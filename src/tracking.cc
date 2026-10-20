#include "runtime_timing.h"
#include "tracking.h"

#include "converter.h"
#include "geometric_tools.h"
#include "keyframe.h"
#include "map_point.h"
#include "matcher.h"
#include "optimizer.h"
#include "pinhole.h"
#include "sophus/se3.hpp"
#include "upnpl_solver.h"

#include <chrono>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <algorithm>
#include <mutex>

using namespace std;

namespace XPLSLAM {

namespace {

bool TrackDiagFrame(const Frame &frame) {
    static const char *range = std::getenv("XPLSLAM_TRACK_DIAG");
    if (!range || !*range)
        return false;
    int first = 0, last = INT_MAX;
    if (std::sscanf(range, "%d:%d", &first, &last) != 2)
        return false;
    return frame.mnId >= static_cast<unsigned long>(first) &&
           frame.mnId <= static_cast<unsigned long>(last);
}

void TrackDiag(const Frame &frame, const char *stage, int matches = -1,
               int inliers = -1) {
    if (!TrackDiagFrame(frame))
        return;
    int detected = 0, stereo = 0, assigned = 0;
    for (int n : frame.Np)
        detected += n;
    for (const auto &depths : frame.mvDepthStereoPoint)
        for (float depth : depths)
            stereo += depth > 0.0f;
    for (const auto &camera_points : frame.mvpMapPoints_cams)
        for (const auto &point : camera_points)
            assigned += point != nullptr;
    std::cerr << "TRACK_DIAG frame=" << frame.mnId << " stage=" << stage
              << " detected=" << detected << " stereo=" << stereo
              << " assigned=" << assigned << " matches=" << matches
              << " inliers=" << inliers << std::endl;
    if (std::strcmp(stage, "detected") == 0 && frame.mnCam >= 2 &&
        !frame.mvFirstToSecondPointMatch.empty()) {
        std::vector<float> vertical_errors;
        const auto &matches01 = frame.mvFirstToSecondPointMatch.front();
        for (size_t i = 0; i < matches01.size(); ++i) {
            const int j = matches01[i];
            if (j >= 0)
                vertical_errors.push_back(std::abs(
                    frame.mvKeysUn_cams[0][i].y -
                    frame.mvKeysUn_cams[1][j].y));
        }
        if (!vertical_errors.empty()) {
            std::sort(vertical_errors.begin(), vertical_errors.end());
            const auto percentile = [&](float ratio) {
                return vertical_errors[static_cast<size_t>(
                    ratio * static_cast<float>(vertical_errors.size() - 1))];
            };
            std::cerr << "STEREO_DIAG frame=" << frame.mnId
                      << " p50_vertical_px=" << percentile(0.5f)
                      << " p90_vertical_px=" << percentile(0.9f)
                      << " p99_vertical_px=" << percentile(0.99f)
                      << std::endl;
        }
    }
}

using TrackingClock = chrono::steady_clock;

// TrackingNow() is the single choke point through which every timing call
// site in this file queries the clock. Gating it here (instead of at each
// call site) makes XPLSLAM_ENABLE_PROFILING=OFF a true zero-cost switch: no
// steady_clock syscall/vDSO query happens anywhere in Track(), and the
// downstream elapsed-time computations collapse to a constant 0.0 that the
// optimizer can fold away, without touching any algorithm control flow.
inline TrackingClock::time_point TrackingNow() {
#ifdef XPLSLAM_ENABLE_PROFILING
    return TrackingClock::now();
#else
    return TrackingClock::time_point();
#endif
}

double TrackingElapsedMilliseconds(const TrackingClock::time_point &begin,
                                   const TrackingClock::time_point &end) {
    return chrono::duration<double, std::milli>(end - begin).count();
}

struct ScopedTrackingTimer {
    explicit ScopedTrackingTimer(double &elapsed_ms)
        : elapsed_ms(elapsed_ms), begin(TrackingNow()) {}

    ~ScopedTrackingTimer() {
        elapsed_ms += TrackingElapsedMilliseconds(begin, TrackingNow());
    }

    double &elapsed_ms;
    TrackingClock::time_point begin;
};

struct TrackTimingStats {
    explicit TrackTimingStats(TrackTimingBreakdown &output)
        : output(output), begin(TrackingNow()) {
        output = {};
    }

    ~TrackTimingStats() {
        output.step_wait_ms = step_wait_ms;
        output.preprocess_ms = preprocess_ms;
        output.map_lock_wait_ms = map_lock_wait_ms;
        output.map_update_check_ms = map_update_check_ms;
        output.initialization_ms = initialization_ms;
        output.pose_tracking_ms = pose_tracking_ms;
        output.local_map_tracking_ms = local_map_tracking_ms;
        output.post_tracking_ms = post_tracking_ms;
        output.trajectory_record_ms = trajectory_record_ms;
        output.total_ms =
            TrackingElapsedMilliseconds(begin, TrackingNow());
    }

    TrackTimingBreakdown &output;
    TrackingClock::time_point begin;
    double step_wait_ms = 0.0;
    double preprocess_ms = 0.0;
    double map_lock_wait_ms = 0.0;
    double map_update_check_ms = 0.0;
    double initialization_ms = 0.0;
    double pose_tracking_ms = 0.0;
    double local_map_tracking_ms = 0.0;
    double post_tracking_ms = 0.0;
    double trajectory_record_ms = 0.0;
};

bool IsSelectedLineConstraint(const Frame &frame, int cam, int index,
                              bool adaptive_enabled) {
    if (!adaptive_enabled) {
        return true;
    }
    return cam >= 0 &&
           cam < static_cast<int>(frame.mvbUseLineConstraint_cams.size()) &&
           index >= 0 &&
           index <
               static_cast<int>(frame.mvbUseLineConstraint_cams[cam].size()) &&
           frame.mvbUseLineConstraint_cams[cam][index];
}

} // namespace

Tracking::Tracking(System *pSys, DBoW3::Vocabulary *pvoc_point,
                   DBoW3::Vocabulary *pvoc_line, Atlas *pAtlas,
                   KeyFrameDatabase *pKFDB, const std::string &strSettingPath,
                   Settings *settings, const std::string &_nameSeq)
    : mState(NO_IMAGES_YET), mTrackedFr(0), mbStep(false),
      mbOnlyTracking(false), mbMapUpdated(false), mbVO(false),
      mpVocabulary_point(pvoc_point), mpVocabulary_line(pvoc_line),
      mpKeyFrameDB(pKFDB), mpSystem(pSys), bStepByStep(false), mpAtlas(pAtlas),
      mnLastRelocFrameId(0), time_recently_lost(5.0), mnInitialFrameId(0),
      mbCreatedMap(false), mnFirstFrameId(0), mpLastKeyFrame(nullptr),
      mpStereoPointIndexInGrid(nullptr) {
    newParameterLoader(settings);

    initID = 0;
    lastID = 0;
    mbInitWith3KFs = false;
    mnNumDataset = 0;
}

Tracking::~Tracking() {}

void Tracking::newParameterLoader(Settings *settings) {
    mnCam = settings->nCams();
    MapPoint::nCam = mnCam;
    MapLine::nCam = mnCam;
    mvpCameras = settings->cameras();
    mvDistCoef.resize(mnCam);
    mvK.resize(mnCam, cv::Mat::eye(3, 3, CV_32F));
    mvK_.resize(mnCam, Eigen::Matrix3f::Identity());
    for (int i = 0; i < mnCam; ++i) {
        mvpCameras[i] = mpAtlas->AddCamera(mvpCameras[i]);
        mvDistCoef[i] = cv::Mat::zeros(4, 1, CV_32F);
        mvK[i].at<float>(0, 0) = mvpCameras[i]->getParameter(0);
        mvK[i].at<float>(1, 1) = mvpCameras[i]->getParameter(1);
        mvK[i].at<float>(0, 2) = mvpCameras[i]->getParameter(2);
        mvK[i].at<float>(1, 2) = mvpCameras[i]->getParameter(3);

        mvK_[i](0, 0) = mvpCameras[i]->getParameter(0);
        mvK_[i](1, 1) = mvpCameras[i]->getParameter(1);
        mvK_[i](0, 2) = mvpCameras[i]->getParameter(2);
        mvK_[i](1, 2) = mvpCameras[i]->getParameter(3);
    }

    mvTbc = settings->Tbc();
    mvStereoPair = settings->stereoPairs();
    mpStereoPointIndexInGrid = &mpSystem->GetStereoPointIndexInGrid();
    mThDepth = settings->thDepth();
    mPointStereoMinDot = settings->pointMinDot();
    mLineStereoMinDot = settings->lineMinDot();
    mAdaptiveLineConfig = settings->adaptiveLines();
    mTrackingRobustnessConfig = settings->trackingRobustness();

    mMaxFrames = settings->fps();
    mMinFrames = 0;
    mbRGB = settings->rgb();

    mImageScale = 1.0f;

    XPLNet::Config xplnet_config = settings->xplnetConfig();
    mpXPLNet = new XPLNet(xplnet_config);
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper) {
    mpLocalMapper = pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing) {
    mpLoopClosing = pLoopClosing;
}

void Tracking::InformOnlyTracking(const bool &flag) { mbOnlyTracking = flag; }

void Tracking::Reset(bool bLocMap) {
    Verbose::PrintMess("System Reseting", Verbose::VERBOSITY_NORMAL);

    // Reset Local Mapping
    if (!bLocMap) {
        Verbose::PrintMess("Reseting Local Mapper...",
                           Verbose::VERBOSITY_NORMAL);
        mpLocalMapper->RequestReset();
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
    }

    // Reset Loop Closing
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    mpLoopClosing->RequestReset();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database...", Verbose::VERBOSITY_NORMAL);
    mpKeyFrameDB->clear();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearAtlas();
    mpAtlas->CreateNewMap();
    mnInitialFrameId = 0;

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();
    mCurrentFrame = Frame();
    mnLastRelocFrameId = 0;
    mLastFrame = Frame();
    mpReferenceKF = nullptr;
    mpLastKeyFrame = nullptr;

    Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

void Tracking::ResetActiveMap(bool bLocMap) {
    Verbose::PrintMess("Active map Reseting", Verbose::VERBOSITY_NORMAL);

    Map *pMap = mpAtlas->GetCurrentMap();

    if (!bLocMap) {
        Verbose::PrintMess("Reseting Local Mapper...",
                           Verbose::VERBOSITY_VERY_VERBOSE);
        mpLocalMapper->RequestResetActiveMap(pMap);
        Verbose::PrintMess("done", Verbose::VERBOSITY_VERY_VERBOSE);
    }

    // Reset Loop Closing
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    mpLoopClosing->RequestResetActiveMap(pMap);
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database", Verbose::VERBOSITY_NORMAL);
    mpKeyFrameDB->clearMap(pMap); // Only clear the active map references
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearMap();

    // KeyFrame::nNextId = mpAtlas->GetLastInitKFid();
    // Frame::nNextId = mnLastInitFrameId;
    mnLastInitFrameId = Frame::nNextId;
    // mnLastRelocFrameId = mnLastInitFrameId;
    mState = NO_IMAGES_YET; // NOT_INITIALIZED;

    list<bool> lbLost;
    // lbLost.reserve(mlbLost.size());
    unsigned int index = mnFirstFrameId;
    cout << "mnFirstFrameId = " << mnFirstFrameId << endl;
    for (Map *pMap : mpAtlas->GetAllMaps()) {
        if (pMap->GetAllKeyFrames().size() > 0) {
            if (index > pMap->GetLowerKFID())
                index = pMap->GetLowerKFID();
        }
    }

    // cout << "First Frame id: " << index << endl;
    int num_lost = 0;
    cout << "mnInitialFrameId = " << mnInitialFrameId << endl;

    for (list<bool>::iterator ilbL = mlbLost.begin(); ilbL != mlbLost.end();
         ilbL++) {
        if (index < mnInitialFrameId)
            lbLost.push_back(*ilbL);
        else {
            lbLost.push_back(true);
            num_lost += 1;
        }

        index++;
    }
    cout << num_lost << " Frames set to lost" << endl;

    mlbLost = lbLost;

    mnInitialFrameId = mCurrentFrame.mnId;
    mnLastRelocFrameId = mCurrentFrame.mnId;

    mCurrentFrame = Frame();
    mLastFrame = Frame();
    mpReferenceKF = nullptr;
    mpLastKeyFrame = nullptr;

    mbVelocity = false;

    Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

Sophus::SE3f Tracking::GrabMultiCamera(const std::vector<cv::Mat> &vIms,
                                       const double &timestamp,
                                       std::string filename) {
    if (runtime_frame) runtime_frame->state_before = mState;
    mLastMultiCameraTiming = {};
    const auto total_begin = TrackingNow();

    auto stage_begin = TrackingNow();
    mvIms = vIms;
    mLastMultiCameraTiming.input_setup_ms =
        TrackingElapsedMilliseconds(stage_begin, TrackingNow());

    stage_begin = TrackingNow();
    mCurrentFrame =
        Frame(mvIms, timestamp, mpXPLNet, mpVocabulary_point, mpVocabulary_line,
              mvK, mvDistCoef, mvTbc, mvStereoPair, mpStereoPointIndexInGrid,
              mPointStereoMinDot, mLineStereoMinDot, mThDepth, mvpCameras);
    if (runtime_frame) runtime_frame->frame_id = mCurrentFrame.mnId;
    TrackDiag(mCurrentFrame, "detected");
    mbCurrentFrameUsesLines = !mAdaptiveLineConfig.enabled;
    mCurrentPoseConstraint = {};
    mnPointMatchesInliers = 0;
    mnLineMatchesInliers = 0;
    if (!mAdaptiveLineConfig.enabled) {
        RuntimeReasonScope runtime_reason(RuntimeLineReason::Always);
        mCurrentFrame.EnsureLineFeatures(Frame::LineFeatureState::StereoReady);
    }
    mLastMultiCameraTiming.frame_construction_ms =
        TrackingElapsedMilliseconds(stage_begin, TrackingNow());
    mLastMultiCameraTiming.frame_id = mCurrentFrame.mnId;
    mLastMultiCameraTiming.frame = mCurrentFrame.GetConstructionTiming();

    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

    Track();
    if (runtime_frame) {
        runtime_frame->state_after = mState;
        runtime_frame->has_pose = mCurrentFrame.HasPose();
    }

    // Lazy line completion happens inside Track(), so publish the final timing
    // rather than the point-only constructor snapshot.
    mLastMultiCameraTiming.frame = mCurrentFrame.GetConstructionTiming();

    stage_begin = TrackingNow();
    Sophus::SE3f pose =
        mCurrentFrame.HasPose() ? mCurrentFrame.GetPose() : Sophus::SE3f();
    mLastMultiCameraTiming.output_pose_ms =
        TrackingElapsedMilliseconds(stage_begin, TrackingNow());
    mLastMultiCameraTiming.total_ms =
        TrackingElapsedMilliseconds(total_begin, TrackingNow());
    return pose;
}

void Tracking::Track() {
    TrackTimingStats timing(mLastMultiCameraTiming.track);

    {
        ScopedTrackingTimer timer(timing.step_wait_ms);
        if (bStepByStep) {
            std::cout << "Tracking: Waiting to the next step" << std::endl;
            while (!mbStep && bStepByStep)
                usleep(500);
            mbStep = false;
        }
    }

    Map *pCurrentMap = nullptr;
    {
        ScopedTrackingTimer timer(timing.preprocess_ms);
        pCurrentMap = mpAtlas->GetCurrentMap();
        if (pCurrentMap == nullptr) {
            std::cerr << "Error: No active map in the atlas." << std::endl;
        }

        if (mState == NO_IMAGES_YET) {
            if (mLastFrame.mTimeStamp > mCurrentFrame.mTimeStamp) {
                cerr << "ERROR: Frame with a timestamp older than previous "
                        "frame detected!"
                     << endl;
                CreateMapInAtlas();
                return;
            }

            mState = NOT_INITIALIZED;
        }

        mLastProcessedState = mState;

        mbCreatedMap = false;
    }

    const auto map_lock_wait_begin = TrackingNow();
    unique_lock<mutex> lock(pCurrentMap->mMutexMapUpdate);
    timing.map_lock_wait_ms =
        TrackingElapsedMilliseconds(map_lock_wait_begin, TrackingNow());

    {
        ScopedTrackingTimer timer(timing.map_update_check_ms);
        mbMapUpdated = false;

        int nCurMapChangeIndex = pCurrentMap->GetMapChangeIndex();
        int nMapChangeIndex = pCurrentMap->GetLastMapChange();
        if (nCurMapChangeIndex > nMapChangeIndex) {
            pCurrentMap->SetLastMapChange(nCurMapChangeIndex);
            mbMapUpdated = true;
        }
    }

    if (mState == NOT_INITIALIZED) {
        ScopedTrackingTimer timer(timing.initialization_ms);
        StereosInitialization();

        if (mState != OK) {
            mLastFrame = Frame(mCurrentFrame);
            return;
        }

        if (mpAtlas->GetAllMaps().size() == 1) {
            mnFirstFrameId = mCurrentFrame.mnId;
        }
    } else {
        bool bOK;

        {
            ScopedTrackingTimer timer(timing.pose_tracking_ms);
            if (!mbOnlyTracking) {
                if (mState == OK) {
                    CheckReplacedInLastFrame();

                    if (mCurrentFrame.mnId < mnLastRelocFrameId + 2 ||
                        !mbVelocity) {
                        bOK = TrackReferenceKeyFrame();
                    } else {
                        bOK = TrackWithMotionModel();
                        if (!bOK) {
                            bOK = TrackReferenceKeyFrame();
                            if (!bOK) {
                                cout << "TrackWithMotionModel and "
                                        "TrackReferenceKeyFrame failed at "
                                        "frame "
                                     << mCurrentFrame.mnId << endl;
                            }
                        }
                    }

                    if (!bOK) {
                        if (pCurrentMap->KeyFramesInMap() > 10) {
                            mState = RECENTLY_LOST;
                            mTimeStampLost = mCurrentFrame.mTimeStamp;
                        } else {
                            mState = LOST;
                        }
                    }
                } else {
                    if (mState == RECENTLY_LOST) {
                        // Keep tracking against the current map while the
                        // previous pose is still close. Relocalization needs
                        // a much larger consensus and can miss a short dip
                        // in temporal matches.
                        bOK = mTrackingRobustnessConfig.max_frame_step_m > 0.0f &&
                              TrackReferenceKeyFrame();
                        if (!bOK)
                            bOK = Relocalization();

                        if (mCurrentFrame.mTimeStamp - mTimeStampLost > 3.0f &&
                            !bOK) {
                            mState = LOST;
                            bOK = false;
                        }
                    } else if (mState == LOST) {
                        if (pCurrentMap->KeyFramesInMap() < 10) {
                            mpSystem->ResetActiveMap();
                        } else
                            CreateMapInAtlas();

                        if (mpLastKeyFrame)
                            mpLastKeyFrame = nullptr;

                        return;
                    }
                }
            } else {
                if (mState == LOST) {
                    bOK = Relocalization();
                }
                if (!mbVO) {
                    if (mbVelocity) {
                        bOK = TrackWithMotionModel();
                    } else {
                        bOK = TrackReferenceKeyFrame();
                    }
                } else {
                    bool bOKLastFrame = false;
                    bool bOKReloc = false;
                    vector<vector<MapPointPtr>> vpMPsLastFrame;
                    vector<vector<MapLinePtr>> vpMLsLastFrame;
                    Sophus::SE3f TbwLastFrame;
                    if (mbVelocity) {
                        bOKLastFrame = TrackWithMotionModel();
                        vpMPsLastFrame = mCurrentFrame.mvpMapPoints_cams;
                        vpMLsLastFrame = mCurrentFrame.mvpMapLines_cams;
                        TbwLastFrame = mCurrentFrame.GetPose();
                    }
                    bOKReloc = Relocalization();

                    if (bOKLastFrame && bOKReloc) {
                        mCurrentFrame.SetPose(TbwLastFrame);
                        mCurrentFrame.mvpMapPoints_cams = vpMPsLastFrame;
                        mCurrentFrame.mvpMapLines_cams = vpMLsLastFrame;

                        if (mbVO) {
                            for (int cam_id = 0; cam_id < mnCam; ++cam_id) {
                                for (int i = 0; i < mCurrentFrame.Np[cam_id];
                                     ++i) {
                                    if (mCurrentFrame
                                            .mvpMapPoints_cams[cam_id][i] &&
                                        !mCurrentFrame
                                             .mvbOutlierPoint_cams[cam_id][i]) {
                                        mCurrentFrame
                                            .mvpMapPoints_cams[cam_id][i]
                                            ->IncreaseFound();
                                    }
                                }
                                for (int i = 0; i < mCurrentFrame.Nl[cam_id];
                                     ++i) {
                                    if (mCurrentFrame
                                            .mvpMapLines_cams[cam_id][i] &&
                                        !mCurrentFrame
                                             .mvbOutlierLine_cams[cam_id][i]) {
                                        mCurrentFrame
                                            .mvpMapLines_cams[cam_id][i]
                                            ->IncreaseFound();
                                    }
                                }
                            }
                        }
                    } else if (bOKReloc) {
                        mbVO = false;
                    }

                    bOK = bOKLastFrame || bOKReloc;
                }
            }
        }

        {
            ScopedTrackingTimer timer(timing.local_map_tracking_ms);
            if (!mCurrentFrame.mpReferenceKF)
                mCurrentFrame.mpReferenceKF = mpReferenceKF;

            if (!mbOnlyTracking) {
                if (bOK) {
                    bOK = TrackLocalMap();
                    if (!bOK)
                        cout << "TrackLocalMap failed" << endl;
                }

            } else {
                if (bOK && !mbVO)
                    bOK = TrackLocalMap();
            }
        }

        {
            ScopedTrackingTimer timer(timing.post_tracking_ms);
            if (bOK)
                mState = OK;
            else if (mState == OK) {
                mState = RECENTLY_LOST;
                mTimeStampLost = mCurrentFrame.mTimeStamp;
            }

            if (bOK || mState == RECENTLY_LOST) {
                if ((bOK ||
                     mTrackingRobustnessConfig.max_frame_step_m <= 0.0f) &&
                    mLastFrame.isSet() && mCurrentFrame.isSet()) {
                    Sophus::SE3f LastTwb = mLastFrame.GetPose().inverse();
                    mVelocity = mCurrentFrame.GetPose() * LastTwb;
                    mbVelocity = true;
                } else {
                    mbVelocity = false;
                    if (mTrackingRobustnessConfig.max_frame_step_m > 0.0f &&
                        mLastFrame.isSet())
                        mCurrentFrame.SetPose(mLastFrame.GetPose());
                }

                for (int c = 0; c < mnCam; ++c) {
                    for (int i = 0; i < mCurrentFrame.Np[c]; ++i) {
                        MapPointPtr pMP = mCurrentFrame.mvpMapPoints_cams[c][i];
                        if (pMP) {
                            if (pMP->Observations() < 1) {
                                mCurrentFrame.mvpMapPoints_cams[c][i] = nullptr;
                                mCurrentFrame.mvbOutlierPoint_cams[c][i] =
                                    false;
                            }
                        }
                    }

                    for (int i = 0; i < mCurrentFrame.Nl[c]; ++i) {
                        MapLinePtr pML = mCurrentFrame.mvpMapLines_cams[c][i];
                        if (pML) {
                            if (pML->Observations() < 1) {
                                mCurrentFrame.mvpMapLines_cams[c][i] = nullptr;
                                mCurrentFrame.mvbOutlierLine_cams[c][i] = false;
                            }
                        }
                    }
                }

                mlpTemporalPoints.clear();

                mlpTemporalLines.clear();

                bool bNeedKF = NeedNewKeyFrame();

                if (bNeedKF && bOK)
                    CreateNewKeyFrame();

                for (int cam_id = 0; cam_id < mnCam; ++cam_id) {
                    for (int i = 0; i < mCurrentFrame.Np[cam_id]; ++i) {
                        if (mCurrentFrame.mvpMapPoints_cams[cam_id][i] &&
                            mCurrentFrame.mvbOutlierPoint_cams[cam_id][i])
                            mCurrentFrame.mvpMapPoints_cams[cam_id][i] =
                                nullptr;
                    }
                    for (int i = 0; i < mCurrentFrame.Nl[cam_id]; ++i) {
                        if (mCurrentFrame.mvpMapLines_cams[cam_id][i] &&
                            mCurrentFrame.mvbOutlierLine_cams[cam_id][i])
                            mCurrentFrame.mvpMapLines_cams[cam_id][i] = nullptr;
                    }
                }
            }

            if (mState == LOST) {
                if (pCurrentMap->KeyFramesInMap() < 10) {
                    mpSystem->ResetActiveMap();
                    return;
                }
                CreateMapInAtlas();
                return;
            }

            if (!mCurrentFrame.mpReferenceKF)
                mCurrentFrame.mpReferenceKF = mpReferenceKF;

            mLastFrame = Frame(mCurrentFrame);
        }
    }

    {
        ScopedTrackingTimer timer(timing.trajectory_record_ms);
        if (mState == OK || mState == RECENTLY_LOST) {
            if (mCurrentFrame.isSet()) {
                Sophus::SE3f Tcr_ =
                    mCurrentFrame.GetPose() *
                    mCurrentFrame.mpReferenceKF->GetPoseInverse();
                mlRelativeFramePoses.push_back(Tcr_);
                mlpReferences.push_back(mCurrentFrame.mpReferenceKF);
                mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
                mlbLost.push_back(mState == LOST);
            } else {
                mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
                mlpReferences.push_back(mlpReferences.back());
                mlFrameTimes.push_back(mlFrameTimes.back());
                mlbLost.push_back(mState == LOST);
            }
        }
    }
}

void Tracking::CreateMapInAtlas() {
    mnLastInitFrameId = mCurrentFrame.mnId;
    mpAtlas->CreateNewMap();

    mnInitialFrameId = mCurrentFrame.mnId + 1;
    mState = NO_IMAGES_YET;

    // Restart the variable with information about the last KF
    mbVelocity = false;
    // mnLastRelocFrameId = mnLastInitFrameId; // The last relocation KF_id
    // is the current id, because it is the new starting point for new map
    Verbose::PrintMess("First frame id in map: " +
                           to_string(mnLastInitFrameId + 1),
                       Verbose::VERBOSITY_NORMAL);
    mbVO = false; // Init value for know if there are enough MapPoints in
                  // the last KF
    if (mpLastKeyFrame)
        mpLastKeyFrame = nullptr;

    if (mpReferenceKF)
        mpReferenceKF = nullptr;

    mLastFrame = Frame();
    mCurrentFrame = Frame();

    mbCreatedMap = true;
}

void Tracking::StereosInitialization() {
    RuntimeReasonScope runtime_reason(RuntimeLineReason::Initialization);
    if (runtime_frame) ++runtime_frame->initialization_calls;
    if (!mCurrentFrame.EnsureLineFeatures(
            Frame::LineFeatureState::StereoReady)) {
        return;
    }
    const int N = mCurrentFrame.mnTotalPoints + mCurrentFrame.mnTotalLines;

    if (N < 500)
        return;

    mCurrentFrame.SetPose(Sophus::SE3f());

    KeyFramePtr pKFini = std::make_shared<KeyFrame>(
        mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);

    if (runtime_frame) runtime_frame->keyframe_created = true;
    mpAtlas->AddKeyFrame(pKFini);

    for (int pair_id = 0; pair_id < mvStereoPair.size(); ++pair_id) {
        const int cam_id1 = mvStereoPair[pair_id].first;
        const int cam_id2 = mvStereoPair[pair_id].second;

        for (int i = 0; i < mCurrentFrame.Np[cam_id1]; ++i) {
            int second_idx =
                mCurrentFrame.mvFirstToSecondPointMatch[pair_id][i];
            float z = mCurrentFrame.mvDepthStereoPoint[pair_id][i];
            if (second_idx != -1 && z > 0.0f &&
                (mThDepth <= 0.0f || z <= mThDepth)) {
                MapPointPtr pMP1 = mCurrentFrame.mvpMapPoints_cams[cam_id1][i];
                MapPointPtr pMP2 =
                    mCurrentFrame.mvpMapPoints_cams[cam_id2][second_idx];

                if (!pMP1 && !pMP2) {
                    Eigen::Vector3f x3D;
                    mCurrentFrame.UnprojectStereoPoint(pair_id, i, x3D);

                    MapPointPtr pNewMP = std::make_shared<MapPoint>(
                        x3D, pKFini, mpAtlas->GetCurrentMap());

                    pNewMP->AddObservation(pKFini, i, cam_id1);
                    pNewMP->AddObservation(pKFini, second_idx, cam_id2);

                    pKFini->AddMapPoint(pNewMP, i, cam_id1);
                    pKFini->AddMapPoint(pNewMP, second_idx, cam_id2);

                    pNewMP->ComputeDistinctiveDescriptors();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints_cams[cam_id1][i] = pNewMP;
                    mCurrentFrame.mvpMapPoints_cams[cam_id2][second_idx] =
                        pNewMP;
                } else if (pMP1 && !pMP2) {
                    if (pMP1->HasObservation(pKFini, cam_id2))
                        continue;
                    mCurrentFrame.mvpMapPoints_cams[cam_id2][second_idx] = pMP1;
                    pMP1->AddObservation(pKFini, second_idx, cam_id2);
                    pKFini->AddMapPoint(pMP1, second_idx, cam_id2);
                    pMP1->ComputeDistinctiveDescriptors();
                } else if (!pMP1 && pMP2) {
                    if (pMP2->HasObservation(pKFini, cam_id1))
                        continue;
                    mCurrentFrame.mvpMapPoints_cams[cam_id1][i] = pMP2;
                    pMP2->AddObservation(pKFini, i, cam_id1);
                    pKFini->AddMapPoint(pMP2, i, cam_id1);
                    pMP2->ComputeDistinctiveDescriptors();
                }
            }
        }

        for (int i = 0; i < mCurrentFrame.Nl[cam_id1]; ++i) {
            int second_idx = mCurrentFrame.mvFirstToSecondLineMatch[pair_id][i];
            float z = mCurrentFrame.mvDepthStereoLine[pair_id][i];
            if (second_idx != -1) {
                MapLinePtr pML1 = mCurrentFrame.mvpMapLines_cams[cam_id1][i];
                MapLinePtr pML2 =
                    mCurrentFrame.mvpMapLines_cams[cam_id2][second_idx];
                if (!pML1 && !pML2) {
                    Eigen::Matrix<float, 6, 1> line3D;
                    mCurrentFrame.UnprojectStereoLine(pair_id, i, line3D);

                    MapLinePtr pNewML = std::make_shared<MapLine>(
                        line3D, pKFini, mpAtlas->GetCurrentMap());

                    pNewML->AddObservation(pKFini, i, cam_id1);
                    pNewML->AddObservation(pKFini, second_idx, cam_id2);

                    pKFini->AddMapLine(pNewML, i, cam_id1);
                    pKFini->AddMapLine(pNewML, second_idx, cam_id2);

                    pNewML->ComputeDistinctiveDescriptors();
                    mpAtlas->AddMapLine(pNewML);

                    mCurrentFrame.mvpMapLines_cams[cam_id1][i] = pNewML;
                    mCurrentFrame.mvpMapLines_cams[cam_id2][second_idx] =
                        pNewML;
                } else if (pML1 && !pML2) {
                    if (pML1->HasObservation(pKFini, cam_id2))
                        continue;
                    mCurrentFrame.mvpMapLines_cams[cam_id2][second_idx] = pML1;
                    pML1->AddObservation(pKFini, second_idx, cam_id2);
                    pKFini->AddMapLine(pML1, second_idx, cam_id2);
                    pML1->ComputeDistinctiveDescriptors();
                } else if (!pML1 && pML2) {
                    if (pML2->HasObservation(pKFini, cam_id1))
                        continue;
                    mCurrentFrame.mvpMapLines_cams[cam_id1][i] = pML2;
                    pML2->AddObservation(pKFini, i, cam_id1);
                    pKFini->AddMapLine(pML2, i, cam_id1);
                    pML2->ComputeDistinctiveDescriptors();
                }
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKFini);

    mLastFrame = Frame(mCurrentFrame);
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKFini;

    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints = mpAtlas->GetAllMapPoints();
    mvpLocalMapLines = mpAtlas->GetAllMapLines();
    mpReferenceKF = pKFini;
    mCurrentFrame.mpReferenceKF = pKFini;

    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);
    mpAtlas->SetReferenceMapLines(mvpLocalMapLines);

    mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

    mState = OK;
}

void Tracking::CheckReplacedInLastFrame() {
    for (int cam_id = 0; cam_id < mnCam; ++cam_id) {
        for (int i = 0; i < mLastFrame.Np[cam_id]; ++i) {
            MapPointPtr pMP = mLastFrame.mvpMapPoints_cams[cam_id][i];
            if (pMP) {
                MapPointPtr pRep = pMP->GetReplaced();
                if (pRep) {
                    mLastFrame.mvpMapPoints_cams[cam_id][i] = pRep;
                }
            }
        }

        for (int i = 0; i < mLastFrame.Nl[cam_id]; ++i) {
            MapLinePtr pML = mLastFrame.mvpMapLines_cams[cam_id][i];
            if (pML) {
                MapLinePtr pRep = pML->GetReplaced();
                if (pRep) {
                    mLastFrame.mvpMapLines_cams[cam_id][i] = pRep;
                }
            }
        }
    }
}

void Tracking::MatchReferenceLines(KeyFramePtr reference) {
    ScopedTrackingTimer timer(
        mLastMultiCameraTiming.track.deferred_line_tracking_ms);
    if (!reference || !mCurrentFrame.EnsureLineFeatures(
                          Frame::LineFeatureState::TrackingReady)) {
        return;
    }
    mCurrentFrame.ComputeLineBoW();
    Matcher matcher(0.9);
    vector<vector<MapLinePtr>> line_matches;
    matcher.SearchLinesByBow(reference, mCurrentFrame, line_matches);
    mCurrentFrame.mvpMapLines_cams = std::move(line_matches);
}

void Tracking::OptimizeCurrentPoseAdaptive(KeyFramePtr line_reference) {
    RuntimeReasonScope runtime_reason(RuntimeLineReason::WeakPoints);
    if (!mAdaptiveLineConfig.enabled) {
        Optimizer::PoseOptimization(&mCurrentFrame);
        mbCurrentFrameUsesLines = true;
        return;
    }

    {
        ScopedTrackingTimer timer(
            mLastMultiCameraTiming.track.point_constraint_analysis_ms);
        ScopedTrackingTimer evaluation_timer(
            mLastMultiCameraTiming.track.point_constraint_evaluation_ms);
        mCurrentPoseConstraint = Optimizer::EvaluatePointConstraints(
            &mCurrentFrame, mAdaptiveLineConfig);
    }
    mbCurrentFrameUsesLines = false;
    if (!mCurrentPoseConstraint.point_constraints_strong) {
        MatchReferenceLines(line_reference);
        mbCurrentFrameUsesLines = mCurrentFrame.HasLineFeatures();
        if (mbCurrentFrameUsesLines) {
            ScopedTrackingTimer timer(
                mLastMultiCameraTiming.track.line_selection_optimization_ms);
            ScopedTrackingTimer selection_timer(
                mLastMultiCameraTiming.track.line_selection_ms);
            mCurrentPoseConstraint = Optimizer::SelectLineConstraints(
                &mCurrentFrame, mAdaptiveLineConfig, mCurrentPoseConstraint);
        }
    }
    // Both branches optimize exactly once, including when no line was selected.
    ScopedTrackingTimer optimization_timer(
        mbCurrentFrameUsesLines
            ? mLastMultiCameraTiming.track.line_selection_optimization_ms
            : mLastMultiCameraTiming.track.point_constraint_analysis_ms);
    Optimizer::PoseOptimizationSelectedLines(
        &mCurrentFrame, mAdaptiveLineConfig, &mCurrentPoseConstraint);
}

bool Tracking::IsPlausibleFrameStep(const char *stage) const {
    const float max_step = mTrackingRobustnessConfig.max_frame_step_m;
    if (max_step <= 0.0f || !mLastFrame.isSet() ||
        !mCurrentFrame.isSet())
        return true;

    const Eigen::Vector3f current_center =
        mCurrentFrame.GetPose().inverse().translation();
    const Eigen::Vector3f last_center =
        mLastFrame.GetPose().inverse().translation();
    const float distance = (current_center - last_center).norm();
    const double elapsed = mCurrentFrame.mTimeStamp - mLastFrame.mTimeStamp;
    const float allowed = max_step *
        std::max(1.0, elapsed * static_cast<double>(mMaxFrames));
    if (std::isfinite(distance) && distance <= allowed)
        return true;

    std::cerr << "Rejected implausible " << stage << " pose at frame "
              << mCurrentFrame.mnId << ": step=" << distance
              << " m, limit=" << allowed << " m" << std::endl;
    return false;
}

bool Tracking::TrackReferenceKeyFrame() {
    mCurrentFrame.ComputeBoW();

    Matcher matcher(0.9);
    vector<vector<MapPointPtr>> vpMapPointMatches;
    vector<vector<MapLinePtr>> vpMapLineMatches;

#ifdef XPLSLAM_ENABLE_PROFILING
    cout << "Reference KF: " << mpReferenceKF->mnFrameId
         << ", Current Frame: " << mCurrentFrame.mnId << endl;
    cout << "Reference KF isBad: " << mpReferenceKF->isBad() << endl;
#endif
    int nmatches = matcher.SearchByBow(mpReferenceKF, mCurrentFrame,
                                       vpMapPointMatches, vpMapLineMatches);
    TrackDiag(mCurrentFrame, "reference_bow", nmatches);

    if (nmatches < 15 && !mAdaptiveLineConfig.enabled) {
        cout << "Not enough matches to reference keyframe: " << nmatches
             << endl;
        return false;
    }

    // cout << "Matches to reference keyframe: " << nmatches << endl;

    mCurrentFrame.mvpMapPoints_cams = vpMapPointMatches;
    mCurrentFrame.mvpMapLines_cams = vpMapLineMatches;

    // KeyFramePtr pLastRefKF = mLastFrame.mpReferenceKF;
    // if (pLastRefKF && !pLastRefKF->isBad()) {
    //     cout << "Last reference KF: " << pLastRefKF->mnFrameId
    //          << ", isBad: " << pLastRefKF->isBad() << endl;
    //     Sophus::SE3f Tlr =
    //         mlRelativeFramePoses.empty()
    //             ? mLastFrame.GetPose() * pLastRefKF->GetPoseInverse()
    //             : mlRelativeFramePoses.back();
    //     mLastFrame.SetPose(Tlr * pLastRefKF->GetPose());
    // }
    // UpdateLastFrame();
    mCurrentFrame.SetPose(mLastFrame.GetPose());

    OptimizeCurrentPoseAdaptive(mpReferenceKF);

    int nmatchesMap = 0;
    for (int cam_id = 0; cam_id < mnCam; ++cam_id) {
        for (int i = 0; i < mCurrentFrame.Np[cam_id]; ++i) {
            if (mCurrentFrame.mvpMapPoints_cams[cam_id][i]) {
                if (mCurrentFrame.mvbOutlierPoint_cams[cam_id][i]) {
                    MapPointPtr pMP =
                        mCurrentFrame.mvpMapPoints_cams[cam_id][i];

                    mCurrentFrame.mvpMapPoints_cams[cam_id][i] = nullptr;
                    mCurrentFrame.mvbOutlierPoint_cams[cam_id][i] = false;
                    pMP->mvbTrackInView[cam_id] = false;
                    pMP->mvnLastFrameSeen[cam_id] = mCurrentFrame.mnId;
                    nmatches--;
                } else if (mCurrentFrame.mvpMapPoints_cams[cam_id][i]
                               ->Observations() > 0) {
                    nmatchesMap++;
                }
            }
        }

        for (int i = 0; i < mCurrentFrame.Nl[cam_id]; ++i) {
            if (mCurrentFrame.mvpMapLines_cams[cam_id][i]) {
                if (mCurrentFrame.mvbOutlierLine_cams[cam_id][i]) {
                    MapLinePtr pML = mCurrentFrame.mvpMapLines_cams[cam_id][i];

                    mCurrentFrame.mvpMapLines_cams[cam_id][i] = nullptr;
                    mCurrentFrame.mvbOutlierLine_cams[cam_id][i] = false;
                    pML->mvbTrackInView[cam_id] = false;
                    pML->mvnLastFrameSeen[cam_id] = mCurrentFrame.mnId;
                    nmatches--;
                } else if (mCurrentFrame.mvpMapLines_cams[cam_id][i]
                                   ->Observations() > 0 &&
                           IsSelectedLineConstraint(
                               mCurrentFrame, cam_id, i,
                               mAdaptiveLineConfig.enabled)) {
                    nmatchesMap++;
                }
            }
        }
    }
    // cout << "Matches to reference keyframe: " << nmatches
    //      << ", matches with map points/lines: " << nmatchesMap << endl;

    TrackDiag(mCurrentFrame, "reference_optimized", nmatches, nmatchesMap);
    return nmatchesMap >= 10 && IsPlausibleFrameStep("reference");
}

bool Tracking::TrackLastFrame() {
    constexpr int inlierTh = 20;

    UpdateLastFrame();
    mLastFrame.ComputeBoW();
    mCurrentFrame.ComputeBoW();

    Matcher matcher(0.9);
    vector<vector<MapPointPtr>> vpMapPointMatches;
    vector<vector<MapLinePtr>> vpMapLineMatches;
    int nmatches = matcher.SearchByBow(mLastFrame, mCurrentFrame,
                                       vpMapPointMatches, vpMapLineMatches);
    // cout << "Matches to last frame: " << nmatches << endl;

    if (nmatches < inlierTh && !mAdaptiveLineConfig.enabled) {
        return false;
    }

    int iter = 30;
    UPnPLSolver solver(mCurrentFrame, vpMapPointMatches, vpMapLineMatches);
    solver.SetRansacParameters(0.99, 10, iter, 6, 0.5, 5.991);

    vector<vector<bool>> vbPointInliers;
    vector<vector<bool>> vbLineInliers;
    int nInliers = 0;
    bool bNoMore = false;
    Eigen::Matrix4f eigTbw;
    if (!solver.iterate(iter, bNoMore, vbPointInliers, vbLineInliers, nInliers,
                        eigTbw)) {
        cout << "Pose estimation failed in TrackLastFrame." << endl;
        return false;
    }

    mCurrentFrame.SetPose(Sophus::SE3f(eigTbw));
    for (int cam = 0; cam < mnCam; ++cam) {
        for (size_t i = 0; i < vpMapPointMatches[cam].size(); ++i) {
            mCurrentFrame.mvpMapPoints_cams[cam][i] =
                vbPointInliers[cam][i] ? vpMapPointMatches[cam][i] : nullptr;
        }
        for (size_t i = 0; i < vpMapLineMatches[cam].size(); ++i) {
            mCurrentFrame.mvpMapLines_cams[cam][i] =
                vbLineInliers[cam][i] ? vpMapLineMatches[cam][i] : nullptr;
        }
    }

    nmatches = nInliers;
    // cout << "Inliers to last frame: " << nmatches << endl;
    OptimizeCurrentPoseAdaptive(mpReferenceKF);

    int nmatchesMap = 0;
    for (int cam_id = 0; cam_id < mnCam; ++cam_id) {
        for (int i = 0; i < mCurrentFrame.Np[cam_id]; ++i) {
            if (mCurrentFrame.mvpMapPoints_cams[cam_id][i]) {
                if (mCurrentFrame.mvbOutlierPoint_cams[cam_id][i]) {
                    MapPointPtr pMP =
                        mCurrentFrame.mvpMapPoints_cams[cam_id][i];

                    mCurrentFrame.mvpMapPoints_cams[cam_id][i] = nullptr;
                    mCurrentFrame.mvbOutlierPoint_cams[cam_id][i] = false;
                    pMP->mvbTrackInView[cam_id] = false;
                    pMP->mvnLastFrameSeen[cam_id] = mCurrentFrame.mnId;
                    nmatches--;
                } else if (mCurrentFrame.mvpMapPoints_cams[cam_id][i]
                               ->Observations() > 0) {
                    nmatchesMap++;
                }
            }
        }

        for (int i = 0; i < mCurrentFrame.Nl[cam_id]; ++i) {
            if (mCurrentFrame.mvpMapLines_cams[cam_id][i]) {
                if (mCurrentFrame.mvbOutlierLine_cams[cam_id][i]) {
                    MapLinePtr pML = mCurrentFrame.mvpMapLines_cams[cam_id][i];

                    mCurrentFrame.mvpMapLines_cams[cam_id][i] = nullptr;
                    mCurrentFrame.mvbOutlierLine_cams[cam_id][i] = false;
                    pML->mvbTrackInView[cam_id] = false;
                    pML->mvnLastFrameSeen[cam_id] = mCurrentFrame.mnId;
                    nmatches--;
                } else if (mCurrentFrame.mvpMapLines_cams[cam_id][i]
                                   ->Observations() > 0 &&
                           IsSelectedLineConstraint(
                               mCurrentFrame, cam_id, i,
                               mAdaptiveLineConfig.enabled)) {
                    nmatchesMap++;
                }
            }
        }
    }

    if (mbOnlyTracking) {
        mbVO = nmatchesMap < 10;
        return nmatches > inlierTh;
    }

    return nmatchesMap >= inlierTh;
}

bool Tracking::TrackWithMotionModel() {
    Matcher matcher(0.9);

    UpdateLastFrame();

    mCurrentFrame.SetPose(mVelocity * mLastFrame.GetPose());
    // cout << setprecision(9) << fixed;
    // cout << "velocity: " << mVelocity.matrix() << endl;

    int inlier_th = 20;

    for (int cam_id = 0; cam_id < mnCam; ++cam_id) {
        fill(mCurrentFrame.mvpMapPoints_cams[cam_id].begin(),
             mCurrentFrame.mvpMapPoints_cams[cam_id].end(), nullptr);
        fill(mCurrentFrame.mvpMapLines_cams[cam_id].begin(),
             mCurrentFrame.mvpMapLines_cams[cam_id].end(), nullptr);
    }

    float th_point = 7;
    float th_line_d = 7;
    float th_line_theta = 0.12;

    int nmatches = matcher.SearchByProjection(
        mCurrentFrame, mLastFrame, th_point, th_line_d, th_line_theta);
    TrackDiag(mCurrentFrame, "motion_projection", nmatches);

    if (nmatches < 30) {
        cout << "Not enough matches with motion model, trying again with a "
                "wider "
                "search radius: "
             << nmatches << endl;
        for (int cam_id = 0; cam_id < mnCam; ++cam_id) {
            fill(mCurrentFrame.mvpMapPoints_cams[cam_id].begin(),
                 mCurrentFrame.mvpMapPoints_cams[cam_id].end(), nullptr);
            fill(mCurrentFrame.mvpMapLines_cams[cam_id].begin(),
                 mCurrentFrame.mvpMapLines_cams[cam_id].end(), nullptr);
        }

        nmatches =
            matcher.SearchByProjection(mCurrentFrame, mLastFrame, th_point * 2,
                                       th_line_d * 2, th_line_theta * 2);
        TrackDiag(mCurrentFrame, "motion_wide", nmatches);
    }

    if (nmatches < inlier_th && !mAdaptiveLineConfig.enabled) {
        return false;
    }

    // chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    OptimizeCurrentPoseAdaptive(mpReferenceKF);
    // chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    // chrono::duration<double> time_used =
    //     chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    // cout << "PoseOptimization time: " << time_used.count() * 1000 << " ms"
    //      << endl;

    int nmatchesMap = 0;
    for (int cam_id = 0; cam_id < mnCam; ++cam_id) {
        for (int i = 0; i < mCurrentFrame.Np[cam_id]; ++i) {
            if (mCurrentFrame.mvpMapPoints_cams[cam_id][i]) {
                if (mCurrentFrame.mvbOutlierPoint_cams[cam_id][i]) {
                    MapPointPtr pMP =
                        mCurrentFrame.mvpMapPoints_cams[cam_id][i];

                    mCurrentFrame.mvpMapPoints_cams[cam_id][i] = nullptr;
                    mCurrentFrame.mvbOutlierPoint_cams[cam_id][i] = false;
                    pMP->mvbTrackInView[cam_id] = false;
                    pMP->mvnLastFrameSeen[cam_id] = mCurrentFrame.mnId;
                    nmatches--;
                } else if (mCurrentFrame.mvpMapPoints_cams[cam_id][i]
                               ->Observations() > 0) {
                    nmatchesMap++;
                }
            }
        }

        for (int i = 0; i < mCurrentFrame.Nl[cam_id]; ++i) {
            if (mCurrentFrame.mvpMapLines_cams[cam_id][i]) {
                if (mCurrentFrame.mvbOutlierLine_cams[cam_id][i]) {
                    MapLinePtr pML = mCurrentFrame.mvpMapLines_cams[cam_id][i];

                    mCurrentFrame.mvpMapLines_cams[cam_id][i] = nullptr;
                    mCurrentFrame.mvbOutlierLine_cams[cam_id][i] = false;
                    pML->mvbTrackInView[cam_id] = false;
                    pML->mvnLastFrameSeen[cam_id] = mCurrentFrame.mnId;
                    nmatches--;
                } else if (mCurrentFrame.mvpMapLines_cams[cam_id][i]
                                   ->Observations() > 0 &&
                           IsSelectedLineConstraint(
                               mCurrentFrame, cam_id, i,
                               mAdaptiveLineConfig.enabled)) {
                    nmatchesMap++;
                }
            }
        }
    }

    if (mbOnlyTracking) {
        mbVO = nmatchesMap < 10;
        return nmatches > inlier_th;
    }

    TrackDiag(mCurrentFrame, "motion_optimized", nmatches, nmatchesMap);
    return nmatchesMap >= inlier_th && IsPlausibleFrameStep("motion");
}

void Tracking::UpdateLastFrame() {
    KeyFramePtr pRef = mLastFrame.mpReferenceKF;
    if (!pRef)
        return;
    Sophus::SE3f Tlr = mlRelativeFramePoses.back();
    mLastFrame.SetPose(Tlr * pRef->GetPose());

    if (mnLastKeyFrameId == mLastFrame.mnId || !mbOnlyTracking)
        return;

    vector<tuple<float, int, int>> vDepthPointIdx;
    vDepthPointIdx.reserve(mLastFrame.mnTotalPoints);

    vector<tuple<float, int, int>> vDepthLineIdx;
    vDepthLineIdx.reserve(mLastFrame.mnTotalLines);

    for (int pair_id = 0; pair_id < mvStereoPair.size(); ++pair_id) {
        const int cam_id1 = mvStereoPair[pair_id].first;
        const int cam_id2 = mvStereoPair[pair_id].second;

        for (int i = 0; i < mLastFrame.Np[cam_id1]; ++i) {
            float z = mLastFrame.mvDepthStereoPoint[pair_id][i];
            if (z > 0)
                vDepthPointIdx.emplace_back(z, pair_id, i);
        }

        if (mLastFrame.HasStereoLineFeatures()) {
            for (int i = 0; i < mLastFrame.Nl[cam_id1]; ++i) {
                float z = mLastFrame.mvDepthStereoLine[pair_id][i];
                if (z > 0)
                    vDepthLineIdx.emplace_back(z, pair_id, i);
            }
        }
    }

    if (!vDepthPointIdx.empty()) {
        sort(vDepthPointIdx.begin(), vDepthPointIdx.end());
        int nPoints = 0;

        for (size_t i = 0; i < vDepthPointIdx.size(); ++i) {
            float z = get<0>(vDepthPointIdx[i]);
            int pair_id = get<1>(vDepthPointIdx[i]);
            int idx = get<2>(vDepthPointIdx[i]);

            int cam_id1 = mvStereoPair[pair_id].first;
            int cam_id2 = mvStereoPair[pair_id].second;

            int second_idx = mLastFrame.mvFirstToSecondPointMatch[pair_id][idx];

            bool bCreateNew = false;

            MapPointPtr pMP = mLastFrame.mvpMapPoints_cams[cam_id1][idx];

            if (!pMP)
                bCreateNew = true;
            else if (pMP->Observations() < 1)
                bCreateNew = true;

            if (bCreateNew) {
                Eigen::Vector3f x3DW;

                mLastFrame.UnprojectStereoPoint(pair_id, idx, x3DW);

                MapPointPtr pNewMP = std::make_shared<MapPoint>(
                    x3DW, mpAtlas->GetCurrentMap(), &mLastFrame, idx, cam_id1);
                mLastFrame.mvpMapPoints_cams[cam_id1][idx] = pNewMP;
                mLastFrame.mvpMapPoints_cams[cam_id2][second_idx] = pNewMP;

                mlpTemporalPoints.push_back(pNewMP);
                nPoints++;
            } else {
                nPoints++;
            }

            if (z > mThDepth && nPoints > 100)
                break;
        }
    }

    if (!vDepthLineIdx.empty()) {
        sort(vDepthLineIdx.begin(), vDepthLineIdx.end());
        int nLines = 0;

        for (size_t i = 0; i < vDepthLineIdx.size(); ++i) {
            float z = get<0>(vDepthLineIdx[i]);
            int pair_id = get<1>(vDepthLineIdx[i]);
            int idx = get<2>(vDepthLineIdx[i]);

            int cam_id1 = mvStereoPair[pair_id].first;
            int cam_id2 = mvStereoPair[pair_id].second;

            int second_idx = mLastFrame.mvFirstToSecondLineMatch[pair_id][idx];

            bool bCreateNew = false;

            MapLinePtr pML = mLastFrame.mvpMapLines_cams[cam_id1][idx];

            if (!pML)
                bCreateNew = true;
            else if (pML->Observations() < 1)
                bCreateNew = true;

            if (bCreateNew) {
                Eigen::Matrix<float, 6, 1> line3DW;

                mLastFrame.UnprojectStereoLine(pair_id, idx, line3DW);

                MapLinePtr pNewML =
                    std::make_shared<MapLine>(line3DW, mpAtlas->GetCurrentMap(),
                                              &mLastFrame, idx, cam_id1);
                mLastFrame.mvpMapLines_cams[cam_id1][idx] = pNewML;
                mLastFrame.mvpMapLines_cams[cam_id2][second_idx] = pNewML;

                mlpTemporalLines.push_back(pNewML);
                nLines++;
            } else {
                nLines++;
            }

            if (z > mThDepth && nLines > 50)
                break;
        }
    }
}

bool Tracking::TrackLocalMap() {
    RuntimeReasonScope runtime_reason(RuntimeLineReason::WeakPoints);
    auto &timing = mLastMultiCameraTiming.track.local_map;
    timing = {};
    ScopedTrackingTimer total_timer(timing.total_ms);

    mTrackedFr++;

    UpdateLocalMap();
    {
        ScopedTrackingTimer timer(timing.search_local_features_ms);
        SearchLocalFeatures();
    }
    TrackDiag(mCurrentFrame, "local_projected");

    {
        ScopedTrackingTimer timer(timing.pose_optimization_ms);
        if (!mAdaptiveLineConfig.enabled) {
            Optimizer::PoseOptimization(&mCurrentFrame);
        } else {
            PoseConstraintSummary point_summary;
            {
                ScopedTrackingTimer analysis_timer(
                    mLastMultiCameraTiming.track.point_constraint_analysis_ms);
                ScopedTrackingTimer evaluation_timer(
                    mLastMultiCameraTiming.track
                        .point_constraint_evaluation_ms);
                point_summary = Optimizer::EvaluatePointConstraints(
                    &mCurrentFrame, mAdaptiveLineConfig);
            }
            const bool needs_line_fallback =
                !point_summary.point_constraints_strong;
            mCurrentPoseConstraint = point_summary;
            if (needs_line_fallback) {
                if (!mCurrentFrame.HasLineFeatures()) {
                    ScopedTrackingTimer line_timer(
                        mLastMultiCameraTiming.track.deferred_line_tracking_ms);
                    if (mCurrentFrame.EnsureLineFeatures(
                            Frame::LineFeatureState::TrackingReady)) {
                        UpdateLocalLines();
                        MatchLocalLines();
                    }
                }
                mbCurrentFrameUsesLines = mCurrentFrame.HasLineFeatures();
                if (mbCurrentFrameUsesLines) {
                    ScopedTrackingTimer selection_timer(
                        mLastMultiCameraTiming.track
                            .line_selection_optimization_ms);
                    ScopedTrackingTimer line_selection_timer(
                        mLastMultiCameraTiming.track.line_selection_ms);
                    mCurrentPoseConstraint = Optimizer::SelectLineConstraints(
                        &mCurrentFrame, mAdaptiveLineConfig, point_summary);
                }
            } else {
                mbCurrentFrameUsesLines = false;
            }
            ScopedTrackingTimer optimization_timer(
                mbCurrentFrameUsesLines ? mLastMultiCameraTiming.track
                                              .line_selection_optimization_ms
                                        : mLastMultiCameraTiming.track
                                              .point_constraint_analysis_ms);
            Optimizer::PoseOptimizationSelectedLines(
                &mCurrentFrame, mAdaptiveLineConfig, &mCurrentPoseConstraint);
        }
    }

    {
        ScopedTrackingTimer timer(timing.inlier_update_ms);
        mnMatchesInliers = 0;
        mnPointMatchesInliers = 0;
        mnLineMatchesInliers = 0;

        for (int c = 0; c < mnCam; ++c) {
            for (int i = 0; i < mCurrentFrame.Np[c]; ++i) {
                if (mCurrentFrame.mvpMapPoints_cams[c][i]) {
                    if (!mCurrentFrame.mvbOutlierPoint_cams[c][i]) {
                        mCurrentFrame.mvpMapPoints_cams[c][i]->IncreaseFound();
                        if (!mbOnlyTracking) {
                            if (mCurrentFrame.mvpMapPoints_cams[c][i]
                                    ->Observations() > 0)
                                ++mnPointMatchesInliers;
                        } else
                            ++mnPointMatchesInliers;
                    } else
                        mCurrentFrame.mvpMapPoints_cams[c][i] = nullptr;
                }
            }

            for (int i = 0; i < mCurrentFrame.Nl[c]; ++i) {
                if (mCurrentFrame.mvpMapLines_cams[c][i]) {
                    if (!mCurrentFrame.mvbOutlierLine_cams[c][i]) {
                        mCurrentFrame.mvpMapLines_cams[c][i]->IncreaseFound();
                        const bool selected =
                            !mAdaptiveLineConfig.enabled ||
                            (c < static_cast<int>(
                                     mCurrentFrame.mvbUseLineConstraint_cams
                                         .size()) &&
                             i < static_cast<int>(
                                     mCurrentFrame.mvbUseLineConstraint_cams[c]
                                         .size()) &&
                             mCurrentFrame.mvbUseLineConstraint_cams[c][i]);
                        if (selected) {
                            if (!mbOnlyTracking) {
                                if (mCurrentFrame.mvpMapLines_cams[c][i]
                                        ->Observations() > 0)
                                    ++mnLineMatchesInliers;
                            } else {
                                ++mnLineMatchesInliers;
                            }
                        }
                    } else
                        mCurrentFrame.mvpMapLines_cams[c][i] = nullptr;
                }
            }
        }

        mnMatchesInliers = mnPointMatchesInliers + mnLineMatchesInliers;

        mpLocalMapper->mnMatchesInliers = mnMatchesInliers;
    }

    {
        ScopedTrackingTimer timer(timing.acceptance_check_ms);
        TrackDiag(mCurrentFrame, "local_optimized", -1, mnPointMatchesInliers);
        if (!IsPlausibleFrameStep("local map"))
            return false;
        if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames &&
            mnMatchesInliers < 50)
            return false;

        if (mnMatchesInliers > 10 && mState == RECENTLY_LOST)
            return true;

        if (mnMatchesInliers < 30)
            return false;
    }

    return true;
}

void Tracking::UpdateLocalMap() {
    auto &timing = mLastMultiCameraTiming.track.local_map;
    {
        ScopedTrackingTimer timer(timing.reference_map_update_ms);
        mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);
        if (mbCurrentFrameUsesLines)
            mpAtlas->SetReferenceMapLines(mvpLocalMapLines);
    }
    {
        ScopedTrackingTimer timer(timing.local_keyframes_update_ms);
        UpdateLocalKeyFrames();
    }
    {
        ScopedTrackingTimer timer(timing.local_points_update_ms);
        UpdateLocalPoints();
    }
    {
        ScopedTrackingTimer timer(timing.local_lines_update_ms);
        if (mbCurrentFrameUsesLines)
            UpdateLocalLines();
    }
}

void Tracking::UpdateLocalKeyFrames() {
    map<KeyFramePtr, int> keyframeCounter;

    for (int cam_id = 0; cam_id < mnCam; ++cam_id) {
        for (int i = 0; i < mCurrentFrame.Np[cam_id]; ++i) {
            MapPointPtr pMP = mCurrentFrame.mvpMapPoints_cams[cam_id][i];
            if (pMP) {
                if (!pMP->isBad()) {
                    const map<KeyFramePtr, vector<int>> observations =
                        pMP->GetObservations();
                    for (map<KeyFramePtr, vector<int>>::const_iterator
                             it = observations.begin(),
                             itend = observations.end();
                         it != itend; ++it) {
                        keyframeCounter[it->first]++;
                    }
                } else {
                    mCurrentFrame.mvpMapPoints_cams[cam_id][i] = nullptr;
                }
            }
        }

        if (!mbCurrentFrameUsesLines)
            continue;
        for (int i = 0; i < mCurrentFrame.Nl[cam_id]; ++i) {
            MapLinePtr pML = mCurrentFrame.mvpMapLines_cams[cam_id][i];
            if (pML) {
                if (!pML->isBad()) {
                    const map<KeyFramePtr, vector<int>> observations =
                        pML->GetObservations();
                    for (map<KeyFramePtr, vector<int>>::const_iterator
                             it = observations.begin(),
                             itend = observations.end();
                         it != itend; ++it) {
                        keyframeCounter[it->first]++;
                    }
                } else {
                    mCurrentFrame.mvpMapLines_cams[cam_id][i] = nullptr;
                }
            }
        }
    }

    int max = 0;
    KeyFramePtr pKFmax = nullptr;

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map.
    // Also check which keyframe shares most points
    for (map<KeyFramePtr, int>::const_iterator it = keyframeCounter.begin(),
                                               itEnd = keyframeCounter.end();
         it != itEnd; it++) {
        KeyFramePtr pKF = it->first;

        if (pKF->isBad())
            continue;

        if (it->second > max) {
            max = it->second;
            pKFmax = pKF;
        }

        mvpLocalKeyFrames.push_back(pKF);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    // Include also some not-already-included keyframes that are neighbors to
    // already-included keyframes
    for (vector<KeyFramePtr>::const_iterator itKF = mvpLocalKeyFrames.begin(),
                                             itEndKF = mvpLocalKeyFrames.end();
         itKF != itEndKF; itKF++) {
        // Limit the number of keyframes
        if (mvpLocalKeyFrames.size() > 80) // 80
            break;

        KeyFramePtr pKF = *itKF;

        const vector<KeyFramePtr> vNeighs =
            pKF->GetBestCovisibilityKeyFrames(10);

        for (vector<KeyFramePtr>::const_iterator itNeighKF = vNeighs.begin(),
                                                 itEndNeighKF = vNeighs.end();
             itNeighKF != itEndNeighKF; itNeighKF++) {
            KeyFramePtr pNeighKF = *itNeighKF;
            if (!pNeighKF->isBad()) {
                if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFramePtr> spChilds = pKF->GetChilds();
        for (set<KeyFramePtr>::const_iterator sit = spChilds.begin(),
                                              send = spChilds.end();
             sit != send; sit++) {
            KeyFramePtr pChildKF = *sit;
            if (!pChildKF->isBad()) {
                if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFramePtr pParent = pKF->GetParent();
        if (pParent) {
            if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                break;
            }
        }
    }

    // cout << "Local KeyFrames: " << mvpLocalKeyFrames.size() << endl;

    if (pKFmax) {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

void Tracking::UpdateLocalPoints() {
    mvpLocalMapPoints.clear();

    for (vector<KeyFramePtr>::const_iterator itKF = mvpLocalKeyFrames.begin(),
                                             itEndKF = mvpLocalKeyFrames.end();
         itKF != itEndKF; itKF++) {
        KeyFramePtr pKF = *itKF;

        const vector<vector<MapPointPtr>> vpMPs = pKF->GetMapPointMatches();

        for (size_t cam_id = 0; cam_id < vpMPs.size(); ++cam_id) {
            for (vector<MapPointPtr>::const_iterator
                     itMP = vpMPs[cam_id].begin(),
                     itEndMP = vpMPs[cam_id].end();
                 itMP != itEndMP; itMP++) {
                MapPointPtr pMP = *itMP;
                if (!pMP)
                    continue;
                if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                    continue;
                if (!pMP->isBad()) {
                    mvpLocalMapPoints.push_back(pMP);
                    pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                }
            }
        }
    }
}

void Tracking::UpdateLocalLines() {
    mvpLocalMapLines.clear();

    for (vector<KeyFramePtr>::const_iterator itKF = mvpLocalKeyFrames.begin(),
                                             itEndKF = mvpLocalKeyFrames.end();
         itKF != itEndKF; itKF++) {
        KeyFramePtr pKF = *itKF;

        const vector<vector<MapLinePtr>> vpMLs = pKF->GetMapLineMatches();

        for (size_t cam_id = 0; cam_id < vpMLs.size(); ++cam_id) {
            for (vector<MapLinePtr>::const_iterator
                     itML = vpMLs[cam_id].begin(),
                     itEndML = vpMLs[cam_id].end();
                 itML != itEndML; itML++) {
                MapLinePtr pML = *itML;
                if (!pML)
                    continue;
                if (pML->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                    continue;
                if (!pML->isBad()) {
                    mvpLocalMapLines.push_back(pML);
                    pML->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                }
            }
        }
    }
}

void Tracking::SearchLocalFeatures() {
    RuntimePhaseScope runtime_phase(RuntimePhase::Matching);
    auto &timing = mLastMultiCameraTiming.track.local_map.search;
    timing = {};
    ScopedTrackingTimer total_timer(timing.total_ms);

    {
        ScopedTrackingTimer timer(timing.current_feature_cleanup_ms);
        for (int cam_id = 0; cam_id < mnCam; ++cam_id) {
            for (vector<MapPointPtr>::iterator
                     vit = mCurrentFrame.mvpMapPoints_cams[cam_id].begin(),
                     vend = mCurrentFrame.mvpMapPoints_cams[cam_id].end();
                 vit != vend; ++vit) {
                MapPointPtr pMP = *vit;
                if (pMP) {
                    if (pMP->isBad()) {
                        *vit = nullptr;
                    } else {
                        pMP->IncreaseVisible();
                        pMP->mvnLastFrameSeen[cam_id] = mCurrentFrame.mnId;
                        pMP->mvbTrackInView[cam_id] = false;
                    }
                }
            }

            if (!mbCurrentFrameUsesLines)
                continue;
            for (vector<MapLinePtr>::iterator
                     vit = mCurrentFrame.mvpMapLines_cams[cam_id].begin(),
                     vend = mCurrentFrame.mvpMapLines_cams[cam_id].end();
                 vit != vend; ++vit) {
                MapLinePtr pML = *vit;
                if (pML) {
                    if (pML->isBad()) {
                        *vit = nullptr;
                    } else {
                        pML->IncreaseVisible();
                        pML->mvnLastFrameSeen[cam_id] = mCurrentFrame.mnId;
                        pML->mvbTrackInView[cam_id] = false;
                    }
                }
            }
        }
    }

    int nToMatch = 0;
    {
        ScopedTrackingTimer timer(timing.point_frustum_check_ms);
        for (vector<MapPointPtr>::iterator vit = mvpLocalMapPoints.begin(),
                                           vend = mvpLocalMapPoints.end();
             vit != vend; ++vit) {
            MapPointPtr pMP = *vit;

            if (pMP->isBad())
                continue;

            for (int cam_id = 0; cam_id < mnCam; ++cam_id) {
                if (pMP->mvnLastFrameSeen[cam_id] == mCurrentFrame.mnId)
                    continue;
                if (mCurrentFrame.isPointInFrustum(pMP, 0.5, cam_id)) {
                    pMP->IncreaseVisible();
                    nToMatch++;
                }
            }
        }
    }

    if (mbCurrentFrameUsesLines) {
        ScopedTrackingTimer timer(timing.line_frustum_check_ms);
        for (vector<MapLinePtr>::iterator vit = mvpLocalMapLines.begin(),
                                          vend = mvpLocalMapLines.end();
             vit != vend; ++vit) {
            MapLinePtr pML = *vit;

            if (pML->isBad())
                continue;

            for (int cam_id = 0; cam_id < mnCam; ++cam_id) {
                if (pML->mvnLastFrameSeen[cam_id] == mCurrentFrame.mnId)
                    continue;
                if (mCurrentFrame.isLineInFrustum(pML, cam_id)) {
                    pML->IncreaseVisible();
                    nToMatch++;
                }
            }
        }
    }

    {
        ScopedTrackingTimer timer(timing.projection_matching_ms);
        if (nToMatch > 0) {
            Matcher matcher(0.8);
            float th_point = 2, th_line = 1;

            if (mCurrentFrame.mnId < mnLastRelocFrameId + 2) {
                th_point = 5;
                th_line = 1;
            }

            if (mState == LOST || mState == RECENTLY_LOST) {
                th_point = 15;
                th_line = 2;
            }

            static const vector<MapLinePtr> no_lines;
            const vector<MapLinePtr> &lines =
                mbCurrentFrameUsesLines ? mvpLocalMapLines : no_lines;
            matcher.SearchByProjection(
                mCurrentFrame, mvpLocalMapPoints, lines, th_point, th_line,
                mpLocalMapper->mbFarPoints, mpLocalMapper->mThFarPoints,
                mpLocalMapper->mbFarLines, mpLocalMapper->mThFarLines);
        }
    }
}

void Tracking::MatchLocalLines() {
    RuntimePhaseScope runtime_phase(RuntimePhase::Matching);
    if (!mCurrentFrame.HasLineFeatures()) {
        return;
    }
    int lines_in_view = 0;
    for (MapLinePtr line : mvpLocalMapLines) {
        if (!line || line->isBad()) {
            continue;
        }
        for (int cam = 0; cam < mnCam; ++cam) {
            if (line->mvnLastFrameSeen[cam] == mCurrentFrame.mnId) {
                continue;
            }
            if (mCurrentFrame.isLineInFrustum(line, cam)) {
                ++lines_in_view;
            }
        }
    }
    if (lines_in_view == 0) {
        return;
    }
    Matcher matcher(0.8);
    static const vector<MapPointPtr> no_points;
    matcher.SearchByProjection(
        mCurrentFrame, no_points, mvpLocalMapLines, 2.0f, 1.0f,
        mpLocalMapper->mbFarPoints, mpLocalMapper->mThFarPoints,
        mpLocalMapper->mbFarLines, mpLocalMapper->mThFarLines);
}

void Tracking::PrepareCurrentFrameLinesForKeyFrame() {
    RuntimeReasonScope runtime_reason(RuntimeLineReason::KeyFrame);
    const bool had_tracking_lines = mCurrentFrame.HasLineFeatures();
    if (!mCurrentFrame.EnsureLineFeatures(
            Frame::LineFeatureState::StereoReady)) {
        return;
    }

    // A strong point-only frame has not searched the local line map. Associate
    // lines at the already accepted point pose, but deliberately do not run a
    // joint pose update here.
    if (!had_tracking_lines) {
        UpdateLocalLines();
        MatchLocalLines();
        Optimizer::ScreenLineAssociationsAtFixedPose(
            &mCurrentFrame, mAdaptiveLineConfig.line_precheck_chi2);
        for (auto &camera_mask : mCurrentFrame.mvbUseLineConstraint_cams) {
            camera_mask.clear();
        }
    }
}

bool Tracking::NeedNewKeyFrame() {
    if (mbOnlyTracking)
        return false;

    if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpAtlas->KeyFramesInMap();

    if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames &&
        nKFs > mMaxFrames)
        return false;

    int nMinObs = 3;
    if (nKFs <= 2)
        nMinObs = 2;

    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);
    if (mbCurrentFrameUsesLines)
        nRefMatches += mpReferenceKF->TrackedMapLines(nMinObs);

    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    float thRefRatio = 0.75;

    const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;

    const bool c1b = (mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames) &&
                     bLocalMappingIdle;
    const bool c1c = mnMatchesInliers < nRefMatches * 0.25;

    const bool c2 =
        (mnMatchesInliers < nRefMatches * thRefRatio) && mnMatchesInliers > 15;

    if ((c1a || c1b || c1c) && c2) {
        if (bLocalMappingIdle)
            return true;
        else {
            mpLocalMapper->InterruptBA();
            if (mpLocalMapper->KeyframesInQueue() < 3)
                return true;
            else {
                return false;
            }
        }
    } else {
        return false;
    }
}

void Tracking::CreateNewKeyFrame() {
    if (!mpLocalMapper->SetNotStop(true))
        return;

    PrepareCurrentFrameLinesForKeyFrame();

    KeyFramePtr pKF = std::make_shared<KeyFrame>(
        mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);

    if (runtime_frame) runtime_frame->keyframe_created = true;
    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = mpReferenceKF;

    if (mpLastKeyFrame) {
        pKF->mPrevKF = mpLastKeyFrame;
        mpLastKeyFrame->mNextKF = pKF;
    }

    mCurrentFrame.UpdatePoseMatrices();

    int maxLine = 100;
    int nNewPoints = 0;

    vector<tuple<float, int, int>> vDepthPointIdx;
    vDepthPointIdx.reserve(mCurrentFrame.mnTotalPoints);

    vector<tuple<float, int, int>> vDepthLineIdx;
    vDepthLineIdx.reserve(mCurrentFrame.mnTotalLines);

    for (int pair_id = 0; pair_id < mvStereoPair.size(); ++pair_id) {
        const int cam_id1 = mvStereoPair[pair_id].first;
        const int cam_id2 = mvStereoPair[pair_id].second;

        for (int i = 0; i < mCurrentFrame.Np[cam_id1]; ++i) {
            float z = mCurrentFrame.mvDepthStereoPoint[pair_id][i];
            if (z > 0)
                vDepthPointIdx.emplace_back(z, pair_id, i);
        }

        for (int i = 0; i < mCurrentFrame.Nl[cam_id1]; ++i) {
            float z = mCurrentFrame.mvDepthStereoLine[pair_id][i];
            if (z > 0)
                vDepthLineIdx.emplace_back(z, pair_id, i);
        }
    }

    if (!vDepthPointIdx.empty()) {
        sort(vDepthPointIdx.begin(), vDepthPointIdx.end());
        for (size_t i = 0; i < vDepthPointIdx.size(); ++i) {
            float z = get<0>(vDepthPointIdx[i]);
            if (z > mThDepth)
                break;
            int pair_id = get<1>(vDepthPointIdx[i]);
            int idx = get<2>(vDepthPointIdx[i]);

            int cam_id1 = mvStereoPair[pair_id].first;
            int cam_id2 = mvStereoPair[pair_id].second;

            int second_idx =
                mCurrentFrame.mvFirstToSecondPointMatch[pair_id][idx];

            if (second_idx == -1) {
                continue;
            }

            MapPointPtr pMP1 = mCurrentFrame.mvpMapPoints_cams[cam_id1][idx];
            MapPointPtr pMP2 =
                mCurrentFrame.mvpMapPoints_cams[cam_id2][second_idx];
            if (pMP1 && pMP1->Observations() < 1) {
                pMP1 = nullptr;
                mCurrentFrame.mvpMapPoints_cams[cam_id1][idx] = nullptr;
            }
            if (pMP2 && pMP2->Observations() < 1) {
                pMP2 = nullptr;
                mCurrentFrame.mvpMapPoints_cams[cam_id2][second_idx] = nullptr;
            }

            if (!pMP1 && !pMP2) {
                Eigen::Vector3f x3DW;

                mCurrentFrame.UnprojectStereoPoint(pair_id, idx, x3DW);

                MapPointPtr pNewMP = std::make_shared<MapPoint>(
                    x3DW, pKF, mpAtlas->GetCurrentMap());
                pNewMP->AddObservation(pKF, idx, cam_id1);
                pNewMP->AddObservation(pKF, second_idx, cam_id2);
                pKF->AddMapPoint(pNewMP, idx, cam_id1);
                pKF->AddMapPoint(pNewMP, second_idx, cam_id2);

                pNewMP->ComputeDistinctiveDescriptors();
                mpAtlas->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints_cams[cam_id1][idx] = pNewMP;
                mCurrentFrame.mvpMapPoints_cams[cam_id2][second_idx] = pNewMP;

                nNewPoints++;
            } else if (pMP1 && !pMP2) {
                if (pMP1->HasObservation(pKF, cam_id2))
                    continue;

                mCurrentFrame.mvpMapPoints_cams[cam_id2][second_idx] = pMP1;
                pMP1->AddObservation(pKF, second_idx, cam_id2);
                pKF->AddMapPoint(pMP1, second_idx, cam_id2);
                pMP1->ComputeDistinctiveDescriptors();
            } else if (!pMP1 && pMP2) {
                if (pMP2->HasObservation(pKF, cam_id1))
                    continue;

                mCurrentFrame.mvpMapPoints_cams[cam_id1][idx] = pMP2;
                pMP2->AddObservation(pKF, idx, cam_id1);
                pKF->AddMapPoint(pMP2, idx, cam_id1);
                pMP2->ComputeDistinctiveDescriptors();
            }

        }
    }

    if (!vDepthLineIdx.empty()) {
        sort(vDepthLineIdx.begin(), vDepthLineIdx.end());
        int nLines = 0;

        for (size_t i = 0; i < vDepthLineIdx.size(); ++i) {
            float z = get<0>(vDepthLineIdx[i]);
            if (z > mThDepth)
                break;
            int pair_id = get<1>(vDepthLineIdx[i]);
            int idx = get<2>(vDepthLineIdx[i]);

            int cam_id1 = mvStereoPair[pair_id].first;
            int cam_id2 = mvStereoPair[pair_id].second;

            int second_idx =
                mCurrentFrame.mvFirstToSecondLineMatch[pair_id][idx];

            if (second_idx == -1) {
                continue;
            }

            MapLinePtr pML1 = mCurrentFrame.mvpMapLines_cams[cam_id1][idx];
            MapLinePtr pML2 =
                mCurrentFrame.mvpMapLines_cams[cam_id2][second_idx];

            if (pML1 && pML1->Observations() < 1) {
                pML1 = nullptr;
                mCurrentFrame.mvpMapLines_cams[cam_id1][idx] = nullptr;
            }
            if (pML2 && pML2->Observations() < 1) {
                pML2 = nullptr;
                mCurrentFrame.mvpMapLines_cams[cam_id2][second_idx] = nullptr;
            }

            if (!pML1 && !pML2) {
                Eigen::Matrix<float, 6, 1> line3DW;

                mCurrentFrame.UnprojectStereoLine(pair_id, idx, line3DW);

                MapLinePtr pNewML = std::make_shared<MapLine>(
                    line3DW, pKF, mpAtlas->GetCurrentMap());
                pNewML->AddObservation(pKF, idx, cam_id1);
                pNewML->AddObservation(pKF, second_idx, cam_id2);
                pKF->AddMapLine(pNewML, idx, cam_id1);
                pKF->AddMapLine(pNewML, second_idx, cam_id2);

                pNewML->ComputeDistinctiveDescriptors();
                mpAtlas->AddMapLine(pNewML);

                mCurrentFrame.mvpMapLines_cams[cam_id1][idx] = pNewML;
                mCurrentFrame.mvpMapLines_cams[cam_id2][second_idx] = pNewML;

                nLines++;
            } else if (pML1 && !pML2) {
                if (pML1->HasObservation(pKF, cam_id2))
                    continue;

                mCurrentFrame.mvpMapLines_cams[cam_id2][second_idx] = pML1;
                pML1->AddObservation(pKF, second_idx, cam_id2);
                pKF->AddMapLine(pML1, second_idx, cam_id2);
                pML1->ComputeDistinctiveDescriptors();
            } else if (!pML1 && pML2) {
                if (pML2->HasObservation(pKF, cam_id1))
                    continue;

                mCurrentFrame.mvpMapLines_cams[cam_id1][idx] = pML2;
                pML2->AddObservation(pKF, idx, cam_id1);
                pKF->AddMapLine(pML2, idx, cam_id1);
                pML2->ComputeDistinctiveDescriptors();
            }

            // if (nLines > maxLine)
            //     break;
        }
    }

    TrackDiag(mCurrentFrame, "new_keyframe_points", nNewPoints);
    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

bool Tracking::Relocalization() {
    RuntimeReasonScope runtime_reason(RuntimeLineReason::Relocalization);
    if (runtime_frame) ++runtime_frame->relocalization_calls;
    mCurrentFrame.ComputeBoW();

    const bool use_relocalization_lines =
        !mAdaptiveLineConfig.enabled || mCurrentFrame.HasLineFeatures();

    vector<KeyFramePtr> vpCandidateKFs =
        mpKeyFrameDB->DetectRelocalizationCandidates(
            &mCurrentFrame, mpAtlas->GetCurrentMap(), use_relocalization_lines);

    // cout << "Relocalization candidates: " << vpCandidateKFs.size() << endl;

    if (vpCandidateKFs.empty()) {
        if (mAdaptiveLineConfig.enabled && !mCurrentFrame.HasLineFeatures() &&
            mCurrentFrame.EnsureLineFeatures(
                Frame::LineFeatureState::TrackingReady)) {
            mbCurrentFrameUsesLines = true;
            return Relocalization();
        }
        cout << "Relocalization failed: No candidates found" << endl;
        return false;
    }

    const int nKFs = vpCandidateKFs.size();

    Matcher matcher(0.75);

    vector<UPnPLSolver *> vpUPnPLsolvers;
    vpUPnPLsolvers.resize(nKFs);

    vector<vector<vector<MapPointPtr>>> vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<vector<vector<MapLinePtr>>> vvpMapLineMatches;
    vvpMapLineMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates = 0;

    for (int i = 0; i < nKFs; ++i) {
        KeyFramePtr pKF = vpCandidateKFs[i];

        if (pKF->isBad())
            vbDiscarded[i] = true;
        else {
            int nmatches =
                matcher.SearchByBow(pKF, mCurrentFrame, vvpMapPointMatches[i],
                                    vvpMapLineMatches[i]);
            // cout << "Candidate " << i << ": " << nmatches << " matches" <<
            // endl;
            if (nmatches < 15) {
                vbDiscarded[i] = true;
                continue;
            } else {
                UPnPLSolver *pSolver = new UPnPLSolver(
                    mCurrentFrame, vvpMapPointMatches[i], vvpMapLineMatches[i]);
                pSolver->SetRansacParameters(0.99, 10, 100, 6, 0.5, 5.991);
                vpUPnPLsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    if (nCandidates == 0 && mAdaptiveLineConfig.enabled &&
        !mCurrentFrame.HasLineFeatures() &&
        mCurrentFrame.EnsureLineFeatures(
            Frame::LineFeatureState::TrackingReady)) {
        mbCurrentFrameUsesLines = true;
        return Relocalization();
    }

    bool bMatch = false;
    Matcher matcher2(0.9);
    auto optimize_candidate_pose = [this](KeyFramePtr candidate) {
        if (!mAdaptiveLineConfig.enabled) {
            return Optimizer::PoseOptimization(&mCurrentFrame);
        }
        OptimizeCurrentPoseAdaptive(candidate);
        return mCurrentPoseConstraint.point_inliers +
               mCurrentPoseConstraint.selected_line_inliers;
    };

    while (nCandidates > 0 && !bMatch) {
        for (int i = 0; i < nKFs; ++i) {
            if (vbDiscarded[i])
                continue;

            vector<vector<bool>> vbPointInliers, vbLineInliers;
            int nInliers;
            bool bNoMore;

            UPnPLSolver *pSolver = vpUPnPLsolvers[i];
            Eigen::Matrix4f eigTbw;
            bool bTbw = pSolver->iterate(5, bNoMore, vbPointInliers,
                                         vbLineInliers, nInliers, eigTbw);

            if (bNoMore) {
                vbDiscarded[i] = true;
                nCandidates--;
            }

            if (bTbw) {
                Sophus::SE3f Tbw(eigTbw);
                mCurrentFrame.SetPose(Tbw);

                set<MapPointPtr> sFoundMapPoints;
                set<MapLinePtr> sFoundMapLines;
                for (size_t cam_id = 0; cam_id < vvpMapPointMatches[i].size();
                     ++cam_id) {
                    for (size_t j = 0; j < vvpMapPointMatches[i][cam_id].size();
                         ++j) {
                        if (vbPointInliers[cam_id][j]) {
                            mCurrentFrame.mvpMapPoints_cams[cam_id][j] =
                                vvpMapPointMatches[i][cam_id][j];
                            sFoundMapPoints.insert(
                                vvpMapPointMatches[i][cam_id][j]);
                        } else {
                            mCurrentFrame.mvpMapPoints_cams[cam_id][j] =
                                nullptr;
                        }
                    }
                    for (size_t j = 0; j < vvpMapLineMatches[i][cam_id].size();
                         ++j) {
                        if (vbLineInliers[cam_id][j]) {
                            mCurrentFrame.mvpMapLines_cams[cam_id][j] =
                                vvpMapLineMatches[i][cam_id][j];
                            sFoundMapLines.insert(
                                vvpMapLineMatches[i][cam_id][j]);
                        } else {
                            mCurrentFrame.mvpMapLines_cams[cam_id][j] = nullptr;
                        }
                    }
                }

                int nGood = optimize_candidate_pose(vpCandidateKFs[i]);

                if (nGood < 10)
                    continue;

                for (int cam_id = 0; cam_id < mnCam; ++cam_id) {
                    for (int j = 0; j < mCurrentFrame.Np[cam_id]; ++j) {
                        if (mCurrentFrame.mvbOutlierPoint_cams[cam_id][j])
                            mCurrentFrame.mvpMapPoints_cams[cam_id][j] =
                                nullptr;
                    }
                    for (int j = 0; j < mCurrentFrame.Nl[cam_id]; ++j) {
                        if (mCurrentFrame.mvbOutlierLine_cams[cam_id][j])
                            mCurrentFrame.mvpMapLines_cams[cam_id][j] = nullptr;
                    }
                }

                if (nGood < 50) {
                    int nadditional = matcher2.SearchByProjection(
                        mCurrentFrame, vpCandidateKFs[i], sFoundMapPoints,
                        sFoundMapLines, 10, 10, 0.12);

                    if (nadditional + nGood >= 50) {
                        nGood = optimize_candidate_pose(vpCandidateKFs[i]);

                        if (nGood > 30 && nGood < 50) {
                            sFoundMapPoints.clear();
                            sFoundMapLines.clear();
                            for (int cam_id = 0; cam_id < mnCam; ++cam_id) {
                                for (int j = 0; j < mCurrentFrame.Np[cam_id];
                                     ++j) {
                                    if (mCurrentFrame
                                            .mvpMapPoints_cams[cam_id][j]) {
                                        sFoundMapPoints.insert(
                                            mCurrentFrame
                                                .mvpMapPoints_cams[cam_id][j]);
                                    }
                                }
                                for (int j = 0; j < mCurrentFrame.Nl[cam_id];
                                     ++j) {
                                    if (mCurrentFrame
                                            .mvpMapLines_cams[cam_id][j]) {
                                        sFoundMapLines.insert(
                                            mCurrentFrame
                                                .mvpMapLines_cams[cam_id][j]);
                                    }
                                }
                            }
                            nadditional = matcher2.SearchByProjection(
                                mCurrentFrame, vpCandidateKFs[i],
                                sFoundMapPoints, sFoundMapLines, 3, 3, 0.08);

                            if (nadditional + nGood >= 50) {
                                nGood =
                                    optimize_candidate_pose(vpCandidateKFs[i]);
                                for (int cam_id = 0; cam_id < mnCam; ++cam_id) {
                                    for (int j = 0;
                                         j < mCurrentFrame.Np[cam_id]; ++j) {
                                        if (mCurrentFrame
                                                .mvbOutlierPoint_cams[cam_id]
                                                                     [j])
                                            mCurrentFrame
                                                .mvpMapPoints_cams[cam_id][j] =
                                                nullptr;
                                    }
                                    for (int j = 0;
                                         j < mCurrentFrame.Nl[cam_id]; ++j) {
                                        if (mCurrentFrame
                                                .mvbOutlierLine_cams[cam_id][j])
                                            mCurrentFrame
                                                .mvpMapLines_cams[cam_id][j] =
                                                nullptr;
                                    }
                                }
                            }
                        }
                    }
                }

                if (nGood >= 50) {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if (!bMatch) {
        if (mAdaptiveLineConfig.enabled && !mCurrentFrame.HasLineFeatures() &&
            mCurrentFrame.EnsureLineFeatures(
                Frame::LineFeatureState::TrackingReady)) {
            mbCurrentFrameUsesLines = true;
            return Relocalization();
        }
        cout << "Relocalization failed: No valid pose found" << endl;
        return false;
    } else {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        cout << "Relocalized! " << endl;
        return true;
    }
}
}; // namespace XPLSLAM
