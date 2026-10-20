#include "local_mapping.h"
#include "geometric_tools.h"
#include "matcher.h"
#include "map_sparsification.h"
#include "optimizer.h"

#include <algorithm>
#include <cmath>
#include <mutex>

using namespace std;

namespace XPLSLAM {

namespace {

constexpr float kLineReprojectionChi2 = 5.991f;
constexpr float kKeyFrameRedundantRatio = 0.7f;
constexpr int kKeyFrameRedundantObs = 3;

} // namespace

LocalMapping::LocalMapping(System *pSys, Atlas *pAtlas,
                           const std::string &_strSeqName)
    : mpSystem(pSys), mbResetRequested(false), mbResetRequestedActiveMap(false),
      mbFinishRequested(false), mbFinished(true), mpAtlas(pAtlas),
      mbAbortBA(false), mbStopped(false), mbStopRequested(false),
      mbNotStop(false), mbAcceptKeyFrames(true), mIdxInit(0), mScale(1.0),
      mInitSect(0), mbNotBA1(true), mbNotBA2(true), mIdxIteration(0) {
    mnMatchesInliers = 0;

    mTinit = 0.f;

    mNumLM = 0;
    mNumKFCulling = 0;
}

void LocalMapping::SetLoopCloser(LoopClosing *pLoopCloser) {
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker) { mpTracker = pTracker; }

void LocalMapping::SetMapSparsification(
    MapSparsification *map_sparsification, int non_local_threshold) {
    mpMapSparsification = map_sparsification;
    mnSparsificationNonLocalThreshold = non_local_threshold;
}

void LocalMapping::RequestStop() {
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::isStopped() {
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

void LocalMapping::Release() {
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if (mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

void LocalMapping::RequestReset() {
    {
        unique_lock<mutex> lock(mMutexReset);
        cout << "LM: Map reset recieved" << endl;
        mbResetRequested = true;
    }
    cout << "LM: Map reset, waiting..." << endl;

    while (1) {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if (!mbResetRequested)
                break;
        }
        usleep(3000);
    }
    cout << "LM: Map reset, Done!!!" << endl;
}

void LocalMapping::RequestResetActiveMap(Map *pMap) {
    {
        unique_lock<mutex> lock(mMutexReset);
        cout << "LM: Active map reset recieved" << endl;
        mbResetRequestedActiveMap = true;
        mpMapToReset = pMap;
    }
    cout << "LM: Active map reset, waiting..." << endl;

    while (1) {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if (!mbResetRequestedActiveMap)
                break;
        }
        usleep(3000);
    }
    cout << "LM: Active map reset, Done!!!" << endl;
}

bool LocalMapping::stopRequested() {
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

bool LocalMapping::AcceptKeyFrames() {
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::InterruptBA() { mbAbortBA = true; }

bool LocalMapping::SetNotStop(bool flag) {
    unique_lock<mutex> lock(mMutexStop);

    if (flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InsertKeyFrame(KeyFramePtr pKF) {
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA = true;
}

void LocalMapping::SetAcceptKeyFrames(bool flag) {
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames = flag;
}

bool LocalMapping::CheckNewKeyFrames() {
    unique_lock<mutex> lock(mMutexNewKFs);
    return (!mlNewKeyFrames.empty());
}

void LocalMapping::Run() {
    mbFinished = false;

    while (1) {
        SetAcceptKeyFrames(false);

        if (CheckNewKeyFrames()) {
            // chrono::steady_clock::time_point t1 =
            // chrono::steady_clock::now();
            ProcessNewKeyFrame();
            MapFeatureCulling();
            CreateNewMapFeatures();
            // chrono::steady_clock::time_point t2 =
            // chrono::steady_clock::now(); chrono::duration<double> time_used =
            //     chrono::duration_cast<chrono::duration<double>>(t2 - t1);
            // cout << "Local Mapping time cost: " << time_used.count() * 1000
            //      << " ms" << endl;

            mbAbortBA = false;

            int num_FixedKF_BA = 0;
            int num_OptKF_BA = 0;
            int num_MPs_BA = 0;
            int num_edges_BA = 0;
            if (!CheckNewKeyFrames())
                SearchInNeighbors();

            if (!CheckNewKeyFrames() && !stopRequested()) {
                if (mpAtlas->KeyFramesInMap() > 2) {
                    // chrono::steady_clock::time_point t1 =
                    //     chrono::steady_clock::now();
                    Optimizer::LocalBundleAdjustment(
                        mpCurrentKeyFrame, &mbAbortBA,
                        mpCurrentKeyFrame->GetMap(), num_FixedKF_BA,
                        num_OptKF_BA, num_MPs_BA, num_edges_BA);
                    // chrono::steady_clock::time_point t2 =
                    //     chrono::steady_clock::now();
                    // chrono::duration<double> time_used =
                    //     chrono::duration_cast<chrono::duration<double>>(t2 -
                    //                                                     t1);
                    // cout << "Local BA time cost: " << time_used.count() *
                    // 1000
                    //      << " ms" << endl;
                }

                // KeyFrameCulling();
            }

            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);

            if (mpMapSparsification) {
                mlpKeyFramesToSparsification.push_back(mpCurrentKeyFrame);
                Map *current_map = mpCurrentKeyFrame->GetMap();
                for (auto it = mlpKeyFramesToSparsification.begin();
                     it != mlpKeyFramesToSparsification.end();) {
                    const KeyFramePtr keyframe = *it;
                    if (!keyframe || keyframe->isBad() ||
                        keyframe->GetMap() != current_map ||
                        keyframe->isSparsified()) {
                        it = mlpKeyFramesToSparsification.erase(it);
                        continue;
                    }

                    const bool is_local =
                        keyframe->mnBALocalForKF == mpCurrentKeyFrame->mnId ||
                        keyframe->mnBAFixedForKF == mpCurrentKeyFrame->mnId;
                    if (keyframe->UpdateCountInLocalMapping(
                            is_local,
                            mnSparsificationNonLocalThreshold)) {
                        mpMapSparsification->InsertKeyFrame(keyframe);
                        it = mlpKeyFramesToSparsification.erase(it);
                    } else {
                        ++it;
                    }
                }
            }

            // mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
        } else if (Stop()) {
            while (isStopped() && !CheckFinish()) {
                usleep(3000);
            }
            if (CheckFinish()) {
                break;
            }
        }

        ResetIfRequested();
        SetAcceptKeyFrames(true);

        if (CheckFinish())
            break;

        usleep(3000);
    }

    SetFinish();
}

void LocalMapping::ProcessNewKeyFrame() {
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    mpCurrentKeyFrame->ComputeBoW();

    const vector<vector<MapPointPtr>> vpMapPoints =
        mpCurrentKeyFrame->GetMapPointMatches();
    unordered_set<MapPointPtr> spUniqueMPs;

    const vector<vector<MapLinePtr>> vpMapLines =
        mpCurrentKeyFrame->GetMapLineMatches();
    unordered_set<MapLinePtr> spUniqueMLs;

    for (int c = 0; c < mpCurrentKeyFrame->mnCam; ++c) {
        for (size_t i = 0; i < vpMapPoints[c].size(); ++i) {
            MapPointPtr pMP = vpMapPoints[c][i];
            if (pMP) {
                if (!pMP->isBad()) {
                    if (!pMP->HasObservation(mpCurrentKeyFrame, c)) {
                        pMP->AddObservation(mpCurrentKeyFrame, i, c);
                        pMP->ComputeDistinctiveDescriptors();
                    } else if (!spUniqueMPs.count(pMP)) {
                        spUniqueMPs.insert(pMP);
                        mlpRecentAddedMapPoints.push_back(pMP);
                    }
                }
            }
        }

        for (size_t i = 0; i < vpMapLines[c].size(); ++i) {
            MapLinePtr pML = vpMapLines[c][i];
            if (pML) {
                if (!pML->isBad()) {
                    if (!pML->HasObservation(mpCurrentKeyFrame, c)) {
                        pML->AddObservation(mpCurrentKeyFrame, i, c);
                        pML->ComputeDistinctiveDescriptors();
                    } else if (!spUniqueMLs.count(pML)) {
                        spUniqueMLs.insert(pML);
                        mlpRecentAddedMapLines.push_back(pML);
                    }
                }
            }
        }
    }

    mpCurrentKeyFrame->UpdateConnections();

    mpAtlas->AddKeyFrame(mpCurrentKeyFrame);
}

void LocalMapping::MapFeatureCulling() {
    list<MapPointPtr>::iterator lit_point = mlpRecentAddedMapPoints.begin(),
                               lend_point = mlpRecentAddedMapPoints.end();
    list<MapLinePtr>::iterator lit_line = mlpRecentAddedMapLines.begin(),
                              lend_line = mlpRecentAddedMapLines.end();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    int nThObs = 3;

    while (lit_point != lend_point) {
        MapPointPtr pMP = *lit_point;
        if (pMP->isBad()) {
            lit_point = mlpRecentAddedMapPoints.erase(lit_point);
        } else if (pMP->GetFoundRatio() < 0.25f) {
            pMP->SetBadFlag();
            lit_point = mlpRecentAddedMapPoints.erase(lit_point);
        } else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 2 &&
                   pMP->Observations() <= nThObs) {
            pMP->SetBadFlag();
            lit_point = mlpRecentAddedMapPoints.erase(lit_point);
        } else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 3) {
            lit_point = mlpRecentAddedMapPoints.erase(lit_point);
        } else {
            ++lit_point;
        }
    }

    while (lit_line != lend_line) {
        MapLinePtr pML = *lit_line;
        if (pML->isBad()) {
            lit_line = mlpRecentAddedMapLines.erase(lit_line);
        } else if (pML->GetFoundRatio() < 0.25f) {
            pML->SetBadFlag();
            lit_line = mlpRecentAddedMapLines.erase(lit_line);
        } else if (((int)nCurrentKFid - (int)pML->mnFirstKFid) >= 2 &&
                   pML->Observations() <= nThObs) {
            pML->SetBadFlag();
            lit_line = mlpRecentAddedMapLines.erase(lit_line);
        } else if (((int)nCurrentKFid - (int)pML->mnFirstKFid) >= 3) {
            lit_line = mlpRecentAddedMapLines.erase(lit_line);
        } else {
            ++lit_line;
        }
    }
}

void LocalMapping::CreateNewMapFeatures() {
    struct PendingMapPoint {
        MapPointPtr point;
        int current_idx;
        int current_cam;
        int neighbor_idx;
        int neighbor_cam;
    };

    struct PendingMapLine {
        MapLinePtr line;
        int current_idx;
        int current_cam;
        int neighbor_idx;
        int neighbor_cam;
    };

    int nn = 10;

    vector<KeyFramePtr> vpNeighKFs =
        mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    Matcher matcher(0.9f);

    vector<GeometricCamera *> vpCameras1 = mpCurrentKeyFrame->mvpCameras;
    int nCam = vpCameras1.size();

    constexpr int kOptimisticAttempts = 2;
    const auto same_pose = [](const Sophus::SE3f &lhs,
                              const Sophus::SE3f &rhs) {
        return (lhs.matrix().array() == rhs.matrix().array()).all();
    };

    for (size_t i = 0; i < vpNeighKFs.size(); ++i) {
        if (i > 0 && CheckNewKeyFrames())
            return;

        KeyFramePtr pKF2 = vpNeighKFs[i];
        Map *pMap = mpCurrentKeyFrame->GetMap();
        if (!pMap || !pKF2 || pKF2->GetMap() != pMap)
            continue;
        vector<GeometricCamera *> vpCameras2 = pKF2->mvpCameras;
        for (int attempt = 0;; ++attempt) {
            // Normal operation is optimistic: all expensive work uses
            // immutable snapshots without holding the global map lock. If
            // concurrent map updates invalidate two consecutive attempts,
            // fall back to the original serialized behavior to guarantee
            // progress without accepting stale geometry.
            unique_lock<mutex> map_update_lock;
            if (attempt >= kOptimisticAttempts)
                map_update_lock = unique_lock<mutex>(pMap->mMutexMapUpdate);

            Matcher::TriangulationMatches matches;
            matcher.SearchForTriangulation(mpCurrentKeyFrame, pKF2, matches);
            const auto &point_snapshot1 = matches.point_snapshot1;
            const auto &point_snapshot2 = matches.point_snapshot2;
            const vector<Sophus::SE3f> &vsophTcw1 = matches.camera_poses1;
            const vector<Sophus::SE3f> &vsophTcw2 = matches.camera_poses2;
            if (vsophTcw1.size() != static_cast<size_t>(nCam) ||
                vsophTcw2.size() != static_cast<size_t>(nCam)) {
                break;
            }

            vector<Eigen::Matrix<float, 3, 4>> veigTcw1(nCam), veigTcw2(nCam);
            vector<Eigen::Matrix3f> vRcw1(nCam), vRwc1(nCam);
            vector<Eigen::Matrix3f> vRcw2(nCam), vRwc2(nCam);
            vector<Eigen::Vector3f> vtcw1(nCam), vtcw2(nCam);
            vector<Eigen::Vector3f> vOw1(nCam), vOw2(nCam);
            for (int c = 0; c < nCam; ++c) {
                veigTcw1[c] = vsophTcw1[c].matrix3x4();
                veigTcw2[c] = vsophTcw2[c].matrix3x4();
                vRcw1[c] = veigTcw1[c].block<3, 3>(0, 0);
                vRcw2[c] = veigTcw2[c].block<3, 3>(0, 0);
                vRwc1[c] = vRcw1[c].transpose();
                vRwc2[c] = vRcw2[c].transpose();
                vtcw1[c] = vsophTcw1[c].translation();
                vtcw2[c] = vsophTcw2[c].translation();
                vOw1[c] = vsophTcw1[c].inverse().translation();
                vOw2[c] = vsophTcw2[c].inverse().translation();
            }

            vector<PendingMapPoint> pending_points;
            vector<PendingMapLine> pending_lines;
            pending_points.reserve(matches.point_matches.size());
            pending_lines.reserve(matches.line_matches.size());

            for (const Matcher::TriangulationMatch &match :
                 matches.point_matches) {
                const int cam1 = match.cam1;
                const int idx1_cam = match.idx1;
                const int cam2 = match.cam2;
                const int idx2_cam = match.idx2;

                if ((vOw1[cam1] - vOw2[cam2]).norm() < 0.1f)
                    continue;

                const cv::Point2f &kp1 =
                    point_snapshot1.mvKeysUn_cams[cam1][idx1_cam];
                const cv::Point2f &kp2 =
                    point_snapshot2.mvKeysUn_cams[cam2][idx2_cam];

                Eigen::Vector3f xn1 = vpCameras1[cam1]->unprojectEig(kp1);
                Eigen::Vector3f xn2 = vpCameras2[cam2]->unprojectEig(kp2);

                Eigen::Vector3f ray1 = vRwc1[cam1] * xn1;
                Eigen::Vector3f ray2 = vRwc2[cam2] * xn2;
                const float cosParallaxRays =
                    ray1.dot(ray2) / (ray1.norm() * ray2.norm());

                Eigen::Vector3f x3D;
                bool goodProj = false;
                if (cosParallaxRays > 0 && cosParallaxRays < 0.9998f) {
                    goodProj = GeometricTools::Triangulate(
                        xn1, xn2, veigTcw1[cam1], veigTcw2[cam2], x3D);
                    if (!goodProj)
                        continue;
                } else {
                    continue;
                }

                float z1 = vRcw1[cam1].row(2).dot(x3D) + vtcw1[cam1].z();
                if (z1 <= 0.0)
                    continue;

                float z2 = vRcw2[cam2].row(2).dot(x3D) + vtcw2[cam2].z();
                if (z2 <= 0.0)
                    continue;

                if (mbFarPoints &&
                    (z1 >= mThFarPoints || z2 >= mThFarPoints)) {
                    continue;
                }

                const float x1 =
                    vRcw1[cam1].row(0).dot(x3D) + vtcw1[cam1].x();
                const float y1 =
                    vRcw1[cam1].row(1).dot(x3D) + vtcw1[cam1].y();
                cv::Point2f uv1 =
                    vpCameras1[cam1]->project(cv::Point3f(x1, y1, z1));
                float errX1 = uv1.x - kp1.x;
                float errY1 = uv1.y - kp1.y;
                if ((errX1 * errX1 + errY1 * errY1) > 5.991)
                    continue;

                const float x2 =
                    vRcw2[cam2].row(0).dot(x3D) + vtcw2[cam2].x();
                const float y2 =
                    vRcw2[cam2].row(1).dot(x3D) + vtcw2[cam2].y();
                cv::Point2f uv2 =
                    vpCameras2[cam2]->project(cv::Point3f(x2, y2, z2));
                float errX2 = uv2.x - kp2.x;
                float errY2 = uv2.y - kp2.y;
                if ((errX2 * errX2 + errY2 * errY2) > 5.991)
                    continue;

                Eigen::Vector3f normal1 = x3D - vOw1[cam1];
                Eigen::Vector3f normal2 = x3D - vOw2[cam2];
                if (normal1.norm() == 0 || normal2.norm() == 0)
                    continue;

                MapPointPtr pMP =
                    std::make_shared<MapPoint>(x3D, mpCurrentKeyFrame, pMap);
                pMP->AddObservation(mpCurrentKeyFrame, idx1_cam, cam1);
                pMP->AddObservation(pKF2, idx2_cam, cam2);
                pending_points.push_back(
                    {pMP, idx1_cam, cam1, idx2_cam, cam2});
            }

            for (const Matcher::TriangulationMatch &match :
                 matches.line_matches) {
                const int cam1 = match.cam1;
                const int idx1_cam = match.idx1;
                const int cam2 = match.cam2;
                const int idx2_cam = match.idx2;

                if ((vOw1[cam1] - vOw2[cam2]).norm() < 0.1f)
                    continue;

                const Eigen::Vector4f &line1 =
                    mpCurrentKeyFrame->mvLineKeysUn_cams[cam1][idx1_cam];
                const Eigen::Vector4f &line2 =
                    pKF2->mvLineKeysUn_cams[cam2][idx2_cam];

                Eigen::Matrix<float, 6, 1> line3D;
                if (!GeometricTools::TriangulateLine(
                        line1, vpCameras1[cam1], vsophTcw1[cam1], line2,
                        vpCameras2[cam2], vsophTcw2[cam2], line3D)) {
                    continue;
                }

                if (!GeometricTools::IsLineDepthPositive(
                        line3D, vpCameras1[cam1], vsophTcw1[cam1], line1) ||
                    !GeometricTools::IsLineDepthPositive(
                        line3D, vpCameras2[cam2], vsophTcw2[cam2], line2)) {
                    continue;
                }

                Eigen::Vector2f reprojection_error1;
                Eigen::Vector2f reprojection_error2;
                if (!GeometricTools::ComputeLineReprojectionError(
                        line3D, vpCameras1[cam1], vsophTcw1[cam1], line1,
                        reprojection_error1) ||
                    !GeometricTools::ComputeLineReprojectionError(
                        line3D, vpCameras2[cam2], vsophTcw2[cam2], line2,
                        reprojection_error2)) {
                    continue;
                }

                const float chi2_1 = reprojection_error1.squaredNorm();
                const float chi2_2 = reprojection_error2.squaredNorm();
                if (chi2_1 > kLineReprojectionChi2 ||
                    chi2_2 > kLineReprojectionChi2) {
                    continue;
                }

                const float depth1 = GeometricTools::LineDepthInCamera(
                    line3D, vsophTcw1[cam1]);
                const float depth2 = GeometricTools::LineDepthInCamera(
                    line3D, vsophTcw2[cam2]);
                if (!std::isfinite(depth1) || !std::isfinite(depth2) ||
                    depth1 <= 0.0 || depth2 <= 0.0) {
                    continue;
                }

                if (mbFarLines &&
                    (depth1 >= mThFarLines || depth2 >= mThFarLines)) {
                    continue;
                }

                MapLinePtr pML =
                    std::make_shared<MapLine>(line3D, mpCurrentKeyFrame, pMap);
                pML->AddObservation(mpCurrentKeyFrame, idx1_cam, cam1);
                pML->AddObservation(pKF2, idx2_cam, cam2);
                pML->ComputeDistinctiveDescriptors();
                pending_lines.push_back(
                    {pML, idx1_cam, cam1, idx2_cam, cam2});
            }

            if (!map_update_lock.owns_lock())
                map_update_lock = unique_lock<mutex>(pMap->mMutexMapUpdate);

            if (mpCurrentKeyFrame->isBad() || pKF2->isBad() ||
                mpCurrentKeyFrame->GetMap() != pMap ||
                pKF2->GetMap() != pMap) {
                map_update_lock.unlock();
                break;
            }

            bool snapshot_current =
                mpCurrentKeyFrame->IsPointFeatureSnapshotCurrent(
                    point_snapshot1) &&
                pKF2->IsPointFeatureSnapshotCurrent(point_snapshot2) &&
                same_pose(mpCurrentKeyFrame->GetPose(), matches.body_pose1) &&
                same_pose(pKF2->GetPose(), matches.body_pose2);

            if (snapshot_current) {
                for (const PendingMapPoint &pending : pending_points) {
                    if (mpCurrentKeyFrame->GetMapPoint(
                            pending.current_idx, pending.current_cam) ||
                        pKF2->GetMapPoint(pending.neighbor_idx,
                                         pending.neighbor_cam)) {
                        snapshot_current = false;
                        break;
                    }
                }
            }
            if (snapshot_current) {
                for (const PendingMapLine &pending : pending_lines) {
                    if (mpCurrentKeyFrame->GetMapLine(
                            pending.current_idx, pending.current_cam) ||
                        pKF2->GetMapLine(pending.neighbor_idx,
                                        pending.neighbor_cam)) {
                        snapshot_current = false;
                        break;
                    }
                }
            }

            if (!snapshot_current) {
                map_update_lock.unlock();
                if (i > 0 && CheckNewKeyFrames())
                    return;
                continue;
            }

            for (const PendingMapPoint &pending : pending_points) {
                mpCurrentKeyFrame->AddMapPoint(
                    pending.point, pending.current_idx, pending.current_cam);
                pKF2->AddMapPoint(pending.point, pending.neighbor_idx,
                                  pending.neighbor_cam);
                mpAtlas->AddMapPoint(pending.point);
            }
            for (const PendingMapLine &pending : pending_lines) {
                mpCurrentKeyFrame->AddMapLine(
                    pending.line, pending.current_idx, pending.current_cam);
                pKF2->AddMapLine(pending.line, pending.neighbor_idx,
                                 pending.neighbor_cam);
                mpAtlas->AddMapLine(pending.line);
            }
            map_update_lock.unlock();

            for (const PendingMapPoint &pending : pending_points) {
                pending.point->ComputeDistinctiveDescriptors();
                mlpRecentAddedMapPoints.push_back(pending.point);
            }
            for (const PendingMapLine &pending : pending_lines)
                mlpRecentAddedMapLines.push_back(pending.line);
            break;
        }
    }
}

void LocalMapping::SearchInNeighbors() {
    int nn = 10;
    const vector<KeyFramePtr> vpNeighKFs =
        mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    vector<KeyFramePtr> vpTargetKFs;
    for (vector<KeyFramePtr>::const_iterator vit = vpNeighKFs.begin(),
                                            itend = vpNeighKFs.end();
         vit != itend; ++vit) {
        KeyFramePtr pKFi = *vit;
        if (pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;
    }

    for (int i = 0, imax = vpTargetKFs.size(); i < imax; i++) {
        const vector<KeyFramePtr> vpSecondNeighKFs =
            vpTargetKFs[i]->GetBestCovisibilityKeyFrames(20);
        for (vector<KeyFramePtr>::const_iterator vit2 = vpSecondNeighKFs.begin(),
                                                vend2 = vpSecondNeighKFs.end();
             vit2 != vend2; vit2++) {
            KeyFramePtr pKFi2 = *vit2;
            if (pKFi2->isBad() ||
                pKFi2->mnFuseTargetForKF == mpCurrentKeyFrame->mnId ||
                pKFi2->mnId == mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
            pKFi2->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;
        }
        if (mbAbortBA)
            break;
    }

    Matcher matcher;
    vector<vector<MapPointPtr>> vpMapPointMatches_cam =
        mpCurrentKeyFrame->GetMapPointMatches();
    vector<vector<MapLinePtr>> vpMapLineMatches_cam =
        mpCurrentKeyFrame->GetMapLineMatches();

    for (vector<KeyFramePtr>::const_iterator vit = vpTargetKFs.begin(),
                                            vend = vpTargetKFs.end();
         vit != vend; ++vit) {
        KeyFramePtr pKFi = *vit;

        matcher.Fuse(pKFi, vpMapPointMatches_cam, vpMapLineMatches_cam, 3, 3,
                     0.08);
    }

    if (mbAbortBA)
        return;

    vector<vector<MapPointPtr>> vpFuseCandidatesPoints;
    vector<vector<MapLinePtr>> vpFuseCandidatesLines;
    int nCam = mpCurrentKeyFrame->mnCam;

    vpFuseCandidatesPoints.resize(nCam);
    vpFuseCandidatesLines.resize(nCam);

    for (int c = 0; c < nCam; ++c) {
        vpFuseCandidatesPoints[c].reserve(vpTargetKFs.size() *
                                          vpMapPointMatches_cam[c].size());
        vpFuseCandidatesLines[c].reserve(vpTargetKFs.size() *
                                         vpMapLineMatches_cam[c].size());
    }

    for (vector<KeyFramePtr>::const_iterator vit = vpTargetKFs.begin(),
                                            vend = vpTargetKFs.end();
         vit != vend; ++vit) {
        KeyFramePtr pKFi = *vit;

        vector<vector<MapPointPtr>> vpMapPointsKFi = pKFi->GetMapPointMatches();
        vector<vector<MapLinePtr>> vpMapLinesKFi = pKFi->GetMapLineMatches();

        for (int c = 0; c < nCam; ++c) {
            for (size_t i = 0; i < vpMapPointsKFi[c].size(); ++i) {
                MapPointPtr pMP = vpMapPointsKFi[c][i];
                if (!pMP)
                    continue;
                if (pMP->isBad() ||
                    pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                    continue;
                pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
                vpFuseCandidatesPoints[c].push_back(pMP);
            }

            for (size_t i = 0; i < vpMapLinesKFi[c].size(); ++i) {
                MapLinePtr pML = vpMapLinesKFi[c][i];
                if (!pML)
                    continue;
                if (pML->isBad() ||
                    pML->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                    continue;
                pML->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
                vpFuseCandidatesLines[c].push_back(pML);
            }
        }
    }

    matcher.Fuse(mpCurrentKeyFrame, vpFuseCandidatesPoints,
                 vpFuseCandidatesLines, 3, 3, 0.08);

    vpMapPointMatches_cam = mpCurrentKeyFrame->GetMapPointMatches();
    vpMapLineMatches_cam = mpCurrentKeyFrame->GetMapLineMatches();

    unordered_set<MapPointPtr> spUniqueMPs;
    unordered_set<MapLinePtr> spUniqueMLs;

    for (int c = 0; c < nCam; ++c) {
        for (size_t i = 0; i < vpMapPointMatches_cam[c].size(); ++i) {
            MapPointPtr pMP = vpMapPointMatches_cam[c][i];
            if (!pMP)
                continue;
            if (pMP->isBad())
                continue;
            if (!spUniqueMPs.count(pMP)) {
                spUniqueMPs.insert(pMP);
                pMP->ComputeDistinctiveDescriptors();
            }
        }
        for (size_t i = 0; i < vpMapLineMatches_cam[c].size(); ++i) {
            MapLinePtr pML = vpMapLineMatches_cam[c][i];
            if (!pML)
                continue;
            if (pML->isBad())
                continue;
            if (!spUniqueMLs.count(pML)) {
                spUniqueMLs.insert(pML);
                pML->ComputeDistinctiveDescriptors();
            }
        }
    }

    mpCurrentKeyFrame->UpdateConnections();
}

void LocalMapping::KeyFrameCulling() {
    const int Nd = 21;
    mpCurrentKeyFrame->UpdateBestCovisibles();
    vector<KeyFramePtr> vpLocalKeyFrames =
        mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    int count = 0;

    for (auto vit = vpLocalKeyFrames.begin(), vend = vpLocalKeyFrames.end();
         vit != vend; ++vit) {
        count++;
        KeyFramePtr pKF = *vit;

        if ((pKF->mnId == pKF->GetMap()->GetInitKFid()) || pKF->isBad())
            continue;

        const vector<vector<MapPointPtr>> vpMapPoints =
            pKF->GetMapPointMatches();
        const vector<vector<MapLinePtr>> vpMapLines = pKF->GetMapLineMatches();

        const int thObs = kKeyFrameRedundantObs;
        int nRedundantObservations = 0;
        int nMPs = 0;
        unordered_set<MapPointPtr> spUniqueMPs;
        int nMLs = 0;
        unordered_set<MapLinePtr> spUniqueMLs;

        for (size_t cam_id = 0; cam_id < vpMapPoints.size(); ++cam_id) {
            for (size_t i = 0, iend = vpMapPoints[cam_id].size(); i < iend;
                 ++i) {
                MapPointPtr pMP = vpMapPoints[cam_id][i];
                if (pMP) {
                    if (!pMP->isBad()) {
                        if (spUniqueMPs.count(pMP))
                            continue;
                        spUniqueMPs.insert(pMP);
                        nMPs++;

                        if (pMP->Observations() > thObs) {
                            const float &score =
                                pKF->mvKeyScores_cams[cam_id][i];

                            const map<KeyFramePtr, vector<int>> observations =
                                pMP->GetObservations();
                            int nObs = 0;
                            for (auto mit = observations.begin(),
                                      mend = observations.end();
                                 mit != mend; ++mit) {
                                KeyFramePtr pKFi = mit->first;
                                if (pKFi == pKF || pKFi->isBad())
                                    continue;
                                vector<int> indexes = mit->second;
                                float best_score = 0.f;
                                for (int c = 0; c < indexes.size(); ++c) {
                                    if (indexes[c] != -1) {
                                        float s =
                                            pKFi->mvKeyScores_cams[c]
                                                                  [indexes[c]];
                                        if (s > best_score)
                                            best_score = s;
                                    }
                                }

                                if (best_score > score) {
                                    nObs++;
                                    if (nObs > thObs)
                                        break;
                                }
                            }

                            if (nObs > thObs) {
                                nRedundantObservations++;
                            }
                        }
                    }
                }
            }

            for (size_t i = 0, iend = vpMapLines[cam_id].size(); i < iend;
                 ++i) {
                MapLinePtr pML = vpMapLines[cam_id][i];
                if (pML) {
                    if (!pML->isBad()) {
                        if (spUniqueMLs.count(pML))
                            continue;
                        spUniqueMLs.insert(pML);
                        nMLs++;

                        if (pML->Observations() > thObs) {
                            const float &score =
                                pKF->mvLineScores_cams[cam_id][i];

                            const map<KeyFramePtr, vector<int>> observations =
                                pML->GetObservations();
                            int nObs = 0;
                            for (auto mit = observations.begin(),
                                      mend = observations.end();
                                 mit != mend; ++mit) {
                                KeyFramePtr pKFi = mit->first;
                                if (pKFi == pKF || pKFi->isBad())
                                    continue;
                                vector<int> indexes = mit->second;
                                float best_score = 0.f;
                                for (int c = 0; c < indexes.size(); ++c) {
                                    if (indexes[c] != -1) {
                                        float s =
                                            pKFi->mvLineScores_cams[c]
                                                                   [indexes[c]];
                                        if (s > best_score)
                                            best_score = s;
                                    }
                                }

                                if (best_score > score) {
                                    nObs++;
                                    if (nObs > thObs)
                                        break;
                                }
                            }

                            if (nObs > thObs) {
                                nRedundantObservations++;
                            }
                        }
                    }
                }
            }
        }

        const int nFeatures = nMPs + nMLs;
        if (nFeatures > 0 &&
            nRedundantObservations > kKeyFrameRedundantRatio * nFeatures) {
            pKF->SetBadFlag();
        }

        if ((count > 20 && mbAbortBA) || count > 100)
            break;
    }
}

bool LocalMapping::Stop() {
    unique_lock<mutex> lock(mMutexStop);
    if (mbStopRequested && !mbNotStop) {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::CheckFinish() {
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::RequestFinish() {
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::isFinished() {
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void LocalMapping::ResetIfRequested() {
    bool executed_reset = false;
    {
        unique_lock<mutex> lock(mMutexReset);
        if (mbResetRequested) {
            executed_reset = true;

            cout << "LM: Reseting Atlas in Local Mapping..." << endl;
            mlNewKeyFrames.clear();
            mlpRecentAddedMapPoints.clear();
            mlpRecentAddedMapLines.clear();
            mlpKeyFramesToSparsification.clear();
            mbResetRequested = false;
            mbResetRequestedActiveMap = false;

            // Inertial parameters
            mTinit = 0.f;
            mbNotBA2 = true;
            mbNotBA1 = true;

            mIdxInit = 0;

            cout << "LM: End reseting Local Mapping..." << endl;
        }

        if (mbResetRequestedActiveMap) {
            executed_reset = true;
            cout << "LM: Reseting current map in Local Mapping..." << endl;
            mlNewKeyFrames.clear();
            mlpRecentAddedMapPoints.clear();
            mlpRecentAddedMapLines.clear();
            for (auto it = mlpKeyFramesToSparsification.begin();
                 it != mlpKeyFramesToSparsification.end();) {
                if (!*it || (*it)->GetMap() == mpMapToReset)
                    it = mlpKeyFramesToSparsification.erase(it);
                else
                    ++it;
            }

            // Inertial parameters
            mTinit = 0.f;
            mbNotBA2 = true;
            mbNotBA1 = true;

            mbResetRequested = false;
            mbResetRequestedActiveMap = false;
            cout << "LM: End reseting Local Mapping..." << endl;
        }
    }
    if (executed_reset)
        cout << "LM: Reset free the mutex" << endl;
}

void LocalMapping::SetFinish() {
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

void LocalMapping::EmptyQueue() {
    while (CheckNewKeyFrames())
        ProcessNewKeyFrame();
}

}; // namespace XPLSLAM
