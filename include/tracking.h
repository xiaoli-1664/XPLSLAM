#pragma once

#include "Vocabulary.h"
#include "frame.h"
#include "keyframe_database.h"
#include "settings.h"
#include "system.h"
#include "tracking_timing.h"
#include "xplnet.h"

#include "geometric_camera.h"

#include <mutex>
#include <unordered_set>
#include <utility>

namespace XPLSLAM {

class Atlas;
class LocalMapping;
class LoopClosing;
class System;
class Settings;

class Tracking {

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Tracking(System *pSys, DBoW3::Vocabulary *pvoc_point,
             DBoW3::Vocabulary *pvoc_line, Atlas *pAtlas,
             KeyFrameDatabase *pKFDB, const std::string &strSettingPath,
             Settings *settings, const std::string &_nameSeq = std::string());

    ~Tracking();

    // Parse the config file
    bool ParseCamParamFile(cv::FileStorage &fSettings);

    Sophus::SE3f GrabMultiCamera(const std::vector<cv::Mat> &vIms,
                                 const double &timestamp, std::string filename);

    const MultiCameraTrackingTiming &GetLastMultiCameraTiming() const {
        return mLastMultiCameraTiming;
    }

    void SetLocalMapper(LocalMapping *pLocalMapper);
    void SetLoopClosing(LoopClosing *pLoopClosing);
    void SetStepByStep(bool bSet);
    bool GetStepByStep();

    // Load new settings
    // The focal lenght should be similar or scale prediction will fail when
    // projecting points
    void ChangeCalibration(const std::string &strSettingPath);

    // Use this function if you have deactivated local mapping and you only want
    // to localize the camera.
    void InformOnlyTracking(const bool &flag);

    KeyFramePtr GetLastKeyFrame() { return mpLastKeyFrame; }

    void CreateMapInAtlas();
    // std::mutex mMutexTracks;

    //--
    void NewDataset();
    int GetNumberDataset();
    int GetMatchesInliers();

    // DEBUG
    void SaveSubTrajectory(std::string strNameFile_frames,
                           std::string strNameFile_kf,
                           std::string strFolder = "");
    void SaveSubTrajectory(std::string strNameFile_frames,
                           std::string strNameFile_kf, Map *pMap);

    float GetImageScale();

#ifdef REGISTER_LOOP
    void RequestStop();
    bool isStopped();
    void Release();
    bool stopRequested();
#endif

  public:
    // Tracking states
    enum eTrackingState {
        SYSTEM_NOT_READY = -1,
        NO_IMAGES_YET = 0,
        NOT_INITIALIZED = 1,
        OK = 2,
        RECENTLY_LOST = 3,
        LOST = 4,
        OK_KLT = 5
    };

    eTrackingState mState;
    eTrackingState mLastProcessedState;

    // Current Frame
    Frame mCurrentFrame;
    Frame mLastFrame;

    std::vector<cv::Mat> mvIms;

    // Lists used to recover the full camera trajectory at the end of the
    // execution. Basically we store the reference keyframe for each frame and
    // its relative transformation
    std::list<Sophus::SE3f> mlRelativeFramePoses;
    std::list<KeyFramePtr> mlpReferences;
    std::list<double> mlFrameTimes;
    std::list<bool> mlbLost;

    // frames with estimated pose
    int mTrackedFr;
    bool mbStep;

    // True if local mapping is deactivated and we are performing only
    // localization
    bool mbOnlyTracking;

    void Reset(bool bLocMap = false);
    void ResetActiveMap(bool bLocMap = false);

    float mMeanTrack;
    bool mbInitWith3KFs;
    double t0;    // time-stamp of first read frame
    double t0vis; // time-stamp of first inserted keyframe
    bool mFastInit = false;

    std::vector<MapPointPtr> GetLocalMapMPS();

    bool mbWriteStats;

#ifdef REGISTER_TIMES
    void LocalMapStats2File();
    void TrackStats2File();
    void PrintTimeStats();

    vector<double> vdRectStereo_ms;
    vector<double> vdResizeImage_ms;
    vector<double> vdORBExtract_ms;
    vector<double> vdStereoMatch_ms;
    vector<double> vdIMUInteg_ms;
    vector<double> vdPosePred_ms;
    vector<double> vdLMTrack_ms;
    vector<double> vdNewKF_ms;
    vector<double> vdTrackTotal_ms;
#endif

  protected:
    // Main tracking function. It is independent of the input sensor.
    void Track();

    void StereosInitialization();

    // void CreateNewMapPoints();
    void CreateInitialMapMonocular();

    void CheckReplacedInLastFrame();
    bool TrackReferenceKeyFrame();
    void UpdateLastFrame();
    bool TrackLastFrame();
    bool TrackWithMotionModel();
    bool IsPlausibleFrameStep(const char *stage) const;

    bool Relocalization();

    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalLines();
    void UpdateLocalKeyFrames();

    bool TrackLocalMap();
    void SearchLocalFeatures();

    void OptimizeCurrentPoseAdaptive(KeyFramePtr line_reference);
    void MatchReferenceLines(KeyFramePtr reference);
    void MatchLocalLines();
    void PrepareCurrentFrameLinesForKeyFrame();

    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();

    bool mbMapUpdated;

    int mnCam;

    // In case of performing only localization, this flag is true when there are
    // no matches to points in the map. Still tracking will continue if there
    // are enough matches with temporal points. In that case we are doing visual
    // odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    bool mbVO;

    // Other Thread Pointers
    LocalMapping *mpLocalMapper;
    LoopClosing *mpLoopClosing;

    XPLNet *mpXPLNet;

    // BoW
    DBoW3::Vocabulary *mpVocabulary_point;
    DBoW3::Vocabulary *mpVocabulary_line;
    KeyFrameDatabase *mpKeyFrameDB;

    // Local Map
    KeyFramePtr mpReferenceKF;
    std::vector<KeyFramePtr> mvpLocalKeyFrames;
    std::vector<MapPointPtr> mvpLocalMapPoints;
    std::vector<MapLinePtr> mvpLocalMapLines;

    // System
    System *mpSystem;

    // Drawers
    bool bStepByStep;

    // Atlas
    Atlas *mpAtlas;

    // Calibration matrix
    std::vector<cv::Mat> mvK;
    std::vector<Eigen::Matrix3f> mvK_;
    std::vector<cv::Mat> mvDistCoef;
    float mImageScale;

    bool mInsertKFsLost;

    // New KeyFrame rules (according to fps)
    int mMinFrames;
    int mMaxFrames;

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two
    // keyframes.
    float mThDepth;

    // Current matches in frame
    int mnMatchesInliers;
    int mnPointMatchesInliers = 0;
    int mnLineMatchesInliers = 0;
    AdaptiveLineConfig mAdaptiveLineConfig;
    TrackingRobustnessConfig mTrackingRobustnessConfig;
    PoseConstraintSummary mCurrentPoseConstraint;
    bool mbCurrentFrameUsesLines = false;

    // Last Frame, KeyFrame and Relocalisation Info
    KeyFramePtr mpLastKeyFrame;
    unsigned int mnLastKeyFrameId;
    unsigned int mnLastRelocFrameId;
    double mTimeStampLost;
    double time_recently_lost;

    unsigned int mnFirstFrameId;
    unsigned int mnInitialFrameId;
    unsigned int mnLastInitFrameId;

    bool mbCreatedMap;

    // Motion Model
    bool mbVelocity{false};
    Sophus::SE3f mVelocity;

    // Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;

    std::list<MapPointPtr> mlpTemporalPoints;
    std::list<MapLinePtr> mlpTemporalLines;

    // int nMapChangeIndex;

    int mnNumDataset;

    std::ofstream f_track_stats;

    std::ofstream f_track_times;
    double mTime_PosePred;
    double mTime_LocalMapTrack;
    double mTime_NewKF_Dec;

    std::vector<GeometricCamera *> mvpCameras;

    int initID, lastID;

    std::vector<Sophus::SE3f> mvTbc;
    std::vector<std::pair<int, int>> mvStereoPair;
    const std::vector<std::vector<std::vector<std::unordered_set<size_t>>>>
        *mpStereoPointIndexInGrid;
    float mPointStereoMinDot;
    float mLineStereoMinDot;
    MultiCameraTrackingTiming mLastMultiCameraTiming;

    void newParameterLoader(Settings *settings);

#ifdef REGISTER_LOOP
    bool Stop();

    bool mbStopped;
    bool mbStopRequested;
    bool mbNotStop;
    std::mutex mMutexStop;
#endif
};

} // namespace XPLSLAM
