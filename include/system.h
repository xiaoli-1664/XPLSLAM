#pragma once

#include <opencv2/core/core.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <thread>
#include <unordered_set>
#include <unistd.h>

#include "Vocabulary.h"
#include "atlas.h"
#include "keyframe_database.h"
#include "local_mapping.h"
#include "loop_closing.h"
#include "map_sparsification.h"
#include "settings.h"
#include "tracking.h"
#include "tracking_timing.h"

namespace XPLSLAM {

class Verbose {
  public:
    enum eLevel {
        VERBOSITY_QUIET = 0,
        VERBOSITY_NORMAL = 1,
        VERBOSITY_VERBOSE = 2,
        VERBOSITY_VERY_VERBOSE = 3,
        VERBOSITY_DEBUG = 4
    };

    static eLevel th;

  public:
    static void PrintMess(std::string str, eLevel lev) {
        if (lev <= th)
            std::cout << str << std::endl;
    }

    static void SetTh(eLevel _th) { th = _th; }
};

class Atlas;
class Tracking;
class LocalMapping;
class LoopClosing;
class Settings;
class FrameDrawer;
class MapDrawer;
class Viewer;

class System {
  public:
    // File type
    enum FileType {
        TEXT_FILE = 0,
        BINARY_FILE = 1,
    };

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // Initialize the SLAM system. It launches the Local Mapping, Loop Closing
    // and Viewer threads.
    System(const std::string &strVocFilePoint,
           const std::string &strVocFileLine,
           const std::string &strSettingsFile, const bool bUseViewer = true,
           const int initFr = 0,
           const std::string &strSequence = std::string());

    Sophus::SE3f TrackMultiCamera(const std::vector<cv::Mat> &vIms,
                                  const double &timestamp,
                                  std::string filename = "");

    ProcessedFrameTiming GetLastProcessedFrameTiming();

    // This stops local mapping thread (map building) and performs only camera
    // tracking.
    void ActivateLocalizationMode();
    // This resumes local mapping thread and performs SLAM again.
    void DeactivateLocalizationMode();

    // Returns true if there have been a big map change (loop closure, global
    // BA) since last call to this function
    bool MapChanged();

    // Reset the system (clear Atlas or the active map)
    void Reset();
    void ResetActiveMap();

    // All threads will be requested to finish.
    // It waits until all threads have finished.
    // This function must be called before saving the trajectory.
    void Shutdown();
    bool isShutDown();

    // Save camera trajectory in the TUM RGB-D dataset format.
    // Only for stereo and RGB-D. This method does not work for monocular.
    // Call first Shutdown()
    // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
    void SaveTrajectoryTUM(const std::string &filename);

    // Save keyframe poses in the TUM RGB-D dataset format.
    // This method works for all sensor input.
    // Call first Shutdown()
    // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
    void SaveKeyFrameTrajectoryTUM(const std::string &filename);

    void SaveTrajectoryEuRoC(const std::string &filename);
    void SaveKeyFrameTrajectoryEuRoC(const std::string &filename);

    void SaveTrajectoryEuRoC(const std::string &filename, Map *pMap);
    void SaveKeyFrameTrajectoryEuRoC(const std::string &filename, Map *pMap);

    // Save data used for initialization debug
    void SaveDebugData(const int &iniIdx);

    // Save camera trajectory in the KITTI dataset format.
    // Only for stereo and RGB-D. This method does not work for monocular.
    // Call first Shutdown()
    // See format details at:
    // http://www.cvlibs.net/datasets/kitti/eval_odometry.php
    void SaveTrajectoryKITTI(const std::string &filename);

    // TODO: Save/Load functions
    // SaveMap(const string &filename);
    // LoadMap(const string &filename);

    // Information from most recent processed frame
    // You can call this right after TrackMonocular (or stereo or RGBD)
    int GetTrackingState();
    std::vector<MapPointPtr> GetTrackedMapPoints();
    std::vector<MapLinePtr> GetTrackedMapLines();
    std::vector<std::vector<cv::Point2f>> GetTrackedKeyPointsUn();
    std::vector<std::vector<Eigen::Vector4f>> GetTrackedKeyLinesUn();

    // For debugging
    bool isLost();
    bool isFinished();

    void ChangeDataset();

    float GetImageScale();
    long unsigned int GetMapPointCount();
    MapSparsification::Statistics GetMapSparsificationStatistics();

    const std::vector<std::vector<std::vector<std::unordered_set<size_t>>>> &
    GetStereoPointIndexInGrid() const {
        return mvStereoPointIndexInGrid;
    }

#ifdef REGISTER_TIMES
    void InsertRectTime(double &time);
    void InsertResizeTime(double &time);
    void InsertTrackTime(double &time);
#endif

  private:
    void SaveAtlas(int type);
    bool LoadAtlas(int type);

    std::string CalculateCheckSum(std::string filename, int type);
    void PrecomputeStereoPointIndexInGrid();

    DBoW3::Vocabulary *mpVocabulary_point;
    DBoW3::Vocabulary *mpVocabulary_line;

    // KeyFrame database for place recognition (relocalization and loop
    // detection).
    KeyFrameDatabase *mpKeyFrameDatabase;

    // Map structure that stores the pointers to all KeyFrames and MapPoints.
    // Map* mpMap;
    Atlas *mpAtlas;

    // Tracker. It receives a frame and computes the associated camera pose.
    // It also decides when to insert a new keyframe, create some new MapPoints
    // and performs relocalization if tracking fails.
    Tracking *mpTracker;

    // Local Mapper. It manages the local map and performs local bundle
    // adjustment.
    LocalMapping *mpLocalMapper;

    // Loop Closer. It searches loops with every new keyframe. If there is a
    // loop it performs a pose graph optimization and full bundle adjustment (in
    // a new thread) afterwards.
    LoopClosing *mpLoopCloser;
    MapSparsification *mpMapSparsification;

    // System threads: Local Mapping, Loop Closing, Viewer.
    // The Tracking thread "lives" in the main execution thread that creates the
    // System object.
    std::thread *mptLocalMapping;
    std::thread *mptLoopClosing;
    std::thread *mptMapSparsification;
    std::thread *mptViewer;

    FrameDrawer *mpFrameDrawer;
    MapDrawer *mpMapDrawer;
    Viewer *mpViewer;

    // Reset flag
    std::mutex mMutexReset;
    bool mbReset;
    bool mbResetActiveMap;

    // Change mode flags
    std::mutex mMutexMode;
    bool mbActivateLocalizationMode;
    bool mbDeactivateLocalizationMode;

    // Shutdown flag
    bool mbShutDown;

    // Tracking state
    int mTrackingState;
    std::vector<std::vector<MapPointPtr>> mTrackedMapPoints;
    std::vector<std::vector<MapLinePtr>> mTrackedMapLines;
    std::vector<std::vector<cv::Point2f>> mTrackedKeyPointsUn;
    std::vector<std::vector<Eigen::Vector4f>> mTrackedKeyLinesUn;
    ProcessedFrameTiming mLastProcessedFrameTiming;
    std::mutex mMutexState;

    //
    std::string mStrLoadAtlasFromFile;
    std::string mStrSaveAtlasToFile;

    std::string mStrVocabularyPointFilePath;
    std::string mStrVocabularyLineFilePath;

    Settings *settings_;
    std::vector<std::vector<std::vector<std::unordered_set<size_t>>>>
        mvStereoPointIndexInGrid;
};

} // namespace XPLSLAM
