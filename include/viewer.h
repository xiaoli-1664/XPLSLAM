#pragma once

#include "frame_drawer.h"
#include "map_drawer.h"
#include "system.h"

#include <mutex>

namespace XPLSLAM {

class Viewer {
  public:
    Viewer(System *pSystem, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer,
           Settings *settings, const std::string &strSettingPath);

    void Run();
    void RequestFinish();
    bool isFinished();

  private:
    bool CheckFinish();
    void SetFinish();
    void LoadViewerParameters(Settings *settings,
                              const std::string &strSettingPath);

    System *mpSystem;
    FrameDrawer *mpFrameDrawer;
    MapDrawer *mpMapDrawer;

    double mT;
    float mImageViewerScale;
    float mViewpointX;
    float mViewpointY;
    float mViewpointZ;
    float mViewpointF;

    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;
};

} // namespace XPLSLAM
