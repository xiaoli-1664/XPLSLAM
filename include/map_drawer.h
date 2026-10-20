#pragma once

#include "atlas.h"
#include "settings.h"

#include <mutex>
#include <pangolin/gl/opengl_render_state.h>
#include <sophus/se3.hpp>

namespace XPLSLAM {

class MapDrawer {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    MapDrawer(Atlas *pAtlas, Settings *settings,
              const std::string &strSettingPath);

    void DrawMapPoints();
    void DrawMapLines();
    void DrawKeyFrames(bool draw_keyframes, bool draw_graph);
    void DrawTrajectory();
    void DrawCurrentCamera(pangolin::OpenGlMatrix &Twb);

    void SetCurrentCameraPose(const Sophus::SE3f &Tbw,
                              const KeyFramePtr &reference_keyframe);
    void SetCurrentCameraExtrinsics(const std::vector<Sophus::SE3f> &vTbc);
    void ClearTrajectory();
    void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M,
                                      pangolin::OpenGlMatrix &MOw);

  private:
    void LoadViewerParameters(const std::string &strSettingPath);
    void DrawFrustum(const float size, const float line_width,
                     const float red, const float green, const float blue);
    void DrawCameraRig(const Sophus::SE3f &Tbw,
                       const std::vector<Sophus::SE3f> &vTbc,
                       const float size, const float line_width,
                       const float red, const float green, const float blue);

    Atlas *mpAtlas;

    float mKeyFrameSize;
    float mKeyFrameLineWidth;
    float mGraphLineWidth;
    float mPointSize;
    float mLineWidth;
    float mCameraSize;
    float mCameraLineWidth;
    float mTrajectoryLineWidth;
    float mLineSegmentHalfLength;

    struct CameraTrajectoryEntry {
        Sophus::SE3f Tbr;
        KeyFramePtr reference_keyframe;
        Eigen::Vector3f recorded_position;
    };

    Sophus::SE3f mCameraPose;
    std::vector<Sophus::SE3f> mvTbc;
    std::vector<CameraTrajectoryEntry> mvCameraTrajectory;
    std::mutex mMutexCamera;
};

} // namespace XPLSLAM
