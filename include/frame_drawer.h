#pragma once

#include "atlas.h"
#include "frame.h"
#include "tracking.h"

#include <mutex>
#include <opencv2/core.hpp>

namespace XPLSLAM {

class FrameDrawer {
  public:
    explicit FrameDrawer(Atlas *pAtlas);

    void Update(const Frame &frame, Tracking::eTrackingState state);
    cv::Mat DrawFrame(float imageScale = 1.0f);

  private:
    static std::string StateToString(Tracking::eTrackingState state);
    void DrawTextInfo(cv::Mat &image, Tracking::eTrackingState state,
                      int tracked_points, int tracked_lines);

    Atlas *mpAtlas;

    std::mutex mMutex;
    std::vector<cv::Mat> mvImages;
    std::vector<std::vector<cv::Point2f>> mvKeys;
    std::vector<std::vector<Eigen::Vector4f>> mvLines;
    std::vector<std::vector<bool>> mvbMapPoints;
    std::vector<std::vector<bool>> mvbMapLines;
    std::vector<std::vector<bool>> mvbOutlierPoints;
    std::vector<std::vector<bool>> mvbOutlierLines;
    Tracking::eTrackingState mState;
};

} // namespace XPLSLAM
