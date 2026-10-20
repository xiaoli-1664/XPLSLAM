#include "frame_drawer.h"

#include <cmath>
#include <opencv2/imgproc.hpp>

using namespace std;

namespace XPLSLAM {

FrameDrawer::FrameDrawer(Atlas *pAtlas)
    : mpAtlas(pAtlas), mState(Tracking::NO_IMAGES_YET) {}

void FrameDrawer::Update(const Frame &frame, Tracking::eTrackingState state) {
    unique_lock<mutex> lock(mMutex);

    mvImages.resize(frame.mvImgs.size());
    for (size_t cam = 0; cam < frame.mvImgs.size(); ++cam) {
        mvImages[cam] = frame.mvImgs[cam].clone();
    }

    mvKeys = frame.mvKeysUn_cams;
    mvLines = frame.mvLineKeysUn_cams;
    mvbOutlierPoints = frame.mvbOutlierPoint_cams;
    mvbOutlierLines = frame.mvbOutlierLine_cams;
    mState = state;

    mvbMapPoints.assign(frame.mvpMapPoints_cams.size(), {});
    for (size_t cam = 0; cam < frame.mvpMapPoints_cams.size(); ++cam) {
        mvbMapPoints[cam].resize(frame.mvpMapPoints_cams[cam].size(), false);
        for (size_t i = 0; i < frame.mvpMapPoints_cams[cam].size(); ++i) {
            mvbMapPoints[cam][i] = frame.mvpMapPoints_cams[cam][i] != nullptr;
        }
    }

    mvbMapLines.assign(frame.mvpMapLines_cams.size(), {});
    for (size_t cam = 0; cam < frame.mvpMapLines_cams.size(); ++cam) {
        mvbMapLines[cam].resize(frame.mvpMapLines_cams[cam].size(), false);
        for (size_t i = 0; i < frame.mvpMapLines_cams[cam].size(); ++i) {
            mvbMapLines[cam][i] = frame.mvpMapLines_cams[cam][i] != nullptr;
        }
    }
}

cv::Mat FrameDrawer::DrawFrame(float imageScale) {
    vector<cv::Mat> images;
    vector<vector<cv::Point2f>> keys;
    vector<vector<Eigen::Vector4f>> lines;
    vector<vector<bool>> map_points;
    vector<vector<bool>> map_lines;
    vector<vector<bool>> outlier_points;
    vector<vector<bool>> outlier_lines;
    Tracking::eTrackingState state;

    {
        unique_lock<mutex> lock(mMutex);
        images = mvImages;
        keys = mvKeys;
        lines = mvLines;
        map_points = mvbMapPoints;
        map_lines = mvbMapLines;
        outlier_points = mvbOutlierPoints;
        outlier_lines = mvbOutlierLines;
        state = mState;
    }

    if (images.empty()) {
        return cv::Mat(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    }

    int tracked_points = 0;
    int tracked_lines = 0;
    vector<cv::Mat> rendered(images.size());
    for (size_t cam = 0; cam < images.size(); ++cam) {
        cv::Mat im;
        if (images[cam].empty()) {
            im = cv::Mat(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
        } else if (images[cam].channels() == 1) {
            cv::cvtColor(images[cam], im, cv::COLOR_GRAY2BGR);
        } else {
            im = images[cam].clone();
        }

        if (cam < lines.size()) {
            for (size_t i = 0; i < lines[cam].size(); ++i) {
                const Eigen::Vector4f &line = lines[cam][i];
                const bool matched = cam < map_lines.size() &&
                                     i < map_lines[cam].size() &&
                                     map_lines[cam][i];
                const bool outlier = cam < outlier_lines.size() &&
                                     i < outlier_lines[cam].size() &&
                                     outlier_lines[cam][i];
                if (!matched || outlier) {
                    continue;
                }
                ++tracked_lines;
                cv::line(im, cv::Point2f(line(0), line(1)),
                         cv::Point2f(line(2), line(3)),
                         cv::Scalar(255, 255, 0), 1,
                         cv::LINE_AA);
            }
        }

        if (cam < keys.size()) {
            for (size_t i = 0; i < keys[cam].size(); ++i) {
                const bool matched = cam < map_points.size() &&
                                     i < map_points[cam].size() &&
                                     map_points[cam][i];
                const bool outlier = cam < outlier_points.size() &&
                                     i < outlier_points[cam].size() &&
                                     outlier_points[cam][i];
                if (!matched || outlier) {
                    continue;
                }
                ++tracked_points;
                cv::circle(im, keys[cam][i], 2, cv::Scalar(0, 255, 0), -1,
                           cv::LINE_AA);
            }
        }

        cv::putText(im, "cam" + to_string(cam), cv::Point(10, 22),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255),
                    2, cv::LINE_AA);
        rendered[cam] = im;
    }

    const int n = static_cast<int>(rendered.size());
    const int cols = n <= 2 ? n : static_cast<int>(ceil(sqrt(n)));
    const int rows = static_cast<int>(ceil(static_cast<float>(n) / cols));
    const int cell_w = rendered[0].cols;
    const int cell_h = rendered[0].rows;
    cv::Mat mosaic(rows * cell_h, cols * cell_w, CV_8UC3,
                   cv::Scalar(20, 20, 20));

    for (int i = 0; i < n; ++i) {
        cv::Mat cell = mosaic(cv::Rect((i % cols) * cell_w,
                                       (i / cols) * cell_h, cell_w, cell_h));
        if (rendered[i].cols == cell_w && rendered[i].rows == cell_h) {
            rendered[i].copyTo(cell);
        } else {
            cv::resize(rendered[i], cell, cv::Size(cell_w, cell_h));
        }
    }

    DrawTextInfo(mosaic, state, tracked_points, tracked_lines);

    if (imageScale != 1.0f && imageScale > 0.0f) {
        cv::resize(mosaic, mosaic, cv::Size(), imageScale, imageScale,
                   cv::INTER_LINEAR);
    }
    return mosaic;
}

string FrameDrawer::StateToString(Tracking::eTrackingState state) {
    switch (state) {
    case Tracking::SYSTEM_NOT_READY:
        return "SYSTEM_NOT_READY";
    case Tracking::NO_IMAGES_YET:
        return "NO_IMAGES_YET";
    case Tracking::NOT_INITIALIZED:
        return "NOT_INITIALIZED";
    case Tracking::OK:
        return "OK";
    case Tracking::RECENTLY_LOST:
        return "RECENTLY_LOST";
    case Tracking::LOST:
        return "LOST";
    case Tracking::OK_KLT:
        return "OK_KLT";
    }
    return "UNKNOWN";
}

void FrameDrawer::DrawTextInfo(cv::Mat &image, Tracking::eTrackingState state,
                               int tracked_points,
                               int tracked_lines) {
    const int baseline = 0;
    const int text_h = 28;
    cv::Mat info(image.rows + text_h, image.cols, image.type());
    image.copyTo(info.rowRange(0, image.rows));
    info.rowRange(image.rows, image.rows + text_h).setTo(cv::Scalar(0, 0, 0));

    const string text = "XPLSLAM " + StateToString(state) +
                        " | tracked points: " + to_string(tracked_points) +
                        " | tracked lines: " + to_string(tracked_lines);
    cv::putText(info, text, cv::Point(10, image.rows + 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1,
                cv::LINE_AA, false);
    (void)baseline;
    image = info;
}

} // namespace XPLSLAM
