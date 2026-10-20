#include "camera_models/pinhole.h"
#include "frame.h"
#include "geometric_tools.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <unordered_set>
#include <vector>

namespace XPLSLAM {

struct FrameTestAccess {
    static void SetCameraExtrinsics(Frame &frame,
                                    const std::vector<Sophus::SE3f> &Tbc) {
        frame.mvTbc = Tbc;
        frame.mvTcb.clear();
        frame.mvRbc.clear();
        frame.mvtbc.clear();
        for (const auto &pose : Tbc) {
            frame.mvTcb.push_back(pose.inverse());
            frame.mvRbc.push_back(pose.rotationMatrix());
            frame.mvtbc.push_back(pose.translation());
        }
    }
};

} // namespace XPLSLAM

namespace {

using namespace XPLSLAM;

bool Check(bool condition, const char *message) {
    if (!condition) {
        std::cerr << message << '\n';
    }
    return condition;
}

struct StereoFixture {
    Pinhole camera{{500.0f, 500.0f, 320.0f, 240.0f}};
    Frame frame;
    std::vector<std::vector<std::vector<std::unordered_set<size_t>>>> candidates;

    StereoFixture() {
        frame.mnCam = 2;
        frame.Np = {1, 1};
        frame.Nl = {1, 1};
        frame.mvpCamera = {&camera, &camera};
        frame.mvStereoPair = {{0, 1}};
        FrameTestAccess::SetCameraExtrinsics(
            frame, {Sophus::SE3f(), Sophus::SE3f(
                Eigen::Matrix3f::Identity(), Eigen::Vector3f(0.25f, 0, 0))});
        frame.mPointStereoMinDot = 0.5f;
        frame.mLineStereoMinDot = 0.5f;
        frame.mvKeysUn_cams.resize(2, std::vector<cv::Point2f>(1));
        frame.mvLineKeysUn_cams.resize(2, std::vector<Eigen::Vector4f>(1));
        cv::Mat descriptor = cv::Mat::zeros(1, 64, CV_32F);
        descriptor.at<float>(0, 0) = 1.0f;
        frame.mvDescriptors_point = {descriptor, descriptor};
        frame.mvDescriptors_line = {descriptor, descriptor};
        frame.mvDepthStereoPoint.resize(1);
        frame.mvFirstToSecondPointMatch.resize(1);
        frame.mvStereo3DPoints.resize(1);
        frame.mvDepthStereoLine.resize(1);
        frame.mvFirstToSecondLineMatch.resize(1);
        frame.mvStereo3DLines.resize(1);

        Frame::mvnMinX = {0, 0};
        Frame::mvnMinY = {0, 0};
        Frame::mvfGridElementWidthInvPoint = {0.1f, 0.1f};
        Frame::mvfGridElementHeightInvPoint = {0.1f, 0.1f};
        frame.mvGrid_point.resize(2);
        for (auto &grid : frame.mvGrid_point) {
            grid.resize(FRAME_GRID_ROWS_POINT);
            for (auto &row : grid) {
                row.resize(FRAME_GRID_COLS_POINT);
            }
        }
        // Let the coarse candidate lookup see the sole right feature. The
        // production matcher must still validate its exact stereo geometry.
        frame.mvGrid_point[1][0][0] = {0};
        candidates.resize(1);
        candidates[0].resize(FRAME_GRID_ROWS_POINT);
        for (auto &row : candidates[0]) {
            row.resize(FRAME_GRID_COLS_POINT, std::unordered_set<size_t>{0});
        }
        frame.mpStereoPointIndexInGrid = &candidates;
    }

    void SetPoint(const Eigen::Vector3f &point) {
        for (int cam = 0; cam < 2; ++cam) {
            const Eigen::Vector3f point_c =
                frame.GetCameraExtrinsic(cam).inverse() * point;
            const Eigen::Vector2f pixel = camera.project(point_c);
            frame.mvKeysUn_cams[cam][0] = {pixel.x(), pixel.y()};
        }
    }

    void SetLine(float depth) {
        for (int cam = 0; cam < 2; ++cam) {
            const Eigen::Vector3f start =
                frame.GetCameraExtrinsic(cam).inverse() *
                Eigen::Vector3f(0.4f, -0.5f, depth);
            const Eigen::Vector3f end =
                frame.GetCameraExtrinsic(cam).inverse() *
                Eigen::Vector3f(0.4f, 0.5f, depth);
            frame.mvLineKeysUn_cams[cam][0].head<2>() = camera.project(start);
            frame.mvLineKeysUn_cams[cam][0].tail<2>() = camera.project(end);
        }
    }
};

} // namespace

int main() {
    using namespace XPLSLAM;
    bool ok = true;
    StereoFixture fixture;
    fixture.SetPoint({0, 0, 5});
    fixture.frame.ComputePointMulticameraMatches();
    ok &= Check(fixture.frame.mvFirstToSecondPointMatch[0][0] == 0 &&
                    std::abs(fixture.frame.mvDepthStereoPoint[0][0] - 5) < 1e-3f,
                "valid stereo point was rejected or has incorrect depth");
    fixture.frame.mvKeysUn_cams[1][0].y += 5.0f;
    fixture.frame.ComputePointMulticameraMatches();
    ok &= Check(fixture.frame.mvFirstToSecondPointMatch[0][0] == -1,
                "off-epipolar point match was accepted");
    fixture.SetPoint({0, 0, 5000});
    fixture.frame.ComputePointMulticameraMatches();
    ok &= Check(fixture.frame.mvFirstToSecondPointMatch[0][0] == -1,
                "near-parallel stereo rays were accepted");
    fixture.SetLine(5);
    fixture.frame.ComputeLineMulticameraMatches();
    ok &= Check(fixture.frame.mvFirstToSecondLineMatch[0][0] == 0,
                "valid stereo line was rejected");
    fixture.SetLine(-5);
    fixture.frame.ComputeLineMulticameraMatches();
    ok &= Check(fixture.frame.mvFirstToSecondLineMatch[0][0] == -1 &&
                    fixture.frame.mvDepthStereoLine[0][0] < 0,
                "behind-camera line was accepted using unsigned distance");

    int x = -1;
    int y = -1;
    constexpr float d_max = 800;
    const float theta_inv = FRAME_GRID_COLS_LINE / GeometricTools::kPi;
    const float d_inv = FRAME_GRID_ROWS_LINE / (2.0f * d_max);
    ok &= Check(GeometricTools::NormalizedLinePolarToGridCell(
                    0, 179.5f * GeometricTools::kPi / 180.0f, d_max,
                    theta_inv, d_inv, FRAME_GRID_COLS_LINE, FRAME_GRID_ROWS_LINE,
                    x, y) && x == FRAME_GRID_COLS_LINE - 1,
                "line near pi disappeared from the polar grid");
    ok &= Check(GeometricTools::NormalizedLinePolarToGridCell(
                    d_max, std::nextafter(GeometricTools::kPi, 0.0f), d_max,
                    theta_inv, d_inv, FRAME_GRID_COLS_LINE, FRAME_GRID_ROWS_LINE,
                    x, y) && x == FRAME_GRID_COLS_LINE - 1 &&
                    y == FRAME_GRID_ROWS_LINE - 1,
                "upper grid boundary was dropped");
    ok &= Check(!GeometricTools::NormalizedLinePolarToGridCell(
                    0, std::numeric_limits<float>::quiet_NaN(), d_max,
                    theta_inv, d_inv, FRAME_GRID_COLS_LINE, FRAME_GRID_ROWS_LINE,
                    x, y),
                "non-finite grid coordinate was accepted");
    return ok ? 0 : 1;
}
