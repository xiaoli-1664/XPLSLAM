#include "graphite_slam_backend.h"

#include "frame.h"
#include "g2o_types.h"
#include "map_line.h"
#include "map_point.h"
#include "pinhole.h"

#include <Eigen/Core>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <utility>
#include <vector>

namespace XPLSLAM::graphite_backend {
namespace {

struct ObservationRef {
    size_t cam_id = 0;
    size_t index = 0;
};

std::array<double, 9> FlattenMatrix(const Eigen::Matrix3d &matrix) {
    std::array<double, 9> values{};
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            values[static_cast<size_t>(3 * row + col)] = matrix(row, col);
        }
    }
    return values;
}

std::array<double, 3> FlattenVector3(const Eigen::Vector3d &vector) {
    return {vector[0], vector[1], vector[2]};
}

std::array<double, 2> FlattenPoint2(const cv::Point2f &point) {
    return {point.x, point.y};
}

std::array<double, 4> FlattenVector4f(const Eigen::Vector4f &vector) {
    return {vector[0], vector[1], vector[2], vector[3]};
}

std::array<double, 6> FlattenVector6d(const Vector6d &vector) {
    return {vector[0], vector[1], vector[2], vector[3], vector[4], vector[5]};
}

Eigen::Matrix3d UnflattenMatrix(const std::array<double, 9> &values) {
    Eigen::Matrix3d matrix;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            matrix(row, col) = values[static_cast<size_t>(3 * row + col)];
        }
    }
    return matrix;
}

Eigen::Vector3d UnflattenVector3(const std::array<double, 3> &values) {
    return Eigen::Vector3d(values[0], values[1], values[2]);
}

FlatPoseVertex FlattenPose(const BasePose &pose) {
    FlatPoseVertex flat;
    flat.Rwb = FlattenMatrix(pose.Rwb);
    flat.twb = FlattenVector3(pose.twb);
    return flat;
}

bool FillCamera(const BasePose &pose, size_t cam_id, FlatPinholeCamera &camera,
                std::array<double, 9> &Rcb, std::array<double, 3> &tcb) {
    const auto *pinhole = dynamic_cast<const Pinhole *>(pose.vpCameras[cam_id]);
    if (!pinhole) {
        return false;
    }

    const PinholeCamera pinhole_camera =
        MakePinholeCamera(pinhole->getParameter(0), pinhole->getParameter(1),
                          pinhole->getParameter(2), pinhole->getParameter(3));
    camera.fx = pinhole_camera.fx;
    camera.fy = pinhole_camera.fy;
    camera.cx = pinhole_camera.cx;
    camera.cy = pinhole_camera.cy;
    camera.toPL = FlattenMatrix(pinhole_camera.toPL);
    Rcb = FlattenMatrix(pose.Rcb[cam_id]);
    tcb = FlattenVector3(pose.tcb[cam_id]);
    return true;
}

void FillLineEndpointCovariance(const Frame &frame, size_t cam_id, int idx,
                                FlatPoseOnlyLineInput &input) {
    if (idx < 0 || cam_id >= frame.mvLineEndpointCovariances_cams.size() ||
        static_cast<size_t>(idx) >=
            frame.mvLineEndpointCovariances_cams[cam_id].size()) {
        return;
    }
    input.has_endpoint_covariance =
        PackLineEndpointCovariance(
            frame.mvLineEndpointCovariances_cams[cam_id][idx],
            input.endpoint_covariance_upper)
            ? 1
            : 0;
}

} // namespace

int PoseOptimization(Frame *frame) {
    if (!frame) {
        return 0;
    }

    const std::vector<int> points_per_camera = frame->Np;
    const std::vector<int> lines_per_camera = frame->Nl;
    const int total_points = frame->mnTotalPoints;
    const int total_lines = frame->mnTotalLines;
    const int num_cameras = frame->mnCam;

    std::vector<FlatPoseOnlyPointInput> points;
    std::vector<FlatPoseOnlyLineInput> lines;
    points.reserve(static_cast<size_t>(total_points));
    lines.reserve(static_cast<size_t>(total_lines));

    std::vector<ObservationRef> point_refs;
    std::vector<ObservationRef> line_refs;
    point_refs.reserve(static_cast<size_t>(total_points));
    line_refs.reserve(static_cast<size_t>(total_lines));

    const BasePose initial_pose(frame);
    const FlatPoseVertex initial_flat_pose = FlattenPose(initial_pose);

    const double sigma_point = 1.0;
    const double point_information = 1.0 / (sigma_point * sigma_point);
    const double line_information = kLineInformationFallback;

    {
        std::unique_lock<std::mutex> lock_point(MapPoint::mGlobalMutex);
        std::unique_lock<std::mutex> lock_line(MapLine::mGlobalMutex);

        for (size_t cam_id = 0; cam_id < static_cast<size_t>(num_cameras);
             ++cam_id) {
            FlatPinholeCamera camera;
            std::array<double, 9> Rcb{};
            std::array<double, 3> tcb{};
            if (!FillCamera(initial_pose, cam_id, camera, Rcb, tcb)) {
                return 0;
            }

            for (int i = 0; i < points_per_camera[cam_id]; ++i) {
                MapPointPtr map_point = frame->mvpMapPoints_cams[cam_id][i];
                if (!map_point) {
                    continue;
                }

                const cv::Point2f point_obs = frame->mvKeysUn_cams[cam_id][i];

                FlatPoseOnlyPointInput input;
                input.camera = camera;
                input.Rcb = Rcb;
                input.tcb = tcb;
                input.point_w =
                    FlattenVector3(map_point->GetWorldPos().cast<double>());
                input.measurement = FlattenPoint2(point_obs);
                input.information = point_information;
                points.push_back(input);
                point_refs.push_back({cam_id, static_cast<size_t>(i)});
            }

            for (int i = 0; i < lines_per_camera[cam_id]; ++i) {
                MapLinePtr map_line = frame->mvpMapLines_cams[cam_id][i];
                if (!map_line) {
                    continue;
                }

                const Eigen::Vector4f line_obs =
                    frame->mvLineKeysUn_cams[cam_id][i];

                FlatPoseOnlyLineInput input;
                input.camera = camera;
                input.Rcb = Rcb;
                input.tcb = tcb;
                input.line_w =
                    FlattenVector6d(map_line->GetWorldPos().cast<double>());
                input.measurement = FlattenVector4f(line_obs);
                FillLineEndpointCovariance(*frame, cam_id, i, input);
                input.information = line_information;
                lines.push_back(input);
                line_refs.push_back({cam_id, static_cast<size_t>(i)});
            }
        }
    }

    const int initial_correspondences =
        static_cast<int>(points.size() + lines.size());
    const auto reject_pose = [&]() {
        for (const ObservationRef &ref : point_refs) {
            frame->mvbOutlierPoint_cams[ref.cam_id][ref.index] = true;
        }
        for (const ObservationRef &ref : line_refs) {
            frame->mvbOutlierLine_cams[ref.cam_id][ref.index] = true;
        }
        return 0;
    };
    if (initial_correspondences < 3) {
        return reject_pose();
    }

    FlatPoseOnlyProblem problem;
    problem.initial_pose = initial_flat_pose;
    problem.points = points.data();
    problem.num_points = points.size();
    problem.lines = lines.data();
    problem.num_lines = lines.size();

    std::vector<uint8_t> point_outliers(points.size(), 0);
    std::vector<uint8_t> line_outliers(lines.size(), 0);
    FlatPoseOnlyResult result;
    result.optimized_pose = initial_flat_pose;
    result.point_outliers = point_outliers.data();
    result.num_point_outliers = point_outliers.size();
    result.line_outliers = line_outliers.data();
    result.num_line_outliers = line_outliers.size();
    const bool timing_enabled =
        std::getenv("XPLSLAM_GRAPHITE_POSE_TIMING") != nullptr;
    const auto optimize_start = std::chrono::steady_clock::now();
    if (!OptimizePoseOnlyFlat(problem, result)) {
        return reject_pose();
    }
    const auto optimize_end = std::chrono::steady_clock::now();
    if (timing_enabled) {
        const double optimize_ms =
            std::chrono::duration<double, std::milli>(optimize_end -
                                                      optimize_start)
                .count();
        std::cerr << "graphite_pose_only_ms=" << optimize_ms
                  << " correspondences=" << initial_correspondences
                  << " outliers=" << result.num_bad << '\n';
    }

    for (size_t i = 0; i < point_refs.size(); ++i) {
        const ObservationRef &ref = point_refs[i];
        frame->mvbOutlierPoint_cams[ref.cam_id][ref.index] =
            point_outliers[i] != 0;
    }

    for (size_t i = 0; i < line_refs.size(); ++i) {
        const ObservationRef &ref = line_refs[i];
        frame->mvbOutlierLine_cams[ref.cam_id][ref.index] =
            line_outliers[i] != 0;
    }

    const Eigen::Matrix3f rotation =
        UnflattenMatrix(result.optimized_pose.Rwb).cast<float>();
    const Eigen::Vector3f translation =
        UnflattenVector3(result.optimized_pose.twb).cast<float>();
    if (!rotation.allFinite() || !translation.allFinite()) {
        return reject_pose();
    }
    const Eigen::Matrix3f R = NormalizeRotation<float>(rotation);
    const Sophus::SE3f Tbw_after = Sophus::SE3f(R, translation).inverse();
    frame->SetPose(Tbw_after);

    return initial_correspondences - result.num_bad;
}

} // namespace XPLSLAM::graphite_backend
