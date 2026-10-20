#include "graphite_slam_backend.h"

#include <algorithm>

namespace XPLSLAM::graphite_backend {
namespace {

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

std::array<double, 2> FlattenVector2(const Eigen::Vector2d &vector) {
    return {vector[0], vector[1]};
}

std::array<double, 4> FlattenVector4(const Eigen::Vector4d &vector) {
    return {vector[0], vector[1], vector[2], vector[3]};
}

std::array<double, 6> FlattenVector6(const Vector6d &vector) {
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

FlatPoseVertex FlattenPoseVertex(const PoseVertex &pose) {
    FlatPoseVertex flat;
    flat.Rwb = FlattenMatrix(pose.Rwb);
    flat.twb = FlattenVector3(pose.twb);
    return flat;
}

PoseVertex UnflattenPoseVertex(const FlatPoseVertex &flat) {
    PoseVertex pose;
    pose.Rwb = UnflattenMatrix(flat.Rwb);
    pose.twb = UnflattenVector3(flat.twb);
    return pose;
}

FlatPinholeCamera FlattenCamera(const PinholeCamera &camera) {
    FlatPinholeCamera flat;
    flat.fx = camera.fx;
    flat.fy = camera.fy;
    flat.cx = camera.cx;
    flat.cy = camera.cy;
    flat.toPL = FlattenMatrix(camera.toPL);
    return flat;
}

FlatPoseOnlyPointInput FlattenPointInput(const PoseOnlyPointInput &input) {
    FlatPoseOnlyPointInput flat;
    flat.camera = FlattenCamera(input.camera);
    flat.Rcb = FlattenMatrix(input.Rcb);
    flat.tcb = FlattenVector3(input.tcb);
    flat.point_w = FlattenVector3(input.point_w);
    flat.measurement = FlattenVector2(input.measurement);
    flat.information = input.information;
    return flat;
}

FlatPoseOnlyLineInput FlattenLineInput(const PoseOnlyLineInput &input) {
    FlatPoseOnlyLineInput flat;
    flat.camera = FlattenCamera(input.camera);
    flat.Rcb = FlattenMatrix(input.Rcb);
    flat.tcb = FlattenVector3(input.tcb);
    flat.line_w = FlattenVector6(input.line_w);
    flat.measurement = FlattenVector4(input.measurement);
    flat.has_endpoint_covariance =
        input.has_endpoint_covariance &&
                PackLineEndpointCovariance(input.endpoint_covariance,
                                           flat.endpoint_covariance_upper)
            ? 1
            : 0;
    flat.information = input.information;
    return flat;
}

} // namespace

bool OptimizePoseOnly(const PoseOnlyProblem &problem, PoseOnlyResult &result) {
    std::vector<FlatPoseOnlyPointInput> flat_points;
    std::vector<FlatPoseOnlyLineInput> flat_lines;
    flat_points.reserve(problem.points.size());
    flat_lines.reserve(problem.lines.size());

    for (const PoseOnlyPointInput &point : problem.points) {
        flat_points.push_back(FlattenPointInput(point));
    }
    for (const PoseOnlyLineInput &line : problem.lines) {
        flat_lines.push_back(FlattenLineInput(line));
    }

    result.optimized_pose = problem.initial_pose;
    result.point_outliers.assign(problem.points.size(), 0);
    result.line_outliers.assign(problem.lines.size(), 0);
    result.num_bad = 0;

    FlatPoseOnlyProblem flat_problem;
    flat_problem.initial_pose = FlattenPoseVertex(problem.initial_pose);
    flat_problem.points = flat_points.data();
    flat_problem.num_points = flat_points.size();
    flat_problem.lines = flat_lines.data();
    flat_problem.num_lines = flat_lines.size();
    flat_problem.chi2_threshold = problem.chi2_threshold;
    flat_problem.huber_delta = problem.huber_delta;

    FlatPoseOnlyResult flat_result;
    flat_result.optimized_pose = flat_problem.initial_pose;
    flat_result.point_outliers = result.point_outliers.data();
    flat_result.num_point_outliers = result.point_outliers.size();
    flat_result.line_outliers = result.line_outliers.data();
    flat_result.num_line_outliers = result.line_outliers.size();
    flat_result.num_bad = 0;

    if (!OptimizePoseOnlyFlat(flat_problem, flat_result)) {
        return false;
    }

    result.optimized_pose = UnflattenPoseVertex(flat_result.optimized_pose);
    result.num_bad = flat_result.num_bad;
    return true;
}

} // namespace XPLSLAM::graphite_backend
