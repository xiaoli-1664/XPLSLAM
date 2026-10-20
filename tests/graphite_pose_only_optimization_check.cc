#include "graphite_slam_backend.h"

#include <cuda_runtime.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

namespace {

namespace gb = XPLSLAM::graphite_backend;

gb::Vector6d MakePluckerLine(const Eigen::Vector3d &p1,
                             const Eigen::Vector3d &p2) {
    gb::Vector6d line;
    const Eigen::Vector3d direction = p2 - p1;
    line.head<3>() = p1.cross(direction);
    line.tail<3>() = direction;
    return line;
}

double PoseChi2(const gb::PoseVertex &pose, const gb::PoseOnlyProblem &problem) {
    double chi2 = 0.0;
    for (const gb::PoseOnlyPointInput &point : problem.points) {
        const gb::Pose camera_pose = gb::MakeCameraPose(pose, point.Rcb,
                                                        point.tcb);
        chi2 += point.information *
                gb::PoseOnlyPointResidual(camera_pose, point.camera,
                                          point.point_w, point.measurement)
                    .squaredNorm();
    }
    for (const gb::PoseOnlyLineInput &line : problem.lines) {
        const gb::Pose camera_pose =
            gb::MakeCameraPose(pose, line.Rcb, line.tcb);
        const Eigen::Vector3d line_2d =
            gb::ProjectWorldLine(camera_pose, line.camera, line.line_w);
        const Eigen::Vector2d residual = gb::PoseOnlyLineResidual(
            camera_pose, line.camera, line.line_w, line.measurement);
        XPLSLAM::PackedLineEndpointCovariance packed{};
        const bool has_covariance =
            line.has_endpoint_covariance &&
            XPLSLAM::PackLineEndpointCovariance(line.endpoint_covariance,
                                                packed);
        const Eigen::Matrix2d information =
            XPLSLAM::LineResidualInformation(line_2d, packed, has_covariance,
                                             line.information);
        chi2 += residual.dot(information * residual);
    }
    return chi2;
}

double RotationAngle(const Eigen::Matrix3d &R) {
    const double cos_angle =
        std::max(-1.0, std::min(1.0, (R.trace() - 1.0) * 0.5));
    return std::acos(cos_angle);
}

void PrintPoseError(const gb::PoseVertex &optimized,
                    const gb::PoseVertex &truth) {
    const double rot_error =
        RotationAngle(optimized.Rwb.transpose() * truth.Rwb);
    const double trans_error = (optimized.twb - truth.twb).norm();
    std::cerr << std::setprecision(12) << "rotation_error=" << rot_error
              << " translation_error=" << trans_error << '\n';
}

} // namespace

int main() {
    int device_count = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (device_status != cudaSuccess || device_count <= 0) {
        std::cerr << "Skipping: CUDA device is not visible to this process\n";
        return 77;
    }

    gb::PoseOnlyProblem problem;
    const gb::PinholeCamera camera =
        gb::MakePinholeCamera(520.0, 515.0, 320.0, 240.0);
    const Eigen::Matrix3d Rcb =
        gb::ExpSO3(Eigen::Vector3d(-0.015, 0.01, 0.02));
    const Eigen::Vector3d tcb(0.04, -0.015, 0.02);

    gb::PoseVertex truth;
    truth.Rwb = gb::ExpSO3(Eigen::Vector3d(0.035, -0.025, 0.018));
    truth.twb = Eigen::Vector3d(0.18, -0.09, 0.27);

    problem.initial_pose.Rwb =
        truth.Rwb * gb::ExpSO3(Eigen::Vector3d(0.018, -0.014, 0.012));
    problem.initial_pose.twb = truth.twb + Eigen::Vector3d(0.08, -0.05, 0.06);

    const gb::Pose truth_pose = gb::MakeCameraPose(truth, Rcb, tcb);
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 6; ++col) {
            gb::PoseOnlyPointInput input;
            input.camera = camera;
            input.Rcb = Rcb;
            input.tcb = tcb;
            input.point_w =
                Eigen::Vector3d(-1.0 + 0.4 * col, -0.45 + 0.3 * row,
                                4.0 + 0.2 * ((row + col) % 3));
            input.measurement =
                gb::ProjectWorldPoint(truth_pose, camera, input.point_w);
            input.information = 1.0;
            problem.points.push_back(input);
        }
    }

    for (int i = 0; i < 8; ++i) {
        const Eigen::Vector3d p1(-0.8 + 0.22 * i, -0.35 + 0.08 * i,
                                 3.6 + 0.12 * (i % 3));
        const Eigen::Vector3d p2(-0.45 + 0.18 * i, 0.28 - 0.05 * i,
                                 4.2 + 0.1 * ((i + 1) % 3));

        gb::PoseOnlyLineInput input;
        input.camera = camera;
        input.Rcb = Rcb;
        input.tcb = tcb;
        input.line_w = MakePluckerLine(p1, p2);
        input.measurement.head<2>() =
            gb::ProjectWorldPoint(truth_pose, camera, p1);
        input.measurement.tail<2>() =
            gb::ProjectWorldPoint(truth_pose, camera, p2);
        input.endpoint_covariance << 1.0 + 0.1 * i, 0.15, 0.08, 0.03, 0.15,
            4.0 + 0.2 * i, 0.04, 0.12, 0.08, 0.04, 2.0 + 0.1 * i, 0.2,
            0.03, 0.12, 0.2, 6.0 + 0.2 * i;
        input.has_endpoint_covariance = true;
        input.information = 0.25;
        problem.lines.push_back(input);
    }

    const double initial_chi2 = PoseChi2(problem.initial_pose, problem);

    gb::PoseOnlyResult result;
    if (!gb::OptimizePoseOnly(problem, result)) {
        std::cerr << "OptimizePoseOnly returned false\n";
        return 1;
    }

    const double final_chi2 = PoseChi2(result.optimized_pose, problem);
    const int outlier_count = result.num_bad;
    const double rotation_error =
        RotationAngle(result.optimized_pose.Rwb.transpose() * truth.Rwb);
    const double translation_error =
        (result.optimized_pose.twb - truth.twb).norm();

    const bool chi2_reduced = final_chi2 < initial_chi2 * 0.05;
    const bool pose_close = rotation_error < 5e-3 && translation_error < 1e-2;
    const bool no_outliers = outlier_count == 0;

    if (!chi2_reduced || !pose_close || !no_outliers) {
        std::cerr << std::setprecision(12)
                  << "Pose-only optimization failed\ninitial_chi2="
                  << initial_chi2 << " final_chi2=" << final_chi2
                  << " outliers=" << outlier_count << '\n';
        PrintPoseError(result.optimized_pose, truth);
        return 1;
    }

    // Zero reprojection error does not make geometry behind the camera valid.
    gb::PoseOnlyProblem behind_problem = problem;
    const gb::Pose body_camera_pose = gb::MakeCameraPose(
        truth, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    const Eigen::Vector3d behind0 =
        truth.Rwb * Eigen::Vector3d(0.3, -0.2, -4.0) + truth.twb;
    const Eigen::Vector3d behind1 =
        truth.Rwb * Eigen::Vector3d(0.3, 0.5, -4.0) + truth.twb;
    gb::PoseOnlyPointInput behind_point;
    behind_point.camera = camera;
    behind_point.point_w = behind0;
    behind_point.measurement =
        gb::ProjectWorldPoint(body_camera_pose, camera, behind0);
    behind_problem.points.push_back(behind_point);
    gb::PoseOnlyLineInput behind_line;
    behind_line.camera = camera;
    behind_line.line_w = MakePluckerLine(behind0, behind1);
    behind_line.measurement.head<2>() =
        gb::ProjectWorldPoint(body_camera_pose, camera, behind0);
    behind_line.measurement.tail<2>() =
        gb::ProjectWorldPoint(body_camera_pose, camera, behind1);
    behind_problem.lines.push_back(behind_line);

    gb::PoseOnlyResult behind_result;
    if (!gb::OptimizePoseOnly(behind_problem, behind_result) ||
        behind_result.point_outliers.back() == 0 ||
        behind_result.line_outliers.back() == 0 || behind_result.num_bad != 2) {
        std::cerr << "Behind-camera point/line were not rejected\n";
        return 1;
    }

    std::cout << std::setprecision(12)
              << "graphite_pose_only_optimization_check passed, initial_chi2="
              << initial_chi2 << ", final_chi2=" << final_chi2
              << ", rotation_error=" << rotation_error
              << ", translation_error=" << translation_error << '\n';
    return 0;
}
