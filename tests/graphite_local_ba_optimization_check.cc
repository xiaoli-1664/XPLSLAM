#include "graphite_local_ba_jacobians.cuh"
#include "graphite_slam_backend.h"
#include "graphite_slam_types.h"

#include <cuda_runtime.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace gb = XPLSLAM::graphite_backend;

using Vector6 = Eigen::Matrix<double, 6, 1>;

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

std::array<double, 6> FlattenVector6(const Vector6 &vector) {
    return {vector[0], vector[1], vector[2],
            vector[3], vector[4], vector[5]};
}

Eigen::Matrix3d Rotation(const gb::FlatPoseVertex &pose) {
    Eigen::Matrix3d R;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            R(row, col) = pose.Rwb[static_cast<size_t>(3 * row + col)];
        }
    }
    return R;
}

Eigen::Vector3d Translation(const gb::FlatPoseVertex &pose) {
    return Eigen::Vector3d(pose.twb[0], pose.twb[1], pose.twb[2]);
}

gb::FlatPoseVertex MakePose(const Eigen::Vector3d &r,
                            const Eigen::Vector3d &t) {
    gb::FlatPoseVertex pose;
    pose.Rwb = FlattenMatrix(gb::ExpSO3(r));
    pose.twb = FlattenVector3(t);
    return pose;
}

Eigen::Vector3d Vector3FromArray(const std::array<double, 3> &values) {
    return Eigen::Vector3d(values[0], values[1], values[2]);
}

gb::Pose MakePoseVertex(const gb::FlatPoseVertex &flat,
                        const std::array<double, 3> &rcb,
                        const std::array<double, 3> &tcb) {
    gb::Pose pose;
    pose.Rwb = Rotation(flat);
    pose.twb = Translation(flat);
    pose.Rcb = gb::ExpSO3(Vector3FromArray(rcb));
    pose.tcb = Vector3FromArray(tcb);
    return pose;
}

gb::Pose MakePoseVertex(const gb::FlatPoseVertex &flat) {
    static const std::array<double, 3> zero = {0.0, 0.0, 0.0};
    return MakePoseVertex(flat, zero, zero);
}

gb::Pose MakePoseFromParameters(const Vector6 &pose_params,
                                const Eigen::Vector3d &rcb,
                                const Eigen::Vector3d &tcb) {
    gb::Pose pose;
    pose.Rwb = gb::ExpSO3(Eigen::Vector3d(pose_params.head<3>()));
    pose.twb = pose_params.tail<3>();
    pose.Rcb = gb::ExpSO3(rcb);
    pose.tcb = tcb;
    return pose;
}

Vector6 MakePluckerLine(const Eigen::Vector3d &p1,
                        const Eigen::Vector3d &p2) {
    Vector6 line;
    const Eigen::Vector3d direction = p2 - p1;
    line.head<3>() = p1.cross(direction);
    line.tail<3>() = direction;
    return line;
}

Vector6 LineFromFlat(const gb::FlatLocalBALineVertex &line) {
    Vector6 line_w;
    for (int i = 0; i < 6; ++i) {
        line_w[i] = line.line_w[static_cast<size_t>(i)];
    }
    return line_w;
}

Eigen::Vector3d PointFromFlat(const gb::FlatLocalBAPointVertex &point) {
    return Eigen::Vector3d(point.position[0], point.position[1],
                           point.position[2]);
}

double RotationAngle(const Eigen::Matrix3d &R) {
    const double cos_angle =
        std::max(-1.0, std::min(1.0, (R.trace() - 1.0) * 0.5));
    return std::acos(cos_angle);
}

Eigen::Vector2d PointResidualReference(const Vector6 &pose_params,
                                       const Eigen::Vector3d &point_w,
                                       const Eigen::Vector3d &rcb,
                                       const Eigen::Vector3d &tcb,
                                       const gb::PinholeCamera &camera,
                                       const Eigen::Vector2d &measurement) {
    return measurement -
           gb::ProjectWorldPoint(
               MakePoseFromParameters(pose_params, rcb, tcb), camera, point_w);
}

Eigen::Vector2d LineResidualReference(const Vector6 &pose_params,
                                      const Vector6 &line_w,
                                      const Eigen::Vector3d &rcb,
                                      const Eigen::Vector3d &tcb,
                                      const gb::PinholeCamera &camera,
                                      const Eigen::Vector4d &measurement) {
    return gb::PoseOnlyLineResidual(
        MakePoseFromParameters(pose_params, rcb, tcb), camera, line_w,
        measurement);
}

struct LineState {
    Eigen::Matrix3d U = Eigen::Matrix3d::Identity();
    Eigen::Matrix2d W = Eigen::Matrix2d::Identity();
    Vector6 line_w = Vector6::Zero();
};

LineState MakeLineState(const Vector6 &line_w) {
    LineState state;
    const Eigen::Vector3d moment = line_w.head<3>();
    const Eigen::Vector3d direction = line_w.tail<3>();
    const double moment_norm = moment.norm();
    const double direction_norm = direction.norm();
    const double md_norm = std::hypot(moment_norm, direction_norm);
    state.U.col(0) = moment / moment_norm;
    state.U.col(1) = direction / direction_norm;
    state.U.col(2) = moment.cross(direction).normalized();
    state.W << moment_norm, -direction_norm, direction_norm, moment_norm;
    state.W /= md_norm;
    state.line_w.head<3>() = state.U.col(0) * state.W(0, 0);
    state.line_w.tail<3>() = state.U.col(1) * state.W(1, 0);
    return state;
}

Eigen::Matrix2d ExpSO2(double theta) {
    Eigen::Matrix2d R;
    R << std::cos(theta), -std::sin(theta), std::sin(theta), std::cos(theta);
    return R;
}

LineState ApplyLineUpdate(LineState state, const Eigen::Vector4d &delta) {
    state.U = state.U * gb::ExpSO3(Eigen::Vector3d(delta.head<3>()));
    state.W = state.W * ExpSO2(delta[3]);
    state.line_w.head<3>() = state.U.col(0) * state.W(0, 0);
    state.line_w.tail<3>() = state.U.col(1) * state.W(1, 0);
    return state;
}

Vector6 RetractPose(const Vector6 &pose, const Vector6 &delta) {
    const Eigen::Matrix3d R = gb::ExpSO3(Eigen::Vector3d(pose.head<3>())) *
        gb::ExpSO3(Eigen::Vector3d(delta.head<3>()));
    const Eigen::AngleAxisd aa(R);
    Vector6 result;
    result.head<3>() = aa.angle() * aa.axis();
    result.tail<3>() = pose.tail<3>() + delta.tail<3>();
    return result;
}

template <int R, int C, typename ResidualFunction>
bool CheckJacobian(const std::string &name,
                   const Eigen::Matrix<double, R, C> &analytic,
                   const Eigen::Matrix<double, C, 1> &x,
                   ResidualFunction residual_function, double abs_tolerance,
                   double rel_tolerance) {
    constexpr double eps = 1e-6;
    Eigen::Matrix<double, R, C> numerical;
    for (int col = 0; col < C; ++col) {
        Eigen::Matrix<double, C, 1> x_plus = x;
        Eigen::Matrix<double, C, 1> x_minus = x;
        x_plus[col] += eps;
        x_minus[col] -= eps;
        numerical.col(col) =
            (residual_function(x_plus) - residual_function(x_minus)) /
            (2.0 * eps);
    }

    const double max_abs = (analytic - numerical).cwiseAbs().maxCoeff();
    const double scale = std::max(
        1.0, std::max(analytic.cwiseAbs().maxCoeff(),
                      numerical.cwiseAbs().maxCoeff()));
    const double rel = max_abs / scale;
    if (max_abs > abs_tolerance && rel > rel_tolerance) {
        std::cerr << std::setprecision(12) << name
                  << " Jacobian check failed, max_abs=" << max_abs
                  << ", rel=" << rel << "\nanalytic:\n"
                  << analytic << "\nnumerical:\n"
                  << numerical << '\n';
        return false;
    }
    return true;
}

bool CheckLocalBAJacobians() {
    const gb::PinholeCamera camera =
        gb::MakePinholeCamera(517.0, 509.0, 323.0, 241.0);
    const Eigen::Vector3d rcb(0.035, -0.018, 0.026);
    const Eigen::Vector3d tcb(0.075, -0.022, 0.031);

    const Vector6 point_pose =
        (Vector6() << 0.12, -0.055, 0.08, 0.28, -0.11, 0.18).finished();
    const Eigen::Vector3d point_w(0.72, -0.38, 4.35);
    const Eigen::Vector2d point_measurement =
        gb::ProjectWorldPoint(
            MakePoseFromParameters(
                (Vector6() << 0.09, -0.04, 0.065, 0.23, -0.08, 0.15)
                    .finished(),
                rcb, tcb),
            camera, Eigen::Vector3d(0.70, -0.35, 4.28)) +
        Eigen::Vector2d(0.4, -0.3);

    Eigen::Vector2d point_residual;
    Eigen::Matrix<double, 2, 6> J_point_pose;
    Eigen::Matrix<double, 2, 3> J_point;
    gb::generated::local_ba_point_residual(
        gb::ExpSO3(Eigen::Vector3d(point_pose.head<3>())),
        Eigen::Vector3d(point_pose.tail<3>()), point_w, rcb, tcb, camera.fx, camera.fy, camera.cx,
        camera.cy, point_measurement, point_residual);
    gb::generated::local_ba_point_jacobian_pose(
        gb::ExpSO3(Eigen::Vector3d(point_pose.head<3>())),
        Eigen::Vector3d(point_pose.tail<3>()), point_w, rcb, tcb, camera.fx, camera.fy, camera.cx,
        camera.cy, point_measurement, J_point_pose);
    gb::generated::local_ba_point_jacobian_point(
        gb::ExpSO3(Eigen::Vector3d(point_pose.head<3>())),
        Eigen::Vector3d(point_pose.tail<3>()), point_w, rcb, tcb, camera.fx, camera.fy, camera.cx,
        camera.cy, point_measurement, J_point);

    const Eigen::Vector2d point_reference = PointResidualReference(
        point_pose, point_w, rcb, tcb, camera, point_measurement);
    if ((point_residual - point_reference).cwiseAbs().maxCoeff() > 1e-10) {
        std::cerr << "Local BA point residual does not match reference\n";
        return false;
    }
    bool ok = CheckJacobian<2, 6>(
        "local_ba_point pose", J_point_pose, Vector6::Zero(),
        [&](const Vector6 &pose) {
            return PointResidualReference(RetractPose(point_pose, pose), point_w, rcb, tcb, camera,
                                          point_measurement);
        },
        2e-5, 2e-7);
    ok = CheckJacobian<2, 3>(
             "local_ba_point point", J_point, point_w,
             [&](const Eigen::Vector3d &point) {
                 return PointResidualReference(point_pose, point, rcb, tcb,
                                               camera, point_measurement);
             },
             2e-5, 2e-7) &&
         ok;

    const Vector6 line_pose =
        (Vector6() << -0.08, 0.045, -0.06, -0.18, 0.07, 0.23).finished();
    const Eigen::Vector3d line_p1(-0.55, -0.28, 3.7);
    const Eigen::Vector3d line_p2(0.18, 0.32, 4.45);
    const LineState line_state =
        MakeLineState(MakePluckerLine(line_p1, line_p2));
    const gb::Pose line_truth_pose = MakePoseFromParameters(
        (Vector6() << -0.065, 0.035, -0.052, -0.15, 0.06, 0.20).finished(),
        rcb, tcb);
    const Eigen::Vector2d line_start =
        gb::ProjectWorldPoint(line_truth_pose, camera, line_p1);
    const Eigen::Vector2d line_end =
        gb::ProjectWorldPoint(line_truth_pose, camera, line_p2);
    const Eigen::Vector4d line_measurement(line_start[0] + 0.2,
                                           line_start[1] - 0.1,
                                           line_end[0] - 0.15,
                                           line_end[1] + 0.25);

    Eigen::Vector2d line_residual;
    Eigen::Matrix<double, 2, 6> J_line_pose;
    Eigen::Matrix<double, 2, 4> J_line;
    Eigen::Vector3d line_2d_residual;
    Eigen::Vector3d line_2d_pose;
    Eigen::Vector3d line_2d_line;
    const Eigen::Vector4d line_delta = Eigen::Vector4d::Zero();
    gb::generated::local_ba_line_residual(
        gb::ExpSO3(Eigen::Vector3d(line_pose.head<3>())),
        Eigen::Vector3d(line_pose.tail<3>()), line_state.U, line_state.W, line_delta, rcb, tcb, camera.fx,
        camera.fy, camera.cx, camera.cy, line_measurement, line_residual,
        line_2d_residual);
    gb::generated::local_ba_line_jacobian_pose(
        gb::ExpSO3(Eigen::Vector3d(line_pose.head<3>())),
        Eigen::Vector3d(line_pose.tail<3>()), line_state.U, line_state.W, line_delta, rcb, tcb, camera.fx,
        camera.fy, camera.cx, camera.cy, line_measurement, J_line_pose,
        line_2d_pose);
    gb::generated::local_ba_line_jacobian_line(
        gb::ExpSO3(Eigen::Vector3d(line_pose.head<3>())),
        Eigen::Vector3d(line_pose.tail<3>()), line_state.U, line_state.W, line_delta, rcb, tcb, camera.fx,
        camera.fy, camera.cx, camera.cy, line_measurement, J_line,
        line_2d_line);

    if ((line_2d_residual - line_2d_pose).cwiseAbs().maxCoeff() > 1e-10 ||
        (line_2d_residual - line_2d_line).cwiseAbs().maxCoeff() > 1e-10) {
        std::cerr << "Generated local BA line projections disagree\n";
        return false;
    }

    const Eigen::Vector2d line_reference = LineResidualReference(
        line_pose, line_state.line_w, rcb, tcb, camera, line_measurement);
    if ((line_residual - line_reference).cwiseAbs().maxCoeff() > 1e-10) {
        std::cerr << "Local BA line residual does not match reference\n";
        return false;
    }
    ok = CheckJacobian<2, 6>(
             "local_ba_line pose", J_line_pose, Vector6::Zero(),
             [&](const Vector6 &pose) {
                 return LineResidualReference(RetractPose(line_pose, pose), line_state.line_w, rcb, tcb,
                                              camera, line_measurement);
             },
             2e-4, 2e-6) &&
         ok;
    ok = CheckJacobian<2, 4>(
             "local_ba_line tangent", J_line, Eigen::Vector4d::Zero(),
             [&](const Eigen::Vector4d &delta) {
                 const LineState updated = ApplyLineUpdate(line_state, delta);
                 return LineResidualReference(line_pose, updated.line_w, rcb,
                                              tcb, camera, line_measurement);
             },
             2e-4, 2e-6) &&
         ok;

    return ok;
}

double LocalBAChi2(
    const std::vector<gb::FlatPoseVertex> &poses,
    const std::vector<gb::FlatLocalBAPointVertex> &points,
    const std::vector<gb::FlatLocalBALineVertex> &lines,
    const std::vector<gb::FlatLocalBAPointObservation> &point_observations,
    const std::vector<gb::FlatLocalBALineObservation> &line_observations) {
    double chi2 = 0.0;
    for (const auto &observation : point_observations) {
        const gb::Pose pose = MakePoseVertex(
            poses[observation.pose], observation.rcb, observation.tcb);
        const Eigen::Vector2d measurement(observation.measurement[0],
                                          observation.measurement[1]);
        const Eigen::Vector2d residual =
            measurement -
            gb::ProjectWorldPoint(pose, gb::MakePinholeCamera(
                                            observation.camera.fx,
                                            observation.camera.fy,
                                            observation.camera.cx,
                                            observation.camera.cy),
                                  PointFromFlat(points[observation.point]));
        chi2 += observation.information * residual.squaredNorm();
    }
    for (const auto &observation : line_observations) {
        const gb::Pose pose = MakePoseVertex(
            poses[observation.pose], observation.rcb, observation.tcb);
        const gb::PinholeCamera camera_model = gb::MakePinholeCamera(
            observation.camera.fx, observation.camera.fy,
            observation.camera.cx, observation.camera.cy);
        const gb::Vector6d line_w = LineFromFlat(lines[observation.line]);
        const Eigen::Vector4d measurement(
            observation.measurement[0], observation.measurement[1],
            observation.measurement[2], observation.measurement[3]);
        const Eigen::Vector2d residual = gb::PoseOnlyLineResidual(
            pose, camera_model, line_w, measurement);
        const Eigen::Vector3d line_2d =
            gb::ProjectWorldLine(pose, camera_model, line_w);
        const bool has_covariance =
            observation.has_endpoint_covariance &&
            XPLSLAM::IsValidLineEndpointCovariance(
                observation.endpoint_covariance_upper);
        const Eigen::Matrix2d information =
            XPLSLAM::LineResidualInformation(
                line_2d, observation.endpoint_covariance_upper,
                has_covariance, observation.information);
        chi2 += residual.dot(information * residual);
    }
    return chi2;
}

} // namespace

int main() {
    if (!CheckLocalBAJacobians()) {
        return 1;
    }

    int device_count = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (device_status != cudaSuccess || device_count <= 0) {
        std::cerr << "Skipping: CUDA device is not visible to this process\n";
        return 77;
    }

    gb::FlatPinholeCamera camera;
    camera.fx = 520.0;
    camera.fy = 515.0;
    camera.cx = 320.0;
    camera.cy = 240.0;
    camera.toPL = FlattenMatrix(
        gb::PinholeLineProjectionMatrix(camera.fx, camera.fy, camera.cx,
                                        camera.cy));
    const std::array<double, 3> rcb =
        FlattenVector3(Eigen::Vector3d(0.032, -0.017, 0.021));
    const std::array<double, 3> tcb = {0.075, -0.018, 0.028};

    std::vector<gb::FlatPoseVertex> truth_poses;
    truth_poses.push_back(MakePose(Eigen::Vector3d::Zero(),
                                   Eigen::Vector3d::Zero()));
    truth_poses.push_back(MakePose(Eigen::Vector3d(0.035, -0.02, 0.018),
                                   Eigen::Vector3d(0.24, -0.08, 0.18)));
    truth_poses.push_back(MakePose(Eigen::Vector3d(0.065, -0.035, 0.03),
                                   Eigen::Vector3d(0.50, -0.12, 0.30)));

    std::vector<gb::FlatPoseVertex> initial_poses = truth_poses;
    initial_poses[2] = MakePose(Eigen::Vector3d(0.075, -0.041, 0.038),
                                Eigen::Vector3d(0.53, -0.138, 0.32));
    const std::vector<uint8_t> fixed_poses = {1, 1, 0};

    std::vector<gb::FlatLocalBAPointVertex> truth_points;
    std::vector<gb::FlatLocalBAPointVertex> initial_points;
    for (int i = 0; i < 18; ++i) {
        const double x = -0.9 + 0.22 * (i % 6);
        const double y = -0.45 + 0.28 * (i / 6);
        const double z = 3.8 + 0.15 * (i % 4);
        gb::FlatLocalBAPointVertex truth;
        truth.position = {x, y, z};
        truth_points.push_back(truth);

        gb::FlatLocalBAPointVertex initial = truth;
        initial.position[0] += 0.010 * ((i % 2) ? 1.0 : -1.0);
        initial.position[1] += 0.008 * ((i % 3) ? -1.0 : 1.0);
        initial.position[2] += 0.012 * ((i % 4) ? 1.0 : -1.0);
        initial_points.push_back(initial);
    }

    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> truth_line_points;
    std::vector<gb::FlatLocalBALineVertex> truth_lines;
    std::vector<gb::FlatLocalBALineVertex> initial_lines;
    for (int i = 0; i < 6; ++i) {
        Eigen::Vector3d p1(-0.75 + 0.18 * i, -0.35 + 0.08 * i,
                           3.6 + 0.08 * (i % 2));
        Eigen::Vector3d p2(-0.45 + 0.15 * i, 0.35 - 0.05 * i,
                           4.25 + 0.1 * (i % 3));
        truth_line_points.emplace_back(p1, p2);

        gb::FlatLocalBALineVertex truth;
        truth.line_w = FlattenVector6(MakePluckerLine(p1, p2));
        truth_lines.push_back(truth);

        p1 += Eigen::Vector3d(0.006, -0.005, 0.005);
        p2 += Eigen::Vector3d(-0.005, 0.004, -0.006);
        gb::FlatLocalBALineVertex initial;
        initial.line_w = FlattenVector6(MakePluckerLine(p1, p2));
        initial_lines.push_back(initial);
    }

    std::vector<gb::FlatLocalBAPointObservation> point_observations;
    for (size_t point_idx = 0; point_idx < truth_points.size(); ++point_idx) {
        for (size_t pose_idx = 0; pose_idx < truth_poses.size(); ++pose_idx) {
            const gb::Pose pose =
                MakePoseVertex(truth_poses[pose_idx], rcb, tcb);
            const Eigen::Vector2d measurement =
                gb::ProjectWorldPoint(pose,
                                      gb::MakePinholeCamera(
                                          camera.fx, camera.fy, camera.cx,
                                          camera.cy),
                                      PointFromFlat(truth_points[point_idx]));
            gb::FlatLocalBAPointObservation observation;
            observation.pose = pose_idx;
            observation.point = point_idx;
            observation.camera = camera;
            observation.rcb = rcb;
            observation.tcb = tcb;
            observation.measurement = {measurement[0], measurement[1]};
            observation.information = 1.0;
            point_observations.push_back(observation);
        }
    }

    std::vector<gb::FlatLocalBALineObservation> line_observations;
    for (size_t line_idx = 0; line_idx < truth_lines.size(); ++line_idx) {
        for (size_t pose_idx = 0; pose_idx < truth_poses.size(); ++pose_idx) {
            const gb::Pose pose =
                MakePoseVertex(truth_poses[pose_idx], rcb, tcb);
            const Eigen::Vector2d start =
                gb::ProjectWorldPoint(pose,
                                      gb::MakePinholeCamera(
                                          camera.fx, camera.fy, camera.cx,
                                          camera.cy),
                                      truth_line_points[line_idx].first);
            const Eigen::Vector2d end =
                gb::ProjectWorldPoint(pose,
                                      gb::MakePinholeCamera(
                                          camera.fx, camera.fy, camera.cx,
                                          camera.cy),
                                      truth_line_points[line_idx].second);
            gb::FlatLocalBALineObservation observation;
            observation.pose = pose_idx;
            observation.line = line_idx;
            observation.camera = camera;
            observation.rcb = rcb;
            observation.tcb = tcb;
            observation.measurement = {start[0], start[1], end[0], end[1]};
            Eigen::Matrix4d endpoint_covariance;
            const double offset = 0.05 * static_cast<double>(line_idx +
                                                              pose_idx);
            endpoint_covariance << 1.0 + offset, 0.12, 0.06, 0.02, 0.12,
                3.5 + offset, 0.03, 0.10, 0.06, 0.03, 2.0 + offset, 0.18,
                0.02, 0.10, 0.18, 5.0 + offset;
            observation.has_endpoint_covariance =
                XPLSLAM::PackLineEndpointCovariance(
                    endpoint_covariance,
                    observation.endpoint_covariance_upper)
                    ? 1
                    : 0;
            if (!observation.has_endpoint_covariance) {
                std::cerr << "Failed to pack local BA endpoint covariance\n";
                return 1;
            }
            observation.information = 0.25;
            line_observations.push_back(observation);
        }
    }

    const double initial_chi2 =
        LocalBAChi2(initial_poses, initial_points, initial_lines,
                   point_observations, line_observations);

    std::vector<gb::FlatPoseVertex> optimized_poses(initial_poses.size());
    std::vector<gb::FlatLocalBAPointVertex> optimized_points(
        initial_points.size());
    std::vector<gb::FlatLocalBALineVertex> optimized_lines(
        initial_lines.size());
    std::vector<uint8_t> point_outliers(point_observations.size(), 0);
    std::vector<uint8_t> line_outliers(line_observations.size(), 0);

    gb::FlatLocalBAProblem problem;
    problem.initial_poses = initial_poses.data();
    problem.fixed_poses = fixed_poses.data();
    problem.num_poses = initial_poses.size();
    problem.initial_points = initial_points.data();
    problem.num_points = initial_points.size();
    problem.initial_lines = initial_lines.data();
    problem.num_lines = initial_lines.size();
    problem.point_observations = point_observations.data();
    problem.num_point_observations = point_observations.size();
    problem.line_observations = line_observations.data();
    problem.num_line_observations = line_observations.size();
    problem.schedule = gb::FlatLocalBASchedule::kTwoStage;
    problem.first_stage_iterations = 5;
    problem.second_stage_iterations = 10;
    problem.single_stage_iterations = 10;
    problem.initial_damping = 1e-4;

    gb::FlatLocalBAResult result;
    result.optimized_poses = optimized_poses.data();
    result.num_optimized_poses = optimized_poses.size();
    result.optimized_points = optimized_points.data();
    result.num_optimized_points = optimized_points.size();
    result.optimized_lines = optimized_lines.data();
    result.num_optimized_lines = optimized_lines.size();
    result.point_outliers = point_outliers.data();
    result.num_point_outliers = point_outliers.size();
    result.line_outliers = line_outliers.data();
    result.num_line_outliers = line_outliers.size();

    if (!gb::OptimizeLocalBAFlat(problem, result)) {
        std::cerr << "OptimizeLocalBAFlat returned false\n";
        return 1;
    }

    for (size_t i = 0; i < initial_poses.size(); ++i) {
        if (fixed_poses[i] &&
            (optimized_poses[i].Rwb != initial_poses[i].Rwb ||
             optimized_poses[i].twb != initial_poses[i].twb)) {
            std::cerr << "Local BA changed a fixed Twb vertex\n";
            return 1;
        }
    }

    const double final_chi2 =
        LocalBAChi2(optimized_poses, optimized_points, optimized_lines,
                   point_observations, line_observations);
    const double rotation_error =
        RotationAngle(Rotation(optimized_poses[2]).transpose() *
                      Rotation(truth_poses[2]));
    const double translation_error =
        (Translation(optimized_poses[2]) - Translation(truth_poses[2])).norm();
    const int point_outlier_count =
        std::count(point_outliers.begin(), point_outliers.end(), uint8_t{1});
    const int line_outlier_count =
        std::count(line_outliers.begin(), line_outliers.end(), uint8_t{1});

    if (final_chi2 >= initial_chi2 * 0.02 || rotation_error > 2e-2 ||
        translation_error > 6e-2 || point_outlier_count != 0 ||
        line_outlier_count != 0 || !result.used_two_stage) {
        std::cerr << std::setprecision(12)
                  << "Local BA optimization failed\ninitial_chi2="
                  << initial_chi2 << " final_chi2=" << final_chi2
                  << " rotation_error=" << rotation_error
                  << " translation_error=" << translation_error
                  << " point_outliers=" << point_outlier_count
                  << " line_outliers=" << line_outlier_count
                  << " used_two_stage=" << result.used_two_stage << '\n';
        return 1;
    }

    std::vector<gb::FlatPoseVertex> strict_optimized_poses(
        initial_poses.size());
    std::vector<gb::FlatLocalBAPointVertex> strict_optimized_points(
        initial_points.size());
    std::vector<gb::FlatLocalBALineVertex> strict_optimized_lines(
        initial_lines.size());
    std::vector<uint8_t> strict_point_outliers(point_observations.size(), 0);
    std::vector<uint8_t> strict_line_outliers(line_observations.size(), 0);

    gb::FlatLocalBAProblem strict_problem = problem;
    strict_problem.schedule = gb::FlatLocalBASchedule::kSingleStage;
    strict_problem.single_stage_iterations = 50;
    strict_problem.huber_delta = 1e12;
    strict_problem.chi2_threshold = 1e12;

    gb::FlatLocalBAResult strict_result;
    strict_result.optimized_poses = strict_optimized_poses.data();
    strict_result.num_optimized_poses = strict_optimized_poses.size();
    strict_result.optimized_points = strict_optimized_points.data();
    strict_result.num_optimized_points = strict_optimized_points.size();
    strict_result.optimized_lines = strict_optimized_lines.data();
    strict_result.num_optimized_lines = strict_optimized_lines.size();
    strict_result.point_outliers = strict_point_outliers.data();
    strict_result.num_point_outliers = strict_point_outliers.size();
    strict_result.line_outliers = strict_line_outliers.data();
    strict_result.num_line_outliers = strict_line_outliers.size();

    if (!gb::OptimizeLocalBAFlat(strict_problem, strict_result)) {
        std::cerr << "Strict OptimizeLocalBAFlat returned false\n";
        return 1;
    }

    const double strict_final_chi2 = LocalBAChi2(
        strict_optimized_poses, strict_optimized_points,
        strict_optimized_lines, point_observations, line_observations);
    const double strict_rotation_error =
        RotationAngle(Rotation(strict_optimized_poses[2]).transpose() *
                      Rotation(truth_poses[2]));
    const double strict_translation_error =
        (Translation(strict_optimized_poses[2]) - Translation(truth_poses[2]))
            .norm();
    if (strict_final_chi2 > 1e-6 || strict_rotation_error > 1e-5 ||
        strict_translation_error > 1e-5) {
        auto diagnose_strict = [&](const char *name, bool use_points,
                                   bool use_lines, bool truth_pose_initial,
                                   bool truth_line_initial) {
            std::vector<gb::FlatPoseVertex> diag_initial_poses = initial_poses;
            if (truth_pose_initial) {
                diag_initial_poses[2] = truth_poses[2];
            }
            std::vector<gb::FlatLocalBALineVertex> diag_initial_lines =
                truth_line_initial ? truth_lines : initial_lines;

            std::vector<gb::FlatPoseVertex> diag_poses(initial_poses.size());
            std::vector<gb::FlatLocalBAPointVertex> diag_points(
                use_points ? initial_points.size() : 0);
            std::vector<gb::FlatLocalBALineVertex> diag_lines(
                use_lines ? diag_initial_lines.size() : 0);
            std::vector<uint8_t> diag_point_outliers(
                use_points ? point_observations.size() : 0, 0);
            std::vector<uint8_t> diag_line_outliers(
                use_lines ? line_observations.size() : 0, 0);

            gb::FlatLocalBAProblem diag_problem = strict_problem;
            diag_problem.initial_poses = diag_initial_poses.data();
            diag_problem.initial_points =
                use_points ? initial_points.data() : nullptr;
            diag_problem.num_points = use_points ? initial_points.size() : 0;
            diag_problem.initial_lines =
                use_lines ? diag_initial_lines.data() : nullptr;
            diag_problem.num_lines =
                use_lines ? diag_initial_lines.size() : 0;
            diag_problem.point_observations =
                use_points ? point_observations.data() : nullptr;
            diag_problem.num_point_observations =
                use_points ? point_observations.size() : 0;
            diag_problem.line_observations =
                use_lines ? line_observations.data() : nullptr;
            diag_problem.num_line_observations =
                use_lines ? line_observations.size() : 0;

            gb::FlatLocalBAResult diag_result;
            diag_result.optimized_poses = diag_poses.data();
            diag_result.num_optimized_poses = diag_poses.size();
            diag_result.optimized_points = diag_points.data();
            diag_result.num_optimized_points = diag_points.size();
            diag_result.optimized_lines = diag_lines.data();
            diag_result.num_optimized_lines = diag_lines.size();
            diag_result.point_outliers = diag_point_outliers.data();
            diag_result.num_point_outliers = diag_point_outliers.size();
            diag_result.line_outliers = diag_line_outliers.data();
            diag_result.num_line_outliers = diag_line_outliers.size();

            const bool ok = gb::OptimizeLocalBAFlat(diag_problem, diag_result);
            if (!ok) {
                std::cerr << name << " strict diagnostic returned false\n";
                return;
            }
            const double chi2 = LocalBAChi2(
                diag_poses, diag_points, diag_lines,
                use_points ? point_observations
                           : std::vector<gb::FlatLocalBAPointObservation>{},
                use_lines ? line_observations
                          : std::vector<gb::FlatLocalBALineObservation>{});
            const double rot =
                RotationAngle(Rotation(diag_poses[2]).transpose() *
                              Rotation(truth_poses[2]));
            const double trans =
                (Translation(diag_poses[2]) - Translation(truth_poses[2]))
                    .norm();
            std::cerr << std::setprecision(12) << name
                      << " strict diagnostic chi2=" << chi2
                      << " rotation_error=" << rot
                      << " translation_error=" << trans << '\n';
        };
        diagnose_strict("point-only", true, false, false, false);
        diagnose_strict("line-only", false, true, false, false);
        diagnose_strict("line-only truth-pose-init", false, true, true, false);
        diagnose_strict("line-only truth-line-init", false, true, false, true);
        std::cerr << std::setprecision(12)
                  << "Strict Local BA convergence failed\ninitial_chi2="
                  << initial_chi2
                  << " strict_final_chi2=" << strict_final_chi2
                  << " strict_rotation_error=" << strict_rotation_error
                  << " strict_translation_error="
                  << strict_translation_error << '\n';
        return 1;
    }

    std::cout << std::setprecision(12)
              << "graphite_local_ba_optimization_check passed, initial_chi2="
              << initial_chi2 << ", final_chi2=" << final_chi2
              << ", rotation_error=" << rotation_error
              << ", translation_error=" << translation_error << '\n';
    return 0;
}
