#include "graphite_essential_graph_jacobians.cuh"
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
#include <vector>

namespace {

namespace gb = XPLSLAM::graphite_backend;

using Vector6 = Eigen::Matrix<double, 6, 1>;

Eigen::Matrix3d Skew(const Eigen::Vector3d &v) {
    Eigen::Matrix3d skew;
    skew << 0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0;
    return skew;
}

Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R) {
    Eigen::Quaterniond q(R);
    q.normalize();
    Eigen::AngleAxisd angle_axis(q);
    if (angle_axis.angle() == 0.0) {
        return Eigen::Vector3d::Zero();
    }
    return angle_axis.axis() * angle_axis.angle();
}

Eigen::Matrix3d InverseLeftJacobianSO3(const Eigen::Vector3d &w) {
    const double theta2 = w.squaredNorm();
    const double theta = std::sqrt(theta2);
    const Eigen::Matrix3d W = Skew(w);
    if (theta < 1e-5) {
        return Eigen::Matrix3d::Identity() - 0.5 * W +
               (1.0 / 12.0) * W * W;
    }
    return Eigen::Matrix3d::Identity() - 0.5 * W +
           (1.0 - theta / (2.0 * std::tan(theta / 2.0))) / theta2 * W * W;
}

Vector6 LogSE3(const Eigen::Matrix3d &R, const Eigen::Vector3d &t) {
    Vector6 log;
    const Eigen::Vector3d omega = LogSO3(R);
    log.head<3>() = omega;
    log.tail<3>() = InverseLeftJacobianSO3(omega) * t;
    return log;
}

gb::FlatPoseVertex MakePose(const Eigen::Vector3d &r,
                            const Eigen::Vector3d &t) {
    gb::FlatPoseVertex pose;
    const Eigen::Matrix3d R = gb::ExpSO3(r);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            pose.Rwb[static_cast<size_t>(3 * row + col)] = R(row, col);
        }
        pose.twb[static_cast<size_t>(row)] = t[row];
    }
    return pose;
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

gb::FlatPoseVertex Compose(const gb::FlatPoseVertex &a,
                           const gb::FlatPoseVertex &b) {
    const Eigen::Matrix3d R = Rotation(a) * Rotation(b);
    const Eigen::Vector3d t = Rotation(a) * Translation(b) + Translation(a);
    gb::FlatPoseVertex pose;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            pose.Rwb[static_cast<size_t>(3 * row + col)] = R(row, col);
        }
        pose.twb[static_cast<size_t>(row)] = t[row];
    }
    return pose;
}

gb::FlatPoseVertex Inverse(const gb::FlatPoseVertex &pose) {
    const Eigen::Matrix3d R = Rotation(pose).transpose();
    const Eigen::Vector3d t = -R * Translation(pose);
    gb::FlatPoseVertex inverse;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            inverse.Rwb[static_cast<size_t>(3 * row + col)] = R(row, col);
        }
        inverse.twb[static_cast<size_t>(row)] = t[row];
    }
    return inverse;
}

gb::FlatSE3Transform Measurement(const gb::FlatPoseVertex &transform) {
    return {transform.Rwb, transform.twb};
}

gb::FlatPoseVertex Transform(const gb::FlatSE3Transform &measurement) {
    return {measurement.rotation, measurement.translation};
}

gb::FlatPoseVertex Retract(const gb::FlatPoseVertex &pose, const Vector6 &delta) {
    const Eigen::Matrix3d R = Rotation(pose) *
        gb::ExpSO3(Eigen::Vector3d(delta.head<3>()));
    return MakePose(LogSO3(R), Translation(pose) + delta.tail<3>());
}

Vector6 GeneratedResidual(const gb::FlatPoseVertex &pose_i,
                          const gb::FlatPoseVertex &pose_j,
                          const gb::FlatPoseVertex &measurement) {
    Vector6 residual;
    gb::generated::essential_graph_residual(
        Rotation(pose_i), Translation(pose_i), Rotation(pose_j), Translation(pose_j),
        LogSO3(Rotation(measurement)), Translation(measurement), residual);
    return residual;
}

double PoseError(const gb::FlatPoseVertex &actual,
                 const gb::FlatPoseVertex &expected) {
    const Eigen::Matrix3d dR = Rotation(actual).transpose() * Rotation(expected);
    const Eigen::Vector3d dt = Translation(actual) - Translation(expected);
    return LogSO3(dR).norm() + dt.norm();
}

double GraphChi2(const std::vector<gb::FlatPoseVertex> &poses,
                 const std::vector<gb::FlatEssentialGraphEdgeInput> &edges) {
    double chi2 = 0.0;
    for (const gb::FlatEssentialGraphEdgeInput &edge : edges) {
        const gb::FlatPoseVertex error_pose =
            Compose(Transform(edge.measurement_Tji),
                    Compose(Inverse(poses[edge.vertex_i]), poses[edge.vertex_j]));
        const Vector6 residual = LogSE3(Rotation(error_pose),
                                        Translation(error_pose));
        chi2 += residual.squaredNorm();
    }
    return chi2;
}

std::array<double, 36> IdentityInformation6() {
    std::array<double, 36> information{};
    for (int i = 0; i < 6; ++i) {
        information[static_cast<size_t>(6 * i + i)] = 1.0;
    }
    return information;
}

bool CheckGeneratedJacobian() {
    const gb::FlatPoseVertex pose_i =
        MakePose(Eigen::Vector3d(0.04, -0.03, 0.02),
                 Eigen::Vector3d(0.2, -0.1, 0.08));
    const gb::FlatPoseVertex pose_j =
        MakePose(Eigen::Vector3d(-0.02, 0.025, -0.015),
                 Eigen::Vector3d(-0.15, 0.18, 0.05));
    gb::FlatPoseVertex measurement = Compose(Inverse(pose_j), pose_i);
    measurement = Compose(MakePose(Eigen::Vector3d(1e-4, -8e-5, 6e-5),
                                   Eigen::Vector3d(9e-5, -7e-5, 5e-5)),
                          measurement);

    Eigen::Matrix<double, 6, 6> Ji;
    Eigen::Matrix<double, 6, 6> Jj;
    gb::generated::essential_graph_jacobian_i(
        Rotation(pose_i), Translation(pose_i), Rotation(pose_j), Translation(pose_j), LogSO3(Rotation(measurement)),
        Translation(measurement), Ji);
    gb::generated::essential_graph_jacobian_j(
        Rotation(pose_i), Translation(pose_i), Rotation(pose_j), Translation(pose_j), LogSO3(Rotation(measurement)),
        Translation(measurement), Jj);

    constexpr double eps = 1e-6;
    Eigen::Matrix<double, 6, 6> numeric_i;
    Eigen::Matrix<double, 6, 6> numeric_j;
    for (int col = 0; col < 6; ++col) {
        Vector6 delta = Vector6::Zero();
        delta[col] = eps;
        numeric_i.col(col) =
            (GeneratedResidual(Retract(pose_i, delta), pose_j, measurement) -
             GeneratedResidual(Retract(pose_i, -delta), pose_j, measurement)) /
            (2.0 * eps);
        numeric_j.col(col) =
            (GeneratedResidual(pose_i, Retract(pose_j, delta), measurement) -
             GeneratedResidual(pose_i, Retract(pose_j, -delta), measurement)) /
            (2.0 * eps);
    }

    const double max_i = (Ji - numeric_i).cwiseAbs().maxCoeff();
    const double max_j = (Jj - numeric_j).cwiseAbs().maxCoeff();
    if (max_i > 2e-5 || max_j > 2e-5) {
        std::cerr << std::setprecision(12)
                  << "generated jacobian mismatch, max_i=" << max_i
                  << " max_j=" << max_j << '\n';
        return false;
    }
    return true;
}

bool CheckOptimization() {
    std::vector<gb::FlatPoseVertex> truth;
    truth.push_back(MakePose(Eigen::Vector3d(0.3, -0.2, 0.15),
                             Eigen::Vector3d(0.8, -0.5, 0.4)));
    truth.push_back(MakePose(Eigen::Vector3d(0.04, -0.02, 0.03),
                             Eigen::Vector3d(0.35, -0.08, 0.12)));
    truth.push_back(MakePose(Eigen::Vector3d(0.08, -0.04, 0.05),
                             Eigen::Vector3d(0.72, -0.16, 0.20)));
    truth.push_back(MakePose(Eigen::Vector3d(0.10, -0.03, 0.08),
                             Eigen::Vector3d(1.05, -0.05, 0.32)));
    truth.push_back(MakePose(Eigen::Vector3d(0.13, -0.01, 0.10),
                             Eigen::Vector3d(1.38, 0.06, 0.45)));

    std::vector<gb::FlatPoseVertex> initial = truth;
    for (size_t i = 1; i < initial.size(); ++i) {
        initial[i] = Compose(
            MakePose(Eigen::Vector3d(0.01 * i, -0.008 * i, 0.006 * i),
                     Eigen::Vector3d(0.04 * i, -0.025 * i, 0.03 * i)),
            initial[i]);
    }

    const std::array<double, 36> information = IdentityInformation6();
    std::vector<std::pair<size_t, size_t>> pairs = {
        {0, 1}, {1, 2}, {2, 3}, {3, 4}, {0, 2}, {1, 3}, {0, 4}};
    std::vector<gb::FlatEssentialGraphEdgeInput> edges;
    for (const auto &[i, j] : pairs) {
        gb::FlatEssentialGraphEdgeInput edge;
        edge.vertex_i = i;
        edge.vertex_j = j;
        edge.measurement_Tji = Measurement(Compose(Inverse(truth[j]), truth[i]));
        edge.information = information;
        edges.push_back(edge);
    }

    std::vector<uint8_t> fixed(initial.size(), 0);
    fixed[0] = 1;
    std::vector<gb::FlatPoseVertex> optimized(initial.size());

    gb::FlatEssentialGraphProblem problem;
    problem.initial_poses = initial.data();
    problem.fixed = fixed.data();
    problem.num_poses = initial.size();
    problem.edges = edges.data();
    problem.num_edges = edges.size();
    problem.iterations = 20;
    problem.initial_damping = 1e-10;

    gb::FlatEssentialGraphResult result;
    result.optimized_poses = optimized.data();
    result.num_optimized_poses = optimized.size();
    if (!gb::OptimizeEssentialGraphFlat(problem, result)) {
        std::cerr << "OptimizeEssentialGraphFlat returned false\n";
        return false;
    }

    if (optimized[0].Rwb != initial[0].Rwb || optimized[0].twb != initial[0].twb) {
        std::cerr << "Essential graph changed the fixed Twb vertex\n";
        return false;
    }
    const double initial_chi2 = GraphChi2(initial, edges);
    const double final_chi2 = GraphChi2(optimized, edges);
    double max_pose_error = 0.0;
    for (size_t i = 0; i < optimized.size(); ++i) {
        max_pose_error = std::max(max_pose_error, PoseError(optimized[i],
                                                            truth[i]));
    }

    if (final_chi2 > initial_chi2 * 1e-6 || max_pose_error > 2e-4) {
        std::cerr << std::setprecision(12)
                  << "essential graph optimization failed, initial_chi2="
                  << initial_chi2 << " final_chi2=" << final_chi2
                  << " result_initial_chi2=" << result.initial_chi2
                  << " result_final_chi2=" << result.final_chi2
                  << " max_pose_error=" << max_pose_error << '\n';
        return false;
    }
    return true;
}

} // namespace

int main() {
    int device_count = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (device_status != cudaSuccess || device_count <= 0) {
        std::cerr << "Skipping: CUDA device is not visible to this process\n";
        return 77;
    }

    if (!CheckGeneratedJacobian() || !CheckOptimization()) {
        return 1;
    }

    std::cout << "graphite_essential_graph_optimization_check passed\n";
    return 0;
}
