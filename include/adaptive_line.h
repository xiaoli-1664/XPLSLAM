#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <vector>

namespace XPLSLAM {

struct AdaptiveLineConfig {
    bool enabled = true;
    int min_point_matches = 50;
    int coverage_rows = 4;
    int coverage_cols = 4;
    double min_coverage_ratio = 0.25;
    double min_normalized_eigenvalue = 0.05;
    double line_precheck_chi2 = 9.21; // Final-pose association check, not selection.
    double min_line_gain = 1e-3;
    int max_selected_lines = 50;
    int target_observations = 50;
};

struct PoseConstraintSummary {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Matrix<double, 6, 6> point_hessian =
        Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 6> normalized_hessian =
        Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> column_scale =
        Eigen::Matrix<double, 6, 1>::Ones();
    int point_matches = 0; // Valid matches at the incoming pose.
    int point_inliers = 0; // Inliers after the single pose optimization.
    int selected_line_inliers = 0;
    int selected_lines = 0;
    double coverage_ratio = 0.0;
    double min_eigenvalue = 0.0;
    bool numerically_valid = false;
    bool point_constraints_strong = false;
};

struct NormalizedPoseHessian {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Matrix<double, 6, 1> scale =
        Eigen::Matrix<double, 6, 1>::Ones();
    Eigen::Matrix<double, 6, 6> hessian =
        Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> eigenvalues =
        Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 6> eigenvectors =
        Eigen::Matrix<double, 6, 6>::Identity();
    bool valid = false;
};

inline NormalizedPoseHessian NormalizePoseHessian(
    const Eigen::Matrix<double, 6, 6> &input) {
    NormalizedPoseHessian result;
    if (!input.allFinite()) {
        return result;
    }
    const double max_diagonal = std::max(1.0, input.diagonal().maxCoeff());
    const double epsilon = 1e-9 * max_diagonal;
    for (int i = 0; i < 6; ++i) {
        if (!std::isfinite(input(i, i)) || input(i, i) < 0.0) {
            return result;
        }
        result.scale(i) =
            1.0 / std::sqrt(std::max(input(i, i), epsilon));
    }
    result.hessian =
        result.scale.asDiagonal() * input * result.scale.asDiagonal();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(
        result.hessian);
    if (solver.info() != Eigen::Success ||
        !solver.eigenvalues().allFinite() ||
        !solver.eigenvectors().allFinite()) {
        return result;
    }
    result.eigenvalues = solver.eigenvalues();
    result.eigenvectors = solver.eigenvectors();
    result.valid = true;
    return result;
}

inline double WeakestDirectionGain(
    const Eigen::Matrix<double, 6, 6> &line_hessian,
    const Eigen::Matrix<double, 6, 1> &scale,
    const Eigen::Matrix<double, 6, 1> &weakest_direction) {
    const Eigen::Matrix<double, 6, 6> normalized_line =
        scale.asDiagonal() * line_hessian * scale.asDiagonal();
    return weakest_direction.dot(normalized_line * weakest_direction);
}

} // namespace XPLSLAM
