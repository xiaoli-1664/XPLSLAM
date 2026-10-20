#include "adaptive_line.h"

#include <Eigen/Core>

#include <cmath>
#include <iostream>
#include <limits>

int main() {
    using Matrix6d = Eigen::Matrix<double, 6, 6>;
    using Vector6d = Eigen::Matrix<double, 6, 1>;
    using XPLSLAM::NormalizePoseHessian;
    using XPLSLAM::WeakestDirectionGain;

    Matrix6d hessian = Matrix6d::Identity();
    hessian(0, 1) = hessian(1, 0) = 0.35;
    hessian(2, 5) = hessian(5, 2) = -0.2;
    const auto reference = NormalizePoseHessian(hessian);
    if (!reference.valid || reference.eigenvalues(0) <= 0.0) {
        std::cerr << "valid Hessian rejected" << std::endl;
        return 1;
    }

    Vector6d unit_change;
    unit_change << 100.0, 10.0, 2.0, 0.01, 0.1, 0.5;
    const Matrix6d rescaled =
        unit_change.asDiagonal() * hessian * unit_change.asDiagonal();
    const auto normalized_rescaled = NormalizePoseHessian(rescaled);
    if (!normalized_rescaled.valid ||
        (reference.hessian - normalized_rescaled.hessian).norm() > 1e-12) {
        std::cerr << "rotation/translation unit scaling changed the normalized "
                     "Hessian"
                  << std::endl;
        return 2;
    }

    const Matrix6d rank_one = Matrix6d::Ones();
    const auto degenerate = NormalizePoseHessian(rank_one);
    if (!degenerate.valid || std::fabs(degenerate.eigenvalues(0)) > 1e-10) {
        std::cerr << "degenerate Hessian was not detected by its minimum "
                     "eigenvalue"
                  << std::endl;
        return 3;
    }

    Matrix6d invalid = Matrix6d::Identity();
    invalid(0, 0) = std::numeric_limits<double>::quiet_NaN();
    if (NormalizePoseHessian(invalid).valid) {
        std::cerr << "non-finite Hessian accepted" << std::endl;
        return 4;
    }

    const Vector6d weakest = Vector6d::Unit(0);
    Matrix6d useful_line = Matrix6d::Zero();
    useful_line(0, 0) = 4.0;
    Matrix6d orthogonal_line = Matrix6d::Zero();
    orthogonal_line(1, 1) = 100.0;
    const double useful_gain =
        WeakestDirectionGain(useful_line, Vector6d::Ones(), weakest);
    const double orthogonal_gain =
        WeakestDirectionGain(orthogonal_line, Vector6d::Ones(), weakest);
    if (!(useful_gain > orthogonal_gain && useful_gain == 4.0)) {
        std::cerr << "weakest-direction gain ordering is incorrect" << std::endl;
        return 5;
    }

    std::cout << "adaptive line constraint check passed" << std::endl;
    return 0;
}
