#include "line_observation_covariance.h"

#include <Eigen/Core>

#include <cmath>
#include <iostream>

int main() {
    Eigen::Matrix4d covariance = Eigen::Matrix4d::Identity();
    covariance(0, 0) = 4.0;
    covariance(1, 1) = 9.0;
    covariance(2, 2) = 16.0;
    covariance(3, 3) = 25.0;
    XPLSLAM::PackedLineEndpointCovariance packed{};
    if (!XPLSLAM::PackLineEndpointCovariance(covariance, packed)) {
        return 1;
    }

    const Eigen::Vector3d line(1.0, 0.0, -10.0);
    const Eigen::Matrix2d with_covariance =
        XPLSLAM::LineResidualInformation(line, packed, true);
    const Eigen::Matrix2d without_covariance =
        XPLSLAM::LineResidualInformation(line, packed, false);
    const Eigen::Matrix2d explicit_fallback =
        XPLSLAM::LineResidualInformation(line, packed, false, 0.25);

#ifdef XPLSLAM_ABLATION_UNIT_LINE_INFORMATION
    const Eigen::Matrix2d expected = Eigen::Matrix2d::Identity();
    const bool ok = with_covariance.isApprox(expected, 1e-12) &&
                    without_covariance.isApprox(expected, 1e-12) &&
                    explicit_fallback.isApprox(expected, 1e-12) &&
                    XPLSLAM::kLineInformationFallback == 1.0;
#else
    Eigen::Matrix2d expected_covariance = Eigen::Matrix2d::Zero();
    expected_covariance(0, 0) = 0.25;
    expected_covariance(1, 1) = 0.0625;
    const bool ok = with_covariance.isApprox(expected_covariance, 1e-12) &&
                    without_covariance.isApprox(
                        0.25 * Eigen::Matrix2d::Identity(), 1e-12) &&
                    explicit_fallback.isApprox(
                        0.25 * Eigen::Matrix2d::Identity(), 1e-12) &&
                    XPLSLAM::kLineInformationFallback == 0.25;
#endif
    if (!ok) {
        std::cerr << "Unexpected line information policy\n";
        return 1;
    }
    return 0;
}
