#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define XPLSLAM_LINE_HD __host__ __device__
#else
#define XPLSLAM_LINE_HD
#endif

namespace XPLSLAM {

using PackedLineEndpointCovariance = std::array<float, 10>;

constexpr double kLegacyLineInformation = 0.25;
#ifdef XPLSLAM_ABLATION_UNIT_LINE_INFORMATION
constexpr double kLineInformationFallback = 1.0;
#else
constexpr double kLineInformationFallback = kLegacyLineInformation;
#endif
constexpr double kLineResidualVarianceFloor = 1e-6;

template <typename T>
XPLSLAM_LINE_HD inline bool IsFiniteLineCovarianceValue(T value) {
#if defined(__CUDA_ARCH__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
}

inline bool PackLineEndpointCovariance(
    const Eigen::Matrix4d &input, PackedLineEndpointCovariance &packed) {
    if (!input.allFinite()) {
        return false;
    }

    const double scale = std::max(1.0, input.cwiseAbs().maxCoeff());
    if ((input - input.transpose()).cwiseAbs().maxCoeff() > 1e-5 * scale) {
        return false;
    }

    const Eigen::Matrix4d covariance = 0.5 * (input + input.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> solver(
        covariance, Eigen::EigenvaluesOnly);
    if (solver.info() != Eigen::Success ||
        solver.eigenvalues().minCoeff() < -1e-6 * scale) {
        return false;
    }

    packed = {static_cast<float>(covariance(0, 0)),
              static_cast<float>(covariance(0, 1)),
              static_cast<float>(covariance(0, 2)),
              static_cast<float>(covariance(0, 3)),
              static_cast<float>(covariance(1, 1)),
              static_cast<float>(covariance(1, 2)),
              static_cast<float>(covariance(1, 3)),
              static_cast<float>(covariance(2, 2)),
              static_cast<float>(covariance(2, 3)),
              static_cast<float>(covariance(3, 3))};
    return true;
}

inline bool PackLineEndpointCovariance(
    const Eigen::Matrix4f &input, PackedLineEndpointCovariance &packed) {
    const Eigen::Matrix4d covariance = input.cast<double>();
    return PackLineEndpointCovariance(covariance, packed);
}

inline Eigen::Matrix4d UnpackLineEndpointCovariance(
    const PackedLineEndpointCovariance &packed) {
    Eigen::Matrix4d covariance;
    covariance << packed[0], packed[1], packed[2], packed[3], packed[1],
        packed[4], packed[5], packed[6], packed[2], packed[5], packed[7],
        packed[8], packed[3], packed[6], packed[8], packed[9];
    return covariance;
}

inline bool IsValidLineEndpointCovariance(
    const PackedLineEndpointCovariance &packed) {
    PackedLineEndpointCovariance validated;
    return PackLineEndpointCovariance(
        UnpackLineEndpointCovariance(packed), validated);
}

// Returns the lower-triangular inverse Cholesky factor W such that
// ||W r||^2 = r^T Sigma_r^-1 r. The endpoint covariance is ordered as
// (x1, y1, x2, y2). The covariance is frozen while differentiating W r.
template <typename T>
XPLSLAM_LINE_HD inline void SetFallbackLineWhitening(
    T fallback_information, T *whitening) {
    const T fallback =
        fallback_information > T(0) &&
                IsFiniteLineCovarianceValue(fallback_information)
            ? fallback_information
            : T(kLineInformationFallback);
    const T scale = sqrt(fallback);
    whitening[0] = scale;
    whitening[1] = T(0);
    whitening[2] = T(0);
    whitening[3] = scale;
}

template <typename T>
XPLSLAM_LINE_HD inline bool ComputeLineResidualWhitening(
    const T *line_2d, const float *endpoint_covariance_upper,
    bool has_endpoint_covariance, T fallback_information, T *whitening) {
#ifdef XPLSLAM_ABLATION_UNIT_LINE_INFORMATION
    (void)line_2d;
    (void)endpoint_covariance_upper;
    (void)has_endpoint_covariance;
    (void)fallback_information;
    whitening[0] = T(1);
    whitening[1] = T(0);
    whitening[2] = T(0);
    whitening[3] = T(1);
    return false;
#else
    if (!has_endpoint_covariance || !endpoint_covariance_upper) {
        SetFallbackLineWhitening(fallback_information, whitening);
        return false;
    }

    const T a = line_2d[0];
    const T b = line_2d[1];
    const T normal_norm = sqrt(a * a + b * b);
    if (!(normal_norm > T(1e-12)) ||
        !IsFiniteLineCovarianceValue(normal_norm)) {
        SetFallbackLineWhitening(fallback_information, whitening);
        return false;
    }

    const T nx = a / normal_norm;
    const T ny = b / normal_norm;
    const T nx2 = nx * nx;
    const T nxy = nx * ny;
    const T ny2 = ny * ny;
    const T c00 = T(endpoint_covariance_upper[0]);
    const T c01 = T(endpoint_covariance_upper[1]);
    const T c02 = T(endpoint_covariance_upper[2]);
    const T c03 = T(endpoint_covariance_upper[3]);
    const T c11 = T(endpoint_covariance_upper[4]);
    const T c12 = T(endpoint_covariance_upper[5]);
    const T c13 = T(endpoint_covariance_upper[6]);
    const T c22 = T(endpoint_covariance_upper[7]);
    const T c23 = T(endpoint_covariance_upper[8]);
    const T c33 = T(endpoint_covariance_upper[9]);

    const T variance0 = nx2 * c00 + T(2) * nxy * c01 + ny2 * c11;
    const T covariance01 =
        nx2 * c02 + nxy * (c03 + c12) + ny2 * c13;
    const T variance1 = nx2 * c22 + T(2) * nxy * c23 + ny2 * c33;
    if (!IsFiniteLineCovarianceValue(variance0) ||
        !IsFiniteLineCovarianceValue(covariance01) ||
        !IsFiniteLineCovarianceValue(variance1)) {
        SetFallbackLineWhitening(fallback_information, whitening);
        return false;
    }

    const T floor = T(kLineResidualVarianceFloor);
    const T l00 = sqrt(variance0 > floor ? variance0 : floor);
    const T l10 = covariance01 / l00;
    const T schur = variance1 - l10 * l10;
    const T l11 = sqrt(schur > floor ? schur : floor);
    if (!IsFiniteLineCovarianceValue(l00) ||
        !IsFiniteLineCovarianceValue(l10) ||
        !IsFiniteLineCovarianceValue(l11)) {
        SetFallbackLineWhitening(fallback_information, whitening);
        return false;
    }

    whitening[0] = T(1) / l00;
    whitening[1] = T(0);
    whitening[2] = -l10 / (l00 * l11);
    whitening[3] = T(1) / l11;
    return true;
#endif
}

inline Eigen::Matrix2d LineResidualWhitening(
    const Eigen::Vector3d &line_2d,
    const PackedLineEndpointCovariance &endpoint_covariance,
    bool has_endpoint_covariance,
    double fallback_information = kLineInformationFallback) {
    double values[4];
    ComputeLineResidualWhitening(line_2d.data(), endpoint_covariance.data(),
                                 has_endpoint_covariance,
                                 fallback_information, values);
    Eigen::Matrix2d whitening;
    whitening << values[0], values[1], values[2], values[3];
    return whitening;
}

inline Eigen::Matrix2d LineResidualInformation(
    const Eigen::Vector3d &line_2d,
    const PackedLineEndpointCovariance &endpoint_covariance,
    bool has_endpoint_covariance,
    double fallback_information = kLineInformationFallback) {
    const Eigen::Matrix2d whitening =
        LineResidualWhitening(line_2d, endpoint_covariance,
                              has_endpoint_covariance, fallback_information);
    return whitening.transpose() * whitening;
}

} // namespace XPLSLAM

#undef XPLSLAM_LINE_HD
