#include "graphite_essential_graph_jacobians.cuh"
#include "graphite_local_ba_jacobians.cuh"
#include "graphite_slam_backend.h"

#include <graphite/factor.hpp>
#include <graphite/graph.hpp>
#include <graphite/loss.hpp>
#include <graphite/optimizer/levenberg_marquardt.hpp>
#include <graphite/solver/cudss.hpp>
#include <graphite/solver/cudss_schur.hpp>
#include <graphite/stream.hpp>
#include <graphite/vertex.hpp>

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

namespace XPLSLAM::graphite_backend {
namespace {

template <typename Values>
bool AreFiniteMapValues(const Values &values) {
    for (double value : values) {
        if (!std::isfinite(value) ||
            std::abs(value) > std::numeric_limits<float>::max()) {
            return false;
        }
    }
    return true;
}

template <typename T, int E>
class SwitchableHuberLoss final : public graphite::Loss<T, E> {
  public:
    T delta;
    bool enabled;

    __device__ __host__ SwitchableHuberLoss()
        : delta(static_cast<T>(100.0)), enabled(false) {}
    __device__ __host__ SwitchableHuberLoss(T delta, bool enabled)
        : delta(delta), enabled(enabled) {}
    __device__ __host__ SwitchableHuberLoss(const SwitchableHuberLoss &other)
        : delta(other.delta), enabled(other.enabled) {}

    __device__ __host__ T loss(const T &x) const override {
        if (!enabled || x <= delta * delta) {
            return x;
        }
        return static_cast<T>(2) * sqrt(x) * delta - delta * delta;
    }

    __device__ __host__ T loss_derivative(const T &x) const override {
        if (!enabled || x <= delta * delta) {
            return static_cast<T>(1);
        }
        return delta / sqrt(x);
    }
};

struct DeviceVec2 {
    double v[2];
};

struct DeviceVec4 {
    double v[4];
};

struct DevicePinholeCamera {
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    double toPL[9] = {};
};

struct DevicePoseVertex {
    double Rwb[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    double twb[3] = {};
};

using DevicePoseGraphVertex = DevicePoseVertex;

struct EssentialGraphEdgeData {
    double measurement_r[3] = {};
    double measurement_t[3] = {};
};

struct DeviceBAPointVertex {
    double position[3] = {};
};

struct DeviceBALineVertex {
    double U[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    double W[4] = {1.0, 0.0, 0.0, 1.0};
    double delta[4] = {};
    double line_w[6] = {};
};

struct LocalBAObservationData {
    DevicePinholeCamera camera;
    double rcb[3] = {};
    double tcb[3] = {};
    double information = 1.0;
};

struct LocalBALineObservationData {
    DevicePinholeCamera camera;
    double rcb[3] = {};
    double tcb[3] = {};
    float endpoint_covariance_upper[10] = {};
    uint8_t has_endpoint_covariance = 0;
    double information = kLineInformationFallback;
};

struct PointFactorData {
    DevicePinholeCamera camera;
    double Rcb[9] = {};
    double tcb[3] = {};
    double point_w[3] = {};
    double information = 1.0;
};

struct LineFactorData {
    DevicePinholeCamera camera;
    double Rcb[9] = {};
    double tcb[3] = {};
    double line_w[6] = {};
    float endpoint_covariance_upper[10] = {};
    uint8_t has_endpoint_covariance = 0;
    double information = kLineInformationFallback;
};

template <typename T>
__device__ __forceinline__ T Mat3At(const double *matrix, int row, int col) {
    return static_cast<T>(matrix[3 * row + col]);
}

template <typename T>
__device__ __forceinline__ void Mat3Mul(const T *a, const T *b, T *out) {
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            out[3 * row + col] = a[3 * row] * b[col] +
                                 a[3 * row + 1] * b[3 + col] +
                                 a[3 * row + 2] * b[6 + col];
        }
    }
}

template <typename T>
__device__ __forceinline__ void Mat3VecMul(const T *matrix, const T *vector,
                                           T *out) {
    for (int row = 0; row < 3; ++row) {
        out[row] = matrix[3 * row] * vector[0] +
                   matrix[3 * row + 1] * vector[1] +
                   matrix[3 * row + 2] * vector[2];
    }
}

template <typename T>
__device__ __forceinline__ void Skew(const T *vector, T *out) {
    out[0] = T(0);
    out[1] = -vector[2];
    out[2] = vector[1];
    out[3] = vector[2];
    out[4] = T(0);
    out[5] = -vector[0];
    out[6] = -vector[1];
    out[7] = vector[0];
    out[8] = T(0);
}

template <typename T>
__device__ __forceinline__ void Cross(const T *a, const T *b, T *out) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

template <typename T>
__device__ __forceinline__ void ExpSO3Array(const T *w, T *rotation) {
    const T d2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    const T d = sqrt(d2);
    T W[9];
    T W2[9];
    Skew(w, W);
    Mat3Mul(W, W, W2);

    const T a = d < T(1e-5) ? T(1) : sin(d) / d;
    const T b = d < T(1e-5) ? T(0.5) : (T(1) - cos(d)) / d2;
    for (int i = 0; i < 9; ++i) {
        rotation[i] = a * W[i] + b * W2[i];
    }
    rotation[0] += T(1);
    rotation[4] += T(1);
    rotation[8] += T(1);
}

template <typename T>
__device__ __forceinline__ void LoadMat3(const double *src, T *dst) {
    for (int i = 0; i < 9; ++i) {
        dst[i] = static_cast<T>(src[i]);
    }
}

template <typename T>
__device__ __forceinline__ void LoadVec3(const double *src, T *dst) {
    for (int i = 0; i < 3; ++i) {
        dst[i] = static_cast<T>(src[i]);
    }
}

template <typename T>
__device__ __forceinline__ void
CameraMatrices(const DevicePoseVertex &pose, const double *Rcb,
               const double *tcb, T *Rcw, T *tcw) {
    T Rcb_t[9];
    T Rbw_t[9];
    T twb_t[3];
    LoadMat3(Rcb, Rcb_t);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            Rbw_t[3 * row + col] = static_cast<T>(pose.Rwb[3 * col + row]);
        }
    }
    LoadVec3(pose.twb, twb_t);
    Mat3Mul(Rcb_t, Rbw_t, Rcw);
    Mat3VecMul(Rcw, twb_t, tcw);
    for (int i = 0; i < 3; ++i) {
        tcw[i] = static_cast<T>(tcb[i]) - tcw[i];
    }
}

template <typename T>
__device__ __forceinline__ void
CameraPointT(const DevicePoseVertex &pose, const PointFactorData &data,
             T *point_c, T *Rcw = nullptr, T *tcw = nullptr) {
    T local_Rcw[9];
    T local_tcw[3];
    T point_w[3];
    CameraMatrices(pose, data.Rcb, data.tcb, local_Rcw, local_tcw);
    LoadVec3(data.point_w, point_w);
    Mat3VecMul(local_Rcw, point_w, point_c);
    for (int i = 0; i < 3; ++i) {
        point_c[i] += local_tcw[i];
    }
    if (Rcw) {
        for (int i = 0; i < 9; ++i) {
            Rcw[i] = local_Rcw[i];
        }
    }
    if (tcw) {
        for (int i = 0; i < 3; ++i) {
            tcw[i] = local_tcw[i];
        }
    }
}

template <typename T>
__device__ __forceinline__ void ProjectPointT(const DevicePinholeCamera &camera,
                                              const T *point_c, T *projection) {
    projection[0] = T(camera.fx) * point_c[0] / point_c[2] + T(camera.cx);
    projection[1] = T(camera.fy) * point_c[1] / point_c[2] + T(camera.cy);
}

template <typename T>
__device__ __forceinline__ void ProjectLineT(const DevicePinholeCamera &camera,
                                             const T *normal_c, T *line_2d) {
    for (int row = 0; row < 3; ++row) {
        line_2d[row] = Mat3At<T>(camera.toPL, row, 0) * normal_c[0] +
                       Mat3At<T>(camera.toPL, row, 1) * normal_c[1] +
                       Mat3At<T>(camera.toPL, row, 2) * normal_c[2];
    }
}

template <typename T>
__device__ __forceinline__ T Line2DErrorT(const T *line_2d, const T *point_2d) {
    const T scale = sqrt(line_2d[0] * line_2d[0] + line_2d[1] * line_2d[1]);
    return (line_2d[0] * point_2d[0] + line_2d[1] * point_2d[1] + line_2d[2]) /
           scale;
}

template <typename T, typename LineData>
__device__ __forceinline__ void
LineResidualWhiteningT(const T *line_2d, const LineData &data,
                       T *whitening) {
    ComputeLineResidualWhitening(
        line_2d, data.endpoint_covariance_upper,
        data.has_endpoint_covariance != 0, T(data.information), whitening);
}

template <typename T, typename LineData>
__device__ __forceinline__ void WhitenLineResidualT(const T *line_2d,
                                                    const LineData &data,
                                                    T *residual) {
    T whitening[4];
    LineResidualWhiteningT(line_2d, data, whitening);
    const T r0 = residual[0];
    const T r1 = residual[1];
    residual[0] = whitening[0] * r0 + whitening[1] * r1;
    residual[1] = whitening[2] * r0 + whitening[3] * r1;
}

template <int Columns, typename T, typename LineData>
__device__ __forceinline__ void WhitenLineJacobianT(const T *line_2d,
                                                    const LineData &data,
                                                    T *jacobian) {
    T whitening[4];
    LineResidualWhiteningT(line_2d, data, whitening);
    for (int col = 0; col < Columns; ++col) {
        const T j0 = jacobian[2 * col];
        const T j1 = jacobian[2 * col + 1];
        jacobian[2 * col] = whitening[0] * j0 + whitening[1] * j1;
        jacobian[2 * col + 1] = whitening[2] * j0 + whitening[3] * j1;
    }
}

template <typename T, typename LineData>
__device__ __forceinline__ T LineMahalanobisSquaredT(
    const T *line_2d, const LineData &data, T residual0, T residual1) {
    T residual[2] = {residual0, residual1};
    WhitenLineResidualT(line_2d, data, residual);
    return residual[0] * residual[0] + residual[1] * residual[1];
}

template <typename T>
__device__ __forceinline__ void
CameraLineNormalT(const DevicePoseVertex &pose, const LineFactorData &data,
                  T *normal_c, T *Rcw = nullptr, T *tcw = nullptr,
                  T *direction_c_out = nullptr) {
    T local_Rcw[9];
    T local_tcw[3];
    T moment[3];
    T direction[3];
    T direction_c[3];
    T rotated_moment[3];
    T cross_term[3];
    CameraMatrices(pose, data.Rcb, data.tcb, local_Rcw, local_tcw);
    LoadVec3(data.line_w, moment);
    LoadVec3(data.line_w + 3, direction);
    Mat3VecMul(local_Rcw, direction, direction_c);
    Mat3VecMul(local_Rcw, moment, rotated_moment);
    Cross(local_tcw, direction_c, cross_term);
    for (int i = 0; i < 3; ++i) {
        normal_c[i] = rotated_moment[i] + cross_term[i];
    }
    if (Rcw) {
        for (int i = 0; i < 9; ++i) {
            Rcw[i] = local_Rcw[i];
        }
    }
    if (tcw) {
        for (int i = 0; i < 3; ++i) {
            tcw[i] = local_tcw[i];
        }
    }
    if (direction_c_out) {
        for (int i = 0; i < 3; ++i) {
            direction_c_out[i] = direction_c[i];
        }
    }
}

struct PoseVertexTraits {
    static constexpr size_t dimension = 6;
    using Vertex = DevicePoseVertex;
    using State = DevicePoseVertex;

    template <typename T>
    __device__ static void parameters(const Vertex &vertex, T *parameters) {
        parameters[0] = static_cast<T>(0);
        parameters[1] = static_cast<T>(0);
        parameters[2] = static_cast<T>(0);
        parameters[3] = static_cast<T>(vertex.twb[0]);
        parameters[4] = static_cast<T>(vertex.twb[1]);
        parameters[5] = static_cast<T>(vertex.twb[2]);
    }

    __device__ static void update(Vertex &vertex, const double *delta) {
        double update_rotation[9];
        double new_rotation[9];
        ExpSO3Array(delta, update_rotation);
        Mat3Mul(vertex.Rwb, update_rotation, new_rotation);
        for (int i = 0; i < 9; ++i) {
            vertex.Rwb[i] = new_rotation[i];
        }
        vertex.twb[0] += delta[3];
        vertex.twb[1] += delta[4];
        vertex.twb[2] += delta[5];
    }

    __device__ static State get_state(const Vertex &vertex) { return vertex; }

    __device__ static void set_state(Vertex &vertex, const State &state) {
        vertex = state;
    }
};

using PoseVertexDescriptor =
    graphite::VertexDescriptor<double, double, PoseVertexTraits>;

template <typename T>
__device__ __forceinline__ Eigen::Matrix<T, 3, 3>
DeviceMatrix3(const double *values) {
    Eigen::Matrix<T, 3, 3> result;
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            result(row, col) = static_cast<T>(values[3 * row + col]);
    return result;
}

template <typename T>
__device__ __forceinline__ Eigen::Matrix<T, 3, 1>
DeviceVector3(const double *values) {
    return Eigen::Matrix<T, 3, 1>(T(values[0]), T(values[1]), T(values[2]));
}

template <typename T>
__device__ __forceinline__ Eigen::Matrix<T, 3, 1>
MeasurementRotation(const EssentialGraphEdgeData &data) {
    Eigen::Matrix<T, 3, 1> measurement;
    for (int i = 0; i < 3; ++i) {
        measurement[i] = static_cast<T>(data.measurement_r[i]);
    }
    return measurement;
}

template <typename T>
__device__ __forceinline__ Eigen::Matrix<T, 3, 1>
MeasurementTranslation(const EssentialGraphEdgeData &data) {
    Eigen::Matrix<T, 3, 1> measurement;
    for (int i = 0; i < 3; ++i) {
        measurement[i] = static_cast<T>(data.measurement_t[i]);
    }
    return measurement;
}

using PoseGraphVertexTraits = PoseVertexTraits;
using PoseGraphVertexDescriptor = PoseVertexDescriptor;

struct EssentialGraphTraits {
    static constexpr size_t dimension = 6;
    using VertexDescriptors =
        std::tuple<PoseGraphVertexDescriptor, PoseGraphVertexDescriptor>;
    using Observation = graphite::Empty;
    using Data = EssentialGraphEdgeData;
    using Loss = graphite::DefaultLoss<double, dimension>;
    using Differentiation = graphite::DifferentiationMode::Manual;

    template <typename T>
    __device__ static void error(const DevicePoseGraphVertex &pose_i,
                                 const DevicePoseGraphVertex &pose_j,
                                 const Data &data, T *residual) {
        const auto Ri = DeviceMatrix3<T>(pose_i.Rwb);
        const auto ti = DeviceVector3<T>(pose_i.twb);
        const auto Rj = DeviceMatrix3<T>(pose_j.Rwb);
        const auto tj = DeviceVector3<T>(pose_j.twb);
        const Eigen::Matrix<T, 3, 1> measurement_r =
            MeasurementRotation<T>(data);
        const Eigen::Matrix<T, 3, 1> measurement_t =
            MeasurementTranslation<T>(data);
        Eigen::Matrix<T, 6, 1> residual_matrix;
        generated::essential_graph_residual(Ri, ti, Rj, tj, measurement_r,
                                            measurement_t, residual_matrix);
        for (int row = 0; row < 6; ++row) {
            residual[row] = residual_matrix[row];
        }
    }

    template <typename T, size_t I>
    __device__ static void jacobian(const DevicePoseGraphVertex &pose_i,
                                    const DevicePoseGraphVertex &pose_j,
                                    const Data &data, T *jacobian) {
        static_assert(I < 2);
        const auto Ri = DeviceMatrix3<T>(pose_i.Rwb);
        const auto ti = DeviceVector3<T>(pose_i.twb);
        const auto Rj = DeviceMatrix3<T>(pose_j.Rwb);
        const auto tj = DeviceVector3<T>(pose_j.twb);
        const Eigen::Matrix<T, 3, 1> measurement_r =
            MeasurementRotation<T>(data);
        const Eigen::Matrix<T, 3, 1> measurement_t =
            MeasurementTranslation<T>(data);
        Eigen::Matrix<T, 6, 6> selected;
        if constexpr (I == 0) {
            generated::essential_graph_jacobian_i(
                Ri, ti, Rj, tj, measurement_r, measurement_t, selected);
        } else {
            generated::essential_graph_jacobian_j(
                Ri, ti, Rj, tj, measurement_r, measurement_t, selected);
        }
        for (int col = 0; col < 6; ++col) {
            for (int row = 0; row < 6; ++row) {
                jacobian[col * 6 + row] = selected(row, col);
            }
        }
    }
};

using EssentialGraphFactor =
    graphite::FactorDescriptor<double, double, EssentialGraphTraits>;

template <typename T>
__device__ __forceinline__ Eigen::Matrix<T, 3, 1>
BAPointParameters(const DeviceBAPointVertex &vertex) {
    return Eigen::Matrix<T, 3, 1>(static_cast<T>(vertex.position[0]),
                                  static_cast<T>(vertex.position[1]),
                                  static_cast<T>(vertex.position[2]));
}

template <typename T>
__device__ __forceinline__ Eigen::Matrix<T, 3, 3>
BALineBaseU(const DeviceBALineVertex &vertex) {
    Eigen::Matrix<T, 3, 3> U;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            U(row, col) = static_cast<T>(vertex.U[3 * row + col]);
        }
    }
    return U;
}

template <typename T>
__device__ __forceinline__ Eigen::Matrix<T, 2, 2>
BALineBaseW(const DeviceBALineVertex &vertex) {
    Eigen::Matrix<T, 2, 2> W;
    W(0, 0) = static_cast<T>(vertex.W[0]);
    W(0, 1) = static_cast<T>(vertex.W[1]);
    W(1, 0) = static_cast<T>(vertex.W[2]);
    W(1, 1) = static_cast<T>(vertex.W[3]);
    return W;
}

template <typename T>
__device__ __forceinline__ Eigen::Matrix<T, 4, 1>
BALineDelta(const DeviceBALineVertex &vertex) {
    return Eigen::Matrix<T, 4, 1>(
        static_cast<T>(vertex.delta[0]), static_cast<T>(vertex.delta[1]),
        static_cast<T>(vertex.delta[2]), static_cast<T>(vertex.delta[3]));
}

template <typename T, typename ObservationData>
__device__ __forceinline__ Eigen::Matrix<T, 3, 1>
BARotationVector(const ObservationData &data) {
    return Eigen::Matrix<T, 3, 1>(static_cast<T>(data.rcb[0]),
                                  static_cast<T>(data.rcb[1]),
                                  static_cast<T>(data.rcb[2]));
}

template <typename T, typename ObservationData>
__device__ __forceinline__ Eigen::Matrix<T, 3, 1>
BATranslation(const ObservationData &data) {
    return Eigen::Matrix<T, 3, 1>(static_cast<T>(data.tcb[0]),
                                  static_cast<T>(data.tcb[1]),
                                  static_cast<T>(data.tcb[2]));
}

template <typename T>
__device__ __forceinline__ Eigen::Matrix<T, 2, 1>
BAMeasurement2(const DeviceVec2 &measurement) {
    return Eigen::Matrix<T, 2, 1>(static_cast<T>(measurement.v[0]),
                                  static_cast<T>(measurement.v[1]));
}

template <typename T>
__device__ __forceinline__ Eigen::Matrix<T, 4, 1>
BAMeasurement4(const DeviceVec4 &measurement) {
    return Eigen::Matrix<T, 4, 1>(
        static_cast<T>(measurement.v[0]), static_cast<T>(measurement.v[1]),
        static_cast<T>(measurement.v[2]), static_cast<T>(measurement.v[3]));
}

__device__ __forceinline__ void RefreshBALineWorld(DeviceBALineVertex &line) {
    double update_rotation[9];
    double current_U[9];
    ExpSO3Array(line.delta, update_rotation);
    Mat3Mul(line.U, update_rotation, current_U);

    const double c = cos(line.delta[3]);
    const double s = sin(line.delta[3]);
    const double current_W00 = line.W[0] * c + line.W[1] * s;
    const double current_W10 = line.W[2] * c + line.W[3] * s;
    for (int row = 0; row < 3; ++row) {
        line.line_w[row] = current_U[3 * row] * current_W00;
        line.line_w[row + 3] = current_U[3 * row + 1] * current_W10;
    }
}

struct BAPointVertexTraits {
    static constexpr size_t dimension = 3;
    using Vertex = DeviceBAPointVertex;
    using State = DeviceBAPointVertex;

    template <typename T>
    __device__ static void parameters(const Vertex &vertex, T *parameters) {
        for (int i = 0; i < 3; ++i) {
            parameters[i] = static_cast<T>(vertex.position[i]);
        }
    }

    __device__ static void update(Vertex &vertex, const double *delta) {
        for (int i = 0; i < 3; ++i) {
            vertex.position[i] += delta[i];
        }
    }

    __device__ static State get_state(const Vertex &vertex) { return vertex; }
    __device__ static void set_state(Vertex &vertex, const State &state) {
        vertex = state;
    }
};

struct BALineVertexTraits {
    static constexpr size_t dimension = 4;
    using Vertex = DeviceBALineVertex;
    using State = DeviceBALineVertex;

    template <typename T>
    __device__ static void parameters(const Vertex &vertex, T *parameters) {
        for (int i = 0; i < 4; ++i) {
            parameters[i] = static_cast<T>(vertex.delta[i]);
        }
    }

    __device__ static void update(Vertex &vertex, const double *delta) {
        for (int i = 0; i < 4; ++i) {
            vertex.delta[i] += delta[i];
        }
        RefreshBALineWorld(vertex);
    }

    __device__ static State get_state(const Vertex &vertex) { return vertex; }
    __device__ static void set_state(Vertex &vertex, const State &state) {
        vertex = state;
    }
};

using BAPointVertexDescriptor =
    graphite::VertexDescriptor<double, double, BAPointVertexTraits>;
using BALineVertexDescriptor =
    graphite::VertexDescriptor<double, double, BALineVertexTraits>;

struct LocalBAPointTraits {
    static constexpr size_t dimension = 2;
    using VertexDescriptors =
        std::tuple<BAPointVertexDescriptor, PoseGraphVertexDescriptor>;
    using Observation = DeviceVec2;
    using Data = LocalBAObservationData;
    using Loss = SwitchableHuberLoss<double, dimension>;
    using Differentiation = graphite::DifferentiationMode::Manual;

    template <typename T>
    __device__ static void
    error(const DeviceBAPointVertex &point, const DevicePoseGraphVertex &pose,
          const Observation &measurement, const Data &data, T *residual) {
        const auto Rwb = DeviceMatrix3<T>(pose.Rwb);
        const auto twb = DeviceVector3<T>(pose.twb);
        const Eigen::Matrix<T, 3, 1> point_params = BAPointParameters<T>(point);
        Eigen::Matrix<T, 2, 1> residual_matrix;
        generated::local_ba_point_residual(
            Rwb, twb, point_params, BARotationVector<T>(data),
            BATranslation<T>(data), static_cast<T>(data.camera.fx),
            static_cast<T>(data.camera.fy), static_cast<T>(data.camera.cx),
            static_cast<T>(data.camera.cy), BAMeasurement2<T>(measurement),
            residual_matrix);
        residual[0] = residual_matrix[0];
        residual[1] = residual_matrix[1];
    }

    template <typename T, size_t I>
    __device__ static void jacobian(const DeviceBAPointVertex &point,
                                    const DevicePoseGraphVertex &pose,
                                    const Observation &measurement,
                                    const Data &data, T *jacobian) {
        static_assert(I < 2);
        const auto Rwb = DeviceMatrix3<T>(pose.Rwb);
        const auto twb = DeviceVector3<T>(pose.twb);
        const Eigen::Matrix<T, 3, 1> point_params = BAPointParameters<T>(point);
        if constexpr (I == 0) {
            Eigen::Matrix<T, 2, 3> J_point;
            generated::local_ba_point_jacobian_point(
                Rwb, twb, point_params, BARotationVector<T>(data),
                BATranslation<T>(data), static_cast<T>(data.camera.fx),
                static_cast<T>(data.camera.fy), static_cast<T>(data.camera.cx),
                static_cast<T>(data.camera.cy), BAMeasurement2<T>(measurement),
                J_point);
            for (int col = 0; col < 3; ++col) {
                for (int row = 0; row < 2; ++row) {
                    jacobian[col * 2 + row] = J_point(row, col);
                }
            }
        } else {
            Eigen::Matrix<T, 2, 6> J_pose;
            generated::local_ba_point_jacobian_pose(
                Rwb, twb, point_params, BARotationVector<T>(data),
                BATranslation<T>(data), static_cast<T>(data.camera.fx),
                static_cast<T>(data.camera.fy), static_cast<T>(data.camera.cx),
                static_cast<T>(data.camera.cy), BAMeasurement2<T>(measurement),
                J_pose);
            for (int col = 0; col < 6; ++col) {
                for (int row = 0; row < 2; ++row) {
                    jacobian[col * 2 + row] = J_pose(row, col);
                }
            }
        }
    }
};

struct LocalBALineTraits {
    static constexpr size_t dimension = 2;
    using VertexDescriptors =
        std::tuple<BALineVertexDescriptor, PoseGraphVertexDescriptor>;
    using Observation = DeviceVec4;
    using Data = LocalBALineObservationData;
    using Loss = SwitchableHuberLoss<double, dimension>;
    using Differentiation = graphite::DifferentiationMode::Manual;

    template <typename T>
    __device__ static void
    error(const DeviceBALineVertex &line, const DevicePoseGraphVertex &pose,
          const Observation &measurement, const Data &data, T *residual) {
        const auto Rwb = DeviceMatrix3<T>(pose.Rwb);
        const auto twb = DeviceVector3<T>(pose.twb);
        Eigen::Matrix<T, 2, 1> residual_matrix;
        Eigen::Matrix<T, 3, 1> line_2d;
        generated::local_ba_line_residual(
            Rwb, twb, BALineBaseU<T>(line), BALineBaseW<T>(line),
            BALineDelta<T>(line), BARotationVector<T>(data),
            BATranslation<T>(data), static_cast<T>(data.camera.fx),
            static_cast<T>(data.camera.fy), static_cast<T>(data.camera.cx),
            static_cast<T>(data.camera.cy), BAMeasurement4<T>(measurement),
            residual_matrix, line_2d);
        residual[0] = residual_matrix[0];
        residual[1] = residual_matrix[1];
        WhitenLineResidualT(line_2d.data(), data, residual);
    }

    template <typename T, size_t I>
    __device__ static void
    jacobian(const DeviceBALineVertex &line, const DevicePoseGraphVertex &pose,
             const Observation &measurement, const Data &data, T *jacobian) {
        static_assert(I < 2);
        const auto Rwb = DeviceMatrix3<T>(pose.Rwb);
        const auto twb = DeviceVector3<T>(pose.twb);
        if constexpr (I == 0) {
            Eigen::Matrix<T, 2, 4> J_line;
            Eigen::Matrix<T, 3, 1> line_2d;
            generated::local_ba_line_jacobian_line(
                Rwb, twb, BALineBaseU<T>(line), BALineBaseW<T>(line),
                BALineDelta<T>(line), BARotationVector<T>(data),
                BATranslation<T>(data), static_cast<T>(data.camera.fx),
                static_cast<T>(data.camera.fy), static_cast<T>(data.camera.cx),
                static_cast<T>(data.camera.cy), BAMeasurement4<T>(measurement),
                J_line, line_2d);
            for (int col = 0; col < 4; ++col) {
                for (int row = 0; row < 2; ++row) {
                    jacobian[col * 2 + row] = J_line(row, col);
                }
            }
            WhitenLineJacobianT<4>(line_2d.data(), data, jacobian);
        } else {
            Eigen::Matrix<T, 2, 6> J_pose;
            Eigen::Matrix<T, 3, 1> line_2d;
            generated::local_ba_line_jacobian_pose(
                Rwb, twb, BALineBaseU<T>(line), BALineBaseW<T>(line),
                BALineDelta<T>(line), BARotationVector<T>(data),
                BATranslation<T>(data), static_cast<T>(data.camera.fx),
                static_cast<T>(data.camera.fy), static_cast<T>(data.camera.cx),
                static_cast<T>(data.camera.cy), BAMeasurement4<T>(measurement),
                J_pose, line_2d);
            for (int col = 0; col < 6; ++col) {
                for (int row = 0; row < 2; ++row) {
                    jacobian[col * 2 + row] = J_pose(row, col);
                }
            }
            WhitenLineJacobianT<6>(line_2d.data(), data, jacobian);
        }
    }
};

using LocalBAPointFactor =
    graphite::FactorDescriptor<double, double, LocalBAPointTraits>;
using LocalBALineFactor =
    graphite::FactorDescriptor<double, double, LocalBALineTraits>;

struct BAPointEvalRef {
    size_t pose = 0;
    size_t point = 0;
    LocalBAObservationData data;
    DeviceVec2 measurement;
};

struct BALineEvalRef {
    size_t pose = 0;
    size_t line = 0;
    LocalBALineObservationData data;
    DeviceVec4 measurement;
};

template <typename T, typename ObservationData>
__device__ __forceinline__ void
LocalBACameraMatrices(const DevicePoseGraphVertex &pose,
                      const ObservationData &data, T *Rcw, T *tcw) {
    double Rcb[9];
    ExpSO3Array(data.rcb, Rcb);
    CameraMatrices(pose, Rcb, data.tcb, Rcw, tcw);
}

template <typename T>
__device__ __forceinline__ bool
LinePointFromBackprojectedRayT(const T *line_c, const T *ray_c, T *point_c) {
    const T kMinLineNorm = T(1e-12);
    const T *moment = line_c;
    const T *direction = line_c + 3;
    const T direction_norm2 = direction[0] * direction[0] +
                              direction[1] * direction[1] +
                              direction[2] * direction[2];
    const T ray_norm2 =
        ray_c[0] * ray_c[0] + ray_c[1] * ray_c[1] + ray_c[2] * ray_c[2];
    if (direction_norm2 < kMinLineNorm || ray_norm2 < kMinLineNorm) {
        return false;
    }

    T point_on_line[3];
    Cross(direction, moment, point_on_line);
    for (int i = 0; i < 3; ++i) {
        point_on_line[i] /= direction_norm2;
    }

    const T direction_dot_ray = direction[0] * ray_c[0] +
                                direction[1] * ray_c[1] +
                                direction[2] * ray_c[2];
    const T point_dot_direction = point_on_line[0] * direction[0] +
                                  point_on_line[1] * direction[1] +
                                  point_on_line[2] * direction[2];
    const T point_dot_ray = point_on_line[0] * ray_c[0] +
                            point_on_line[1] * ray_c[1] +
                            point_on_line[2] * ray_c[2];
    const T denom =
        direction_norm2 * ray_norm2 - direction_dot_ray * direction_dot_ray;
    if (fabs(denom) < kMinLineNorm) {
        return false;
    }

    const T line_parameter =
        (direction_dot_ray * point_dot_ray - ray_norm2 * point_dot_direction) /
        denom;
    for (int i = 0; i < 3; ++i) {
        point_c[i] = point_on_line[i] + line_parameter * direction[i];
    }
    return isfinite(point_c[0]) && isfinite(point_c[1]) && isfinite(point_c[2]);
}

template <typename T>
__device__ __forceinline__ bool IsLineDepthPositiveT(
    const T *normal_c, const T *direction_c,
    const DevicePinholeCamera &camera, const DeviceVec4 &measurement) {
    T line_c[6];
    for (int i = 0; i < 3; ++i) {
        line_c[i] = normal_c[i];
        line_c[i + 3] = direction_c[i];
    }
    const T ray0[3] = {
        T((measurement.v[0] - camera.cx) / camera.fx),
        T((measurement.v[1] - camera.cy) / camera.fy), T(1)};
    const T ray1[3] = {
        T((measurement.v[2] - camera.cx) / camera.fx),
        T((measurement.v[3] - camera.cy) / camera.fy), T(1)};
    T point0[3];
    T point1[3];
    return LinePointFromBackprojectedRayT(line_c, ray0, point0) &&
           LinePointFromBackprojectedRayT(line_c, ray1, point1) &&
           point0[2] > T(1e-9) && point1[2] > T(1e-9);
}

__global__ void EvaluateLocalBAPointOutliersKernel(
    const DevicePoseGraphVertex *poses, const DeviceBAPointVertex *points,
    const BAPointEvalRef *refs, uint8_t *outliers, int *num_bad,
    double chi2_threshold, size_t count) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }

    const BAPointEvalRef &ref = refs[idx];
    double Rcw[9];
    double tcw[3];
    LocalBACameraMatrices(poses[ref.pose], ref.data, Rcw, tcw);
    double point_c[3];
    Mat3VecMul(Rcw, points[ref.point].position, point_c);
    for (int i = 0; i < 3; ++i) {
        point_c[i] += tcw[i];
    }
    double projection[2];
    ProjectPointT(ref.data.camera, point_c, projection);
    const double rx = ref.measurement.v[0] - projection[0];
    const double ry = ref.measurement.v[1] - projection[1];
    const bool depth_bad = !(point_c[2] > 0.0) || !isfinite(point_c[2]);
    const double chi2 = ref.data.information * (rx * rx + ry * ry);
    const bool chi2_bad = !isfinite(chi2) || chi2 > chi2_threshold;
    const bool is_outlier = depth_bad || chi2_bad;
    outliers[idx] = is_outlier ? 1 : 0;
    if (is_outlier) {
        atomicAdd(num_bad, 1);
    }
}

__global__ void EvaluateLocalBALineOutliersKernel(
    const DevicePoseGraphVertex *poses, const DeviceBALineVertex *lines,
    const BALineEvalRef *refs, uint8_t *outliers, int *num_bad,
    double chi2_threshold, size_t count) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }

    const BALineEvalRef &ref = refs[idx];
    double Rcw[9];
    double tcw[3];
    LocalBACameraMatrices(poses[ref.pose], ref.data, Rcw, tcw);
    const DeviceBALineVertex &line = lines[ref.line];
    double direction_c[3];
    double rotated_moment[3];
    double cross_term[3];
    double normal_c[3];
    Mat3VecMul(Rcw, line.line_w + 3, direction_c);
    Mat3VecMul(Rcw, line.line_w, rotated_moment);
    Cross(tcw, direction_c, cross_term);
    for (int i = 0; i < 3; ++i) {
        normal_c[i] = rotated_moment[i] + cross_term[i];
    }

    double line_2d[3];
    ProjectLineT(ref.data.camera, normal_c, line_2d);
    const double start[2] = {ref.measurement.v[0], ref.measurement.v[1]};
    const double end[2] = {ref.measurement.v[2], ref.measurement.v[3]};
    const double r0 = Line2DErrorT(line_2d, start);
    const double r1 = Line2DErrorT(line_2d, end);

    const bool depth_ok = IsLineDepthPositiveT(
        normal_c, direction_c, ref.data.camera, ref.measurement);
    const double chi2 = LineMahalanobisSquaredT(line_2d, ref.data, r0, r1);
    const bool chi2_bad = !isfinite(chi2) || chi2 > chi2_threshold;
    const bool is_outlier = !depth_ok || chi2_bad;
    outliers[idx] = is_outlier ? 1 : 0;
    if (is_outlier) {
        atomicAdd(num_bad, 1);
    }
}

struct PoseOnlyPointTraits {
    static constexpr size_t dimension = 2;
    using VertexDescriptors = std::tuple<PoseVertexDescriptor>;
    using Observation = DeviceVec2;
    using Data = PointFactorData;
    using Loss = SwitchableHuberLoss<double, dimension>;
    using Differentiation = graphite::DifferentiationMode::Manual;

    template <typename T>
    __device__ static void error(const DevicePoseVertex &pose,
                                 const Observation &measurement,
                                 const Data &data, T *residual) {
        T point_c[3];
        T projection[2];
        CameraPointT(pose, data, point_c);
        ProjectPointT(data.camera, point_c, projection);
        residual[0] = T(measurement.v[0]) - projection[0];
        residual[1] = T(measurement.v[1]) - projection[1];
    }

    template <typename T, size_t I>
    __device__ static void jacobian(const DevicePoseVertex &pose,
                                    const Observation &, const Data &data,
                                    T *jacobian) {
        static_assert(I == 0);
        Eigen::Matrix<T, 2, 6> J;
        generated::pose_only_point_jacobian_pose(
            DeviceMatrix3<T>(pose.Rwb), DeviceVector3<T>(pose.twb),
            DeviceVector3<T>(data.point_w), DeviceMatrix3<T>(data.Rcb),
            DeviceVector3<T>(data.tcb), T(data.camera.fx), T(data.camera.fy), J);
        for (int col = 0; col < 6; ++col)
            for (int row = 0; row < 2; ++row)
                jacobian[2 * col + row] = J(row, col);
    }
};

struct PoseOnlyLineTraits {
    static constexpr size_t dimension = 2;
    using VertexDescriptors = std::tuple<PoseVertexDescriptor>;
    using Observation = DeviceVec4;
    using Data = LineFactorData;
    using Loss = SwitchableHuberLoss<double, dimension>;
    using Differentiation = graphite::DifferentiationMode::Manual;

    template <typename T>
    __device__ static void error(const DevicePoseVertex &pose,
                                 const Observation &measurement,
                                 const Data &data, T *residual) {
        T normal_c[3];
        T line_2d[3];
        T start[2] = {T(measurement.v[0]), T(measurement.v[1])};
        T end[2] = {T(measurement.v[2]), T(measurement.v[3])};
        CameraLineNormalT(pose, data, normal_c);
        ProjectLineT(data.camera, normal_c, line_2d);
        residual[0] = Line2DErrorT(line_2d, start);
        residual[1] = Line2DErrorT(line_2d, end);
        WhitenLineResidualT(line_2d, data, residual);
    }

    template <typename T, size_t I>
    __device__ static void jacobian(const DevicePoseVertex &pose,
                                    const Observation &measurement,
                                    const Data &data, T *jacobian) {
        static_assert(I == 0);
        Eigen::Matrix<T, 6, 1> line_w;
        for (int i = 0; i < 6; ++i) line_w[i] = T(data.line_w[i]);
        Eigen::Matrix<T, 2, 6> J;
        Eigen::Matrix<T, 3, 1> line_2d;
        generated::pose_only_line_jacobian_pose(
            DeviceMatrix3<T>(pose.Rwb), DeviceVector3<T>(pose.twb), line_w,
            DeviceMatrix3<T>(data.Rcb), DeviceVector3<T>(data.tcb),
            DeviceMatrix3<T>(data.camera.toPL), BAMeasurement4<T>(measurement),
            J, line_2d);
        for (int col = 0; col < 6; ++col)
            for (int row = 0; row < 2; ++row)
                jacobian[2 * col + row] = J(row, col);
        WhitenLineJacobianT<6>(line_2d.data(), data, jacobian);
    }
};

using PointFactor =
    graphite::FactorDescriptor<double, double, PoseOnlyPointTraits>;
using LineFactor =
    graphite::FactorDescriptor<double, double, PoseOnlyLineTraits>;

template <typename Factor>
void SetFactorState(Factor &factor, const std::vector<size_t> &factor_ids,
                    const std::vector<uint8_t> &outliers, bool robust_enabled,
                    double huber_delta) {
    const typename Factor::LossType loss(huber_delta, robust_enabled);
    for (size_t i = 0; i < factor_ids.size(); ++i) {
        factor.set_active(factor_ids[i], outliers[i] ? 1 : 0);
        factor.set_loss(factor_ids[i], loss);
    }
}

template <typename Factor>
void SetFactorStateLocal(Factor &factor, const std::vector<uint8_t> &outliers,
                         bool robust_enabled, double huber_delta) {
    const typename Factor::LossType loss(huber_delta, robust_enabled);
    const size_t count = std::min(factor.internal_count(), outliers.size());
    for (size_t i = 0; i < count; ++i) {
        factor.set_active_local(i, outliers[i] ? 1 : 0);
        factor.set_loss_local(i, loss);
    }
}

int ActiveFactorCount(const std::vector<uint8_t> &point_outliers,
                      const std::vector<uint8_t> &line_outliers) {
    int active = 0;
    for (uint8_t outlier : point_outliers) {
        active += outlier == 0 ? 1 : 0;
    }
    for (uint8_t outlier : line_outliers) {
        active += outlier == 0 ? 1 : 0;
    }
    return active;
}

__global__ void
EvaluatePointOutliersKernel(DevicePoseVertex pose, const PointFactorData *data,
                            const DeviceVec2 *measurements, uint8_t *outliers,
                            int *num_bad, double chi2_threshold, size_t count) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }

    double point_c[3];
    double projection[2];
    CameraPointT(pose, data[idx], point_c);
    ProjectPointT(data[idx].camera, point_c, projection);
    const double rx = measurements[idx].v[0] - projection[0];
    const double ry = measurements[idx].v[1] - projection[1];
    const double chi2 = data[idx].information * (rx * rx + ry * ry);
    const bool is_outlier = !isfinite(chi2) || chi2 > chi2_threshold ||
                            !(point_c[2] > 0.0) || !isfinite(point_c[2]);
    outliers[idx] = is_outlier ? 1 : 0;
    if (is_outlier) {
        atomicAdd(num_bad, 1);
    }
}

__global__ void
EvaluateLineOutliersKernel(DevicePoseVertex pose, const LineFactorData *data,
                           const DeviceVec4 *measurements, uint8_t *outliers,
                           int *num_bad, double chi2_threshold, size_t count) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }

    double normal_c[3];
    double direction_c[3];
    double line_2d[3];
    const double start[2] = {measurements[idx].v[0], measurements[idx].v[1]};
    const double end[2] = {measurements[idx].v[2], measurements[idx].v[3]};
    CameraLineNormalT<double>(pose, data[idx], normal_c, nullptr, nullptr,
                             direction_c);
    ProjectLineT(data[idx].camera, normal_c, line_2d);
    const double r0 = Line2DErrorT(line_2d, start);
    const double r1 = Line2DErrorT(line_2d, end);
    const double chi2 = LineMahalanobisSquaredT(line_2d, data[idx], r0, r1);
    const bool is_outlier = !isfinite(chi2) || chi2 > chi2_threshold ||
                            !IsLineDepthPositiveT(normal_c, direction_c,
                                                  data[idx].camera,
                                                  measurements[idx]);
    outliers[idx] = is_outlier ? 1 : 0;
    if (is_outlier) {
        atomicAdd(num_bad, 1);
    }
}

template <typename Factor> size_t FactorCount(const Factor *factor) {
    return factor ? factor->internal_count() : 0;
}

int EvaluateOutliersGpu(const DevicePoseVertex &pose,
                        const PointFactor *point_factor,
                        const LineFactor *line_factor, double chi2_threshold,
                        thrust::device_vector<uint8_t> &point_outliers_device,
                        thrust::device_vector<uint8_t> &line_outliers_device,
                        std::vector<uint8_t> &point_outliers_host,
                        std::vector<uint8_t> &line_outliers_host) {
    thrust::device_vector<int> num_bad_device(1);
    thrust::fill(num_bad_device.begin(), num_bad_device.end(), 0);

    constexpr size_t threads_per_block = 256;
    const size_t point_count = FactorCount(point_factor);
    if (point_count > 0) {
        const size_t blocks =
            (point_count + threads_per_block - 1) / threads_per_block;
        EvaluatePointOutliersKernel<<<blocks, threads_per_block>>>(
            pose, point_factor->data.data().get(),
            point_factor->device_obs.data().get(),
            point_outliers_device.data().get(), num_bad_device.data().get(),
            chi2_threshold, point_count);
    }

    const size_t line_count = FactorCount(line_factor);
    if (line_count > 0) {
        const size_t blocks =
            (line_count + threads_per_block - 1) / threads_per_block;
        EvaluateLineOutliersKernel<<<blocks, threads_per_block>>>(
            pose, line_factor->data.data().get(),
            line_factor->device_obs.data().get(),
            line_outliers_device.data().get(), num_bad_device.data().get(),
            chi2_threshold, line_count);
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        return -1;
    }
    if (point_count > 0) {
        thrust::copy(point_outliers_device.begin(), point_outliers_device.end(),
                     point_outliers_host.begin());
    }
    if (line_count > 0) {
        thrust::copy(line_outliers_device.begin(), line_outliers_device.end(),
                     line_outliers_host.begin());
    }
    int num_bad = 0;
    if (cudaMemcpy(&num_bad, num_bad_device.data().get(), sizeof(int),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return -1;
    }
    return num_bad;
}

void PrintCudaStatus(const char *label) {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        std::cerr << label << " CUDA error: " << cudaGetErrorString(status)
                  << '\n';
    }
}

template <typename Factor>
void PrintFirstResiduals(const char *label, const Factor &factor) {
    thrust::host_vector<double> residuals = factor.residuals;
    const size_t count = std::min<size_t>(residuals.size(), 6);
    std::cerr << label << " residuals";
    for (size_t i = 0; i < count; ++i) {
        std::cerr << ' ' << residuals[i];
    }
    std::cerr << '\n';
}

DevicePinholeCamera ToDeviceCamera(const FlatPinholeCamera &camera) {
    DevicePinholeCamera device;
    device.fx = camera.fx;
    device.fy = camera.fy;
    device.cx = camera.cx;
    device.cy = camera.cy;
    std::copy(camera.toPL.begin(), camera.toPL.end(), device.toPL);
    return device;
}

DevicePoseVertex ToDevicePoseVertex(const FlatPoseVertex &pose) {
    DevicePoseVertex device;
    std::copy(pose.Rwb.begin(), pose.Rwb.end(), device.Rwb);
    std::copy(pose.twb.begin(), pose.twb.end(), device.twb);
    return device;
}

FlatPoseVertex FromDevicePoseVertexFlat(const DevicePoseVertex &device) {
    FlatPoseVertex pose;
    std::copy(device.Rwb, device.Rwb + 9, pose.Rwb.begin());
    std::copy(device.twb, device.twb + 3, pose.twb.begin());
    return pose;
}

DeviceVec2 ToDeviceVec2(const std::array<double, 2> &vector) {
    return DeviceVec2{{vector[0], vector[1]}};
}

DeviceVec4 ToDeviceVec4(const std::array<double, 4> &vector) {
    return DeviceVec4{{vector[0], vector[1], vector[2], vector[3]}};
}

PointFactorData ToPointFactorData(const FlatPoseOnlyPointInput &input) {
    PointFactorData data;
    data.camera = ToDeviceCamera(input.camera);
    std::copy(input.Rcb.begin(), input.Rcb.end(), data.Rcb);
    std::copy(input.tcb.begin(), input.tcb.end(), data.tcb);
    std::copy(input.point_w.begin(), input.point_w.end(), data.point_w);
    data.information = input.information;
    return data;
}

LineFactorData ToLineFactorData(const FlatPoseOnlyLineInput &input) {
    LineFactorData data;
    data.camera = ToDeviceCamera(input.camera);
    std::copy(input.Rcb.begin(), input.Rcb.end(), data.Rcb);
    std::copy(input.tcb.begin(), input.tcb.end(), data.tcb);
    std::copy(input.line_w.begin(), input.line_w.end(), data.line_w);
    std::copy(input.endpoint_covariance_upper.begin(),
              input.endpoint_covariance_upper.end(),
              data.endpoint_covariance_upper);
    data.has_endpoint_covariance =
        input.has_endpoint_covariance &&
                IsValidLineEndpointCovariance(input.endpoint_covariance_upper)
            ? 1
            : 0;
    data.information = input.information;
    return data;
}

Eigen::Matrix3d ToEigenMatrix3(const std::array<double, 9> &values) {
    Eigen::Matrix3d matrix;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            matrix(row, col) = values[static_cast<size_t>(3 * row + col)];
        }
    }
    return matrix;
}

Eigen::Vector3d RotationVectorFromMatrix(const std::array<double, 9> &values) {
    Eigen::Quaterniond q(ToEigenMatrix3(values));
    q.normalize();
    const Eigen::AngleAxisd angle_axis(q);
    if (!std::isfinite(angle_axis.angle()) || angle_axis.angle() == 0.0) {
        return Eigen::Vector3d::Zero();
    }
    return angle_axis.axis() * angle_axis.angle();
}

DevicePoseGraphVertex ToDevicePoseGraphVertex(const FlatPoseVertex &pose) {
    return ToDevicePoseVertex(pose);
}

FlatPoseVertex
FromDevicePoseGraphVertexFlat(const DevicePoseGraphVertex &device) {
    return FromDevicePoseVertexFlat(device);
}

EssentialGraphEdgeData
ToEssentialGraphEdgeData(const FlatEssentialGraphEdgeInput &input) {
    EssentialGraphEdgeData data;
    const Eigen::Vector3d measurement_r =
        RotationVectorFromMatrix(input.measurement_Tji.rotation);
    for (int i = 0; i < 3; ++i) {
        data.measurement_r[i] = measurement_r[i];
        data.measurement_t[i] =
            input.measurement_Tji.translation[static_cast<size_t>(i)];
    }
    return data;
}

DeviceBAPointVertex ToDeviceBAPointVertex(const FlatLocalBAPointVertex &point) {
    DeviceBAPointVertex device;
    std::copy(point.position.begin(), point.position.end(), device.position);
    return device;
}

FlatLocalBAPointVertex
FromDeviceBAPointVertexFlat(const DeviceBAPointVertex &device) {
    FlatLocalBAPointVertex point;
    std::copy(device.position, device.position + 3, point.position.begin());
    return point;
}

bool ToDeviceBALineVertex(const FlatLocalBALineVertex &line,
                          DeviceBALineVertex &device) {
    Eigen::Matrix<double, 6, 1> line_w;
    for (int i = 0; i < 6; ++i) {
        line_w[i] = line.line_w[static_cast<size_t>(i)];
    }

    const Eigen::Vector3d moment = line_w.head<3>();
    const Eigen::Vector3d direction = line_w.tail<3>();
    const double moment_norm = moment.norm();
    const double direction_norm = direction.norm();
    const double md_norm = std::hypot(moment_norm, direction_norm);
    const Eigen::Vector3d u3 = moment.cross(direction);
    const double u3_norm = u3.norm();

    if (moment_norm < 1e-9 || direction_norm < 1e-9 || md_norm < 1e-9 ||
        u3_norm < 1e-12 || !line_w.allFinite()) {
        return false;
    }

    const Eigen::Vector3d u1 = moment / moment_norm;
    const Eigen::Vector3d u2 = direction / direction_norm;
    const Eigen::Vector3d u3_normalized = u3 / u3_norm;
    for (int row = 0; row < 3; ++row) {
        device.U[3 * row] = u1[row];
        device.U[3 * row + 1] = u2[row];
        device.U[3 * row + 2] = u3_normalized[row];
    }
    device.W[0] = moment_norm / md_norm;
    device.W[1] = -direction_norm / md_norm;
    device.W[2] = direction_norm / md_norm;
    device.W[3] = moment_norm / md_norm;
    for (int i = 0; i < 6; ++i) {
        device.line_w[i] = line_w[i] / md_norm;
    }
    return true;
}

FlatLocalBALineVertex
FromDeviceBALineVertexFlat(const DeviceBALineVertex &device) {
    FlatLocalBALineVertex line;
    std::copy(device.line_w, device.line_w + 6, line.line_w.begin());
    return line;
}

LocalBAObservationData
ToLocalBAObservationData(const FlatLocalBAPointObservation &observation) {
    LocalBAObservationData data;
    data.camera = ToDeviceCamera(observation.camera);
    for (int i = 0; i < 3; ++i) {
        data.rcb[i] = observation.rcb[static_cast<size_t>(i)];
        data.tcb[i] = observation.tcb[static_cast<size_t>(i)];
    }
    data.information = observation.information;
    return data;
}

LocalBALineObservationData
ToLocalBAObservationData(const FlatLocalBALineObservation &observation) {
    LocalBALineObservationData data;
    data.camera = ToDeviceCamera(observation.camera);
    for (int i = 0; i < 3; ++i) {
        data.rcb[i] = observation.rcb[static_cast<size_t>(i)];
        data.tcb[i] = observation.tcb[static_cast<size_t>(i)];
    }
    std::copy(observation.endpoint_covariance_upper.begin(),
              observation.endpoint_covariance_upper.end(),
              data.endpoint_covariance_upper);
    data.has_endpoint_covariance =
        observation.has_endpoint_covariance &&
                IsValidLineEndpointCovariance(
                    observation.endpoint_covariance_upper)
            ? 1
            : 0;
    data.information = observation.information;
    return data;
}

const double *
CustomInformationOrNull(const std::array<double, 36> &information) {
    for (double value : information) {
        if (value != 0.0) {
            return information.data();
        }
    }
    return nullptr;
}

double ComputeEssentialGraphChi2(PoseGraphVertexDescriptor &pose_descriptor,
                                 EssentialGraphFactor &edge_factor) {
    pose_descriptor.to_device();
    edge_factor.initialize_device_ids(0);
    edge_factor.to_device();
    edge_factor.compute_error();
    cudaDeviceSynchronize();
    return edge_factor.chi2();
}

template <typename T>
void PrefetchManagedRange(T *data, size_t count, int device_id,
                          cudaStream_t stream) {
    if (!data || count == 0) {
        return;
    }
    cudaMemPrefetchAsync(data, count * sizeof(T), device_id, stream);
}

void PrefetchEssentialGraphToDevice(
    graphite::managed_vector<DevicePoseGraphVertex> &pose_storage,
    PoseGraphVertexDescriptor &pose_descriptor,
    EssentialGraphFactor &edge_factor) {
    int device_id = 0;
    if (cudaGetDevice(&device_id) != cudaSuccess) {
        return;
    }
    cudaStream_t stream = 0;
    PrefetchManagedRange(pose_storage.data().get(), pose_storage.size(),
                         device_id, stream);
    PrefetchManagedRange(pose_descriptor.active_state.data().get(),
                         pose_descriptor.active_state.size(), device_id,
                         stream);
    PrefetchManagedRange(pose_descriptor.block_ids.data().get(),
                         pose_descriptor.block_ids.size(), device_id, stream);
    PrefetchManagedRange(edge_factor.data.data().get(), edge_factor.data.size(),
                         device_id, stream);
    PrefetchManagedRange(edge_factor.precision_matrices.data().get(),
                         edge_factor.precision_matrices.size(), device_id,
                         stream);
    PrefetchManagedRange(edge_factor.loss.data().get(), edge_factor.loss.size(),
                         device_id, stream);
    cudaStreamSynchronize(stream);
}

void PrefetchPoseGraphToHost(
    graphite::managed_vector<DevicePoseGraphVertex> &pose_storage) {
    PrefetchManagedRange(pose_storage.data().get(), pose_storage.size(),
                         cudaCpuDeviceId, 0);
    cudaStreamSynchronize(0);
}

void PrefetchLocalBAToDevice(
    graphite::managed_vector<DevicePoseGraphVertex> &pose_storage,
    graphite::managed_vector<DeviceBAPointVertex> &point_storage,
    graphite::managed_vector<DeviceBALineVertex> &line_storage,
    PoseGraphVertexDescriptor &pose_descriptor,
    BAPointVertexDescriptor &point_descriptor,
    BALineVertexDescriptor &line_descriptor, LocalBAPointFactor *point_factor,
    LocalBALineFactor *line_factor) {
    int device_id = 0;
    if (cudaGetDevice(&device_id) != cudaSuccess) {
        return;
    }
    cudaStream_t stream = 0;
    PrefetchManagedRange(pose_storage.data().get(), pose_storage.size(),
                         device_id, stream);
    PrefetchManagedRange(point_storage.data().get(), point_storage.size(),
                         device_id, stream);
    PrefetchManagedRange(line_storage.data().get(), line_storage.size(),
                         device_id, stream);

    PrefetchManagedRange(pose_descriptor.active_state.data().get(),
                         pose_descriptor.active_state.size(), device_id,
                         stream);
    PrefetchManagedRange(pose_descriptor.block_ids.data().get(),
                         pose_descriptor.block_ids.size(), device_id, stream);
    PrefetchManagedRange(point_descriptor.active_state.data().get(),
                         point_descriptor.active_state.size(), device_id,
                         stream);
    PrefetchManagedRange(point_descriptor.block_ids.data().get(),
                         point_descriptor.block_ids.size(), device_id, stream);
    PrefetchManagedRange(line_descriptor.active_state.data().get(),
                         line_descriptor.active_state.size(), device_id,
                         stream);
    PrefetchManagedRange(line_descriptor.block_ids.data().get(),
                         line_descriptor.block_ids.size(), device_id, stream);

    if (point_factor) {
        PrefetchManagedRange(point_factor->device_obs.data().get(),
                             point_factor->device_obs.size(), device_id,
                             stream);
        PrefetchManagedRange(point_factor->data.data().get(),
                             point_factor->data.size(), device_id, stream);
        PrefetchManagedRange(point_factor->precision_matrices.data().get(),
                             point_factor->precision_matrices.size(), device_id,
                             stream);
        PrefetchManagedRange(point_factor->loss.data().get(),
                             point_factor->loss.size(), device_id, stream);
    }
    if (line_factor) {
        PrefetchManagedRange(line_factor->device_obs.data().get(),
                             line_factor->device_obs.size(), device_id, stream);
        PrefetchManagedRange(line_factor->data.data().get(),
                             line_factor->data.size(), device_id, stream);
        PrefetchManagedRange(line_factor->precision_matrices.data().get(),
                             line_factor->precision_matrices.size(), device_id,
                             stream);
        PrefetchManagedRange(line_factor->loss.data().get(),
                             line_factor->loss.size(), device_id, stream);
    }
    cudaStreamSynchronize(stream);
}

void PrefetchLocalBAToHost(
    graphite::managed_vector<DevicePoseGraphVertex> &pose_storage,
    graphite::managed_vector<DeviceBAPointVertex> &point_storage,
    graphite::managed_vector<DeviceBALineVertex> &line_storage) {
    PrefetchManagedRange(pose_storage.data().get(), pose_storage.size(),
                         cudaCpuDeviceId, 0);
    PrefetchManagedRange(point_storage.data().get(), point_storage.size(),
                         cudaCpuDeviceId, 0);
    PrefetchManagedRange(line_storage.data().get(), line_storage.size(),
                         cudaCpuDeviceId, 0);
    cudaStreamSynchronize(0);
}

double ComputeLocalBAChi2(PoseGraphVertexDescriptor &pose_descriptor,
                          BAPointVertexDescriptor &point_descriptor,
                          BALineVertexDescriptor &line_descriptor,
                          LocalBAPointFactor *point_factor,
                          LocalBALineFactor *line_factor) {
    pose_descriptor.to_device();
    point_descriptor.to_device();
    line_descriptor.to_device();
    double chi2 = 0.0;
    if (point_factor) {
        point_factor->initialize_device_ids(0);
        point_factor->to_device();
        point_factor->compute_error();
        chi2 += point_factor->chi2();
    }
    if (line_factor) {
        line_factor->initialize_device_ids(0);
        line_factor->to_device();
        line_factor->compute_error();
        chi2 += line_factor->chi2();
    }
    cudaDeviceSynchronize();
    return chi2;
}

int EvaluateLocalBAOutliersGpu(
    const graphite::managed_vector<DevicePoseGraphVertex> &pose_storage,
    const graphite::managed_vector<DeviceBAPointVertex> &point_storage,
    const graphite::managed_vector<DeviceBALineVertex> &line_storage,
    const thrust::device_vector<BAPointEvalRef> &point_refs,
    const thrust::device_vector<BALineEvalRef> &line_refs,
    double chi2_threshold,
    thrust::device_vector<uint8_t> &point_outliers_device,
    thrust::device_vector<uint8_t> &line_outliers_device,
    std::vector<uint8_t> &point_outliers_host,
    std::vector<uint8_t> &line_outliers_host) {
    thrust::device_vector<int> num_bad_device(1);
    thrust::fill(num_bad_device.begin(), num_bad_device.end(), 0);

    constexpr size_t threads_per_block = 256;
    const size_t point_count = point_refs.size();
    if (point_count > 0) {
        const size_t blocks =
            (point_count + threads_per_block - 1) / threads_per_block;
        EvaluateLocalBAPointOutliersKernel<<<blocks, threads_per_block>>>(
            pose_storage.data().get(), point_storage.data().get(),
            point_refs.data().get(), point_outliers_device.data().get(),
            num_bad_device.data().get(), chi2_threshold, point_count);
    }

    const size_t line_count = line_refs.size();
    if (line_count > 0) {
        const size_t blocks =
            (line_count + threads_per_block - 1) / threads_per_block;
        EvaluateLocalBALineOutliersKernel<<<blocks, threads_per_block>>>(
            pose_storage.data().get(), line_storage.data().get(),
            line_refs.data().get(), line_outliers_device.data().get(),
            num_bad_device.data().get(), chi2_threshold, line_count);
    }

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return -1;
    }
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        return -1;
    }

    if (point_count > 0) {
        thrust::copy(point_outliers_device.begin(), point_outliers_device.end(),
                     point_outliers_host.begin());
    }
    if (line_count > 0) {
        thrust::copy(line_outliers_device.begin(), line_outliers_device.end(),
                     line_outliers_host.begin());
    }

    int num_bad = 0;
    status = cudaMemcpy(&num_bad, num_bad_device.data().get(), sizeof(int),
                        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        return -1;
    }
    return num_bad;
}

bool RunEssentialGraphSolver(graphite::Graph<double, double> &graph,
                             graphite::Solver<double, double> &solver,
                             const FlatEssentialGraphProblem &problem) {
    graphite::StreamPool streams(8);
    graphite::optimizer::LevenbergMarquardtOptions<double, double> options;
    options.solver = &solver;
    options.streams = &streams;
    options.optimization_level = 0;
    options.initial_damping = problem.initial_damping;
    options.iterations = problem.iterations;
    options.use_identity = true;
    options.verbose = std::getenv("XPLSLAM_GRAPHITE_VERBOSE") != nullptr;
    try {
        return graphite::optimizer::levenberg_marquardt(&graph, &options);
    } catch (const std::exception &e) {
        if (options.verbose) {
            std::cerr << "EssentialGraph solver failed: " << e.what() << '\n';
        }
    } catch (...) {
        if (options.verbose) {
            std::cerr
                << "EssentialGraph solver failed with unknown exception\n";
        }
    }
    return false;
}

graphite::cudssSolverOptions LocalBACudssOptions() {
    graphite::cudssSolverOptions options;
    options.hybrid_memory = 0;
    return options;
}

struct LocalBASolverWorkspace {
    graphite::StreamPool streams;
    graphite::cudssSchurSolver<double, double> solver;

    LocalBASolverWorkspace() : streams(8), solver(LocalBACudssOptions()) {}
};

LocalBASolverWorkspace &GetLocalBASolverWorkspace() {
    thread_local std::unique_ptr<LocalBASolverWorkspace> workspace;
    if (!workspace) {
        workspace = std::make_unique<LocalBASolverWorkspace>();
    }
    return *workspace;
}

bool RunLocalBASolver(graphite::Graph<double, double> &graph,
                      const FlatLocalBAProblem &problem, size_t iterations,
                      LocalBASolverWorkspace &workspace) {
    graphite::optimizer::LevenbergMarquardtOptions<double, double> options;
    options.solver = &workspace.solver;
    options.streams = &workspace.streams;
    options.optimization_level = 0;
    options.initial_damping = problem.initial_damping;
    options.iterations = iterations;
    options.use_identity = true;
    options.stop_flag = problem.stop_flag;
    options.verbose = std::getenv("XPLSLAM_GRAPHITE_VERBOSE") != nullptr;
    try {
        return graphite::optimizer::levenberg_marquardt2(&graph, &options);
    } catch (const std::exception &e) {
        if (options.verbose) {
            std::cerr << "LocalBA solver failed: " << e.what() << '\n';
        }
    } catch (...) {
        if (options.verbose) {
            std::cerr << "LocalBA solver failed with unknown exception\n";
        }
    }
    return false;
}

} // namespace

bool OptimizePoseOnlyFlatInternal(const FlatPoseOnlyProblem &problem,
                                  FlatPoseOnlyResult &result) {
    int device_count = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (device_status != cudaSuccess || device_count <= 0) {
        return false;
    }

    result.optimized_pose = problem.initial_pose;
    result.num_bad = 0;
    if (!AreFiniteMapValues(problem.initial_pose.Rwb) ||
        !AreFiniteMapValues(problem.initial_pose.twb)) {
        return false;
    }

    std::vector<uint8_t> point_outliers(problem.num_points, 0);
    std::vector<uint8_t> line_outliers(problem.num_lines, 0);
    thrust::device_vector<uint8_t> point_outliers_device(problem.num_points);
    thrust::device_vector<uint8_t> line_outliers_device(problem.num_lines);

    graphite::managed_vector<DevicePoseVertex> pose_storage;
    pose_storage.reserve(1);
    pose_storage.push_back(ToDevicePoseVertex(problem.initial_pose));

    PoseVertexDescriptor pose_descriptor;
    pose_descriptor.reserve(1);
    pose_descriptor.add_vertex(0, pose_storage.data().get(), false);

    graphite::Graph<double, double> graph;
    graph.add_descriptor(&pose_descriptor);

    std::unique_ptr<PointFactor> point_factor;
    std::unique_ptr<LineFactor> line_factor;
    std::vector<size_t> point_factor_ids;
    std::vector<size_t> line_factor_ids;

    if (problem.num_points > 0) {
        point_factor = std::make_unique<PointFactor>(&pose_descriptor);
        point_factor->reserve(problem.num_points);
        point_factor_ids.reserve(problem.num_points);
        for (size_t i = 0; i < problem.num_points; ++i) {
            const FlatPoseOnlyPointInput &input = problem.points[i];
            PointFactorData data = ToPointFactorData(input);
            const std::array<double, 4> information = {input.information, 0.0,
                                                       0.0, input.information};
            point_factor_ids.push_back(point_factor->add_factor(
                {0}, ToDeviceVec2(input.measurement), information.data(), data,
                SwitchableHuberLoss<double, 2>(problem.huber_delta, true)));
        }
        graph.add_descriptor(point_factor.get());
    }

    if (problem.num_lines > 0) {
        line_factor = std::make_unique<LineFactor>(&pose_descriptor);
        line_factor->reserve(problem.num_lines);
        line_factor_ids.reserve(problem.num_lines);
        for (size_t i = 0; i < problem.num_lines; ++i) {
            const FlatPoseOnlyLineInput &input = problem.lines[i];
            LineFactorData data = ToLineFactorData(input);
            const std::array<double, 4> information = {1.0, 0.0, 0.0, 1.0};
            line_factor_ids.push_back(line_factor->add_factor(
                {0}, ToDeviceVec4(input.measurement), information.data(), data,
                SwitchableHuberLoss<double, 2>(problem.huber_delta, true)));
        }
        graph.add_descriptor(line_factor.get());
    }
    graph.scale_system(true);

    graphite::StreamPool streams(8);
    graphite::cudssSolverOptions cudss_options;
    graphite::cudssSolver<double, double> solver(cudss_options);
    graphite::optimizer::LevenbergMarquardtOptions<double, double> options;
    options.solver = &solver;
    options.streams = &streams;
    options.optimization_level = 0;
    options.initial_damping = 1e-6;
    options.use_identity = true;
    options.verbose = std::getenv("XPLSLAM_GRAPHITE_VERBOSE") != nullptr;

    constexpr int iterations[4] = {10, 10, 10, 10};
    for (int outer = 0; outer < 4; ++outer) {
        const DevicePoseVertex previous_pose = pose_storage[0];
        pose_storage[0] = ToDevicePoseVertex(problem.initial_pose);

        if (ActiveFactorCount(point_outliers, line_outliers) < 10) {
            if (outer == 0) {
                return false;
            }
            pose_storage[0] = previous_pose;
            break;
        }

        const bool robust_enabled = outer < 3;
        if (point_factor) {
            SetFactorState(*point_factor, point_factor_ids, point_outliers,
                           robust_enabled, problem.huber_delta);
        }
        if (line_factor) {
            SetFactorState(*line_factor, line_factor_ids, line_outliers,
                           robust_enabled, problem.huber_delta);
        }

        if (options.verbose) {
            pose_descriptor.to_device();
            if (point_factor) {
                point_factor->initialize_device_ids(options.optimization_level);
                point_factor->to_device();
                point_factor->compute_error();
                const double point_chi2 = point_factor->chi2();
                PrintCudaStatus("point factor preflight");
                std::cerr << "preflight outer=" << outer
                          << " point_active=" << point_factor->active_count()
                          << " point_chi2=" << point_chi2 << '\n';
                PrintFirstResiduals("point", *point_factor);
            }
            if (line_factor) {
                line_factor->initialize_device_ids(options.optimization_level);
                line_factor->to_device();
                line_factor->compute_error();
                const double line_chi2 = line_factor->chi2();
                PrintCudaStatus("line factor preflight");
                std::cerr << "preflight outer=" << outer
                          << " line_active=" << line_factor->active_count()
                          << " line_chi2=" << line_chi2 << '\n';
                PrintFirstResiduals("line", *line_factor);
            }
        }

        options.iterations = iterations[outer];
        if (!graphite::optimizer::levenberg_marquardt2(&graph, &options)) {
            return false;
        }

        if (cudaDeviceSynchronize() != cudaSuccess ||
            !AreFiniteMapValues(pose_storage[0].Rwb) ||
            !AreFiniteMapValues(pose_storage[0].twb)) {
            return false;
        }
        result.num_bad = EvaluateOutliersGpu(
            pose_storage[0], point_factor.get(), line_factor.get(),
            problem.chi2_threshold, point_outliers_device, line_outliers_device,
            point_outliers, line_outliers);
        if (result.num_bad < 0) {
            return false;
        }
    }

    cudaDeviceSynchronize();
    result.optimized_pose = FromDevicePoseVertexFlat(pose_storage[0]);
    if (problem.num_points > 0) {
        std::copy(point_outliers.begin(), point_outliers.end(),
                  result.point_outliers);
    }
    if (problem.num_lines > 0) {
        std::copy(line_outliers.begin(), line_outliers.end(),
                  result.line_outliers);
    }
    return true;
}

bool OptimizeLocalBAFlatInternal(const FlatLocalBAProblem &problem,
                                 FlatLocalBAResult &result) {
    int device_count = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (device_status != cudaSuccess || device_count <= 0) {
        return false;
    }

    if (problem.num_poses == 0 || (problem.num_point_observations == 0 &&
                                   problem.num_line_observations == 0)) {
        return false;
    }

    size_t free_pose_count = 0;
    for (size_t i = 0; i < problem.num_poses; ++i) {
        const bool fixed = problem.fixed_poses && problem.fixed_poses[i] != 0;
        free_pose_count += fixed ? 0 : 1;
    }
    if (free_pose_count == 0) {
        return false;
    }

    result.initial_chi2 = 0.0;
    result.final_chi2 = 0.0;
    result.used_two_stage = false;

    graphite::managed_vector<DevicePoseGraphVertex> pose_storage;
    graphite::managed_vector<DeviceBAPointVertex> point_storage;
    graphite::managed_vector<DeviceBALineVertex> line_storage;
    pose_storage.reserve(problem.num_poses);
    point_storage.reserve(problem.num_points);
    line_storage.reserve(problem.num_lines);

    for (size_t i = 0; i < problem.num_poses; ++i) {
        const DevicePoseGraphVertex pose =
            ToDevicePoseGraphVertex(problem.initial_poses[i]);
        pose_storage.push_back(pose);
    }
    for (size_t i = 0; i < problem.num_points; ++i) {
        const DeviceBAPointVertex point =
            ToDeviceBAPointVertex(problem.initial_points[i]);
        point_storage.push_back(point);
    }
    for (size_t i = 0; i < problem.num_lines; ++i) {
        DeviceBALineVertex line;
        if (!ToDeviceBALineVertex(problem.initial_lines[i], line)) {
            return false;
        }
        line_storage.push_back(line);
    }

    PoseGraphVertexDescriptor pose_descriptor;
    BAPointVertexDescriptor point_descriptor;
    BALineVertexDescriptor line_descriptor;
    pose_descriptor.reserve(problem.num_poses);
    point_descriptor.reserve(problem.num_points);
    line_descriptor.reserve(problem.num_lines);
    point_descriptor.set_eliminate(true);
    line_descriptor.set_eliminate(true);

    for (size_t i = 0; i < problem.num_poses; ++i) {
        const bool fixed = problem.fixed_poses && problem.fixed_poses[i] != 0;
        pose_descriptor.add_vertex(i, pose_storage.data().get() + i, fixed);
    }
    const size_t point_id_offset = problem.num_poses;
    const size_t line_id_offset = problem.num_poses + problem.num_points;
    for (size_t i = 0; i < problem.num_points; ++i) {
        point_descriptor.add_vertex(point_id_offset + i,
                                    point_storage.data().get() + i, false);
    }
    for (size_t i = 0; i < problem.num_lines; ++i) {
        line_descriptor.add_vertex(line_id_offset + i,
                                   line_storage.data().get() + i, false);
    }

    std::unique_ptr<LocalBAPointFactor> point_factor;
    std::unique_ptr<LocalBALineFactor> line_factor;
    std::vector<BAPointEvalRef> point_refs_host;
    std::vector<BALineEvalRef> line_refs_host;
    point_refs_host.reserve(problem.num_point_observations);
    line_refs_host.reserve(problem.num_line_observations);

    if (problem.num_point_observations > 0) {
        point_factor = std::make_unique<LocalBAPointFactor>(&point_descriptor,
                                                            &pose_descriptor);
        point_factor->reserve(problem.num_point_observations);
        for (size_t i = 0; i < problem.num_point_observations; ++i) {
            const FlatLocalBAPointObservation &observation =
                problem.point_observations[i];
            if (observation.pose >= problem.num_poses ||
                observation.point >= problem.num_points ||
                observation.information <= 0.0 ||
                !std::isfinite(observation.information)) {
                return false;
            }
            const std::array<double, 4> information = {
                observation.information, 0.0, 0.0, observation.information};
            const LocalBAObservationData data =
                ToLocalBAObservationData(observation);
            const DeviceVec2 measurement =
                ToDeviceVec2(observation.measurement);
            point_factor->add_factor_sequential(
                {point_id_offset + observation.point, observation.pose},
                measurement, information.data(), data,
                SwitchableHuberLoss<double, 2>(problem.huber_delta, true));
            point_refs_host.push_back(BAPointEvalRef{
                observation.pose, observation.point, data, measurement});
        }
    }

    if (problem.num_line_observations > 0) {
        line_factor = std::make_unique<LocalBALineFactor>(&line_descriptor,
                                                          &pose_descriptor);
        line_factor->reserve(problem.num_line_observations);
        for (size_t i = 0; i < problem.num_line_observations; ++i) {
            const FlatLocalBALineObservation &observation =
                problem.line_observations[i];
            if (observation.pose >= problem.num_poses ||
                observation.line >= problem.num_lines ||
                observation.information <= 0.0 ||
                !std::isfinite(observation.information)) {
                return false;
            }
            const std::array<double, 4> information = {1.0, 0.0, 0.0, 1.0};
            const LocalBALineObservationData data =
                ToLocalBAObservationData(observation);
            const DeviceVec4 measurement =
                ToDeviceVec4(observation.measurement);
            line_factor->add_factor_sequential(
                {line_id_offset + observation.line, observation.pose},
                measurement, information.data(), data,
                SwitchableHuberLoss<double, 2>(problem.huber_delta, true));
            line_refs_host.push_back(BALineEvalRef{
                observation.pose, observation.line, data, measurement});
        }
    }

    graphite::Graph<double, double> graph;
    graph.add_descriptor(&pose_descriptor);
    if (problem.num_point_observations > 0) {
        graph.add_descriptor(&point_descriptor);
    }
    if (problem.num_line_observations > 0) {
        graph.add_descriptor(&line_descriptor);
    }
    if (point_factor) {
        graph.add_descriptor(point_factor.get());
    }
    if (line_factor) {
        graph.add_descriptor(line_factor.get());
    }
    graph.scale_system(false);

    thrust::device_vector<BAPointEvalRef> point_refs_device(
        point_refs_host.begin(), point_refs_host.end());
    thrust::device_vector<BALineEvalRef> line_refs_device(
        line_refs_host.begin(), line_refs_host.end());
    std::vector<uint8_t> point_outliers(problem.num_point_observations, 0);
    std::vector<uint8_t> line_outliers(problem.num_line_observations, 0);
    thrust::device_vector<uint8_t> point_outliers_device(
        problem.num_point_observations);
    thrust::device_vector<uint8_t> line_outliers_device(
        problem.num_line_observations);

    const bool validate_chi2 =
        std::getenv("XPLSLAM_GRAPHITE_LBA_VALIDATE_CHI2") != nullptr;
    if (validate_chi2) {
        result.initial_chi2 = ComputeLocalBAChi2(
            pose_descriptor, point_descriptor, line_descriptor,
            point_factor.get(), line_factor.get());
    }

    LocalBASolverWorkspace &solver_workspace = GetLocalBASolverWorkspace();

    auto prefetch = [&]() {
        PrefetchLocalBAToDevice(pose_storage, point_storage, line_storage,
                                pose_descriptor, point_descriptor,
                                line_descriptor, point_factor.get(),
                                line_factor.get());
    };

    auto set_factor_state = [&](bool robust_enabled) {
        if (point_factor) {
            SetFactorStateLocal(*point_factor, point_outliers, robust_enabled,
                                problem.huber_delta);
        }
        if (line_factor) {
            SetFactorStateLocal(*line_factor, line_outliers, robust_enabled,
                                problem.huber_delta);
        }
    };

    auto run_stage = [&](size_t iterations, bool robust_enabled) {
        if (iterations == 0) {
            return true;
        }
        set_factor_state(robust_enabled);

        prefetch();

        const bool ok =
            RunLocalBASolver(graph, problem, iterations, solver_workspace);

        const cudaError_t sync_status = cudaDeviceSynchronize();
        return ok && sync_status == cudaSuccess;
    };

    auto evaluate_outliers = [&]() {
        return EvaluateLocalBAOutliersGpu(
            pose_storage, point_storage, line_storage, point_refs_device,
            line_refs_device, problem.chi2_threshold, point_outliers_device,
            line_outliers_device, point_outliers, line_outliers);
    };

    bool optimized = false;
    switch (problem.schedule) {
    case FlatLocalBASchedule::kSingleStage:
        optimized = run_stage(problem.single_stage_iterations, true);
        if (optimized && evaluate_outliers() < 0) {
            optimized = false;
        }
        break;
    case FlatLocalBASchedule::kTwoStage: {
        optimized = run_stage(problem.first_stage_iterations, true);
        int num_bad = optimized ? evaluate_outliers() : -1;
        if (optimized && num_bad >= 0) {
            result.used_two_stage = true;
            optimized = run_stage(problem.second_stage_iterations, false);
            if (optimized && evaluate_outliers() < 0) {
                optimized = false;
            }
        } else {
            optimized = false;
        }
        break;
    }
    case FlatLocalBASchedule::kAutoMerge: {
        optimized = run_stage(problem.first_stage_iterations, true);
        const int num_bad = optimized ? evaluate_outliers() : -1;
        if (!optimized || num_bad < 0) {
            optimized = false;
            break;
        }

        result.used_two_stage = true;
        optimized = run_stage(problem.second_stage_iterations, false);
        if (optimized && evaluate_outliers() < 0) {
            optimized = false;
        }
        break;
    }
    }

    if (!optimized) {
        return false;
    }

    if (validate_chi2) {
        result.final_chi2 = ComputeLocalBAChi2(
            pose_descriptor, point_descriptor, line_descriptor,
            point_factor.get(), line_factor.get());
    }

    PrefetchLocalBAToHost(pose_storage, point_storage, line_storage);

    for (size_t i = 0; i < problem.num_poses; ++i) {
        result.optimized_poses[i] =
            FromDevicePoseGraphVertexFlat(pose_storage[i]);
        if (!AreFiniteMapValues(result.optimized_poses[i].Rwb) ||
            !AreFiniteMapValues(result.optimized_poses[i].twb)) {
            return false;
        }
    }
    for (size_t i = 0; i < problem.num_points; ++i) {
        result.optimized_points[i] =
            FromDeviceBAPointVertexFlat(point_storage[i]);
        if (!AreFiniteMapValues(result.optimized_points[i].position)) {
            return false;
        }
    }
    for (size_t i = 0; i < problem.num_lines; ++i) {
        result.optimized_lines[i] = FromDeviceBALineVertexFlat(line_storage[i]);
        if (!AreFiniteMapValues(result.optimized_lines[i].line_w)) {
            return false;
        }
    }
    if (problem.num_point_observations > 0) {
        std::copy(point_outliers.begin(), point_outliers.end(),
                  result.point_outliers);
    }
    if (problem.num_line_observations > 0) {
        std::copy(line_outliers.begin(), line_outliers.end(),
                  result.line_outliers);
    }
    return true;
}

bool OptimizeEssentialGraphFlatInternal(
    const FlatEssentialGraphProblem &problem,
    FlatEssentialGraphResult &result) {
    int device_count = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (device_status != cudaSuccess || device_count <= 0) {
        return false;
    }

    if (problem.num_poses == 0) {
        return false;
    }

    result.initial_chi2 = 0.0;
    result.final_chi2 = 0.0;
    for (size_t i = 0; i < problem.num_poses; ++i) {
        result.optimized_poses[i] = problem.initial_poses[i];
    }

    if (problem.num_edges == 0) {
        return true;
    }

    graphite::managed_vector<DevicePoseGraphVertex> pose_storage;
    pose_storage.reserve(problem.num_poses);
    for (size_t i = 0; i < problem.num_poses; ++i) {
        DevicePoseGraphVertex vertex =
            ToDevicePoseGraphVertex(problem.initial_poses[i]);
        pose_storage.push_back(vertex);
    }

    PoseGraphVertexDescriptor pose_descriptor;
    pose_descriptor.reserve(problem.num_poses);
    for (size_t i = 0; i < problem.num_poses; ++i) {
        const bool fixed = problem.fixed && problem.fixed[i] != 0;
        pose_descriptor.add_vertex(i, pose_storage.data().get() + i, fixed);
    }

    EssentialGraphFactor edge_factor(&pose_descriptor, &pose_descriptor);
    edge_factor.reserve(problem.num_edges);
    for (size_t i = 0; i < problem.num_edges; ++i) {
        const FlatEssentialGraphEdgeInput &edge = problem.edges[i];
        if (edge.vertex_i >= problem.num_poses ||
            edge.vertex_j >= problem.num_poses) {
            return false;
        }
        edge_factor.add_factor({edge.vertex_i, edge.vertex_j},
                               graphite::Empty(),
                               CustomInformationOrNull(edge.information),
                               ToEssentialGraphEdgeData(edge));
    }

    graphite::Graph<double, double> graph;
    graph.add_descriptor(&pose_descriptor);
    graph.add_descriptor(&edge_factor);
    graph.scale_system(false);

    const bool validate_chi2 =
        std::getenv("XPLSLAM_GRAPHITE_EG_VALIDATE_CHI2") != nullptr;
    if (validate_chi2) {
        result.initial_chi2 =
            ComputeEssentialGraphChi2(pose_descriptor, edge_factor);
    }
    PrefetchEssentialGraphToDevice(pose_storage, pose_descriptor, edge_factor);

    bool optimized = false;
    {
        graphite::cudssSolverOptions cudss_options;
        cudss_options.hybrid_memory = 0;
        graphite::cudssSolver<double, double> solver(cudss_options);
        std::chrono::steady_clock::time_point start =
            std::chrono::steady_clock::now();
        optimized = RunEssentialGraphSolver(graph, solver, problem);
        std::chrono::steady_clock::time_point end =
            std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "EssentialGraph optimization with cudssSolver took "
                  << elapsed_seconds.count() * 1000 << " ms" << std::endl;
    }

    if (!optimized) {
        return false;
    }

    cudaDeviceSynchronize();
    if (validate_chi2) {
        result.final_chi2 =
            ComputeEssentialGraphChi2(pose_descriptor, edge_factor);
    }
    PrefetchPoseGraphToHost(pose_storage);
    for (size_t i = 0; i < problem.num_poses; ++i) {
        result.optimized_poses[i] =
            FromDevicePoseGraphVertexFlat(pose_storage[i]);
    }
    return true;
}

bool OptimizePoseOnlyFlat(const FlatPoseOnlyProblem &flat_problem,
                          FlatPoseOnlyResult &flat_result) {
    if (flat_problem.num_points > 0 && !flat_problem.points) {
        return false;
    }
    if (flat_problem.num_lines > 0 && !flat_problem.lines) {
        return false;
    }
    if (flat_result.num_point_outliers < flat_problem.num_points ||
        flat_result.num_line_outliers < flat_problem.num_lines) {
        return false;
    }

    return OptimizePoseOnlyFlatInternal(flat_problem, flat_result);
}

bool OptimizeEssentialGraphFlat(const FlatEssentialGraphProblem &flat_problem,
                                FlatEssentialGraphResult &flat_result) {
    if (flat_problem.num_poses > 0 && !flat_problem.initial_poses) {
        return false;
    }
    if (flat_problem.num_edges > 0 && !flat_problem.edges) {
        return false;
    }
    if (flat_result.num_optimized_poses < flat_problem.num_poses ||
        (flat_problem.num_poses > 0 && !flat_result.optimized_poses)) {
        return false;
    }
    if (flat_problem.iterations == 0) {
        return false;
    }

    return OptimizeEssentialGraphFlatInternal(flat_problem, flat_result);
}

bool OptimizeLocalBAFlat(const FlatLocalBAProblem &flat_problem,
                         FlatLocalBAResult &flat_result) {
    if (flat_problem.num_poses == 0 || !flat_problem.initial_poses) {
        return false;
    }
    if (flat_problem.num_points > 0 && !flat_problem.initial_points) {
        return false;
    }
    if (flat_problem.num_lines > 0 && !flat_problem.initial_lines) {
        return false;
    }
    if (flat_problem.num_point_observations > 0 &&
        !flat_problem.point_observations) {
        return false;
    }
    if (flat_problem.num_line_observations > 0 &&
        !flat_problem.line_observations) {
        return false;
    }
    if (flat_result.num_optimized_poses < flat_problem.num_poses ||
        !flat_result.optimized_poses) {
        return false;
    }
    if (flat_result.num_optimized_points < flat_problem.num_points ||
        (flat_problem.num_points > 0 && !flat_result.optimized_points)) {
        return false;
    }
    if (flat_result.num_optimized_lines < flat_problem.num_lines ||
        (flat_problem.num_lines > 0 && !flat_result.optimized_lines)) {
        return false;
    }
    if (flat_result.num_point_outliers < flat_problem.num_point_observations ||
        (flat_problem.num_point_observations > 0 &&
         !flat_result.point_outliers)) {
        return false;
    }
    if (flat_result.num_line_outliers < flat_problem.num_line_observations ||
        (flat_problem.num_line_observations > 0 &&
         !flat_result.line_outliers)) {
        return false;
    }
    if (flat_problem.single_stage_iterations == 0 &&
        flat_problem.first_stage_iterations == 0 &&
        flat_problem.second_stage_iterations == 0) {
        return false;
    }

    return OptimizeLocalBAFlatInternal(flat_problem, flat_result);
}

} // namespace XPLSLAM::graphite_backend
