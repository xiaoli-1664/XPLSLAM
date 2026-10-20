#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>

#if defined(__CUDACC__)
#define XPLSLAM_GRAPHITE_HD __host__ __device__
#else
#define XPLSLAM_GRAPHITE_HD
#endif

namespace XPLSLAM::graphite_backend {

using Vector6d = Eigen::Matrix<double, 6, 1>;

// Absolute body-to-world pose, updated by Rwb *= Exp(dtheta), twb += dt_world.
struct PoseVertex {
    Eigen::Matrix3d Rwb = Eigen::Matrix3d::Identity();
    Eigen::Vector3d twb = Eigen::Vector3d::Zero();
};

struct PinholeCamera {
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    Eigen::Matrix3d toPL = Eigen::Matrix3d::Zero();
};

struct Pose {
    Eigen::Matrix3d Rwb = Eigen::Matrix3d::Identity();
    Eigen::Vector3d twb = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Rcb = Eigen::Matrix3d::Identity();
    Eigen::Vector3d tcb = Eigen::Vector3d::Zero();
};

XPLSLAM_GRAPHITE_HD inline Pose MakeCameraPose(
    const PoseVertex &vertex, const Eigen::Matrix3d &Rcb,
    const Eigen::Vector3d &tcb) {
    Pose pose;
    pose.Rwb = vertex.Rwb;
    pose.twb = vertex.twb;
    pose.Rcb = Rcb;
    pose.tcb = tcb;
    return pose;
}

XPLSLAM_GRAPHITE_HD inline Eigen::Matrix3d
PinholeLineProjectionMatrix(double fx, double fy, double cx, double cy) {
    Eigen::Matrix3d to_pl;
    to_pl.setZero();
    to_pl(0, 0) = fy;
    to_pl(1, 1) = fx;
    to_pl(2, 0) = -fy * cx;
    to_pl(2, 1) = -fx * cy;
    to_pl(2, 2) = fx * fy;
    return to_pl;
}

XPLSLAM_GRAPHITE_HD inline PinholeCamera
MakePinholeCamera(double fx, double fy, double cx, double cy) {
    PinholeCamera camera;
    camera.fx = fx;
    camera.fy = fy;
    camera.cx = cx;
    camera.cy = cy;
    camera.toPL = PinholeLineProjectionMatrix(fx, fy, cx, cy);
    return camera;
}

XPLSLAM_GRAPHITE_HD inline Eigen::Matrix3d CameraRotation(const Pose &pose) {
    return pose.Rcb * pose.Rwb.transpose();
}

XPLSLAM_GRAPHITE_HD inline Eigen::Vector3d CameraTranslation(const Pose &pose) {
    return pose.tcb - CameraRotation(pose) * pose.twb;
}

XPLSLAM_GRAPHITE_HD inline Eigen::Vector3d CameraPoint(
    const Pose &pose, const Eigen::Vector3d &point_w) {
    return CameraRotation(pose) * point_w + CameraTranslation(pose);
}

XPLSLAM_GRAPHITE_HD inline Eigen::Vector2d ProjectPoint(
    const PinholeCamera &camera, const Eigen::Vector3d &point_c) {
    Eigen::Vector2d projection;
    projection[0] = camera.fx * point_c[0] / point_c[2] + camera.cx;
    projection[1] = camera.fy * point_c[1] / point_c[2] + camera.cy;
    return projection;
}

XPLSLAM_GRAPHITE_HD inline Eigen::Vector2d ProjectWorldPoint(
    const Pose &pose, const PinholeCamera &camera,
    const Eigen::Vector3d &point_w) {
    return ProjectPoint(camera, CameraPoint(pose, point_w));
}

XPLSLAM_GRAPHITE_HD inline Eigen::Vector3d CameraLineNormal(
    const Pose &pose, const Vector6d &line_w) {
    const Eigen::Matrix3d Rcw = CameraRotation(pose);
    const Eigen::Vector3d tcw = CameraTranslation(pose);
    const Eigen::Vector3d direction_c = Rcw * line_w.tail<3>();
    return Rcw * line_w.head<3>() + tcw.cross(direction_c);
}

XPLSLAM_GRAPHITE_HD inline Eigen::Vector3d ProjectWorldLine(
    const Pose &pose, const PinholeCamera &camera, const Vector6d &line_w) {
    return camera.toPL * CameraLineNormal(pose, line_w);
}

XPLSLAM_GRAPHITE_HD inline double Line2DError(
    const Eigen::Vector3d &line_2d, const Eigen::Vector2d &point_2d) {
    const double scale =
        sqrt(line_2d[0] * line_2d[0] + line_2d[1] * line_2d[1]);
    return (line_2d[0] * point_2d[0] + line_2d[1] * point_2d[1] +
            line_2d[2]) /
           scale;
}

XPLSLAM_GRAPHITE_HD inline Eigen::Vector2d PoseOnlyPointResidual(
    const Pose &pose, const PinholeCamera &camera,
    const Eigen::Vector3d &point_w, const Eigen::Vector2d &measurement) {
    return measurement - ProjectWorldPoint(pose, camera, point_w);
}

XPLSLAM_GRAPHITE_HD inline Eigen::Vector2d PoseOnlyLineResidual(
    const Pose &pose, const PinholeCamera &camera, const Vector6d &line_w,
    const Eigen::Vector4d &measurement) {
    const Eigen::Vector3d line_2d = ProjectWorldLine(pose, camera, line_w);
    Eigen::Vector2d residual;
    residual[0] = Line2DError(line_2d, measurement.head<2>());
    residual[1] = Line2DError(line_2d, measurement.tail<2>());
    return residual;
}

XPLSLAM_GRAPHITE_HD inline bool IsPointDepthPositive(
    const Pose &pose, const Eigen::Vector3d &point_w) {
    return CameraPoint(pose, point_w)[2] > 0.0;
}

template <typename T>
XPLSLAM_GRAPHITE_HD inline Eigen::Matrix<T, 3, 3>
SkewMatrix(const Eigen::Matrix<T, 3, 1> &v) {
    Eigen::Matrix<T, 3, 3> skew;
    skew << T(0), -v[2], v[1], v[2], T(0), -v[0], -v[1], v[0], T(0);
    return skew;
}

template <typename T>
XPLSLAM_GRAPHITE_HD inline Eigen::Matrix<T, 3, 3>
ExpSO3(const Eigen::Matrix<T, 3, 1> &w) {
    const T d2 = w.squaredNorm();
    const T d = sqrt(d2);
    const Eigen::Matrix<T, 3, 3> W = SkewMatrix(w);

    Eigen::Matrix<T, 3, 3> rotation = Eigen::Matrix<T, 3, 3>::Identity();
    if (d < T(1e-5)) {
        rotation += W + T(0.5) * W * W;
    } else {
        rotation += W * sin(d) / d + W * W * (T(1) - cos(d)) / d2;
    }
    return rotation;
}

template <typename T>
XPLSLAM_GRAPHITE_HD inline Eigen::Matrix<T, 2, 3>
ProjectJacobian(const PinholeCamera &camera,
                const Eigen::Matrix<T, 3, 1> &point_c) {
    Eigen::Matrix<T, 2, 3> jacobian;
    const T inv_z = T(1) / point_c[2];
    const T inv_z2 = inv_z * inv_z;
    jacobian << T(camera.fx) * inv_z, T(0),
        -T(camera.fx) * point_c[0] * inv_z2, T(0), T(camera.fy) * inv_z,
        -T(camera.fy) * point_c[1] * inv_z2;
    return jacobian;
}

template <typename T>
XPLSLAM_GRAPHITE_HD inline Eigen::Matrix<T, 3, 1>
Line2DJacobian(const Eigen::Matrix<T, 3, 1> &line_2d,
               const Eigen::Matrix<T, 2, 1> &point_2d) {
    const T a = line_2d[0];
    const T b = line_2d[1];
    const T c = line_2d[2];
    const T x = point_2d[0];
    const T y = point_2d[1];
    const T s2 = a * a + b * b;
    const T s = sqrt(s2);
    const T d = a * x + b * y + c;

    Eigen::Matrix<T, 3, 1> jacobian;
    jacobian[0] = x / s - d * a / (s2 * s);
    jacobian[1] = y / s - d * b / (s2 * s);
    jacobian[2] = T(1) / s;
    return jacobian;
}

} // namespace XPLSLAM::graphite_backend

#undef XPLSLAM_GRAPHITE_HD
