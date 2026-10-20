#include "geometric_tools.h"

#include "camera_models/geometric_camera.h"
#include "keyframe.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace XPLSLAM {

constexpr float kMinDepth = 1e-9f;
constexpr float kMinLineDirectionNorm = 1e-2f;
constexpr float kMinLineSegmentLength = 1e-6f;
constexpr float kMinPlaneNorm = 1e-6f;
constexpr float kMaxTriangulationPlaneCos = 0.9998f;
constexpr double kMinLineNormDouble = 1e-12;

template <typename Scalar>
bool LinePointFromBackprojectedRayImpl(
    const Eigen::Matrix<Scalar, 6, 1> &line_c,
    const Eigen::Matrix<Scalar, 3, 1> &ray_c,
    Eigen::Matrix<Scalar, 3, 1> &point_c, const Scalar min_line_norm) {
    const Eigen::Matrix<Scalar, 3, 1> moment = line_c.template head<3>();
    const Eigen::Matrix<Scalar, 3, 1> direction = line_c.template tail<3>();
    const Scalar direction_norm2 = direction.squaredNorm();
    const Scalar ray_norm2 = ray_c.squaredNorm();
    if (direction_norm2 < min_line_norm || ray_norm2 < min_line_norm ||
        !moment.allFinite() || !direction.allFinite() || !ray_c.allFinite()) {
        return false;
    }

    const Eigen::Matrix<Scalar, 3, 1> point_on_line =
        direction.cross(moment) / direction_norm2;
    const Scalar direction_dot_ray = direction.dot(ray_c);
    const Scalar point_dot_direction = point_on_line.dot(direction);
    const Scalar point_dot_ray = point_on_line.dot(ray_c);
    const Scalar denom =
        direction_norm2 * ray_norm2 - direction_dot_ray * direction_dot_ray;
    if (std::abs(denom) < min_line_norm) {
        return false;
    }

    const Scalar line_parameter =
        (direction_dot_ray * point_dot_ray - ray_norm2 * point_dot_direction) /
        denom;
    point_c = point_on_line + line_parameter * direction;
    return point_c.allFinite();
}

Eigen::Matrix3f GeometricTools::ComputeF12(const KeyFrame &kf1, int cam_id1,
                                           const KeyFrame &kf2, int cam_id2) {
    if (!kf1.mvpCameras[cam_id1] || !kf2.mvpCameras[cam_id2]) {
        return Eigen::Matrix3f::Zero();
    }

    Sophus::SE3<float> Tc1w = kf1.GetCameraPose(cam_id1);
    Sophus::Matrix3<float> Rc1w = Tc1w.rotationMatrix();
    Sophus::SE3<float>::TranslationMember tc1w = Tc1w.translation();

    Sophus::SE3<float> Tc2w = kf2.GetCameraPose(cam_id2);
    Sophus::Matrix3<float> Rc2w = Tc2w.rotationMatrix();
    Sophus::SE3<float>::TranslationMember tc2w = Tc2w.translation();

    Sophus::Matrix3<float> Rc1c2 = Rc1w * Rc2w.transpose();
    Eigen::Vector3f tc1c2 = -Rc1c2 * tc2w + tc1w;

    Eigen::Matrix3f tc1c2x = Sophus::SO3f::hat(tc1c2);

    const Eigen::Matrix3f K1 = kf1.mvpCameras[cam_id1]->toK_();
    const Eigen::Matrix3f K2 = kf2.mvpCameras[cam_id2]->toK_();

    return K1.transpose().inverse() * tc1c2x * Rc1c2 * K2.inverse();
}

bool GeometricTools::Triangulate(Eigen::Vector3f &x_c1, Eigen::Vector3f &x_c2,
                                 Eigen::Matrix<float, 3, 4> &Tc1w,
                                 Eigen::Matrix<float, 3, 4> &Tc2w,
                                 Eigen::Vector3f &x3D) {
    Eigen::Matrix4f A;
    A.block<1, 4>(0, 0) =
        x_c1(0) * Tc1w.block<1, 4>(2, 0) - Tc1w.block<1, 4>(0, 0);
    A.block<1, 4>(1, 0) =
        x_c1(1) * Tc1w.block<1, 4>(2, 0) - Tc1w.block<1, 4>(1, 0);
    A.block<1, 4>(2, 0) =
        x_c2(0) * Tc2w.block<1, 4>(2, 0) - Tc2w.block<1, 4>(0, 0);
    A.block<1, 4>(3, 0) =
        x_c2(1) * Tc2w.block<1, 4>(2, 0) - Tc2w.block<1, 4>(1, 0);

    Eigen::JacobiSVD<Eigen::Matrix4f> svd(A, Eigen::ComputeFullV);

    Eigen::Vector4f x3Dh = svd.matrixV().col(3);

    if (x3Dh(3) == 0)
        return false;

    // Euclidean coordinates
    x3D = x3Dh.head(3) / x3Dh(3);

    return true;
}

bool GeometricTools::ImagePointToGridCell(const cv::Point2f &kp, float min_x,
                                          float min_y, float width_inv,
                                          float height_inv, int cols, int rows,
                                          int &pos_x, int &pos_y) {
    pos_x = static_cast<int>(std::floor((kp.x - min_x) * width_inv));
    pos_y = static_cast<int>(std::floor((kp.y - min_y) * height_inv));

    return pos_x >= 0 && pos_x < cols && pos_y >= 0 && pos_y < rows;
}

bool GeometricTools::NormalizeLinePolar(float &d, float &theta) {
    if (!std::isfinite(d) || !std::isfinite(theta)) {
        return false;
    }

    theta = std::fmod(theta, 2.0f * kPi);
    if (theta < 0.0f) {
        theta += 2.0f * kPi;
    }
    if (theta >= kPi) {
        theta -= kPi;
        d = -d;
    }

    return std::isfinite(d) && std::isfinite(theta);
}

bool GeometricTools::NormalizedLinePolarToGridCell(float d, float theta,
                                                   float d_max, float theta_inv,
                                                   float d_inv, int cols,
                                                   int rows, int &pos_x,
                                                   int &pos_y) {
    if (!std::isfinite(d) || !std::isfinite(theta) ||
        !std::isfinite(d_max) || !std::isfinite(theta_inv) ||
        !std::isfinite(d_inv) || d_max <= 0.0f || theta_inv <= 0.0f ||
        d_inv <= 0.0f || cols <= 0 || rows <= 0 || theta < 0.0f ||
        theta >= kPi || d < -d_max || d > d_max) {
        return false;
    }

    // Cells are half-open bins, matching the floor/ceil bounds used by
    // Frame and KeyFrame searches. Clamp the upper boundary for roundoff.
    pos_x = static_cast<int>(std::min(
        std::floor(theta * theta_inv), static_cast<float>(cols - 1)));
    pos_y = static_cast<int>(std::min(
        std::floor((d + d_max) * d_inv), static_cast<float>(rows - 1)));

    return pos_x >= 0 && pos_x < cols && pos_y >= 0 && pos_y < rows;
}

bool GeometricTools::LinePolarToGridCell(float d, float theta, float d_max,
                                         float theta_inv, float d_inv, int cols,
                                         int rows, int &pos_x, int &pos_y) {
    if (!NormalizeLinePolar(d, theta)) {
        return false;
    }

    return NormalizedLinePolarToGridCell(d, theta, d_max, theta_inv, d_inv,
                                         cols, rows, pos_x, pos_y);
}

bool GeometricTools::ImageLineToNormalizedPolar(const Eigen::Vector4f &line,
                                                Eigen::Vector2f &line_polar) {
    const float x1 = line(0);
    const float y1 = line(1);
    const float x2 = line(2);
    const float y2 = line(3);

    const float dx = x2 - x1;
    const float dy = y2 - y1;
    if (dx * dx + dy * dy < kMinLineSegmentLength * kMinLineSegmentLength) {
        return false;
    }

    float theta = std::atan2(dx, -dy);
    float d = x1 * std::cos(theta) + y1 * std::sin(theta);
    if (!NormalizeLinePolar(d, theta)) {
        return false;
    }

    line_polar << d, theta;
    return true;
}

bool GeometricTools::ImageLineToNormalizedPolar(const Eigen::Vector3f &line,
                                                float &d, float &theta) {
    const float norm = std::hypot(line(0), line(1));
    if (norm < kMinLineDirectionNorm || !line.allFinite()) {
        return false;
    }

    d = -line(2) / norm;
    theta = std::atan2(line(1), line(0));
    return NormalizeLinePolar(d, theta);
}

Eigen::Matrix<float, 3, 4>
GeometricTools::MakeProjectionMatrix(const Sophus::SE3f &Tcw) {
    Eigen::Matrix<float, 3, 4> projection = Eigen::Matrix<float, 3, 4>::Zero();
    projection.block<3, 3>(0, 0) = Tcw.rotationMatrix();
    projection.col(3) = Tcw.translation();
    return projection;
}

bool GeometricTools::BuildObservationPlane(const Eigen::VectorXf &line,
                                           GeometricCamera *camera,
                                           Eigen::Vector4f &plane) {
    if (camera == nullptr || line.size() < 4) {
        return false;
    }

    const cv::Point2f start(line(0), line(1));
    const cv::Point2f end(line(2), line(3));
    const Eigen::Vector3f ray_start = camera->unprojectEig(start);
    const Eigen::Vector3f ray_end = camera->unprojectEig(end);
    Eigen::Vector3f normal = ray_start.cross(ray_end);
    const float normal_norm = normal.norm();
    if (normal_norm < kMinPlaneNorm || !normal.allFinite()) {
        return false;
    }

    normal /= normal_norm;
    plane << normal, 0.0f;
    return plane.allFinite();
}

bool GeometricTools::TriangulateLine(const Eigen::Vector4f &line1,
                                     GeometricCamera *camera1,
                                     const Sophus::SE3f &Tcw1,
                                     const Eigen::Vector4f &line2,
                                     GeometricCamera *camera2,
                                     const Sophus::SE3f &Tcw2,
                                     Eigen::Matrix<float, 6, 1> &line_w) {
    Eigen::Vector4f plane1_c;
    Eigen::Vector4f plane2_c;
    if (!BuildObservationPlane(line1, camera1, plane1_c) ||
        !BuildObservationPlane(line2, camera2, plane2_c)) {
        return false;
    }

    const Eigen::Vector4f plane1_w = Tcw1.matrix().transpose() * plane1_c;
    const Eigen::Vector4f plane2_w = Tcw2.matrix().transpose() * plane2_c;
    const Eigen::Vector3f normal1 = plane1_w.head<3>();
    const Eigen::Vector3f normal2 = plane2_w.head<3>();

    const float normal1_norm = normal1.norm();
    const float normal2_norm = normal2.norm();
    if (normal1_norm < kMinLineDirectionNorm ||
        normal2_norm < kMinLineDirectionNorm || !plane1_w.allFinite() ||
        !plane2_w.allFinite()) {
        return false;
    }

    const float cos_angle =
        std::abs(normal1.dot(normal2) / (normal1_norm * normal2_norm));
    if (!std::isfinite(cos_angle) || cos_angle > kMaxTriangulationPlaneCos) {
        return false;
    }

    Eigen::Vector3f direction = normal1.cross(normal2);
    const float direction_norm = direction.norm();
    if (direction_norm < kMinLineDirectionNorm || !direction.allFinite()) {
        return false;
    }

    Eigen::Vector3f moment = plane1_w(3) * normal2 - plane2_w(3) * normal1;
    if (!moment.allFinite() || moment.norm() < kMinLineDirectionNorm) {
        return false;
    }

    direction /= direction_norm;
    moment /= direction_norm;

    line_w.head<3>() = moment;
    line_w.tail<3>() = direction;
    return line_w.allFinite();
}

float GeometricTools::LineDepthInCamera(
    const Eigen::Matrix<float, 6, 1> &line_w, const Sophus::SE3f &Tcw) {
    Eigen::Matrix<float, 6, 1> line_c;
    TransformLineToCamera(line_w, Tcw, line_c);
    const Eigen::Vector3f direction_c = line_c.tail<3>();
    const Eigen::Vector3f moment_c = line_c.head<3>();
    const float direction_norm = direction_c.norm();
    if (direction_norm < kMinLineDirectionNorm) {
        return std::numeric_limits<float>::infinity();
    }

    return moment_c.norm() / direction_norm;
}

bool GeometricTools::LinePointFromBackprojectedRay(
    const Eigen::Matrix<float, 6, 1> &line_c, const Eigen::Vector3f &ray_c,
    Eigen::Vector3f &point_c) {
    return LinePointFromBackprojectedRayImpl(line_c, ray_c, point_c,
                                             kMinLineDirectionNorm);
}

bool GeometricTools::LinePointFromBackprojectedRay(
    const Eigen::Matrix<double, 6, 1> &line_c, const Eigen::Vector3d &ray_c,
    Eigen::Vector3d &point_c) {
    return LinePointFromBackprojectedRayImpl(line_c, ray_c, point_c,
                                             kMinLineNormDouble);
}

bool GeometricTools::IsLineDepthPositive(
    const Eigen::Matrix<float, 6, 1> &line_w, GeometricCamera *camera,
    const Sophus::SE3f &Tcw, const Eigen::Vector4f &line_obs) {
    if (camera == nullptr) {
        return false;
    }

    Eigen::Matrix<float, 6, 1> line_c;
    TransformLineToCamera(line_w, Tcw, line_c);

    const Eigen::Vector3f ray_c1 =
        camera->unprojectEig(cv::Point2f(line_obs(0), line_obs(1)));
    const Eigen::Vector3f ray_c2 =
        camera->unprojectEig(cv::Point2f(line_obs(2), line_obs(3)));

    Eigen::Vector3f point_c1;
    Eigen::Vector3f point_c2;
    if (!LinePointFromBackprojectedRay(line_c, ray_c1, point_c1) ||
        !LinePointFromBackprojectedRay(line_c, ray_c2, point_c2)) {
        return false;
    }

    return point_c1.z() > kMinDepth && point_c2.z() > kMinDepth;
}

bool GeometricTools::ComputeLineReprojectionError(
    const Eigen::Matrix<float, 6, 1> &line_w, GeometricCamera *camera,
    const Sophus::SE3f &Tcw, const Eigen::Vector4f &line_obs,
    Eigen::Vector2f &errors) {
    if (camera == nullptr) {
        return false;
    }

    Eigen::Matrix<float, 6, 1> line_c;
    TransformLineToCamera(line_w, Tcw, line_c);

    const Eigen::Vector3f line_2d = camera->projectLine(line_c);
    const float norm = std::hypot(line_2d(0), line_2d(1));
    if (norm < kMinLineDirectionNorm || !line_2d.allFinite()) {
        return false;
    }

    errors(0) =
        (line_2d(0) * line_obs(0) + line_2d(1) * line_obs(1) + line_2d(2)) /
        norm;
    errors(1) =
        (line_2d(0) * line_obs(2) + line_2d(1) * line_obs(3) + line_2d(2)) /
        norm;
    return errors.allFinite();
}

} // namespace XPLSLAM
