#include <boost/serialization/export.hpp>

#include "pinhole.h"

namespace XPLSLAM {

long unsigned int GeometricCamera::nNextId = 0;

cv::Point2f Pinhole::project(const cv::Point3f &p3D) {
    return cv::Point2f(mvParameters[0] * p3D.x / p3D.z + mvParameters[2],
                       mvParameters[1] * p3D.y / p3D.z + mvParameters[3]);
}

Eigen::Vector2d Pinhole::project(const Eigen::Vector3d &v3D) {
    Eigen::Vector2d res;
    res[0] = mvParameters[0] * v3D[0] / v3D[2] + mvParameters[2];
    res[1] = mvParameters[1] * v3D[1] / v3D[2] + mvParameters[3];

    return res;
}

Eigen::Vector2f Pinhole::project(const Eigen::Vector3f &v3D) {
    Eigen::Vector2f res;
    res[0] = mvParameters[0] * v3D[0] / v3D[2] + mvParameters[2];
    res[1] = mvParameters[1] * v3D[1] / v3D[2] + mvParameters[3];

    return res;
}

Eigen::Vector3d
Pinhole::projectLine(const Eigen::Matrix<double, 6, 1> &line3D) {
    Eigen::Matrix3d P;
    P << mvParameters[1], 0., 0., 0., mvParameters[0], 0.,
        -mvParameters[1] * mvParameters[2], -mvParameters[0] * mvParameters[3],
        mvParameters[0] * mvParameters[1];
    return P * line3D.head<3>();
}

Eigen::Vector3f Pinhole::projectLine(const Eigen::Matrix<float, 6, 1> &line3D) {
    Eigen::Matrix3f P;
    P << mvParameters[1], 0.f, 0.f, 0.f, mvParameters[0], 0.f,
        -mvParameters[1] * mvParameters[2], -mvParameters[0] * mvParameters[3],
        mvParameters[0] * mvParameters[1];
    return P * line3D.head<3>();
}

Eigen::Vector2f Pinhole::projectMat(const cv::Point3f &p3D) {
    cv::Point2f point = this->project(p3D);
    return Eigen::Vector2f(point.x, point.y);
}

float Pinhole::uncertainty2(const Eigen::Matrix<double, 2, 1> &p2D) {
    return 1.0;
}

Eigen::Vector3f Pinhole::unprojectEig(const cv::Point2f &p2D) {
    return Eigen::Vector3f((p2D.x - mvParameters[2]) / mvParameters[0],
                           (p2D.y - mvParameters[3]) / mvParameters[1], 1.f);
}

Eigen::Vector3d Pinhole::unprojectEig(const Eigen::Vector2d &p2D) {
    return Eigen::Vector3d((p2D.x() - mvParameters[2]) / mvParameters[0],
                           (p2D.y() - mvParameters[3]) / mvParameters[1], 1.0);
}

cv::Point3f Pinhole::unproject(const cv::Point2f &p2D) {
    return cv::Point3f((p2D.x - mvParameters[2]) / mvParameters[0],
                       (p2D.y - mvParameters[3]) / mvParameters[1], 1.f);
}

Eigen::Matrix<double, 2, 3> Pinhole::projectJac(const Eigen::Vector3d &v3D) {
    Eigen::Matrix<double, 2, 3> Jac;
    Jac(0, 0) = mvParameters[0] / v3D[2];
    Jac(0, 1) = 0.f;
    Jac(0, 2) = -mvParameters[0] * v3D[0] / (v3D[2] * v3D[2]);
    Jac(1, 0) = 0.f;
    Jac(1, 1) = mvParameters[1] / v3D[2];
    Jac(1, 2) = -mvParameters[1] * v3D[1] / (v3D[2] * v3D[2]);

    return Jac;
}

cv::Mat Pinhole::toK() const {
    cv::Mat K = (cv::Mat_<float>(3, 3) << mvParameters[0], 0.f, mvParameters[2],
                 0.f, mvParameters[1], mvParameters[3], 0.f, 0.f, 1.f);
    return K;
}

Eigen::Matrix3f Pinhole::toK_() const {
    Eigen::Matrix3f K;
    K << mvParameters[0], 0.f, mvParameters[2], 0.f, mvParameters[1],
        mvParameters[3], 0.f, 0.f, 1.f;
    return K;
}

Eigen::Matrix3d Pinhole::toPL_() const {
    Eigen::Matrix3d PL;
    PL << mvParameters[1], 0., 0., 0., mvParameters[0], 0.,
        -mvParameters[1] * mvParameters[2], -mvParameters[0] * mvParameters[3],
        mvParameters[0] * mvParameters[1];
    return PL;
}

bool Pinhole::epipolarConstrain(GeometricCamera *camera2,
                                const cv::Point2f &kp1, const cv::Point2f &kp2,
                                const Eigen::Matrix3f &R12,
                                const Eigen::Vector3f &t12,
                                const float sigmaLevel, const float unc) {
    // Compute Fundamental Matrix
    Eigen::Matrix3f t12x = Sophus::SO3f::hat(t12);
    Eigen::Matrix3f K1 = this->toK_();
    Eigen::Matrix3f K2 = camera2->toK_();
    Eigen::Matrix3f F12 = K1.transpose().inverse() * t12x * R12 * K2.inverse();

    // Epipolar line in second image l = x1'F12 = [a b c]
    const float a = kp1.x * F12(0, 0) + kp1.y * F12(1, 0) + F12(2, 0);
    const float b = kp1.x * F12(0, 1) + kp1.y * F12(1, 1) + F12(2, 1);
    const float c = kp1.x * F12(0, 2) + kp1.y * F12(1, 2) + F12(2, 2);

    const float num = a * kp2.x + b * kp2.y + c;

    const float den = a * a + b * b;

    if (den == 0)
        return false;

    const float dsqr = num * num / den;

    return dsqr < 3.84 * unc;
}

std::ostream &operator<<(std::ostream &os, const Pinhole &ph) {
    os << ph.mvParameters[0] << " " << ph.mvParameters[1] << " "
       << ph.mvParameters[2] << " " << ph.mvParameters[3];
    return os;
}

std::istream &operator>>(std::istream &is, Pinhole &ph) {
    float nextParam;
    for (size_t i = 0; i < 4; i++) {
        assert(is.good()); // Make sure the input stream is good
        is >> nextParam;
        ph.mvParameters[i] = nextParam;
    }
    return is;
}

bool Pinhole::IsEqual(const GeometricCamera &camera) const {
    if (camera.GetType() != GeometricCamera::CAM_PINHOLE)
        return false;

    const auto &pinhole_camera = static_cast<const Pinhole &>(camera);

    if (size() != pinhole_camera.size())
        return false;

    bool is_same_camera = true;
    for (size_t i = 0; i < size(); ++i) {
        if (abs(mvParameters[i] - pinhole_camera.getParameter(i)) > 1e-6) {
            is_same_camera = false;
            break;
        }
    }
    return is_same_camera;
}
} // namespace XPLSLAM
