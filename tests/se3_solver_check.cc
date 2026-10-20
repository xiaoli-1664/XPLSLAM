#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#define protected public
#include "SE3Solver.h"
#undef protected

#include "camera_models/pinhole.h"

namespace {

bool NearlyEqual(const Eigen::Matrix3f &a, const Eigen::Matrix3f &b,
                 float tolerance) {
    return (a - b).norm() <= tolerance;
}

bool NearlyEqual(const Eigen::Vector3f &a, const Eigen::Vector3f &b,
                 float tolerance) {
    return (a - b).norm() <= tolerance;
}

} // namespace

int main() {
    using namespace XPLSLAM;

    Pinhole camera({500.0f, 500.0f, 320.0f, 240.0f});

    auto kf1 = std::make_shared<KeyFrame>();
    auto kf2 = std::make_shared<KeyFrame>();
    kf1->mnCam = 1;
    kf2->mnCam = 1;
    kf1->mvpCameras = {&camera};
    kf2->mvpCameras = {&camera};
    kf1->mvTbc = {Sophus::SE3f()};
    kf2->mvTbc = {Sophus::SE3f()};

    const Eigen::Matrix3f R12 =
        (Eigen::AngleAxisf(0.25f, Eigen::Vector3f::UnitZ()) *
         Eigen::AngleAxisf(-0.17f, Eigen::Vector3f::UnitY()) *
         Eigen::AngleAxisf(0.11f, Eigen::Vector3f::UnitX()))
            .toRotationMatrix();
    const Eigen::Vector3f t12(0.35f, -0.18f, 0.42f);

    const std::vector<Eigen::Vector3f> points1 = {
        {0.0f, 0.0f, 4.0f},     {1.0f, 0.2f, 4.7f},
        {-0.6f, 0.9f, 5.1f},    {0.4f, -1.1f, 5.4f},
        {1.3f, 1.0f, 6.0f},     {-1.2f, -0.5f, 4.9f},
    };

    SE3Solver solver;
    solver.mpKF1 = kf1;
    solver.mpKF2 = kf2;
    solver.mnCam = 1;
    solver.vpCamera1 = {&camera};
    solver.vpCamera2 = {&camera};
    solver.mvTcb = {Sophus::SE3f()};
    solver.mvMPCam.assign(points1.size(), {0, 0});
    solver.mvpMapPoints1.assign(points1.size(), nullptr);
    solver.mvpMapLines1.clear();
    solver.mvX3Db1.reserve(points1.size());
    solver.mvX3Db2.reserve(points1.size());
    solver.mvP1im1.reserve(points1.size());
    solver.mvP2im2.reserve(points1.size());
    solver.mvAllIndices.reserve(points1.size());

    for (size_t i = 0; i < points1.size(); ++i) {
        const Eigen::Vector3f &x1 = points1[i];
        const Eigen::Vector3f x2 = R12.transpose() * (x1 - t12);

        solver.mvX3Db1.push_back(x1);
        solver.mvX3Db2.push_back(x2);
        solver.mvP1im1.push_back(camera.project(x1));
        solver.mvP2im2.push_back(camera.project(x2));
        solver.mvAllIndices.push_back(i);
    }

    solver.SetRansacParameters(0.99, 3, 100);

    bool no_more = false;
    bool converged = false;
    int inliers = 0;
    const Eigen::Matrix4f T12_est =
        solver.iterate(100, no_more, inliers, converged);

    const Eigen::Matrix3f R12_est = T12_est.block<3, 3>(0, 0);
    const Eigen::Vector3f t12_est = T12_est.block<3, 1>(0, 3);

    if (!converged) {
        std::cerr << "SE3Solver did not converge" << std::endl;
        return 1;
    }
    if (inliers != static_cast<int>(points1.size())) {
        std::cerr << "Expected " << points1.size() << " inliers, got "
                  << inliers << std::endl;
        return 1;
    }
    if (!NearlyEqual(R12_est, R12, 1e-4f)) {
        std::cerr << "Rotation mismatch\nExpected:\n"
                  << R12 << "\nActual:\n"
                  << R12_est << std::endl;
        return 1;
    }
    if (!NearlyEqual(t12_est, t12, 1e-4f)) {
        std::cerr << "Translation mismatch\nExpected: " << t12.transpose()
                  << "\nActual: " << t12_est.transpose() << std::endl;
        return 1;
    }

    return 0;
}
