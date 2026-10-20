#pragma once

#include <assert.h>

#include "geometric_camera.h"

namespace XPLSLAM {
class Pinhole : public GeometricCamera {

    friend class boost::serialization::access;

    template <class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar &boost::serialization::base_object<GeometricCamera>(*this);
    }

  public:
    Pinhole() {
        mvParameters.resize(4);
        mnId = nNextId++;
        mnType = CAM_PINHOLE;
    }
    Pinhole(const std::vector<float> _vParameters)
        : GeometricCamera(_vParameters) {
        assert(mvParameters.size() == 4);
        mnId = nNextId++;
        mnType = CAM_PINHOLE;
    }

    Pinhole(const Pinhole &pinhole) : GeometricCamera(pinhole.mvParameters) {
        assert(mvParameters.size() == 4);
        mnId = nNextId++;
        mnType = CAM_PINHOLE;
    }

    ~Pinhole() {}

    cv::Point2f project(const cv::Point3f &p3D) override;
    Eigen::Vector2d project(const Eigen::Vector3d &v3D) override;
    Eigen::Vector2f project(const Eigen::Vector3f &v3D) override;
    Eigen::Vector3f
    projectLine(const Eigen::Matrix<float, 6, 1> &line3D) override;
    Eigen::Vector3d
    projectLine(const Eigen::Matrix<double, 6, 1> &line3D) override;
    Eigen::Vector2f projectMat(const cv::Point3f &p3D) override;

    float uncertainty2(const Eigen::Matrix<double, 2, 1> &p2D);

    Eigen::Vector3f unprojectEig(const cv::Point2f &p2D);
    cv::Point3f unproject(const cv::Point2f &p2D);
    Eigen::Vector3d unprojectEig(const Eigen::Vector2d &p2D) override;

    Eigen::Matrix<double, 2, 3> projectJac(const Eigen::Vector3d &v3D);

    cv::Mat toK() const;
    Eigen::Matrix3f toK_() const;
    Eigen::Matrix3d toPL_() const;

    bool epipolarConstrain(GeometricCamera *camera2, const cv::Point2f &kp1,
                           const cv::Point2f &kp2, const Eigen::Matrix3f &R12,
                           const Eigen::Vector3f &t12, const float sigmaLevel,
                           const float unc) override;

    bool matchAndtriangulate(const cv::Point2f &kp1, const cv::Point2f &kp2,
                             const GeometricCamera &otherCamera,
                             Sophus::SE3f &Tcw1, Sophus::SE3f &Tcw2,
                             const float sigmaLevel1, const float sigmaLevel2,
                             Eigen::Vector3f &x3Dtriangulated) override {
        return false;
    }

    friend std::ostream &operator<<(std::ostream &os, const Pinhole &ph);
    friend std::istream &operator>>(std::istream &os, Pinhole &ph);

    bool IsEqual(const GeometricCamera &camera) const;

  private:
};
} // namespace XPLSLAM
