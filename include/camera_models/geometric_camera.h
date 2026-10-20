#pragma once

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/serialization/access.hpp>
#include <boost/serialization/assume_abstract.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

#include <sophus/se3.hpp>

#include <Eigen/Geometry>

#include "converter.h"
#include "geometric_tools.h"

namespace XPLSLAM {
class GeometricCamera {

    friend class boost::serialization::access;

    template <class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & mnId;
        ar & mnType;
        ar & mvParameters;
    }

  public:
    GeometricCamera() {}
    GeometricCamera(const std::vector<float> &_vParameters)
        : mvParameters(_vParameters) {}
    virtual ~GeometricCamera() = default;

    virtual cv::Point2f project(const cv::Point3f &p3D) = 0;
    virtual Eigen::Vector2d project(const Eigen::Vector3d &v3D) = 0;
    virtual Eigen::Vector2f project(const Eigen::Vector3f &v3D) = 0;
    virtual Eigen::Vector3d
    projectLine(const Eigen::Matrix<double, 6, 1> &line3D) = 0;
    virtual Eigen::Vector3f
    projectLine(const Eigen::Matrix<float, 6, 1> &line3D) = 0;
    virtual Eigen::Vector2f projectMat(const cv::Point3f &p3D) = 0;

    virtual float uncertainty2(const Eigen::Matrix<double, 2, 1> &p2D) = 0;

    virtual Eigen::Vector3f unprojectEig(const cv::Point2f &p2D) = 0;
    virtual Eigen::Vector3d unprojectEig(const Eigen::Vector2d &p2D) = 0;
    virtual cv::Point3f unproject(const cv::Point2f &p2D) = 0;

    virtual Eigen::Matrix<double, 2, 3>
    projectJac(const Eigen::Vector3d &v3D) = 0;

    virtual cv::Mat toK() const = 0;
    virtual Eigen::Matrix3f toK_() const = 0;
    virtual Eigen::Matrix3d toPL_() const = 0;

    virtual bool epipolarConstrain(GeometricCamera *otherCamera,
                                   const cv::Point2f &kp1,
                                   const cv::Point2f &kp2,
                                   const Eigen::Matrix3f &R12,
                                   const Eigen::Vector3f &t12,
                                   const float sigmaLevel, const float unc) = 0;

    float getParameter(const int i) const { return mvParameters[i]; }
    void setParameter(const float p, const size_t i) { mvParameters[i] = p; }

    size_t size() const { return mvParameters.size(); }

    virtual bool matchAndtriangulate(const cv::Point2f &kp1,
                                     const cv::Point2f &kp2,
                                     const GeometricCamera &otherCamera,
                                     Sophus::SE3f &Tcw1, Sophus::SE3f &Tcw2,
                                     const float sigmaLevel1,
                                     const float sigmaLevel2,
                                     Eigen::Vector3f &x3Dtriangulated) = 0;

    unsigned int GetId() const { return mnId; }

    unsigned int GetType() const { return mnType; }

    const static unsigned int CAM_PINHOLE = 0;
    const static unsigned int CAM_FISHEYE = 1;

    static long unsigned int nNextId;

  protected:
    std::vector<float> mvParameters;

    unsigned int mnId;

    unsigned int mnType;
};
} // namespace XPLSLAM
