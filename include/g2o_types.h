#pragma once

#include "g2o/core/base_binary_edge.h"
#include "g2o/core/base_multi_edge.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/core/base_vertex.h"
#include "g2o/core/eigen_types.h"
#include "g2o/types/sba/types_sba.h"

#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Memory.h>
#include <exception>
#include <opencv2/core/core.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "frame.h"
#include "keyframe.h"
#include "line_observation_covariance.h"
#include "se3quat.h"
#include "vertex_se3_expmap.h"
#include "vertex_se3.h"

namespace XPLSLAM {

class KeyFrame;
class Frame;
class GeometricCamera;

using Vector6d = Eigen::Matrix<double, 6, 1>;

Eigen::Matrix3d ExpSO3(const double x, const double y, const double z);
Eigen::Matrix3d ExpSO3(const Eigen::Vector3d &w);
Eigen::Matrix2d ExpSO2(const double phi);

Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R);

Eigen::Matrix3d InverseRightJacobianSO3(const Eigen::Vector3d &v);
Eigen::Matrix3d RightJacobianSO3(const Eigen::Vector3d &v);
Eigen::Matrix3d RightJacobianSO3(const double x, const double y,
                                 const double z);

Eigen::Matrix3d Skew(const Eigen::Vector3d &w);
Eigen::Matrix3d InverseRightJacobianSO3(const double x, const double y,
                                        const double z);

Eigen::Vector3d Line2DJac(const Eigen::Vector3d &l, const Eigen::Vector2d &x);

double Line2DError(const Eigen::Vector3d &l, const Eigen::Vector2d &x);

class LineEndpointCovarianceHolder {
  public:
    bool setEndpointCovariance(const Eigen::Matrix4d &covariance) {
        has_endpoint_covariance_ =
            PackLineEndpointCovariance(covariance, endpoint_covariance_);
        return has_endpoint_covariance_;
    }

    void setEndpointCovariance(
        const PackedLineEndpointCovariance &covariance) {
        endpoint_covariance_ = covariance;
        has_endpoint_covariance_ = true;
    }

    bool hasEndpointCovariance() const { return has_endpoint_covariance_; }

  protected:
    Eigen::Matrix2d lineResidualInformation(
        const Eigen::Vector3d &line_2d) const {
        return LineResidualInformation(line_2d, endpoint_covariance_,
                                       has_endpoint_covariance_);
    }

  private:
    PackedLineEndpointCovariance endpoint_covariance_{};
    bool has_endpoint_covariance_ = false;
};

template <typename T = double>
Eigen::Matrix<T, 3, 3> NormalizeRotation(const Eigen::Matrix<T, 3, 3> &R) {
    Eigen::JacobiSVD<Eigen::Matrix<T, 3, 3>> svd(R, Eigen::ComputeFullU |
                                                        Eigen::ComputeFullV);
    return svd.matrixU() * svd.matrixV().transpose();
}

class BasePose {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    BasePose()
        : Rwb(Eigen::Matrix3d::Identity()), twb(Eigen::Vector3d::Zero()),
          its(0) {}
    BasePose(KeyFramePtr pKF);
    BasePose(Frame *pF);

    // Split right update: Rwb *= Exp(dtheta), twb += dt_world.
    void Update(const double *pu);

    Eigen::Matrix3d CameraRotation(int cam_id) const {
        return Rcb[cam_id] * Rwb.transpose();
    }
    Eigen::Vector3d CameraTranslation(int cam_id) const {
        return tcb[cam_id] - CameraRotation(cam_id) * twb;
    }
    Sophus::SE3d ToTbw() const {
        return Sophus::SE3d(NormalizeRotation<double>(Rwb), twb).inverse();
    }

    Eigen::Vector2d ProjectPoint(const Eigen::Vector3d &Xw,
                                 const int cam_id) const;
    Eigen::Vector3d ProjectLine(const Eigen::Matrix<double, 6, 1> &line_w,
                                const int cam_id) const;
    bool isPointDepthPositive(const Eigen::Vector3d &Xw,
                              const int cam_id) const;
    bool isLineDepthPositive(const Eigen::Matrix<double, 6, 1> &line_w,
                             const int cam_id,
                             const Eigen::Vector4d &line_obs) const;

  public:
    Eigen::Matrix3d Rwb;
    Eigen::Vector3d twb;

    std::vector<Eigen::Matrix3d> Rcb;
    std::vector<Eigen::Vector3d> tcb;
    std::vector<GeometricCamera *> vpCameras;

    int its;
};

Eigen::Matrix<double, 2, 6>
PointOnlyPoseJacobian(const BasePose &pose, const Eigen::Vector3d &Xw,
                      int cam_id);

Eigen::Matrix<double, 2, 6>
LineOnlyPoseJacobian(const BasePose &pose,
                     const Eigen::Matrix<double, 6, 1> &line_w,
                     const Eigen::Vector4d &measurement, int cam_id);

class BaseSE3 {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    BaseSE3()
        : Rbw(Eigen::Matrix3d::Identity()), tbw(Eigen::Vector3d::Zero()),
          its(0) {}
    BaseSE3(const g2o::SE3Quat &Tbw)
        : Rbw(Tbw.rotation()), tbw(Tbw.translation()), its(0) {}

    void Update(const double *pu);

    Eigen::Matrix3d Rbw;
    Eigen::Vector3d tbw;

    int its;
};

class BaseLine {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    BaseLine() {}
    BaseLine(MapLinePtr pML);

    void Update(const double *pu);

  public:
    Eigen::Matrix3d U;
    Eigen::Matrix2d W;

    Eigen::Matrix<double, 6, 1> lineW;
    int its;
};

class VertexPose : public g2o::BaseVertex<6, BasePose> {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexPose() {}
    VertexPose(KeyFramePtr pKF) { setEstimate(BasePose(pKF)); }
    VertexPose(Frame *pF) { setEstimate(BasePose(pF)); }

    virtual bool read(std::istream &is) { return false; }
    virtual bool write(std::ostream &os) const { return false; }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl(const double *update) {
        _estimate.Update(update);
        updateCache();
    }
};

class VertexoSE3 : public g2o::BaseVertex<6, BaseSE3> {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexoSE3() {}
    VertexoSE3(g2o::SE3Quat Tbw) { setEstimate(BaseSE3(Tbw)); }
    virtual bool read(std::istream &is) { return false; }
    virtual bool write(std::ostream &os) const { return false; }

    virtual void setToOriginImpl() { _estimate = BaseSE3(g2o::SE3Quat()); }

    virtual void oplusImpl(const double *update_) {
        _estimate.Update(update_);
        updateCache();
    }

    Eigen::Vector2d ProjectPoint(const Eigen::Vector3d &Xw, const int cam_id,
                                 int frame) const;
    Eigen::Vector2d ProjectPointInverse(const Eigen::Vector3d &Xw,
                                        const int cam_id, int frame) const;
    Eigen::Vector3d ProjectLine(const Eigen::Matrix<double, 6, 1> &line_w,
                                const int cam_id, int frame) const;
    Eigen::Vector3d
    ProjectLineInverse(const Eigen::Matrix<double, 6, 1> &line_w,
                       const int cam_id, int frame) const;

    std::vector<GeometricCamera *> vpCameras1;
    std::vector<GeometricCamera *> vpCameras2;
    std::vector<Sophus::SE3d> vTcb;
};

class VertexLine : public g2o::BaseVertex<4, BaseLine> {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexLine() {}
    VertexLine(MapLinePtr pML) { setEstimate(BaseLine(pML)); }

    virtual bool read(std::istream &is) { return false; }
    virtual bool write(std::ostream &os) const { return false; }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl(const double *update) {
        _estimate.Update(update);
        updateCache();
    }
};

class EdgeProjectPointOnlyPose
    : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeProjectPointOnlyPose(int _cam_id) : cam_id(_cam_id) {}

    bool read(std::istream &is) { return false; }
    bool write(std::ostream &os) const { return false; }

    void computeError() {
        const VertexPose *vPose = static_cast<const VertexPose *>(_vertices[0]);
        Eigen::Vector2d obs(_measurement);
        _error = obs - vPose->estimate().ProjectPoint(Xw, cam_id);
    }

    bool isDepthPositive() {
        const VertexPose *vPose = static_cast<const VertexPose *>(_vertices[0]);
        return vPose->estimate().isPointDepthPositive(Xw, cam_id);
    }

    virtual void linearizeOplus();

  public:
    int cam_id;
    Eigen::Vector3d Xw;
};

class EdgeProjectPoint
    : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexPointXYZ,
                                 VertexPose> {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeProjectPoint(int _cam_id) : cam_id(_cam_id) {}

    bool read(std::istream &is) { return false; }
    bool write(std::ostream &os) const { return false; }

    void computeError() {
        const g2o::VertexPointXYZ *vPoint =
            static_cast<const g2o::VertexPointXYZ *>(_vertices[0]);
        const VertexPose *vPose = static_cast<const VertexPose *>(_vertices[1]);
        Eigen::Vector2d obs(_measurement);
        _error =
            obs - vPose->estimate().ProjectPoint(vPoint->estimate(), cam_id);
    }

    bool isDepthPositive() {
        const g2o::VertexPointXYZ *vPoint =
            static_cast<const g2o::VertexPointXYZ *>(_vertices[0]);
        const VertexPose *vPose = static_cast<const VertexPose *>(_vertices[1]);
        return vPose->estimate().isPointDepthPositive(vPoint->estimate(),
                                                      cam_id);
    }

    virtual void linearizeOplus();

  public:
    int cam_id;
};

class EdgeProjectLineOnlyPose
    : public g2o::BaseUnaryEdge<2, Eigen::Vector4d, VertexPose>,
      public LineEndpointCovarianceHolder {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeProjectLineOnlyPose(int _cam_id) : cam_id(_cam_id) {}

    bool read(std::istream &is) { return false; }
    bool write(std::ostream &os) const { return false; }

    void computeError() {
        const VertexPose *vPose = static_cast<const VertexPose *>(_vertices[0]);
        Eigen::Vector4d obs(_measurement);
        Eigen::Vector3d l = vPose->estimate().ProjectLine(line_w, cam_id);
        if (hasEndpointCovariance()) {
            setInformation(lineResidualInformation(l));
        }
        _error[0] = Line2DError(l, obs.head<2>());
        _error[1] = Line2DError(l, obs.tail<2>());
    }

    bool isDepthPositive() {
        const VertexPose *vPose = static_cast<const VertexPose *>(_vertices[0]);
        return vPose->estimate().isLineDepthPositive(line_w, cam_id,
                                                     _measurement);
    }

    virtual void linearizeOplus();

  public:
    int cam_id;
    Eigen::Matrix<double, 6, 1> line_w;
};

class EdgeProjectLine
    : public g2o::BaseBinaryEdge<2, Eigen::Vector4d, VertexLine, VertexPose>,
      public LineEndpointCovarianceHolder {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeProjectLine(int _cam_id) : cam_id(_cam_id) {}

    bool read(std::istream &is) { return false; }
    bool write(std::ostream &os) const { return false; }

    void computeError() {
        const VertexLine *vLine = static_cast<const VertexLine *>(_vertices[0]);
        const VertexPose *vPose = static_cast<const VertexPose *>(_vertices[1]);
        Eigen::Vector4d obs(_measurement);
        Eigen::Vector3d l =
            vPose->estimate().ProjectLine(vLine->estimate().lineW, cam_id);
        if (hasEndpointCovariance()) {
            setInformation(lineResidualInformation(l));
        }
        _error[0] = Line2DError(l, obs.head<2>());
        _error[1] = Line2DError(l, obs.tail<2>());
    }

    bool isDepthPositive() {
        const VertexLine *vLine = static_cast<const VertexLine *>(_vertices[0]);
        const VertexPose *vPose = static_cast<const VertexPose *>(_vertices[1]);
        return vPose->estimate().isLineDepthPositive(vLine->estimate().lineW,
                                                     cam_id, _measurement);
    }

    virtual void linearizeOplus();

  public:
    int cam_id;
};

class EdgeSE3ProjectPoint
    : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexPointXYZ,
                                 VertexoSE3> {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3ProjectPoint(int _cam_id) : cam_id(_cam_id) {}

    bool read(std::istream &is) { return false; }
    bool write(std::ostream &os) const { return false; }

    void computeError() {
        const g2o::VertexPointXYZ *vPoint =
            static_cast<const g2o::VertexPointXYZ *>(_vertices[0]);
        const VertexoSE3 *vSE3 = static_cast<const VertexoSE3 *>(_vertices[1]);
        Eigen::Vector2d obs(_measurement);
        _error = obs - vSE3->ProjectPoint(vPoint->estimate(), cam_id, 1);
    }

    virtual void linearizeOplus();

  public:
    int cam_id;
};

class EdgeInverseSE3ProjectPoint
    : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexPointXYZ,
                                 VertexoSE3> {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeInverseSE3ProjectPoint(int _cam_id) : cam_id(_cam_id) {}

    bool read(std::istream &is) { return false; }
    bool write(std::ostream &os) const { return false; }

    void computeError() {
        const g2o::VertexPointXYZ *vPoint =
            static_cast<const g2o::VertexPointXYZ *>(_vertices[0]);
        const VertexoSE3 *vSE3 = static_cast<const VertexoSE3 *>(_vertices[1]);
        Eigen::Vector2d obs(_measurement);
        _error = obs - vSE3->ProjectPointInverse(vPoint->estimate(), cam_id, 2);
    }

    virtual void linearizeOplus();

  public:
    int cam_id;
};

class EdgeSE3ProjectLine
    : public g2o::BaseBinaryEdge<2, Eigen::Vector4d, VertexLine, VertexoSE3>,
      public LineEndpointCovarianceHolder {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3ProjectLine(int _cam_id) : cam_id(_cam_id) {}

    bool read(std::istream &is) { return false; }
    bool write(std::ostream &os) const { return false; }

    void computeError() {
        const VertexLine *vLine = static_cast<const VertexLine *>(_vertices[0]);
        const VertexoSE3 *vSE3 = static_cast<const VertexoSE3 *>(_vertices[1]);
        Eigen::Vector4d obs(_measurement);
        Eigen::Vector3d l =
            vSE3->ProjectLine(vLine->estimate().lineW, cam_id, 1);
        if (hasEndpointCovariance()) {
            setInformation(lineResidualInformation(l));
        }
        _error[0] = Line2DError(l, obs.head<2>());
        _error[1] = Line2DError(l, obs.tail<2>());
    }

    virtual void linearizeOplus();

    int cam_id;
};

class EdgeInverseSE3ProjectLine
    : public g2o::BaseBinaryEdge<2, Eigen::Vector4d, VertexLine, VertexoSE3>,
      public LineEndpointCovarianceHolder {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeInverseSE3ProjectLine(int _cam_id) : cam_id(_cam_id) {}

    bool read(std::istream &is) { return false; }
    bool write(std::ostream &os) const { return false; }

    void computeError() {
        const VertexLine *vLine = static_cast<const VertexLine *>(_vertices[0]);
        const VertexoSE3 *vSE3 = static_cast<const VertexoSE3 *>(_vertices[1]);
        Eigen::Vector4d obs(_measurement);
        Eigen::Vector3d l =
            vSE3->ProjectLineInverse(vLine->estimate().lineW, cam_id, 2);
        if (hasEndpointCovariance()) {
            setInformation(lineResidualInformation(l));
        }
        _error[0] = Line2DError(l, obs.head<2>());
        _error[1] = Line2DError(l, obs.tail<2>());
    }

    virtual void linearizeOplus();

  public:
    int cam_id;
};

// Absolute body-to-world pose; tangent order is [rotation, world translation].
class VertexPoseGraph : public g2o::BaseVertex<6, g2o::SE3Quat> {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    bool read(std::istream &) override { return false; }
    bool write(std::ostream &) const override { return false; }
    void setToOriginImpl() override { setEstimate(g2o::SE3Quat()); }
    void oplusImpl(const double *update) override {
        const Eigen::Vector3d rotation(update[0], update[1], update[2]);
        const Eigen::Vector3d translation(update[3], update[4], update[5]);
        setEstimate(g2o::SE3Quat(
            estimate().rotation().toRotationMatrix() * ExpSO3(rotation),
            estimate().translation() + translation));
    }
};

class EdgeEssentialGraphSE3
    : public g2o::BaseBinaryEdge<6, g2o::SE3Quat, VertexPoseGraph,
                                 VertexPoseGraph> {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool read(std::istream &is) { return false; }
    bool write(std::ostream &os) const { return false; }

    void computeError() {
        const auto *v1 =
            static_cast<const VertexPoseGraph *>(_vertices[0]);
        const auto *v2 =
            static_cast<const VertexPoseGraph *>(_vertices[1]);
        _error = (_measurement * v1->estimate().inverse() * v2->estimate())
                     .log();
    }

    virtual void linearizeOplus();
};

}; // namespace XPLSLAM
