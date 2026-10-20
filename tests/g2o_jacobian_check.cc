#include "g2o_types.h"
#include "pinhole.h"

#include "g2o/core/jacobian_workspace.h"
#include "g2o/types/slam3d/vertex_pointxyz.h"

#include <Eigen/Core>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

using XPLSLAM::BaseLine;
using XPLSLAM::BasePose;
using XPLSLAM::BaseSE3;
using XPLSLAM::EdgeInverseSE3ProjectLine;
using XPLSLAM::EdgeInverseSE3ProjectPoint;
using XPLSLAM::EdgeEssentialGraphSE3;
using XPLSLAM::EdgeProjectLine;
using XPLSLAM::EdgeProjectLineOnlyPose;
using XPLSLAM::EdgeProjectPoint;
using XPLSLAM::EdgeProjectPointOnlyPose;
using XPLSLAM::EdgeSE3ProjectLine;
using XPLSLAM::EdgeSE3ProjectPoint;
using XPLSLAM::GeometricCamera;
using XPLSLAM::Line2DError;
using XPLSLAM::Line2DJac;
using XPLSLAM::Pinhole;
using XPLSLAM::VertexLine;
using XPLSLAM::VertexoSE3;
using XPLSLAM::VertexPose;

constexpr double kPointJacobianTolerance = 1e-4;
constexpr double kLineJacobianTolerance = 1e-4;
constexpr double kPoseGraphJacobianTolerance = 2e-4;
constexpr double kJacobianRelativeTolerance = 2e-6;
constexpr double kErrorTolerance = 1e-8;
constexpr int kCoveredEdgeTypes = 9;

struct SampleConfig {
    std::string name;
    double rot_limit;
    double trans_limit;
    double xy_limit;
    double z_min;
    double z_max;
    double min_line_length;
};

class NumericEdgeProjectPointOnlyPose
    : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit NumericEdgeProjectPointOnlyPose(int cam_id_in)
        : cam_id(cam_id_in) {}

    bool read(std::istream &) override { return false; }
    bool write(std::ostream &) const override { return false; }

    void computeError() override {
        const auto *vPose = static_cast<const VertexPose *>(_vertices[0]);
        _error = _measurement - vPose->estimate().ProjectPoint(Xw, cam_id);
    }

    int cam_id;
    Eigen::Vector3d Xw;
};

class NumericEdgeProjectPoint
    : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexPointXYZ,
                                 VertexPose> {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit NumericEdgeProjectPoint(int cam_id_in) : cam_id(cam_id_in) {}

    bool read(std::istream &) override { return false; }
    bool write(std::ostream &) const override { return false; }

    void computeError() override {
        const auto *vPoint =
            static_cast<const g2o::VertexPointXYZ *>(_vertices[0]);
        const auto *vPose = static_cast<const VertexPose *>(_vertices[1]);
        _error = _measurement -
                 vPose->estimate().ProjectPoint(vPoint->estimate(), cam_id);
    }

    int cam_id;
};

class NumericEdgeProjectLineOnlyPose
    : public g2o::BaseUnaryEdge<2, Eigen::Vector4d, VertexPose> {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit NumericEdgeProjectLineOnlyPose(int cam_id_in)
        : cam_id(cam_id_in) {}

    bool read(std::istream &) override { return false; }
    bool write(std::ostream &) const override { return false; }

    void computeError() override {
        const auto *vPose = static_cast<const VertexPose *>(_vertices[0]);
        const Eigen::Vector3d line =
            vPose->estimate().ProjectLine(line_w, cam_id);
        _error[0] = Line2DError(line, _measurement.head<2>());
        _error[1] = Line2DError(line, _measurement.tail<2>());
    }

    int cam_id;
    Eigen::Matrix<double, 6, 1> line_w;
};

class NumericEdgeProjectLine
    : public g2o::BaseBinaryEdge<2, Eigen::Vector4d, VertexLine, VertexPose> {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit NumericEdgeProjectLine(int cam_id_in) : cam_id(cam_id_in) {}

    bool read(std::istream &) override { return false; }
    bool write(std::ostream &) const override { return false; }

    void computeError() override {
        const auto *vLine = static_cast<const VertexLine *>(_vertices[0]);
        const auto *vPose = static_cast<const VertexPose *>(_vertices[1]);
        const Eigen::Vector3d line = vPose->estimate().ProjectLine(
            vLine->estimate().lineW, cam_id);
        _error[0] = Line2DError(line, _measurement.head<2>());
        _error[1] = Line2DError(line, _measurement.tail<2>());
    }

    int cam_id;
};

class NumericEdgeSE3ProjectPoint
    : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexPointXYZ,
                                 VertexoSE3> {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    NumericEdgeSE3ProjectPoint(int cam_id_in, int frame_in, bool inverse_in)
        : cam_id(cam_id_in), frame(frame_in), inverse(inverse_in) {}

    bool read(std::istream &) override { return false; }
    bool write(std::ostream &) const override { return false; }

    void computeError() override {
        const auto *vPoint =
            static_cast<const g2o::VertexPointXYZ *>(_vertices[0]);
        const auto *vSE3 = static_cast<const VertexoSE3 *>(_vertices[1]);
        const Eigen::Vector2d projection =
            inverse
                ? vSE3->ProjectPointInverse(vPoint->estimate(), cam_id, frame)
                : vSE3->ProjectPoint(vPoint->estimate(), cam_id, frame);
        _error = _measurement - projection;
    }

    int cam_id;
    int frame;
    bool inverse;
};

class NumericEdgeSE3ProjectLine
    : public g2o::BaseBinaryEdge<2, Eigen::Vector4d, VertexLine, VertexoSE3> {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    NumericEdgeSE3ProjectLine(int cam_id_in, int frame_in, bool inverse_in)
        : cam_id(cam_id_in), frame(frame_in), inverse(inverse_in) {}

    bool read(std::istream &) override { return false; }
    bool write(std::ostream &) const override { return false; }

    void computeError() override {
        const auto *vLine = static_cast<const VertexLine *>(_vertices[0]);
        const auto *vSE3 = static_cast<const VertexoSE3 *>(_vertices[1]);
        const Eigen::Vector3d line =
            inverse ? vSE3->ProjectLineInverse(vLine->estimate().lineW, cam_id,
                                               frame)
                    : vSE3->ProjectLine(vLine->estimate().lineW, cam_id,
                                        frame);
        _error[0] = Line2DError(line, _measurement.head<2>());
        _error[1] = Line2DError(line, _measurement.tail<2>());
    }

    int cam_id;
    int frame;
    bool inverse;
};

class NumericEdgeEssentialGraphSE3
    : public g2o::BaseBinaryEdge<6, g2o::SE3Quat, XPLSLAM::VertexPoseGraph,
                                 XPLSLAM::VertexPoseGraph> {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool read(std::istream &) override { return false; }
    bool write(std::ostream &) const override { return false; }

    void computeError() override {
        const auto *v1 =
            static_cast<const XPLSLAM::VertexPoseGraph *>(_vertices[0]);
        const auto *v2 =
            static_cast<const XPLSLAM::VertexPoseGraph *>(_vertices[1]);
        _error =
            (_measurement * v1->estimate().inverse() * v2->estimate()).log();
    }
};

BasePose MakePose(const std::vector<GeometricCamera *> &cameras,
                  std::mt19937 &rng, const SampleConfig &config) {
    std::uniform_real_distribution<double> rot_dist(-config.rot_limit,
                                                    config.rot_limit);
    std::uniform_real_distribution<double> trans_dist(-config.trans_limit,
                                                      config.trans_limit);

    BasePose pose;
    pose.Rwb =
        XPLSLAM::ExpSO3(Eigen::Vector3d(rot_dist(rng), rot_dist(rng),
                                        rot_dist(rng)));
    pose.twb =
        Eigen::Vector3d(trans_dist(rng), trans_dist(rng), trans_dist(rng));
    pose.Rcb.resize(cameras.size());
    pose.tcb.resize(cameras.size());
    pose.vpCameras.resize(cameras.size());
    for (size_t cam_id = 0; cam_id < cameras.size(); ++cam_id) {
        pose.Rcb[cam_id] =
            XPLSLAM::ExpSO3(Eigen::Vector3d(rot_dist(rng), rot_dist(rng),
                                            rot_dist(rng)));
        pose.tcb[cam_id] =
            Eigen::Vector3d(trans_dist(rng), trans_dist(rng), trans_dist(rng));
        pose.vpCameras[cam_id] = cameras[cam_id];
    }
    pose.its = 0;
    return pose;
}

Eigen::Matrix<double, 6, 1>
MakePluckerLine(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2) {
    Eigen::Matrix<double, 6, 1> line;
    const Eigen::Vector3d direction = p2 - p1;
    line.head<3>() = p1.cross(direction);
    line.tail<3>() = direction;
    return line;
}

BaseLine MakeBaseLine(const Eigen::Matrix<double, 6, 1> &line_w) {
    BaseLine line;
    const Eigen::Vector3d moment = line_w.head<3>();
    const Eigen::Vector3d direction = line_w.tail<3>();
    const double moment_norm = moment.norm();
    const double direction_norm = direction.norm();
    const double md_norm = std::hypot(moment_norm, direction_norm);

    line.U.col(0) = moment / moment_norm;
    line.U.col(1) = direction / direction_norm;
    line.U.col(2) = moment.cross(direction).normalized();
    line.W << moment_norm, -direction_norm, direction_norm, moment_norm;
    line.W /= md_norm;
    line.lineW = line_w / md_norm;
    return line;
}

BaseSE3 MakeSE3(const BasePose &pose) {
    BaseSE3 se3;
    se3.Rbw = pose.Rwb.transpose();
    se3.tbw = -se3.Rbw * pose.twb;
    se3.its = 0;
    return se3;
}

Eigen::Vector4d MakeLineObservation(const Eigen::Vector3d &line_2d) {
    Eigen::Vector4d observation;
    const double a = line_2d[0];
    const double b = line_2d[1];
    const double c = line_2d[2];

    if (std::abs(b) > std::abs(a)) {
        const double x1 = 260.0;
        const double x2 = 380.0;
        observation << x1, -(a * x1 + c) / b, x2, -(a * x2 + c) / b;
    } else {
        const double y1 = 180.0;
        const double y2 = 300.0;
        observation << -(b * y1 + c) / a, y1, -(b * y2 + c) / a, y2;
    }

    return observation;
}

struct Sample {
    BasePose pose;
    BaseLine line;
    Eigen::Vector3d point_w;
    Eigen::Vector4d line_obs;
    Eigen::Vector2d point_obs;
};

Sample MakeSample(const std::vector<GeometricCamera *> &cameras,
                  std::mt19937 &rng, const SampleConfig &config) {
    std::uniform_real_distribution<double> xy_dist(-config.xy_limit,
                                                   config.xy_limit);
    std::uniform_real_distribution<double> z_dist(config.z_min, config.z_max);

    for (int attempt = 0; attempt < 1000; ++attempt) {
        Sample sample;
        sample.pose = MakePose(cameras, rng, config);
        sample.point_w =
            Eigen::Vector3d(xy_dist(rng), xy_dist(rng), z_dist(rng));

        const Eigen::Vector3d line_start(xy_dist(rng), xy_dist(rng),
                                         z_dist(rng));
        const Eigen::Vector3d line_end(xy_dist(rng), xy_dist(rng),
                                       z_dist(rng));
        if ((line_end - line_start).norm() < config.min_line_length) {
            continue;
        }

        bool valid_depth = true;
        for (size_t cam_id = 0; cam_id < cameras.size(); ++cam_id) {
            if (!sample.pose.isPointDepthPositive(sample.point_w, cam_id) ||
                !sample.pose.isPointDepthPositive(line_start, cam_id) ||
                !sample.pose.isPointDepthPositive(line_end, cam_id)) {
                valid_depth = false;
                break;
            }
        }
        if (!valid_depth) {
            continue;
        }

        const Eigen::Matrix<double, 6, 1> plucker =
            MakePluckerLine(line_start, line_end);
        if (plucker.head<3>().norm() < 1e-8 ||
            plucker.tail<3>().norm() < 1e-8 ||
            std::abs(plucker.head<3>().dot(plucker.tail<3>())) > 1e-8) {
            continue;
        }

        sample.line = MakeBaseLine(plucker);
        sample.point_obs = sample.pose.ProjectPoint(sample.point_w, 0);

        const Eigen::Vector3d projected_line =
            sample.pose.ProjectLine(sample.line.lineW, 0);
        if (projected_line.head<2>().norm() < 1e-8 ||
            !projected_line.allFinite() || !sample.point_obs.allFinite()) {
            continue;
        }

        sample.line_obs = MakeLineObservation(projected_line);
        if (!sample.line_obs.allFinite()) {
            continue;
        }

        bool valid_projection = true;
        for (size_t cam_id = 0; cam_id < cameras.size(); ++cam_id) {
            const Eigen::Vector2d point_obs =
                sample.pose.ProjectPoint(sample.point_w, cam_id);
            const Eigen::Vector3d line_obs =
                sample.pose.ProjectLine(sample.line.lineW, cam_id);
            if (!point_obs.allFinite() || !line_obs.allFinite() ||
                line_obs.head<2>().norm() < 1e-8 ||
                !MakeLineObservation(line_obs).allFinite()) {
                valid_projection = false;
                break;
            }
        }
        if (!valid_projection) {
            continue;
        }

        return sample;
    }

    throw std::runtime_error("failed to generate a valid jacobian sample");
}

void PrepareVertex(VertexPose &vertex, const BasePose &pose, int id) {
    vertex.setId(id);
    vertex.setFixed(false);
    vertex.setEstimate(pose);
}

void PrepareVertex(VertexoSE3 &vertex, const BaseSE3 &pose,
                   const BasePose &base_pose,
                   const std::vector<GeometricCamera *> &cameras1,
                   const std::vector<GeometricCamera *> &cameras2, int id) {
    vertex.setId(id);
    vertex.setFixed(false);
    vertex.setEstimate(pose);
    vertex.vpCameras1.resize(cameras1.size());
    vertex.vpCameras2.resize(cameras2.size());
    vertex.vTcb.resize(base_pose.Rcb.size());
    for (size_t cam_id = 0; cam_id < base_pose.Rcb.size(); ++cam_id) {
        vertex.vpCameras1[cam_id] = cameras1[cam_id];
        vertex.vpCameras2[cam_id] = cameras2[cam_id];
        vertex.vTcb[cam_id] =
            Sophus::SE3d(base_pose.Rcb[cam_id], base_pose.tcb[cam_id]);
    }
}

void PrepareVertex(VertexLine &vertex, const BaseLine &line, int id) {
    vertex.setId(id);
    vertex.setFixed(false);
    vertex.setEstimate(line);
}

void PrepareVertex(g2o::VertexPointXYZ &vertex, const Eigen::Vector3d &point,
                   int id) {
    vertex.setId(id);
    vertex.setFixed(false);
    vertex.setEstimate(point);
}

template <typename EdgeType>
void AllocateWorkspace(const EdgeType &edge, g2o::JacobianWorkspace &workspace) {
    workspace.updateSize(&edge, true);
    workspace.allocate();
    workspace.setZero();
}

template <typename EdgeType>
void LinearizeUnaryPointPose(EdgeType &edge,
                             g2o::JacobianWorkspace &workspace) {
    AllocateWorkspace(edge, workspace);
    using Base = g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose>;
    static_cast<Base &>(edge).linearizeOplus(workspace);
}

template <typename EdgeType>
void LinearizeUnaryLinePose(EdgeType &edge,
                            g2o::JacobianWorkspace &workspace) {
    AllocateWorkspace(edge, workspace);
    using Base = g2o::BaseUnaryEdge<2, Eigen::Vector4d, VertexPose>;
    static_cast<Base &>(edge).linearizeOplus(workspace);
}

template <typename EdgeType>
void LinearizeBinaryPoint(EdgeType &edge, g2o::JacobianWorkspace &workspace) {
    AllocateWorkspace(edge, workspace);
    using Base =
        g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexPointXYZ,
                            VertexPose>;
    static_cast<Base &>(edge).linearizeOplus(workspace);
}

template <typename EdgeType>
void LinearizeBinaryLine(EdgeType &edge, g2o::JacobianWorkspace &workspace) {
    AllocateWorkspace(edge, workspace);
    using Base =
        g2o::BaseBinaryEdge<2, Eigen::Vector4d, VertexLine, VertexPose>;
    static_cast<Base &>(edge).linearizeOplus(workspace);
}

template <typename EdgeType>
void LinearizeBinarySE3Point(EdgeType &edge,
                             g2o::JacobianWorkspace &workspace) {
    AllocateWorkspace(edge, workspace);
    using Base =
        g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexPointXYZ,
                            VertexoSE3>;
    static_cast<Base &>(edge).linearizeOplus(workspace);
}

template <typename EdgeType>
void LinearizeBinarySE3Line(EdgeType &edge,
                            g2o::JacobianWorkspace &workspace) {
    AllocateWorkspace(edge, workspace);
    using Base =
        g2o::BaseBinaryEdge<2, Eigen::Vector4d, VertexLine, VertexoSE3>;
    static_cast<Base &>(edge).linearizeOplus(workspace);
}

template <typename EdgeType>
void LinearizeBinaryEssentialGraph(EdgeType &edge,
                                   g2o::JacobianWorkspace &workspace) {
    AllocateWorkspace(edge, workspace);
    using Base =
        g2o::BaseBinaryEdge<6, g2o::SE3Quat, XPLSLAM::VertexPoseGraph,
                            XPLSLAM::VertexPoseGraph>;
    static_cast<Base &>(edge).linearizeOplus(workspace);
}

bool CheckErrorZero(const std::string &name, const Eigen::VectorXd &error) {
    const double max_abs = error.cwiseAbs().maxCoeff();
    if (!error.allFinite() || max_abs > kErrorTolerance) {
        std::cerr << name << " error check failed, max_abs=" << max_abs
                  << "\nerror:\n"
                  << error.transpose() << '\n';
        return false;
    }
    return true;
}

bool CheckErrorFinite(const std::string &name, const Eigen::VectorXd &error) {
    if (!error.allFinite()) {
        std::cerr << name << " error finite check failed\nerror:\n"
                  << error.transpose() << '\n';
        return false;
    }
    return true;
}

bool CheckError(const std::string &name, const Eigen::VectorXd &error,
                double residual_scale) {
    return residual_scale == 0.0 ? CheckErrorZero(name, error)
                         : CheckErrorFinite(name, error);
}

Eigen::Vector2d PointResidualOffset(double residual_scale) {
    return residual_scale * Eigen::Vector2d(1.25, -0.85);
}

Eigen::Vector4d LineResidualOffset(double residual_scale) {
    return residual_scale * Eigen::Vector4d(1.1, -0.7, -0.9, 1.3);
}

std::string ResidualCaseName(double residual_scale) {
    if (residual_scale == 0.0) {
        return "zero";
    }
    return "residual_scale_" + std::to_string(residual_scale);
}

bool CheckMatrixNear(const std::string &name, const std::string &block_name,
                     const Eigen::MatrixXd &analytic,
                     const Eigen::MatrixXd &numeric, double tolerance) {
    const Eigen::MatrixXd diff = analytic - numeric;
    const double max_abs = diff.cwiseAbs().maxCoeff();
    const double denom = std::max(1.0, numeric.cwiseAbs().maxCoeff());
    const double max_rel = max_abs / denom;

    if (!analytic.allFinite() || !numeric.allFinite() ||
        (max_abs > tolerance && max_rel > kJacobianRelativeTolerance)) {
        std::cerr << std::setprecision(12) << name << " " << block_name
                  << " jacobian check failed, max_abs=" << max_abs
                  << ", max_rel=" << max_rel
                  << ", abs_tolerance=" << tolerance
                  << ", rel_tolerance=" << kJacobianRelativeTolerance
                  << "\nanalytic:\n"
                  << analytic << "\nnumeric:\n"
                  << numeric << "\ndiff:\n"
                  << diff << '\n';
        return false;
    }

    return true;
}

template <int VertexN, typename AnalyticEdgeType, typename NumericEdgeType>
bool CompareBlock(const std::string &name, const std::string &block_name,
                  const AnalyticEdgeType &analytic_edge,
                  const NumericEdgeType &numeric_edge, double tolerance) {
    const Eigen::MatrixXd analytic =
        analytic_edge.template jacobianOplusXn<VertexN>();
    const Eigen::MatrixXd numeric =
        numeric_edge.template jacobianOplusXn<VertexN>();
    return CheckMatrixNear(name, block_name, analytic, numeric, tolerance);
}

bool CheckPoseVertexUpdate(const std::vector<GeometricCamera *> &cameras,
                           const SampleConfig &config) {
    std::mt19937 rng(7);
    const BasePose pose = MakePose(cameras, rng, config);
    VertexPose vertex;
    vertex.setEstimate(pose);

    const double update[6] = {0.01, -0.02, 0.015, 0.03, -0.01, 0.02};
    vertex.oplus(update);

    const Eigen::Matrix3d expected_rotation =
        pose.Rwb *
        XPLSLAM::ExpSO3(Eigen::Vector3d(update[0], update[1], update[2]));
    const Eigen::Vector3d expected_translation =
        pose.twb + Eigen::Vector3d(update[3], update[4], update[5]);

    const double rotation_error =
        (vertex.estimate().Rwb - expected_rotation).cwiseAbs().maxCoeff();
    const double translation_error =
        (vertex.estimate().twb - expected_translation).cwiseAbs().maxCoeff();
    if (rotation_error > 1e-12 || translation_error > 1e-12 ||
        !vertex.estimate().Rwb.allFinite() ||
        !vertex.estimate().twb.allFinite()) {
        std::cerr << "VertexPose update check failed, rotation_error="
                  << rotation_error
                  << ", translation_error=" << translation_error << '\n';
        return false;
    }

    return true;
}

bool CheckLineVertexUpdate(const BaseLine &line) {
    VertexLine vertex;
    vertex.setEstimate(line);

    BaseLine expected = line;
    const double update[4] = {0.01, -0.015, 0.02, -0.01};
    expected.Update(update);
    vertex.oplus(update);

    const double line_error =
        (vertex.estimate().lineW - expected.lineW).cwiseAbs().maxCoeff();
    const double rotation_error =
        (vertex.estimate().U - expected.U).cwiseAbs().maxCoeff();
    const double w_error =
        (vertex.estimate().W - expected.W).cwiseAbs().maxCoeff();
    const double plucker_dot =
        std::abs(vertex.estimate().lineW.head<3>().dot(
            vertex.estimate().lineW.tail<3>()));

    if (line_error > 1e-12 || rotation_error > 1e-12 || w_error > 1e-12 ||
        plucker_dot > 1e-12 || !vertex.estimate().lineW.allFinite()) {
        std::cerr << "VertexLine update check failed, line_error="
                  << line_error << ", rotation_error=" << rotation_error
                  << ", W_error=" << w_error
                  << ", plucker_dot=" << plucker_dot << '\n';
        return false;
    }

    return true;
}

bool CheckSE3VertexUpdate(const BasePose &pose) {
    VertexoSE3 vertex;
    const BaseSE3 se3 = MakeSE3(pose);
    vertex.setEstimate(se3);

    const double update[6] = {0.01, -0.02, 0.015, 0.03, -0.01, 0.02};
    vertex.oplus(update);

    const Eigen::Matrix3d expected_rotation =
        se3.Rbw *
        XPLSLAM::ExpSO3(Eigen::Vector3d(update[0], update[1], update[2]));
    const Eigen::Vector3d expected_translation =
        se3.tbw + Eigen::Vector3d(update[3], update[4], update[5]);

    const double rotation_error =
        (vertex.estimate().Rbw - expected_rotation).cwiseAbs().maxCoeff();
    const double translation_error =
        (vertex.estimate().tbw - expected_translation).cwiseAbs().maxCoeff();
    if (rotation_error > 1e-12 || translation_error > 1e-12 ||
        !vertex.estimate().Rbw.allFinite() ||
        !vertex.estimate().tbw.allFinite()) {
        std::cerr << "VertexoSE3 update check failed, rotation_error="
                  << rotation_error
                  << ", translation_error=" << translation_error << '\n';
        return false;
    }

    return true;
}

bool CheckPointOnlyPoseEdge(const Sample &sample, int iteration,
                            int cam_id, double residual_scale) {
    VertexPose pose_vertex;
    PrepareVertex(pose_vertex, sample.pose, 0);

    const Eigen::Vector2d observation =
        sample.pose.ProjectPoint(sample.point_w, cam_id) +
        PointResidualOffset(residual_scale);

    EdgeProjectPointOnlyPose analytic(cam_id);
    analytic.setVertex(0, &pose_vertex);
    analytic.setMeasurement(observation);
    analytic.Xw = sample.point_w;
    analytic.computeError();

    NumericEdgeProjectPointOnlyPose numeric(cam_id);
    numeric.setVertex(0, &pose_vertex);
    numeric.setMeasurement(observation);
    numeric.Xw = sample.point_w;
    numeric.computeError();

    const std::string name = "EdgeProjectPointOnlyPose[" +
                             ResidualCaseName(residual_scale) + ":cam" +
                             std::to_string(cam_id) + ":" +
                             std::to_string(iteration) + "]";
    bool ok = CheckError(name, analytic.error(), residual_scale);
    ok = CheckError(name + " numeric", numeric.error(), residual_scale) && ok;

    g2o::JacobianWorkspace analytic_workspace;
    g2o::JacobianWorkspace numeric_workspace;
    LinearizeUnaryPointPose(analytic, analytic_workspace);
    LinearizeUnaryPointPose(numeric, numeric_workspace);

    ok = CompareBlock<0>(name, "pose", analytic, numeric,
                         kPointJacobianTolerance) &&
         ok;
    return ok;
}

bool CheckPointEdge(const Sample &sample, int iteration, int cam_id,
                    double residual_scale) {
    VertexPose pose_vertex;
    g2o::VertexPointXYZ point_vertex;
    PrepareVertex(pose_vertex, sample.pose, 1);
    PrepareVertex(point_vertex, sample.point_w, 0);

    const Eigen::Vector2d observation =
        sample.pose.ProjectPoint(sample.point_w, cam_id) +
        PointResidualOffset(residual_scale);

    EdgeProjectPoint analytic(cam_id);
    analytic.setVertex(0, &point_vertex);
    analytic.setVertex(1, &pose_vertex);
    analytic.setMeasurement(observation);
    analytic.computeError();

    NumericEdgeProjectPoint numeric(cam_id);
    numeric.setVertex(0, &point_vertex);
    numeric.setVertex(1, &pose_vertex);
    numeric.setMeasurement(observation);
    numeric.computeError();

    const std::string name = "EdgeProjectPoint[" +
                             ResidualCaseName(residual_scale) + ":cam" +
                             std::to_string(cam_id) + ":" +
                             std::to_string(iteration) + "]";
    bool ok = CheckError(name, analytic.error(), residual_scale);
    ok = CheckError(name + " numeric", numeric.error(), residual_scale) && ok;

    g2o::JacobianWorkspace analytic_workspace;
    g2o::JacobianWorkspace numeric_workspace;
    LinearizeBinaryPoint(analytic, analytic_workspace);
    LinearizeBinaryPoint(numeric, numeric_workspace);

    ok = CompareBlock<0>(name, "point", analytic, numeric,
                         kPointJacobianTolerance) &&
         ok;
    ok = CompareBlock<1>(name, "pose", analytic, numeric,
                         kPointJacobianTolerance) &&
         ok;
    return ok;
}

bool CheckLineOnlyPoseEdge(const Sample &sample, int iteration,
                           int cam_id, double residual_scale) {
    VertexPose pose_vertex;
    PrepareVertex(pose_vertex, sample.pose, 0);

    const Eigen::Vector4d observation =
        MakeLineObservation(sample.pose.ProjectLine(sample.line.lineW,
                                                    cam_id)) +
        LineResidualOffset(residual_scale);

    EdgeProjectLineOnlyPose analytic(cam_id);
    analytic.setVertex(0, &pose_vertex);
    analytic.setMeasurement(observation);
    analytic.line_w = sample.line.lineW;
    analytic.computeError();

    NumericEdgeProjectLineOnlyPose numeric(cam_id);
    numeric.setVertex(0, &pose_vertex);
    numeric.setMeasurement(observation);
    numeric.line_w = sample.line.lineW;
    numeric.computeError();

    const std::string name = "EdgeProjectLineOnlyPose[" +
                             ResidualCaseName(residual_scale) + ":cam" +
                             std::to_string(cam_id) + ":" +
                             std::to_string(iteration) + "]";
    bool ok = CheckError(name, analytic.error(), residual_scale);
    ok = CheckError(name + " numeric", numeric.error(), residual_scale) && ok;

    g2o::JacobianWorkspace analytic_workspace;
    g2o::JacobianWorkspace numeric_workspace;
    LinearizeUnaryLinePose(analytic, analytic_workspace);
    LinearizeUnaryLinePose(numeric, numeric_workspace);

    ok = CompareBlock<0>(name, "pose", analytic, numeric,
                         kLineJacobianTolerance) &&
         ok;
    return ok;
}

bool CheckLineEdge(const Sample &sample, int iteration, int cam_id,
                   double residual_scale) {
    VertexPose pose_vertex;
    VertexLine line_vertex;
    PrepareVertex(pose_vertex, sample.pose, 1);
    PrepareVertex(line_vertex, sample.line, 0);

    const Eigen::Vector4d observation =
        MakeLineObservation(sample.pose.ProjectLine(sample.line.lineW,
                                                    cam_id)) +
        LineResidualOffset(residual_scale);

    EdgeProjectLine analytic(cam_id);
    analytic.setVertex(0, &line_vertex);
    analytic.setVertex(1, &pose_vertex);
    analytic.setMeasurement(observation);
    analytic.computeError();

    NumericEdgeProjectLine numeric(cam_id);
    numeric.setVertex(0, &line_vertex);
    numeric.setVertex(1, &pose_vertex);
    numeric.setMeasurement(observation);
    numeric.computeError();

    const std::string name = "EdgeProjectLine[" +
                             ResidualCaseName(residual_scale) + ":cam" +
                             std::to_string(cam_id) + ":" +
                             std::to_string(iteration) + "]";
    bool ok = CheckError(name, analytic.error(), residual_scale);
    ok = CheckError(name + " numeric", numeric.error(), residual_scale) && ok;

    g2o::JacobianWorkspace analytic_workspace;
    g2o::JacobianWorkspace numeric_workspace;
    LinearizeBinaryLine(analytic, analytic_workspace);
    LinearizeBinaryLine(numeric, numeric_workspace);

    ok = CompareBlock<0>(name, "line", analytic, numeric,
                         kLineJacobianTolerance) &&
         ok;
    ok = CompareBlock<1>(name, "pose", analytic, numeric,
                         kLineJacobianTolerance) &&
         ok;
    return ok;
}

template <typename AnalyticEdge>
bool CheckSE3PointEdge(const Sample &sample,
                       const std::vector<GeometricCamera *> &cameras1,
                       const std::vector<GeometricCamera *> &cameras2,
                       int frame, int cam_id,
                       bool inverse, const std::string &edge_name,
                       int iteration, double residual_scale) {
    VertexoSE3 se3_vertex;
    g2o::VertexPointXYZ point_vertex;
    PrepareVertex(se3_vertex, MakeSE3(sample.pose), sample.pose, cameras1,
                  cameras2, 1);
    PrepareVertex(point_vertex, sample.point_w, 0);

    const Eigen::Vector2d observation =
        (inverse ? se3_vertex.ProjectPointInverse(sample.point_w, cam_id, frame)
                 : se3_vertex.ProjectPoint(sample.point_w, cam_id, frame)) +
        PointResidualOffset(residual_scale);

    AnalyticEdge analytic(cam_id);
    analytic.setVertex(0, &point_vertex);
    analytic.setVertex(1, &se3_vertex);
    analytic.setMeasurement(observation);
    analytic.computeError();

    NumericEdgeSE3ProjectPoint numeric(cam_id, frame, inverse);
    numeric.setVertex(0, &point_vertex);
    numeric.setVertex(1, &se3_vertex);
    numeric.setMeasurement(observation);
    numeric.computeError();

    const std::string name = edge_name + "[" +
                             ResidualCaseName(residual_scale) + ":cam" +
                             std::to_string(cam_id) + ":" +
                             std::to_string(iteration) + "]";
    bool ok = CheckError(name, analytic.error(), residual_scale);
    ok = CheckError(name + " numeric", numeric.error(), residual_scale) && ok;

    g2o::JacobianWorkspace analytic_workspace;
    g2o::JacobianWorkspace numeric_workspace;
    LinearizeBinarySE3Point(analytic, analytic_workspace);
    LinearizeBinarySE3Point(numeric, numeric_workspace);

    ok = CompareBlock<0>(name, "point", analytic, numeric,
                         kPointJacobianTolerance) &&
         ok;
    ok = CompareBlock<1>(name, "pose", analytic, numeric,
                         kPointJacobianTolerance) &&
         ok;
    return ok;
}

template <typename AnalyticEdge>
bool CheckSE3LineEdge(const Sample &sample,
                      const std::vector<GeometricCamera *> &cameras1,
                      const std::vector<GeometricCamera *> &cameras2,
                      int frame, int cam_id,
                      bool inverse, const std::string &edge_name,
                      int iteration, double residual_scale) {
    VertexoSE3 se3_vertex;
    VertexLine line_vertex;
    PrepareVertex(se3_vertex, MakeSE3(sample.pose), sample.pose, cameras1,
                  cameras2, 1);
    PrepareVertex(line_vertex, sample.line, 0);

    const Eigen::Vector3d projected_line =
        inverse
            ? se3_vertex.ProjectLineInverse(sample.line.lineW, cam_id, frame)
            : se3_vertex.ProjectLine(sample.line.lineW, cam_id, frame);
    const Eigen::Vector4d observation =
        MakeLineObservation(projected_line) + LineResidualOffset(residual_scale);

    AnalyticEdge analytic(cam_id);
    analytic.setVertex(0, &line_vertex);
    analytic.setVertex(1, &se3_vertex);
    analytic.setMeasurement(observation);
    analytic.computeError();

    NumericEdgeSE3ProjectLine numeric(cam_id, frame, inverse);
    numeric.setVertex(0, &line_vertex);
    numeric.setVertex(1, &se3_vertex);
    numeric.setMeasurement(observation);
    numeric.computeError();

    const std::string name = edge_name + "[" +
                             ResidualCaseName(residual_scale) + ":cam" +
                             std::to_string(cam_id) + ":" +
                             std::to_string(iteration) + "]";
    bool ok = CheckError(name, analytic.error(), residual_scale);
    ok = CheckError(name + " numeric", numeric.error(), residual_scale) && ok;

    g2o::JacobianWorkspace analytic_workspace;
    g2o::JacobianWorkspace numeric_workspace;
    LinearizeBinarySE3Line(analytic, analytic_workspace);
    LinearizeBinarySE3Line(numeric, numeric_workspace);

    ok = CompareBlock<0>(name, "line", analytic, numeric,
                         kLineJacobianTolerance) &&
         ok;
    ok = CompareBlock<1>(name, "pose", analytic, numeric,
                         kLineJacobianTolerance) &&
         ok;
    return ok;
}

bool CheckEssentialGraphEdge(int case_id, const g2o::SE3Quat &Tiw,
                             const g2o::SE3Quat &Tjw,
                             const g2o::SE3Quat &measurement) {
    XPLSLAM::VertexPoseGraph vertex_i;
    vertex_i.setId(0);
    vertex_i.setFixed(false);
    vertex_i.setEstimate(Tiw.inverse());

    XPLSLAM::VertexPoseGraph vertex_j;
    vertex_j.setId(1);
    vertex_j.setFixed(false);
    vertex_j.setEstimate(Tjw.inverse());

    EdgeEssentialGraphSE3 analytic;
    analytic.setVertex(0, &vertex_i);
    analytic.setVertex(1, &vertex_j);
    analytic.setMeasurement(measurement);
    analytic.computeError();

    NumericEdgeEssentialGraphSE3 numeric;
    numeric.setVertex(0, &vertex_i);
    numeric.setVertex(1, &vertex_j);
    numeric.setMeasurement(measurement);
    numeric.computeError();

    const std::string name =
        "EdgeEssentialGraphSE3[" + std::to_string(case_id) + "]";
    bool ok = CheckErrorFinite(name, analytic.error());
    ok = CheckErrorFinite(name + " numeric", numeric.error()) && ok;

    g2o::JacobianWorkspace analytic_workspace;
    g2o::JacobianWorkspace numeric_workspace;
    LinearizeBinaryEssentialGraph(analytic, analytic_workspace);
    LinearizeBinaryEssentialGraph(numeric, numeric_workspace);

    ok = CompareBlock<0>(name, "pose_i", analytic, numeric,
                         kPoseGraphJacobianTolerance) &&
         ok;
    ok = CompareBlock<1>(name, "pose_j", analytic, numeric,
                         kPoseGraphJacobianTolerance) &&
         ok;
    return ok;
}

bool CheckAllEdges(const Sample &sample,
                   const std::vector<GeometricCamera *> &cameras1,
                   const std::vector<GeometricCamera *> &cameras2,
                   int iteration, int cam_id, double residual_scale) {
    bool ok = true;
    ok = CheckPointOnlyPoseEdge(sample, iteration, cam_id, residual_scale) && ok;
    ok = CheckPointEdge(sample, iteration, cam_id, residual_scale) && ok;
    ok = CheckLineOnlyPoseEdge(sample, iteration, cam_id, residual_scale) && ok;
    ok = CheckLineEdge(sample, iteration, cam_id, residual_scale) && ok;
    ok = CheckSE3PointEdge<EdgeSE3ProjectPoint>(
             sample, cameras1, cameras2, 1, cam_id, false,
             "EdgeSE3ProjectPoint", iteration, residual_scale) &&
         ok;
    ok = CheckSE3PointEdge<EdgeInverseSE3ProjectPoint>(
             sample, cameras1, cameras2, 2, cam_id, true,
             "EdgeInverseSE3ProjectPoint", iteration, residual_scale) &&
         ok;
    ok = CheckSE3LineEdge<EdgeSE3ProjectLine>(
             sample, cameras1, cameras2, 1, cam_id, false,
             "EdgeSE3ProjectLine", iteration, residual_scale) &&
         ok;
    ok = CheckSE3LineEdge<EdgeInverseSE3ProjectLine>(
             sample, cameras1, cameras2, 2, cam_id, true,
             "EdgeInverseSE3ProjectLine", iteration, residual_scale) &&
         ok;
    return ok;
}

bool CheckEssentialGraphEdges() {
    bool ok = true;

    const g2o::SE3Quat Tiw(
        XPLSLAM::ExpSO3(Eigen::Vector3d(0.03, -0.02, 0.015)),
        Eigen::Vector3d(0.12, -0.04, 0.08));
    const g2o::SE3Quat Tjw(
        XPLSLAM::ExpSO3(Eigen::Vector3d(-0.025, 0.035, -0.02)),
        Eigen::Vector3d(-0.18, 0.11, 0.05));
    const g2o::SE3Quat Tji = Tjw * Tiw.inverse();

    ok = CheckEssentialGraphEdge(0, Tiw, Tjw, Tji) && ok;

    const g2o::SE3Quat small_residual(
        XPLSLAM::ExpSO3(Eigen::Vector3d(1e-4, -8e-5, 6e-5)),
        Eigen::Vector3d(9e-5, -7e-5, 5e-5));
    ok = CheckEssentialGraphEdge(1, Tiw, Tjw, small_residual * Tji) && ok;

    const g2o::SE3Quat larger_residual(
        XPLSLAM::ExpSO3(Eigen::Vector3d(0.45, -0.3, 0.2)),
        Eigen::Vector3d(0.9, -0.7, 0.5));
    ok = CheckEssentialGraphEdge(2, Tiw, Tjw, larger_residual * Tji) && ok;
    return ok;
}

} // namespace

int main() {
    auto camera1_0 =
        std::make_unique<Pinhole>(std::vector<float>{525.0f, 520.0f, 320.0f,
                                                     240.0f});
    auto camera1_1 =
        std::make_unique<Pinhole>(std::vector<float>{610.0f, 590.0f, 300.0f,
                                                     220.0f});
    auto camera1_2 =
        std::make_unique<Pinhole>(std::vector<float>{480.0f, 515.0f, 340.0f,
                                                     255.0f});
    auto camera1_3 =
        std::make_unique<Pinhole>(std::vector<float>{700.0f, 680.0f, 310.0f,
                                                     235.0f});
    auto camera2_0 =
        std::make_unique<Pinhole>(std::vector<float>{500.0f, 505.0f, 300.0f,
                                                     250.0f});
    auto camera2_1 =
        std::make_unique<Pinhole>(std::vector<float>{455.0f, 470.0f, 330.0f,
                                                     210.0f});
    auto camera2_2 =
        std::make_unique<Pinhole>(std::vector<float>{640.0f, 625.0f, 315.0f,
                                                     245.0f});
    auto camera2_3 =
        std::make_unique<Pinhole>(std::vector<float>{560.0f, 545.0f, 295.0f,
                                                     265.0f});
    std::vector<GeometricCamera *> cameras1 = {camera1_0.get(),
                                               camera1_1.get(),
                                               camera1_2.get(),
                                               camera1_3.get()};
    std::vector<GeometricCamera *> cameras2 = {camera2_0.get(),
                                               camera2_1.get(),
                                               camera2_2.get(),
                                               camera2_3.get()};

    const std::vector<SampleConfig> configs = {
        {"normal", 0.12, 0.18, 1.0, 3.5, 7.0, 0.4},
        {"challenging", 0.32, 0.45, 2.0, 2.0, 12.0, 0.25},
    };
    const std::vector<unsigned int> seeds = {42, 31415, 27182};
    const std::vector<double> residual_scales = {0.0, 1.0, 8.0};
    constexpr int kNumSamplesPerSeed = 8;

    bool ok = true;
    int total_samples = 0;

    for (const SampleConfig &config : configs) {
        ok = CheckPoseVertexUpdate(cameras1, config) && ok;
        for (unsigned int seed : seeds) {
            std::mt19937 rng(seed);
            for (int i = 0; i < kNumSamplesPerSeed; ++i) {
                const Sample sample = MakeSample(cameras1, rng, config);
                const int iteration = total_samples++;
                if (iteration == 0) {
                    ok = CheckSE3VertexUpdate(sample.pose) && ok;
                    ok = CheckLineVertexUpdate(sample.line) && ok;
                }
                for (int cam_id = 0; cam_id < static_cast<int>(cameras1.size());
                     ++cam_id) {
                    for (double residual_scale : residual_scales) {
                        ok = CheckAllEdges(sample, cameras1, cameras2, iteration,
                                           cam_id, residual_scale) &&
                             ok;
                    }
                }
            }
        }
    }
    ok = CheckEssentialGraphEdges() && ok;

    if (!ok) {
        std::cerr << "g2o jacobian check failed\n";
        return 1;
    }

    std::cout << "g2o jacobian check passed for " << total_samples
              << " samples across " << cameras1.size() << " cameras and "
              << kCoveredEdgeTypes << " edge types\n";
    return 0;
}
