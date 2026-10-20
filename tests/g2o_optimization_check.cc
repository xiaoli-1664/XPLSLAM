#include "g2o_types.h"
#include "pinhole.h"

#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/types/sba/edge_se3_expmap.h"
#include "g2o/types/sba/vertex_se3_expmap.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/slam3d/vertex_pointxyz.h"

#include <Eigen/Core>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using XPLSLAM::BaseLine;
using XPLSLAM::BasePose;
using XPLSLAM::BaseSE3;
using XPLSLAM::EdgeEssentialGraphSE3;
using XPLSLAM::EdgeInverseSE3ProjectLine;
using XPLSLAM::EdgeInverseSE3ProjectPoint;
using XPLSLAM::EdgeProjectLine;
using XPLSLAM::EdgeProjectLineOnlyPose;
using XPLSLAM::EdgeProjectPoint;
using XPLSLAM::EdgeProjectPointOnlyPose;
using XPLSLAM::EdgeSE3ProjectLine;
using XPLSLAM::EdgeSE3ProjectPoint;
using XPLSLAM::GeometricCamera;
using XPLSLAM::Pinhole;
using XPLSLAM::VertexLine;
using XPLSLAM::VertexoSE3;
using XPLSLAM::VertexPose;

constexpr int kLeftCamId = 0;
constexpr int kRightCamId = 1;
constexpr double kStereoBaseline = 0.2;
constexpr double kInitialChi2Min = 1e-3;
constexpr double kFinalChi2Max = 1e-8;
constexpr double kPoseRotationTolerance = 1e-5;
constexpr double kPoseTranslationTolerance = 1e-5;

struct FeatureSet {
    std::vector<Eigen::Vector3d> points;
    std::vector<Eigen::Matrix<double, 6, 1>> lines;
};

struct OptimizationResult {
    double initial_chi2 = 0.0;
    double final_chi2 = 0.0;
    int iterations = 0;
    BasePose pose;
};

struct SE3OptimizationResult {
    double initial_chi2 = 0.0;
    double final_chi2 = 0.0;
    int iterations = 0;
    BaseSE3 transform;
};

struct PoseGraphResult {
    double initial_chi2 = 0.0;
    double final_chi2 = 0.0;
    int iterations = 0;
    g2o::SE3Quat pose;
};

BasePose MakePose(const std::vector<GeometricCamera *> &cameras,
                  const Eigen::Vector3d &rot,
                  const Eigen::Vector3d &trans) {
    BasePose pose;
    pose.Rwb = XPLSLAM::ExpSO3(rot);
    pose.twb = trans;
    pose.Rcb.resize(cameras.size());
    pose.tcb.resize(cameras.size());
    pose.vpCameras.resize(cameras.size());
    pose.Rcb[kLeftCamId] = Eigen::Matrix3d::Identity();
    pose.tcb[kLeftCamId] = Eigen::Vector3d::Zero();
    pose.vpCameras[kLeftCamId] = cameras[kLeftCamId];
    pose.Rcb[kRightCamId] = Eigen::Matrix3d::Identity();
    pose.tcb[kRightCamId] = Eigen::Vector3d(-kStereoBaseline, 0.0, 0.0);
    pose.vpCameras[kRightCamId] = cameras[kRightCamId];
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

Eigen::Matrix<double, 6, 1>
TransformLine(const Eigen::Matrix<double, 6, 1> &line,
              const Eigen::Matrix3d &R, const Eigen::Vector3d &t) {
    Eigen::Matrix<double, 6, 1> transformed;
    transformed.tail<3>() = R * line.tail<3>();
    transformed.head<3>() = R * line.head<3>() + t.cross(transformed.tail<3>());
    return transformed;
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

Eigen::Vector4d MakeLineObservation(const Eigen::Vector3d &line_2d) {
    Eigen::Vector4d observation;
    const double a = line_2d[0];
    const double b = line_2d[1];
    const double c = line_2d[2];

    if (std::abs(b) > std::abs(a)) {
        const double x1 = 180.0;
        const double x2 = 460.0;
        observation << x1, -(a * x1 + c) / b, x2, -(a * x2 + c) / b;
    } else {
        const double y1 = 120.0;
        const double y2 = 360.0;
        observation << -(b * y1 + c) / a, y1, -(b * y2 + c) / a, y2;
    }

    return observation;
}

FeatureSet MakeFeatureSet() {
    FeatureSet features;
    features.points = {
        {-1.0, -0.7, 4.5}, {-0.2, -0.9, 5.2}, {0.8, -0.6, 4.8},
        {-0.9, 0.1, 6.0},  {0.1, 0.2, 4.2},   {1.0, 0.1, 5.8},
        {-0.6, 0.9, 5.0},  {0.3, 0.8, 6.3},   {1.1, 0.9, 4.9},
        {-1.2, 0.4, 7.0},  {0.6, -1.1, 6.7},  {1.3, -0.2, 5.5},
    };

    const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>
        line_endpoints = {
            {{-1.3, -0.8, 4.4}, {-0.3, 0.7, 5.6}},
            {{0.6, -1.1, 4.8}, {1.4, 0.5, 6.0}},
            {{-0.9, 0.4, 5.2}, {0.7, 1.1, 4.7}},
            {{-1.4, 0.9, 6.2}, {0.2, -0.4, 5.1}},
            {{0.2, -0.7, 6.4}, {1.5, 0.8, 5.3}},
            {{-0.5, -1.0, 5.8}, {1.0, -0.1, 6.6}},
            {{-1.1, 0.2, 4.9}, {0.5, 1.3, 6.1}},
            {{0.7, 0.3, 4.6}, {1.6, -0.9, 5.9}},
        };

    features.lines.reserve(line_endpoints.size());
    for (const auto &endpoints : line_endpoints) {
        features.lines.push_back(
            MakePluckerLine(endpoints.first, endpoints.second));
    }

    return features;
}

std::unique_ptr<g2o::SparseOptimizer> MakePoseOnlyOptimizer() {
    auto optimizer = std::make_unique<g2o::SparseOptimizer>();
    auto linear_solver = std::make_unique<
        g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
    auto block_solver =
        std::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver));
    optimizer->setAlgorithm(
        new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver)));
    optimizer->setVerbose(false);
    return optimizer;
}

std::unique_ptr<g2o::SparseOptimizer> MakeBundleOptimizer() {
    auto optimizer = std::make_unique<g2o::SparseOptimizer>();
    auto linear_solver = std::make_unique<
        g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>();
    auto block_solver =
        std::make_unique<g2o::BlockSolverX>(std::move(linear_solver));
    optimizer->setAlgorithm(
        new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver)));
    optimizer->setVerbose(false);
    return optimizer;
}

double GraphChi2(g2o::SparseOptimizer &optimizer) {
    double chi2 = 0.0;
    for (auto *edge_base : optimizer.edges()) {
        auto *edge = static_cast<g2o::OptimizableGraph::Edge *>(edge_base);
        edge->computeError();
        chi2 += edge->chi2();
    }
    return chi2;
}

double RotationError(const BasePose &lhs, const BasePose &rhs) {
    return XPLSLAM::LogSO3(lhs.Rwb.transpose() * rhs.Rwb).norm();
}

double TranslationError(const BasePose &lhs, const BasePose &rhs) {
    return (lhs.twb - rhs.twb).norm();
}

bool CheckResult(const std::string &name, const OptimizationResult &result,
                 const BasePose &truth, double rotation_tolerance,
                 double translation_tolerance) {
    const double rotation_error = RotationError(result.pose, truth);
    const double translation_error = TranslationError(result.pose, truth);
    const bool ok = result.iterations > 0 &&
                    result.initial_chi2 > kInitialChi2Min &&
                    result.final_chi2 < result.initial_chi2 * 1e-6 &&
                    result.final_chi2 < kFinalChi2Max &&
                    rotation_error < rotation_tolerance &&
                    translation_error < translation_tolerance;
    if (!ok) {
        std::cerr << name << " failed: iterations=" << result.iterations
                  << ", initial_chi2=" << result.initial_chi2
                  << ", final_chi2=" << result.final_chi2
                  << ", rotation_error=" << rotation_error
                  << ", translation_error=" << translation_error << '\n';
    }
    return ok;
}

bool CheckResidualResult(const std::string &name,
                         const OptimizationResult &result) {
    const bool ok = result.iterations > 0 &&
                    result.initial_chi2 > kInitialChi2Min &&
                    result.final_chi2 < result.initial_chi2 * 1e-6 &&
                    result.final_chi2 < kFinalChi2Max &&
                    result.pose.Rwb.allFinite() && result.pose.twb.allFinite();
    if (!ok) {
        std::cerr << name << " failed: iterations=" << result.iterations
                  << ", initial_chi2=" << result.initial_chi2
                  << ", final_chi2=" << result.final_chi2 << '\n';
    }
    return ok;
}

bool CheckSE3Result(const std::string &name, const SE3OptimizationResult &result,
                    const g2o::SE3Quat &truth, double rotation_tolerance,
                    double translation_tolerance) {
    const double rotation_error =
        XPLSLAM::LogSO3(result.transform.Rbw.transpose() *
                        truth.rotation().toRotationMatrix())
            .norm();
    const double translation_error =
        (result.transform.tbw - truth.translation()).norm();
    const bool ok = result.iterations > 0 &&
                    result.initial_chi2 > kInitialChi2Min &&
                    result.final_chi2 < result.initial_chi2 * 1e-6 &&
                    result.final_chi2 < kFinalChi2Max &&
                    rotation_error < rotation_tolerance &&
                    translation_error < translation_tolerance;
    if (!ok) {
        std::cerr << name << " failed: iterations=" << result.iterations
                  << ", initial_chi2=" << result.initial_chi2
                  << ", final_chi2=" << result.final_chi2
                  << ", rotation_error=" << rotation_error
                  << ", translation_error=" << translation_error << '\n';
    }
    return ok;
}

bool CheckPoseGraphResult(const std::string &name,
                          const PoseGraphResult &result,
                          const g2o::SE3Quat &truth,
                          double rotation_tolerance,
                          double translation_tolerance) {
    const g2o::SE3Quat error = result.pose * truth.inverse();
    const double rotation_error =
        XPLSLAM::LogSO3(error.rotation().toRotationMatrix()).norm();
    const double translation_error = error.translation().norm();
    const bool ok = result.iterations > 0 &&
                    result.initial_chi2 > kInitialChi2Min &&
                    result.final_chi2 < result.initial_chi2 * 1e-6 &&
                    result.final_chi2 < kFinalChi2Max &&
                    rotation_error < rotation_tolerance &&
                    translation_error < translation_tolerance;
    if (!ok) {
        std::cerr << name << " failed: iterations=" << result.iterations
                  << ", initial_chi2=" << result.initial_chi2
                  << ", final_chi2=" << result.final_chi2
                  << ", rotation_error=" << rotation_error
                  << ", translation_error=" << translation_error << '\n';
    }
    return ok;
}

OptimizationResult RunPoseOnlyCase(const std::string &name,
                                   const std::vector<GeometricCamera *> &cameras,
                                   const FeatureSet &features,
                                   bool use_points, bool use_lines) {
    const BasePose truth =
        MakePose(cameras, Eigen::Vector3d(0.05, -0.04, 0.03),
                 Eigen::Vector3d(0.18, -0.08, 0.12));
    const BasePose initial =
        MakePose(cameras, Eigen::Vector3d(0.07, -0.065, 0.055),
                 Eigen::Vector3d(0.25, -0.13, 0.17));

    auto optimizer = MakePoseOnlyOptimizer();
    auto *pose_vertex = new VertexPose();
    pose_vertex->setId(0);
    pose_vertex->setFixed(false);
    pose_vertex->setEstimate(initial);
    optimizer->addVertex(pose_vertex);

    int edge_id = 0;
    if (use_points) {
        for (const Eigen::Vector3d &point : features.points) {
            for (int cam_id = 0; cam_id < static_cast<int>(cameras.size());
                 ++cam_id) {
                auto *edge = new EdgeProjectPointOnlyPose(cam_id);
                edge->setId(edge_id++);
                edge->setVertex(0, pose_vertex);
                edge->Xw = point;
                edge->setMeasurement(truth.ProjectPoint(point, cam_id));
                edge->setInformation(Eigen::Matrix2d::Identity());
                optimizer->addEdge(edge);
            }
        }
    }

    if (use_lines) {
        for (const auto &line : features.lines) {
            for (int cam_id = 0; cam_id < static_cast<int>(cameras.size());
                 ++cam_id) {
                auto *edge = new EdgeProjectLineOnlyPose(cam_id);
                edge->setId(edge_id++);
                edge->setVertex(0, pose_vertex);
                edge->line_w = line;
                edge->setMeasurement(
                    MakeLineObservation(truth.ProjectLine(line, cam_id)));
                edge->setInformation(Eigen::Matrix2d::Identity());
                edge->setEndpointCovariance(Eigen::Matrix4d::Identity());
                optimizer->addEdge(edge);
            }
        }
    }

    optimizer->initializeOptimization();
    const double initial_chi2 = GraphChi2(*optimizer);
    const int iterations = optimizer->optimize(30);
    const double final_chi2 = GraphChi2(*optimizer);

    OptimizationResult result;
    result.initial_chi2 = initial_chi2;
    result.final_chi2 = final_chi2;
    result.iterations = iterations;
    result.pose = pose_vertex->estimate();
    std::cout << name << ": initial_chi2=" << initial_chi2
              << ", final_chi2=" << final_chi2 << '\n';
    return result;
}

OptimizationResult RunBundleCase(const std::string &name,
                                 const std::vector<GeometricCamera *> &cameras,
                                 const FeatureSet &features, bool use_points,
                                 bool use_lines) {
    const BasePose fixed_truth =
        MakePose(cameras, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    const BasePose truth =
        MakePose(cameras, Eigen::Vector3d(0.04, -0.03, 0.025),
                 Eigen::Vector3d(0.22, -0.05, 0.10));
    const BasePose initial =
        MakePose(cameras, Eigen::Vector3d(0.06, -0.052, 0.045),
                 Eigen::Vector3d(0.29, -0.10, 0.15));

    auto optimizer = MakeBundleOptimizer();
    auto *fixed_pose_vertex = new VertexPose();
    fixed_pose_vertex->setId(0);
    fixed_pose_vertex->setFixed(true);
    fixed_pose_vertex->setEstimate(fixed_truth);
    optimizer->addVertex(fixed_pose_vertex);

    auto *pose_vertex = new VertexPose();
    pose_vertex->setId(1);
    pose_vertex->setFixed(false);
    pose_vertex->setEstimate(initial);
    optimizer->addVertex(pose_vertex);

    int next_vertex_id = 2;
    int next_edge_id = 0;

    if (use_points) {
        for (const Eigen::Vector3d &point : features.points) {
            auto *point_vertex = new g2o::VertexPointXYZ();
            point_vertex->setId(next_vertex_id++);
            point_vertex->setMarginalized(true);
            point_vertex->setFixed(false);
            point_vertex->setEstimate(
                point + Eigen::Vector3d(0.03, -0.025, 0.04));
            optimizer->addVertex(point_vertex);

            const VertexPose *pose_vertices[2] = {fixed_pose_vertex,
                                                  pose_vertex};
            const BasePose *poses[2] = {&fixed_truth, &truth};
            for (int i = 0; i < 2; ++i) {
                for (int cam_id = 0; cam_id < static_cast<int>(cameras.size());
                     ++cam_id) {
                    auto *edge = new EdgeProjectPoint(cam_id);
                    edge->setId(next_edge_id++);
                    edge->setVertex(0, point_vertex);
                    edge->setVertex(
                        1, const_cast<VertexPose *>(pose_vertices[i]));
                    edge->setMeasurement(poses[i]->ProjectPoint(point, cam_id));
                    edge->setInformation(Eigen::Matrix2d::Identity());
                    optimizer->addEdge(edge);
                }
            }
        }
    }

    if (use_lines) {
        for (const auto &line : features.lines) {
            auto *line_vertex = new VertexLine();
            line_vertex->setId(next_vertex_id++);
            line_vertex->setMarginalized(true);
            line_vertex->setFixed(false);
            line_vertex->setEstimate(MakeBaseLine(line));
            optimizer->addVertex(line_vertex);

            const VertexPose *pose_vertices[2] = {fixed_pose_vertex,
                                                  pose_vertex};
            const BasePose *poses[2] = {&fixed_truth, &truth};
            for (int i = 0; i < 2; ++i) {
                for (int cam_id = 0; cam_id < static_cast<int>(cameras.size());
                     ++cam_id) {
                    auto *edge = new EdgeProjectLine(cam_id);
                    edge->setId(next_edge_id++);
                    edge->setVertex(0, line_vertex);
                    edge->setVertex(
                        1, const_cast<VertexPose *>(pose_vertices[i]));
                    edge->setMeasurement(MakeLineObservation(
                        poses[i]->ProjectLine(line, cam_id)));
                    edge->setInformation(Eigen::Matrix2d::Identity());
                    edge->setEndpointCovariance(Eigen::Matrix4d::Identity());
                    optimizer->addEdge(edge);
                }
            }
        }
    }

    optimizer->initializeOptimization();
    const double initial_chi2 = GraphChi2(*optimizer);
    const int iterations = optimizer->optimize(30);
    const double final_chi2 = GraphChi2(*optimizer);

    OptimizationResult result;
    result.initial_chi2 = initial_chi2;
    result.final_chi2 = final_chi2;
    result.iterations = iterations;
    result.pose = pose_vertex->estimate();
    std::cout << name << ": initial_chi2=" << initial_chi2
              << ", final_chi2=" << final_chi2 << '\n';
    return result;
}

SE3OptimizationResult RunRelativeSE3Case(
    const std::string &name, const std::vector<GeometricCamera *> &cameras,
    const FeatureSet &features, bool use_points, bool use_lines) {
    const g2o::SE3Quat truth(
        XPLSLAM::ExpSO3(Eigen::Vector3d(0.04, -0.03, 0.035)),
        Eigen::Vector3d(0.16, -0.04, 0.08));
    const g2o::SE3Quat initial(
        XPLSLAM::ExpSO3(Eigen::Vector3d(0.07, -0.055, 0.06)),
        Eigen::Vector3d(0.23, -0.09, 0.13));

    auto optimizer = MakeBundleOptimizer();
    auto *se3_vertex = new VertexoSE3(initial);
    se3_vertex->setId(0);
    se3_vertex->setFixed(false);
    se3_vertex->vpCameras1 = cameras;
    se3_vertex->vpCameras2 = cameras;
    se3_vertex->vTcb.assign(cameras.size(), Sophus::SE3d());
    optimizer->addVertex(se3_vertex);

    int next_vertex_id = 1;
    int next_edge_id = 0;
    if (use_points) {
        for (const Eigen::Vector3d &point1 : features.points) {
            const Eigen::Vector3d point2 = truth.inverse() * point1;

            auto *point1_vertex = new g2o::VertexPointXYZ();
            point1_vertex->setId(next_vertex_id++);
            point1_vertex->setFixed(true);
            point1_vertex->setEstimate(point1);
            optimizer->addVertex(point1_vertex);

            auto *point2_vertex = new g2o::VertexPointXYZ();
            point2_vertex->setId(next_vertex_id++);
            point2_vertex->setFixed(true);
            point2_vertex->setEstimate(point2);
            optimizer->addVertex(point2_vertex);

            for (int cam_id = 0; cam_id < static_cast<int>(cameras.size());
                 ++cam_id) {
                auto *edge12 = new EdgeSE3ProjectPoint(cam_id);
                edge12->setId(next_edge_id++);
                edge12->setVertex(0, point2_vertex);
                edge12->setVertex(1, se3_vertex);
                edge12->setMeasurement(cameras[cam_id]->project(point1));
                edge12->setInformation(Eigen::Matrix2d::Identity());
                optimizer->addEdge(edge12);

                auto *edge21 = new EdgeInverseSE3ProjectPoint(cam_id);
                edge21->setId(next_edge_id++);
                edge21->setVertex(0, point1_vertex);
                edge21->setVertex(1, se3_vertex);
                edge21->setMeasurement(cameras[cam_id]->project(point2));
                edge21->setInformation(Eigen::Matrix2d::Identity());
                optimizer->addEdge(edge21);
            }
        }
    }

    if (use_lines) {
        const Eigen::Matrix3d R21 =
            truth.rotation().toRotationMatrix().transpose();
        const Eigen::Vector3d t21 = -(R21 * truth.translation());
        for (const auto &line1 : features.lines) {
            const auto line2 = TransformLine(line1, R21, t21);

            auto *line1_vertex = new VertexLine();
            line1_vertex->setId(next_vertex_id++);
            line1_vertex->setFixed(true);
            line1_vertex->setEstimate(MakeBaseLine(line1));
            optimizer->addVertex(line1_vertex);

            auto *line2_vertex = new VertexLine();
            line2_vertex->setId(next_vertex_id++);
            line2_vertex->setFixed(true);
            line2_vertex->setEstimate(MakeBaseLine(line2));
            optimizer->addVertex(line2_vertex);

            for (int cam_id = 0; cam_id < static_cast<int>(cameras.size());
                 ++cam_id) {
                auto *edge12 = new EdgeSE3ProjectLine(cam_id);
                edge12->setId(next_edge_id++);
                edge12->setVertex(0, line2_vertex);
                edge12->setVertex(1, se3_vertex);
                edge12->setMeasurement(
                    MakeLineObservation(cameras[cam_id]->projectLine(line1)));
                edge12->setInformation(Eigen::Matrix2d::Identity());
                edge12->setEndpointCovariance(Eigen::Matrix4d::Identity());
                optimizer->addEdge(edge12);

                auto *edge21 = new EdgeInverseSE3ProjectLine(cam_id);
                edge21->setId(next_edge_id++);
                edge21->setVertex(0, line1_vertex);
                edge21->setVertex(1, se3_vertex);
                edge21->setMeasurement(
                    MakeLineObservation(cameras[cam_id]->projectLine(line2)));
                edge21->setInformation(Eigen::Matrix2d::Identity());
                edge21->setEndpointCovariance(Eigen::Matrix4d::Identity());
                optimizer->addEdge(edge21);
            }
        }
    }

    optimizer->initializeOptimization();
    const double initial_chi2 = GraphChi2(*optimizer);
    const int iterations = optimizer->optimize(30);
    const double final_chi2 = GraphChi2(*optimizer);

    SE3OptimizationResult result;
    result.initial_chi2 = initial_chi2;
    result.final_chi2 = final_chi2;
    result.iterations = iterations;
    result.transform = se3_vertex->estimate();
    std::cout << name << ": initial_chi2=" << initial_chi2
              << ", final_chi2=" << final_chi2 << '\n';
    return result;
}

PoseGraphResult RunEssentialGraphCase(const std::string &name) {
    const g2o::SE3Quat Tiw_fixed(
        XPLSLAM::ExpSO3(Eigen::Vector3d(0.3, -0.2, 0.15)),
        Eigen::Vector3d(0.8, -0.5, 0.4));
    const g2o::SE3Quat Tjw_truth(
        XPLSLAM::ExpSO3(Eigen::Vector3d(0.08, -0.04, 0.05)),
        Eigen::Vector3d(0.35, -0.12, 0.18));
    const g2o::SE3Quat Tjw_initial(
        XPLSLAM::ExpSO3(Eigen::Vector3d(0.12, -0.075, 0.09)),
        Eigen::Vector3d(0.48, -0.22, 0.27));

    auto optimizer = MakePoseOnlyOptimizer();
    auto *vertex_i = new XPLSLAM::VertexPoseGraph();
    vertex_i->setId(0);
    vertex_i->setFixed(true);
    vertex_i->setEstimate(Tiw_fixed.inverse());
    optimizer->addVertex(vertex_i);

    auto *vertex_j = new XPLSLAM::VertexPoseGraph();
    vertex_j->setId(1);
    vertex_j->setFixed(false);
    vertex_j->setEstimate(Tjw_initial.inverse());
    optimizer->addVertex(vertex_j);

    const g2o::SE3Quat Tji = Tjw_truth * Tiw_fixed.inverse();
    auto *edge = new EdgeEssentialGraphSE3();
    edge->setVertex(0, vertex_i);
    edge->setVertex(1, vertex_j);
    edge->setMeasurement(Tji);
    edge->information() = Eigen::Matrix<double, 6, 6>::Identity();
    optimizer->addEdge(edge);

    optimizer->initializeOptimization();
    const double initial_chi2 = GraphChi2(*optimizer);
    const int iterations = optimizer->optimize(20);
    const double final_chi2 = GraphChi2(*optimizer);

    PoseGraphResult result;
    result.initial_chi2 = initial_chi2;
    result.final_chi2 = final_chi2;
    result.iterations = iterations;
    result.pose = vertex_j->estimate().inverse();
    std::cout << name << ": initial_chi2=" << initial_chi2
              << ", final_chi2=" << final_chi2 << '\n';
    return result;
}

bool CheckLineObservationCovariance(
    const std::vector<GeometricCamera *> &cameras) {
    Eigen::Matrix4d covariance;
    covariance << 4.0, 1.0, 0.5, 0.2, 1.0, 9.0, 0.3, 0.4, 0.5, 0.3,
        16.0, 2.0, 0.2, 0.4, 2.0, 25.0;

    XPLSLAM::PackedLineEndpointCovariance packed;
    if (!XPLSLAM::PackLineEndpointCovariance(covariance, packed)) {
        std::cerr << "Failed to pack a valid endpoint covariance\n";
        return false;
    }

    const Eigen::Matrix2d vertical_covariance =
        (Eigen::Matrix2d() << 4.0, 0.5, 0.5, 16.0).finished();
    const Eigen::Matrix2d horizontal_covariance =
        (Eigen::Matrix2d() << 9.0, 0.4, 0.4, 25.0).finished();
    const Eigen::Matrix2d vertical_information =
        XPLSLAM::LineResidualInformation(Eigen::Vector3d(2.0, 0.0, 7.0),
                                         packed, true);
    const Eigen::Matrix2d horizontal_information =
        XPLSLAM::LineResidualInformation(Eigen::Vector3d(0.0, -3.0, 4.0),
                                         packed, true);
    if (!vertical_information.isApprox(vertical_covariance.inverse(), 1e-6) ||
        !horizontal_information.isApprox(horizontal_covariance.inverse(),
                                         1e-6)) {
        std::cerr << "Endpoint-to-line covariance propagation is incorrect\n";
        return false;
    }

    const Eigen::Vector3d oblique_line(2.0, -5.0, 11.0);
    const Eigen::Matrix2d whitening = XPLSLAM::LineResidualWhitening(
        oblique_line, packed, true);
    const Eigen::Matrix2d information = XPLSLAM::LineResidualInformation(
        oblique_line, packed, true);
    const Eigen::Matrix2d scaled_information =
        XPLSLAM::LineResidualInformation(-7.0 * oblique_line, packed, true);
    if (!(whitening.transpose() * whitening).isApprox(information, 1e-10) ||
        !scaled_information.isApprox(information, 1e-10)) {
        std::cerr << "Line whitening or scale invariance is incorrect\n";
        return false;
    }

    const Eigen::Matrix2d fallback = XPLSLAM::LineResidualInformation(
        oblique_line, {}, false, XPLSLAM::kLegacyLineInformation);
    if (!fallback.isApprox(0.25 * Eigen::Matrix2d::Identity(), 1e-12)) {
        std::cerr << "Legacy line-information fallback is incorrect\n";
        return false;
    }

    Eigen::Matrix4d invalid = covariance;
    invalid(0, 0) = -1.0;
    if (XPLSLAM::PackLineEndpointCovariance(invalid, packed)) {
        std::cerr << "Invalid endpoint covariance was accepted\n";
        return false;
    }

    BasePose pose = MakePose(cameras, Eigen::Vector3d::Zero(),
                             Eigen::Vector3d::Zero());
    VertexPose vertex;
    vertex.setEstimate(pose);
    EdgeProjectLineOnlyPose edge(kLeftCamId);
    edge.setVertex(0, &vertex);
    edge.line_w = MakePluckerLine(Eigen::Vector3d(-0.8, -0.3, 3.5),
                                  Eigen::Vector3d(0.7, 0.5, 4.2));
    edge.setMeasurement(MakeLineObservation(
        pose.ProjectLine(edge.line_w, kLeftCamId)));
    edge.setInformation(Eigen::Matrix2d::Identity() * 0.25);
    if (!edge.setEndpointCovariance(covariance)) {
        std::cerr << "g2o line edge rejected a valid covariance\n";
        return false;
    }
    edge.computeError();
    const Eigen::Matrix2d information0 = edge.information();

    pose.Rwb = XPLSLAM::ExpSO3(Eigen::Vector3d(0.25, -0.18, 0.12));
    vertex.setEstimate(pose);
    edge.computeError();
    const Eigen::Vector3d projected_line =
        pose.ProjectLine(edge.line_w, kLeftCamId);
    const Eigen::Matrix2d expected_information =
        XPLSLAM::LineResidualInformation(projected_line, packed, true);
    if (!edge.information().isApprox(expected_information, 1e-8) ||
        edge.information().isApprox(information0, 1e-5)) {
        std::cerr << "g2o line information did not follow the projected line\n";
        return false;
    }
    return true;
}

} // namespace

int main() {
    auto left_camera =
        std::make_unique<Pinhole>(std::vector<float>{525.0f, 520.0f, 320.0f,
                                                     240.0f});
    auto right_camera =
        std::make_unique<Pinhole>(std::vector<float>{525.0f, 520.0f, 320.0f,
                                                     240.0f});
    const std::vector<GeometricCamera *> cameras = {left_camera.get(),
                                                   right_camera.get()};
    const FeatureSet features = MakeFeatureSet();
    bool ok = true;

    ok = CheckLineObservationCovariance(cameras) && ok;

    ok = CheckResult("pose-only points",
                     RunPoseOnlyCase("pose-only points", cameras,
                                     features, true, false),
                     MakePose(cameras, Eigen::Vector3d(0.05, -0.04, 0.03),
                              Eigen::Vector3d(0.18, -0.08, 0.12)),
                     kPoseRotationTolerance, kPoseTranslationTolerance) &&
         ok;
    ok = CheckResult("pose-only points+lines",
                     RunPoseOnlyCase("pose-only points+lines", cameras,
                                     features, true, true),
                     MakePose(cameras, Eigen::Vector3d(0.05, -0.04, 0.03),
                              Eigen::Vector3d(0.18, -0.08, 0.12)),
                     kPoseRotationTolerance, kPoseTranslationTolerance) &&
         ok;
    ok = CheckResult("pose-only lines",
                     RunPoseOnlyCase("pose-only lines", cameras, features,
                                     false, true),
                     MakePose(cameras, Eigen::Vector3d(0.05, -0.04, 0.03),
                              Eigen::Vector3d(0.18, -0.08, 0.12)),
                     5e-5, 5e-5) &&
         ok;

    ok = CheckResult("bundle points",
                     RunBundleCase("bundle points", cameras, features, true,
                                   false),
                     MakePose(cameras, Eigen::Vector3d(0.04, -0.03, 0.025),
                              Eigen::Vector3d(0.22, -0.05, 0.10)),
                     kPoseRotationTolerance, kPoseTranslationTolerance) &&
         ok;
    ok = CheckResult("bundle points+lines",
                     RunBundleCase("bundle points+lines", cameras, features,
                                   true, true),
                     MakePose(cameras, Eigen::Vector3d(0.04, -0.03, 0.025),
                              Eigen::Vector3d(0.22, -0.05, 0.10)),
                     kPoseRotationTolerance, kPoseTranslationTolerance) &&
         ok;
    ok = CheckResult("bundle lines",
                     RunBundleCase("bundle lines", cameras, features, false,
                                   true),
                     MakePose(cameras, Eigen::Vector3d(0.04, -0.03, 0.025),
                              Eigen::Vector3d(0.22, -0.05, 0.10)),
                     1e-4, 1e-4) &&
         ok;

    const g2o::SE3Quat relative_truth(
        XPLSLAM::ExpSO3(Eigen::Vector3d(0.04, -0.03, 0.035)),
        Eigen::Vector3d(0.16, -0.04, 0.08));
    ok = CheckSE3Result("relative SE3 points",
                        RunRelativeSE3Case("relative SE3 points", cameras,
                                           features, true, false),
                        relative_truth, kPoseRotationTolerance,
                        kPoseTranslationTolerance) &&
         ok;
    ok = CheckSE3Result("relative SE3 points+lines",
                        RunRelativeSE3Case("relative SE3 points+lines",
                                           cameras, features, true, true),
                        relative_truth, kPoseRotationTolerance,
                        kPoseTranslationTolerance) &&
         ok;
    ok = CheckSE3Result("relative SE3 lines",
                        RunRelativeSE3Case("relative SE3 lines", cameras,
                                           features, false, true),
                        relative_truth, 1e-4, 1e-4) &&
         ok;

    const g2o::SE3Quat essential_truth(
        XPLSLAM::ExpSO3(Eigen::Vector3d(0.08, -0.04, 0.05)),
        Eigen::Vector3d(0.35, -0.12, 0.18));
    ok = CheckPoseGraphResult("essential graph",
                              RunEssentialGraphCase("essential graph"),
                              essential_truth, kPoseRotationTolerance,
                              kPoseTranslationTolerance) &&
         ok;

    if (!ok) {
        std::cerr << "g2o optimization check failed\n";
        return 1;
    }

    std::cout << "g2o optimization check passed\n";
    return 0;
}
