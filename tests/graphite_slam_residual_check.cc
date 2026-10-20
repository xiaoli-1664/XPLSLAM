#include "g2o_types.h"
#include "graphite_slam_types.h"
#include "graphite_local_ba_jacobians.cuh"
#include "graphite_essential_graph_jacobians.cuh"
#include "g2o/core/jacobian_workspace.h"
#include "pinhole.h"

#include <Eigen/Core>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace gb = XPLSLAM::graphite_backend;

constexpr double kTolerance = 1e-9;

gb::PinholeCamera MakeGraphiteCamera(const XPLSLAM::Pinhole &camera) {
    return gb::MakePinholeCamera(camera.getParameter(0), camera.getParameter(1),
                                 camera.getParameter(2),
                                 camera.getParameter(3));
}

gb::Pose MakeGraphitePose(const XPLSLAM::BasePose &pose, int cam_id) {
    gb::Pose graphite_pose;
    graphite_pose.Rwb = pose.Rwb;
    graphite_pose.twb = pose.twb;
    graphite_pose.Rcb = pose.Rcb[cam_id];
    graphite_pose.tcb = pose.tcb[cam_id];
    return graphite_pose;
}

gb::Vector6d MakePluckerLine(const Eigen::Vector3d &p1,
                             const Eigen::Vector3d &p2) {
    gb::Vector6d line;
    const Eigen::Vector3d direction = p2 - p1;
    line.head<3>() = p1.cross(direction);
    line.tail<3>() = direction;
    return line;
}

Eigen::Vector4d MakeLineObservation(const Eigen::Vector3d &line_2d) {
    Eigen::Vector4d observation;
    const double a = line_2d[0];
    const double b = line_2d[1];
    const double c = line_2d[2];

    if (std::abs(b) > std::abs(a)) {
        const double x1 = 240.0;
        const double x2 = 420.0;
        observation << x1, -(a * x1 + c) / b, x2, -(a * x2 + c) / b;
    } else {
        const double y1 = 150.0;
        const double y2 = 330.0;
        observation << -(b * y1 + c) / a, y1, -(b * y2 + c) / a, y2;
    }

    return observation;
}

bool CheckVectorNear(const std::string &name, const Eigen::VectorXd &actual,
                     const Eigen::VectorXd &expected) {
    const Eigen::VectorXd diff = actual - expected;
    const double max_abs = diff.cwiseAbs().maxCoeff();
    if (!actual.allFinite() || !expected.allFinite() || max_abs > kTolerance) {
        std::cerr << std::setprecision(12) << name
                  << " mismatch, max_abs=" << max_abs << "\nactual:\n"
                  << actual.transpose() << "\nexpected:\n"
                  << expected.transpose() << "\ndiff:\n"
                  << diff.transpose() << '\n';
        return false;
    }
    return true;
}

bool CheckBoolEqual(const std::string &name, bool actual, bool expected) {
    if (actual != expected) {
        std::cerr << name << " mismatch, actual=" << actual
                  << ", expected=" << expected << '\n';
        return false;
    }
    return true;
}

XPLSLAM::BasePose MakeReferencePose(XPLSLAM::Pinhole &camera) {
    XPLSLAM::BasePose pose;
    pose.Rwb = XPLSLAM::ExpSO3(Eigen::Vector3d(0.04, -0.03, 0.02));
    pose.twb = Eigen::Vector3d(0.2, -0.1, 0.3);
    pose.Rcb.push_back(
        XPLSLAM::ExpSO3(Eigen::Vector3d(-0.02, 0.015, 0.01)));
    pose.tcb.push_back(Eigen::Vector3d(0.05, -0.02, 0.01));
    pose.vpCameras.push_back(&camera);
    pose.its = 0;
    return pose;
}

using Matrix26 = Eigen::Matrix<double, 2, 6>;
using Matrix6 = Eigen::Matrix<double, 6, 6>;
using Vector6 = Eigen::Matrix<double, 6, 1>;

bool CheckMatrix(const std::string &name, const Eigen::MatrixXd &actual,
                 const Eigen::MatrixXd &expected, double abs_tol = 2e-5,
                 double rel_tol = 2e-7) {
    const double error = (actual - expected).cwiseAbs().maxCoeff();
    const double scale = std::max(1.0, expected.cwiseAbs().maxCoeff());
    if (!actual.allFinite() || !expected.allFinite() ||
        (error > abs_tol && error / scale > rel_tol)) {
        std::cerr << name << " mismatch: abs=" << error
                  << ", rel=" << error / scale << '\n';
        return false;
    }
    return true;
}

template <int Rows, int Cols, typename Function>
Eigen::Matrix<double, Rows, Cols> NumericalJacobian(Function f) {
    constexpr double eps = 1e-6;
    Eigen::Matrix<double, Rows, Cols> J;
    for (int col = 0; col < Cols; ++col) {
        Eigen::Matrix<double, Cols, 1> delta =
            Eigen::Matrix<double, Cols, 1>::Zero();
        delta[col] = eps;
        J.col(col) = (f(delta) - f(-delta)) / (2 * eps);
    }
    return J;
}

XPLSLAM::BasePose Retract(XPLSLAM::BasePose pose, const Vector6 &delta) {
    pose.Update(delta.data());
    return pose;
}

bool CheckGeneratedJacobians(XPLSLAM::Pinhole &camera) {
    bool ok = true;
    const gb::PinholeCamera model = MakeGraphiteCamera(camera);
    const Eigen::Vector3d rcb(0.18, -0.09, 0.06);
    const Eigen::Vector3d tcb(0.21, -0.13, 0.07);
    const std::vector<Eigen::Vector3d> rotations = {
        {0, 0, 0}, {1e-9, -2e-9, 3e-9}, {0.3, -0.2, 0.4},
        {-0.7, 0.9, -0.5}, {3.12, 0.02, -0.01}, {0.02, -3.12, 0.01},
        {0.01, -0.02, 3.12}};
    for (size_t sample = 0; sample < rotations.size(); ++sample) {
        XPLSLAM::BasePose pose = MakeReferencePose(camera);
        pose.Rwb = XPLSLAM::ExpSO3(rotations[sample]);
        pose.twb = Eigen::Vector3d(1.2, -0.8, 0.6);
        pose.Rcb[0] = XPLSLAM::ExpSO3(rcb);
        pose.tcb[0] = tcb;
        const Eigen::Vector3d point_w =
            pose.Rwb * Eigen::Vector3d(0.7, -0.2, 4.8) + pose.twb;
        const Eigen::Vector3d p1 =
            pose.Rwb * Eigen::Vector3d(-0.5, -0.3, 3.7) + pose.twb;
        const Eigen::Vector3d p2 =
            pose.Rwb * Eigen::Vector3d(0.6, 0.5, 4.5) + pose.twb;
        const Vector6 line_w = MakePluckerLine(p1, p2);
        const Eigen::Vector2d point_measurement =
            pose.ProjectPoint(point_w, 0) + Eigen::Vector2d(0.6, -0.4);
        const Eigen::Vector4d line_measurement =
            MakeLineObservation(pose.ProjectLine(line_w, 0)) +
            Eigen::Vector4d(0.2, -0.3, 0.4, -0.1);
        const gb::Pose graphite_pose = MakeGraphitePose(pose, 0);

        // Independent Tbw projection verifies that inversion preserves geometry.
        const Sophus::SE3d Tbw = pose.ToTbw();
        const Eigen::Vector3d point_c = pose.Rcb[0] * (Tbw * point_w) + tcb;
        ok &= CheckVectorNear("Tbw/Twb point projection",
            gb::ProjectWorldPoint(graphite_pose, model, point_w), camera.project(point_c));
        ok &= CheckVectorNear("Tbw/Twb point round trip", Tbw.inverse() * (Tbw * point_w), point_w);
        const Eigen::Matrix3d Rcw = pose.Rcb[0] * Tbw.rotationMatrix();
        const Eigen::Vector3d tcw = pose.Rcb[0] * Tbw.translation() + tcb;
        const Eigen::Vector3d normal_c =
            Rcw * line_w.head<3>() + tcw.cross(Rcw * line_w.tail<3>());
        ok &= CheckVectorNear("Tbw/Twb line projection",
            gb::ProjectWorldLine(graphite_pose, model, line_w), model.toPL * normal_c);
        const Eigen::Vector3d behind =
            pose.Rwb * Eigen::Vector3d(0.0, 0.0, -4.0) + pose.twb;
        ok &= CheckBoolEqual("Twb depth", pose.isPointDepthPositive(behind, 0), false);

        // Correcting world geometry using the new Twb must preserve body coordinates.
        const Vector6 correction =
            (Vector6() << 0.08, -0.04, 0.06, 0.4, -0.3, 0.2).finished();
        const XPLSLAM::BasePose corrected = Retract(pose, correction);
        const Sophus::SE3d world_correction = corrected.ToTbw().inverse() * Tbw;
        const Eigen::Vector3d corrected_point = world_correction * point_w;
        ok &= CheckVectorNear("map point correction", corrected.ToTbw() * corrected_point, Tbw * point_w);
        const Vector6 corrected_line = MakePluckerLine(world_correction * p1, world_correction * p2);
        ok &= CheckVectorNear("map line correction", corrected.ProjectLine(corrected_line, 0), pose.ProjectLine(line_w, 0));

        Matrix26 Jpoint, Jline, Jba;
        Eigen::Vector3d line_2d;
        gb::generated::pose_only_point_jacobian_pose(
            pose.Rwb, pose.twb, point_w, pose.Rcb[0], tcb, model.fx, model.fy, Jpoint);
        gb::generated::pose_only_line_jacobian_pose(
            pose.Rwb, pose.twb, line_w, pose.Rcb[0], tcb, model.toPL,
            line_measurement, Jline, line_2d);
        ok &= CheckMatrix("point g2o/generated", Jpoint,
                          XPLSLAM::PointOnlyPoseJacobian(pose, point_w, 0), 1e-9, 1e-10);
        ok &= CheckMatrix("line g2o/generated", Jline,
                          XPLSLAM::LineOnlyPoseJacobian(pose, line_w, line_measurement, 0), 1e-9, 1e-10);
        ok &= CheckMatrix("point right tangent", Jpoint,
            NumericalJacobian<2, 6>([&](const Vector6 &d) -> Eigen::Vector2d {
                return point_measurement - Retract(pose, d).ProjectPoint(point_w, 0);
            }));
        ok &= CheckMatrix("line right tangent", Jline,
            NumericalJacobian<2, 6>([&](const Vector6 &d) -> Eigen::Vector2d {
                const Eigen::Vector3d line = Retract(pose, d).ProjectLine(line_w, 0);
                return Eigen::Vector2d(XPLSLAM::Line2DError(line, line_measurement.head<2>()),
                                       XPLSLAM::Line2DError(line, line_measurement.tail<2>()));
            }));
        gb::generated::local_ba_point_jacobian_pose(
            pose.Rwb, pose.twb, point_w, rcb, tcb, model.fx, model.fy, model.cx, model.cy,
            point_measurement, Jba);
        ok &= CheckMatrix("BA/pose-only point", Jba, Jpoint, 1e-9, 1e-10);
        Eigen::Vector2d generated_point_residual;
        gb::generated::local_ba_point_residual(
            pose.Rwb, pose.twb, point_w, rcb, tcb, model.fx, model.fy, model.cx, model.cy,
            point_measurement, generated_point_residual);
        ok &= CheckVectorNear("BA point residual", generated_point_residual,
                              point_measurement - pose.ProjectPoint(point_w, 0));
        Eigen::Matrix<double, 2, 3> Jlandmark;
        gb::generated::local_ba_point_jacobian_point(
            pose.Rwb, pose.twb, point_w, rcb, tcb, model.fx, model.fy, model.cx, model.cy,
            point_measurement, Jlandmark);
        ok &= CheckMatrix("point landmark", Jlandmark,
            NumericalJacobian<2, 3>([&](const Eigen::Vector3d &d) -> Eigen::Vector2d {
                return point_measurement - pose.ProjectPoint(point_w + d, 0);
            }));

        XPLSLAM::BaseLine line;
        const double mn = line_w.head<3>().norm(), dn = line_w.tail<3>().norm();
        line.U.col(0) = line_w.head<3>() / mn;
        line.U.col(1) = line_w.tail<3>() / dn;
        line.U.col(2) = line.U.col(0).cross(line.U.col(1));
        line.W << mn, -dn, dn, mn;
        line.W /= std::hypot(mn, dn);
        line.its = 0;
        for (const Eigen::Vector4d &delta :
             {Eigen::Vector4d(0, 0, 0, 0), Eigen::Vector4d(0.08, -0.05, 0.03, 0.04)}) {
            XPLSLAM::BaseLine current = line;
            current.Update(delta.data());
            gb::generated::local_ba_line_jacobian_pose(
                pose.Rwb, pose.twb, line.U, line.W, delta, rcb, tcb,
                model.fx, model.fy, model.cx, model.cy, line_measurement, Jba, line_2d);
            ok &= CheckMatrix("BA/pose-only line", Jba,
                XPLSLAM::LineOnlyPoseJacobian(pose, current.lineW, line_measurement, 0), 1e-9, 1e-10);
            Eigen::Vector2d generated_line_residual;
            gb::generated::local_ba_line_residual(
                pose.Rwb, pose.twb, line.U, line.W, delta, rcb, tcb,
                model.fx, model.fy, model.cx, model.cy, line_measurement, generated_line_residual, line_2d);
            ok &= CheckVectorNear("BA line residual", generated_line_residual,
                gb::PoseOnlyLineResidual(graphite_pose, model, current.lineW, line_measurement));
            Eigen::Matrix<double, 2, 4> Jlandmark_line;
            gb::generated::local_ba_line_jacobian_line(
                pose.Rwb, pose.twb, line.U, line.W, delta, rcb, tcb,
                model.fx, model.fy, model.cx, model.cy, line_measurement, Jlandmark_line, line_2d);
            ok &= CheckMatrix("BA line chart", Jlandmark_line,
                NumericalJacobian<2, 4>([&](const Eigen::Vector4d &d) -> Eigen::Vector2d {
                    XPLSLAM::BaseLine updated = line;
                    const Eigen::Vector4d chart = delta + d;
                    updated.Update(chart.data());
                    return gb::PoseOnlyLineResidual(graphite_pose, model, updated.lineW, line_measurement);
                }));
        }
    }
    return ok;
}

bool CheckEssentialGraphJacobians() {
    bool ok = true;
    const std::vector<Eigen::Vector3d> errors = {
        {0, 0, 0}, {1e-9, -2e-9, 3e-9}, {1e-7, -8e-8, 6e-8},
        {1e-6, -8e-7, 6e-7}, {1e-5, -8e-6, 6e-6},
        {1e-4, -8e-5, 6e-5},
        {0.45, -0.3, 0.2}, {3.10, 0.02, -0.01}, {0.01, 3.10, 0.02},
        {0.02, -0.01, 3.10}};
    const g2o::SE3Quat Twi(XPLSLAM::ExpSO3(Eigen::Vector3d(0.4, -0.3, 0.2)),
                          Eigen::Vector3d(1.2, -0.8, 0.6));
    const g2o::SE3Quat Twj(XPLSLAM::ExpSO3(Eigen::Vector3d(-0.2, 0.5, -0.3)),
                          Eigen::Vector3d(-0.6, 0.9, 0.7));
    for (size_t sample = 0; sample < errors.size(); ++sample) {
        const Eigen::Vector3d translation = sample == 0 ? Eigen::Vector3d::Zero().eval() :
            Eigen::Vector3d(0.6, -0.4, 0.3);
        const g2o::SE3Quat measurement =
            g2o::SE3Quat(XPLSLAM::ExpSO3(errors[sample]), translation) * Twj.inverse() * Twi;
        XPLSLAM::VertexPoseGraph vi, vj;
        vi.setEstimate(Twi); vj.setEstimate(Twj);
        XPLSLAM::EdgeEssentialGraphSE3 edge;
        edge.setVertex(0, &vi); edge.setVertex(1, &vj); edge.setMeasurement(measurement);
        g2o::JacobianWorkspace workspace;
        workspace.updateSize(&edge, true); workspace.allocate(); workspace.setZero();
        using EdgeBase = g2o::BaseBinaryEdge<6, g2o::SE3Quat,
            XPLSLAM::VertexPoseGraph, XPLSLAM::VertexPoseGraph>;
        static_cast<EdgeBase &>(edge).linearizeOplus(workspace);
        Matrix6 Ji, Jj;
        const Eigen::Matrix3d Ri = Twi.rotation().toRotationMatrix(), Rj = Twj.rotation().toRotationMatrix();
        const Eigen::AngleAxisd aa(measurement.rotation());
        const Eigen::Vector3d mr = aa.angle() * aa.axis();
        gb::generated::essential_graph_jacobian_i(
            Ri, Twi.translation(), Rj, Twj.translation(), mr, measurement.translation(), Ji);
        gb::generated::essential_graph_jacobian_j(
            Ri, Twi.translation(), Rj, Twj.translation(), mr, measurement.translation(), Jj);
        Vector6 residual;
        gb::generated::essential_graph_residual(
            Ri, Twi.translation(), Rj, Twj.translation(), mr, measurement.translation(), residual);
        edge.computeError();
        ok &= CheckMatrix("essential residual " + std::to_string(sample), residual, edge.error(), 1e-9, 1e-9);
        ok &= CheckMatrix("essential g2o/generated i " + std::to_string(sample), Ji, edge.jacobianOplusXn<0>());
        ok &= CheckMatrix("essential g2o/generated j " + std::to_string(sample), Jj, edge.jacobianOplusXn<1>());
        for (int vertex = 0; vertex < 2; ++vertex) {
            const Matrix6 numeric = NumericalJacobian<6, 6>([&](const Vector6 &delta) -> Vector6 {
                XPLSLAM::VertexPoseGraph perturbed;
                perturbed.setEstimate(vertex == 0 ? Twi : Twj);
                perturbed.oplus(delta.data());
                const g2o::SE3Quat Ti = vertex == 0 ? perturbed.estimate() : Twi;
                const g2o::SE3Quat Tj = vertex == 1 ? perturbed.estimate() : Twj;
                return (measurement * Ti.inverse() * Tj).log();
            });
            ok &= CheckMatrix("essential right tangent " + std::to_string(sample), vertex == 0 ? Ji : Jj, numeric);
        }
    }
    return ok;
}

} // namespace

int main() {
    XPLSLAM::Pinhole camera({520.0f, 515.0f, 320.0f, 240.0f});
    const XPLSLAM::BasePose reference_pose = MakeReferencePose(camera);
    const gb::Pose pose = MakeGraphitePose(reference_pose, 0);
    const gb::PinholeCamera graphite_camera = MakeGraphiteCamera(camera);

    bool ok = true;

    const Eigen::Vector3d point_w(0.1, -0.15, 4.0);
    const Eigen::Vector2d point_projection =
        reference_pose.ProjectPoint(point_w, 0);
    const Eigen::Vector2d point_measurement =
        point_projection + Eigen::Vector2d(1.25, -0.85);
    ok &= CheckVectorNear(
        "point projection",
        gb::ProjectWorldPoint(pose, graphite_camera, point_w),
        point_projection);
    ok &= CheckVectorNear(
        "pose-only point residual",
        gb::PoseOnlyPointResidual(pose, graphite_camera, point_w,
                                  point_measurement),
        point_measurement - point_projection);
    ok &= CheckBoolEqual("point positive depth",
                         gb::IsPointDepthPositive(pose, point_w),
                         reference_pose.isPointDepthPositive(point_w, 0));

    const Eigen::Vector3d line_p1(0.2, -0.1, 3.2);
    const Eigen::Vector3d line_p2(-0.15, 0.25, 4.1);
    const gb::Vector6d line_w = MakePluckerLine(line_p1, line_p2);
    const Eigen::Vector3d line_projection =
        reference_pose.ProjectLine(line_w, 0);
    const Eigen::Vector4d line_measurement =
        MakeLineObservation(line_projection) +
        Eigen::Vector4d(1.1, -0.7, -0.9, 1.3);

    Eigen::Vector2d expected_line_residual;
    expected_line_residual[0] =
        XPLSLAM::Line2DError(line_projection, line_measurement.head<2>());
    expected_line_residual[1] =
        XPLSLAM::Line2DError(line_projection, line_measurement.tail<2>());

    ok &= CheckVectorNear("line projection",
                          gb::ProjectWorldLine(pose, graphite_camera, line_w),
                          line_projection);
    ok &= CheckVectorNear(
        "pose-only line residual",
        gb::PoseOnlyLineResidual(pose, graphite_camera, line_w,
                                 line_measurement),
        expected_line_residual);

    ok &= CheckGeneratedJacobians(camera);
    ok &= CheckEssentialGraphJacobians();

    if (!ok) {
        return 1;
    }

    std::cout << "graphite_slam_residual_check passed\n";
    return 0;
}
