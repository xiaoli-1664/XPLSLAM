#include "g2o_types.h"
#include "geometric_camera.h"
#include "map_line.h"

#include <cmath>
#include <exception>

namespace XPLSLAM {

using namespace std;

namespace {

constexpr double kMinDepth = 1e-9;

GeometricCamera *FrameCamera(const VertexoSE3 *vSE3, int cam_id, int frame) {
    return frame == 1 ? vSE3->vpCameras1[cam_id] : vSE3->vpCameras2[cam_id];
}

void SE3CameraPose(const VertexoSE3 *vSE3, int cam_id, Eigen::Matrix3d &Rcb,
                   Eigen::Matrix3d &Rcw, Eigen::Vector3d &tcw) {
    const BaseSE3 &pose = vSE3->estimate();
    Rcb = vSE3->vTcb[cam_id].rotationMatrix();
    Rcw = Rcb * pose.Rbw;
    tcw = Rcb * pose.tbw + vSE3->vTcb[cam_id].translation();
}

void InverseSE3CameraPose(const VertexoSE3 *vSE3, int cam_id,
                          Eigen::Matrix3d &Rcb, Eigen::Matrix3d &Rcw,
                          Eigen::Vector3d &tcw) {
    const BaseSE3 &pose = vSE3->estimate();
    Rcb = vSE3->vTcb[cam_id].rotationMatrix();
    Rcw = Rcb * pose.Rbw.transpose();
    tcw = vSE3->vTcb[cam_id].translation() - Rcw * pose.tbw;
}

void LinearizeSE3ProjectLine(const VertexLine *vLine, const VertexoSE3 *vSE3,
                             int cam_id, int frame,
                             const Eigen::Vector4d &measurement,
                             Eigen::Matrix<double, 2, 4> &jacLine,
                             Eigen::Matrix<double, 2, 6> &jacPose,
                             Eigen::Vector3d &projected_line) {
    const BaseLine &line = vLine->estimate();

    Eigen::Matrix3d U = line.U;
    Eigen::Matrix2d W = line.W;
    Eigen::Vector3d u1 = U.col(0);
    Eigen::Vector3d u2 = U.col(1);
    Eigen::Vector3d u3 = U.col(2);
    double w1 = W(0, 0);
    double w2 = W(1, 0);

    Eigen::Matrix3d Rcb;
    Eigen::Matrix3d Rcw;
    Eigen::Vector3d tcw;
    SE3CameraPose(vSE3, cam_id, Rcb, Rcw, tcw);

    projected_line = vSE3->ProjectLine(line.lineW, cam_id, frame);
    const Eigen::Vector3d &l = projected_line;

    Eigen::Matrix<double, 2, 3> de_dl;
    de_dl.row(0) = Line2DJac(l, measurement.head<2>()).transpose();
    de_dl.row(1) = Line2DJac(l, measurement.tail<2>()).transpose();

    Eigen::Matrix<double, 2, 3> de_dnc =
        de_dl * FrameCamera(vSE3, cam_id, frame)->toPL_();

    Eigen::Matrix<double, 3, 6> dnc_dPose;
    dnc_dPose.block<3, 3>(0, 0) = -Rcw * Skew(line.lineW.head<3>()) -
                                  Skew(tcw) * Rcw * Skew(line.lineW.tail<3>());
    dnc_dPose.block<3, 3>(0, 3) = -Skew(Rcw * line.lineW.tail<3>()) * Rcb;

    Eigen::Matrix<double, 3, 6> dnc_dLw;
    dnc_dLw << Rcw, Skew(tcw) * Rcw;

    Eigen::Matrix<double, 6, 4> dLw_dUW;
    dLw_dUW.setZero();
    dLw_dUW.col(0).tail<3>() = w2 * u3;
    dLw_dUW.col(1).head<3>() = -w1 * u3;
    dLw_dUW.col(2) << w1 * u2, -w2 * u1;
    dLw_dUW.col(3) << -u1 * w2, u2 * w1;

    jacLine = de_dnc * dnc_dLw * dLw_dUW;
    jacPose = de_dnc * dnc_dPose;
}

void LinearizeInverseSE3ProjectLine(const VertexLine *vLine,
                                    const VertexoSE3 *vSE3, int cam_id,
                                    int frame,
                                    const Eigen::Vector4d &measurement,
                                    Eigen::Matrix<double, 2, 4> &jacLine,
                                    Eigen::Matrix<double, 2, 6> &jacPose,
                                    Eigen::Vector3d &projected_line) {
    const BaseLine &line = vLine->estimate();
    const BaseSE3 &pose = vSE3->estimate();

    Eigen::Matrix3d U = line.U;
    Eigen::Matrix2d W = line.W;
    Eigen::Vector3d u1 = U.col(0);
    Eigen::Vector3d u2 = U.col(1);
    Eigen::Vector3d u3 = U.col(2);
    double w1 = W(0, 0);
    double w2 = W(1, 0);

    Eigen::Matrix3d Rcb;
    Eigen::Matrix3d Rcw;
    Eigen::Vector3d tcw;
    InverseSE3CameraPose(vSE3, cam_id, Rcb, Rcw, tcw);

    projected_line =
        vSE3->ProjectLineInverse(line.lineW, cam_id, frame);
    const Eigen::Vector3d &l = projected_line;

    Eigen::Matrix<double, 2, 3> de_dl;
    de_dl.row(0) = Line2DJac(l, measurement.head<2>()).transpose();
    de_dl.row(1) = Line2DJac(l, measurement.tail<2>()).transpose();

    Eigen::Matrix<double, 2, 3> de_dnc =
        de_dl * FrameCamera(vSE3, cam_id, frame)->toPL_();

    const Eigen::Matrix3d Rwb = pose.Rbw.transpose();
    const Eigen::Vector3d nw_b = Rwb * line.lineW.head<3>();
    const Eigen::Vector3d dw_b = Rwb * line.lineW.tail<3>();
    const Eigen::Vector3d tw_b = Rwb * pose.tbw;
    const Eigen::Vector3d dc = Rcw * line.lineW.tail<3>();

    Eigen::Matrix<double, 3, 6> dnc_dPose;
    dnc_dPose.block<3, 3>(0, 0) = Rcb * Skew(nw_b) +
                                  Skew(dc) * Rcb * Skew(tw_b) +
                                  Skew(tcw) * Rcb * Skew(dw_b);
    dnc_dPose.block<3, 3>(0, 3) = Skew(dc) * Rcw;

    Eigen::Matrix<double, 3, 6> dnc_dLw;
    dnc_dLw << Rcw, Skew(tcw) * Rcw;

    Eigen::Matrix<double, 6, 4> dLw_dUW;
    dLw_dUW.setZero();
    dLw_dUW.col(0).tail<3>() = w2 * u3;
    dLw_dUW.col(1).head<3>() = -w1 * u3;
    dLw_dUW.col(2) << w1 * u2, -w2 * u1;
    dLw_dUW.col(3) << -u1 * w2, u2 * w1;

    jacLine = de_dnc * dnc_dLw * dLw_dUW;
    jacPose = de_dnc * dnc_dPose;
}

} // namespace

BasePose::BasePose(KeyFramePtr pKF) : its(0) {
    const Sophus::SE3d Twb = pKF->GetPoseInverse().cast<double>();
    Rwb = Twb.rotationMatrix();
    twb = Twb.translation();

    int n_cams = pKF->mnCam;

    tcb.resize(n_cams);
    Rcb.resize(n_cams);
    vpCameras.resize(n_cams);

    vector<Sophus::SE3f> vTbc = pKF->GetCameraExtrinsics();

    for (int c = 0; c < n_cams; ++c) {
        vpCameras[c] = pKF->mvpCameras[c];

        Sophus::SE3d Tcb = vTbc[c].inverse().cast<double>();
        Rcb[c] = Tcb.rotationMatrix();
        tcb[c] = Tcb.translation();
    }
}

BasePose::BasePose(Frame *pF) : its(0) {
    const Sophus::SE3d Twb = pF->GetPose().cast<double>().inverse();
    Rwb = Twb.rotationMatrix();
    twb = Twb.translation();

    int n_cams = pF->mnCam;

    tcb.resize(n_cams);
    Rcb.resize(n_cams);
    vpCameras.resize(n_cams);

    vector<Sophus::SE3f> vTbc = pF->GetCameraExtrinsics();

    for (int c = 0; c < n_cams; ++c) {
        vpCameras[c] = pF->mvpCamera[c];

        Sophus::SE3d Tcb = vTbc[c].inverse().cast<double>();
        Rcb[c] = Tcb.rotationMatrix();
        tcb[c] = Tcb.translation();
    }
}

void BasePose::Update(const double *pu) {
    Eigen::Vector3d w(pu[0], pu[1], pu[2]);
    Eigen::Vector3d t(pu[3], pu[4], pu[5]);

    Rwb = Rwb * ExpSO3(w);
    twb = twb + t;

    its++;
    if (its >= 3) {
        Rwb = NormalizeRotation(Rwb);
        its = 0;
    }
}

void BaseSE3::Update(const double *pu) {
    Eigen::Vector3d w(pu[0], pu[1], pu[2]);
    Eigen::Vector3d t(pu[3], pu[4], pu[5]);

    Rbw = Rbw * ExpSO3(w);
    tbw = tbw + t;

    its++;
    if (its >= 3) {
        Rbw = NormalizeRotation(Rbw);
        its = 0;
    }
}

Eigen::Vector2d BasePose::ProjectPoint(const Eigen::Vector3d &Xw,
                                       const int cam_id) const {
    Eigen::Vector3d Xc = CameraRotation(cam_id) * Xw + CameraTranslation(cam_id);

    return vpCameras[cam_id]->project(Xc);
}

Eigen::Vector2d VertexoSE3::ProjectPoint(const Eigen::Vector3d &Xw,
                                         const int cam_id, int frame) const {
    Eigen::Vector3d Xc = vTcb[cam_id] * (_estimate.Rbw * Xw + _estimate.tbw);
    return frame == 1 ? vpCameras1[cam_id]->project(Xc)
                      : vpCameras2[cam_id]->project(Xc);
}

Eigen::Vector2d VertexoSE3::ProjectPointInverse(const Eigen::Vector3d &Xw,
                                                const int cam_id,
                                                int frame) const {
    Eigen::Vector3d Xc =
        vTcb[cam_id] * (_estimate.Rbw.transpose() * (Xw - _estimate.tbw));
    return frame == 1 ? vpCameras1[cam_id]->project(Xc)
                      : vpCameras2[cam_id]->project(Xc);
}

Eigen::Vector3d BasePose::ProjectLine(const Eigen::Matrix<double, 6, 1> &line_w,
                                      const int cam_id) const {
    Eigen::Matrix3d Rcw = CameraRotation(cam_id);
    Eigen::Vector3d tcw = CameraTranslation(cam_id);

    Eigen::Matrix<double, 6, 1> line_c;
    line_c.head<3>() =
        Rcw * line_w.head<3>() + tcw.cross(Rcw * line_w.tail<3>());
    line_c.tail<3>() = Rcw * line_w.tail<3>();
    return vpCameras[cam_id]->projectLine(line_c);
}

Eigen::Vector3d
VertexoSE3::ProjectLine(const Eigen::Matrix<double, 6, 1> &line_w,
                        const int cam_id, int frame) const {
    Eigen::Matrix3d Rcw = vTcb[cam_id].rotationMatrix() * _estimate.Rbw;
    Eigen::Vector3d tcw = vTcb[cam_id].rotationMatrix() * _estimate.tbw +
                          vTcb[cam_id].translation();

    Eigen::Matrix<double, 6, 1> line_c;
    line_c.head<3>() =
        Rcw * line_w.head<3>() + tcw.cross(Rcw * line_w.tail<3>());
    line_c.tail<3>() = Rcw * line_w.tail<3>();
    return frame == 1 ? vpCameras1[cam_id]->projectLine(line_c)
                      : vpCameras2[cam_id]->projectLine(line_c);
}

Eigen::Vector3d
VertexoSE3::ProjectLineInverse(const Eigen::Matrix<double, 6, 1> &line_w,
                               const int cam_id, int frame) const {
    Eigen::Matrix3d Rcw =
        vTcb[cam_id].rotationMatrix() * _estimate.Rbw.transpose();
    Eigen::Vector3d tcw = vTcb[cam_id].translation() - Rcw * _estimate.tbw;

    Eigen::Matrix<double, 6, 1> line_c;
    line_c.head<3>() =
        Rcw * line_w.head<3>() + tcw.cross(Rcw * line_w.tail<3>());
    line_c.tail<3>() = Rcw * line_w.tail<3>();
    return frame == 1 ? vpCameras1[cam_id]->projectLine(line_c)
                      : vpCameras2[cam_id]->projectLine(line_c);
}

bool BasePose::isPointDepthPositive(const Eigen::Vector3d &Xw,
                                    const int cam_id) const {
    const Eigen::Vector3d point_c =
        CameraRotation(cam_id) * Xw + CameraTranslation(cam_id);
    return point_c.allFinite() && point_c.z() > 0.0;
}

bool BasePose::isLineDepthPositive(const Eigen::Matrix<double, 6, 1> &line_w,
                                   const int cam_id,
                                   const Eigen::Vector4d &line_obs) const {
    Eigen::Matrix3d Rcw = CameraRotation(cam_id);
    Eigen::Vector3d tcw = CameraTranslation(cam_id);

    Eigen::Matrix<double, 6, 1> line_c;
    line_c.head<3>() =
        Rcw * line_w.head<3>() + tcw.cross(Rcw * line_w.tail<3>());
    line_c.tail<3>() = Rcw * line_w.tail<3>();

    Eigen::Vector3d line_obs_c1 =
        vpCameras[cam_id]->unprojectEig(line_obs.head<2>());
    Eigen::Vector3d line_obs_c2 =
        vpCameras[cam_id]->unprojectEig(line_obs.tail<2>());

    Eigen::Vector3d point_c1;
    Eigen::Vector3d point_c2;
    if (!GeometricTools::LinePointFromBackprojectedRay(line_c, line_obs_c1,
                                                       point_c1) ||
        !GeometricTools::LinePointFromBackprojectedRay(line_c, line_obs_c2,
                                                       point_c2)) {
        return false;
    }

    return point_c1.z() > kMinDepth && point_c2.z() > kMinDepth;
}

BaseLine::BaseLine(MapLinePtr pML) : its(0) {
    lineW = pML->GetWorldPos().cast<double>();

    Eigen::Vector3d m = lineW.head<3>();
    Eigen::Vector3d d = lineW.tail<3>();

    double m_norm = m.norm();
    double d_norm = d.norm();
    double md_norm = hypot(m_norm, d_norm);

    U.col(0) = m / m_norm;
    U.col(1) = d / d_norm;
    U.col(2) = (m.cross(d)).normalized();

    W << m_norm, -d_norm, d_norm, m_norm;
    W /= md_norm;

    lineW.head<3>() = U.col(0) * W(0, 0);
    lineW.tail<3>() = U.col(1) * W(1, 0);
}

void BaseLine::Update(const double *pu) {
    Eigen::Vector3d theta(pu[0], pu[1], pu[2]);

    double phi = pu[3];

    U = U * ExpSO3(theta);
    W = W * ExpSO2(phi);

    its++;
    if (its >= 3) {
        U = NormalizeRotation(U);

        its = 0;
    }

    lineW.head<3>() = U.col(0) * W(0, 0);
    lineW.tail<3>() = U.col(1) * W(1, 0);
}

namespace {
Eigen::Matrix<double, 3, 6> LineCameraPoseJacobian(
    const BasePose &pose, const Vector6d &line_w, int cam_id) {
    const Eigen::Vector3d db = pose.Rwb.transpose() * line_w.tail<3>();
    const Eigen::Vector3d nb = pose.Rwb.transpose() *
        (line_w.head<3>() - pose.twb.cross(line_w.tail<3>()));
    Eigen::Matrix<double, 3, 6> J;
    J.leftCols<3>() = pose.Rcb[cam_id] * Skew(nb) +
        Skew(pose.tcb[cam_id]) * pose.Rcb[cam_id] * Skew(db);
    J.rightCols<3>() = pose.CameraRotation(cam_id) * Skew(line_w.tail<3>());
    return J;
}
} // namespace

void EdgeProjectPointOnlyPose::linearizeOplus() {
    const VertexPose *vPose = static_cast<const VertexPose *>(_vertices[0]);
    _jacobianOplusXi = PointOnlyPoseJacobian(vPose->estimate(), Xw, cam_id);
}

Eigen::Matrix<double, 2, 6>
PointOnlyPoseJacobian(const BasePose &pose, const Eigen::Vector3d &Xw,
                      int cam_id) {
    const Eigen::Matrix3d Rcw = pose.CameraRotation(cam_id);
    const Eigen::Vector3d tcw =
        pose.CameraTranslation(cam_id);
    const Eigen::Vector3d Xc = Rcw * Xw + tcw;

    Eigen::Matrix<double, 3, 6> dXc_dPose;
    const Eigen::Vector3d Xb = pose.Rwb.transpose() * (Xw - pose.twb);
    dXc_dPose << pose.Rcb[cam_id] * Skew(Xb), -Rcw;
    return -pose.vpCameras[cam_id]->projectJac(Xc) * dXc_dPose;
}

void EdgeProjectPoint::linearizeOplus() {
    const g2o::VertexPointXYZ *vPoint =
        static_cast<const g2o::VertexPointXYZ *>(_vertices[0]);
    const VertexPose *vPose = static_cast<const VertexPose *>(_vertices[1]);
    const BasePose &pose = vPose->estimate();
    Eigen::Vector3d Xw = vPoint->estimate();

    Eigen::Matrix3d Rcw = pose.CameraRotation(cam_id);
    Eigen::Vector3d tcw = pose.CameraTranslation(cam_id);

    Eigen::Vector3d Xc = Rcw * Xw + tcw;

    Eigen::Matrix<double, 3, 6> dXc_dPose;
    const Eigen::Vector3d Xb = pose.Rwb.transpose() * (Xw - pose.twb);
    dXc_dPose << pose.Rcb[cam_id] * Skew(Xb), -Rcw;

    _jacobianOplusXi = -pose.vpCameras[cam_id]->projectJac(Xc) * Rcw;
    _jacobianOplusXj = -pose.vpCameras[cam_id]->projectJac(Xc) * dXc_dPose;
}

void EdgeProjectLineOnlyPose::linearizeOplus() {
    const VertexPose *vPose = static_cast<const VertexPose *>(_vertices[0]);
    const BasePose &pose = vPose->estimate();

    Eigen::Vector3d l = pose.ProjectLine(line_w, cam_id);
    if (hasEndpointCovariance()) {
        setInformation(lineResidualInformation(l));
    }

    _jacobianOplusXi =
        LineOnlyPoseJacobian(pose, line_w, _measurement, cam_id);
}

Eigen::Matrix<double, 2, 6>
LineOnlyPoseJacobian(const BasePose &pose,
                     const Eigen::Matrix<double, 6, 1> &line_w,
                     const Eigen::Vector4d &measurement, int cam_id) {
    const Eigen::Vector3d l = pose.ProjectLine(line_w, cam_id);

    Eigen::Matrix<double, 2, 3> de_dl;
    de_dl.row(0) = Line2DJac(l, measurement.head<2>()).transpose();
    de_dl.row(1) = Line2DJac(l, measurement.tail<2>()).transpose();

    const Eigen::Matrix<double, 3, 6> dnc_dPose =
        LineCameraPoseJacobian(pose, line_w, cam_id);
    return de_dl * pose.vpCameras[cam_id]->toPL_() * dnc_dPose;
}

void EdgeProjectLine::linearizeOplus() {
    const VertexLine *vLine = static_cast<const VertexLine *>(_vertices[0]);
    const VertexPose *vPose = static_cast<const VertexPose *>(_vertices[1]);
    const BasePose &pose = vPose->estimate();
    const BaseLine &line = vLine->estimate();

    Eigen::Matrix3d U = line.U;
    Eigen::Matrix2d W = line.W;
    Eigen::Vector3d u1 = U.col(0);
    Eigen::Vector3d u2 = U.col(1);
    Eigen::Vector3d u3 = U.col(2);
    double w1 = W(0, 0);
    double w2 = W(1, 0);

    Eigen::Matrix3d Rcw = pose.CameraRotation(cam_id);
    Eigen::Vector3d tcw = pose.CameraTranslation(cam_id);

    Eigen::Vector3d l = pose.ProjectLine(line.lineW, cam_id);
    if (hasEndpointCovariance()) {
        setInformation(lineResidualInformation(l));
    }

    Eigen::Matrix<double, 2, 3> de_dl;
    de_dl.row(0) = Line2DJac(l, _measurement.head<2>()).transpose();
    de_dl.row(1) = Line2DJac(l, _measurement.tail<2>()).transpose();

    Eigen::Matrix3d dl_dnc = pose.vpCameras[cam_id]->toPL_();

    Eigen::Matrix<double, 2, 3> de_dnc = de_dl * dl_dnc;

    const Eigen::Matrix<double, 3, 6> dnc_dPose =
        LineCameraPoseJacobian(pose, line.lineW, cam_id);

    Eigen::Matrix<double, 3, 6> dnc_dLw;
    dnc_dLw << Rcw, Skew(tcw) * Rcw;

    Eigen::Matrix<double, 6, 4> dLw_dUW;
    dLw_dUW.setZero();
    dLw_dUW.col(0).tail<3>() = w2 * u3;
    dLw_dUW.col(1).head<3>() = -w1 * u3;
    dLw_dUW.col(2) << w1 * u2, -w2 * u1;
    dLw_dUW.col(3) << -u1 * w2, u2 * w1;

    _jacobianOplusXi = de_dnc * dnc_dLw * dLw_dUW;
    _jacobianOplusXj = de_dnc * dnc_dPose;
}

void EdgeSE3ProjectPoint::linearizeOplus() {
    const g2o::VertexPointXYZ *vPoint =
        static_cast<const g2o::VertexPointXYZ *>(_vertices[0]);
    const VertexoSE3 *vSE3 = static_cast<const VertexoSE3 *>(_vertices[1]);
    Eigen::Vector3d Xw = vPoint->estimate();

    Eigen::Matrix3d Rcb;
    Eigen::Matrix3d Rcw;
    Eigen::Vector3d tcw;
    SE3CameraPose(vSE3, cam_id, Rcb, Rcw, tcw);

    Eigen::Vector3d Xc = Rcw * Xw + tcw;

    Eigen::Matrix<double, 3, 6> dXc_dPose;
    dXc_dPose << -Rcw * Skew(Xw), Rcb;

    Eigen::Matrix<double, 2, 3> jacProj =
        FrameCamera(vSE3, cam_id, 1)->projectJac(Xc);
    _jacobianOplusXi = -jacProj * Rcw;
    _jacobianOplusXj = -jacProj * dXc_dPose;
}

void EdgeInverseSE3ProjectPoint::linearizeOplus() {
    const g2o::VertexPointXYZ *vPoint =
        static_cast<const g2o::VertexPointXYZ *>(_vertices[0]);
    const VertexoSE3 *vSE3 = static_cast<const VertexoSE3 *>(_vertices[1]);
    Eigen::Vector3d Xw = vPoint->estimate();

    Eigen::Matrix3d Rcb;
    Eigen::Matrix3d Rcw;
    Eigen::Vector3d tcw;
    InverseSE3CameraPose(vSE3, cam_id, Rcb, Rcw, tcw);

    Eigen::Vector3d Xc = Rcw * Xw + tcw;

    Eigen::Matrix<double, 3, 6> dXc_dPose;
    dXc_dPose << Rcb * Skew(vSE3->estimate().Rbw.transpose() *
                            (Xw - vSE3->estimate().tbw)),
        -Rcw;

    Eigen::Matrix<double, 2, 3> jacProj =
        FrameCamera(vSE3, cam_id, 2)->projectJac(Xc);
    _jacobianOplusXi = -jacProj * Rcw;
    _jacobianOplusXj = -jacProj * dXc_dPose;
}

void EdgeSE3ProjectLine::linearizeOplus() {
    const VertexLine *vLine = static_cast<const VertexLine *>(_vertices[0]);
    const VertexoSE3 *vSE3 = static_cast<const VertexoSE3 *>(_vertices[1]);
    Eigen::Matrix<double, 2, 4> jacLine;
    Eigen::Matrix<double, 2, 6> jacPose;
    Eigen::Vector3d projected_line;
    LinearizeSE3ProjectLine(vLine, vSE3, cam_id, 1, _measurement, jacLine,
                            jacPose, projected_line);
    if (hasEndpointCovariance()) {
        setInformation(lineResidualInformation(projected_line));
    }
    _jacobianOplusXi = jacLine;
    _jacobianOplusXj = jacPose;
}

void EdgeInverseSE3ProjectLine::linearizeOplus() {
    const VertexLine *vLine = static_cast<const VertexLine *>(_vertices[0]);
    const VertexoSE3 *vSE3 = static_cast<const VertexoSE3 *>(_vertices[1]);
    Eigen::Matrix<double, 2, 4> jacLine;
    Eigen::Matrix<double, 2, 6> jacPose;
    Eigen::Vector3d projected_line;
    LinearizeInverseSE3ProjectLine(vLine, vSE3, cam_id, 2, _measurement,
                                   jacLine, jacPose, projected_line);
    if (hasEndpointCovariance()) {
        setInformation(lineResidualInformation(projected_line));
    }
    _jacobianOplusXi = jacLine;
    _jacobianOplusXj = jacPose;
}

void EdgeEssentialGraphSE3::linearizeOplus() {
    const auto *vi = static_cast<const VertexPoseGraph *>(_vertices[0]);
    const auto *vj = static_cast<const VertexPoseGraph *>(_vertices[1]);
    const g2o::SE3Quat &Twi = vi->estimate();
    const g2o::SE3Quat &Twj = vj->estimate();
    const Vector6d error = (_measurement * Twi.inverse() * Twj).log();

    // Sophus orders [translation, rotation]; g2o and these vertices use
    // [rotation, translation]. Permute both rows and columns.
    auto inverse_left = [](const Vector6d &e) {
        if (e.head<3>().squaredNorm() < 1e-6) {
            // Stable Bernoulli expansion near zero: the closed-form SE(3)
            // cross block subtracts terms proportional to theta^-4.
            Eigen::Matrix<double, 6, 6> ad = Eigen::Matrix<double, 6, 6>::Zero();
            ad.topLeftCorner<3, 3>() = Skew(e.head<3>());
            ad.bottomRightCorner<3, 3>() = ad.topLeftCorner<3, 3>();
            ad.bottomLeftCorner<3, 3>() = Skew(e.tail<3>());
            const Eigen::Matrix<double, 6, 6> ad2 = ad * ad;
            const Eigen::Matrix<double, 6, 6> ad4 = ad2 * ad2;
            Eigen::Matrix<double, 6, 6> result =
                Eigen::Matrix<double, 6, 6>::Identity() - 0.5 * ad +
                ad2 / 12.0 - ad4 / 720.0 + ad4 * ad2 / 30240.0;
            return result;
        }
        Vector6d sophus_e;
        sophus_e << e.tail<3>(), e.head<3>();
        const Eigen::Matrix<double, 6, 6> original =
            Sophus::SE3d::leftJacobianInverse(sophus_e);
        Eigen::Matrix<double, 6, 6> result;
        result << original.bottomRightCorner<3, 3>(),
                  original.bottomLeftCorner<3, 3>(),
                  original.topRightCorner<3, 3>(),
                  original.topLeftCorner<3, 3>();
        return result;
    };
    Eigen::Matrix<double, 6, 6> Di = Eigen::Matrix<double, 6, 6>::Identity();
    Eigen::Matrix<double, 6, 6> Dj = Di;
    Di.bottomRightCorner<3, 3>() = Twi.rotation().toRotationMatrix().transpose();
    Dj.bottomRightCorner<3, 3>() = Twj.rotation().toRotationMatrix().transpose();
    _jacobianOplusXi = -inverse_left(error) * _measurement.adj() * Di;
    _jacobianOplusXj = inverse_left(-error) * Dj;
}

Eigen::Matrix3d ExpSO3(const Eigen::Vector3d &w) {
    return ExpSO3(w[0], w[1], w[2]);
}

Eigen::Matrix3d ExpSO3(const double x, const double y, const double z) {
    const double d2 = x * x + y * y + z * z;
    const double d = sqrt(d2);
    Eigen::Matrix3d W;
    W << 0.0, -z, y, z, 0.0, -x, -y, x, 0.0;
    if (d < 1e-5) {
        Eigen::Matrix3d res = Eigen::Matrix3d::Identity() + W + 0.5 * W * W;
        return NormalizeRotation(res);
    } else {
        Eigen::Matrix3d res = Eigen::Matrix3d::Identity() + W * sin(d) / d +
                              W * W * (1.0 - cos(d)) / d2;
        return NormalizeRotation(res);
    }
}

Eigen::Matrix2d ExpSO2(const double theta) {
    Eigen::Matrix2d R;
    R << cos(theta), -sin(theta), sin(theta), cos(theta);
    return R;
}

Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R) {
    const double tr = R(0, 0) + R(1, 1) + R(2, 2);
    Eigen::Vector3d w;
    w << (R(2, 1) - R(1, 2)) / 2, (R(0, 2) - R(2, 0)) / 2,
        (R(1, 0) - R(0, 1)) / 2;
    const double costheta = (tr - 1.0) * 0.5f;
    if (costheta > 1 || costheta < -1)
        return w;
    const double theta = acos(costheta);
    const double s = sin(theta);
    if (fabs(s) < 1e-5)
        return w;
    else
        return theta * w / s;
}

Eigen::Matrix3d InverseRightJacobianSO3(const Eigen::Vector3d &v) {
    return InverseRightJacobianSO3(v[0], v[1], v[2]);
}

Eigen::Matrix3d InverseRightJacobianSO3(const double x, const double y,
                                        const double z) {
    const double d2 = x * x + y * y + z * z;
    const double d = sqrt(d2);

    Eigen::Matrix3d W;
    W << 0.0, -z, y, z, 0.0, -x, -y, x, 0.0;
    if (d < 1e-5)
        return Eigen::Matrix3d::Identity();
    else
        return Eigen::Matrix3d::Identity() + W / 2 +
               W * W * (1.0 / d2 - (1.0 + cos(d)) / (2.0 * d * sin(d)));
}

Eigen::Matrix3d RightJacobianSO3(const Eigen::Vector3d &v) {
    return RightJacobianSO3(v[0], v[1], v[2]);
}

Eigen::Matrix3d RightJacobianSO3(const double x, const double y,
                                 const double z) {
    const double d2 = x * x + y * y + z * z;
    const double d = sqrt(d2);

    Eigen::Matrix3d W;
    W << 0.0, -z, y, z, 0.0, -x, -y, x, 0.0;
    if (d < 1e-5) {
        return Eigen::Matrix3d::Identity();
    } else {
        return Eigen::Matrix3d::Identity() - W * (1.0 - cos(d)) / d2 +
               W * W * (d - sin(d)) / (d2 * d);
    }
}

Eigen::Matrix3d Skew(const Eigen::Vector3d &w) {
    Eigen::Matrix3d W;
    W << 0.0, -w[2], w[1], w[2], 0.0, -w[0], -w[1], w[0], 0.0;
    return W;
}

Eigen::Vector3d Line2DJac(const Eigen::Vector3d &l, const Eigen::Vector2d &x) {
    double s = hypot(l[0], l[1]);
    double s3 = s * s * s;
    double d = l[0] * x[0] + l[1] * x[1] + l[2];

    return {x[0] / s - l[0] * d / s3, x[1] / s - l[1] * d / s3, 1.0 / s};
}

double Line2DError(const Eigen::Vector3d &l, const Eigen::Vector2d &x) {
    double s = hypot(l[0], l[1]);
    double d = l[0] * x[0] + l[1] * x[1] + l[2];
    return d / s;
}

}; // namespace XPLSLAM
