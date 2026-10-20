#include "runtime_timing.h"
#include "optimizer.h"

#include "edge_se3.h"
#include "edge_se3_expmap.h"
#include "g2o/core/eigen_types.h"
#include "g2o_types.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include "converter.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/sparse_block_matrix.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/types/sim3/types_seven_dof_expmap.h"
#include "se3quat.h"
#include "sophus/se3.hpp"
#include "vertex_pointxyz.h"
#include "vertex_se3_expmap.h"

#ifdef GRAPHITE
#include "camera_models/pinhole.h"
#include "graphite_slam_backend.h"
#endif

namespace XPLSLAM {

using namespace std;

namespace {

using Matrix6d = Eigen::Matrix<double, 6, 6>;

template <typename Edge>
bool IsReprojectionOutlier(Edge *edge, double chi2_threshold) {
    // Also refresh excluded edges at the current estimate before classifying.
    edge->computeError();
    const double chi2 = edge->chi2();
    return !std::isfinite(chi2) || chi2 > chi2_threshold ||
           !edge->isDepthPositive();
}

bool IsFinitePose(const BasePose &pose) {
    return pose.Rwb.cast<float>().allFinite() &&
           pose.twb.cast<float>().allFinite();
}

bool HasFiniteLocalBAEstimates(const g2o::SparseOptimizer &optimizer) {
    for (const auto &entry : optimizer.vertices()) {
        if (const auto *pose = dynamic_cast<const VertexPose *>(entry.second)) {
            if (!IsFinitePose(pose->estimate())) {
                return false;
            }
        } else if (const auto *point =
                       dynamic_cast<const g2o::VertexPointXYZ *>(entry.second)) {
            if (!point->estimate().cast<float>().allFinite()) {
                return false;
            }
        } else if (const auto *line =
                       dynamic_cast<const VertexLine *>(entry.second)) {
            if (!line->estimate().lineW.cast<float>().allFinite()) {
                return false;
            }
        }
    }
    return true;
}

double MinimumEigenvalue(const Matrix6d &hessian, bool &valid) {
    Eigen::SelfAdjointEigenSolver<Matrix6d> solver(hessian);
    valid = solver.info() == Eigen::Success && solver.eigenvalues().allFinite();
    return valid ? solver.eigenvalues()(0) : 0.0;
}

Eigen::Matrix<double, 6, 1> HessianColumnScale(const Matrix6d &hessian,
                                               bool &valid) {
    Eigen::Matrix<double, 6, 1> scale;
    const double max_diag = std::max(1.0, hessian.diagonal().maxCoeff());
    const double eps = 1e-9 * max_diag;
    valid = hessian.allFinite() && std::isfinite(eps);
    for (int i = 0; i < 6; ++i) {
        const double diagonal = hessian(i, i);
        if (!std::isfinite(diagonal) || diagonal < 0.0) {
            valid = false;
        }
        scale(i) = 1.0 / std::sqrt(std::max(diagonal, eps));
    }
    valid = valid && scale.allFinite();
    return scale;
}

Matrix6d ScaledHessian(const Matrix6d &hessian,
                       const Eigen::Matrix<double, 6, 1> &scale) {
    return scale.asDiagonal() * hessian * scale.asDiagonal();
}

template <typename FrameLike>
bool GetLineEndpointCovariance(const FrameLike &frame, size_t cam_id, int idx,
                               PackedLineEndpointCovariance &covariance) {
    if (idx < 0 || cam_id >= frame.mvLineEndpointCovariances_cams.size() ||
        static_cast<size_t>(idx) >=
            frame.mvLineEndpointCovariances_cams[cam_id].size()) {
        return false;
    }
    return PackLineEndpointCovariance(
        frame.mvLineEndpointCovariances_cams[cam_id][idx], covariance);
}

template <typename Edge, typename FrameLike>
void SetLineEndpointCovariance(Edge *edge, const FrameLike &frame,
                               size_t cam_id, int idx) {
    PackedLineEndpointCovariance covariance;
    if (GetLineEndpointCovariance(frame, cam_id, idx, covariance)) {
        edge->setEndpointCovariance(covariance);
    }
}

#ifdef GRAPHITE
template <typename FrameLike>
void SetFlatLineEndpointCovariance(
    PackedLineEndpointCovariance &packed,
    uint8_t &has_covariance, const FrameLike &frame, size_t cam_id, int idx) {
    has_covariance =
        GetLineEndpointCovariance(frame, cam_id, idx, packed) ? 1 : 0;
}
#endif

Eigen::Matrix<double, 6, 1>
TransformLineToBody(const Eigen::Matrix<float, 6, 1> &line_w,
                    const Eigen::Matrix3f &Rbw, const Eigen::Vector3f &tbw) {
    Eigen::Matrix<float, 6, 1> line_b;
    line_b.head<3>() =
        Rbw * line_w.head<3>() + tbw.cross(Rbw * line_w.tail<3>());
    line_b.tail<3>() = Rbw * line_w.tail<3>();
    return line_b.cast<double>();
}

bool MakeBaseLineEstimate(const Eigen::Matrix<double, 6, 1> &line_w,
                          BaseLine &line) {
    const Eigen::Vector3d moment = line_w.head<3>();
    const Eigen::Vector3d direction = line_w.tail<3>();
    const double moment_norm = moment.norm();
    const double direction_norm = direction.norm();
    const double md_norm = std::hypot(moment_norm, direction_norm);

    if (moment_norm < 1e-9 || direction_norm < 1e-9 || md_norm < 1e-9 ||
        !line_w.allFinite()) {
        return false;
    }

    line.U.col(0) = moment / moment_norm;
    line.U.col(1) = direction / direction_norm;
    line.U.col(2) = moment.cross(direction).normalized();
    line.W << moment_norm, -direction_norm, direction_norm, moment_norm;
    line.W /= md_norm;
    line.lineW = line_w / md_norm;
    return line.U.allFinite() && line.W.allFinite() && line.lineW.allFinite();
}

g2o::SE3Quat ToG2OSE3Quat(const Sophus::SE3d &T) {
    return g2o::SE3Quat(T.unit_quaternion(), T.translation());
}

Sophus::SE3d ToSophusSE3(const g2o::SE3Quat &T) {
    return Sophus::SE3d(
        NormalizeRotation<double>(T.rotation().toRotationMatrix()),
        T.translation());
}

g2o::Sim3 ToG2OSim3(const g2o::SE3Quat &T) {
    return g2o::Sim3(T.rotation(), T.translation(), 1.0);
}

#ifdef GRAPHITE
graphite_backend::FlatPoseVertex ToFlatPoseVertexFromTbw(const g2o::SE3Quat &Tbw) {
    graphite_backend::FlatPoseVertex flat;
    const g2o::SE3Quat Twb = Tbw.inverse();
    const Eigen::Matrix3d rotation = Twb.rotation().toRotationMatrix();
    const Eigen::Vector3d translation = Twb.translation();
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            flat.Rwb[static_cast<size_t>(3 * row + col)] = rotation(row, col);
        }
        flat.twb[static_cast<size_t>(row)] = translation[row];
    }
    return flat;
}

g2o::SE3Quat TbwFromFlatPoseVertex(const graphite_backend::FlatPoseVertex &pose) {
    Eigen::Matrix3d rotation;
    Eigen::Vector3d translation;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            rotation(row, col) = pose.Rwb[static_cast<size_t>(3 * row + col)];
        }
        translation[row] = pose.twb[static_cast<size_t>(row)];
    }
    return g2o::SE3Quat(NormalizeRotation<double>(rotation), translation).inverse();
}

std::array<double, 9> FlattenMatrix(const Eigen::Matrix3d &matrix) {
    std::array<double, 9> values{};
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            values[static_cast<size_t>(3 * row + col)] = matrix(row, col);
        }
    }
    return values;
}

std::array<double, 3> FlattenRotationVector(const Eigen::Matrix3d &matrix) {
    Eigen::Quaterniond q(matrix);
    q.normalize();
    const Eigen::AngleAxisd angle_axis(q);
    if (!std::isfinite(angle_axis.angle()) || angle_axis.angle() == 0.0) {
        return {0.0, 0.0, 0.0};
    }
    const Eigen::Vector3d rotation = angle_axis.axis() * angle_axis.angle();
    return {rotation[0], rotation[1], rotation[2]};
}

std::array<double, 3> FlattenVector3(const Eigen::Vector3d &vector) {
    return {vector[0], vector[1], vector[2]};
}

std::array<double, 6>
FlattenVector6(const Eigen::Matrix<double, 6, 1> &vector) {
    return {vector[0], vector[1], vector[2], vector[3], vector[4], vector[5]};
}

graphite_backend::FlatPoseVertex ToFlatPoseVertex(const BasePose &pose) {
    graphite_backend::FlatPoseVertex flat;
    flat.Rwb = FlattenMatrix(pose.Rwb);
    flat.twb = FlattenVector3(pose.twb);
    return flat;
}

Eigen::Vector3f
FromFlatPointVertex(const graphite_backend::FlatLocalBAPointVertex &point) {
    return Eigen::Vector3f(static_cast<float>(point.position[0]),
                           static_cast<float>(point.position[1]),
                           static_cast<float>(point.position[2]));
}

PluckerLine
FromFlatLineVertex(const graphite_backend::FlatLocalBALineVertex &line) {
    PluckerLine line_w;
    for (int i = 0; i < 6; ++i) {
        line_w[i] = static_cast<float>(line.line_w[static_cast<size_t>(i)]);
    }
    return line_w;
}

struct GraphiteKeyFrameData {
    graphite_backend::FlatPoseVertex pose;
    std::vector<graphite_backend::FlatPinholeCamera> cameras;
    std::vector<std::array<double, 3>> rcb;
    std::vector<std::array<double, 3>> tcb;
};

struct GraphiteLocalBAData {
    std::vector<KeyFramePtr> poses;
    std::vector<uint8_t> fixed_poses;
    std::vector<uint8_t> update_poses;
    std::vector<GraphiteKeyFrameData> keyframes;
    std::unordered_map<KeyFramePtr, size_t> pose_index;

    std::vector<MapPointPtr> points;
    std::vector<graphite_backend::FlatLocalBAPointVertex> initial_points;
    std::unordered_map<MapPointPtr, size_t> point_index;

    std::vector<MapLinePtr> lines;
    std::vector<graphite_backend::FlatLocalBALineVertex> initial_lines;
    std::unordered_map<MapLinePtr, size_t> line_index;

    std::vector<graphite_backend::FlatLocalBAPointObservation>
        point_observations;
    std::vector<std::tuple<KeyFramePtr, MapPointPtr, int>> point_refs;
    std::vector<graphite_backend::FlatLocalBALineObservation> line_observations;
    std::vector<std::tuple<KeyFramePtr, MapLinePtr, int>> line_refs;
};

bool FillGraphiteKeyFrameData(KeyFramePtr pKF, GraphiteKeyFrameData &data) {
    if (!pKF || pKF->isBad())
        return false;

    const BasePose pose(pKF);
    data.pose = ToFlatPoseVertex(pose);
    data.cameras.resize(static_cast<size_t>(pKF->mnCam));
    data.rcb.resize(static_cast<size_t>(pKF->mnCam));
    data.tcb.resize(static_cast<size_t>(pKF->mnCam));

    for (size_t cam_id = 0; cam_id < static_cast<size_t>(pKF->mnCam);
         ++cam_id) {
        if (cam_id >= pose.vpCameras.size() || !pose.vpCameras[cam_id])
            return false;
        const auto *pinhole =
            dynamic_cast<const Pinhole *>(pose.vpCameras[cam_id]);
        if (!pinhole)
            return false;

        graphite_backend::FlatPinholeCamera camera;
        camera.fx = pinhole->getParameter(0);
        camera.fy = pinhole->getParameter(1);
        camera.cx = pinhole->getParameter(2);
        camera.cy = pinhole->getParameter(3);
        camera.toPL = FlattenMatrix(pinhole->toPL_());
        data.cameras[cam_id] = camera;
        data.rcb[cam_id] = FlattenRotationVector(pose.Rcb[cam_id]);
        data.tcb[cam_id] = FlattenVector3(pose.tcb[cam_id]);
    }
    return true;
}

void ReserveGraphiteLocalBAData(GraphiteLocalBAData &data, size_t num_poses,
                                size_t num_points, size_t num_lines) {
    data.poses.reserve(num_poses);
    data.fixed_poses.reserve(num_poses);
    data.update_poses.reserve(num_poses);
    data.keyframes.reserve(num_poses);
    data.pose_index.reserve(num_poses);

    data.points.reserve(num_points);
    data.initial_points.reserve(num_points);
    data.point_index.reserve(num_points);
    data.point_observations.reserve(num_points * 12);
    data.point_refs.reserve(num_points * 12);

    data.lines.reserve(num_lines);
    data.initial_lines.reserve(num_lines);
    data.line_index.reserve(num_lines);
    data.line_observations.reserve(num_lines * 8);
    data.line_refs.reserve(num_lines * 8);
}

bool AddGraphitePose(GraphiteLocalBAData &data, KeyFramePtr pKF, bool fixed,
                     bool update_pose) {
    auto it = data.pose_index.find(pKF);
    if (it != data.pose_index.end()) {
        const size_t idx = it->second;
        data.fixed_poses[idx] = fixed ? 1 : data.fixed_poses[idx];
        data.update_poses[idx] = update_pose ? 1 : data.update_poses[idx];
        return true;
    }

    GraphiteKeyFrameData keyframe_data;
    if (!FillGraphiteKeyFrameData(pKF, keyframe_data)) {
        return false;
    }

    const size_t index = data.poses.size();
    data.pose_index[pKF] = index;
    data.poses.push_back(pKF);
    data.fixed_poses.push_back(fixed ? 1 : 0);
    data.update_poses.push_back(update_pose ? 1 : 0);
    data.keyframes.push_back(std::move(keyframe_data));
    return true;
}

void AddGraphitePoint(GraphiteLocalBAData &data, MapPointPtr pMP) {
    if (!pMP || data.point_index.count(pMP))
        return;
    const size_t index = data.points.size();
    data.point_index[pMP] = index;
    data.points.push_back(pMP);
    graphite_backend::FlatLocalBAPointVertex point;
    const Eigen::Vector3d position = pMP->GetWorldPos().cast<double>();
    point.position = FlattenVector3(position);
    data.initial_points.push_back(point);
}

void AddGraphiteLine(GraphiteLocalBAData &data, MapLinePtr pML) {
    if (!pML || data.line_index.count(pML))
        return;
    const size_t index = data.lines.size();
    data.line_index[pML] = index;
    data.lines.push_back(pML);
    graphite_backend::FlatLocalBALineVertex line;
    line.line_w = FlattenVector6(pML->GetWorldPos().cast<double>());
    data.initial_lines.push_back(line);
}

bool AddGraphitePointObservation(GraphiteLocalBAData &data, MapPointPtr pMP,
                                 KeyFramePtr pKFi, size_t cam_id, int idx) {
    auto pose_it = data.pose_index.find(pKFi);
    auto point_it = data.point_index.find(pMP);
    if (pose_it == data.pose_index.end() || point_it == data.point_index.end())
        return true;
    const size_t pose_idx = pose_it->second;
    if (cam_id >= data.keyframes[pose_idx].cameras.size() ||
        cam_id >= pKFi->mvKeysUn_cams.size() || idx < 0 ||
        static_cast<size_t>(idx) >= pKFi->mvKeysUn_cams[cam_id].size()) {
        return false;
    }

    const cv::Point2f &ptUn = pKFi->mvKeysUn_cams[cam_id][idx];
    graphite_backend::FlatLocalBAPointObservation observation;
    observation.pose = pose_idx;
    observation.point = point_it->second;
    observation.camera = data.keyframes[pose_idx].cameras[cam_id];
    observation.rcb = data.keyframes[pose_idx].rcb[cam_id];
    observation.tcb = data.keyframes[pose_idx].tcb[cam_id];
    observation.measurement = {ptUn.x, ptUn.y};
    observation.information = 1.0;
    data.point_observations.push_back(observation);
    data.point_refs.emplace_back(pKFi, pMP, static_cast<int>(cam_id));
    return true;
}

bool AddGraphiteLineObservation(GraphiteLocalBAData &data, MapLinePtr pML,
                                KeyFramePtr pKFi, size_t cam_id, int idx) {
    auto pose_it = data.pose_index.find(pKFi);
    auto line_it = data.line_index.find(pML);
    if (pose_it == data.pose_index.end() || line_it == data.line_index.end())
        return true;
    const size_t pose_idx = pose_it->second;
    if (cam_id >= data.keyframes[pose_idx].cameras.size() ||
        cam_id >= pKFi->mvLineKeysUn_cams.size() || idx < 0 ||
        static_cast<size_t>(idx) >= pKFi->mvLineKeysUn_cams[cam_id].size()) {
        return false;
    }

    const Eigen::Vector4f &lineUn = pKFi->mvLineKeysUn_cams[cam_id][idx];
    graphite_backend::FlatLocalBALineObservation observation;
    observation.pose = pose_idx;
    observation.line = line_it->second;
    observation.camera = data.keyframes[pose_idx].cameras[cam_id];
    observation.rcb = data.keyframes[pose_idx].rcb[cam_id];
    observation.tcb = data.keyframes[pose_idx].tcb[cam_id];
    observation.measurement = {lineUn[0], lineUn[1], lineUn[2], lineUn[3]};
    SetFlatLineEndpointCovariance(observation.endpoint_covariance_upper,
                                  observation.has_endpoint_covariance, *pKFi,
                                  cam_id, idx);
    observation.information = kLineInformationFallback;
    data.line_observations.push_back(observation);
    data.line_refs.emplace_back(pKFi, pML, static_cast<int>(cam_id));
    return true;
}

bool RunGraphiteLocalBA(GraphiteLocalBAData &data, Map *pMap,
                        graphite_backend::FlatLocalBASchedule schedule,
                        bool *pbStopFlag, bool increase_change_index) {
    if (!pMap || data.poses.empty() ||
        (data.point_observations.empty() && data.line_observations.empty())) {
        return false;
    }

    std::vector<graphite_backend::FlatPoseVertex> optimized_poses(
        data.poses.size());
    std::vector<graphite_backend::FlatLocalBAPointVertex> optimized_points(
        data.points.size());
    std::vector<graphite_backend::FlatLocalBALineVertex> optimized_lines(
        data.lines.size());
    std::vector<uint8_t> point_outliers(data.point_observations.size(), 0);
    std::vector<uint8_t> line_outliers(data.line_observations.size(), 0);

    std::vector<graphite_backend::FlatPoseVertex> initial_poses;
    initial_poses.reserve(data.keyframes.size());
    for (const GraphiteKeyFrameData &keyframe : data.keyframes) {
        initial_poses.push_back(keyframe.pose);
    }

    graphite_backend::FlatLocalBAProblem problem;
    problem.initial_poses = initial_poses.data();
    problem.fixed_poses = data.fixed_poses.data();
    problem.num_poses = initial_poses.size();
    problem.initial_points = data.initial_points.data();
    problem.num_points = data.initial_points.size();
    problem.initial_lines = data.initial_lines.data();
    problem.num_lines = data.initial_lines.size();
    problem.point_observations = data.point_observations.data();
    problem.num_point_observations = data.point_observations.size();
    problem.line_observations = data.line_observations.data();
    problem.num_line_observations = data.line_observations.size();
    problem.schedule = schedule;
    problem.single_stage_iterations = 10;
    problem.first_stage_iterations = 5;
    problem.second_stage_iterations = 10;
    problem.initial_damping = 1e-4;
    problem.chi2_threshold = 5.991;
    problem.huber_delta = std::sqrt(5.991);
    problem.stop_flag = pbStopFlag;

    graphite_backend::FlatLocalBAResult result;
    result.optimized_poses = optimized_poses.data();
    result.num_optimized_poses = optimized_poses.size();
    result.optimized_points = optimized_points.data();
    result.num_optimized_points = optimized_points.size();
    result.optimized_lines = optimized_lines.data();
    result.num_optimized_lines = optimized_lines.size();
    result.point_outliers = point_outliers.data();
    result.num_point_outliers = point_outliers.size();
    result.line_outliers = line_outliers.data();
    result.num_line_outliers = line_outliers.size();

    const bool success = graphite_backend::OptimizeLocalBAFlat(problem, result);
    if (!success) {
        return false;
    }

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    for (size_t i = 0; i < point_outliers.size(); ++i) {
        if (!point_outliers[i])
            continue;
        KeyFramePtr pKFi;
        MapPointPtr pMPi;
        int cam_id;
        tie(pKFi, pMPi, cam_id) = data.point_refs[i];
        if (pKFi && pMPi && !pMPi->isBad()) {
            pKFi->EraseMapPointMatch(pMPi, cam_id);
            pMPi->EraseObservation(pKFi, cam_id);
        }
    }
    for (size_t i = 0; i < line_outliers.size(); ++i) {
        if (!line_outliers[i])
            continue;
        KeyFramePtr pKFi;
        MapLinePtr pMLi;
        int cam_id;
        tie(pKFi, pMLi, cam_id) = data.line_refs[i];
        if (pKFi && pMLi && !pMLi->isBad()) {
            pKFi->EraseMapLineMatch(pMLi, cam_id);
            pMLi->EraseObservation(pKFi, cam_id);
        }
    }

    for (size_t i = 0; i < data.poses.size(); ++i) {
        KeyFramePtr pKFi = data.poses[i];
        if (!data.update_poses[i] || !pKFi || pKFi->isBad())
            continue;
        const g2o::SE3Quat Tbw = TbwFromFlatPoseVertex(optimized_poses[i]);
        pKFi->SetPose(ToSophusSE3(Tbw).cast<float>());
    }

    for (size_t i = 0; i < data.points.size(); ++i) {
        MapPointPtr pMP = data.points[i];
        if (pMP && !pMP->isBad()) {
            pMP->SetWorldPos(FromFlatPointVertex(optimized_points[i]));
        }
    }
    for (size_t i = 0; i < data.lines.size(); ++i) {
        MapLinePtr pML = data.lines[i];
        if (pML && !pML->isBad()) {
            pML->SetWorldPos(FromFlatLineVertex(optimized_lines[i]));
        }
    }

    if (increase_change_index) {
        pMap->IncreaseChangeIndex();
    }
    return true;
}

bool AddGraphiteFixedPoseIfNeeded(GraphiteLocalBAData &data, KeyFramePtr pKFi,
                                  Map *pCurrentMap,
                                  unordered_set<KeyFramePtr> &fixed_keyframes,
                                  int &num_fixedKF) {
    if (!pKFi || pKFi->isBad() || pKFi->GetMap() != pCurrentMap) {
        return true;
    }
    if (!fixed_keyframes.insert(pKFi).second) {
        return true;
    }
    if (!AddGraphitePose(data, pKFi, true, false)) {
        return false;
    }
    ++num_fixedKF;
    return true;
}

bool TryLocalBundleAdjustmentWithGraphiteOnePass(
    Map *pMap, Map *pCurrentMap, KeyFramePtr pKF,
    const list<KeyFramePtr> &lLocalKeyFrames,
    const list<MapPointPtr> &lLocalMapPoints,
    const list<MapLinePtr> &lLocalMapLines, bool *pbStopFlag, int &num_fixedKF,
    int &num_OptKF, int &num_MPs, int &num_edges) {
    if (!pMap || !pCurrentMap || !pKF) {
        return false;
    }

    GraphiteLocalBAData data;
    ReserveGraphiteLocalBAData(data, lLocalKeyFrames.size() + 16,
                               lLocalMapPoints.size(), lLocalMapLines.size());

    int graphite_num_fixedKF = 0;
    for (KeyFramePtr pKFi : lLocalKeyFrames) {
        if (!pKFi || pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
            continue;
        const bool fixed = pKFi->mnId == pMap->GetInitKFid();
        graphite_num_fixedKF += fixed ? 1 : 0;
        if (!AddGraphitePose(data, pKFi, fixed, true))
            return false;
    }

    for (MapPointPtr pMP : lLocalMapPoints) {
        if (pMP && !pMP->isBad() && pMP->GetMap() == pCurrentMap) {
            AddGraphitePoint(data, pMP);
        }
    }
    for (MapLinePtr pML : lLocalMapLines) {
        if (pML && !pML->isBad() && pML->GetMap() == pCurrentMap) {
            AddGraphiteLine(data, pML);
        }
    }

    unordered_set<KeyFramePtr> fixed_keyframes;
    fixed_keyframes.reserve(data.points.size() / 4 + data.lines.size() + 16);
    vector<pair<KeyFramePtr, vector<int>>> observations;
    observations.reserve(16);

    for (MapPointPtr pMP : data.points) {
        observations.clear();
        pMP->GetObservations(observations);
        for (const auto &observation : observations) {
            KeyFramePtr pKFi = observation.first;
            if (!pKFi || pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
                continue;
            if (pKFi->mnBALocalForKF != pKF->mnId &&
                !AddGraphiteFixedPoseIfNeeded(data, pKFi, pCurrentMap,
                                              fixed_keyframes,
                                              graphite_num_fixedKF)) {
                return false;
            }
            for (size_t cam_id = 0; cam_id < observation.second.size();
                 ++cam_id) {
                const int idx = observation.second[cam_id];
                if (idx != -1 && !AddGraphitePointObservation(data, pMP, pKFi,
                                                              cam_id, idx)) {
                    return false;
                }
            }
        }
    }

    for (MapLinePtr pML : data.lines) {
        observations.clear();
        pML->GetObservations(observations);
        for (const auto &observation : observations) {
            KeyFramePtr pKFi = observation.first;
            if (!pKFi || pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
                continue;
            if (pKFi->mnBALocalForKF != pKF->mnId &&
                !AddGraphiteFixedPoseIfNeeded(data, pKFi, pCurrentMap,
                                              fixed_keyframes,
                                              graphite_num_fixedKF)) {
                return false;
            }
            for (size_t cam_id = 0; cam_id < observation.second.size();
                 ++cam_id) {
                const int idx = observation.second[cam_id];
                if (idx != -1 &&
                    !AddGraphiteLineObservation(data, pML, pKFi, cam_id, idx)) {
                    return false;
                }
            }
        }
    }

    if (graphite_num_fixedKF == 0) {
        return false;
    }

    num_MPs = static_cast<int>(data.points.size() + data.lines.size());
    num_edges = static_cast<int>(data.point_observations.size() +
                                 data.line_observations.size());

    const bool success = RunGraphiteLocalBA(
        data, pMap, graphite_backend::FlatLocalBASchedule::kSingleStage,
        pbStopFlag, true);
    if (success) {
        num_fixedKF = graphite_num_fixedKF;
        num_OptKF = lLocalKeyFrames.size();
    }
    return success;
}

bool TryLocalBundleAdjustmentWithGraphite(
    Map *pMap, Map *pCurrentMap, const list<KeyFramePtr> &lLocalKeyFrames,
    const list<KeyFramePtr> &lFixedKeyFrames,
    const list<MapPointPtr> &lLocalMapPoints,
    const list<MapLinePtr> &lLocalMapLines,
    const unordered_map<MapPointPtr, map<KeyFramePtr, vector<int>>>
        &mPointObservations,
    const unordered_map<MapLinePtr, map<KeyFramePtr, vector<int>>>
        &mLineObservations,
    bool *pbStopFlag, int &num_MPs, int &num_edges) {
    GraphiteLocalBAData data;
    ReserveGraphiteLocalBAData(data,
                               lLocalKeyFrames.size() + lFixedKeyFrames.size(),
                               lLocalMapPoints.size(), lLocalMapLines.size());

    for (KeyFramePtr pKFi : lLocalKeyFrames) {
        if (!pKFi || pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
            continue;
        const bool fixed = pKFi->mnId == pMap->GetInitKFid();
        if (!AddGraphitePose(data, pKFi, fixed, true))
            return false;
    }
    for (KeyFramePtr pKFi : lFixedKeyFrames) {
        if (!pKFi || pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
            continue;
        if (!AddGraphitePose(data, pKFi, true, false))
            return false;
    }

    for (MapPointPtr pMP : lLocalMapPoints) {
        if (pMP && !pMP->isBad() && pMP->GetMap() == pCurrentMap) {
            AddGraphitePoint(data, pMP);
        }
    }
    for (MapLinePtr pML : lLocalMapLines) {
        if (pML && !pML->isBad() && pML->GetMap() == pCurrentMap) {
            AddGraphiteLine(data, pML);
        }
    }

    for (MapPointPtr pMP : data.points) {
        const auto obs_it = mPointObservations.find(pMP);
        if (obs_it == mPointObservations.end())
            continue;
        for (const auto &observation : obs_it->second) {
            KeyFramePtr pKFi = observation.first;
            if (!pKFi || pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
                continue;
            for (size_t cam_id = 0; cam_id < observation.second.size();
                 ++cam_id) {
                const int idx = observation.second[cam_id];
                if (idx != -1 && !AddGraphitePointObservation(data, pMP, pKFi,
                                                              cam_id, idx)) {
                    return false;
                }
            }
        }
    }

    for (MapLinePtr pML : data.lines) {
        const auto obs_it = mLineObservations.find(pML);
        if (obs_it == mLineObservations.end())
            continue;
        for (const auto &observation : obs_it->second) {
            KeyFramePtr pKFi = observation.first;
            if (!pKFi || pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
                continue;
            for (size_t cam_id = 0; cam_id < observation.second.size();
                 ++cam_id) {
                const int idx = observation.second[cam_id];
                if (idx != -1 &&
                    !AddGraphiteLineObservation(data, pML, pKFi, cam_id, idx)) {
                    return false;
                }
            }
        }
    }

    num_MPs = static_cast<int>(data.points.size() + data.lines.size());
    num_edges = static_cast<int>(data.point_observations.size() +
                                 data.line_observations.size());
    return RunGraphiteLocalBA(
        data, pMap, graphite_backend::FlatLocalBASchedule::kSingleStage,
        pbStopFlag, true);
}

bool TryMergeLocalBundleAdjustmentWithGraphite(
    KeyFramePtr pMainKF, const vector<KeyFramePtr> &vpAdjustKF,
    const vector<KeyFramePtr> &vpFixedKF, const vector<MapPointPtr> &vpMPs,
    const vector<MapLinePtr> &vpMLs, unsigned long maxKFid, bool *pbStopFlag) {
    if (!pMainKF)
        return false;
    Map *pCurrentMap = pMainKF->GetMap();
    GraphiteLocalBAData data;
    ReserveGraphiteLocalBAData(data, vpAdjustKF.size() + vpFixedKF.size(),
                               vpMPs.size(), vpMLs.size());

    for (KeyFramePtr pKFi : vpFixedKF) {
        if (pKFi && !pKFi->isBad() && pKFi->GetMap() == pCurrentMap &&
            !AddGraphitePose(data, pKFi, true, false)) {
            return false;
        }
    }
    for (KeyFramePtr pKFi : vpAdjustKF) {
        if (pKFi && !pKFi->isBad() && pKFi->GetMap() == pCurrentMap &&
            !AddGraphitePose(data, pKFi, false, true)) {
            return false;
        }
    }

    for (MapPointPtr pMP : vpMPs) {
        if (pMP && !pMP->isBad() && pMP->GetMap() == pCurrentMap) {
            AddGraphitePoint(data, pMP);
        }
    }
    for (MapLinePtr pML : vpMLs) {
        if (pML && !pML->isBad() && pML->GetMap() == pCurrentMap) {
            AddGraphiteLine(data, pML);
        }
    }

    vector<pair<KeyFramePtr, vector<int>>> observations;
    observations.reserve(16);
    for (MapPointPtr pMP : data.points) {
        observations.clear();
        pMP->GetObservations(observations);
        for (const auto &observation : observations) {
            KeyFramePtr pKFi = observation.first;
            if (!pKFi || pKFi->isBad() || pKFi->mnId > maxKFid ||
                pKFi->mnBALocalForMerge != pMainKF->mnId)
                continue;
            for (size_t cam_id = 0; cam_id < observation.second.size();
                 ++cam_id) {
                const int idx = observation.second[cam_id];
                if (idx != -1 && !AddGraphitePointObservation(data, pMP, pKFi,
                                                              cam_id, idx)) {
                    return false;
                }
            }
        }
    }

    for (MapLinePtr pML : data.lines) {
        observations.clear();
        pML->GetObservations(observations);
        for (const auto &observation : observations) {
            KeyFramePtr pKFi = observation.first;
            if (!pKFi || pKFi->isBad() || pKFi->mnId > maxKFid ||
                pKFi->mnBALocalForMerge != pMainKF->mnId)
                continue;
            for (size_t cam_id = 0; cam_id < observation.second.size();
                 ++cam_id) {
                const int idx = observation.second[cam_id];
                if (idx != -1 &&
                    !AddGraphiteLineObservation(data, pML, pKFi, cam_id, idx)) {
                    return false;
                }
            }
        }
    }

    return RunGraphiteLocalBA(data, pCurrentMap,
                              graphite_backend::FlatLocalBASchedule::kAutoMerge,
                              pbStopFlag, false);
}

bool TryOptimizeEssentialGraphWithGraphite(
    Map *pMap, KeyFramePtr pLoopKF, KeyFramePtr pCurKF,
    const LoopClosing::KeyFrameAndPose &NonCorrectedSE3,
    const LoopClosing::KeyFrameAndPose &CorrectedSE3,
    const std::map<KeyFramePtr, std::set<KeyFramePtr>> &LoopConnections) {
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    const vector<KeyFramePtr> vpKFs = pMap->GetAllKeyFrames();

    const unsigned int nMaxKFid = pMap->GetMaxKFid();
    vector<g2o::SE3Quat, Eigen::aligned_allocator<g2o::SE3Quat>> vScw(nMaxKFid +
                                                                      1);
    vector<g2o::SE3Quat, Eigen::aligned_allocator<g2o::SE3Quat>> vCorrectedSwc(
        nMaxKFid + 1);
    vector<size_t> vIdToLocal(nMaxKFid + 1, std::numeric_limits<size_t>::max());
    vector<KeyFramePtr> vLocalToKF;
    vector<graphite_backend::FlatPoseVertex> vInitialPoses;
    vector<uint8_t> vFixed;

    for (KeyFramePtr pKF : vpKFs) {
        if (!pKF || pKF->isBad())
            continue;

        const int nIDi = pKF->mnId;
        LoopClosing::KeyFrameAndPose::const_iterator it =
            CorrectedSE3.find(pKF);
        if (it != CorrectedSE3.end()) {
            vScw[nIDi] = it->second;
        } else {
            Sophus::SE3d Tbw = pKF->GetPose().cast<double>();
            vScw[nIDi] = ToG2OSE3Quat(Tbw);
        }

        vIdToLocal[nIDi] = vInitialPoses.size();
        vLocalToKF.push_back(pKF);
        vInitialPoses.push_back(ToFlatPoseVertexFromTbw(vScw[nIDi]));
        vFixed.push_back(pKF->mnId == pMap->GetInitKFid() ? 1 : 0);
    }

    const int minFeat = 200;
    vector<graphite_backend::FlatEssentialGraphEdgeInput> vEdges;
    set<pair<long unsigned int, long unsigned int>> sInsertedEdges;

    auto HasVertex = [&](const long unsigned int id) {
        return id <= nMaxKFid &&
               vIdToLocal[id] != std::numeric_limits<size_t>::max();
    };

    auto GetNonCorrectedScw = [&](KeyFramePtr pKF) {
        LoopClosing::KeyFrameAndPose::const_iterator it =
            NonCorrectedSE3.find(pKF);
        if (it != NonCorrectedSE3.end())
            return it->second;
        return vScw[pKF->mnId];
    };

    auto AddEdge = [&](const long unsigned int nIDi,
                       const long unsigned int nIDj, const g2o::SE3Quat &Tji) {
        if (!HasVertex(nIDi) || !HasVertex(nIDj))
            return;

        graphite_backend::FlatEssentialGraphEdgeInput edge;
        edge.vertex_i = vIdToLocal[nIDi];
        edge.vertex_j = vIdToLocal[nIDj];
        const Eigen::Matrix3d rotation = Tji.rotation().toRotationMatrix();
        edge.measurement_Tji.rotation = FlattenMatrix(rotation);
        edge.measurement_Tji.translation = FlattenVector3(Tji.translation());
        vEdges.push_back(edge);
    };

    for (map<KeyFramePtr, set<KeyFramePtr>>::const_iterator
             mit = LoopConnections.begin(),
             mend = LoopConnections.end();
         mit != mend; mit++) {
        KeyFramePtr pKF = mit->first;
        if (!pKF || pKF->isBad())
            continue;

        const long unsigned int nIDi = pKF->mnId;
        const set<KeyFramePtr> &spConnections = mit->second;
        const g2o::SE3Quat Tiw = vScw[nIDi];
        const g2o::SE3Quat Twi = Tiw.inverse();

        for (set<KeyFramePtr>::const_iterator sit = spConnections.begin(),
                                              send = spConnections.end();
             sit != send; sit++) {
            KeyFramePtr pKFn = *sit;
            if (!pKFn || pKFn->isBad())
                continue;

            const long unsigned int nIDj = pKFn->mnId;
            if ((nIDi != pCurKF->mnId || nIDj != pLoopKF->mnId) &&
                pKF->GetWeight(pKFn) < minFeat)
                continue;

            const g2o::SE3Quat Tjw = vScw[nIDj];
            AddEdge(nIDi, nIDj, Tjw * Twi);
            sInsertedEdges.insert(make_pair(min(nIDi, nIDj), max(nIDi, nIDj)));
        }
    }

    for (KeyFramePtr pKF : vpKFs) {
        if (!pKF || pKF->isBad())
            continue;

        const long unsigned int nIDi = pKF->mnId;
        const g2o::SE3Quat Twi = GetNonCorrectedScw(pKF).inverse();
        KeyFramePtr pParentKF = pKF->GetParent();

        if (pParentKF && !pParentKF->isBad()) {
            const long unsigned int nIDj = pParentKF->mnId;
            const g2o::SE3Quat Tjw = GetNonCorrectedScw(pParentKF);
            AddEdge(nIDi, nIDj, Tjw * Twi);
        }

        const set<KeyFramePtr> sLoopEdges = pKF->GetLoopEdges();
        for (set<KeyFramePtr>::const_iterator sit = sLoopEdges.begin(),
                                              send = sLoopEdges.end();
             sit != send; sit++) {
            KeyFramePtr pLKF = *sit;
            if (!pLKF || pLKF->isBad())
                continue;

            const long unsigned int nIDj = pLKF->mnId;
            if (nIDj < nIDi) {
                const g2o::SE3Quat Tlw = GetNonCorrectedScw(pLKF);
                AddEdge(nIDi, nIDj, Tlw * Twi);
            }
        }

        const vector<KeyFramePtr> vpConnectedKFs =
            pKF->GetCovisiblesByWeight(minFeat);
        for (vector<KeyFramePtr>::const_iterator vit = vpConnectedKFs.begin();
             vit != vpConnectedKFs.end(); vit++) {
            KeyFramePtr pKFn = *vit;
            if (!pKFn || pKFn->isBad() || pKFn == pParentKF ||
                pKF->hasChild(pKFn))
                continue;

            const long unsigned int nIDj = pKFn->mnId;
            if (nIDj >= nIDi)
                continue;

            if (sInsertedEdges.count(
                    make_pair(min(nIDi, nIDj), max(nIDi, nIDj))))
                continue;

            const g2o::SE3Quat Tnw = GetNonCorrectedScw(pKFn);
            AddEdge(nIDi, nIDj, Tnw * Twi);
        }
    }

    if (vInitialPoses.empty() || vEdges.empty())
        return false;

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used =
        chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "loop preparation time cost: " << time_used.count() * 1000 << " ms"
         << endl;

    t1 = chrono::steady_clock::now();

    vector<graphite_backend::FlatPoseVertex> vOptimized(vInitialPoses.size());
    graphite_backend::FlatEssentialGraphProblem problem;
    problem.initial_poses = vInitialPoses.data();
    problem.fixed = vFixed.data();
    problem.num_poses = vInitialPoses.size();
    problem.edges = vEdges.data();
    problem.num_edges = vEdges.size();
    problem.iterations = 20;
    problem.initial_damping = 1e-16;

    graphite_backend::FlatEssentialGraphResult result;
    result.optimized_poses = vOptimized.data();
    result.num_optimized_poses = vOptimized.size();
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "Graphite preparation time cost: " << time_used.count() * 1000
         << " ms" << endl;

    t1 = chrono::steady_clock::now();
    bool success =
        graphite_backend::OptimizeEssentialGraphFlat(problem, result);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "Graphite optimization time cost: " << time_used.count() * 1000
         << " ms" << endl;
    if (!success) {
        return false;
    }

    t1 = chrono::steady_clock::now();
    auto map_update_section_start = t1;
    auto detail_start = chrono::steady_clock::now();
    const vector<MapPointPtr> vpMPs = pMap->GetAllMapPoints();
    const vector<MapLinePtr> vpMLs = pMap->GetAllMapLines();
    auto detail_end = chrono::steady_clock::now();
    const double map_vector_copy_ms =
        chrono::duration<double, std::milli>(detail_end - detail_start).count();

    cout << "vertices and edges added: " << vInitialPoses.size() << " , "
         << vEdges.size() << endl;

    detail_start = chrono::steady_clock::now();
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);
    detail_end = chrono::steady_clock::now();
    const double map_lock_wait_ms =
        chrono::duration<double, std::milli>(detail_end - detail_start).count();
    cout << "Loop closed edge optimization done." << endl;

    detail_start = chrono::steady_clock::now();
    size_t num_updated_kfs = 0;
    for (size_t local_id = 0; local_id < vLocalToKF.size(); ++local_id) {
        KeyFramePtr pKF = vLocalToKF[local_id];
        if (!pKF || pKF->isBad())
            continue;

        const long unsigned int nIDi = pKF->mnId;
        const g2o::SE3Quat correctedTiw =
            TbwFromFlatPoseVertex(vOptimized[local_id]);
        vCorrectedSwc[nIDi] = correctedTiw.inverse();
        pKF->SetPose(ToSophusSE3(correctedTiw).cast<float>());
        num_updated_kfs++;
    }
    detail_end = chrono::steady_clock::now();
    const double map_kf_update_ms =
        chrono::duration<double, std::milli>(detail_end - detail_start).count();

    detail_start = chrono::steady_clock::now();
    const double map_point_normal_ms = 0.0;
    size_t num_updated_mps = 0;
    size_t num_skipped_mps = 0;
    for (MapPointPtr pMP : vpMPs) {
        if (!pMP || pMP->isBad()) {
            num_skipped_mps++;
            continue;
        }

        int nIDr = -1;
        if (pMP->mnCorrectedByKF == pCurKF->mnId) {
            nIDr = pMP->mnCorrectedReference;
        } else {
            KeyFramePtr pRefKF = pMP->GetReferenceKeyFrame();
            nIDr = pRefKF->mnId;
        }

        const g2o::SE3Quat Trw = vScw[nIDr];
        const g2o::SE3Quat correctedTwr = vCorrectedSwc[nIDr];
        const Eigen::Vector3d p3Dw = pMP->GetWorldPos().cast<double>();
        const Eigen::Vector3d correctedP3Dw = correctedTwr.map(Trw.map(p3Dw));
        pMP->SetWorldPos(correctedP3Dw.cast<float>());
        num_updated_mps++;
    }
    detail_end = chrono::steady_clock::now();
    const double map_point_update_ms =
        chrono::duration<double, std::milli>(detail_end - detail_start).count();

    detail_start = chrono::steady_clock::now();
    size_t num_updated_mls = 0;
    size_t num_skipped_mls = 0;
    for (MapLinePtr pML : vpMLs) {
        if (!pML || pML->isBad()) {
            num_skipped_mls++;
            continue;
        }

        int nIDr = -1;
        if (pML->mnCorrectedByKF == pCurKF->mnId) {
            nIDr = pML->mnCorrectedReference;
        } else {
            KeyFramePtr pRefKF = pML->GetReferenceKeyFrame();
            nIDr = pRefKF->mnId;
        }

        const g2o::SE3Quat Trw = vScw[nIDr];
        const g2o::SE3Quat correctedTwr = vCorrectedSwc[nIDr];
        PluckerLine lineW = pML->GetWorldPos();
        PluckerLine lineR =
            TransformLineToBody(lineW,
                                Trw.rotation().toRotationMatrix().cast<float>(),
                                Trw.translation().cast<float>())
                .cast<float>();
        PluckerLine correctedLineW =
            TransformLineToBody(
                lineR, correctedTwr.rotation().toRotationMatrix().cast<float>(),
                correctedTwr.translation().cast<float>())
                .cast<float>();
        pML->SetWorldPos(correctedLineW);
        num_updated_mls++;
    }
    detail_end = chrono::steady_clock::now();
    const double map_line_update_ms =
        chrono::duration<double, std::milli>(detail_end - detail_start).count();

    detail_start = chrono::steady_clock::now();
    pMap->IncreaseChangeIndex();
    detail_end = chrono::steady_clock::now();
    const double map_change_index_ms =
        chrono::duration<double, std::milli>(detail_end - detail_start).count();
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    const double map_total_ms =
        chrono::duration<double, std::milli>(t2 - map_update_section_start)
            .count();
    cout << "Map update detail: vector_copy=" << map_vector_copy_ms
         << " ms, lock_wait=" << map_lock_wait_ms
         << " ms, keyframes=" << map_kf_update_ms << " ms (" << num_updated_kfs
         << "), map_points=" << map_point_update_ms << " ms ("
         << num_updated_mps << " updated, " << num_skipped_mps
         << " skipped, normal=" << map_point_normal_ms
         << " ms), map_lines=" << map_line_update_ms << " ms ("
         << num_updated_mls << " updated, " << num_skipped_mls
         << " skipped), change_index=" << map_change_index_ms
         << " ms, total=" << map_total_ms << " ms" << endl;
    cout << "Map update time cost: " << time_used.count() * 1000 << " ms"
         << endl;
    return true;
}
#endif

} // namespace

int Optimizer::PoseOptimization(Frame *pFrame) {
    RuntimePhaseScope runtime_phase(RuntimePhase::Pose);
    RuntimeCallScope runtime_call("pose");
    g2o::SparseOptimizer optimizer;

    auto linearSolver = std::make_unique<
        g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();

    auto solver_ptr =
        std::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver));

    g2o::OptimizationAlgorithmGaussNewton *solver =
        new g2o::OptimizationAlgorithmGaussNewton(std::move(solver_ptr));
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences = 0;

    VertexPose *vPose = new VertexPose(pFrame);
    vPose->setId(0);
    vPose->setFixed(false);
    optimizer.addVertex(vPose);

    const vector<int> Np = pFrame->Np;
    const vector<int> Nl = pFrame->Nl;

    const int total_points = pFrame->mnTotalPoints;
    const int total_lines = pFrame->mnTotalLines;

    vector<EdgeProjectPointOnlyPose *> vpEdgesPoint;
    vector<EdgeProjectLineOnlyPose *> vpEdgesLine;

    vector<pair<size_t, size_t>> vnCamIndexEdgePoint;
    vector<pair<size_t, size_t>> vnCamIndexEdgeLine;

    int n_cam = pFrame->mnCam;

    vpEdgesPoint.reserve(static_cast<size_t>(total_points));
    vpEdgesLine.reserve(static_cast<size_t>(total_lines));
    vnCamIndexEdgePoint.reserve(static_cast<size_t>(total_points));
    vnCamIndexEdgeLine.reserve(static_cast<size_t>(total_lines));

    const float deltaMono = sqrt(5.991);
    const double sigma_point = 1.0;

    {
        unique_lock<mutex> lock(MapPoint::mGlobalMutex);
        unique_lock<mutex> lock_line(MapLine::mGlobalMutex);

        for (size_t cam_id = 0; cam_id < static_cast<size_t>(n_cam); ++cam_id) {
            for (int i = 0; i < Np[cam_id]; ++i) {
                MapPointPtr pMP = pFrame->mvpMapPoints_cams[cam_id][i];
                if (pMP) {
                    nInitialCorrespondences++;

                    cv::Point2f ptUn = pFrame->mvKeysUn_cams[cam_id][i];

                    Eigen::Matrix<double, 2, 1> obs;
                    obs << ptUn.x, ptUn.y;

                    EdgeProjectPointOnlyPose *e =
                        new EdgeProjectPointOnlyPose(cam_id);

                    e->setVertex(0,
                                 dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                     optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    e->setInformation(Eigen::Matrix2d::Identity() *
                                      (1.0 / (sigma_point * sigma_point)));

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    e->Xw = pMP->GetWorldPos().cast<double>();

                    optimizer.addEdge(e);

                    vpEdgesPoint.push_back(e);
                    vnCamIndexEdgePoint.emplace_back(cam_id, i);
                }
            }

            for (int i = 0; i < Nl[cam_id]; ++i) {
                MapLinePtr pML = pFrame->mvpMapLines_cams[cam_id][i];
                const bool has_selection_mask =
                    cam_id < pFrame->mvbUseLineConstraint_cams.size() &&
                    pFrame->mvbUseLineConstraint_cams[cam_id].size() ==
                        static_cast<size_t>(Nl[cam_id]);
                const bool selected =
                    !has_selection_mask ||
                    pFrame->mvbUseLineConstraint_cams[cam_id][i];
                if (pML && selected) {
                    nInitialCorrespondences++;

                    Eigen::Vector4f lineUn =
                        pFrame->mvLineKeysUn_cams[cam_id][i];

                    Eigen::Matrix<double, 4, 1> obs;
                    obs << lineUn(0), lineUn(1), lineUn(2), lineUn(3);

                    EdgeProjectLineOnlyPose *e =
                        new EdgeProjectLineOnlyPose(cam_id);

                    e->setVertex(0,
                                 dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                     optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    e->setInformation(Eigen::Matrix2d::Identity() *
                                      kLineInformationFallback);
                    SetLineEndpointCovariance(e, *pFrame, cam_id, i);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    e->line_w = pML->GetWorldPos().cast<double>();

                    optimizer.addEdge(e);

                    vpEdgesLine.push_back(e);
                    vnCamIndexEdgeLine.emplace_back(cam_id, i);
                }
            }
        }
    }

    if (runtime_frame) runtime_call.call.line_edges_added = vpEdgesLine.size();
    std::vector<bool> runtime_used(runtime_frame ? vpEdgesLine.size() : 0, false);
    const auto reject_pose = [&]() {
        for (const auto &ref : vnCamIndexEdgePoint) {
            pFrame->mvbOutlierPoint_cams[std::get<0>(ref)][std::get<1>(ref)] = true;
        }
        for (const auto &ref : vnCamIndexEdgeLine) {
            pFrame->mvbOutlierLine_cams[std::get<0>(ref)][std::get<1>(ref)] = true;
        }
        return 0;
    };
    if (nInitialCorrespondences < 3 || !IsFinitePose(vPose->estimate())) {
        return reject_pose();
    }

    // Non-finite residuals must not enter the first linear system.
    for (auto *edge : vpEdgesPoint) {
        edge->computeError();
        if (!std::isfinite(edge->chi2())) {
            edge->setLevel(1);
        }
    }
    for (auto *edge : vpEdgesLine) {
        edge->computeError();
        if (!std::isfinite(edge->chi2())) {
            edge->setLevel(1);
        }
    }

    const float chi2Mono = 5.991;
    const int its[4] = {10, 10, 10, 10};

    int nBad = 0;
    for (size_t it = 0; it < 4; ++it) {
        BasePose pose(pFrame);
        BasePose pose_before =
            static_cast<VertexPose *>(optimizer.vertex(0))->estimate();
        vPose->setEstimate(pose);

        optimizer.initializeOptimization(0);
        if (optimizer.activeEdges().size() < 10) {
            if (it == 0) {
                return reject_pose();
            }
            vPose->setEstimate(pose_before);
            break;
        }
        if (runtime_frame) {
            ++runtime_call.call.solve_rounds;
            // Count active line observations handed to an actual optimize call,
            // including a later failed solve; added-but-inactive edges do not count.
            for (size_t index = 0; index < vpEdgesLine.size(); ++index) {
                if (std::find(optimizer.activeEdges().begin(), optimizer.activeEdges().end(),
                              vpEdgesLine[index]) != optimizer.activeEdges().end()) {
                    ++runtime_call.call.active_line_round_sum;
                    if (!runtime_used[index]) {
                        runtime_used[index] = true;
                        ++runtime_call.call.line_residuals_used;
                    }
                }
            }
        }
        const int runtime_iterations = optimizer.optimize(its[it]);
        if (runtime_frame) {
            runtime_call.call.solve_iterations += std::max(0, runtime_iterations);
            runtime_call.call.failed_solve_rounds += runtime_iterations <= 0;
        }
        if (!IsFinitePose(vPose->estimate())) {
            return reject_pose();
        }

        nBad = 0;

        for (size_t i = 0; i < vpEdgesPoint.size(); ++i) {
            EdgeProjectPointOnlyPose *e = vpEdgesPoint[i];

            size_t cam_id, idx;
            tie(cam_id, idx) = vnCamIndexEdgePoint[i];

            if (IsReprojectionOutlier(e, chi2Mono)) {
                pFrame->mvbOutlierPoint_cams[cam_id][idx] = true;
                e->setLevel(1);
                nBad++;
            } else {
                pFrame->mvbOutlierPoint_cams[cam_id][idx] = false;
                e->setLevel(0);
            }

            if (it == 2) {
                e->setRobustKernel(0);
            }
        }

        for (size_t i = 0; i < vpEdgesLine.size(); ++i) {
            EdgeProjectLineOnlyPose *e = vpEdgesLine[i];

            size_t cam_id, idx;
            tie(cam_id, idx) = vnCamIndexEdgeLine[i];

            if (IsReprojectionOutlier(e, chi2Mono)) {
                pFrame->mvbOutlierLine_cams[cam_id][idx] = true;
                e->setLevel(1);
                nBad++;
            } else {
                pFrame->mvbOutlierLine_cams[cam_id][idx] = false;
                e->setLevel(0);
            }

            if (it == 2) {
                e->setRobustKernel(0);
            }
        }
    }

    VertexPose *vPoseAfter = static_cast<VertexPose *>(optimizer.vertex(0));
    BasePose pose_after = vPoseAfter->estimate();

    const Sophus::SE3f Tbw_after = pose_after.ToTbw().cast<float>();
    pFrame->SetPose(Tbw_after);

    return nInitialCorrespondences - nBad;
}

PoseConstraintSummary Optimizer::EvaluatePointConstraints(
    Frame *pFrame, const AdaptiveLineConfig &config) {
    RuntimePhaseScope runtime_phase(RuntimePhase::Selection);
    RuntimeCallScope runtime_call("evaluation");
    pFrame->mvbUseLineConstraint_cams.resize(pFrame->mnCam);
    for (int cam = 0; cam < pFrame->mnCam; ++cam) {
        pFrame->mvbUseLineConstraint_cams[cam].assign(
            static_cast<size_t>(std::max(pFrame->Nl[cam], 0)), false);
    }

    // Evaluate at the incoming pose; optimization happens after branch selection.
    PoseConstraintSummary summary;
    BasePose pose(pFrame);
    const int coverage_cells_per_camera =
        config.coverage_rows * config.coverage_cols;
    std::vector<std::vector<bool>> occupied(
        pFrame->mnCam,
        std::vector<bool>(static_cast<size_t>(coverage_cells_per_camera),
                          false));

    for (int cam = 0; cam < pFrame->mnCam; ++cam) {
        for (int i = 0; i < pFrame->Np[cam]; ++i) {
            MapPointPtr point = pFrame->mvpMapPoints_cams[cam][i];
            if (!point || pFrame->mvbOutlierPoint_cams[cam][i]) {
                continue;
            }
            const Eigen::Vector3d world = point->GetWorldPos().cast<double>();
            if (!world.allFinite() || !pose.isPointDepthPositive(world, cam)) {
                continue;
            }
            const Eigen::Matrix<double, 2, 6> jacobian =
                PointOnlyPoseJacobian(pose, world, cam);
            if (!jacobian.allFinite()) {
                continue;
            }
            const cv::Point2f &key = pFrame->mvKeysUn_cams[cam][i];
            if (!std::isfinite(key.x) || !std::isfinite(key.y)) {
                continue;
            }
            summary.point_hessian.noalias() += jacobian.transpose() * jacobian;
            ++summary.point_matches;
            const float width =
                std::max(1.0f, Frame::mvnMaxX[cam] - Frame::mvnMinX[cam]);
            const float height =
                std::max(1.0f, Frame::mvnMaxY[cam] - Frame::mvnMinY[cam]);
            const int col = std::clamp(
                static_cast<int>((key.x - Frame::mvnMinX[cam]) / width *
                                 config.coverage_cols),
                0, config.coverage_cols - 1);
            const int row = std::clamp(
                static_cast<int>((key.y - Frame::mvnMinY[cam]) / height *
                                 config.coverage_rows),
                0, config.coverage_rows - 1);
            occupied[cam][static_cast<size_t>(row * config.coverage_cols + col)] =
                true;
        }
    }

    int occupied_count = 0;
    for (const auto &camera_cells : occupied) {
        occupied_count += static_cast<int>(
            std::count(camera_cells.begin(), camera_cells.end(), true));
    }
    const int total_cells = pFrame->mnCam * coverage_cells_per_camera;
    summary.coverage_ratio =
        total_cells > 0 ? static_cast<double>(occupied_count) / total_cells : 0.0;

    const NormalizedPoseHessian normalized =
        NormalizePoseHessian(summary.point_hessian);
    summary.column_scale = normalized.scale;
    summary.normalized_hessian = normalized.hessian;
    summary.min_eigenvalue =
        normalized.valid ? normalized.eigenvalues(0) : 0.0;
    summary.numerically_valid = normalized.valid;
    summary.point_constraints_strong =
        summary.numerically_valid &&
        summary.point_matches >= config.min_point_matches &&
        summary.coverage_ratio >= config.min_coverage_ratio &&
        summary.min_eigenvalue >= config.min_normalized_eigenvalue;
    runtime_call.call.points = summary.point_matches;
    runtime_call.call.point_strong = summary.point_constraints_strong;
    return summary;
}

PoseConstraintSummary Optimizer::SelectLineConstraints(
    Frame *pFrame, const AdaptiveLineConfig &config,
    const PoseConstraintSummary &point_summary) {
    RuntimePhaseScope runtime_phase(RuntimePhase::Selection);
    RuntimeCallScope runtime_call("selection");
    struct Candidate {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        int cam = -1;
        int index = -1;
        Matrix6d hessian = Matrix6d::Zero();
        Matrix6d normalized_hessian = Matrix6d::Zero();
        bool selected = false;
    };

    PoseConstraintSummary result = point_summary;
    pFrame->mvbUseLineConstraint_cams.resize(pFrame->mnCam);
    for (int cam = 0; cam < pFrame->mnCam; ++cam) {
        pFrame->mvbUseLineConstraint_cams[cam].assign(
            static_cast<size_t>(std::max(pFrame->Nl[cam], 0)), false);
    }

    BasePose pose(pFrame);
    std::vector<Candidate, Eigen::aligned_allocator<Candidate>> candidates;
    Matrix6d all_hessian = point_summary.point_hessian;
    for (int cam = 0; cam < pFrame->mnCam; ++cam) {
        for (int i = 0; i < pFrame->Nl[cam]; ++i) {
            MapLinePtr line = pFrame->mvpMapLines_cams[cam][i];
            if (!line) {
                continue;
            }
            // Candidate eligibility does not validate the association.
            // Unselected lines are checked at the final optimized pose.
            pFrame->mvbOutlierLine_cams[cam][i] = true;
            const Eigen::Matrix<double, 6, 1> world =
                line->GetWorldPos().cast<double>();
            const Eigen::Vector4d obs =
                pFrame->mvLineKeysUn_cams[cam][i].cast<double>();
            if (!world.allFinite() || !obs.allFinite() ||
                !pose.isLineDepthPositive(world, cam, obs)) {
                continue;
            }
            const Eigen::Vector3d projected = pose.ProjectLine(world, cam);
            if (!projected.allFinite()) {
                continue;
            }
            PackedLineEndpointCovariance packed;
            const bool has_covariance =
                GetLineEndpointCovariance(*pFrame, cam, i, packed);
            const Eigen::Matrix2d information = LineResidualInformation(
                projected, packed, has_covariance);
            if (!information.allFinite()) {
                continue;
            }
            const Eigen::Matrix<double, 2, 6> jacobian =
                LineOnlyPoseJacobian(pose, world, obs, cam);
            if (!jacobian.allFinite()) {
                continue;
            }
            Candidate candidate;
            candidate.cam = cam;
            candidate.index = i;
            candidate.hessian = jacobian.transpose() * information * jacobian;
            if (!candidate.hessian.allFinite()) {
                continue;
            }
            all_hessian += candidate.hessian;
            candidates.push_back(candidate);
        }
    }

    runtime_call.call.candidates = candidates.size();
    bool scale_valid = false;
    const Eigen::Matrix<double, 6, 1> fixed_scale =
        HessianColumnScale(all_hessian, scale_valid);
    if (!scale_valid) {
        result.numerically_valid = false;
        return result;
    }

    for (Candidate &candidate : candidates) {
        candidate.normalized_hessian =
            ScaledHessian(candidate.hessian, fixed_scale);
    }
    Matrix6d current = point_summary.point_hessian;
    for (int selection = 0;
         selection < config.max_selected_lines && !candidates.empty();
         ++selection) {
        const Matrix6d normalized = ScaledHessian(current, fixed_scale);
        Eigen::SelfAdjointEigenSolver<Matrix6d> eigen_solver(normalized);
        if (eigen_solver.info() != Eigen::Success ||
            !eigen_solver.eigenvalues().allFinite() ||
            !eigen_solver.eigenvectors().allFinite()) {
            result.numerically_valid = false;
            break;
        }
        const double current_min_eigenvalue = eigen_solver.eigenvalues()(0);
        if (current_min_eigenvalue >= config.min_normalized_eigenvalue &&
            point_summary.point_matches + result.selected_lines >=
                config.target_observations) {
            break;
        }
        const Eigen::Matrix<double, 6, 1> weakest =
            eigen_solver.eigenvectors().col(0);
        int best = -1;
        double best_gain = -std::numeric_limits<double>::infinity();
        for (size_t candidate_index = 0; candidate_index < candidates.size();
             ++candidate_index) {
            const Candidate &candidate = candidates[candidate_index];
            if (candidate.selected) {
                continue;
            }
            const double gain =
                weakest.dot(candidate.normalized_hessian * weakest);
            if (gain > best_gain) {
                best_gain = gain;
                best = static_cast<int>(candidate_index);
            }
        }
        if (best < 0 || !std::isfinite(best_gain) ||
            best_gain < config.min_line_gain) {
            break;
        }
        Candidate &chosen = candidates[static_cast<size_t>(best)];
        chosen.selected = true;
        pFrame->mvbUseLineConstraint_cams[chosen.cam][chosen.index] = true;
        current += chosen.hessian;
        ++result.selected_lines;
        ++runtime_call.call.selected;
    }

    result.column_scale = fixed_scale;
    result.normalized_hessian = ScaledHessian(current, fixed_scale);
    bool eigen_valid = false;
    result.min_eigenvalue =
        MinimumEigenvalue(result.normalized_hessian, eigen_valid);
    result.numerically_valid = eigen_valid;
    return result;
}

int Optimizer::PoseOptimizationSelectedLines(
    Frame *pFrame, const AdaptiveLineConfig &config,
    PoseConstraintSummary *summary) {
    const int inliers = PoseOptimization(pFrame);
    // Reuse optimizer decisions for selected lines; only check unused ones.
    ScreenLineAssociationsAtFixedPose(pFrame, config.line_precheck_chi2, true);
    if (summary != nullptr) {
        summary->point_inliers = 0;
        summary->selected_line_inliers = 0;
        for (int cam = 0; cam < pFrame->mnCam; ++cam) {
            for (int i = 0; i < pFrame->Np[cam]; ++i) {
                if (pFrame->mvpMapPoints_cams[cam][i] &&
                    !pFrame->mvbOutlierPoint_cams[cam][i]) {
                    ++summary->point_inliers;
                }
            }
            const bool has_mask =
                cam < static_cast<int>(pFrame->mvbUseLineConstraint_cams.size());
            for (int i = 0; i < pFrame->Nl[cam]; ++i) {
                if (has_mask &&
                    i < static_cast<int>(
                            pFrame->mvbUseLineConstraint_cams[cam].size()) &&
                    pFrame->mvbUseLineConstraint_cams[cam][i] &&
                    pFrame->mvpMapLines_cams[cam][i] &&
                    !pFrame->mvbOutlierLine_cams[cam][i]) {
                    ++summary->selected_line_inliers;
                }
            }
        }
    }
    return inliers;
}

int Optimizer::ScreenLineAssociationsAtFixedPose(Frame *pFrame,
                                                 double chi2_threshold,
                                                 bool only_unselected) {
    BasePose pose(pFrame);
    int accepted = 0;
    for (int cam = 0; cam < pFrame->mnCam; ++cam) {
        for (int i = 0; i < pFrame->Nl[cam]; ++i) {
            MapLinePtr line = pFrame->mvpMapLines_cams[cam][i];
            if (!line) {
                continue;
            }
            const bool selected =
                cam < static_cast<int>(pFrame->mvbUseLineConstraint_cams.size()) &&
                i < static_cast<int>(pFrame->mvbUseLineConstraint_cams[cam].size()) &&
                pFrame->mvbUseLineConstraint_cams[cam][i];
            if (only_unselected && selected) {
                continue;
            }
            pFrame->mvbOutlierLine_cams[cam][i] = true;
            const Eigen::Matrix<double, 6, 1> world =
                line->GetWorldPos().cast<double>();
            const Eigen::Vector4d obs =
                pFrame->mvLineKeysUn_cams[cam][i].cast<double>();
            if (!world.allFinite() || !obs.allFinite() ||
                !pose.isLineDepthPositive(world, cam, obs)) {
                pFrame->mvpMapLines_cams[cam][i] = nullptr;
                continue;
            }
            const Eigen::Vector3d projected = pose.ProjectLine(world, cam);
            Eigen::Vector2d error;
            error << Line2DError(projected, obs.head<2>()),
                Line2DError(projected, obs.tail<2>());
            PackedLineEndpointCovariance packed;
            const bool has_covariance =
                GetLineEndpointCovariance(*pFrame, cam, i, packed);
            const Eigen::Matrix2d information = LineResidualInformation(
                projected, packed, has_covariance);
            const double chi2 = error.dot(information * error);
            if (!projected.allFinite() || !error.allFinite() ||
                !information.allFinite() || !std::isfinite(chi2) ||
                chi2 > chi2_threshold) {
                pFrame->mvpMapLines_cams[cam][i] = nullptr;
                continue;
            }
            pFrame->mvbOutlierLine_cams[cam][i] = false;
            ++accepted;
        }
    }
    return accepted;
}

void Optimizer::LocalBundleAdjustment(KeyFramePtr pKF, bool *pbStopFlag,
                                      Map *pMap, int &num_fixedKF,
                                      int &num_OptKF, int &num_MPs,
                                      int &num_edges) {
    list<KeyFramePtr> lLocalKeyFrames;

    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;
    Map *pCurrentMap = pKF->GetMap();

    const vector<KeyFramePtr> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for (int i = 0; i < vNeighKFs.size(); ++i) {
        KeyFramePtr pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;
        if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap) {
            lLocalKeyFrames.push_back(pKFi);
        }
    }

    num_fixedKF = 0;
    list<MapPointPtr> lLocalMapPoints;
    list<MapLinePtr> lLocalMapLines;
    vector<MapPointPtr> vpLocalMapPointCandidates;
    vector<MapLinePtr> vpLocalMapLineCandidates;
    unordered_set<MapPointPtr> spLocalMapPoints;
    unordered_set<MapLinePtr> spLocalMapLines;

    for (list<KeyFramePtr>::iterator lit = lLocalKeyFrames.begin(),
                                     lend = lLocalKeyFrames.end();
         lit != lend; ++lit) {
        KeyFramePtr pKFi = *lit;
        if (pKFi->mnId == pMap->GetInitKFid()) {
            num_fixedKF = 1;
        }

        pKFi->AppendMapPoints(vpLocalMapPointCandidates, spLocalMapPoints);
        pKFi->AppendMapLines(vpLocalMapLineCandidates, spLocalMapLines);
    }

    for (MapPointPtr pMP : vpLocalMapPointCandidates) {
        if (pMP->GetMap() != pCurrentMap)
            continue;
        lLocalMapPoints.push_back(pMP);
        pMP->mnBALocalForKF = pKF->mnId;
    }
    for (MapLinePtr pML : vpLocalMapLineCandidates) {
        if (pML->GetMap() != pCurrentMap)
            continue;
        lLocalMapLines.push_back(pML);
        pML->mnBALocalForKF = pKF->mnId;
    }

#ifdef GRAPHITE
    if (!pbStopFlag || !(*pbStopFlag)) {
        int graphite_fixedKF = 0;
        int graphite_OptKF = 0;
        int graphite_MPs = 0;
        int graphite_edges = 0;
        if (TryLocalBundleAdjustmentWithGraphiteOnePass(
                pMap, pCurrentMap, pKF, lLocalKeyFrames, lLocalMapPoints,
                lLocalMapLines, pbStopFlag, graphite_fixedKF, graphite_OptKF,
                graphite_MPs, graphite_edges)) {
            num_fixedKF = graphite_fixedKF;
            num_OptKF = graphite_OptKF;
            num_MPs = graphite_MPs;
            num_edges = graphite_edges;
            return;
        }
    }
#endif

    list<KeyFramePtr> lFixedKeyFrames;
    unordered_map<MapPointPtr, map<KeyFramePtr, vector<int>>>
        mPointObservations;
    unordered_map<MapLinePtr, map<KeyFramePtr, vector<int>>> mLineObservations;
    mPointObservations.reserve(lLocalMapPoints.size());
    mLineObservations.reserve(lLocalMapLines.size());
    for (auto lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end();
         lit != lend; ++lit) {
        const auto insertion =
            mPointObservations.emplace(*lit, (*lit)->GetObservations());
        const map<KeyFramePtr, vector<int>> &observations =
            insertion.first->second;
        for (auto mit = observations.begin(), mend = observations.end();
             mit != mend; ++mit) {
            KeyFramePtr pKFi = mit->first;
            if (pKFi->mnBALocalForKF != pKF->mnId &&
                pKFi->mnBAFixedForKF != pKF->mnId) {
                pKFi->mnBAFixedForKF = pKF->mnId;
                if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap) {
                    lFixedKeyFrames.push_back(pKFi);
                }
            }
        }
    }

    for (auto lit = lLocalMapLines.begin(), lend = lLocalMapLines.end();
         lit != lend; ++lit) {
        const auto insertion =
            mLineObservations.emplace(*lit, (*lit)->GetObservations());
        const map<KeyFramePtr, vector<int>> &observations =
            insertion.first->second;
        for (auto mit = observations.begin(), mend = observations.end();
             mit != mend; ++mit) {
            KeyFramePtr pKFi = mit->first;
            if (pKFi->mnBALocalForKF != pKF->mnId &&
                pKFi->mnBAFixedForKF != pKF->mnId) {
                pKFi->mnBAFixedForKF = pKF->mnId;
                if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap) {
                    lFixedKeyFrames.push_back(pKFi);
                }
            }
        }
    }

    num_fixedKF += lFixedKeyFrames.size();

    if (num_fixedKF == 0) {
        return;
    }

    g2o::SparseOptimizer optimizer;
    auto linearSolver = std::make_unique<
        g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();

    auto solver_ptr =
        std::make_unique<g2o::BlockSolverX>(std::move(linearSolver));

    g2o::OptimizationAlgorithmLevenberg *solver =
        new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;

    for (auto lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end();
         lit != lend; ++lit) {
        KeyFramePtr pKFi = *lit;
        VertexPose *vPose = new VertexPose(pKFi);
        vPose->setId(pKFi->mnId);
        vPose->setFixed(pKFi->mnId == pMap->GetInitKFid());
        optimizer.addVertex(vPose);
        if (pKFi->mnId > maxKFid) {
            maxKFid = pKFi->mnId;
        }
    }
    num_OptKF = lLocalKeyFrames.size();

    for (auto lit = lFixedKeyFrames.begin(), lend = lFixedKeyFrames.end();
         lit != lend; ++lit) {
        KeyFramePtr pKFi = *lit;
        VertexPose *vPose = new VertexPose(pKFi);
        vPose->setId(pKFi->mnId);
        vPose->setFixed(true);
        optimizer.addVertex(vPose);
        if (pKFi->mnId > maxKFid) {
            maxKFid = pKFi->mnId;
        }
    }

    size_t nExpectedPointSize = 0;
    for (const auto &cachedObservations : mPointObservations) {
        for (const auto &observation : cachedObservations.second) {
            KeyFramePtr pKFi = observation.first;
            if (pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
                continue;
            for (int idx : observation.second)
                nExpectedPointSize += idx != -1;
        }
    }
    size_t nExpectedLineSize = 0;
    for (const auto &cachedObservations : mLineObservations) {
        for (const auto &observation : cachedObservations.second) {
            KeyFramePtr pKFi = observation.first;
            if (pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
                continue;
            for (int idx : observation.second)
                nExpectedLineSize += idx != -1;
        }
    }

    vector<EdgeProjectPoint *> vpEdgesPoint;
    vpEdgesPoint.reserve(nExpectedPointSize);

    vector<EdgeProjectLine *> vpEdgesLine;
    vpEdgesLine.reserve(nExpectedLineSize);

    vector<KeyFramePtr> vpEdgeKFPoint;
    vpEdgeKFPoint.reserve(nExpectedPointSize);

    vector<KeyFramePtr> vpEdgeKFLine;
    vpEdgeKFLine.reserve(nExpectedLineSize);

    vector<MapPointPtr> vpEdgeMP;
    vpEdgeMP.reserve(nExpectedPointSize);

    vector<MapLinePtr> vpEdgeML;
    vpEdgeML.reserve(nExpectedLineSize);

    const float thHuber = sqrt(5.991);

    unsigned long maxMPid = maxKFid + 1;

    int nEdges = 0;

    const double sigma_point = 1.0;

    for (auto lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end();
         lit != lend; ++lit) {
        MapPointPtr pMP = *lit;
        g2o::VertexPointXYZ *vPoint = new g2o::VertexPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
        unsigned long id = pMP->mnId + maxKFid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        if (id > maxMPid) {
            maxMPid = id;
        }

        const map<KeyFramePtr, vector<int>> &observations =
            mPointObservations.at(pMP);

        for (auto mit = observations.begin(), mend = observations.end();
             mit != mend; ++mit) {
            KeyFramePtr pKFi = mit->first;

            if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap) {
                for (size_t cam_id = 0; cam_id < mit->second.size(); ++cam_id) {
                    int idx = mit->second[cam_id];
                    if (idx == -1)
                        continue;

                    const cv::Point2f &ptUn = pKFi->mvKeysUn_cams[cam_id][idx];
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << ptUn.x, ptUn.y;

                    EdgeProjectPoint *e = new EdgeProjectPoint(cam_id);

                    e->setVertex(0,
                                 dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                     optimizer.vertex(id)));
                    e->setVertex(1,
                                 dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                     optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    e->setInformation(Eigen::Matrix2d::Identity() *
                                      (1.0 / (sigma_point * sigma_point)));

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber);

                    optimizer.addEdge(e);
                    vpEdgesPoint.push_back(e);
                    vpEdgeKFPoint.push_back(pKFi);
                    vpEdgeMP.push_back(pMP);

                    nEdges++;
                }
            }
        }
    }

    for (auto lit = lLocalMapLines.begin(), lend = lLocalMapLines.end();
         lit != lend; ++lit) {
        MapLinePtr pML = *lit;
        VertexLine *vLine = new VertexLine(pML);
        unsigned long id = pML->mnId + maxMPid + 1;
        vLine->setId(id);
        vLine->setMarginalized(true);
        optimizer.addVertex(vLine);

        const map<KeyFramePtr, vector<int>> &observations =
            mLineObservations.at(pML);

        for (auto mit = observations.begin(), mend = observations.end();
             mit != mend; ++mit) {
            KeyFramePtr pKFi = mit->first;

            if (!pKFi->isBad() && pKFi->GetMap() == pCurrentMap) {
                for (size_t cam_id = 0; cam_id < mit->second.size(); ++cam_id) {
                    int idx = mit->second[cam_id];
                    if (idx == -1)
                        continue;

                    const Eigen::Vector4f &lineUn =
                        pKFi->mvLineKeysUn_cams[cam_id][idx];

                    EdgeProjectLine *e = new EdgeProjectLine(cam_id);

                    e->setVertex(0,
                                 dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                     optimizer.vertex(id)));
                    e->setVertex(1,
                                 dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                     optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(lineUn.cast<double>());
                    e->setInformation(Eigen::Matrix2d::Identity() *
                                      kLineInformationFallback);
                    SetLineEndpointCovariance(e, *pKFi, cam_id, idx);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber);

                    optimizer.addEdge(e);
                    vpEdgesLine.push_back(e);
                    vpEdgeKFLine.push_back(pKFi);
                    vpEdgeML.push_back(pML);

                    nEdges++;
                }
            }
        }
    }

    num_edges = nEdges;

    if (pbStopFlag)
        if (*pbStopFlag)
            return;

    optimizer.initializeOptimization();
    optimizer.optimize(10);

    if (!HasFiniteLocalBAEstimates(optimizer)) {
        return;
    }

    vector<tuple<KeyFramePtr, MapPointPtr, int>> vToEraseMP;
    vector<tuple<KeyFramePtr, MapLinePtr, int>> vToEraseML;

    vToEraseMP.reserve(vpEdgesPoint.size());
    vToEraseML.reserve(vpEdgesLine.size());

    for (size_t i = 0; i < vpEdgesPoint.size(); ++i) {
        EdgeProjectPoint *e = vpEdgesPoint[i];
        MapPointPtr pMP = vpEdgeMP[i];

        if (pMP->isBad())
            continue;

        if (IsReprojectionOutlier(e, 5.991)) {
            KeyFramePtr pKFi = vpEdgeKFPoint[i];
            vToEraseMP.emplace_back(pKFi, pMP, e->cam_id);
        }
    }

    for (size_t i = 0; i < vpEdgesLine.size(); ++i) {
        EdgeProjectLine *e = vpEdgesLine[i];
        MapLinePtr pML = vpEdgeML[i];

        if (pML->isBad())
            continue;

        if (IsReprojectionOutlier(e, 5.991)) {
            KeyFramePtr pKFi = vpEdgeKFLine[i];
            vToEraseML.emplace_back(pKFi, pML, e->cam_id);
        }
    }

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    if (!vToEraseMP.empty()) {
        for (auto &t : vToEraseMP) {
            KeyFramePtr pKFi;
            MapPointPtr pMPi;
            int cam_id;
            tie(pKFi, pMPi, cam_id) = t;
            pKFi->EraseMapPointMatch(pMPi, cam_id);
            pMPi->EraseObservation(pKFi, cam_id);
        }
    }

    if (!vToEraseML.empty()) {
        for (auto &t : vToEraseML) {
            KeyFramePtr pKFi;
            MapLinePtr pMLi;
            int cam_id;
            tie(pKFi, pMLi, cam_id) = t;
            pKFi->EraseMapLineMatch(pMLi, cam_id);
            pMLi->EraseObservation(pKFi, cam_id);
        }
    }

    for (auto lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end();
         lit != lend; ++lit) {
        KeyFramePtr pKFi = *lit;
        VertexPose *vPose =
            static_cast<VertexPose *>(optimizer.vertex(pKFi->mnId));
        BasePose pose = vPose->estimate();
        const Sophus::SE3f Tbw = pose.ToTbw().cast<float>();
        pKFi->SetPose(Tbw);
    }

    for (auto lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end();
         lit != lend; ++lit) {
        MapPointPtr pMP = *lit;
        g2o::VertexPointXYZ *vPoint = static_cast<g2o::VertexPointXYZ *>(
            optimizer.vertex(pMP->mnId + maxKFid + 1));
        pMP->SetWorldPos(vPoint->estimate().cast<float>());
    }

    for (auto lit = lLocalMapLines.begin(), lend = lLocalMapLines.end();
         lit != lend; ++lit) {
        MapLinePtr pML = *lit;
        VertexLine *vLine = static_cast<VertexLine *>(
            optimizer.vertex(pML->mnId + maxMPid + 1));
        BaseLine line = vLine->estimate();
        pML->SetWorldPos(line.lineW.cast<float>());
    }

    pMap->IncreaseChangeIndex();
}

int Optimizer::OptimizeSE3(KeyFramePtr pKF1, KeyFramePtr pKF2,
                           vector<vector<MapPointPtr>> &vpMatchesMP1,
                           vector<vector<MapLinePtr>> &vpMatchesML1,
                           g2o::SE3Quat &g2oT12, const float th2) {
    g2o::SparseOptimizer optimizer;
    auto linearSolver = std::make_unique<
        g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>();
    auto solver_ptr =
        std::make_unique<g2o::BlockSolverX>(std::move(linearSolver));

    g2o::OptimizationAlgorithmLevenberg *solver =
        new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));
    optimizer.setAlgorithm(solver);

    const Eigen::Matrix3f R1w = pKF1->GetRotation();
    const Eigen::Vector3f t1w = pKF1->GetTranslation();
    const Eigen::Matrix3f R2w = pKF2->GetRotation();
    const Eigen::Vector3f t2w = pKF2->GetTranslation();

    int nCam = pKF1->mnCam;

    VertexoSE3 *vSE3 = new VertexoSE3(g2oT12);
    vSE3->setId(0);
    vSE3->setFixed(false);
    vSE3->vpCameras1 = pKF1->mvpCameras;
    vSE3->vpCameras2 = pKF2->mvpCameras;
    vector<Sophus::SE3f> vTbc = pKF1->GetCameraExtrinsics();
    for (size_t cam_id = 0; cam_id < vTbc.size(); ++cam_id) {
        vSE3->vTcb.push_back(vTbc[cam_id].inverse().cast<double>());
    }
    optimizer.addVertex(vSE3);

    int Np = 0;
    int Nl = 0;

    for (size_t cam_id = 0; cam_id < nCam; ++cam_id) {
        Np += vpMatchesMP1[cam_id].size();
        Nl += vpMatchesML1[cam_id].size();
    }

    const vector<vector<MapPointPtr>> vpMapPoints1 = pKF1->GetMapPointMatches();
    const vector<vector<MapLinePtr>> vpMapLines1 = pKF1->GetMapLineMatches();

    vector<EdgeSE3ProjectPoint *> vpEdgesPoint12;
    vector<EdgeSE3ProjectLine *> vpEdgesLine12;
    vector<EdgeInverseSE3ProjectPoint *> vpEdgesPoint21;
    vector<EdgeInverseSE3ProjectLine *> vpEdgesLine21;
    vector<pair<size_t, size_t>> vnCamIndexEdgePoint;
    vector<pair<size_t, size_t>> vnCamIndexEdgeLine;

    vpEdgesPoint12.reserve(2 * Np);
    vpEdgesLine12.reserve(2 * Nl);
    vpEdgesPoint21.reserve(2 * Np);
    vpEdgesLine21.reserve(2 * Nl);
    vnCamIndexEdgePoint.reserve(2 * Np);
    vnCamIndexEdgeLine.reserve(2 * Nl);

    const float deltaHuber = sqrt(th2);

    int nCorrespondences = 0;

    int idx = -1;
    double sigma_point = 1.0;
    for (size_t cam_id = 0; cam_id < nCam; ++cam_id) {
        for (int i = 0; i < vpMatchesMP1[cam_id].size(); ++i) {
            idx++;
            if (!vpMatchesMP1[cam_id][i])
                continue;
            MapPointPtr pMP1 = vpMapPoints1[cam_id][i];
            MapPointPtr pMP2 = vpMatchesMP1[cam_id][i];

            if (!pMP1 || !pMP2 || pMP1->isBad() || pMP2->isBad())
                continue;

            const vector<int> vi2 = pMP2->GetIndexInKeyFrame(pKF2);
            if (all_of(vi2.begin(), vi2.end(),
                       [](int idx) { return idx == -1; })) {
                continue;
            }

            const int id1 = 2 * idx + 1;
            const int id2 = 2 * idx + 2;

            g2o::VertexPointXYZ *vPoint1 = new g2o::VertexPointXYZ();
            Eigen::Vector3f P3D1w = pMP1->GetWorldPos();
            Eigen::Vector3f P3D1b = R1w * P3D1w + t1w;
            vPoint1->setEstimate(P3D1b.cast<double>());
            vPoint1->setId(id1);
            vPoint1->setFixed(true);
            optimizer.addVertex(vPoint1);

            g2o::VertexPointXYZ *vPoint2 = new g2o::VertexPointXYZ();
            Eigen::Vector3f P3D2w = pMP2->GetWorldPos();
            Eigen::Vector3f P3D2b = R2w * P3D2w + t2w;
            vPoint2->setEstimate(P3D2b.cast<double>());
            vPoint2->setId(id2);
            vPoint2->setFixed(true);
            optimizer.addVertex(vPoint2);

            nCorrespondences++;

            Eigen::Matrix<double, 2, 1> obs1;
            const cv::Point2f &pt1 = pKF1->mvKeysUn_cams[cam_id][i];
            obs1 << pt1.x, pt1.y;

            EdgeSE3ProjectPoint *e12 = new EdgeSE3ProjectPoint(cam_id);

            e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                  optimizer.vertex(id2)));
            e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                  optimizer.vertex(0)));
            e12->setMeasurement(obs1);
            e12->setInformation(Eigen::Matrix2d::Identity() *
                                (1.0 / (sigma_point * sigma_point)));

            g2o::RobustKernelHuber *rk1 = new g2o::RobustKernelHuber;
            e12->setRobustKernel(rk1);
            rk1->setDelta(deltaHuber);
            optimizer.addEdge(e12);

            Eigen::Matrix<double, 2, 1> obs2;
            EdgeInverseSE3ProjectPoint *e21;
            for (int j = 0; j < vi2.size(); ++j) {
                if (vi2[j] == -1)
                    continue;
                const cv::Point2f &pt2 = pKF2->mvKeysUn_cams[j][vi2[j]];
                obs2 << pt2.x, pt2.y;
                e21 = new EdgeInverseSE3ProjectPoint(j);
                e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                      optimizer.vertex(id1)));
                e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                      optimizer.vertex(0)));
                e21->setMeasurement(obs2);
                e21->setInformation(Eigen::Matrix2d::Identity() *
                                    (1.0 / (sigma_point * sigma_point)));

                g2o::RobustKernelHuber *rk2 = new g2o::RobustKernelHuber;
                e21->setRobustKernel(rk2);
                rk2->setDelta(deltaHuber);
                optimizer.addEdge(e21);

                break;
            }

            vpEdgesPoint12.push_back(e12);
            vpEdgesPoint21.push_back(e21);
            vnCamIndexEdgePoint.emplace_back(cam_id, i);
        }

        for (int i = 0; i < vpMatchesML1[cam_id].size(); ++i) {
            idx++;
            MapLinePtr pML1 = vpMapLines1[cam_id][i];
            MapLinePtr pML2 = vpMatchesML1[cam_id][i];

            if (!pML1 || !pML2 || pML1->isBad() || pML2->isBad())
                continue;

            const vector<int> vi2 = pML2->GetIndexInKeyFrame(pKF2);
            int cam_id2 = -1;
            int i2 = -1;
            for (int j = 0; j < vi2.size(); ++j) {
                if (vi2[j] != -1) {
                    cam_id2 = j;
                    i2 = vi2[j];
                    break;
                }
            }
            if (cam_id2 == -1)
                continue;

            const int id1 = 2 * idx + 1;
            const int id2 = 2 * idx + 2;

            BaseLine line1;
            BaseLine line2;
            if (!MakeBaseLineEstimate(
                    TransformLineToBody(pML1->GetWorldPos(), R1w, t1w),
                    line1) ||
                !MakeBaseLineEstimate(
                    TransformLineToBody(pML2->GetWorldPos(), R2w, t2w),
                    line2)) {
                continue;
            }

            VertexLine *vLine1 = new VertexLine();
            vLine1->setEstimate(line1);
            vLine1->setId(id1);
            vLine1->setFixed(true);
            optimizer.addVertex(vLine1);

            VertexLine *vLine2 = new VertexLine();
            vLine2->setEstimate(line2);
            vLine2->setId(id2);
            vLine2->setFixed(true);
            optimizer.addVertex(vLine2);

            nCorrespondences++;

            const Eigen::Vector4f &obs1 = pKF1->mvLineKeysUn_cams[cam_id][i];

            EdgeSE3ProjectLine *e12 = new EdgeSE3ProjectLine(cam_id);
            e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                  optimizer.vertex(id2)));
            e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                  optimizer.vertex(0)));
            e12->setMeasurement(obs1.cast<double>());
            e12->setInformation(Eigen::Matrix2d::Identity() *
                                kLineInformationFallback);
            SetLineEndpointCovariance(e12, *pKF1, cam_id, i);

            g2o::RobustKernelHuber *rk1 = new g2o::RobustKernelHuber;
            e12->setRobustKernel(rk1);
            rk1->setDelta(deltaHuber);
            optimizer.addEdge(e12);

            const Eigen::Vector4f &obs2 = pKF2->mvLineKeysUn_cams[cam_id2][i2];

            EdgeInverseSE3ProjectLine *e21 =
                new EdgeInverseSE3ProjectLine(cam_id2);
            e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                  optimizer.vertex(id1)));
            e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                  optimizer.vertex(0)));
            e21->setMeasurement(obs2.cast<double>());
            e21->setInformation(Eigen::Matrix2d::Identity() *
                                kLineInformationFallback);
            SetLineEndpointCovariance(e21, *pKF2,
                                      static_cast<size_t>(cam_id2), i2);

            g2o::RobustKernelHuber *rk2 = new g2o::RobustKernelHuber;
            e21->setRobustKernel(rk2);
            rk2->setDelta(deltaHuber);
            optimizer.addEdge(e21);

            vpEdgesLine12.push_back(e12);
            vpEdgesLine21.push_back(e21);
            vnCamIndexEdgeLine.emplace_back(cam_id, i);
        }
    }

    optimizer.initializeOptimization();
    optimizer.optimize(5);

    int nBad = 0;
    for (size_t i = 0; i < vpEdgesPoint12.size(); ++i) {
        EdgeSE3ProjectPoint *e12 = vpEdgesPoint12[i];
        EdgeInverseSE3ProjectPoint *e21 = vpEdgesPoint21[i];

        if (!e12 || !e21)
            continue;

        if (e12->chi2() > th2 || e21->chi2() > th2) {
            size_t cam_id, idx;
            tie(cam_id, idx) = vnCamIndexEdgePoint[i];
            vpMatchesMP1[cam_id][idx] = nullptr;
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdgesPoint12[i] = nullptr;
            vpEdgesPoint21[i] = nullptr;
            nBad++;

            continue;
        }

        e12->setRobustKernel(0);
        e21->setRobustKernel(0);
    }

    for (size_t i = 0; i < vpEdgesLine12.size(); ++i) {
        EdgeSE3ProjectLine *e12 = vpEdgesLine12[i];
        EdgeInverseSE3ProjectLine *e21 = vpEdgesLine21[i];

        if (!e12 || !e21)
            continue;

        if (e12->chi2() > th2 || e21->chi2() > th2) {
            size_t cam_id, idx;
            tie(cam_id, idx) = vnCamIndexEdgeLine[i];
            vpMatchesML1[cam_id][idx] = nullptr;
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdgesLine12[i] = nullptr;
            vpEdgesLine21[i] = nullptr;
            nBad++;

            continue;
        }

        e12->setRobustKernel(0);
        e21->setRobustKernel(0);
    }

    int nMoreIterations = 0;
    if (nBad > 0)
        nMoreIterations = 10;
    else
        nMoreIterations = 5;

    if (nCorrespondences - nBad < 10) {
        return 0;
    }

    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);

    int nIn = 0;
    for (size_t i = 0; i < vpEdgesPoint12.size(); ++i) {
        EdgeSE3ProjectPoint *e12 = vpEdgesPoint12[i];
        EdgeInverseSE3ProjectPoint *e21 = vpEdgesPoint21[i];

        if (!e12 || !e21)
            continue;

        e12->computeError();
        e21->computeError();

        if (e12->chi2() > th2 || e21->chi2() > th2) {
            size_t cam_id, idx;
            tie(cam_id, idx) = vnCamIndexEdgePoint[i];
            vpMatchesMP1[cam_id][idx] = nullptr;
            nBad++;
        } else {
            nIn++;
        }
    }

    for (size_t i = 0; i < vpEdgesLine12.size(); ++i) {
        EdgeSE3ProjectLine *e12 = vpEdgesLine12[i];
        EdgeInverseSE3ProjectLine *e21 = vpEdgesLine21[i];

        if (!e12 || !e21)
            continue;

        e12->computeError();
        e21->computeError();

        if (e12->chi2() > th2 || e21->chi2() > th2) {
            size_t cam_id, idx;
            tie(cam_id, idx) = vnCamIndexEdgeLine[i];
            vpMatchesML1[cam_id][idx] = nullptr;
            nBad++;
        } else {
            nIn++;
        }
    }

    VertexoSE3 *vSE3_after = static_cast<VertexoSE3 *>(optimizer.vertex(0));
    g2oT12.setRotation(
        Eigen::Quaterniond(vSE3_after->estimate().Rbw.cast<double>()));
    g2oT12.setTranslation(vSE3_after->estimate().tbw.cast<double>());
    return nIn;
}

void Optimizer::LocalBundleAdjustment(KeyFramePtr pMainKF,
                                      std::vector<KeyFramePtr> vpAdjustKF,
                                      std::vector<KeyFramePtr> vpFixedKF,
                                      bool *pbStopFlag) {
    vector<MapPointPtr> vpMPs;
    vector<MapLinePtr> vpMLs;

    g2o::SparseOptimizer optimizer;
    auto linearSolver = std::make_unique<
        g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
    auto solver_ptr =
        std::make_unique<g2o::BlockSolverX>(std::move(linearSolver));
    g2o::OptimizationAlgorithmLevenberg *solver =
        new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned maxKFid = 0;
    set<KeyFramePtr> spKeyFrameBA;

    Map *pCurrentMap = pMainKF->GetMap();

    for (KeyFramePtr pKFi : vpFixedKF) {
        if (pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
            continue;

        pKFi->mnBALocalForMerge = pMainKF->mnId;

        VertexPose *vPose = new VertexPose(pKFi);
        vPose->setId(pKFi->mnId);
        vPose->setFixed(true);
        optimizer.addVertex(vPose);
        if (pKFi->mnId > maxKFid) {
            maxKFid = pKFi->mnId;
        }

        set<MapPointPtr> spViewMPs = pKFi->GetMapPoints();
        set<MapLinePtr> spViewMLs = pKFi->GetMapLines();
        for (MapPointPtr pMP : spViewMPs) {
            if (!pMP || pMP->isBad() || pMP->GetMap() != pCurrentMap)
                continue;
            if (pMP->mnBALocalForMerge == pMainKF->mnId)
                continue;
            vpMPs.push_back(pMP);
            pMP->mnBALocalForMerge = pMainKF->mnId;
        }
        for (MapLinePtr pML : spViewMLs) {
            if (!pML || pML->isBad() || pML->GetMap() != pCurrentMap)
                continue;
            if (pML->mnBALocalForMerge == pMainKF->mnId)
                continue;
            vpMLs.push_back(pML);
            pML->mnBALocalForMerge = pMainKF->mnId;
        }
        spKeyFrameBA.insert(pKFi);
    }

    set<KeyFramePtr> spAdjustKF(vpAdjustKF.begin(), vpAdjustKF.end());
    for (KeyFramePtr pKFi : vpAdjustKF) {
        if (pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
            continue;

        pKFi->mnBALocalForMerge = pMainKF->mnId;

        VertexPose *vPose = new VertexPose(pKFi);
        vPose->setId(pKFi->mnId);
        vPose->setFixed(false);
        optimizer.addVertex(vPose);
        if (pKFi->mnId > maxKFid) {
            maxKFid = pKFi->mnId;
        }

        set<MapPointPtr> spViewMPs = pKFi->GetMapPoints();
        set<MapLinePtr> spViewMLs = pKFi->GetMapLines();
        for (MapPointPtr pMP : spViewMPs) {
            if (!pMP || pMP->isBad() || pMP->GetMap() != pCurrentMap)
                continue;
            if (pMP->mnBALocalForMerge == pMainKF->mnId)
                continue;
            vpMPs.push_back(pMP);
            pMP->mnBALocalForMerge = pMainKF->mnId;
        }
        for (MapLinePtr pML : spViewMLs) {
            if (!pML || pML->isBad() || pML->GetMap() != pCurrentMap)
                continue;
            if (pML->mnBALocalForMerge == pMainKF->mnId)
                continue;
            vpMLs.push_back(pML);
            pML->mnBALocalForMerge = pMainKF->mnId;
        }
        spKeyFrameBA.insert(pKFi);
    }

#ifdef GRAPHITE
    if (!pbStopFlag || !(*pbStopFlag)) {
        if (TryMergeLocalBundleAdjustmentWithGraphite(pMainKF, vpAdjustKF,
                                                      vpFixedKF, vpMPs, vpMLs,
                                                      maxKFid, pbStopFlag)) {
            return;
        }
    }
#endif

    const int nExpectedPointSize =
        (vpAdjustKF.size() + vpFixedKF.size()) * vpMPs.size();
    const int nExpectedLineSize =
        (vpAdjustKF.size() + vpFixedKF.size()) * vpMLs.size();

    vector<EdgeProjectPoint *> vpEdgesPoint;
    vpEdgesPoint.reserve(nExpectedPointSize);

    vector<EdgeProjectLine *> vpEdgesLine;
    vpEdgesLine.reserve(nExpectedLineSize);

    vector<KeyFramePtr> vpEdgeKFPoint;
    vpEdgeKFPoint.reserve(nExpectedPointSize);

    vector<KeyFramePtr> vpEdgeKFLine;
    vpEdgeKFLine.reserve(nExpectedLineSize);

    vector<MapPointPtr> vpEdgeMP;
    vpEdgeMP.reserve(nExpectedPointSize);

    vector<MapLinePtr> vpEdgeML;
    vpEdgeML.reserve(nExpectedLineSize);

    const float thHuber = sqrt(5.991);

    int maxMPid = maxKFid + 1;

    const double sigma_point = 1.0;

    for (unsigned int i = 0; i < vpMPs.size(); ++i) {
        MapPointPtr pMP = vpMPs[i];
        if (pMP->isBad())
            continue;

        g2o::VertexPointXYZ *vPoint = new g2o::VertexPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
        unsigned long id = pMP->mnId + maxKFid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        if (id > maxMPid) {
            maxMPid = id;
        }

        const map<KeyFramePtr, vector<int>> observations =
            pMP->GetObservations();

        for (auto mit = observations.begin(), mend = observations.end();
             mit != mend; ++mit) {
            KeyFramePtr pKFi = mit->first;

            if (pKFi->isBad() || pKFi->mnId > maxKFid ||
                pKFi->mnBALocalForMerge != pMainKF->mnId)
                continue;

            for (size_t cam_id = 0; cam_id < mit->second.size(); ++cam_id) {
                int idx = mit->second[cam_id];
                if (idx == -1)
                    continue;

                const cv::Point2f &ptUn = pKFi->mvKeysUn_cams[cam_id][idx];
                Eigen::Matrix<double, 2, 1> obs;
                obs << ptUn.x, ptUn.y;

                EdgeProjectPoint *e = new EdgeProjectPoint(cam_id);

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                    optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                    optimizer.vertex(pKFi->mnId)));
                e->setMeasurement(obs);
                e->setInformation(Eigen::Matrix2d::Identity() *
                                  (1.0 / (sigma_point * sigma_point)));

                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuber);

                optimizer.addEdge(e);
                vpEdgesPoint.push_back(e);
                vpEdgeKFPoint.push_back(pKFi);
                vpEdgeMP.push_back(pMP);
            }
        }
    }

    for (unsigned int i = 0; i < vpMLs.size(); ++i) {
        MapLinePtr pML = vpMLs[i];
        if (pML->isBad())
            continue;

        VertexLine *vLine = new VertexLine(pML);
        unsigned long id = pML->mnId + maxMPid + 1;
        vLine->setId(id);
        vLine->setMarginalized(true);
        optimizer.addVertex(vLine);

        const map<KeyFramePtr, vector<int>> observations =
            pML->GetObservations();

        for (auto mit = observations.begin(), mend = observations.end();
             mit != mend; ++mit) {
            KeyFramePtr pKFi = mit->first;

            if (pKFi->isBad() || pKFi->mnId > maxKFid ||
                pKFi->mnBALocalForMerge != pMainKF->mnId)
                continue;

            for (size_t cam_id = 0; cam_id < mit->second.size(); ++cam_id) {
                int idx = mit->second[cam_id];
                if (idx == -1)
                    continue;

                const Eigen::Vector4f &lineUn =
                    pKFi->mvLineKeysUn_cams[cam_id][idx];

                EdgeProjectLine *e = new EdgeProjectLine(cam_id);

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                    optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                    optimizer.vertex(pKFi->mnId)));
                e->setMeasurement(lineUn.cast<double>());
                e->setInformation(Eigen::Matrix2d::Identity() *
                                  kLineInformationFallback);
                SetLineEndpointCovariance(e, *pKFi, cam_id, idx);

                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuber);

                optimizer.addEdge(e);
                vpEdgesLine.push_back(e);
                vpEdgeKFLine.push_back(pKFi);
                vpEdgeML.push_back(pML);
            }
        }
    }

    if (pbStopFlag)
        if (*pbStopFlag)
            return;

    optimizer.initializeOptimization();
    optimizer.optimize(5);

    if (!HasFiniteLocalBAEstimates(optimizer)) {
        return;
    }

    bool bDoMore = true;

    if (pbStopFlag)
        if (*pbStopFlag)
            bDoMore = false;

    if (bDoMore) {
        for (size_t i = 0; i < vpEdgesPoint.size(); ++i) {
            EdgeProjectPoint *e = vpEdgesPoint[i];
            MapPointPtr pMP = vpEdgeMP[i];

            if (pMP->isBad())
                continue;

            if (IsReprojectionOutlier(e, 5.991)) {
                e->setLevel(1);
            }
            e->setRobustKernel(0);
        }

        for (size_t i = 0; i < vpEdgesLine.size(); ++i) {
            EdgeProjectLine *e = vpEdgesLine[i];
            MapLinePtr pML = vpEdgeML[i];

            if (pML->isBad())
                continue;

            if (IsReprojectionOutlier(e, 5.991)) {
                e->setLevel(1);
            }
            e->setRobustKernel(0);
        }
        optimizer.initializeOptimization(0);
        optimizer.optimize(10);
        if (!HasFiniteLocalBAEstimates(optimizer)) {
            return;
        }
    }

    vector<tuple<KeyFramePtr, MapPointPtr, int>> vToEraseMP;
    vector<tuple<KeyFramePtr, MapLinePtr, int>> vToEraseML;
    vToEraseMP.reserve(vpEdgesPoint.size());
    vToEraseML.reserve(vpEdgesLine.size());

    for (size_t i = 0, iend = vpEdgesPoint.size(); i < iend; ++i) {
        EdgeProjectPoint *e = vpEdgesPoint[i];
        MapPointPtr pMP = vpEdgeMP[i];

        if (pMP->isBad())
            continue;

        if (IsReprojectionOutlier(e, 5.991)) {
            KeyFramePtr pKFi = vpEdgeKFPoint[i];
            vToEraseMP.emplace_back(pKFi, pMP, e->cam_id);
        }
    }

    for (size_t i = 0, iend = vpEdgesLine.size(); i < iend; ++i) {
        EdgeProjectLine *e = vpEdgesLine[i];
        MapLinePtr pML = vpEdgeML[i];

        if (pML->isBad())
            continue;

        if (IsReprojectionOutlier(e, 5.991)) {
            KeyFramePtr pKFi = vpEdgeKFLine[i];
            vToEraseML.emplace_back(pKFi, pML, e->cam_id);
        }
    }

    unique_lock<mutex> lock(pCurrentMap->mMutexMapUpdate);

    if (!vToEraseMP.empty()) {
        for (auto &t : vToEraseMP) {
            KeyFramePtr pKFi;
            MapPointPtr pMPi;
            int cam_id;
            tie(pKFi, pMPi, cam_id) = t;
            pKFi->EraseMapPointMatch(pMPi, cam_id);
            pMPi->EraseObservation(pKFi, cam_id);
        }
    }
    if (!vToEraseML.empty()) {
        for (auto &t : vToEraseML) {
            KeyFramePtr pKFi;
            MapLinePtr pMLi;
            int cam_id;
            tie(pKFi, pMLi, cam_id) = t;
            pKFi->EraseMapLineMatch(pMLi, cam_id);
            pMLi->EraseObservation(pKFi, cam_id);
        }
    }

    for (KeyFramePtr pKFi : vpAdjustKF) {
        if (pKFi->isBad())
            continue;

        VertexPose *vPose =
            static_cast<VertexPose *>(optimizer.vertex(pKFi->mnId));
        BasePose pose = vPose->estimate();
        const Sophus::SE3f Tbw = pose.ToTbw().cast<float>();
        pKFi->SetPose(Tbw);
    }

    for (MapPointPtr pMP : vpMPs) {
        if (pMP->isBad())
            continue;

        g2o::VertexPointXYZ *vPoint = static_cast<g2o::VertexPointXYZ *>(
            optimizer.vertex(pMP->mnId + maxKFid + 1));
        pMP->SetWorldPos(vPoint->estimate().cast<float>());
    }
    for (MapLinePtr pML : vpMLs) {
        if (pML->isBad())
            continue;

        VertexLine *vLine = static_cast<VertexLine *>(
            optimizer.vertex(pML->mnId + maxMPid + 1));
        BaseLine line = vLine->estimate();
        pML->SetWorldPos(line.lineW.cast<float>());
    }
}

void Optimizer::OptimizeEssentialGraph(
    KeyFramePtr pCurKF, std::vector<KeyFramePtr> &vpFixedKFs,
    std::vector<KeyFramePtr> &vpFixedCorrectedKFs,
    std::vector<KeyFramePtr> &vpNonFixedKFs,
    std::vector<MapPointPtr> &vpNonCorrectedMPs,
    std::vector<MapLinePtr> &vpNonCorrectedMLs) {
    g2o::SparseOptimizer optimizer;
    auto linearSolver = std::make_unique<
        g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
    auto solver_ptr =
        std::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver));
    g2o::OptimizationAlgorithmLevenberg *solver =
        new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    Map *pMap = pCurKF->GetMap();
    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    vector<g2o::SE3Quat, Eigen::aligned_allocator<g2o::SE3Quat>> vTbw(nMaxKFid +
                                                                      1);
    vector<g2o::SE3Quat, Eigen::aligned_allocator<g2o::SE3Quat>> vCorrectedTwb(
        nMaxKFid + 1);
    vector<VertexPoseGraph *> vpVertices(nMaxKFid + 1);

    vector<bool> vpGoodPose(nMaxKFid + 1);
    vector<bool> vpBadPose(nMaxKFid + 1);

    const int minFeat = 200;

    for (KeyFramePtr pKFi : vpFixedKFs) {
        if (pKFi->isBad())
            continue;

        VertexPoseGraph *vSE3 = new VertexPoseGraph();

        const int nIDi = pKFi->mnId;

        Sophus::SE3d Tbw = pKFi->GetPose().cast<double>();
        g2o::SE3Quat Tiw = ToG2OSE3Quat(Tbw);

        vCorrectedTwb[nIDi] = Tiw.inverse();
        vSE3->setEstimate(Tiw.inverse());

        vSE3->setFixed(true);

        vSE3->setId(nIDi);
        vSE3->setMarginalized(false);

        optimizer.addVertex(vSE3);

        vpVertices[nIDi] = vSE3;

        vpGoodPose[nIDi] = true;
        vpBadPose[nIDi] = false;
    }

    set<unsigned long> sIdKF;
    for (KeyFramePtr pKFi : vpFixedCorrectedKFs) {
        if (pKFi->isBad())
            continue;

        VertexPoseGraph *vSE3 = new VertexPoseGraph();

        const int nIDi = pKFi->mnId;

        Sophus::SE3d Tbw = pKFi->GetPose().cast<double>();
        g2o::SE3Quat Tiw = ToG2OSE3Quat(Tbw);

        vCorrectedTwb[nIDi] = Tiw.inverse();
        vSE3->setEstimate(Tiw.inverse());

        Sophus::SE3d Tbw_bef = pKFi->mTbwBefMerge.cast<double>();
        vTbw[nIDi] = ToG2OSE3Quat(Tbw_bef);

        vSE3->setFixed(true);

        vSE3->setId(nIDi);
        vSE3->setMarginalized(false);

        optimizer.addVertex(vSE3);

        vpVertices[nIDi] = vSE3;

        sIdKF.insert(nIDi);

        vpGoodPose[nIDi] = true;
        vpBadPose[nIDi] = true;
    }

    for (KeyFramePtr pKFi : vpNonFixedKFs) {
        if (pKFi->isBad())
            continue;

        const int nIDi = pKFi->mnId;

        if (sIdKF.count(
                nIDi)) // It has already added in the corrected merge KFs
            continue;

        VertexPoseGraph *vSE3 = new VertexPoseGraph();

        Sophus::SE3d Tbw = pKFi->GetPose().cast<double>();
        g2o::SE3Quat Tiw = ToG2OSE3Quat(Tbw);

        vTbw[nIDi] = Tiw;
        vSE3->setEstimate(Tiw.inverse());

        vSE3->setFixed(false);

        vSE3->setId(nIDi);
        vSE3->setMarginalized(false);

        optimizer.addVertex(vSE3);

        vpVertices[nIDi] = vSE3;

        sIdKF.insert(nIDi);

        vpGoodPose[nIDi] = false;
        vpBadPose[nIDi] = true;
    }

    vector<KeyFramePtr> vpKFs;
    vpKFs.reserve(vpFixedKFs.size() + vpFixedCorrectedKFs.size() +
                  vpNonFixedKFs.size());
    vpKFs.insert(vpKFs.end(), vpFixedKFs.begin(), vpFixedKFs.end());
    vpKFs.insert(vpKFs.end(), vpFixedCorrectedKFs.begin(),
                 vpFixedCorrectedKFs.end());
    vpKFs.insert(vpKFs.end(), vpNonFixedKFs.begin(), vpNonFixedKFs.end());
    set<KeyFramePtr> spKFs(vpKFs.begin(), vpKFs.end());

    const Eigen::Matrix<double, 6, 6> matLambda =
        Eigen::Matrix<double, 6, 6>::Identity();

    for (KeyFramePtr pKFi : vpKFs) {
        int num_connections = 0;
        const int nIDi = pKFi->mnId;

        g2o::SE3Quat Twi;
        if (vpBadPose[nIDi])
            Twi = vTbw[nIDi].inverse();
        else if (vpGoodPose[nIDi])
            Twi = vCorrectedTwb[nIDi];

        KeyFramePtr pParentKFi = pKFi->GetParent();

        // Spanning tree edge
        if (pParentKFi && spKFs.find(pParentKFi) != spKFs.end()) {
            int nIDj = pParentKFi->mnId;

            g2o::SE3Quat Tjw;
            bool bHasRelation = false;

            if (vpGoodPose[nIDi] && vpGoodPose[nIDj]) {
                Tjw = vCorrectedTwb[nIDj].inverse();
                bHasRelation = true;
            } else if (vpBadPose[nIDi] && vpBadPose[nIDj]) {
                Tjw = vTbw[nIDj];
                bHasRelation = true;
            }

            if (bHasRelation) {
                g2o::SE3Quat Tji = Tjw * Twi;

                EdgeEssentialGraphSE3 *e = new EdgeEssentialGraphSE3();
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                    optimizer.vertex(nIDj)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                    optimizer.vertex(nIDi)));
                e->setMeasurement(Tji);

                e->information() = matLambda;
                optimizer.addEdge(e);

                num_connections++;
            }
        }

        // Loop edges
        const set<KeyFramePtr> sLoopEdges = pKFi->GetLoopEdges();
        for (set<KeyFramePtr>::const_iterator sit = sLoopEdges.begin(),
                                              send = sLoopEdges.end();
             sit != send; sit++) {
            KeyFramePtr pLKF = *sit;
            if (spKFs.find(pLKF) != spKFs.end() && pLKF->mnId < pKFi->mnId) {
                g2o::SE3Quat Tlw;
                bool bHasRelation = false;

                if (vpGoodPose[nIDi] && vpGoodPose[pLKF->mnId]) {
                    Tlw = vCorrectedTwb[pLKF->mnId].inverse();
                    bHasRelation = true;
                } else if (vpBadPose[nIDi] && vpBadPose[pLKF->mnId]) {
                    Tlw = vTbw[pLKF->mnId];
                    bHasRelation = true;
                }

                if (bHasRelation) {
                    g2o::SE3Quat Tli = Tlw * Twi;
                    EdgeEssentialGraphSE3 *el = new EdgeEssentialGraphSE3();
                    el->setVertex(1,
                                  dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                      optimizer.vertex(pLKF->mnId)));
                    el->setVertex(0,
                                  dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                      optimizer.vertex(nIDi)));
                    el->setMeasurement(Tli);
                    el->information() = matLambda;
                    optimizer.addEdge(el);

                    num_connections++;
                }
            }
        }

        // Covisibility graph edges
        const vector<KeyFramePtr> vpConnectedKFs =
            pKFi->GetCovisiblesByWeight(minFeat);
        for (vector<KeyFramePtr>::const_iterator vit = vpConnectedKFs.begin();
             vit != vpConnectedKFs.end(); vit++) {
            KeyFramePtr pKFn = *vit;
            if (pKFn && pKFn != pParentKFi && !pKFi->hasChild(pKFn) &&
                !sLoopEdges.count(pKFn) && spKFs.find(pKFn) != spKFs.end()) {
                if (!pKFn->isBad() && pKFn->mnId < pKFi->mnId) {

                    g2o::SE3Quat Tnw = vTbw[pKFn->mnId];
                    bool bHasRelation = false;

                    if (vpGoodPose[nIDi] && vpGoodPose[pKFn->mnId]) {
                        Tnw = vCorrectedTwb[pKFn->mnId].inverse();
                        bHasRelation = true;
                    } else if (vpBadPose[nIDi] && vpBadPose[pKFn->mnId]) {
                        Tnw = vTbw[pKFn->mnId];
                        bHasRelation = true;
                    }

                    if (bHasRelation) {
                        g2o::SE3Quat Tni = Tnw * Twi;

                        EdgeEssentialGraphSE3 *en = new EdgeEssentialGraphSE3();
                        en->setVertex(
                            1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                   optimizer.vertex(pKFn->mnId)));
                        en->setVertex(
                            0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                   optimizer.vertex(nIDi)));
                        en->setMeasurement(Tni);
                        en->information() = matLambda;
                        optimizer.addEdge(en);

                        num_connections++;
                    }
                }
            }
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 pose recovery.
    for (KeyFramePtr pKFi : vpNonFixedKFs) {
        if (pKFi->isBad())
            continue;

        const int nIDi = pKFi->mnId;

        VertexPoseGraph *vSE3 =
            static_cast<VertexPoseGraph *>(optimizer.vertex(nIDi));
        g2o::SE3Quat CorrectedTiw = vSE3->estimate().inverse();
        vCorrectedTwb[nIDi] = CorrectedTiw.inverse();
        Sophus::SE3d Tbw = ToSophusSE3(CorrectedTiw);

        pKFi->mTbwBefMerge = pKFi->GetPose();
        pKFi->mTwbBefMerge = pKFi->GetPoseInverse();
        pKFi->SetPose(Tbw.cast<float>());
    }

    for (MapPointPtr pMPi : vpNonCorrectedMPs) {
        if (pMPi->isBad())
            continue;

        KeyFramePtr pRefKF = pMPi->GetReferenceKeyFrame();
        while (pRefKF->isBad()) {
            if (!pRefKF) {
                break;
            }

            pMPi->EraseObservation(pRefKF);
            pRefKF = pMPi->GetReferenceKeyFrame();
        }

        if (vpBadPose[pRefKF->mnId]) {
            Sophus::SE3f TNonCorrectedwr = pRefKF->mTwbBefMerge;
            Sophus::SE3f Twr = pRefKF->GetPoseInverse();

            Eigen::Vector3f eigCorrectedP3Dw =
                Twr * TNonCorrectedwr.inverse() * pMPi->GetWorldPos();
            pMPi->SetWorldPos(eigCorrectedP3Dw);
        } else {
            cout << "ERROR: MapPoint has a reference KF from another map"
                 << endl;
        }
    }

    for (MapLinePtr pMLi : vpNonCorrectedMLs) {
        if (pMLi->isBad())
            continue;

        KeyFramePtr pRefKF = pMLi->GetReferenceKeyFrame();
        while (pRefKF->isBad()) {
            if (!pRefKF) {
                break;
            }

            pMLi->EraseObservation(pRefKF);
            pRefKF = pMLi->GetReferenceKeyFrame();
        }

        if (vpBadPose[pRefKF->mnId]) {
            Sophus::SE3f TNonCorrectedrw = pRefKF->mTwbBefMerge.inverse();
            Sophus::SE3f Twr = pRefKF->GetPoseInverse();

            Eigen::Matrix<float, 6, 1> lineW = pMLi->GetWorldPos();
            Eigen::Matrix<float, 6, 1> liner =
                TransformLineToBody(lineW, TNonCorrectedrw.rotationMatrix(),
                                    TNonCorrectedrw.translation())
                    .cast<float>();
            Eigen::Matrix<float, 6, 1> lineCorrectedw =
                TransformLineToBody(liner, Twr.rotationMatrix(),
                                    Twr.translation())
                    .cast<float>();

            pMLi->SetWorldPos(lineCorrectedw);
        } else {
            cout << "ERROR: MapLine has a reference KF from another map"
                 << endl;
        }
    }
}

void Optimizer::OptimizeEssentialGraph(
    Map *pMap, KeyFramePtr pLoopKF, KeyFramePtr pCurKF,
    const LoopClosing::KeyFrameAndPose &NonCorrectedSE3,
    const LoopClosing::KeyFrameAndPose &CorrectedSE3,
    const std::map<KeyFramePtr, std::set<KeyFramePtr>> &LoopConnections,
    const bool bFixScale) {
#ifdef GRAPHITE
    if (TryOptimizeEssentialGraphWithGraphite(pMap, pLoopKF, pCurKF,
                                              NonCorrectedSE3, CorrectedSE3,
                                              LoopConnections)) {
        return;
    }
#endif
    cout << "Optimizing Essential Graph with " << LoopConnections.size()
         << " KFs" << endl;

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    auto linearSolver = std::make_unique<
        g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
    auto solver_ptr =
        std::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver));
    g2o::OptimizationAlgorithmLevenberg *solver =
        new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));

    (void)bFixScale;

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    const vector<KeyFramePtr> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPointPtr> vpMPs = pMap->GetAllMapPoints();
    const vector<MapLinePtr> vpMLs = pMap->GetAllMapLines();

    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    vector<g2o::SE3Quat, Eigen::aligned_allocator<g2o::SE3Quat>> vScw(nMaxKFid +
                                                                      1);
    vector<g2o::SE3Quat, Eigen::aligned_allocator<g2o::SE3Quat>> vCorrectedSwc(
        nMaxKFid + 1);
    vector<VertexPoseGraph *> vpVertices(nMaxKFid + 1, nullptr);

    const int minFeat = 200;

    for (KeyFramePtr pKF : vpKFs) {
        if (!pKF || pKF->isBad())
            continue;

        VertexPoseGraph *vSE3 = new VertexPoseGraph();

        const int nIDi = pKF->mnId;
        LoopClosing::KeyFrameAndPose::const_iterator it =
            CorrectedSE3.find(pKF);

        if (it != CorrectedSE3.end()) {
            vScw[nIDi] = it->second;
            vSE3->setEstimate(vScw[nIDi].inverse());
        } else {
            Sophus::SE3d Tbw = pKF->GetPose().cast<double>();
            vScw[nIDi] = ToG2OSE3Quat(Tbw);
            vSE3->setEstimate(vScw[nIDi].inverse());
        }

        if (pKF->mnId == pMap->GetInitKFid())
            vSE3->setFixed(true);

        vSE3->setId(nIDi);
        vSE3->setMarginalized(false);

        optimizer.addVertex(vSE3);
        vpVertices[nIDi] = vSE3;
    }

    set<pair<long unsigned int, long unsigned int>> sInsertedEdges;
    const Eigen::Matrix<double, 6, 6> matLambda =
        Eigen::Matrix<double, 6, 6>::Identity();

    auto GetNonCorrectedScw = [&](KeyFramePtr pKF) {
        LoopClosing::KeyFrameAndPose::const_iterator it =
            NonCorrectedSE3.find(pKF);
        if (it != NonCorrectedSE3.end())
            return it->second;
        return vScw[pKF->mnId];
    };

    auto AddEdge = [&](const long unsigned int nIDi,
                       const long unsigned int nIDj, const g2o::SE3Quat &Tji) {
        EdgeEssentialGraphSE3 *e = new EdgeEssentialGraphSE3();
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                            optimizer.vertex(nIDj)));
        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                            optimizer.vertex(nIDi)));
        e->setMeasurement(Tji);
        e->information() = matLambda;
        optimizer.addEdge(e);
    };

    // Loop edges introduced by the current loop closure.
    for (map<KeyFramePtr, set<KeyFramePtr>>::const_iterator
             mit = LoopConnections.begin(),
             mend = LoopConnections.end();
         mit != mend; mit++) {
        KeyFramePtr pKF = mit->first;
        if (!pKF || pKF->isBad())
            continue;

        const long unsigned int nIDi = pKF->mnId;

        const set<KeyFramePtr> &spConnections = mit->second;
        const g2o::SE3Quat Tiw = vScw[nIDi];
        const g2o::SE3Quat Twi = Tiw.inverse();

        for (set<KeyFramePtr>::const_iterator sit = spConnections.begin(),
                                              send = spConnections.end();
             sit != send; sit++) {
            KeyFramePtr pKFn = *sit;
            if (!pKFn || pKFn->isBad())
                continue;

            const long unsigned int nIDj = pKFn->mnId;

            if ((nIDi != pCurKF->mnId || nIDj != pLoopKF->mnId) &&
                pKF->GetWeight(pKFn) < minFeat)
                continue;

            const g2o::SE3Quat Tjw = vScw[nIDj];
            const g2o::SE3Quat Tji = Tjw * Twi;

            AddEdge(nIDi, nIDj, Tji);
            sInsertedEdges.insert(make_pair(min(nIDi, nIDj), max(nIDi, nIDj)));
        }
    }

    // Spanning tree, previous loop edges, and covisibility graph edges.
    for (KeyFramePtr pKF : vpKFs) {
        if (!pKF || pKF->isBad())
            continue;

        const long unsigned int nIDi = pKF->mnId;

        const g2o::SE3Quat Twi = GetNonCorrectedScw(pKF).inverse();

        KeyFramePtr pParentKF = pKF->GetParent();

        if (pParentKF && !pParentKF->isBad()) {
            const long unsigned int nIDj = pParentKF->mnId;
            const g2o::SE3Quat Tjw = GetNonCorrectedScw(pParentKF);
            AddEdge(nIDi, nIDj, Tjw * Twi);
        }

        const set<KeyFramePtr> sLoopEdges = pKF->GetLoopEdges();
        for (set<KeyFramePtr>::const_iterator sit = sLoopEdges.begin(),
                                              send = sLoopEdges.end();
             sit != send; sit++) {
            KeyFramePtr pLKF = *sit;
            if (!pLKF || pLKF->isBad())
                continue;

            const long unsigned int nIDj = pLKF->mnId;
            if (nIDj < nIDi) {
                const g2o::SE3Quat Tlw = GetNonCorrectedScw(pLKF);
                AddEdge(nIDi, nIDj, Tlw * Twi);
            }
        }

        const vector<KeyFramePtr> vpConnectedKFs =
            pKF->GetCovisiblesByWeight(minFeat);
        for (vector<KeyFramePtr>::const_iterator vit = vpConnectedKFs.begin();
             vit != vpConnectedKFs.end(); vit++) {
            KeyFramePtr pKFn = *vit;
            if (!pKFn || pKFn->isBad() || pKFn == pParentKF ||
                pKF->hasChild(pKFn))
                continue;

            const long unsigned int nIDj = pKFn->mnId;
            if (nIDj >= nIDi)
                continue;

            if (sInsertedEdges.count(
                    make_pair(min(nIDi, nIDj), max(nIDi, nIDj))))
                continue;

            const g2o::SE3Quat Tnw = GetNonCorrectedScw(pKFn);
            AddEdge(nIDi, nIDj, Tnw * Twi);
        }
    }

    optimizer.initializeOptimization();
    optimizer.optimize(20);

    cout << "vertices and edges added: " << optimizer.vertices().size() << " , "
         << optimizer.edges().size() << endl;

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);
    cout << "Loop closed edge optimization done." << endl;

    for (KeyFramePtr pKF : vpKFs) {
        if (!pKF || pKF->isBad())
            continue;

        const long unsigned int nIDi = pKF->mnId;

        g2o::SE3Quat correctedTiw =
            static_cast<VertexPoseGraph *>(optimizer.vertex(nIDi))
                ->estimate().inverse();
        vCorrectedSwc[nIDi] = correctedTiw.inverse();

        Sophus::SE3d Tiw = ToSophusSE3(correctedTiw);
        pKF->SetPose(Tiw.cast<float>());
    }

    for (MapPointPtr pMP : vpMPs) {
        if (!pMP || pMP->isBad())
            continue;

        int nIDr = -1;
        if (pMP->mnCorrectedByKF == pCurKF->mnId) {
            nIDr = pMP->mnCorrectedReference;
        } else {
            KeyFramePtr pRefKF = pMP->GetReferenceKeyFrame();
            nIDr = pRefKF->mnId;
        }

        const g2o::SE3Quat Trw = vScw[nIDr];
        const g2o::SE3Quat correctedTwr = vCorrectedSwc[nIDr];

        Eigen::Vector3d p3Dw = pMP->GetWorldPos().cast<double>();
        Eigen::Vector3d correctedP3Dw = correctedTwr.map(Trw.map(p3Dw));
        pMP->SetWorldPos(correctedP3Dw.cast<float>());
    }

    for (MapLinePtr pML : vpMLs) {
        if (!pML || pML->isBad())
            continue;

        int nIDr = -1;
        if (pML->mnCorrectedByKF == pCurKF->mnId) {
            nIDr = pML->mnCorrectedReference;
        } else {
            KeyFramePtr pRefKF = pML->GetReferenceKeyFrame();
            nIDr = pRefKF->mnId;
        }

        const g2o::SE3Quat Trw = vScw[nIDr];
        const g2o::SE3Quat correctedTwr = vCorrectedSwc[nIDr];

        PluckerLine lineW = pML->GetWorldPos();
        PluckerLine lineR =
            TransformLineToBody(lineW,
                                Trw.rotation().toRotationMatrix().cast<float>(),
                                Trw.translation().cast<float>())
                .cast<float>();
        PluckerLine correctedLineW =
            TransformLineToBody(
                lineR, correctedTwr.rotation().toRotationMatrix().cast<float>(),
                correctedTwr.translation().cast<float>())
                .cast<float>();
        pML->SetWorldPos(correctedLineW);
    }

    pMap->IncreaseChangeIndex();
}

void Optimizer::GlobalBundleAdjustemnt(Map *pMap, int nIterations,
                                       bool *pbStopFlag,
                                       const unsigned long nLoopKF,
                                       const bool bRobust) {
    vector<KeyFramePtr> vpKFs = pMap->GetAllKeyFrames();
    vector<MapPointPtr> vpMP = pMap->GetAllMapPoints();
    vector<MapLinePtr> vpML = pMap->GetAllMapLines();
    BundleAdjustment(vpKFs, vpMP, vpML, nIterations, pbStopFlag, nLoopKF,
                     bRobust);
}

void Optimizer::BundleAdjustment(const std::vector<KeyFramePtr> &vpKFs,
                                 const std::vector<MapPointPtr> &vpMP,
                                 const std::vector<MapLinePtr> &vpML,
                                 int nIterations, bool *pbStopFlag,
                                 const unsigned long nLoopKF,
                                 const bool bRobust) {
    vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());

    vector<bool> vbNotIncludedML;
    vbNotIncludedML.resize(vpML.size());

    Map *pMap = vpKFs[0]->GetMap();

    g2o::SparseOptimizer optimizer;
    auto linearSolver = std::make_unique<
        g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
    auto solver_ptr =
        std::make_unique<g2o::BlockSolverX>(std::move(linearSolver));
    g2o::OptimizationAlgorithmLevenberg *solver =
        new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    const int nExpectedPointsSize = vpKFs.size() * vpMP.size();
    const int nExpectedLinesSize = vpKFs.size() * vpML.size();

    vector<EdgeProjectPoint *> vpEdgesPoint;
    vpEdgesPoint.reserve(nExpectedPointsSize);

    vector<EdgeProjectLine *> vpEdgesLine;
    vpEdgesLine.reserve(nExpectedLinesSize);

    vector<KeyFramePtr> vpEdgeKFPoint;
    vpEdgeKFPoint.reserve(nExpectedPointsSize);

    vector<KeyFramePtr> vpEdgeKFLine;
    vpEdgeKFLine.reserve(nExpectedLinesSize);

    vector<MapPointPtr> vpEdgeMP;
    vpEdgeMP.reserve(nExpectedPointsSize);

    vector<MapLinePtr> vpEdgeML;
    vpEdgeML.reserve(nExpectedLinesSize);

    for (size_t i = 0; i < vpKFs.size(); ++i) {
        KeyFramePtr pKFi = vpKFs[i];
        if (pKFi->isBad())
            continue;

        VertexPose *vPose = new VertexPose(pKFi);
        vPose->setId(pKFi->mnId);
        vPose->setFixed(pKFi->mnId == pMap->GetInitKFid());
        optimizer.addVertex(vPose);
        if (pKFi->mnId > maxKFid)
            maxKFid = pKFi->mnId;
    }

    const float thHuber = sqrt(5.991);

    unsigned long maxMPid = maxKFid + 1;

    const double sigma_point = 1.0;

    for (size_t i = 0; i < vpMP.size(); ++i) {
        MapPointPtr pMP = vpMP[i];

        g2o::VertexPointXYZ *vPoint = new g2o::VertexPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
        unsigned long id = pMP->mnId + maxKFid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        if (id > maxMPid) {
            maxMPid = id;
        }

        int nEdges = 0;

        const map<KeyFramePtr, vector<int>> observations =
            pMP->GetObservations();

        for (auto mit = observations.begin(), mend = observations.end();
             mit != mend; ++mit) {
            KeyFramePtr pKFi = mit->first;

            if (pKFi->isBad() || pKFi->mnId > maxKFid)
                continue;
            if (optimizer.vertex(id) == nullptr ||
                optimizer.vertex(pKFi->mnId) == nullptr)
                continue;

            for (size_t cam_id = 0; cam_id < mit->second.size(); ++cam_id) {
                int idx = mit->second[cam_id];
                if (idx == -1)
                    continue;

                const cv::Point2f &ptUn = pKFi->mvKeysUn_cams[cam_id][idx];
                Eigen::Matrix<double, 2, 1> obs;
                obs << ptUn.x, ptUn.y;

                EdgeProjectPoint *e = new EdgeProjectPoint(cam_id);

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                    optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                    optimizer.vertex(pKFi->mnId)));
                e->setMeasurement(obs);
                e->setInformation(Eigen::Matrix2d::Identity() *
                                  (1.0 / (sigma_point * sigma_point)));

                if (bRobust) {
                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber);
                }

                optimizer.addEdge(e);
                vpEdgesPoint.push_back(e);
                vpEdgeKFPoint.push_back(pKFi);
                vpEdgeMP.push_back(pMP);

                nEdges++;
            }
        }
        if (nEdges == 0) {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i] = true;
        } else {
            vbNotIncludedMP[i] = false;
        }
    }

    for (size_t i = 0; i < vpML.size(); ++i) {
        MapLinePtr pML = vpML[i];
        VertexLine *vLine = new VertexLine(pML);
        unsigned long id = pML->mnId + maxMPid + 1;
        vLine->setId(id);
        vLine->setMarginalized(true);
        optimizer.addVertex(vLine);

        int nEdges = 0;

        const map<KeyFramePtr, vector<int>> observations =
            pML->GetObservations();

        for (auto mit = observations.begin(), mend = observations.end();
             mit != mend; ++mit) {
            KeyFramePtr pKFi = mit->first;

            if (pKFi->isBad() || pKFi->mnId > maxKFid)
                continue;
            if (optimizer.vertex(id) == nullptr ||
                optimizer.vertex(pKFi->mnId) == nullptr)
                continue;

            for (size_t cam_id = 0; cam_id < mit->second.size(); ++cam_id) {
                int idx = mit->second[cam_id];
                if (idx == -1)
                    continue;

                const Eigen::Vector4f &lineUn =
                    pKFi->mvLineKeysUn_cams[cam_id][idx];

                EdgeProjectLine *e = new EdgeProjectLine(cam_id);

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                    optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                    optimizer.vertex(pKFi->mnId)));
                e->setMeasurement(lineUn.cast<double>());
                e->setInformation(Eigen::Matrix2d::Identity() *
                                  kLineInformationFallback);
                SetLineEndpointCovariance(e, *pKFi, cam_id, idx);

                if (bRobust) {
                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber);
                }

                optimizer.addEdge(e);
                vpEdgesLine.push_back(e);
                vpEdgeKFLine.push_back(pKFi);
                vpEdgeML.push_back(pML);

                nEdges++;
            }
        }

        if (nEdges == 0) {
            optimizer.removeVertex(vLine);
            vbNotIncludedML[i] = true;
        } else {
            vbNotIncludedML[i] = false;
        }
    }

    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    for (size_t i = 0; i < vpKFs.size(); ++i) {
        KeyFramePtr pKF = vpKFs[i];
        if (pKF->isBad())
            continue;

        VertexPose *vPose =
            static_cast<VertexPose *>(optimizer.vertex(pKF->mnId));
        BasePose pose = vPose->estimate();
        if (nLoopKF == pMap->GetOriginKF()->mnId) {
            const Sophus::SE3f Tbw = pose.ToTbw().cast<float>();
            pKF->SetPose(Tbw.cast<float>());
        } else {
            const Sophus::SE3f Tbw = pose.ToTbw().cast<float>();
            pKF->mTbwGBA = Tbw.cast<float>();
            pKF->mnBAGlobalForKF = nLoopKF;
        }
    }

    for (size_t i = 0; i < vpMP.size(); ++i) {
        if (vbNotIncludedMP[i])
            continue;

        MapPointPtr pMP = vpMP[i];
        if (pMP->isBad())
            continue;
        g2o::VertexPointXYZ *vPoint = static_cast<g2o::VertexPointXYZ *>(
            optimizer.vertex(pMP->mnId + maxKFid + 1));

        if (nLoopKF == pMap->GetOriginKF()->mnId) {
            pMP->SetWorldPos(vPoint->estimate().cast<float>());
        } else {
            pMP->mPosGBA = vPoint->estimate().cast<float>();
            pMP->mnBAGlobalForKF = nLoopKF;
        }
    }

    for (size_t i = 0; i < vpML.size(); ++i) {
        if (vbNotIncludedML[i])
            continue;

        MapLinePtr pML = vpML[i];
        if (pML->isBad())
            continue;
        VertexLine *vLine = static_cast<VertexLine *>(
            optimizer.vertex(pML->mnId + maxMPid + 1));

        if (nLoopKF == pMap->GetOriginKF()->mnId) {
            pML->SetWorldPos(vLine->estimate().lineW.cast<float>());
        } else {
            pML->mPosGBA = vLine->estimate().lineW.cast<float>();
            pML->mnBAGlobalForKF = nLoopKF;
        }
    }
}

}; // namespace XPLSLAM
