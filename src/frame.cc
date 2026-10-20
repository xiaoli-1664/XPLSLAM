#include "runtime_timing.h"
#include "frame.h"
#include "camera_models/pinhole.h"
#include "converter.h"
#include "geometric_camera.h"
#include "geometric_tools.h"
#include "keyframe.h"
#include "map_line.h"
#include "map_point.h"
#include "matcher.h"
#include "sophus/se3.hpp"

#include <algorithm>
#include <c10/cuda/CUDAGuard.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>

using namespace std;

namespace XPLSLAM {

long unsigned int Frame::nNextId = 0;
bool Frame::mbInitialComputations = true;
vector<float> Frame::vcx, Frame::vcy, Frame::vfx, Frame::vfy, Frame::vinvfx,
    Frame::vinvfy;
vector<float> Frame::mvnMinX, Frame::mvnMinY, Frame::mvnMaxX, Frame::mvnMaxY,
    Frame::mvLineDMax;
vector<float> Frame::mvfGridElementWidthInvPoint,
    Frame::mvfGridElementHeightInvPoint, Frame::mvfGridElementHeightInvLine,
    Frame::mvfGridElementWidthInvLine;

namespace {

using FrameClock = chrono::steady_clock;

// Single choke point for every timing call site in this file: gating it
// makes XPLSLAM_ENABLE_PROFILING=OFF a zero-cost switch (no steady_clock
// query anywhere in Frame construction) without touching call sites.
inline FrameClock::time_point FrameNow() {
#ifdef XPLSLAM_ENABLE_PROFILING
    return FrameClock::now();
#else
    return FrameClock::time_point();
#endif
}

double FrameElapsedMilliseconds(const FrameClock::time_point &begin,
                                const FrameClock::time_point &end) {
    return chrono::duration<double, std::milli>(end - begin).count();
}

constexpr int kDescriptorDim = 64;
constexpr float kMinTriangulationDepth = 1e-2f;
constexpr float kMinLineDirectionNorm = 1e-2f;
constexpr float kStereoLoweRatio = 0.9f;
constexpr float kStereoEpipolarChi2 = 3.84f;
constexpr float kStereoReprojectionChi2 = 5.991f;
// Reject almost parallel rays without imposing the temporal triangulation
// cutoff on useful, distant stereo points (sin(angle) > 0.001).
constexpr float kMinStereoParallaxSinSquared = 1e-6f;

int ComputeFeatureVectorLevelsup(const DBoW3::Vocabulary *vocabulary) {
    return std::max(vocabulary ? vocabulary->getDepthLevels() - 2 : 0, 0);
}

torch::Tensor ToCpuFloatContiguous(const torch::Tensor &tensor) {
    if (!tensor.defined()) {
        return torch::zeros({0}, torch::TensorOptions().dtype(torch::kFloat32));
    }

    return tensor
        .to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32))
        .contiguous();
}

cv::Mat TensorToDescriptorMat(const torch::Tensor &tensor, const char *name) {
    if (!tensor.defined()) {
        return cv::Mat(0, kDescriptorDim, CV_32F);
    }

    if (tensor.dim() != 2 || tensor.size(1) != kDescriptorDim) {
        throw std::runtime_error(std::string(name) +
                                 " must have shape [N, 64]");
    }

    auto tensor_cpu = ToCpuFloatContiguous(tensor);
    const int rows = static_cast<int>(tensor_cpu.size(0));
    cv::Mat descriptors(rows, kDescriptorDim, CV_32F);
    if (rows > 0) {
        std::memcpy(descriptors.ptr<float>(), tensor_cpu.data_ptr<float>(),
                    static_cast<size_t>(rows) * kDescriptorDim * sizeof(float));
    }
    return descriptors;
}

float DescriptorDot(const cv::Mat &descriptors_a, int idx_a,
                    const cv::Mat &descriptors_b, int idx_b) {
    return descriptors_a.row(idx_a).dot(descriptors_b.row(idx_b));
}

float CosineDistanceSquared(float score) {
    score = std::clamp(score, -1.0f, 1.0f);
    return std::max(0.0f, 2.0f - 2.0f * score);
}

bool PassCosineLoweRatio(float best_score, float second_best_score,
                         float ratio) {
    if (ratio <= 0.0f) {
        return false;
    }

    if (second_best_score == std::numeric_limits<float>::lowest()) {
        return true;
    }

    const float best_dist2 = CosineDistanceSquared(best_score);
    const float second_dist2 = CosineDistanceSquared(second_best_score);
    if (second_dist2 <= std::numeric_limits<float>::epsilon()) {
        return false;
    }

    const float distance_ratio = std::min(ratio, 1.0f);
    return best_dist2 < distance_ratio * distance_ratio * second_dist2;
}

void BuildFlattenedDescriptors(
    const std::vector<cv::Mat> &camera_descriptors, cv::Mat &descriptors,
    std::vector<std::pair<int, int>> &row_to_cam_row) {
    std::vector<cv::Mat> non_empty_descriptors;
    row_to_cam_row.clear();

    size_t row_count = 0;
    for (const cv::Mat &camera_descriptor : camera_descriptors) {
        row_count += static_cast<size_t>(std::max(camera_descriptor.rows, 0));
    }
    row_to_cam_row.reserve(row_count);
    non_empty_descriptors.reserve(camera_descriptors.size());

    for (int cam = 0; cam < static_cast<int>(camera_descriptors.size());
         ++cam) {
        const cv::Mat &camera_descriptor = camera_descriptors[cam];
        if (camera_descriptor.empty() || camera_descriptor.rows <= 0) {
            continue;
        }

        non_empty_descriptors.push_back(camera_descriptor);
        for (int row = 0; row < camera_descriptor.rows; ++row) {
            row_to_cam_row.emplace_back(cam, row);
        }
    }

    if (non_empty_descriptors.empty()) {
        descriptors.release();
        return;
    }

    cv::vconcat(non_empty_descriptors, descriptors);
}

} // namespace

Frame::Frame()
    : mTbw(Sophus::SE3f()), mRwb(Eigen::Matrix3f::Identity()),
      mOw(Eigen::Vector3f::Zero()), mRbw(Eigen::Matrix3f::Identity()),
      mtbw(Eigen::Vector3f::Zero()), mbHasPose(false),
      mVw(Eigen::Vector3f::Zero()), mbHasVelocity(false), mpvoc_point(nullptr),
      mpvoc_line(nullptr), mnCam(0), mpxplnet(nullptr), mTimeStamp(0.0),
      mThDepth(0.0f), mnTotalPoints(0), mnTotalLines(0), mnCloseMPs(0),
      mnCloseMLs(0), mpLastKeyFrame(nullptr), mpPrevFrame(NULL), mnId(0),
      mpReferenceKF(nullptr), mnDataset(0), mbIsSet(false),
      mPointStereoMinDot(0.0f), mLineStereoMinDot(0.0f),
      mpStereoPointIndexInGrid(nullptr) {
#ifdef REGISTER_TIMES
    mTimeStereoMatch = 0;
    mTimeORB_Ext = 0;
#endif
}

Frame::Frame(const Frame &frame)
    : mTbw(frame.mTbw), mRwb(frame.mRwb), mOw(frame.mOw), mRbw(frame.mRbw),
      mtbw(frame.mtbw), mbHasPose(frame.mbHasPose), mvTbc(frame.mvTbc),
      mvTcb(frame.mvTcb), mvRbc(frame.mvRbc), mvtbc(frame.mvtbc),
      mVw(frame.mVw), mbHasVelocity(frame.mbHasVelocity),
      mpvoc_point(frame.mpvoc_point), mpvoc_line(frame.mpvoc_line),
      mpxplnet(frame.mpxplnet), mnCam(frame.mnCam),
      mTimeStamp(frame.mTimeStamp), mvK_(frame.mvK_), mThDepth(frame.mThDepth),
      Np(frame.Np), Nl(frame.Nl), mnTotalPoints(frame.mnTotalPoints),
      mnTotalLines(frame.mnTotalLines), mvKeys_cams(frame.mvKeys_cams),
      mvKeyScores_cams(frame.mvKeyScores_cams),
      mDescriptors_point(frame.mDescriptors_point.clone()),
      mvPointDescriptorRowToCamRow(frame.mvPointDescriptorRowToCamRow),
      mDescriptors_line(frame.mDescriptors_line.clone()),
      mvLineDescriptorRowToCamRow(frame.mvLineDescriptorRowToCamRow),
      mvLineKeys_cams(frame.mvLineKeys_cams),
      mvKeysUn_cams(frame.mvKeysUn_cams),
      mvLineKeysUn_cams(frame.mvLineKeysUn_cams),
      mvLineScores_cams(frame.mvLineScores_cams),
      mvLineEndpointCovariances_cams(frame.mvLineEndpointCovariances_cams),
      mvLinePolarUn_cams(frame.mvLinePolarUn_cams),
      mvpMapPoints_cams(frame.mvpMapPoints_cams),
      mvpMapLines_cams(frame.mvpMapLines_cams),
      mBowVec_point(frame.mBowVec_point), mFeatVec_point(frame.mFeatVec_point),
      mBowVec_line(frame.mBowVec_line), mFeatVec_line(frame.mFeatVec_line),
      mvbOutlierPoint_cams(frame.mvbOutlierPoint_cams),
      mvbOutlierLine_cams(frame.mvbOutlierLine_cams),
      mvbUseLineConstraint_cams(frame.mvbUseLineConstraint_cams),
      mnCloseMPs(frame.mnCloseMPs), mnCloseMLs(frame.mnCloseMLs),
      mvGrid_point(frame.mvGrid_point), mvGrid_line(frame.mvGrid_line),
      mpLastKeyFrame(frame.mpLastKeyFrame), mpPrevFrame(frame.mpPrevFrame),
      mnId(frame.mnId), mpReferenceKF(frame.mpReferenceKF),
      mNameFile(frame.mNameFile), mnDataset(frame.mnDataset),
      mbIsSet(frame.mbIsSet),
      mConstructionTiming(frame.mConstructionTiming),
      mLineFeatureState(frame.mLineFeatureState),
      mDeferredLineToken(0),
      mvpCamera(frame.mvpCamera),
      mvStereoPair(frame.mvStereoPair),
      mPointStereoMinDot(frame.mPointStereoMinDot),
      mLineStereoMinDot(frame.mLineStereoMinDot),
      mvDepthStereoPoint(frame.mvDepthStereoPoint),
      mvFirstToSecondPointMatch(frame.mvFirstToSecondPointMatch),
      mvStereo3DPoints(frame.mvStereo3DPoints),
      mpStereoPointIndexInGrid(frame.mpStereoPointIndexInGrid),
      mvDepthStereoLine(frame.mvDepthStereoLine),
      mvFirstToSecondLineMatch(frame.mvFirstToSecondLineMatch),
      mvStereo3DLines(frame.mvStereo3DLines) {
    mvK.resize(mnCam);
    mDistCoef.resize(mnCam);
    mvDescriptors_point.resize(mnCam);
    mvDescriptors_line.resize(mnCam);
    mvImgs.resize(frame.mvImgs.size());

    for (int cam = 0; cam < mnCam; ++cam) {
        mvK[cam] = frame.mvK[cam].clone();
        mDistCoef[cam] = frame.mDistCoef[cam].clone();
        mvDescriptors_point[cam] = frame.mvDescriptors_point[cam].clone();
        mvDescriptors_line[cam] = frame.mvDescriptors_line[cam].clone();
        if (cam < static_cast<int>(mvImgs.size()) &&
            cam < static_cast<int>(frame.mvImgs.size())) {
            mvImgs[cam] = frame.mvImgs[cam].clone();
        }
    }
}

Frame::Frame(const vector<cv::Mat> &imgs, const double &timeStamp,
             XPLNet *extractor, DBoW3::Vocabulary *voc_point,
             DBoW3::Vocabulary *voc_line, vector<cv::Mat> &vK,
             vector<cv::Mat> &vdistCoef, const vector<Sophus::SE3f> &vTbc,
             const vector<pair<int, int>> &vStereoPair,
             const Frame::StereoPointIndexGrid *pStereoPointIndexInGrid,
             const float &pointStereoMinDot, const float &lineStereoMinDot,
             const float &thDepth, vector<GeometricCamera *> vpCamera,
             Frame *pPrevF)
    : mpvoc_point(voc_point), mpvoc_line(voc_line), mpxplnet(extractor),
      mTimeStamp(timeStamp), mThDepth(thDepth), mnTotalPoints(0),
      mnTotalLines(0), mpPrevFrame(pPrevF), mpReferenceKF(NULL), mbIsSet(false),
      mvpCamera(vpCamera), mbHasPose(false), mbHasVelocity(false),
      mvStereoPair(vStereoPair), mPointStereoMinDot(pointStereoMinDot),
      mLineStereoMinDot(lineStereoMinDot),
      mpStereoPointIndexInGrid(pStereoPointIndexInGrid) {
    RuntimePhaseScope runtime_phase(RuntimePhase::BaseFeatures);
    mConstructionTiming = {};
    const auto construction_begin = FrameNow();
    auto stage_begin = construction_begin;
    mnId = nNextId++;
    mnCam = imgs.size();
    mvK.resize(mnCam);
    mvK_.resize(mnCam);
    mDistCoef.resize(mnCam);
    mvImgs.resize(mnCam);

    mvTbc.resize(mnCam);
    mvTcb.resize(mnCam);
    mvRbc.resize(mnCam);
    mvtbc.resize(mnCam);

    Np.assign(mnCam, 0);
    Nl.assign(mnCam, 0);
    mvGrid_point.resize(mnCam);
    mvGrid_line.resize(mnCam);
    mvKeys_cams.resize(mnCam);
    mvKeysUn_cams.resize(mnCam);
    mvKeyScores_cams.resize(mnCam);
    mvDescriptors_point.resize(mnCam);
    mvLineKeys_cams.resize(mnCam);
    mvLineKeysUn_cams.resize(mnCam);
    mvLineScores_cams.resize(mnCam);
    mvLineEndpointCovariances_cams.resize(mnCam);
    mvLinePolarUn_cams.resize(mnCam);
    mvDescriptors_line.resize(mnCam);
    mvpMapPoints_cams.resize(mnCam);
    mvpMapLines_cams.resize(mnCam);
    mvbOutlierPoint_cams.resize(mnCam);
    mvbOutlierLine_cams.resize(mnCam);

    mvDepthStereoPoint.resize(mvStereoPair.size());
    mvFirstToSecondPointMatch.resize(mvStereoPair.size());
    mvStereo3DPoints.resize(mvStereoPair.size());
    mvDepthStereoLine.resize(mvStereoPair.size());
    mvFirstToSecondLineMatch.resize(mvStereoPair.size());
    mvStereo3DLines.resize(mvStereoPair.size());

    if (mpStereoPointIndexInGrid == nullptr) {
        throw std::runtime_error("Stereo point grid candidate pointer is null");
    }
    if (mpStereoPointIndexInGrid->size() != mvStereoPair.size()) {
        throw std::runtime_error(
            "Stereo point grid candidate count does not match stereo pairs");
    }

    for (int cam = 0; cam < mnCam; ++cam) {
        mvK[cam] = vK[cam].clone();
        mvK_[cam] = Converter::toMatrix3f(vK[cam]);
        mDistCoef[cam] = vdistCoef[cam].clone();
        mvImgs[cam] = imgs[cam].clone();

        mvTbc[cam] = vTbc[cam];
        mvTcb[cam] = vTbc[cam].inverse();
        mvRbc[cam] = mvTbc[cam].rotationMatrix();
        mvtbc[cam] = mvTbc[cam].translation();
    }
    mConstructionTiming.setup_ms =
        FrameElapsedMilliseconds(stage_begin, FrameNow());

    ExtractFeatures(imgs);

    stage_begin = FrameNow();
    if (mbInitialComputations) {
        ComputeImageBounds(imgs);

        mvfGridElementHeightInvPoint.resize(mnCam);
        mvfGridElementWidthInvPoint.resize(mnCam);
        mvfGridElementHeightInvLine.resize(mnCam);
        mvfGridElementWidthInvLine.resize(mnCam);
        vfx.resize(mnCam);
        vfy.resize(mnCam);
        vcx.resize(mnCam);
        vcy.resize(mnCam);
        vinvfx.resize(mnCam);
        vinvfy.resize(mnCam);

        for (int cam = 0; cam < mnCam; ++cam) {
            mvfGridElementWidthInvPoint[cam] =
                static_cast<float>(FRAME_GRID_COLS_POINT) /
                (mvnMaxX[cam] - mvnMinX[cam]);
            mvfGridElementHeightInvPoint[cam] =
                static_cast<float>(FRAME_GRID_ROWS_POINT) /
                (mvnMaxY[cam] - mvnMinY[cam]);
            mvfGridElementWidthInvLine[cam] =
                static_cast<float>(FRAME_GRID_COLS_LINE) / GeometricTools::kPi;
            mvfGridElementHeightInvLine[cam] =
                mvLineDMax[cam] > 0.0f
                    ? static_cast<float>(FRAME_GRID_ROWS_LINE) /
                          (2.0f * mvLineDMax[cam])
                    : 0.0f;

            vfx[cam] = vK[cam].at<float>(0, 0);
            vfy[cam] = vK[cam].at<float>(1, 1);
            vcx[cam] = vK[cam].at<float>(0, 2);
            vcy[cam] = vK[cam].at<float>(1, 2);
            vinvfx[cam] = 1.0f / vfx[cam];
            vinvfy[cam] = 1.0f / vfy[cam];
        }

        mbInitialComputations = false;
    }
    mConstructionTiming.initial_computations_ms =
        FrameElapsedMilliseconds(stage_begin, FrameNow());

    stage_begin = FrameNow();
    BuildFlattenedDescriptors(mvDescriptors_point, mDescriptors_point,
                              mvPointDescriptorRowToCamRow);
    mConstructionTiming.descriptor_flatten_ms =
        FrameElapsedMilliseconds(stage_begin, FrameNow());

    stage_begin = FrameNow();
    AssignPointFeaturesToGrid();
    mConstructionTiming.grid_assignment_ms =
        FrameElapsedMilliseconds(stage_begin, FrameNow());

    stage_begin = FrameNow();
    ComputePointMulticameraMatches();
    mConstructionTiming.multicamera_matching_ms =
        FrameElapsedMilliseconds(stage_begin, FrameNow());
    mConstructionTiming.total_ms =
        FrameElapsedMilliseconds(construction_begin, FrameNow());
}

void Frame::ExtractFeatures(const vector<cv::Mat> &imgs) {
    if (mpxplnet == nullptr) {
        throw std::runtime_error("XPLNet extractor is null");
    }
    if (imgs.size() != static_cast<size_t>(mnCam)) {
        throw std::runtime_error("Input image count does not match frame");
    }

    c10::cuda::CUDAStreamGuard frontend_stream_guard(
        mpxplnet->GetFrontendStream());

    auto stage_begin = FrameNow();
    DeferredPointDetectionBatch deferred = mpxplnet->detectPoints(imgs);
    vector<DetectionResult> &results = deferred.detections;
    mDeferredLineToken = deferred.token;
    mLineFeatureState = LineFeatureState::None;
    mConstructionTiming.xplnet_detect_ms =
        FrameElapsedMilliseconds(stage_begin, FrameNow());
    mConstructionTiming.xplnet = mpxplnet->GetLastTiming();
    if (results.size() != static_cast<size_t>(mnCam)) {
        throw std::runtime_error("XPLNet result count does not match frame");
    }

    stage_begin = FrameNow();
    Np.assign(mnCam, 0);
    Nl.assign(mnCam, 0);
    mvLinePolarUn_cams.resize(mnCam);
    mnTotalPoints = 0;
    mnTotalLines = 0;
    mnCloseMPs = 0;
    mnCloseMLs = 0;

    for (int cam = 0; cam < mnCam; ++cam) {
        const DetectionResult &result = results[cam];

        auto keypoints_cpu = ToCpuFloatContiguous(result.keypoints);
        if (keypoints_cpu.dim() != 2 || keypoints_cpu.size(1) != 2) {
            throw std::runtime_error("Point keypoints must have shape [N, 2]");
        }
        auto keypoint_scores_cpu = ToCpuFloatContiguous(result.keypoint_scores);

        auto lines_cpu = ToCpuFloatContiguous(result.lines);
        if (lines_cpu.dim() != 3 || lines_cpu.size(1) != 2 ||
            lines_cpu.size(2) != 2) {
            throw std::runtime_error(
                "Line keypoints must have shape [M, 2, 2]");
        }
        auto line_scores_cpu = ToCpuFloatContiguous(result.line_scores);
        auto line_covariances_cpu =
            ToCpuFloatContiguous(result.line_endpoint_covariances);

        cv::Mat point_descriptors =
            TensorToDescriptorMat(result.descriptors, "Point descriptors");
        cv::Mat line_descriptors =
            TensorToDescriptorMat(result.line_descriptors, "Line descriptors");

        const int64_t num_points = keypoints_cpu.size(0);
        const int64_t num_lines = lines_cpu.size(0);
        if (point_descriptors.rows != num_points) {
            throw std::runtime_error(
                "Point descriptor count does not match keypoints");
        }
        if (keypoint_scores_cpu.dim() != 1 ||
            keypoint_scores_cpu.size(0) != num_points) {
            throw std::runtime_error(
                "Point score count does not match keypoints");
        }
        if (line_descriptors.rows != num_lines) {
            throw std::runtime_error(
                "Line descriptor count does not match keylines");
        }
        if (line_scores_cpu.dim() != 1 ||
            line_scores_cpu.size(0) != num_lines) {
            throw std::runtime_error(
                "Line score count does not match keylines");
        }
        if (line_covariances_cpu.dim() != 3 ||
            line_covariances_cpu.size(0) != num_lines ||
            line_covariances_cpu.size(1) != 4 ||
            line_covariances_cpu.size(2) != 4) {
            throw std::runtime_error(
                "Line endpoint covariances must have shape [M, 4, 4]");
        }

        auto &keypoints = mvKeys_cams[cam];
        auto &keypoint_scores = mvKeyScores_cams[cam];
        keypoints.clear();
        keypoints.reserve(static_cast<size_t>(num_points));
        keypoint_scores.clear();
        keypoint_scores.reserve(static_cast<size_t>(num_points));
        const float *keypoint_ptr =
            num_points > 0 ? keypoints_cpu.data_ptr<float>() : nullptr;
        const float *keypoint_score_ptr =
            num_points > 0 ? keypoint_scores_cpu.data_ptr<float>() : nullptr;
        for (int64_t i = 0; i < num_points; ++i) {
            const size_t base = static_cast<size_t>(i) * 2;
            keypoints.emplace_back(keypoint_ptr[base], keypoint_ptr[base + 1]);
            keypoint_scores.push_back(keypoint_score_ptr[i]);
        }
        mvKeysUn_cams[cam] = keypoints;
        mvDescriptors_point[cam] = std::move(point_descriptors);
        mvpMapPoints_cams[cam].assign(static_cast<size_t>(num_points), nullptr);
        mvbOutlierPoint_cams[cam].assign(static_cast<size_t>(num_points),
                                         false);

        auto &keylines = mvLineKeys_cams[cam];
        auto &line_scores = mvLineScores_cams[cam];
        auto &line_covariances = mvLineEndpointCovariances_cams[cam];
        auto &keyline_polars = mvLinePolarUn_cams[cam];
        keylines.clear();
        keylines.reserve(static_cast<size_t>(num_lines));
        line_scores.clear();
        line_scores.reserve(static_cast<size_t>(num_lines));
        line_covariances.clear();
        line_covariances.reserve(static_cast<size_t>(num_lines));
        keyline_polars.clear();
        keyline_polars.reserve(static_cast<size_t>(num_lines));
        const float *line_ptr =
            num_lines > 0 ? lines_cpu.data_ptr<float>() : nullptr;
        const float *line_score_ptr =
            num_lines > 0 ? line_scores_cpu.data_ptr<float>() : nullptr;
        const float *line_covariance_ptr =
            num_lines > 0 ? line_covariances_cpu.data_ptr<float>() : nullptr;
        const float min_x =
            cam < static_cast<int>(mvnMinX.size()) ? mvnMinX[cam] : 0.0f;
        const float min_y =
            cam < static_cast<int>(mvnMinY.size()) ? mvnMinY[cam] : 0.0f;
        for (int64_t i = 0; i < num_lines; ++i) {
            const size_t base = static_cast<size_t>(i) * 4;
            Eigen::Vector4f line;
            line << line_ptr[base], line_ptr[base + 1], line_ptr[base + 2],
                line_ptr[base + 3];
            Eigen::Vector2f line_polar;
            if (!GeometricTools::ImageLineToNormalizedPolar(line, line_polar)) {
                line_polar.setConstant(std::numeric_limits<float>::quiet_NaN());
            }
            keylines.emplace_back(line);
            line_scores.push_back(line_score_ptr[i]);
            Eigen::Matrix4f covariance;
            const size_t covariance_base = static_cast<size_t>(i) * 16;
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    covariance(row, col) =
                        line_covariance_ptr[covariance_base + row * 4 + col];
                }
            }
            line_covariances.emplace_back(covariance);
            keyline_polars.emplace_back(line_polar);
        }
        mvLineKeysUn_cams[cam] = keylines;
        mvDescriptors_line[cam] = std::move(line_descriptors);
        mvpMapLines_cams[cam].assign(static_cast<size_t>(num_lines), nullptr);
        mvbOutlierLine_cams[cam].assign(static_cast<size_t>(num_lines), false);
        mvbUseLineConstraint_cams.resize(mnCam);
        mvbUseLineConstraint_cams[cam].clear();

        Np[cam] = static_cast<int>(num_points);
        Nl[cam] = static_cast<int>(num_lines);
        mnTotalPoints += Np[cam];
        mnTotalLines += Nl[cam];
    }
    mConstructionTiming.feature_materialization_ms =
        FrameElapsedMilliseconds(stage_begin, FrameNow());
}

void Frame::MaterializeLines(const vector<DetectionResult> &results) {
    if (results.size() != static_cast<size_t>(mnCam)) {
        throw std::runtime_error("XPLNet line result count does not match frame");
    }
    Nl.assign(mnCam, 0);
    mnTotalLines = 0;
    mnCloseMLs = 0;
    mvLinePolarUn_cams.resize(mnCam);
    mvbUseLineConstraint_cams.resize(mnCam);

    for (int cam = 0; cam < mnCam; ++cam) {
        const DetectionResult &result = results[cam];
        auto lines_cpu = ToCpuFloatContiguous(result.lines);
        auto scores_cpu = ToCpuFloatContiguous(result.line_scores);
        auto covariances_cpu =
            ToCpuFloatContiguous(result.line_endpoint_covariances);
        if (lines_cpu.dim() != 3 || lines_cpu.size(1) != 2 ||
            lines_cpu.size(2) != 2) {
            throw std::runtime_error("Line keypoints must have shape [M, 2, 2]");
        }
        const int64_t count = lines_cpu.size(0);
        cv::Mat descriptors =
            TensorToDescriptorMat(result.line_descriptors, "Line descriptors");
        if (descriptors.rows != count || scores_cpu.dim() != 1 ||
            scores_cpu.size(0) != count || covariances_cpu.dim() != 3 ||
            covariances_cpu.size(0) != count ||
            covariances_cpu.size(1) != 4 || covariances_cpu.size(2) != 4) {
            throw std::runtime_error("Inconsistent deferred line result shapes");
        }

        auto &lines = mvLineKeys_cams[cam];
        auto &scores = mvLineScores_cams[cam];
        auto &covariances = mvLineEndpointCovariances_cams[cam];
        auto &polars = mvLinePolarUn_cams[cam];
        lines.clear();
        scores.clear();
        covariances.clear();
        polars.clear();
        lines.reserve(static_cast<size_t>(count));
        scores.reserve(static_cast<size_t>(count));
        covariances.reserve(static_cast<size_t>(count));
        polars.reserve(static_cast<size_t>(count));
        const float *line_ptr = count > 0 ? lines_cpu.data_ptr<float>() : nullptr;
        const float *score_ptr = count > 0 ? scores_cpu.data_ptr<float>() : nullptr;
        const float *covariance_ptr =
            count > 0 ? covariances_cpu.data_ptr<float>() : nullptr;
        for (int64_t i = 0; i < count; ++i) {
            const size_t base = static_cast<size_t>(i) * 4;
            Eigen::Vector4f line;
            line << line_ptr[base], line_ptr[base + 1], line_ptr[base + 2],
                line_ptr[base + 3];
            Eigen::Vector2f polar;
            if (!GeometricTools::ImageLineToNormalizedPolar(line, polar)) {
                polar.setConstant(std::numeric_limits<float>::quiet_NaN());
            }
            lines.push_back(line);
            scores.push_back(score_ptr[i]);
            Eigen::Matrix4f covariance;
            const size_t covariance_base = static_cast<size_t>(i) * 16;
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    covariance(row, col) =
                        covariance_ptr[covariance_base + row * 4 + col];
                }
            }
            covariances.push_back(covariance);
            polars.push_back(polar);
        }
        mvLineKeysUn_cams[cam] = lines;
        mvDescriptors_line[cam] = std::move(descriptors);
        mvpMapLines_cams[cam].assign(static_cast<size_t>(count), nullptr);
        mvbOutlierLine_cams[cam].assign(static_cast<size_t>(count), false);
        mvbUseLineConstraint_cams[cam].clear();
        Nl[cam] = static_cast<int>(count);
        mnTotalLines += Nl[cam];
    }
}

bool Frame::EnsureLineFeatures(LineFeatureState target) {
    if (static_cast<int>(mLineFeatureState) >= static_cast<int>(target)) {
        return true;
    }
    if (mLineFeatureState == LineFeatureState::None) {
        if (mpxplnet == nullptr || mDeferredLineToken == 0) {
            return false;
        }
        RuntimePhaseScope runtime_phase(RuntimePhase::SparseLines);
        if (runtime_frame) {
            if (runtime_frame->line_extraction_calls++ == 0)
                runtime_frame->first_line_reason = runtime_frame->reason;
        }
        c10::cuda::CUDAStreamGuard frontend_stream_guard(
            mpxplnet->GetFrontendStream());
        const auto begin = FrameNow();
        vector<DetectionResult> line_results =
            mpxplnet->completeLines(mDeferredLineToken);
        mDeferredLineToken = 0;
        MaterializeLines(line_results);
        if (runtime_frame) {
            runtime_frame->detected_lines += mnTotalLines;
            for (const auto &result : line_results)
                runtime_frame->network_before_dedup += result.line_candidates_before_dedup;
        }
        BuildFlattenedDescriptors(mvDescriptors_line, mDescriptors_line,
                                  mvLineDescriptorRowToCamRow);
        AssignLineFeaturesToGrid();
        mLineFeatureState = LineFeatureState::TrackingReady;
        mConstructionTiming.deferred_line_ms +=
            FrameElapsedMilliseconds(begin, FrameNow());
        mConstructionTiming.xplnet = mpxplnet->GetLastTiming();
    }
    if (target == LineFeatureState::StereoReady &&
        mLineFeatureState == LineFeatureState::TrackingReady) {
        const auto begin = FrameNow();
        ComputeLineMulticameraMatches();
        mConstructionTiming.deferred_line_stereo_ms +=
            FrameElapsedMilliseconds(begin, FrameNow());
        mLineFeatureState = LineFeatureState::StereoReady;
    }
    return static_cast<int>(mLineFeatureState) >= static_cast<int>(target);
}

void Frame::cvMatToGPUTorch(const vector<cv::Mat> &imgs,
                            torch::Tensor &tensor) {
    if (imgs.empty()) {
        throw std::runtime_error("Input image batch is empty");
    }

    int B = imgs.size();
    int C = imgs[0].channels();
    int H = imgs[0].rows;
    int W = imgs[0].cols;

    auto cpu_tensor =
        torch::empty({B, C, H, W}, torch::TensorOptions().dtype(torch::kUInt8));

    const size_t channel_bytes = static_cast<size_t>(H) * W;
    const size_t image_bytes = static_cast<size_t>(C) * channel_bytes;

    for (size_t i = 0; i < B; ++i) {
        const cv::Mat &img = imgs[i];
        if (img.empty()) {
            throw std::runtime_error("Input image is empty");
        }
        if (img.type() != CV_8UC(C)) {
            throw std::runtime_error("Input image must be of type CV_8UC" +
                                     std::to_string(C));
        }
        if (img.rows != H || img.cols != W) {
            throw std::runtime_error("Input image size mismatch");
        }

        uint8_t *dst = cpu_tensor.data_ptr<uint8_t>() + i * image_bytes;

        if (C == 1) {
            if (img.isContinuous()) {
                std::memcpy(dst, img.data, channel_bytes);
            } else {
                for (int r = 0; r < H; ++r) {
                    const uint8_t *src_row = img.ptr<uint8_t>(r);
                    std::memcpy(dst + static_cast<size_t>(r) * W, src_row, W);
                }
            }
            continue;
        }

        for (int c = 0; c < C; ++c) {
            uint8_t *dst_channel = dst + static_cast<size_t>(c) * channel_bytes;
            for (int r = 0; r < H; ++r) {
                const uint8_t *src_row = img.ptr<uint8_t>(r);
                for (int col = 0; col < W; ++col) {
                    dst_channel[static_cast<size_t>(r) * W + col] =
                        src_row[col * C + c];
                }
            }
        }
    }

    if (!torch::cuda::is_available()) {
        throw std::runtime_error(
            "CUDA is not available, cannot move tensor to GPU");
    }

    tensor = cpu_tensor.to(torch::kCUDA);
}

void Frame::SetPose(const Sophus::SE3<float> &Tbw) {
    mTbw = Tbw;

    UpdatePoseMatrices();
    mbIsSet = true;
    mbHasPose = true;
}

void Frame::UpdatePoseMatrices() {
    Sophus::SE3<float> Twb = mTbw.inverse();
    mRwb = Twb.rotationMatrix();
    mOw = Twb.translation();
    mRbw = mTbw.rotationMatrix();
    mtbw = mTbw.translation();
}

void Frame::SetVelocity(Eigen::Vector3f Vwb) {
    mVw = Vwb;
    mbHasVelocity = true;
}

Eigen::Vector3f Frame::GetVelocity() const { return mVw; }

void Frame::ComputeImageBounds(const vector<cv::Mat> &imgs) {
    mvnMinX.resize(mnCam);
    mvnMaxX.resize(mnCam);
    mvnMinY.resize(mnCam);
    mvnMaxY.resize(mnCam);
    mvLineDMax.resize(mnCam);
    for (int cam = 0; cam < mnCam; ++cam) {
        mvnMinX[cam] = 0.0f;
        mvnMaxX[cam] = static_cast<float>(imgs[cam].cols);
        mvnMinY[cam] = 0.0f;
        mvnMaxY[cam] = static_cast<float>(imgs[cam].rows);
        mvLineDMax[cam] = std::hypot(mvnMaxX[cam] - mvnMinX[cam],
                                     mvnMaxY[cam] - mvnMinY[cam]);
    }
}

Sophus::SE3f Frame::GetCameraPose(int cam_id) const {
    if (cam_id < 0 || cam_id >= static_cast<int>(mvTcb.size())) {
        throw std::out_of_range("Camera id out of range");
    }

    return mvTcb[cam_id] * mTbw;
}

vector<Sophus::SE3f> Frame::GetCameraExtrinsics() const { return mvTbc; }

Sophus::SE3f Frame::GetCameraExtrinsic(int cam_id) const {
    return mvTbc[cam_id];
}

bool Frame::PointPosInGrid(const cv::Point2f &kp, int &posX, int &posY) {
    if (mnCam <= 0) {
        return false;
    }

    return GeometricTools::ImagePointToGridCell(
        kp, mvnMinX[0], mvnMinY[0], mvfGridElementWidthInvPoint[0],
        mvfGridElementHeightInvPoint[0], FRAME_GRID_COLS_POINT,
        FRAME_GRID_ROWS_POINT, posX, posY);
}

bool Frame::LinePosInGrid(const float &d, const float &theta, int &posX,
                          int &posY) {
    if (mnCam <= 0) {
        return false;
    }

    if (mvLineDMax.empty() || mvfGridElementWidthInvLine.empty() ||
        mvfGridElementHeightInvLine.empty()) {
        return false;
    }

    return GeometricTools::LinePolarToGridCell(
        d, theta, mvLineDMax[0], mvfGridElementWidthInvLine[0],
        mvfGridElementHeightInvLine[0], FRAME_GRID_COLS_LINE,
        FRAME_GRID_ROWS_LINE, posX, posY);
}

bool Frame::isSet() const { return mbIsSet; }

void Frame::AssignFeaturesToGrid() {
    AssignPointFeaturesToGrid();
    AssignLineFeaturesToGrid();
}

void Frame::AssignPointFeaturesToGrid() {
    for (int cam = 0; cam < mnCam; ++cam) {
        mvGrid_point[cam].assign(
            FRAME_GRID_ROWS_POINT,
            std::vector<std::vector<size_t>>(FRAME_GRID_COLS_POINT));

        for (size_t idx = 0; idx < mvKeysUn_cams[cam].size(); ++idx) {
            int pos_x = 0;
            int pos_y = 0;
            if (!GeometricTools::ImagePointToGridCell(
                    mvKeysUn_cams[cam][idx], mvnMinX[cam], mvnMinY[cam],
                    mvfGridElementWidthInvPoint[cam],
                    mvfGridElementHeightInvPoint[cam], FRAME_GRID_COLS_POINT,
                    FRAME_GRID_ROWS_POINT, pos_x, pos_y)) {
                continue;
            }

            mvGrid_point[cam][pos_y][pos_x].push_back(idx);
        }

    }
}

void Frame::AssignLineFeaturesToGrid() {
    for (int cam = 0; cam < mnCam; ++cam) {
        mvGrid_line[cam].assign(
            FRAME_GRID_ROWS_LINE,
            std::vector<std::vector<size_t>>(FRAME_GRID_COLS_LINE));
        for (size_t idx = 0; idx < mvLinePolarUn_cams[cam].size(); ++idx) {
            int pos_x = 0;
            int pos_y = 0;
            const Eigen::Vector2f &line_polar = mvLinePolarUn_cams[cam][idx];
            if (!GeometricTools::NormalizedLinePolarToGridCell(
                    line_polar(0), line_polar(1), mvLineDMax[cam],
                    mvfGridElementWidthInvLine[cam],
                    mvfGridElementHeightInvLine[cam], FRAME_GRID_COLS_LINE,
                    FRAME_GRID_ROWS_LINE, pos_x, pos_y)) {
                continue;
            }

            mvGrid_line[cam][pos_y][pos_x].push_back(idx);
        }
    }
}

void Frame::ComputeMulticameraMatches() {
    ComputeMulticameraMatchesImpl(true, true);
}

void Frame::ComputePointMulticameraMatches() {
    ComputeMulticameraMatchesImpl(true, false);
}

void Frame::ComputeLineMulticameraMatches() {
    ComputeMulticameraMatchesImpl(false, true);
}

void Frame::ComputeMulticameraMatchesImpl(bool include_points,
                                          bool include_lines) {
    RuntimePhaseScope runtime_phase(RuntimePhase::Matching);
    for (size_t pair_idx = 0; pair_idx < mvStereoPair.size(); ++pair_idx) {
        const int first_cam = mvStereoPair[pair_idx].first;
        const int second_cam = mvStereoPair[pair_idx].second;

        if (first_cam < 0 || first_cam >= mnCam || second_cam < 0 ||
            second_cam >= mnCam) {
            continue;
        }
        if (mvpCamera[first_cam] == nullptr ||
            mvpCamera[second_cam] == nullptr) {
            continue;
        }

        const size_t num_first_points =
            static_cast<size_t>(std::max(Np[first_cam], 0));
        const size_t num_second_points =
            static_cast<size_t>(std::max(Np[second_cam], 0));
        const size_t num_first_lines =
            static_cast<size_t>(std::max(Nl[first_cam], 0));
        const size_t num_second_lines =
            static_cast<size_t>(std::max(Nl[second_cam], 0));

        if (include_points) {
            mvDepthStereoPoint[pair_idx].assign(num_first_points, -1.0f);
            mvFirstToSecondPointMatch[pair_idx].assign(num_first_points, -1);
            mvStereo3DPoints[pair_idx].assign(num_first_points,
                                              Eigen::Vector3f::Zero());
        }

        if (include_lines) {
            mvDepthStereoLine[pair_idx].assign(num_first_lines, -1.0f);
            mvFirstToSecondLineMatch[pair_idx].assign(num_first_lines, -1);
            mvStereo3DLines[pair_idx].assign(num_first_lines,
                                             Eigen::VectorXf::Zero(6));
        }

        const Sophus::SE3f Tc2c1 = mvTcb[second_cam] * mvTbc[first_cam];
        const Eigen::Matrix3f Rc2c1 = Tc2c1.rotationMatrix();
        const Eigen::Vector3f tc2c1 = Tc2c1.translation();

        if (include_points && num_first_points > 0 && num_second_points > 0 &&
            mpStereoPointIndexInGrid != nullptr &&
            pair_idx < mpStereoPointIndexInGrid->size()) {
            const auto &pair_grid = (*mpStereoPointIndexInGrid)[pair_idx];
            const bool has_valid_pair_grid =
                pair_grid.size() == FRAME_GRID_ROWS_POINT;

            Eigen::Matrix<float, 3, 4> Tc1w =
                Eigen::Matrix<float, 3, 4>::Zero();
            Tc1w.block<3, 3>(0, 0).setIdentity();
            Eigen::Matrix<float, 3, 4> Tc2w =
                GeometricTools::MakeProjectionMatrix(Tc2c1);
            const Eigen::Matrix3f F21 =
                mvpCamera[second_cam]->toK_().inverse().transpose() *
                Sophus::SO3f::hat(tc2c1) * Rc2c1 *
                mvpCamera[first_cam]->toK_().inverse();

            std::vector<int> proposed_matches(num_first_points, -1);
            std::vector<float> proposed_scores(
                num_first_points, std::numeric_limits<float>::lowest());
            std::vector<float> best_score_for_second(
                num_second_points, std::numeric_limits<float>::lowest());
            std::vector<int> best_first_for_second(num_second_points, -1);

            for (size_t i = 0; i < num_first_points; ++i) {
                int first_grid_x = 0;
                int first_grid_y = 0;
                if (!GeometricTools::ImagePointToGridCell(
                        mvKeysUn_cams[first_cam][i], mvnMinX[first_cam],
                        mvnMinY[first_cam],
                        mvfGridElementWidthInvPoint[first_cam],
                        mvfGridElementHeightInvPoint[first_cam],
                        FRAME_GRID_COLS_POINT, FRAME_GRID_ROWS_POINT,
                        first_grid_x, first_grid_y)) {
                    continue;
                }

                if (!has_valid_pair_grid ||
                    first_grid_y >= static_cast<int>(pair_grid.size()) ||
                    first_grid_x >=
                        static_cast<int>(pair_grid[first_grid_y].size())) {
                    continue;
                }

                int best_second = -1;
                float best_score = std::numeric_limits<float>::lowest();
                float second_best_score = std::numeric_limits<float>::lowest();
                for (size_t flat_grid_idx :
                     pair_grid[first_grid_y][first_grid_x]) {
                    const int second_grid_y =
                        static_cast<int>(flat_grid_idx / FRAME_GRID_COLS_POINT);
                    const int second_grid_x =
                        static_cast<int>(flat_grid_idx % FRAME_GRID_COLS_POINT);
                    if (second_grid_y < 0 ||
                        second_grid_y >= FRAME_GRID_ROWS_POINT ||
                        second_grid_x < 0 ||
                        second_grid_x >= FRAME_GRID_COLS_POINT) {
                        continue;
                    }

                    for (size_t candidate_idx :
                         mvGrid_point[second_cam][second_grid_y]
                                     [second_grid_x]) {
                        const int second_idx = static_cast<int>(candidate_idx);
                        const float score = DescriptorDot(
                            mvDescriptors_point[first_cam], static_cast<int>(i),
                            mvDescriptors_point[second_cam], second_idx);
                        if (score > best_score) {
                            second_best_score = best_score;
                            best_score = score;
                            best_second = second_idx;
                        } else if (score > second_best_score) {
                            second_best_score = score;
                        }
                    }
                }

                if (best_second < 0 || best_score < mPointStereoMinDot ||
                    !PassCosineLoweRatio(best_score, second_best_score,
                                         kStereoLoweRatio)) {
                    continue;
                }

                proposed_matches[i] = best_second;
                proposed_scores[i] = best_score;
                if (best_score > best_score_for_second[best_second]) {
                    best_score_for_second[best_second] = best_score;
                    best_first_for_second[best_second] = static_cast<int>(i);
                }
            }

            for (size_t i = 0; i < num_first_points; ++i) {
                const int second_idx = proposed_matches[i];
                if (second_idx < 0 ||
                    best_first_for_second[second_idx] != static_cast<int>(i)) {
                    continue;
                }

                // Geometry is checked only for descriptor-selected matches:
                // constant work per match, not per descriptor candidate.
                const cv::Point2f &key_first = mvKeysUn_cams[first_cam][i];
                const cv::Point2f &key_second =
                    mvKeysUn_cams[second_cam][second_idx];
                const Eigen::Vector3f pixel_first(key_first.x, key_first.y, 1.0f);
                const Eigen::Vector3f pixel_second(key_second.x, key_second.y,
                                                   1.0f);
                const Eigen::Vector3f epiline_second = F21 * pixel_first;
                const Eigen::Vector3f epiline_first =
                    F21.transpose() * pixel_second;
                const float epipolar_error = pixel_second.dot(epiline_second);
                const float epipolar_error2 = epipolar_error * epipolar_error;
                const float normal_first2 = epiline_first.head<2>().squaredNorm();
                const float normal_second2 = epiline_second.head<2>().squaredNorm();
                if (!epiline_first.allFinite() || !epiline_second.allFinite() ||
                    !std::isfinite(epipolar_error2) ||
                    !(normal_first2 > 0.0f) || !(normal_second2 > 0.0f) ||
                    epipolar_error2 > kStereoEpipolarChi2 * normal_first2 ||
                    epipolar_error2 > kStereoEpipolarChi2 * normal_second2) {
                    continue;
                }

                Eigen::Vector3f ray_first = mvpCamera[first_cam]->unprojectEig(
                    key_first);
                Eigen::Vector3f ray_second =
                    mvpCamera[second_cam]->unprojectEig(key_second);
                const Eigen::Vector3f ray_second_in_first =
                    Rc2c1.transpose() * ray_second;
                const float ray_norm_product =
                    ray_first.squaredNorm() * ray_second_in_first.squaredNorm();
                const float ray_cross_norm2 =
                    ray_first.cross(ray_second_in_first).squaredNorm();
                if (!std::isfinite(ray_norm_product) ||
                    !std::isfinite(ray_cross_norm2) ||
                    !(ray_cross_norm2 >
                      kMinStereoParallaxSinSquared * ray_norm_product)) {
                    continue;
                }
                Eigen::Vector3f point_c1;
                if (!GeometricTools::Triangulate(ray_first, ray_second, Tc1w,
                                                 Tc2w, point_c1) ||
                    !point_c1.allFinite() ||
                    point_c1.z() <= kMinTriangulationDepth) {
                    continue;
                }

                const Eigen::Vector3f point_c2 = Rc2c1 * point_c1 + tc2c1;
                if (!point_c2.allFinite() ||
                    point_c2.z() <= kMinTriangulationDepth) {
                    continue;
                }

                const Eigen::Vector2f projection_first =
                    mvpCamera[first_cam]->project(point_c1);
                const Eigen::Vector2f projection_second =
                    mvpCamera[second_cam]->project(point_c2);
                if (!projection_first.allFinite() ||
                    !projection_second.allFinite() ||
                    (projection_first - pixel_first.head<2>()).squaredNorm() >
                        kStereoReprojectionChi2 ||
                    (projection_second - pixel_second.head<2>()).squaredNorm() >
                        kStereoReprojectionChi2) {
                    continue;
                }

                mvDepthStereoPoint[pair_idx][i] = point_c1.z();
                mvFirstToSecondPointMatch[pair_idx][i] = second_idx;
                mvStereo3DPoints[pair_idx][i] = point_c1;
            }
        }

        if (!include_lines || num_first_lines == 0 || num_second_lines == 0) {
            continue;
        }

        std::vector<int> best_second_for_first(num_first_lines, -1);
        std::vector<float> best_score_for_first(
            num_first_lines, std::numeric_limits<float>::lowest());
        std::vector<float> second_best_score_for_first(
            num_first_lines, std::numeric_limits<float>::lowest());
        for (size_t i = 0; i < num_first_lines; ++i) {
            for (size_t j = 0; j < num_second_lines; ++j) {
                const float score = DescriptorDot(
                    mvDescriptors_line[first_cam], static_cast<int>(i),
                    mvDescriptors_line[second_cam], static_cast<int>(j));
                if (score > best_score_for_first[i]) {
                    second_best_score_for_first[i] = best_score_for_first[i];
                    best_score_for_first[i] = score;
                    best_second_for_first[i] = static_cast<int>(j);
                } else if (score > second_best_score_for_first[i]) {
                    second_best_score_for_first[i] = score;
                }
            }
        }

        std::vector<int> best_first_for_second(num_second_lines, -1);
        std::vector<float> best_score_for_second(
            num_second_lines, std::numeric_limits<float>::lowest());
        for (size_t j = 0; j < num_second_lines; ++j) {
            for (size_t i = 0; i < num_first_lines; ++i) {
                const float score = DescriptorDot(
                    mvDescriptors_line[first_cam], static_cast<int>(i),
                    mvDescriptors_line[second_cam], static_cast<int>(j));
                if (score > best_score_for_second[j]) {
                    best_score_for_second[j] = score;
                    best_first_for_second[j] = static_cast<int>(i);
                }
            }
        }

        const Eigen::Matrix4f Tc2c1_matrix = Tc2c1.matrix();
        for (size_t i = 0; i < num_first_lines; ++i) {
            const int second_idx = best_second_for_first[i];
            if (second_idx < 0 ||
                best_first_for_second[second_idx] != static_cast<int>(i) ||
                best_score_for_first[i] < mLineStereoMinDot ||
                !PassCosineLoweRatio(best_score_for_first[i],
                                     second_best_score_for_first[i],
                                     kStereoLoweRatio)) {
                continue;
            }

            Eigen::Vector4f plane_first;
            Eigen::Vector4f plane_second_c2;
            if (!GeometricTools::BuildObservationPlane(
                    mvLineKeysUn_cams[first_cam][i], mvpCamera[first_cam],
                    plane_first) ||
                !GeometricTools::BuildObservationPlane(
                    mvLineKeysUn_cams[second_cam][second_idx],
                    mvpCamera[second_cam], plane_second_c2)) {
                continue;
            }

            const Eigen::Vector4f plane_second_c1 =
                Tc2c1_matrix.transpose() * plane_second_c2;
            const Eigen::Vector3f normal_first = plane_first.head<3>();
            const Eigen::Vector3f normal_second = plane_second_c1.head<3>();
            const float distance_first = plane_first(3);
            const float distance_second = plane_second_c1(3);

            Eigen::Vector3f direction = normal_first.cross(normal_second);
            const float direction_norm = direction.norm();
            if (direction_norm < kMinLineDirectionNorm ||
                !direction.allFinite()) {
                continue;
            }

            Eigen::Vector3f moment =
                distance_first * normal_second - distance_second * normal_first;
            if (!moment.allFinite() || moment.norm() < kMinLineDirectionNorm) {
                continue;
            }

            const float depth = moment.norm() / direction_norm;
            if (!std::isfinite(depth)) {
                continue;
            }

            direction /= direction_norm;
            moment /= direction_norm;

            Eigen::Matrix<float, 6, 1> plucker_line;
            plucker_line << moment, direction;
            if (!plucker_line.allFinite()) {
                continue;
            }

            const Eigen::Vector4f &line_first = mvLineKeysUn_cams[first_cam][i];
            const Eigen::Vector4f &line_second =
                mvLineKeysUn_cams[second_cam][second_idx];
            const Sophus::SE3f first_pose;
            if (!GeometricTools::IsLineDepthPositive(
                    plucker_line, mvpCamera[first_cam], first_pose, line_first) ||
                !GeometricTools::IsLineDepthPositive(
                    plucker_line, mvpCamera[second_cam], Tc2c1, line_second)) {
                continue;
            }
            Eigen::Vector2f error_first;
            Eigen::Vector2f error_second;
            if (!GeometricTools::ComputeLineReprojectionError(
                    plucker_line, mvpCamera[first_cam], first_pose, line_first,
                    error_first) ||
                !GeometricTools::ComputeLineReprojectionError(
                    plucker_line, mvpCamera[second_cam], Tc2c1, line_second,
                    error_second) ||
                error_first.squaredNorm() > kStereoReprojectionChi2 ||
                error_second.squaredNorm() > kStereoReprojectionChi2) {
                continue;
            }

            mvDepthStereoLine[pair_idx][i] = depth;
            mvFirstToSecondLineMatch[pair_idx][i] = second_idx;
            mvStereo3DLines[pair_idx][i] = plucker_line;
        }
    }
}

void Frame::ComputeBoW() {
    RuntimePhaseScope runtime_phase(RuntimePhase::Matching);
    if (mBowVec_point.empty()) {
        vector<cv::Mat> vCurrentDesc =
            Converter::toDescriptorVector(mDescriptors_point);
        const int levelsup_point = ComputeFeatureVectorLevelsup(mpvoc_point);
        mpvoc_point->transform(vCurrentDesc, mBowVec_point, mFeatVec_point,
                               levelsup_point);
    }
    ComputeLineBoW();
}

void Frame::ComputeLineBoW() {
    RuntimePhaseScope runtime_phase(RuntimePhase::Matching);
    if (HasLineFeatures() && mBowVec_line.empty()) {
        vector<cv::Mat> vCurrentDesc =
            Converter::toDescriptorVector(mDescriptors_line);
        const int levelsup_line = ComputeFeatureVectorLevelsup(mpvoc_line);
        mpvoc_line->transform(vCurrentDesc, mBowVec_line, mFeatVec_line,
                              levelsup_line);
    }
}

void Frame::UnprojectStereoPoint(const int &pair_id, const int &i,
                                 Eigen::Vector3f &x3D) {
    const int cam_first = mvStereoPair[pair_id].first;
    Eigen::Vector3f x3C1 = mvStereo3DPoints[pair_id][i];

    x3D = mRwb * (mvRbc[cam_first] * x3C1 + mvtbc[cam_first]) + mOw;
}

void Frame::UnprojectStereoLine(const int &pair_id, const int &i,
                                Eigen::Matrix<float, 6, 1> &line3D) {
    const int cam_first = mvStereoPair[pair_id].first;
    Eigen::VectorXf lineC1 = mvStereo3DLines[pair_id][i];

    Eigen::Matrix3f Rwc = mRwb * mvRbc[cam_first];
    Eigen::Vector3f twc = mRwb * mvtbc[cam_first] + mOw;

    line3D.head<3>() =
        Rwc * lineC1.head<3>() + twc.cross(Rwc * lineC1.tail<3>());
    line3D.tail<3>() = Rwc * lineC1.tail<3>();
}

vector<size_t> Frame::GetPointFeaturesInArea(const float &x, const float &y,
                                             const float &r, int cam_id) const {
    vector<size_t> indices;
    indices.reserve(Np[cam_id]);

    const int nMinCellX =
        max(0, (int)floor((x - mvnMinX[cam_id] - r) *
                          mvfGridElementWidthInvPoint[cam_id]));
    if (nMinCellX >= FRAME_GRID_COLS_POINT) {
        return indices;
    }

    const int nMaxCellX = min(static_cast<int>(FRAME_GRID_COLS_POINT) - 1,
                              (int)ceil((x - mvnMinX[cam_id] + r) *
                                        mvfGridElementWidthInvPoint[cam_id]));
    if (nMaxCellX < 0) {
        return indices;
    }

    const int nMinCellY =
        max(0, (int)floor((y - mvnMinY[cam_id] - r) *
                          mvfGridElementHeightInvPoint[cam_id]));
    if (nMinCellY >= FRAME_GRID_ROWS_POINT) {
        return indices;
    }

    const int nMaxCellY = min(static_cast<int>(FRAME_GRID_ROWS_POINT) - 1,
                              (int)ceil((y - mvnMinY[cam_id] + r) *
                                        mvfGridElementHeightInvPoint[cam_id]));
    if (nMaxCellY < 0) {
        return indices;
    }

    for (int ix = nMinCellX; ix <= nMaxCellX; ++ix) {
        for (int iy = nMinCellY; iy <= nMaxCellY; ++iy) {
            const vector<size_t> &cell_indices = mvGrid_point[cam_id][iy][ix];
            if (cell_indices.empty()) {
                continue;
            }

            for (size_t idx : cell_indices) {
                const cv::Point2f &kp = mvKeysUn_cams[cam_id][idx];

                const float dist_x = kp.x - x;
                const float dist_y = kp.y - y;

                if (fabs(dist_x) < r && fabs(dist_y) < r) {
                    indices.push_back(idx);
                }
            }
        }
    }

    return indices;
}

vector<size_t> Frame::GetLineFeaturesInArea(const float &d, const float &theta,
                                            const float &r_d,
                                            const float &r_theta,
                                            int cam_id) const {
    vector<size_t> indices;

    if (!HasLineFeatures() || cam_id < 0 || cam_id >= mnCam ||
        cam_id >= static_cast<int>(mvGrid_line.size()) ||
        mvGrid_line[cam_id].size() != FRAME_GRID_ROWS_LINE) {
        return indices;
    }

    indices.reserve(Nl[cam_id]);

    float query_d = d;
    float query_theta = theta;
    if (!GeometricTools::NormalizeLinePolar(query_d, query_theta)) {
        return indices;
    }

    auto appendLineCells = [&](float center_d, float min_theta,
                               float max_theta) {
        min_theta = std::max(0.0f, min_theta);
        max_theta = std::min(GeometricTools::kPi, max_theta);
        if (min_theta > max_theta) {
            return;
        }

        const int nMinCellX =
            max(0, static_cast<int>(
                       floor(min_theta * mvfGridElementWidthInvLine[cam_id])));
        if (nMinCellX >= FRAME_GRID_COLS_LINE) {
            return;
        }

        const int nMaxCellX =
            min(static_cast<int>(FRAME_GRID_COLS_LINE) - 1,
                static_cast<int>(
                    ceil(max_theta * mvfGridElementWidthInvLine[cam_id])));
        if (nMaxCellX < 0) {
            return;
        }

        const int nMinCellY = max(
            0, static_cast<int>(floor((center_d + mvLineDMax[cam_id] - r_d) *
                                      mvfGridElementHeightInvLine[cam_id])));
        if (nMinCellY >= FRAME_GRID_ROWS_LINE) {
            return;
        }

        const int nMaxCellY =
            min(static_cast<int>(FRAME_GRID_ROWS_LINE) - 1,
                static_cast<int>(ceil((center_d + mvLineDMax[cam_id] + r_d) *
                                      mvfGridElementHeightInvLine[cam_id])));
        if (nMaxCellY < 0) {
            return;
        }

        for (int ix = nMinCellX; ix <= nMaxCellX; ++ix) {
            for (int iy = nMinCellY; iy <= nMaxCellY; ++iy) {
                const vector<size_t> &cell_indices =
                    mvGrid_line[cam_id][iy][ix];
                if (cell_indices.empty()) {
                    continue;
                }

                for (size_t idx : cell_indices) {
                    if (idx >= mvLinePolarUn_cams[cam_id].size()) {
                        continue;
                    }
                    const Eigen::Vector2f &line_polar =
                        mvLinePolarUn_cams[cam_id][idx];
                    if (!line_polar.allFinite()) {
                        continue;
                    }

                    const float direct_theta =
                        fabs(line_polar(1) - query_theta);
                    const float direct_d = fabs(line_polar(0) - query_d);
                    const float flipped_theta =
                        GeometricTools::kPi - direct_theta;
                    const float flipped_d = fabs(line_polar(0) + query_d);
                    if ((direct_d < r_d && direct_theta < r_theta) ||
                        (flipped_d < r_d && flipped_theta < r_theta)) {
                        indices.push_back(idx);
                    }
                }
            }
        }
    };

    appendLineCells(query_d, query_theta - r_theta, query_theta + r_theta);
    if (query_theta - r_theta < 0.0f) {
        appendLineCells(-query_d, GeometricTools::kPi + query_theta - r_theta,
                        GeometricTools::kPi);
    }
    if (query_theta + r_theta >= GeometricTools::kPi) {
        appendLineCells(-query_d, 0.0f,
                        query_theta + r_theta - GeometricTools::kPi);
    }

    return indices;
}

bool Frame::isPointInFrustum(MapPointPtr pMP, float viewingCosLimit,
                             int cam_id) {
    (void)viewingCosLimit;
    pMP->mvbTrackInView[cam_id] = false;
    pMP->mvTrackProjX[cam_id] = -1;
    pMP->mvTrackProjY[cam_id] = -1;

    Eigen::Vector3f Pw = pMP->GetWorldPos();

    const Eigen::Vector3f Pc = mvTcb[cam_id] * mTbw * Pw;
    const float Pc_dist = Pc.norm();

    if (Pc.z() < 0)
        return false;

    const Eigen::Vector2f uv = mvpCamera[cam_id]->project(Pc);

    if (uv(0) < mvnMinX[cam_id] || uv(0) > mvnMaxX[cam_id] ||
        uv(1) < mvnMinY[cam_id] || uv(1) > mvnMaxY[cam_id]) {
        return false;
    }

    pMP->mvTrackProjX[cam_id] = uv(0);
    pMP->mvTrackProjY[cam_id] = uv(1);

    pMP->mvbTrackInView[cam_id] = true;
    pMP->mvTrackDepth[cam_id] = Pc_dist;

    return true;
}

bool Frame::isLineInFrustum(MapLinePtr pML, int cam_id) {
    pML->mvbTrackInView[cam_id] = false;
    pML->mvTrackProjD[cam_id] = -1;
    pML->mvTrackProjTheta[cam_id] = -1;

    Eigen::Matrix<float, 6, 1> lineW = pML->GetWorldPos();

    const Sophus::SE3f Tcw = mvTcb[cam_id] * mTbw;
    const Eigen::Matrix3f Rcw = Tcw.rotationMatrix();
    const Eigen::Vector3f tcw = Tcw.translation();

    Eigen::Matrix<float, 6, 1> lineC;
    lineC.head<3>() = Rcw * lineW.head<3>() + tcw.cross(Rcw * lineW.tail<3>());
    lineC.tail<3>() = Rcw * lineW.tail<3>();

    Eigen::Vector3f line2d = mvpCamera[cam_id]->projectLine(lineC);

    float d, theta;

    if (!GeometricTools::ImageLineToNormalizedPolar(line2d, d, theta)) {
        return false;
    }

    if (!IsLineInImage(line2d, cam_id)) {
        return false;
    }

    float depth = lineC.head<3>().norm() / lineC.tail<3>().norm();

    pML->mvbTrackInView[cam_id] = true;
    pML->mvTrackProjD[cam_id] = d;
    pML->mvTrackProjTheta[cam_id] = theta;
    pML->mvTrackDepth[cam_id] = depth;

    return true;
}

bool Frame::IsPointInImage(const float &x, const float &y, int cam_id) const {
    return (x >= mvnMinX[cam_id] && x < mvnMaxX[cam_id] &&
            y >= mvnMinY[cam_id] && y < mvnMaxY[cam_id]);
}

bool Frame::IsLineInImage(const Eigen::Vector3f &line, int cam_id) const {
    if (cam_id < 0 || cam_id >= mnCam || !line.allFinite()) {
        return false;
    }

    const float a = line(0);
    const float b = line(1);
    const float c = line(2);
    if (std::abs(a) < 1e-6f && std::abs(b) < 1e-6f) {
        return false;
    }

    const float min_x = static_cast<float>(mvnMinX[cam_id]);
    const float max_x = static_cast<float>(mvnMaxX[cam_id]);
    const float min_y = static_cast<float>(mvnMinY[cam_id]);
    const float max_y = static_cast<float>(mvnMaxY[cam_id]);

    const float v1 = a * min_x + b * min_y + c;
    const float v2 = a * max_x + b * min_y + c;
    const float v3 = a * max_x + b * max_y + c;
    const float v4 = a * min_x + b * max_y + c;

    const float min_value = std::min(std::min(v1, v2), std::min(v3, v4));
    const float max_value = std::max(std::max(v1, v2), std::max(v3, v4));
    return min_value <= 0.0f && max_value >= 0.0f;
}

}; // namespace XPLSLAM
