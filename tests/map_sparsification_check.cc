#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <iostream>
#include <memory>
#include <vector>

#define private public
#define protected public
#include "keyframe.h"
#include "map_sparsification.h"
#undef protected
#undef private

#include "map.h"
#include "map_point.h"

namespace {

using namespace XPLSLAM;

std::vector<std::vector<std::vector<size_t>>> MakeGrid() {
    return std::vector<std::vector<std::vector<size_t>>>(
        FRAME_GRID_ROWS_POINT,
        std::vector<std::vector<size_t>>(FRAME_GRID_COLS_POINT));
}

bool Check(bool condition, const char *message) {
    if (!condition)
        std::cerr << message << std::endl;
    return condition;
}

} // namespace

int main() {
    using namespace XPLSLAM;

    MapPoint::nCam = 2;
    Map map;
    auto keyframe = std::make_shared<KeyFrame>();
    keyframe->mnId = 1;
    keyframe->mnCam = 2;
    keyframe->mpMap = &map;
    keyframe->mpKeyFrameDB = nullptr;
    keyframe->mpVocabulary_point = nullptr;
    keyframe->mpVocabulary_line = nullptr;
    keyframe->mbBad = false;
    keyframe->mvnMinX = {0, 0};
    keyframe->mvnMinY = {0, 0};
    keyframe->mvnMaxX = {640, 640};
    keyframe->mvnMaxY = {480, 480};
    keyframe->mvfGridElementWidthInvPoint = {
        static_cast<float>(FRAME_GRID_COLS_POINT) / 640.0f,
        static_cast<float>(FRAME_GRID_COLS_POINT) / 640.0f};
    keyframe->mvfGridElementHeightInvPoint = {
        static_cast<float>(FRAME_GRID_ROWS_POINT) / 480.0f,
        static_cast<float>(FRAME_GRID_ROWS_POINT) / 480.0f};
    keyframe->mvKeysUn_cams = {
        {cv::Point2f(100.0f, 100.0f), cv::Point2f(105.0f, 105.0f)},
        {cv::Point2f(100.0f, 100.0f)}};
    keyframe->mvKeyScores_cams = {{1.0f, 0.5f}, {1.0f}};
    keyframe->Np = {2, 1};
    keyframe->mnTotalPoints = 3;
    keyframe->mvpMapPoints_cams = {
        std::vector<MapPointPtr>(2), std::vector<MapPointPtr>(1)};
    keyframe->mvpMapLines_cams.resize(2);
    keyframe->mvGrid_point = {MakeGrid(), MakeGrid()};

    cv::Mat cam0_descriptors(2, 64, CV_32F, cv::Scalar(1.0f));
    cv::Mat cam1_descriptors(1, 64, CV_32F, cv::Scalar(2.0f));
    keyframe->mDescriptors_point.push_back(cam0_descriptors);
    keyframe->mDescriptors_point.push_back(cam1_descriptors);
    keyframe->mvDescriptors_point = {
        keyframe->mDescriptors_point.rowRange(0, 2),
        keyframe->mDescriptors_point.rowRange(2, 3)};
    keyframe->mvPointDescriptorRowToCamRow = {{0, 0}, {0, 1}, {1, 0}};
    keyframe->mDescriptors_line =
        cv::Mat(2, 64, CV_32F, cv::Scalar(7.0f));
    const cv::Mat line_descriptors_before =
        keyframe->mDescriptors_line.clone();

    auto high_value = std::make_shared<MapPoint>(
        Eigen::Vector3f(0.0f, 0.0f, 3.0f), keyframe, &map);
    auto low_value = std::make_shared<MapPoint>(
        Eigen::Vector3f(1.0f, 0.0f, 3.0f), keyframe, &map);
    high_value->AddObservation(keyframe, 0, 0);
    high_value->AddObservation(keyframe, 0, 1);
    low_value->AddObservation(keyframe, 1, 0);
    keyframe->AddMapPoint(high_value, 0, 0);
    keyframe->AddMapPoint(low_value, 1, 0);
    keyframe->AddMapPoint(high_value, 0, 1);
    map.AddKeyFrame(keyframe);
    map.AddMapPoint(high_value);
    map.AddMapPoint(low_value);

    keyframe->mvGrid_point[0][12][8] = {0, 1};
    keyframe->mvGrid_point[1][12][8] = {0};

    MapSparsificationConfig config;
    config.enabled = true;
    config.min_points_per_keyframe = 1;
    config.lambda = 500.0f;
    config.grid_lambda = 10.0f;
    config.window_length = 30;
    config.non_local_keyframes = 1;
    config.mip_gap = 0.0;
    config.solver_threads = 1;
    MapSparsification sparsification(config, nullptr);

    const MapPointWeakPtr weak_low = low_value;
    const auto result = sparsification.Sparsify({keyframe});
    if (!Check(result.solved, "MILP did not produce a solution") ||
        !Check(result.candidates == 2,
               "multi-camera point was duplicated in the MILP") ||
        !Check(result.deleted == 1,
               "unexpected number of sparsified map points") ||
        !Check(low_value->isBad(), "lower-value point was not removed") ||
        !Check(!high_value->isBad(), "higher-value point was removed")) {
        return 1;
    }

    low_value.reset();
    if (!Check(weak_low.expired(),
               "sparsified MapPoint still has a strong owner")) {
        return 1;
    }

    auto stale_compaction = keyframe->PreparePointFeatureCompaction();
    keyframe->EraseMapPointMatch(high_value, 0);
    if (!Check(!keyframe->CommitPointFeatureCompaction(
                   std::move(stale_compaction)),
               "stale point feature compaction was committed")) {
        return 1;
    }
    keyframe->AddMapPoint(high_value, 0, 0);

    keyframe->CompactPointFeatures();
    if (!Check(keyframe->isSparsified(), "keyframe was not marked compacted") ||
        !Check(keyframe->Np == std::vector<int>({1, 1}),
               "per-camera point counts were not compacted") ||
        !Check(keyframe->mnTotalPoints == 2,
               "total point count was not compacted") ||
        !Check(keyframe->mDescriptors_point.rows == 2,
               "point descriptors were not compacted") ||
        !Check(high_value->GetIndexInKeyFrame(keyframe, 0) == 0 &&
                   high_value->GetIndexInKeyFrame(keyframe, 1) == 0,
               "multi-camera observation indices were not updated") ||
        !Check(cv::norm(keyframe->mDescriptors_line,
                        line_descriptors_before, cv::NORM_INF) == 0.0,
               "map line descriptors changed during point compaction")) {
        return 1;
    }

    return 0;
}
