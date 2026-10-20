#pragma once

#include "graphite_slam_types.h"
#include "line_observation_covariance.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace XPLSLAM {

class Frame;

namespace graphite_backend {

struct PoseOnlyPointInput {
    PinholeCamera camera;
    Eigen::Matrix3d Rcb = Eigen::Matrix3d::Identity();
    Eigen::Vector3d tcb = Eigen::Vector3d::Zero();
    Eigen::Vector3d point_w = Eigen::Vector3d::Zero();
    Eigen::Vector2d measurement = Eigen::Vector2d::Zero();
    double information = 1.0;
};

struct PoseOnlyLineInput {
    PinholeCamera camera;
    Eigen::Matrix3d Rcb = Eigen::Matrix3d::Identity();
    Eigen::Vector3d tcb = Eigen::Vector3d::Zero();
    Vector6d line_w = Vector6d::Zero();
    Eigen::Vector4d measurement = Eigen::Vector4d::Zero();
    Eigen::Matrix4d endpoint_covariance = Eigen::Matrix4d::Zero();
    bool has_endpoint_covariance = false;
    double information = kLineInformationFallback;
};

struct PoseOnlyProblem {
    PoseVertex initial_pose;
    std::vector<PoseOnlyPointInput> points;
    std::vector<PoseOnlyLineInput> lines;
    double chi2_threshold = 5.991;
    double huber_delta = 2.4476519360399265;
};

struct PoseOnlyResult {
    PoseVertex optimized_pose;
    std::vector<uint8_t> point_outliers;
    std::vector<uint8_t> line_outliers;
    int num_bad = 0;
};

struct FlatPoseVertex {
    std::array<double, 9> Rwb{};
    std::array<double, 3> twb{};
};

struct FlatPinholeCamera {
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    std::array<double, 9> toPL{};
};

struct FlatPoseOnlyPointInput {
    FlatPinholeCamera camera;
    std::array<double, 9> Rcb{};
    std::array<double, 3> tcb{};
    std::array<double, 3> point_w{};
    std::array<double, 2> measurement{};
    double information = 1.0;
};

struct FlatPoseOnlyLineInput {
    FlatPinholeCamera camera;
    std::array<double, 9> Rcb{};
    std::array<double, 3> tcb{};
    std::array<double, 6> line_w{};
    std::array<double, 4> measurement{};
    PackedLineEndpointCovariance endpoint_covariance_upper{};
    uint8_t has_endpoint_covariance = 0;
    double information = kLineInformationFallback;
};

struct FlatPoseOnlyProblem {
    FlatPoseVertex initial_pose;
    const FlatPoseOnlyPointInput *points = nullptr;
    size_t num_points = 0;
    const FlatPoseOnlyLineInput *lines = nullptr;
    size_t num_lines = 0;
    double chi2_threshold = 5.991;
    double huber_delta = 2.4476519360399265;
};

struct FlatPoseOnlyResult {
    FlatPoseVertex optimized_pose;
    uint8_t *point_outliers = nullptr;
    size_t num_point_outliers = 0;
    uint8_t *line_outliers = nullptr;
    size_t num_line_outliers = 0;
    int num_bad = 0;
};

// A relative transform is not an absolute Twb pose vertex.
struct FlatSE3Transform {
    std::array<double, 9> rotation{};
    std::array<double, 3> translation{};
};

struct FlatEssentialGraphEdgeInput {
    size_t vertex_i = 0;
    size_t vertex_j = 0;
    FlatSE3Transform measurement_Tji;
    std::array<double, 36> information{};
};

struct FlatEssentialGraphProblem {
    const FlatPoseVertex *initial_poses = nullptr;
    const uint8_t *fixed = nullptr;
    size_t num_poses = 0;
    const FlatEssentialGraphEdgeInput *edges = nullptr;
    size_t num_edges = 0;
    size_t iterations = 20;
    double initial_damping = 1e-16;
};

struct FlatEssentialGraphResult {
    FlatPoseVertex *optimized_poses = nullptr;
    size_t num_optimized_poses = 0;
    double initial_chi2 = 0.0;
    double final_chi2 = 0.0;
};

struct FlatLocalBAPointVertex {
    std::array<double, 3> position{};
};

struct FlatLocalBALineVertex {
    std::array<double, 6> line_w{};
};

struct FlatLocalBAPointObservation {
    size_t pose = 0;
    size_t point = 0;
    FlatPinholeCamera camera;
    std::array<double, 3> rcb{};
    std::array<double, 3> tcb{};
    std::array<double, 2> measurement{};
    double information = 1.0;
};

struct FlatLocalBALineObservation {
    size_t pose = 0;
    size_t line = 0;
    FlatPinholeCamera camera;
    std::array<double, 3> rcb{};
    std::array<double, 3> tcb{};
    std::array<double, 4> measurement{};
    PackedLineEndpointCovariance endpoint_covariance_upper{};
    uint8_t has_endpoint_covariance = 0;
    double information = kLineInformationFallback;
};

enum class FlatLocalBASchedule : uint8_t {
    kSingleStage = 0,
    kTwoStage = 1,
    kAutoMerge = 2,
};

struct FlatLocalBAProblem {
    const FlatPoseVertex *initial_poses = nullptr;
    const uint8_t *fixed_poses = nullptr;
    size_t num_poses = 0;
    const FlatLocalBAPointVertex *initial_points = nullptr;
    size_t num_points = 0;
    const FlatLocalBALineVertex *initial_lines = nullptr;
    size_t num_lines = 0;
    const FlatLocalBAPointObservation *point_observations = nullptr;
    size_t num_point_observations = 0;
    const FlatLocalBALineObservation *line_observations = nullptr;
    size_t num_line_observations = 0;
    FlatLocalBASchedule schedule = FlatLocalBASchedule::kSingleStage;
    size_t single_stage_iterations = 10;
    size_t first_stage_iterations = 5;
    size_t second_stage_iterations = 10;
    double initial_damping = 1e-4;
    double chi2_threshold = 5.991;
    double huber_delta = 2.4476519360399265;
    bool *stop_flag = nullptr;
};

struct FlatLocalBAResult {
    FlatPoseVertex *optimized_poses = nullptr;
    size_t num_optimized_poses = 0;
    FlatLocalBAPointVertex *optimized_points = nullptr;
    size_t num_optimized_points = 0;
    FlatLocalBALineVertex *optimized_lines = nullptr;
    size_t num_optimized_lines = 0;
    uint8_t *point_outliers = nullptr;
    size_t num_point_outliers = 0;
    uint8_t *line_outliers = nullptr;
    size_t num_line_outliers = 0;
    double initial_chi2 = 0.0;
    double final_chi2 = 0.0;
    bool used_two_stage = false;
};

bool OptimizePoseOnly(const PoseOnlyProblem &problem, PoseOnlyResult &result);
bool OptimizePoseOnlyFlat(const FlatPoseOnlyProblem &problem,
                          FlatPoseOnlyResult &result);
bool OptimizeEssentialGraphFlat(const FlatEssentialGraphProblem &problem,
                                FlatEssentialGraphResult &result);
bool OptimizeLocalBAFlat(const FlatLocalBAProblem &problem,
                         FlatLocalBAResult &result);

int PoseOptimization(Frame *frame);

} // namespace graphite_backend
} // namespace XPLSLAM
