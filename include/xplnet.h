#pragma once

#include <Eigen/Core>
#include <Eigen/SVD>
#include <NvInfer.h>
#include <boost/functional/hash.hpp>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <opencv2/core/cuda.hpp>
#include <opencv2/line_descriptor.hpp>
#include <opencv2/opencv.hpp>
#include <torch/nn/options/vision.h>
#include <torch/torch.h>
#include <tuple>
#include <cstdint>
#include <vector>

#include "driver_types.h"
#include "tracking_timing.h"
#include "trt_engine.h"

namespace XPLSLAM {

struct DetectionResult {
    torch::Tensor keypoints;        // [N, 2] (x, y)
    torch::Tensor keypoint_scores;  // [N]
    torch::Tensor descriptors;      // [N, 64]
    torch::Tensor lines;            // [M, 2, 2] ((x1, y1), (x2, y2))
    torch::Tensor line_scores;      // [M]
    torch::Tensor line_descriptors; // [M, 64]
    torch::Tensor line_endpoint_covariances; // [M, 4, 4] (x1, y1, x2, y2)
    int64_t line_candidates_before_dedup = 0;
    int64_t line_candidates_after_dedup = 0;
};

struct LineMatch {
    int64_t query_idx = -1;
    int64_t train_idx = -1;
    float score = 0.0f;
};

struct XPLNetFeatureGrids {
    torch::Tensor point_bins;       // [N, 2] (point_x, point_y)
    torch::Tensor inter_point_bins; // [N, 2] (inter_x, inter_y)
    torch::Tensor line_bins;        // [M, 2] (rho_bin, theta_bin)
    torch::Tensor line_valid_mask;  // [M]
    int line_rho_offset = 0;
};

struct XPLNetFrontendResult {
    DetectionResult detection;
    XPLNetFeatureGrids grids;
};

struct DeferredPointDetectionBatch {
    std::vector<DetectionResult> detections;
    uint64_t token = 0;
};

class XPLNet {
  public:
    struct Config {
        std::string xplnet_onnx;
        std::string verify_onnx;
        std::string line_description_onnx;

        TrtBuildConfig xplnet_build_config;
        TrtBuildConfig verify_build_config;
        TrtBuildConfig line_description_build_config;

        int top_k_kp = 200;
        int top_k_junc = 200;
        float kp_th = 0.05f;
        float junc_th = 0.1f;
        float line_th = 0.5f;

        bool dedup_lines = true;
        float dedup_angle_deg = 5.0f;
        float dedup_midpoint_cell = 8.0f;
        float dedup_overlap_ratio = 0.7f;

        int max_verify_segments = 50000;
        int max_ldesc_lines = 500;

        int samples = 32;
        int verify_dim = 128;
        int ldesc_dim = 64;

        int resize_width = 800;
        int resize_height = 608;
    };

    struct GridConfig {
        int point_grid_size = 8;
        int inter_grid_size = 6;
        float delta_rho = 8.0f;
        float delta_theta = 0.1f;
    };
    explicit XPLNet(const Config &config);

    std::vector<DetectionResult> detect(const std::vector<cv::Mat> &imgs,
                                        cudaStream_t stream = nullptr);
    DeferredPointDetectionBatch
    detectPoints(const std::vector<cv::Mat> &imgs,
                 cudaStream_t stream = nullptr);
    std::vector<DetectionResult> completeLines(uint64_t token,
                                               cudaStream_t stream = nullptr);
    std::vector<DetectionResult> detect(const torch::Tensor &img,
                                        cudaStream_t stream = nullptr);
    std::vector<XPLNetFrontendResult>
    detectWithGrids(const torch::Tensor &img,
                    const std::vector<cv::Size> &sizes,
                    const GridConfig &grid_config, cudaStream_t stream);

    const XPLNetTimingBreakdown &GetLastTiming() const {
        return mLastTiming;
    }

    c10::cuda::CUDAStream GetFrontendStream() const {
        return frontend_stream_;
    }

  private:
    friend struct XPLNetTestAccess;

    void preprocess(const torch::Tensor &img);
    void preprocess(const std::vector<cv::Mat> &imgs,
                    std::vector<float> &scale_w,
                    std::vector<float> &scale_h);
    std::vector<DetectionResult>
    postprocessDetections(int64_t batch_size,
                          const std::vector<float> &scale_w,
                          const std::vector<float> &scale_h,
                          bool include_lines = true);
    torch::Tensor getKptsHeatmap(const torch::Tensor &kpts,
                                 float softmax_temp = 1.0f) const;
    torch::Tensor nms(const torch::Tensor &heatmap, float th, int kernel,
                      bool as_float) const;

    static std::pair<torch::Tensor, torch::Tensor>
    refineJunctionsAndCovariances(const torch::Tensor &junction_logits,
                                  const torch::Tensor &junctions,
                                  const torch::Tensor &junction_valid);

    static torch::Tensor
    restoreImageCoordinates(const torch::Tensor &coordinates,
                            const torch::Tensor &coordinate_scale);

    static torch::Tensor
    scaleCovariances(const torch::Tensor &covariances,
                     const torch::Tensor &coordinate_scale);

    static torch::Tensor assembleLineEndpointCovariances(
        const torch::Tensor &junction_covariances,
        const torch::Tensor &batch_indices,
        const torch::Tensor &start_indices,
        const torch::Tensor &end_indices);

    torch::Tensor interpolateSparse(
        const torch::Tensor &feat, const torch::Tensor &points, int full_h,
        int full_w,
        torch::nn::functional::GridSampleFuncOptions::mode_t mode) const;

    std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>,
               std::vector<torch::Tensor>, std::vector<torch::Tensor>,
               std::vector<int64_t>, std::vector<int64_t>>
    processLines(const torch::Tensor &desc_map, const torch::Tensor &edge_desc,
                 const torch::Tensor &junc_desc, const torch::Tensor &junctions,
                 const torch::Tensor &junction_covariances,
                 const torch::Tensor &junc_valid);

    torch::Tensor generateLineMap(const torch::Tensor &junc_valid) const;

    std::pair<torch::Tensor, torch::Tensor>
    sampleSegmentsMap(const torch::Tensor &junctions,
                      const torch::Tensor &line_map, int num_sample) const;

    torch::Tensor verificationPreprocess(const torch::Tensor &lines_sample,
                                         const torch::Tensor &edge_desc,
                                         const torch::Tensor &junc_desc) const;

    torch::Tensor deduplicateLinesFast(const torch::Tensor &lines,
                                       const torch::Tensor &scores,
                                       const torch::Tensor &batch_idx) const;
    XPLNetFeatureGrids buildGrids(const DetectionResult &result,
                                  const cv::Size &image_size,
                                  const GridConfig &grid_config) const;

    torch::Device device_;
    c10::cuda::CUDAStream frontend_stream_;

    Config config_;

    TrtEngine xplnet_engine_;
    TrtEngine verify_engine_;
    TrtEngine line_description_engine_;
    XPLNetTimingBreakdown mLastTiming;

    uint64_t next_deferred_token_ = 0;
    uint64_t deferred_token_ = 0;
    bool deferred_valid_ = false;
    int64_t deferred_batch_size_ = 0;
    std::vector<float> deferred_scale_ws_;
    std::vector<float> deferred_scale_hs_;
    torch::Tensor deferred_desc_map_;
    torch::Tensor deferred_edge_desc_;
    torch::Tensor deferred_junc_desc_;
    torch::Tensor deferred_junctions_;
    torch::Tensor deferred_junction_covariances_;
    torch::Tensor deferred_junction_valid_;

};

} // namespace XPLSLAM
