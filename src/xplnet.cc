#include "runtime_timing.h"
#include "xplnet.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/argsort.h>
#include <ATen/ops/searchsorted.h>
#include <algorithm>
#include <array>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Optional.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <torch/enum.h>
#include <torch/nn/functional.h>
#include <torch/nn/options/vision.h>
#include <unordered_map>

using torch::indexing::Slice;
namespace XPLSLAM {

using namespace std;

namespace F = torch::nn::functional;

namespace {

// Single choke point for every timing call site in this file: gating it
// makes XPLSLAM_ENABLE_PROFILING=OFF a zero-cost switch (no steady_clock
// query anywhere in the XPLNet forward path) without touching call sites.
inline chrono::steady_clock::time_point XPLNetNow() {
#ifdef XPLSLAM_ENABLE_PROFILING
    return chrono::steady_clock::now();
#else
    return chrono::steady_clock::time_point();
#endif
}

double ElapsedMilliseconds(const chrono::steady_clock::time_point &begin,
                           const chrono::steady_clock::time_point &end) {
    return chrono::duration<double, std::milli>(end - begin).count();
}

class ScopedWallTimer {
  public:
    explicit ScopedWallTimer(double &elapsed_ms)
        : elapsed_ms_(elapsed_ms), begin_(XPLNetNow()) {
        elapsed_ms_ = 0.0;
    }

    ~ScopedWallTimer() {
        elapsed_ms_ =
            ElapsedMilliseconds(begin_, XPLNetNow());
    }

  private:
    double &elapsed_ms_;
    chrono::steady_clock::time_point begin_;
};

int ClampGridIndex(int value, int size) {
    if (size <= 0) {
        return 0;
    }
    return std::max(0, std::min(value, size - 1));
}

struct ResizeAxis {
    std::vector<int> lo;
    std::vector<int> hi;
    std::vector<float> weight;
};

ResizeAxis BuildResizeAxis(int src_size, int dst_size) {
    ResizeAxis axis;
    axis.lo.resize(dst_size);
    axis.hi.resize(dst_size);
    axis.weight.resize(dst_size);

    const float scale = static_cast<float>(src_size) / dst_size;
    for (int i = 0; i < dst_size; ++i) {
        const float src = (static_cast<float>(i) + 0.5f) * scale - 0.5f;
        int lo = static_cast<int>(std::floor(src));
        float weight = src - static_cast<float>(lo);

        if (lo < 0) {
            lo = 0;
            weight = 0.0f;
        } else if (lo >= src_size - 1) {
            lo = src_size - 1;
            weight = 0.0f;
        }

        axis.lo[i] = lo;
        axis.hi[i] = std::min(lo + 1, src_size - 1);
        axis.weight[i] = weight;
    }
    return axis;
}

inline float Sum3U8(const uchar *p) {
    return static_cast<float>(p[0]) + static_cast<float>(p[1]) +
           static_cast<float>(p[2]);
}

void ResizeMeanToFloatTensor(const cv::Mat &img, int target_w, int target_h,
                             float *dst) {
    const ResizeAxis x_axis = BuildResizeAxis(img.cols, target_w);
    const ResizeAxis y_axis = BuildResizeAxis(img.rows, target_h);
    const int channels = img.channels();

    auto resize_rows = [&](const cv::Range &range) {
        for (int y = range.start; y < range.end; ++y) {
            const int y0 = y_axis.lo[y];
            const int y1 = y_axis.hi[y];
            const float wy = y_axis.weight[y];
            const float inv_wy = 1.0f - wy;
            const uchar *row0 = img.ptr<uchar>(y0);
            const uchar *row1 = img.ptr<uchar>(y1);
            float *dst_row = dst + static_cast<size_t>(y) * target_w;

            if (channels == 1) {
                constexpr float inv255 = 1.0f / 255.0f;
                for (int x = 0; x < target_w; ++x) {
                    const int x0 = x_axis.lo[x];
                    const int x1 = x_axis.hi[x];
                    const float wx = x_axis.weight[x];
                    const float inv_wx = 1.0f - wx;

                    const float top = (static_cast<float>(row0[x0]) * inv_wx +
                                       static_cast<float>(row0[x1]) * wx);
                    const float bottom =
                        (static_cast<float>(row1[x0]) * inv_wx +
                         static_cast<float>(row1[x1]) * wx);
                    dst_row[x] = (top * inv_wy + bottom * wy) * inv255;
                }
            } else {
                constexpr float inv3_255 = 1.0f / (3.0f * 255.0f);
                for (int x = 0; x < target_w; ++x) {
                    const int x0 = x_axis.lo[x];
                    const int x1 = x_axis.hi[x];
                    const float wx = x_axis.weight[x];
                    const float inv_wx = 1.0f - wx;

                    const float top =
                        Sum3U8(row0 + static_cast<size_t>(x0) * 3) * inv_wx +
                        Sum3U8(row0 + static_cast<size_t>(x1) * 3) * wx;
                    const float bottom =
                        Sum3U8(row1 + static_cast<size_t>(x0) * 3) * inv_wx +
                        Sum3U8(row1 + static_cast<size_t>(x1) * 3) * wx;
                    dst_row[x] = (top * inv_wy + bottom * wy) * inv3_255;
                }
            }
        }
    };

    cv::parallel_for_(cv::Range(0, target_h), resize_rows);
}

c10::cuda::CUDAStream
DetectStream(cudaStream_t stream,
             const c10::cuda::CUDAStream &frontend_stream) {
    if (stream != nullptr) {
        return c10::cuda::getStreamFromExternal(stream,
                                                c10::cuda::current_device());
    }
    return frontend_stream;
}

} // namespace

XPLNet::XPLNet(const Config &config)
    : device_(torch::kCUDA),
      frontend_stream_(c10::cuda::getStreamFromPool(true)), config_(config) {
    std::cout << "Building XPLNet engines..." << std::endl;
    xplnet_engine_.build(config.xplnet_onnx, config.xplnet_build_config);
    verify_engine_.build(config.verify_onnx, config.verify_build_config);
    line_description_engine_.build(config.line_description_onnx,
                                   config.line_description_build_config);
}

void XPLNet::preprocess(const torch::Tensor &img) {
    int B = img.size(0);
    int C = img.size(1);
    if (C != 1 && C != 3) {
        throw std::runtime_error(
            "XPLNet input tensor must have 1 or 3 channels");
    }

    torch::Tensor img_float = img.to(torch::kFloat32); // [B, 1, H, W]
    img_float.div_(255.0);

    if (C == 3) {
        img_float = img_float.mean(1, true);
        C = 1;
    }

    int target_h = config_.resize_height;
    int target_w = config_.resize_width;

    xplnet_engine_.set_input_shape("image", {B, C, target_h, target_w});
    torch::Tensor img_trt = xplnet_engine_.get_input_buffer("image");

    if (!img.is_contiguous() || !img_trt.is_contiguous()) {
        std::cerr << "Input image tensor must be contiguous and output buffer "
                     "must be contiguous"
                  << std::endl;
    }

    int num_elements = B * C * target_h * target_w;

    torch::Tensor flat_trt = img_trt.view({-1});

    torch::Tensor flat_view = flat_trt.slice(0, 0, num_elements);

    torch::Tensor dst_view = flat_view.view({B, C, target_h, target_w});

    if (!dst_view.is_contiguous()) {
        std::cerr << "Output view must be contiguous" << std::endl;
    }

    at::upsample_bilinear2d_out(dst_view, img_float, {target_h, target_w},
                                false, c10::nullopt, c10::nullopt);
}

void XPLNet::preprocess(const std::vector<cv::Mat> &imgs,
                        std::vector<float> &scale_ws,
                        std::vector<float> &scale_hs) {
    if (imgs.empty()) {
        throw std::runtime_error("XPLNet image batch is empty");
    }

    const int B = static_cast<int>(imgs.size());
    const int target_h = config_.resize_height;
    const int target_w = config_.resize_width;
    if (target_h <= 0 || target_w <= 0) {
        throw std::runtime_error("XPLNet resize dimensions must be positive");
    }

    scale_ws.resize(B);
    scale_hs.resize(B);

    torch::Tensor batch = torch::empty(
        {B, 1, target_h, target_w},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

    const size_t image_values = static_cast<size_t>(target_h) * target_w;

    for (int i = 0; i < B; ++i) {
        const cv::Mat &img = imgs[i];
        if (img.empty()) {
            throw std::runtime_error("XPLNet input image is empty");
        }
        if (img.depth() != CV_8U ||
            (img.channels() != 1 && img.channels() != 3)) {
            throw std::runtime_error(
                "XPLNet input image must be CV_8UC1 or CV_8UC3");
        }

        scale_ws[i] =
            static_cast<float>(img.cols) / static_cast<float>(target_w);
        scale_hs[i] =
            static_cast<float>(img.rows) / static_cast<float>(target_h);

        float *dst =
            batch.data_ptr<float>() + static_cast<size_t>(i) * image_values;
        ResizeMeanToFloatTensor(img, target_w, target_h, dst);
    }

    xplnet_engine_.copy_input("image", batch);
}

torch::Tensor XPLNet::getKptsHeatmap(const torch::Tensor &kpts,
                                     float softmax_temp) const {
    auto scores = torch::softmax(kpts * softmax_temp, 1).slice(1, 0, 64);
    auto B = scores.size(0);
    auto H = scores.size(2);
    auto W = scores.size(3);
    auto heatmap = scores.permute({0, 2, 3, 1}).reshape({B, H, W, 8, 8});
    heatmap = heatmap.permute({0, 1, 3, 2, 4}).reshape({B, 1, H * 8, W * 8});
    return heatmap;
}

torch::Tensor XPLNet::interpolateSparse(
    const torch::Tensor &feat, const torch::Tensor &points, int full_h,
    int full_w,
    torch::nn::functional::GridSampleFuncOptions::mode_t mode) const {
    auto x = points.index({Slice(), Slice(), 0})
                 .to(torch::kFloat32)
                 .div(static_cast<float>(full_w - 1))
                 .mul(2.0)
                 .sub(1.0)
                 .clamp(-1.0 + 1e-6, 1.0 - 1e-6);
    auto y = points.index({Slice(), Slice(), 1})
                 .to(torch::kFloat32)
                 .div(static_cast<float>(full_h - 1))
                 .mul(2.0)
                 .sub(1.0)
                 .clamp(-1.0 + 1e-6, 1.0 - 1e-6);

    auto grid = torch::stack({x, y}, -1).unsqueeze(2);
    auto sampled = F::grid_sample(
        feat, grid, F::GridSampleFuncOptions().mode(mode).align_corners(false));
    return sampled.squeeze(-1).transpose(1, 2);
}

torch::Tensor XPLNet::nms(const torch::Tensor &heatmap, float th, int kernel,
                          bool as_float) const {

    const auto b = heatmap.size(0);
    auto local = torch::max_pool2d(heatmap, {kernel, kernel}, {1, 1},
                                   {kernel / 2, kernel / 2});
    auto pos = (heatmap == local) & (heatmap > th);

    std::vector<torch::Tensor> batched;
    batched.reserve(b);
    int64_t pad_val = 0;
    for (int64_t i = 0; i < b; ++i) {
        auto nz = torch::nonzero(pos[i][0]);
        auto xy = torch::flip(nz, {1});
        if (as_float) {
            xy = xy.to(torch::kFloat32);
        }
        pad_val = std::max<int64_t>(pad_val, xy.size(0));
        batched.push_back(xy);
    }

    const auto dtype = as_float ? torch::kFloat32 : torch::kLong;
    auto out = torch::zeros(
        {b, pad_val, 2}, torch::TensorOptions().dtype(dtype).device(device_));

    for (int64_t i = 0; i < b; ++i) {
        if (batched[i].size(0) > 0) {
            out.index_put_({i, Slice(0, batched[i].size(0)), Slice()},
                           batched[i]);
        }
    }
    return out;
}

pair<torch::Tensor, torch::Tensor> XPLNet::refineJunctionsAndCovariances(
    const torch::Tensor &junction_logits, const torch::Tensor &junctions,
    const torch::Tensor &junction_valid) {
    constexpr int64_t kRadius = 3;
    constexpr float kTemperature = 1.0f;
    constexpr float kCovarianceFloor = 1e-4f;

    if (junction_logits.dim() != 4 || junction_logits.size(1) != 1) {
        throw invalid_argument(
            "junction_logits must have shape [B, 1, H, W]");
    }
    if (junctions.dim() != 3 || junctions.size(2) != 2) {
        throw invalid_argument("junctions must have shape [B, N, 2]");
    }
    if (junction_valid.dim() != 2 ||
        junction_valid.size(0) != junctions.size(0) ||
        junction_valid.size(1) != junctions.size(1)) {
        throw invalid_argument("junction_valid must have shape [B, N]");
    }
    if (junction_logits.size(0) != junctions.size(0)) {
        throw invalid_argument("junction batch size does not match logits");
    }

    const int64_t batch_size = junction_logits.size(0);
    const int64_t height = junction_logits.size(2);
    const int64_t width = junction_logits.size(3);
    if (height <= 0 || width <= 0) {
        throw invalid_argument("junction logits must have positive dimensions");
    }

    auto junctions_float = junctions.to(torch::kFloat32);
    auto logits_float = junction_logits.to(torch::kFloat32);
    auto offset_axis = torch::arange(
        -kRadius, kRadius + 1,
        junctions_float.options().dtype(torch::kFloat32));
    const int64_t window_size = 2 * kRadius + 1;
    auto offset_x = offset_axis.repeat({window_size});
    auto offset_y = offset_axis.repeat_interleave(window_size);
    auto offsets =
        torch::stack({offset_x, offset_y}, -1).view({1, 1, -1, 2});

    auto sample_coords = junctions_float.unsqueeze(2) + offsets;
    auto sample_x = sample_coords.index({Slice(), Slice(), Slice(), 0})
                        .round()
                        .to(torch::kLong);
    auto sample_y = sample_coords.index({Slice(), Slice(), Slice(), 1})
                        .round()
                        .to(torch::kLong);
    auto sample_valid = (sample_x >= 0) & (sample_x < width) &
                        (sample_y >= 0) & (sample_y < height);

    auto sample_x_clamped = sample_x.clamp(0, width - 1);
    auto sample_y_clamped = sample_y.clamp(0, height - 1);
    auto batch_indices =
        torch::arange(batch_size,
                      junctions_float.options().dtype(torch::kLong))
            .view({batch_size, 1, 1});
    auto logits_2d = logits_float.index({Slice(), 0});
    auto local_logits = logits_2d.index(
        {batch_indices, sample_y_clamped, sample_x_clamped});
    local_logits = local_logits.masked_fill(~sample_valid,
                                            -numeric_limits<float>::infinity());

    auto weights = torch::softmax(local_logits / kTemperature, -1);
    auto finite_weights = torch::isfinite(weights).all(-1, true);
    weights = torch::where(finite_weights, weights, torch::zeros_like(weights));

    auto refined = (weights.unsqueeze(-1) * sample_coords).sum(2);
    auto centered = sample_coords - refined.unsqueeze(2);
    auto covariances =
        (weights.unsqueeze(-1).unsqueeze(-1) * centered.unsqueeze(-1) *
         centered.unsqueeze(-2))
            .sum(2);
    covariances =
        covariances +
        kCovarianceFloor *
            torch::eye(2, covariances.options()).view({1, 1, 2, 2});

    auto valid_xy = junction_valid.to(torch::kBool).unsqueeze(-1);
    refined = torch::where(valid_xy, refined, torch::zeros_like(refined));
    covariances = torch::where(valid_xy.unsqueeze(-1), covariances,
                               torch::zeros_like(covariances));
    return {refined, covariances};
}

torch::Tensor
XPLNet::restoreImageCoordinates(const torch::Tensor &coordinates,
                                const torch::Tensor &coordinate_scale) {
    // NMS/refinement use integer pixel centers. Invert align_corners(false),
    // also used by BuildResizeAxis: (x + 0.5) * scale - 0.5.
    auto scale = coordinate_scale.to(coordinates.options());
    return coordinates * scale + (scale - 1.0) * 0.5;
}

torch::Tensor
XPLNet::scaleCovariances(const torch::Tensor &covariances,
                         const torch::Tensor &coordinate_scale) {
    if (covariances.dim() < 2 ||
        covariances.size(-2) != covariances.size(-1)) {
        throw invalid_argument("covariances must contain square matrices");
    }
    const int64_t dimension = covariances.size(-1);
    if (coordinate_scale.dim() != 1 ||
        coordinate_scale.numel() != dimension) {
        throw invalid_argument(
            "coordinate_scale must match the covariance dimension");
    }

    auto scale = coordinate_scale.to(covariances.options());
    vector<int64_t> row_shape(covariances.dim(), 1);
    vector<int64_t> col_shape(covariances.dim(), 1);
    row_shape[row_shape.size() - 2] = dimension;
    col_shape[col_shape.size() - 1] = dimension;
    return covariances * scale.view(row_shape) * scale.view(col_shape);
}

torch::Tensor XPLNet::assembleLineEndpointCovariances(
    const torch::Tensor &junction_covariances,
    const torch::Tensor &batch_indices, const torch::Tensor &start_indices,
    const torch::Tensor &end_indices) {
    if (junction_covariances.dim() != 4 ||
        junction_covariances.size(2) != 2 ||
        junction_covariances.size(3) != 2) {
        throw invalid_argument(
            "junction_covariances must have shape [B, N, 2, 2]");
    }
    if (batch_indices.dim() != 1 || start_indices.dim() != 1 ||
        end_indices.dim() != 1 ||
        batch_indices.numel() != start_indices.numel() ||
        batch_indices.numel() != end_indices.numel()) {
        throw invalid_argument("line endpoint indices must be aligned vectors");
    }

    // The network supplies one marginal covariance per junction. There is no
    // estimated cross-correlation between distinct endpoints, so each line
    // covariance is block diagonal in (x1, y1, x2, y2) order.
    auto line_covariances = torch::zeros(
        {batch_indices.numel(), 4, 4}, junction_covariances.options());
    line_covariances.index_put_(
        {Slice(), Slice(0, 2), Slice(0, 2)},
        junction_covariances.index({batch_indices, start_indices}));
    line_covariances.index_put_(
        {Slice(), Slice(2, 4), Slice(2, 4)},
        junction_covariances.index({batch_indices, end_indices}));
    return line_covariances;
}

torch::Tensor XPLNet::generateLineMap(const torch::Tensor &junc_valid) const {
    auto line_map =
        junc_valid.unsqueeze(2).logical_and(junc_valid.unsqueeze(1));
    return torch::triu(line_map, 1);
}

std::pair<torch::Tensor, torch::Tensor>
XPLNet::sampleSegmentsMap(const torch::Tensor &junctions,
                          const torch::Tensor &line_map, int num_sample) const {
    const auto b = junctions.size(0);
    const auto n = junctions.size(1);
    auto idx = torch::triu_indices(
        n, n, 1, torch::TensorOptions().dtype(torch::kLong).device(device_));
    auto i_idx = idx[0];
    auto j_idx = idx[1];
    const auto m = i_idx.numel();

    if (m == 0) {
        return {torch::zeros({b, num_sample, 0, 2}, junctions.options()),
                torch::zeros({b, 0}, line_map.options().dtype(torch::kBool))};
    }

    auto p0 = junctions.index({Slice(), i_idx, Slice()});
    auto p1 = junctions.index({Slice(), j_idx, Slice()});
    auto seg_valid = line_map.index({Slice(), i_idx, j_idx}).to(torch::kBool);
    auto t = torch::linspace(0.0, 1.0, num_sample,
                             junctions.options().dtype(torch::kFloat32))
                 .view({1, num_sample, 1, 1});
    auto sampled = p0.unsqueeze(1) * (1.0 - t) + p1.unsqueeze(1) * t;
    return {sampled, seg_valid};
}

torch::Tensor
XPLNet::verificationPreprocess(const torch::Tensor &lines_sample,
                               const torch::Tensor &edge_desc,
                               const torch::Tensor &junc_desc) const {
    auto x = lines_sample.index({Slice(), Slice(), Slice(), 0});
    auto y = lines_sample.index({Slice(), Slice(), Slice(), 1});

    const auto h = junc_desc.size(2);
    const auto w = junc_desc.size(3);
    auto grid = torch::stack({x.div(static_cast<float>(4 * w - 1))
                                  .mul(2.0)
                                  .sub(1.0)
                                  .clamp(-1.0 + 1e-6, 1.0 - 1e-6),
                              y.div(static_cast<float>(4 * h - 1))
                                  .mul(2.0)
                                  .sub(1.0)
                                  .clamp(-1.0 + 1e-6, 1.0 - 1e-6)},
                             -1);

    auto grid_junc =
        torch::stack({grid.index({Slice(), 0}),
                      grid.index({Slice(), lines_sample.size(1) - 1})},
                     1);
    auto grid_edge = grid.index({Slice(), Slice(1, lines_sample.size(1) - 1)});

    auto feat_junc = F::grid_sample(
        junc_desc, grid_junc,
        F::GridSampleFuncOptions().mode(torch::kBilinear).align_corners(false));
    auto feat_edge = F::grid_sample(
        edge_desc, grid_edge,
        F::GridSampleFuncOptions().mode(torch::kBilinear).align_corners(false));

    auto feat_start = feat_junc.index({Slice(), Slice(), 0});
    auto feat_end = feat_junc.index({Slice(), Slice(), 1});
    auto feat_edge_flat = feat_edge.flatten(1, 2);
    return torch::cat({feat_start, feat_edge_flat, feat_end}, 1);
}

torch::Tensor
XPLNet::deduplicateLinesFast(const torch::Tensor &lines,
                             const torch::Tensor &scores,
                             const torch::Tensor &batch_idx) const {
    const auto n = lines.size(0);
    auto opts = torch::TensorOptions().dtype(torch::kLong).device(device_);
    if (!config_.dedup_lines || n <= 1) {
        return torch::arange(n, opts);
    }

    auto lines_cpu = lines.to(torch::kFloat32).to(torch::kCPU).contiguous();
    auto scores_cpu = scores.to(torch::kFloat32).to(torch::kCPU).contiguous();
    auto batch_cpu = batch_idx.to(torch::kLong).to(torch::kCPU).contiguous();

    const float angle_step_deg = std::max(config_.dedup_angle_deg, 0.5f);
    const int angle_bins =
        std::max(1, static_cast<int>(std::ceil(180.0f / angle_step_deg)));
    const float angle_step_rad =
        static_cast<float>(M_PI) / static_cast<float>(angle_bins);
    const float angle_thresh_rad =
        config_.dedup_angle_deg * static_cast<float>(M_PI) / 180.0f;
    const float midpoint_cell = std::max(config_.dedup_midpoint_cell, 1.0f);
    const float normal_dist_thresh = 0.5f * midpoint_cell;
    const float overlap_thresh = config_.dedup_overlap_ratio;
    const float min_len = 1e-3f;

    auto line_acc = lines_cpu.accessor<float, 3>();
    auto batch_ptr = batch_cpu.data_ptr<int64_t>();

    struct LineGeom {
        float x0;
        float y0;
        float x1;
        float y1;
        float mx;
        float my;
        float dx;
        float dy;
        float nx;
        float ny;
        float len;
        float proj_min;
        float proj_max;
        int angle_bin;
        std::array<int, 3> cell_x;
        std::array<int, 3> cell_y;
        int64_t batch;
        bool valid;
    };

    std::vector<LineGeom> geom(static_cast<size_t>(n));
    geom.reserve(static_cast<size_t>(n));

    auto make_key = [](int64_t batch, int angle_bin, int cell_x,
                       int cell_y) -> uint64_t {
        const uint64_t ub = static_cast<uint64_t>(batch & 0xffff);
        const uint64_t ua = static_cast<uint64_t>(angle_bin & 0x1ff);
        const uint64_t ux = static_cast<uint64_t>(cell_x & 0x7ffff);
        const uint64_t uy = static_cast<uint64_t>(cell_y & 0x7ffff);
        return (ub << 44) | (ua << 35) | (ux << 18) | uy;
    };

    for (int64_t i = 0; i < n; ++i) {
        float x0 = line_acc[i][0][0];
        float y0 = line_acc[i][0][1];
        float x1 = line_acc[i][1][0];
        float y1 = line_acc[i][1][1];

        float dx = x1 - x0;
        float dy = y1 - y0;
        if (dx < 0.0f || (dx == 0.0f && dy < 0.0f)) {
            std::swap(x0, x1);
            std::swap(y0, y1);
            dx = -dx;
            dy = -dy;
        }

        const float len = std::sqrt(dx * dx + dy * dy);
        auto &g = geom[static_cast<size_t>(i)];
        g.batch = batch_ptr[i];
        g.valid = len >= min_len;
        g.x0 = x0;
        g.y0 = y0;
        g.x1 = x1;
        g.y1 = y1;
        g.mx = 0.5f * (x0 + x1);
        g.my = 0.5f * (y0 + y1);
        g.len = len;
        g.cell_x = {static_cast<int>(std::floor(g.mx / midpoint_cell)),
                    static_cast<int>(std::floor(x0 / midpoint_cell)),
                    static_cast<int>(std::floor(x1 / midpoint_cell))};
        g.cell_y = {static_cast<int>(std::floor(g.my / midpoint_cell)),
                    static_cast<int>(std::floor(y0 / midpoint_cell)),
                    static_cast<int>(std::floor(y1 / midpoint_cell))};

        if (!g.valid) {
            g.dx = 1.0f;
            g.dy = 0.0f;
            g.nx = 0.0f;
            g.ny = 1.0f;
            g.proj_min = x0;
            g.proj_max = x1;
            g.angle_bin = 0;
            continue;
        }

        g.dx = dx / len;
        g.dy = dy / len;
        g.nx = -g.dy;
        g.ny = g.dx;
        float angle = std::atan2(g.dy, g.dx);
        if (angle < 0.0f) {
            angle += static_cast<float>(M_PI);
        }
        g.angle_bin =
            std::min(angle_bins - 1,
                     static_cast<int>(std::floor(angle / angle_step_rad)));
        const float proj0 = x0 * g.dx + y0 * g.dy;
        const float proj1 = x1 * g.dx + y1 * g.dy;
        g.proj_min = std::min(proj0, proj1);
        g.proj_max = std::max(proj0, proj1);
    }

    auto order =
        std::get<1>(torch::sort(scores_cpu, /*dim=*/0, /*descending=*/true));
    auto order_ptr = order.data_ptr<int64_t>();

    std::unordered_map<uint64_t, std::vector<int64_t>> buckets;
    buckets.reserve(static_cast<size_t>(n * 4));
    std::vector<int64_t> keep;
    keep.reserve(static_cast<size_t>(n));
    std::vector<int> seen(static_cast<size_t>(n), -1);
    int seen_tag = 0;

    for (int64_t rank = 0; rank < n; ++rank) {
        const int64_t idx = order_ptr[rank];
        const auto &cur = geom[static_cast<size_t>(idx)];
        if (!cur.valid) {
            keep.push_back(idx);
            continue;
        }

        bool suppressed = false;
        ++seen_tag;
        for (int da = -1; da <= 1 && !suppressed; ++da) {
            const int angle_bin =
                (cur.angle_bin + da + angle_bins) % angle_bins;
            for (int anchor = 0; anchor < 3 && !suppressed; ++anchor) {
                for (int gx = -1; gx <= 1 && !suppressed; ++gx) {
                    for (int gy = -1; gy <= 1 && !suppressed; ++gy) {
                        const uint64_t key = make_key(cur.batch, angle_bin,
                                                      cur.cell_x[anchor] + gx,
                                                      cur.cell_y[anchor] + gy);
                        auto it = buckets.find(key);
                        if (it == buckets.end()) {
                            continue;
                        }

                        for (const int64_t kept_idx : it->second) {
                            if (seen[static_cast<size_t>(kept_idx)] ==
                                seen_tag) {
                                continue;
                            }
                            seen[static_cast<size_t>(kept_idx)] = seen_tag;

                            const auto &ref =
                                geom[static_cast<size_t>(kept_idx)];
                            const float cos_theta = std::clamp(
                                cur.dx * ref.dx + cur.dy * ref.dy, -1.0f, 1.0f);
                            const float angle_diff = std::acos(cos_theta);
                            if (angle_diff > angle_thresh_rad) {
                                continue;
                            }

                            const float normal_dist =
                                std::fabs((cur.mx - ref.mx) * ref.nx +
                                          (cur.my - ref.my) * ref.ny);
                            if (normal_dist > normal_dist_thresh) {
                                continue;
                            }

                            const float cur_proj0 =
                                cur.x0 * ref.dx + cur.y0 * ref.dy;
                            const float cur_proj1 =
                                cur.x1 * ref.dx + cur.y1 * ref.dy;
                            const float cur_min =
                                std::min(cur_proj0, cur_proj1);
                            const float cur_max =
                                std::max(cur_proj0, cur_proj1);
                            const float overlap =
                                std::min(cur_max, ref.proj_max) -
                                std::max(cur_min, ref.proj_min);
                            if (overlap <= 0.0f) {
                                continue;
                            }

                            const float overlap_ratio =
                                overlap /
                                std::max(std::min(cur.len, ref.len), min_len);
                            if (overlap_ratio >= overlap_thresh) {
                                suppressed = true;
                                break;
                            }
                        }
                    }
                }
            }
        }

        if (suppressed) {
            continue;
        }

        keep.push_back(idx);
        for (int anchor = 0; anchor < 3; ++anchor) {
            bool duplicate_anchor = false;
            for (int prev = 0; prev < anchor; ++prev) {
                if (cur.cell_x[anchor] == cur.cell_x[prev] &&
                    cur.cell_y[anchor] == cur.cell_y[prev]) {
                    duplicate_anchor = true;
                    break;
                }
            }
            if (duplicate_anchor) {
                continue;
            }

            const uint64_t key =
                make_key(cur.batch, cur.angle_bin, cur.cell_x[anchor],
                         cur.cell_y[anchor]);
            buckets[key].push_back(idx);
        }
    }

    return torch::tensor(keep, opts);
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<int64_t>, std::vector<int64_t>>
XPLNet::processLines(const torch::Tensor &desc_map,
                     const torch::Tensor &edge_desc,
                     const torch::Tensor &junc_desc,
                     const torch::Tensor &junctions,
                     const torch::Tensor &junction_covariances,
                     const torch::Tensor &junc_valid) {
    auto &timing = mLastTiming.postprocess.line;
    timing = {};
    ScopedWallTimer total_timer(timing.total_ms);

    const auto B = junctions.size(0);
    timing.junction_slots_per_image = junctions.size(1);
    const auto Cp = desc_map.size(1);
    const auto Hp = desc_map.size(2);
    const auto Wp = desc_map.size(3);
    if (junction_covariances.dim() != 4 ||
        junction_covariances.size(0) != junctions.size(0) ||
        junction_covariances.size(1) != junctions.size(1) ||
        junction_covariances.size(2) != 2 ||
        junction_covariances.size(3) != 2) {
        throw invalid_argument(
            "junction_covariances must have shape [B, N, 2, 2]");
    }

    std::vector<torch::Tensor> lines_batch;
    std::vector<torch::Tensor> scores_batch;
    std::vector<torch::Tensor> lines_desc_batch;
    std::vector<torch::Tensor> line_covariances_batch;
    std::vector<int64_t> before_dedup_batch;
    std::vector<int64_t> after_dedup_batch;
    lines_batch.reserve(B);
    scores_batch.reserve(B);
    lines_desc_batch.reserve(B);
    line_covariances_batch.reserve(B);
    before_dedup_batch.assign(B, 0);
    after_dedup_batch.assign(B, 0);

    auto stage_begin = XPLNetNow();
    auto line_map = generateLineMap(junc_valid);
    auto sampled_pair = sampleSegmentsMap(junctions, line_map, config_.samples);
    auto sampled_points = sampled_pair.first;
    auto seg_valid = sampled_pair.second;
    timing.candidate_pairs_per_image = sampled_points.size(2);
    if (runtime_frame)
        runtime_frame->network_pair_slots += B * sampled_points.size(2);
    timing.candidate_generation_ms =
        ElapsedMilliseconds(stage_begin, XPLNetNow());

    if (sampled_points.size(2) == 0) {
        stage_begin = XPLNetNow();
        for (int64_t i = 0; i < B; ++i) {
            lines_batch.push_back(torch::zeros({0, 2, 2}, junctions.options()));
            scores_batch.push_back(torch::zeros({0}, torch::kFloat32));
            lines_desc_batch.push_back(
                torch::zeros({0, Cp}, desc_map.options()));
            line_covariances_batch.push_back(
                torch::zeros({0, 4, 4}, junction_covariances.options()));
            after_dedup_batch[i] = 0;
        }
        timing.result_packaging_ms =
            ElapsedMilliseconds(stage_begin, XPLNetNow());
        return {lines_batch, scores_batch, lines_desc_batch,
                line_covariances_batch, before_dedup_batch,
                after_dedup_batch};
    }

    const auto S = sampled_points.size(1);

    stage_begin = XPLNetNow();
    auto sampled_descs =
        verificationPreprocess(sampled_points, edge_desc, junc_desc);
    timing.verification_preprocess_ms =
        ElapsedMilliseconds(stage_begin, XPLNetNow());
    if (sampled_descs.size(2) == 0) {
        stage_begin = XPLNetNow();
        for (int64_t i = 0; i < B; ++i) {
            lines_batch.push_back(torch::zeros({0, 2, 2}, junctions.options()));
            scores_batch.push_back(torch::zeros({0}, torch::kFloat32));
            lines_desc_batch.push_back(
                torch::zeros({0, Cp}, desc_map.options()));
            line_covariances_batch.push_back(
                torch::zeros({0, 4, 4}, junction_covariances.options()));
            after_dedup_batch[i] = 0;
        }
        timing.result_packaging_ms =
            ElapsedMilliseconds(stage_begin, XPLNetNow());
        return {lines_batch, scores_batch, lines_desc_batch,
                line_covariances_batch, before_dedup_batch,
                after_dedup_batch};
    }

    stage_begin = XPLNetNow();
    verify_engine_.copy_input("line_features", sampled_descs);
    verify_engine_.infer(at::cuda::getCurrentCUDAStream());
    auto line_scores = torch::softmax(verify_engine_.get_output("logits"), 1)
                           .index({Slice(), 1, Slice()});
    timing.verification_trt_enqueue_ms =
        ElapsedMilliseconds(stage_begin, XPLNetNow());

    stage_begin = XPLNetNow();
    auto idx = torch::triu_indices(
        junctions.size(1), junctions.size(1), 1,
        torch::TensorOptions().dtype(torch::kLong).device(device_));

    auto i_idx = idx[0];
    auto j_idx = idx[1];

    auto valid_where = torch::where(seg_valid);
    auto b_idx = valid_where[0];
    if (runtime_frame) runtime_frame->network_valid_pairs += b_idx.numel();
    auto n_idx = valid_where[1];

    if (b_idx.numel() == 0) {
        timing.filtering_dedup_ms =
            ElapsedMilliseconds(stage_begin, XPLNetNow());
        stage_begin = XPLNetNow();
        for (int64_t i = 0; i < B; ++i) {
            lines_batch.push_back(torch::zeros({0, 2, 2}, junctions.options()));
            scores_batch.push_back(torch::zeros({0}, torch::kFloat32));
            lines_desc_batch.push_back(
                torch::zeros({0, Cp}, desc_map.options()));
            line_covariances_batch.push_back(
                torch::zeros({0, 4, 4}, junction_covariances.options()));
            after_dedup_batch[i] = 0;
        }
        timing.result_packaging_ms =
            ElapsedMilliseconds(stage_begin, XPLNetNow());
        return {lines_batch, scores_batch, lines_desc_batch,
                line_covariances_batch, before_dedup_batch,
                after_dedup_batch};
    }
    auto i_valid = i_idx.index_select(0, n_idx);
    auto j_valid = j_idx.index_select(0, n_idx);

    auto p0 = junctions.index({b_idx, i_valid, Slice()});
    auto p1 = junctions.index({b_idx, j_valid, Slice()});
    auto lines = torch::stack({p0, p1}, 1);
    auto line_covariances = assembleLineEndpointCovariances(
        junction_covariances, b_idx, i_valid, j_valid);
    auto lines_sample = sampled_points.index({b_idx, Slice(), n_idx, Slice()});
    auto scores = line_scores.index({b_idx, n_idx});
    auto scores_mask = scores > config_.line_th;

    auto final_lines = lines.index({scores_mask});
    auto final_scores = scores.index({scores_mask});
    auto final_line_covariances = line_covariances.index({scores_mask});
    auto final_lines_sample = lines_sample.index({scores_mask});
    auto final_b_idx = b_idx.index({scores_mask});

    auto long_opts = torch::TensorOptions().dtype(torch::kLong).device(device_);
    auto before_counts = torch::zeros({B}, long_opts);
    if (final_b_idx.numel() > 0) {
        before_counts.scatter_add_(0, final_b_idx,
                                   torch::ones_like(final_b_idx, long_opts));
        auto before_counts_cpu = before_counts.to(torch::kCPU).contiguous();
        auto before_ptr = before_counts_cpu.data_ptr<int64_t>();
        for (int64_t i = 0; i < B; ++i) {
            before_dedup_batch[i] = before_ptr[i];
        }
    }

    if (final_lines.size(0) == 0) {
        timing.filtering_dedup_ms =
            ElapsedMilliseconds(stage_begin, XPLNetNow());
        stage_begin = XPLNetNow();
        for (int64_t i = 0; i < B; ++i) {
            lines_batch.push_back(torch::zeros({0, 2, 2}, junctions.options()));
            scores_batch.push_back(torch::zeros({0}, torch::kFloat32));
            lines_desc_batch.push_back(
                torch::zeros({0, Cp}, desc_map.options()));
            line_covariances_batch.push_back(
                torch::zeros({0, 4, 4}, junction_covariances.options()));
            after_dedup_batch[i] = 0;
        }
        timing.result_packaging_ms =
            ElapsedMilliseconds(stage_begin, XPLNetNow());
        return {lines_batch, scores_batch, lines_desc_batch,
                line_covariances_batch, before_dedup_batch,
                after_dedup_batch};
    }

    if (config_.dedup_lines && final_lines.size(0) > 1) {
        auto keep_idx =
            deduplicateLinesFast(final_lines, final_scores, final_b_idx);
        final_lines = final_lines.index_select(0, keep_idx);
        final_scores = final_scores.index_select(0, keep_idx);
        final_line_covariances =
            final_line_covariances.index_select(0, keep_idx);
        final_lines_sample = final_lines_sample.index_select(0, keep_idx);
        final_b_idx = final_b_idx.index_select(0, keep_idx);
    }
    timing.filtering_dedup_ms =
        ElapsedMilliseconds(stage_begin, XPLNetNow());

    stage_begin = XPLNetNow();
    torch::Tensor lines_descs_flat;
    auto counts = torch::zeros({B}, long_opts);
    counts.scatter_add_(0, final_b_idx,
                        torch::ones_like(final_b_idx, long_opts));
    auto counts_cpu = counts.to(torch::kCPU).contiguous();
    auto counts_ptr = counts_cpu.data_ptr<int64_t>();
    for (int64_t i = 0; i < B; ++i) {
        after_dedup_batch[i] = counts_ptr[i];
    }
    const int64_t max_count = counts.max().item<int64_t>();

    auto padded =
        torch::zeros({B, max_count * S, 2}, final_lines_sample.options());

    auto sorted_idx = torch::argsort(final_b_idx, true, 0, false);
    auto sorted_b = final_b_idx.index_select(0, sorted_idx);

    auto batch_offsets = torch::cumsum(counts, 0) - counts;
    auto pos_sorted = torch::arange(final_b_idx.size(0), long_opts) -
                      batch_offsets.index_select(0, sorted_b);

    auto pos_in_batch = torch::empty_like(final_b_idx);
    pos_in_batch.scatter_(0, sorted_idx, pos_sorted);

    auto offsets = pos_in_batch * S;
    auto s_range = torch::arange(S, long_opts).unsqueeze(0);
    auto flat_indices = offsets.unsqueeze(1) + s_range;

    auto padded_flat = padded.view({B * max_count * S, 2});
    auto linear_idx = final_b_idx.unsqueeze(1) * max_count * S + flat_indices;

    padded_flat.index_put_({linear_idx.reshape(-1)},
                           final_lines_sample.reshape({-1, 2}));

    auto all_descs =
        interpolateSparse(desc_map, padded, 8 * Hp, 8 * Wp, torch::kBilinear);
    all_descs = all_descs.permute({0, 2, 1});
    all_descs = all_descs.view({B, Cp, max_count, S});

    line_description_engine_.copy_input("line_samples", all_descs);
    timing.line_descriptor_preprocess_ms =
        ElapsedMilliseconds(stage_begin, XPLNetNow());

    stage_begin = XPLNetNow();
    line_description_engine_.infer(at::cuda::getCurrentCUDAStream());
    auto lines_descs_all =
        line_description_engine_.get_output("line_descriptors");
    timing.line_descriptor_trt_enqueue_ms =
        ElapsedMilliseconds(stage_begin, XPLNetNow());

    stage_begin = XPLNetNow();
    if (lines_descs_all.dim() == 2) {
        lines_descs_all = lines_descs_all.view({B, max_count, -1});
    }

    // lines_descs_all =
    //     F::normalize(lines_descs_all, F::NormalizeFuncOptions().dim(-1));

    lines_descs_flat = lines_descs_all.index({final_b_idx, pos_in_batch});

    auto sorted_order = torch::argsort(final_b_idx, true, 0, false);
    sorted_b = final_b_idx.index_select(0, sorted_order);

    auto split_starts =
        torch::searchsorted(sorted_b, torch::arange(B, long_opts));

    for (int64_t b = 0; b < B; ++b) {
        const int64_t start = split_starts[b].item<int64_t>();
        const int64_t end = (b + 1 < B) ? split_starts[b + 1].item<int64_t>()
                                        : final_b_idx.size(0);
        auto ord = sorted_order.slice(0, start, end);

        lines_batch.emplace_back(final_lines.index_select(0, ord));
        scores_batch.emplace_back(final_scores.index_select(0, ord));
        lines_desc_batch.emplace_back(lines_descs_flat.index_select(0, ord));
        line_covariances_batch.emplace_back(
            final_line_covariances.index_select(0, ord));
    }
    timing.result_packaging_ms =
        ElapsedMilliseconds(stage_begin, XPLNetNow());

    return {lines_batch, scores_batch, lines_desc_batch, line_covariances_batch,
            before_dedup_batch, after_dedup_batch};
}

std::vector<DetectionResult> XPLNet::detect(const std::vector<cv::Mat> &imgs,
                                            cudaStream_t stream) {
    mLastTiming = {};
    deferred_valid_ = false;
    const auto total_begin = XPLNetNow();
    torch::InferenceMode guard(true);
    auto detect_stream = DetectStream(stream, frontend_stream_);
    c10::cuda::CUDAStreamGuard stream_guard(detect_stream);
    cudaStream_t raw_stream = detect_stream.stream();

    std::vector<float> scale_ws;
    std::vector<float> scale_hs;
    auto stage_begin = XPLNetNow();
    preprocess(imgs, scale_ws, scale_hs);
    mLastTiming.preprocess_ms =
        ElapsedMilliseconds(stage_begin, XPLNetNow());

    stage_begin = XPLNetNow();
    xplnet_engine_.infer(raw_stream);
    mLastTiming.frontend_trt_enqueue_ms =
        ElapsedMilliseconds(stage_begin, XPLNetNow());

    stage_begin = XPLNetNow();
    auto results = postprocessDetections(static_cast<int64_t>(imgs.size()),
                                         scale_ws, scale_hs);
    mLastTiming.postprocess_ms =
        ElapsedMilliseconds(stage_begin, XPLNetNow());

    if (stream == nullptr) {
        stage_begin = XPLNetNow();
        detect_stream.synchronize();
        mLastTiming.final_stream_sync_ms =
            ElapsedMilliseconds(stage_begin, XPLNetNow());
    }
    mLastTiming.total_ms =
        ElapsedMilliseconds(total_begin, XPLNetNow());
    return results;
}

DeferredPointDetectionBatch
XPLNet::detectPoints(const std::vector<cv::Mat> &imgs, cudaStream_t stream) {
    mLastTiming = {};
    deferred_valid_ = false;
    const auto total_begin = XPLNetNow();
    torch::InferenceMode guard(true);
    auto detect_stream = DetectStream(stream, frontend_stream_);
    c10::cuda::CUDAStreamGuard stream_guard(detect_stream);
    cudaStream_t raw_stream = detect_stream.stream();

    std::vector<float> scale_ws;
    std::vector<float> scale_hs;
    auto stage_begin = XPLNetNow();
    preprocess(imgs, scale_ws, scale_hs);
    mLastTiming.preprocess_ms =
        ElapsedMilliseconds(stage_begin, XPLNetNow());

    stage_begin = XPLNetNow();
    xplnet_engine_.infer(raw_stream);
    mLastTiming.frontend_trt_enqueue_ms =
        ElapsedMilliseconds(stage_begin, XPLNetNow());

    stage_begin = XPLNetNow();
    DeferredPointDetectionBatch batch;
    batch.detections = postprocessDetections(
        static_cast<int64_t>(imgs.size()), scale_ws, scale_hs, false);
    mLastTiming.postprocess_ms =
        ElapsedMilliseconds(stage_begin, XPLNetNow());
    batch.token = deferred_token_;

    if (stream == nullptr) {
        stage_begin = XPLNetNow();
        detect_stream.synchronize();
        mLastTiming.final_stream_sync_ms =
            ElapsedMilliseconds(stage_begin, XPLNetNow());
    }
    mLastTiming.total_ms =
        ElapsedMilliseconds(total_begin, XPLNetNow());
    return batch;
}

std::vector<DetectionResult> XPLNet::completeLines(uint64_t token,
                                                   cudaStream_t stream) {
    if (!deferred_valid_ || token == 0 || token != deferred_token_) {
        throw std::runtime_error(
            "Deferred XPLNet line token is stale or already consumed");
    }
    torch::InferenceMode guard(true);
    auto detect_stream = DetectStream(stream, frontend_stream_);
    c10::cuda::CUDAStreamGuard stream_guard(detect_stream);
    const auto stage_begin = XPLNetNow();

    auto lines_result = processLines(
        deferred_desc_map_, deferred_edge_desc_, deferred_junc_desc_,
        deferred_junctions_, deferred_junction_covariances_,
        deferred_junction_valid_);
    auto &lines_batch = std::get<0>(lines_result);
    auto &scores_batch = std::get<1>(lines_result);
    auto &descriptors_batch = std::get<2>(lines_result);
    auto &covariances_batch = std::get<3>(lines_result);
    auto &before_dedup_batch = std::get<4>(lines_result);
    auto &after_dedup_batch = std::get<5>(lines_result);

    std::vector<DetectionResult> results;
    results.reserve(static_cast<size_t>(deferred_batch_size_));
    for (int64_t bi = 0; bi < deferred_batch_size_; ++bi) {
        DetectionResult result;
        auto line_scale =
            torch::tensor({deferred_scale_ws_[bi], deferred_scale_hs_[bi]},
                          lines_batch[bi].options().dtype(torch::kFloat32))
                .view({1, 1, 2});
        result.lines = restoreImageCoordinates(
            lines_batch[bi].to(torch::kFloat32), line_scale);
        auto covariance_scale = torch::tensor(
            {deferred_scale_ws_[bi], deferred_scale_hs_[bi],
             deferred_scale_ws_[bi], deferred_scale_hs_[bi]},
            covariances_batch[bi].options().dtype(torch::kFloat32));
        result.line_endpoint_covariances = scaleCovariances(
            covariances_batch[bi].to(torch::kFloat32), covariance_scale);
        result.line_scores = scores_batch[bi];
        result.line_descriptors = descriptors_batch[bi];
        result.line_candidates_before_dedup = before_dedup_batch[bi];
        result.line_candidates_after_dedup = after_dedup_batch[bi];
        results.push_back(std::move(result));
    }

    deferred_valid_ = false;
    deferred_desc_map_ = torch::Tensor();
    deferred_edge_desc_ = torch::Tensor();
    deferred_junc_desc_ = torch::Tensor();
    deferred_junctions_ = torch::Tensor();
    deferred_junction_covariances_ = torch::Tensor();
    deferred_junction_valid_ = torch::Tensor();
    mLastTiming.postprocess.line_processing_ms +=
        ElapsedMilliseconds(stage_begin, XPLNetNow());
    if (stream == nullptr) {
        detect_stream.synchronize();
    }
    return results;
}

std::vector<DetectionResult> XPLNet::detect(const torch::Tensor &img,
                                            cudaStream_t stream) {
    mLastTiming = {};
    deferred_valid_ = false;
    const auto total_begin = XPLNetNow();
    auto detect_stream = DetectStream(stream, frontend_stream_);
    c10::cuda::CUDAStreamGuard stream_guard(detect_stream);
    cudaStream_t raw_stream = detect_stream.stream();

    auto stage_begin = XPLNetNow();
    preprocess(img);
    mLastTiming.preprocess_ms =
        ElapsedMilliseconds(stage_begin, XPLNetNow());

    stage_begin = XPLNetNow();
    xplnet_engine_.infer(raw_stream);
    mLastTiming.frontend_trt_enqueue_ms =
        ElapsedMilliseconds(stage_begin, XPLNetNow());

    const int64_t b = img.size(0);
    const std::vector<float> scale_ws(
        static_cast<size_t>(b),
        img.size(3) / static_cast<float>(config_.resize_width));
    const std::vector<float> scale_hs(
        static_cast<size_t>(b),
        img.size(2) / static_cast<float>(config_.resize_height));
    stage_begin = XPLNetNow();
    auto results = postprocessDetections(b, scale_ws, scale_hs);
    mLastTiming.postprocess_ms =
        ElapsedMilliseconds(stage_begin, XPLNetNow());

    if (stream == nullptr) {
        stage_begin = XPLNetNow();
        detect_stream.synchronize();
        mLastTiming.final_stream_sync_ms =
            ElapsedMilliseconds(stage_begin, XPLNetNow());
    }
    mLastTiming.total_ms =
        ElapsedMilliseconds(total_begin, XPLNetNow());
    return results;
}

std::vector<DetectionResult>
XPLNet::postprocessDetections(int64_t b, const std::vector<float> &scale_ws,
                              const std::vector<float> &scale_hs,
                              bool include_lines) {
    auto &timing = mLastTiming.postprocess;
    timing = {};
    ScopedWallTimer total_timer(timing.total_ms);

    if (scale_ws.size() != static_cast<size_t>(b) ||
        scale_hs.size() != static_cast<size_t>(b)) {
        throw std::runtime_error("XPLNet scale count does not match batch");
    }

    const auto h = config_.resize_height;
    const auto w = config_.resize_width;

    auto stage_begin = XPLNetNow();
    auto desc_map = xplnet_engine_.get_output("desc_map");
    auto keypoint_map = xplnet_engine_.get_output("keypoint_logits");
    auto heatmap = xplnet_engine_.get_output("heatmap");
    auto junction_logits = xplnet_engine_.get_output("junction_logits");
    auto edge_desc = xplnet_engine_.get_output("edge_desc");
    auto junc_desc = xplnet_engine_.get_output("junc_desc");
    timing.output_access_ms =
        ElapsedMilliseconds(stage_begin, XPLNetNow());

    stage_begin = XPLNetNow();
    auto junctions_prob = torch::sigmoid(junction_logits);
    auto keypoint_h = getKptsHeatmap(keypoint_map);

    auto mkpts = nms(keypoint_h, config_.kp_th, 5, true);
    auto junc_pts = nms(junctions_prob, config_.junc_th, 5, true);

    auto kp_scores =
        (interpolateSparse(keypoint_h, mkpts, h, w, torch::kNearest) *
         interpolateSparse(heatmap, mkpts, h, w, torch::kBilinear))
            .squeeze(-1);
    auto junc_scores =
        interpolateSparse(junctions_prob, junc_pts, h, w, torch::kBilinear)
            .squeeze(-1);

    kp_scores = kp_scores.masked_fill(torch::all(mkpts == 0, -1), -1);
    junc_scores = junc_scores.masked_fill(torch::all(junc_pts == 0, -1), -1);

    auto kp_idxs = torch::argsort(kp_scores, -1, true);
    auto mkpts_x =
        torch::gather(mkpts.index({Slice(), Slice(), 0}), -1, kp_idxs)
            .slice(1, 0, config_.top_k_kp);
    auto mkpts_y =
        torch::gather(mkpts.index({Slice(), Slice(), 1}), -1, kp_idxs)
            .slice(1, 0, config_.top_k_kp);

    mkpts = torch::stack({mkpts_x, mkpts_y}, -1);
    kp_scores =
        torch::gather(kp_scores, -1, kp_idxs).slice(1, 0, config_.top_k_kp);

    auto junc_idxs = torch::argsort(junc_scores, -1, true);
    auto juncs_x =
        torch::gather(junc_pts.index({Slice(), Slice(), 0}), -1, junc_idxs)
            .slice(1, 0, config_.top_k_junc);
    auto juncs_y =
        torch::gather(junc_pts.index({Slice(), Slice(), 1}), -1, junc_idxs)
            .slice(1, 0, config_.top_k_junc);

    junc_pts = torch::stack({juncs_x, juncs_y}, -1);
    junc_scores = torch::gather(junc_scores, -1, junc_idxs)
                      .slice(1, 0, config_.top_k_junc);

    auto kp_valid = kp_scores > 0;
    auto junc_valid = junc_scores > 0;

    auto refined_junctions = refineJunctionsAndCovariances(
        junction_logits, junc_pts, junc_valid);
    junc_pts = refined_junctions.first;
    auto junction_covariances = refined_junctions.second;

    auto desc_map_n =
        F::normalize(desc_map, F::NormalizeFuncOptions().p(2).dim(1));
    timing.point_junction_decode_ms =
        ElapsedMilliseconds(stage_begin, XPLNetNow());

    std::vector<torch::Tensor> lines_batch;
    std::vector<torch::Tensor> scores_batch;
    std::vector<torch::Tensor> lines_desc_batch;
    std::vector<torch::Tensor> line_covariances_batch;
    std::vector<int64_t> line_before_dedup_batch(static_cast<size_t>(b), 0);
    std::vector<int64_t> line_after_dedup_batch(static_cast<size_t>(b), 0);
    if (include_lines) {
        stage_begin = XPLNetNow();
        auto lines_result = processLines(desc_map_n, edge_desc, junc_desc,
                                         junc_pts, junction_covariances,
                                         junc_valid);
        lines_batch = std::move(std::get<0>(lines_result));
        scores_batch = std::move(std::get<1>(lines_result));
        lines_desc_batch = std::move(std::get<2>(lines_result));
        line_covariances_batch = std::move(std::get<3>(lines_result));
        line_before_dedup_batch = std::move(std::get<4>(lines_result));
        line_after_dedup_batch = std::move(std::get<5>(lines_result));
        for (int64_t bi = 0; bi < b; ++bi) {
            timing.line.accepted_before_dedup +=
                line_before_dedup_batch[bi];
            timing.line.accepted_after_dedup += line_after_dedup_batch[bi];
        }
        timing.line_processing_ms =
            ElapsedMilliseconds(stage_begin, XPLNetNow());
    } else {
        deferred_batch_size_ = b;
        deferred_scale_ws_ = scale_ws;
        deferred_scale_hs_ = scale_hs;
        deferred_desc_map_ = desc_map_n;
        deferred_edge_desc_ = edge_desc;
        deferred_junc_desc_ = junc_desc;
        deferred_junctions_ = junc_pts;
        deferred_junction_covariances_ = junction_covariances;
        deferred_junction_valid_ = junc_valid;
        deferred_token_ = ++next_deferred_token_;
        if (deferred_token_ == 0) {
            deferred_token_ = ++next_deferred_token_;
        }
        deferred_valid_ = true;

        lines_batch.reserve(static_cast<size_t>(b));
        scores_batch.reserve(static_cast<size_t>(b));
        lines_desc_batch.reserve(static_cast<size_t>(b));
        line_covariances_batch.reserve(static_cast<size_t>(b));
        for (int64_t bi = 0; bi < b; ++bi) {
            lines_batch.push_back(
                torch::zeros({0, 2, 2}, junc_pts.options()));
            scores_batch.push_back(torch::zeros({0}, junc_pts.options()));
            lines_desc_batch.push_back(
                torch::zeros({0, config_.ldesc_dim}, desc_map_n.options()));
            line_covariances_batch.push_back(
                torch::zeros({0, 4, 4}, junction_covariances.options()));
        }
    }

    stage_begin = XPLNetNow();
    auto descs = F::normalize(
        interpolateSparse(desc_map_n, mkpts, h, w, torch::kBicubic),
        F::NormalizeFuncOptions().p(2).dim(-1));
    timing.point_descriptor_sampling_ms =
        ElapsedMilliseconds(stage_begin, XPLNetNow());

    stage_begin = XPLNetNow();
    std::vector<float> scale_values(static_cast<size_t>(b) * 2);
    for (int64_t bi = 0; bi < b; ++bi) {
        const size_t base = static_cast<size_t>(bi) * 2;
        scale_values[base] = scale_ws[bi];
        scale_values[base + 1] = scale_hs[bi];
    }
    auto scale = torch::from_blob(scale_values.data(), {b, 1, 2},
                                  torch::TensorOptions().dtype(torch::kFloat32))
                     .clone()
                     .to(mkpts.device());
    mkpts = restoreImageCoordinates(mkpts, scale);

    std::vector<DetectionResult> results;
    results.reserve(b);
    for (int64_t bi = 0; bi < b; ++bi) {
        DetectionResult res;
        auto valid_kp = torch::nonzero(kp_valid[bi]).squeeze(1);
        res.keypoints = mkpts.index({bi, valid_kp, Slice()});
        res.keypoint_scores = kp_scores.index({bi, valid_kp});
        res.descriptors = descs.index({bi, valid_kp, Slice()});

        auto line_scale =
            torch::tensor({scale_ws[bi], scale_hs[bi]},
                          lines_batch[bi].options().dtype(torch::kFloat32))
                .view({1, 1, 2});
        res.lines = restoreImageCoordinates(
            lines_batch[bi].to(torch::kFloat32), line_scale);
        auto covariance_scale = torch::tensor(
            {scale_ws[bi], scale_hs[bi], scale_ws[bi], scale_hs[bi]},
            line_covariances_batch[bi].options().dtype(torch::kFloat32));
        res.line_endpoint_covariances = scaleCovariances(
            line_covariances_batch[bi].to(torch::kFloat32), covariance_scale);
        res.line_scores = scores_batch[bi];
        res.line_descriptors = lines_desc_batch[bi];
        res.line_candidates_before_dedup = line_before_dedup_batch[bi];
        res.line_candidates_after_dedup = line_after_dedup_batch[bi];
        results.push_back(std::move(res));
    }
    timing.result_packaging_ms =
        ElapsedMilliseconds(stage_begin, XPLNetNow());

    return results;
}

XPLNetFeatureGrids XPLNet::buildGrids(const DetectionResult &result,
                                      const cv::Size &image_size,
                                      const GridConfig &grid_config) const {
    XPLNetFeatureGrids grids;

    const int point_cols =
        std::max(1, static_cast<int>(std::ceil(
                        image_size.width /
                        static_cast<float>(grid_config.point_grid_size))));
    const int point_rows =
        std::max(1, static_cast<int>(std::ceil(
                        image_size.height /
                        static_cast<float>(grid_config.point_grid_size))));

    const int inter_cols =
        std::max(1, static_cast<int>(std::ceil(
                        image_size.width /
                        static_cast<float>(grid_config.inter_grid_size))));
    const int inter_rows =
        std::max(1, static_cast<int>(std::ceil(
                        image_size.height /
                        static_cast<float>(grid_config.inter_grid_size))));

    if (result.keypoints.defined()) {
        auto keypoints = result.keypoints.to(torch::kFloat32);
        if (keypoints.numel() > 0) {
            auto point_x = torch::clamp(
                torch::floor(keypoints.index({Slice(), 0}) /
                             static_cast<float>(grid_config.point_grid_size))
                    .to(torch::kInt64),
                0, point_cols - 1);
            auto point_y = torch::clamp(
                torch::floor(keypoints.index({Slice(), 1}) /
                             static_cast<float>(grid_config.point_grid_size))
                    .to(torch::kInt64),
                0, point_rows - 1);
            auto inter_x = torch::clamp(
                torch::floor(keypoints.index({Slice(), 0}) /
                             static_cast<float>(grid_config.inter_grid_size))
                    .to(torch::kInt64),
                0, inter_cols - 1);
            auto inter_y = torch::clamp(
                torch::floor(keypoints.index({Slice(), 1}) /
                             static_cast<float>(grid_config.inter_grid_size))
                    .to(torch::kInt64),
                0, inter_rows - 1);
            grids.point_bins = torch::stack({point_x, point_y}, 1);
            grids.inter_point_bins = torch::stack({inter_x, inter_y}, 1);
        }
    }

    const float diag =
        std::sqrt(static_cast<float>(image_size.width * image_size.width +
                                     image_size.height * image_size.height));
    grids.line_rho_offset =
        std::max(1, static_cast<int>(std::ceil(diag / grid_config.delta_rho)));
    const int rho_bins = 2 * grids.line_rho_offset + 1;
    const int theta_bins =
        std::max(1, static_cast<int>(std::ceil(static_cast<float>(M_PI) /
                                               grid_config.delta_theta))) +
        1;

    if (result.lines.defined()) {
        auto lines = result.lines.to(torch::kFloat32);
        if (lines.numel() > 0) {
            auto x1 = lines.index({Slice(), 0, 0});
            auto y1 = lines.index({Slice(), 0, 1});
            auto x2 = lines.index({Slice(), 1, 0});
            auto y2 = lines.index({Slice(), 1, 1});
            auto dx = x2 - x1;
            auto dy = y2 - y1;
            auto length = torch::sqrt(dx * dx + dy * dy);
            auto valid = length >= 1e-3f;
            auto theta = torch::atan2(dy, dx);
            theta = torch::where(theta < -static_cast<float>(M_PI) / 2.0f,
                                 theta + static_cast<float>(M_PI), theta);
            theta = torch::where(theta >= static_cast<float>(M_PI) / 2.0f,
                                 theta - static_cast<float>(M_PI), theta);
            auto rho = (x1 * y2 - y1 * x2) / torch::clamp_min(length, 1e-3f);
            auto rho_bin = torch::clamp(
                torch::round(rho / grid_config.delta_rho).to(torch::kInt64) +
                    grids.line_rho_offset,
                0, rho_bins - 1);
            auto theta_bin = torch::clamp(
                torch::round((theta + static_cast<float>(M_PI) / 2.0f) /
                             grid_config.delta_theta)
                    .to(torch::kInt64),
                0, theta_bins - 1);
            grids.line_bins = torch::stack({rho_bin, theta_bin}, 1);
            grids.line_valid_mask = valid;
        }
    }

    return grids;
}

std::vector<XPLNetFrontendResult>
XPLNet::detectWithGrids(const torch::Tensor &img,
                        const std::vector<cv::Size> &sizes,
                        const GridConfig &grid_config, cudaStream_t stream) {
    auto detections = detect(img, stream);
    if (detections.size() != sizes.size()) {
        throw std::runtime_error("XPLNet detectWithGrids size mismatch");
    }

    auto detect_stream = DetectStream(stream, frontend_stream_);
    c10::cuda::CUDAStreamGuard stream_guard(detect_stream);

    std::vector<XPLNetFrontendResult> results;
    results.reserve(detections.size());
    for (size_t i = 0; i < detections.size(); ++i) {
        XPLNetFrontendResult result;
        result.grids = buildGrids(detections[i], sizes[i], grid_config);
        result.detection = std::move(detections[i]);
        results.push_back(std::move(result));
    }
    if (stream == nullptr) {
        detect_stream.synchronize();
    }
    return results;
}

} // namespace XPLSLAM
