#include "xplnet.h"
#include "settings.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <string>

namespace XPLSLAM {

struct XPLNetTestAccess {
    static std::pair<torch::Tensor, torch::Tensor>
    RefineJunctionsAndCovariances(const torch::Tensor &junction_logits,
                                  const torch::Tensor &junctions,
                                  const torch::Tensor &junction_valid) {
        return XPLNet::refineJunctionsAndCovariances(
            junction_logits, junctions, junction_valid);
    }

    static torch::Tensor RestoreImageCoordinates(
        const torch::Tensor &coordinates,
        const torch::Tensor &coordinate_scale) {
        return XPLNet::restoreImageCoordinates(coordinates, coordinate_scale);
    }

    static torch::Tensor ScaleCovariances(
        const torch::Tensor &covariances,
        const torch::Tensor &coordinate_scale) {
        return XPLNet::scaleCovariances(covariances, coordinate_scale);
    }

    static torch::Tensor AssembleLineEndpointCovariances(
        const torch::Tensor &junction_covariances,
        const torch::Tensor &batch_indices,
        const torch::Tensor &start_indices,
        const torch::Tensor &end_indices) {
        return XPLNet::assembleLineEndpointCovariances(
            junction_covariances, batch_indices, start_indices, end_indices);
    }

    static std::vector<DetectionResult>
    PostprocessDetections(XPLNet &xplnet, int64_t batch_size,
                          const std::vector<float> &scale_w,
                          const std::vector<float> &scale_h) {
        return xplnet.postprocessDetections(batch_size, scale_w, scale_h);
    }
};

} // namespace XPLSLAM

namespace {

bool Check(bool condition, const char *message) {
    if (!condition)
        std::cerr << message << std::endl;
    return condition;
}

bool TensorClose(const torch::Tensor &actual, const torch::Tensor &expected,
                 float tolerance = 1e-5f) {
    if (actual.sizes() != expected.sizes())
        return false;
    if (actual.numel() == 0)
        return true;
    return (actual - expected).abs().max().item<float>() <= tolerance;
}

bool CheckLineCovariances(const XPLSLAM::DetectionResult &result,
                          int image_width, int image_height,
                          const char *image_name) {
    bool ok = true;
    const auto &lines = result.lines;
    const auto &covariances = result.line_endpoint_covariances;
    ok &= Check(lines.dim() == 3 && lines.size(1) == 2 && lines.size(2) == 2,
                "GPU line output does not have shape [N, 2, 2]");
    ok &= Check(covariances.dim() == 3 && covariances.size(1) == 4 &&
                    covariances.size(2) == 4,
                "GPU covariance output does not have shape [N, 4, 4]");
    if (!ok)
        return false;

    ok &= Check(lines.size(0) > 0, "GPU inference returned no lines");
    ok &= Check(covariances.size(0) == lines.size(0),
                "GPU line and covariance counts differ");
    ok &= Check(torch::isfinite(lines).all().item<bool>(),
                "GPU line output contains a non-finite value");
    ok &= Check(torch::isfinite(covariances).all().item<bool>(),
                "GPU covariance output contains a non-finite value");
    if (lines.size(0) == 0 || covariances.size(0) != lines.size(0))
        return false;

    const auto flat_lines = lines.reshape({-1, 2});
    ok &= Check((flat_lines.index({torch::indexing::Slice(), 0}) >= 0)
                    .all()
                    .item<bool>() &&
                    (flat_lines.index({torch::indexing::Slice(), 0}) <
                     image_width)
                        .all()
                        .item<bool>() &&
                    (flat_lines.index({torch::indexing::Slice(), 1}) >= 0)
                        .all()
                        .item<bool>() &&
                    (flat_lines.index({torch::indexing::Slice(), 1}) <
                     image_height)
                        .all()
                        .item<bool>(),
                "GPU refined endpoints are outside the image");
    ok &= Check((lines - lines.round()).abs().max().item<float>() > 1e-4f,
                "GPU endpoints did not contain a sub-pixel refinement");

    auto covariances_cpu =
        covariances.to(torch::kFloat32).to(torch::kCPU).contiguous();
    auto covariance = covariances_cpu.accessor<float, 3>();
    for (int64_t i = 0; i < covariances_cpu.size(0); ++i) {
        for (int row = 0; row < 4; ++row) {
            ok &= Check(covariance[i][row][row] > 0.0f,
                        "GPU covariance has a non-positive diagonal");
            for (int col = 0; col < 4; ++col) {
                ok &= Check(std::fabs(covariance[i][row][col] -
                                      covariance[i][col][row]) <= 1e-4f,
                            "GPU covariance is not symmetric");
                if ((row < 2) != (col < 2)) {
                    ok &= Check(std::fabs(covariance[i][row][col]) <= 1e-6f,
                                "GPU endpoint cross-covariance is not zero");
                }
            }
        }
        for (int offset : {0, 2}) {
            const float determinant =
                covariance[i][offset][offset] *
                    covariance[i][offset + 1][offset + 1] -
                covariance[i][offset][offset + 1] *
                    covariance[i][offset + 1][offset];
            ok &= Check(determinant >= -1e-4f,
                        "GPU endpoint covariance is not positive semidefinite");
        }
    }

    std::cout << image_name << ": " << lines.size(0)
              << " lines, covariance shape [" << covariances.size(0)
              << ", 4, 4]" << std::endl;
    return ok;
}

bool RunGpuCheck(const std::string &config_path) {
    using XPLSLAM::Settings;
    using XPLSLAM::XPLNet;
    using XPLSLAM::XPLNetTestAccess;

    Settings settings(config_path);
    auto config = settings.xplnetConfig();
    config.top_k_kp = 64;
    config.top_k_junc = 24;
    config.kp_th = 1e-4f;
    config.junc_th = 1e-4f;
    config.line_th = 0.0f;

    XPLNet xplnet(config);
    cv::Mat first(config.resize_height, config.resize_width, CV_8UC1,
                  cv::Scalar(16));
    for (int y = 48; y < first.rows; y += 80) {
        cv::line(first, {24, y}, {first.cols - 25, y}, cv::Scalar(224), 3,
                 cv::LINE_AA);
    }
    for (int x = 56; x < first.cols; x += 96) {
        cv::line(first, {x, 24}, {x, first.rows - 25}, cv::Scalar(192), 2,
                 cv::LINE_AA);
    }
    cv::line(first, {20, 20}, {first.cols - 21, first.rows - 21},
             cv::Scalar(255), 4, cv::LINE_AA);
    cv::line(first, {first.cols - 21, 20}, {20, first.rows - 21},
             cv::Scalar(255), 4, cv::LINE_AA);

    cv::Mat second;
    cv::flip(first, second, 1);
    std::vector<cv::Mat> images{first, second};
    auto baseline = xplnet.detect(images);

    bool ok = Check(baseline.size() == images.size(),
                    "GPU inference returned the wrong batch size");
    if (!ok)
        return false;
    ok &= CheckLineCovariances(baseline[0], first.cols, first.rows,
                               "GPU image 0");
    ok &= CheckLineCovariances(baseline[1], second.cols, second.rows,
                               "GPU image 1");

    auto deferred = xplnet.detectPoints(images);
    ok &= Check(deferred.token != 0,
                "deferred point inference returned an invalid token");
    ok &= Check(deferred.detections.size() == baseline.size(),
                "deferred point inference returned the wrong batch size");
    if (deferred.detections.size() != baseline.size())
        return false;
    for (size_t i = 0; i < baseline.size(); ++i) {
        const auto &point_result = deferred.detections[i];
        ok &= Check(point_result.lines.size(0) == 0,
                    "point-only inference unexpectedly materialized lines");
        ok &= Check(TensorClose(point_result.keypoints, baseline[i].keypoints),
                    "deferred keypoints differ from eager inference");
        ok &= Check(TensorClose(point_result.keypoint_scores,
                                baseline[i].keypoint_scores),
                    "deferred keypoint scores differ from eager inference");
        ok &= Check(TensorClose(point_result.descriptors,
                                baseline[i].descriptors),
                    "deferred point descriptors differ from eager inference");
    }

    auto completed_lines = xplnet.completeLines(deferred.token);
    ok &= Check(completed_lines.size() == baseline.size(),
                "deferred line completion returned the wrong batch size");
    if (completed_lines.size() != baseline.size())
        return false;
    for (size_t i = 0; i < baseline.size(); ++i) {
        ok &= Check(TensorClose(completed_lines[i].lines, baseline[i].lines,
                                1e-4f),
                    "deferred line endpoints differ from eager inference");
        ok &= Check(TensorClose(completed_lines[i].line_scores,
                                baseline[i].line_scores, 1e-4f),
                    "deferred line scores differ from eager inference");
        ok &= Check(TensorClose(completed_lines[i].line_descriptors,
                                baseline[i].line_descriptors, 1e-4f),
                    "deferred line descriptors differ from eager inference");
        ok &= Check(TensorClose(
                        completed_lines[i].line_endpoint_covariances,
                        baseline[i].line_endpoint_covariances, 1e-4f),
                    "deferred line covariances differ from eager inference");
        ok &= Check(completed_lines[i].line_candidates_before_dedup ==
                        baseline[i].line_candidates_before_dedup &&
                        completed_lines[i].line_candidates_after_dedup ==
                            baseline[i].line_candidates_after_dedup,
                    "deferred line deduplication counts differ from eager "
                    "inference");
    }

    const std::vector<float> scale_w{2.0f, 0.5f};
    const std::vector<float> scale_h{3.0f, 1.5f};
    auto scaled = XPLNetTestAccess::PostprocessDetections(
        xplnet, static_cast<int64_t>(images.size()), scale_w, scale_h);
    ok &= Check(scaled.size() == baseline.size(),
                "rescaled GPU postprocess returned the wrong batch size");
    if (scaled.size() != baseline.size())
        return false;

    for (size_t i = 0; i < baseline.size(); ++i) {
        auto line_scale = torch::tensor(
                              {scale_w[i], scale_h[i]},
                              baseline[i].lines.options().dtype(torch::kFloat32))
                              .view({1, 1, 2});
        auto covariance_scale = torch::tensor(
            {scale_w[i], scale_h[i], scale_w[i], scale_h[i]},
            baseline[i]
                .line_endpoint_covariances.options()
                .dtype(torch::kFloat32));
        auto expected_covariances =
            baseline[i].line_endpoint_covariances *
            covariance_scale.view({1, 4, 1}) *
            covariance_scale.view({1, 1, 4});
        ok &= Check(TensorClose(scaled[i].lines,
                                (baseline[i].lines + 0.5f) * line_scale - 0.5f,
                                5e-4f),
                    "GPU line pixel-center restoration is incorrect");
        ok &= Check(TensorClose(
                        scaled[i].keypoints,
                        (baseline[i].keypoints + 0.5f) *
                                line_scale.view({1, 2}) -
                            0.5f,
                        5e-4f),
                    "GPU point pixel-center restoration is incorrect");
        ok &= Check(TensorClose(scaled[i].line_endpoint_covariances,
                                expected_covariances, 1e-3f),
                    "GPU covariance resize scaling is incorrect");
    }
    return ok;
}

} // namespace

int main(int argc, char **argv) {
    using XPLSLAM::XPLNetTestAccess;

    bool ok = true;
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
    const auto long_options = torch::TensorOptions().dtype(torch::kLong);

    // Original 16x24 -> network 8x8, align_corners(false): network centers
    // (0,0) and (3,5) sample original centers (0.5,1) and (6.5,16).
    auto resize_points =
        torch::tensor({0.0f, 0.0f, 3.0f, 5.0f}, float_options).view({1, 2, 2});
    auto resize_scale =
        torch::tensor({2.0f, 3.0f}, float_options).view({1, 1, 2});
    auto expected_points =
        torch::tensor({0.5f, 1.0f, 6.5f, 16.0f}, float_options).view({1, 2, 2});
    ok &= Check(TensorClose(XPLNetTestAccess::RestoreImageCoordinates(
                                resize_points, resize_scale),
                            expected_points),
                "resize inversion lost the half-pixel offset");
    ok &= Check(torch::equal(XPLNetTestAccess::RestoreImageCoordinates(
                                expected_points, torch::ones_like(resize_scale)),
                            expected_points),
                "identity resizing changed sub-pixel coordinates");
    ok &= Check(XPLNetTestAccess::RestoreImageCoordinates(
                    torch::empty({0, 2, 2}, float_options), resize_scale)
                        .numel() == 0,
                "empty line coordinate restoration failed");

    auto logits = torch::full({1, 1, 9, 9},
                              -std::numeric_limits<float>::infinity(),
                              float_options);
    logits.index_put_({0, 0, 4, 4}, 0.0f);
    logits.index_put_({0, 0, 4, 5}, 0.0f);
    auto original_logits = logits.clone();
    auto peaks = torch::tensor({4.0f, 4.0f}, float_options).view({1, 1, 2});
    auto valid = torch::ones({1, 1}, torch::kBool);

    auto refined_result = XPLNetTestAccess::RefineJunctionsAndCovariances(
        logits, peaks, valid);
    auto refined = refined_result.first;
    auto covariance = refined_result.second;
    auto expected_refined =
        torch::tensor({4.5f, 4.0f}, float_options).view({1, 1, 2});
    auto expected_covariance =
        torch::tensor({0.2501f, 0.0f, 0.0f, 0.0001f}, float_options)
            .view({1, 1, 2, 2});
    ok &= Check(TensorClose(refined, expected_refined),
                "soft-argmax did not produce the expected sub-pixel point");
    ok &= Check(TensorClose(covariance, expected_covariance),
                "local moments did not produce the expected covariance");
    ok &= Check(torch::isfinite(covariance).all().item<bool>(),
                "interior covariance contains a non-finite value");
    ok &= Check(torch::equal(logits, original_logits),
                "raw junction logits were modified during refinement");

    auto border_logits = torch::zeros({1, 1, 5, 5}, float_options);
    auto border_peaks = torch::zeros({1, 2, 2}, float_options);
    auto border_valid = torch::tensor({true, false}, torch::kBool).view({1, 2});
    auto border_result = XPLNetTestAccess::RefineJunctionsAndCovariances(
        border_logits, border_peaks, border_valid);
    auto expected_border_point = torch::tensor({1.5f, 1.5f}, float_options);
    auto expected_border_covariance =
        torch::tensor({1.2501f, 0.0f, 0.0f, 1.2501f}, float_options)
            .view({2, 2});
    ok &= Check(TensorClose(border_result.first.index({0, 0}),
                            expected_border_point),
                "border soft-argmax did not renormalize the truncated window");
    ok &= Check(TensorClose(border_result.second.index({0, 0}),
                            expected_border_covariance),
                "border covariance is incorrect");
    ok &= Check(TensorClose(border_result.first.index({0, 1}), torch::zeros({2})),
                "invalid junction coordinate was not zeroed");
    ok &= Check(TensorClose(border_result.second.index({0, 1}),
                            torch::zeros({2, 2})),
                "invalid junction covariance was not zeroed");

    auto covariance_2d =
        torch::tensor({2.0f, 0.5f, 0.5f, 3.0f}, float_options)
            .view({1, 2, 2});
    auto scale_2d = torch::tensor({2.0f, 3.0f}, float_options);
    auto scaled_2d =
        XPLNetTestAccess::ScaleCovariances(covariance_2d, scale_2d);
    auto expected_scaled_2d =
        torch::tensor({8.0f, 3.0f, 3.0f, 27.0f}, float_options)
            .view({1, 2, 2});
    ok &= Check(TensorClose(scaled_2d, expected_scaled_2d),
                "2D covariance scaling is incorrect");

    auto junction_covariances = torch::tensor(
        {2.0f, 0.2f, 0.2f, 3.0f, 4.0f, -0.1f, -0.1f, 5.0f},
        float_options);
    junction_covariances = junction_covariances.view({1, 2, 2, 2});
    auto batch_indices = torch::zeros({1}, long_options);
    auto start_indices = torch::zeros({1}, long_options);
    auto end_indices = torch::ones({1}, long_options);
    auto line_covariance = XPLNetTestAccess::AssembleLineEndpointCovariances(
        junction_covariances, batch_indices, start_indices, end_indices);
    ok &= Check(line_covariance.dim() == 3 && line_covariance.size(0) == 1 &&
                    line_covariance.size(1) == 4 &&
                    line_covariance.size(2) == 4,
                "line endpoint covariance does not have shape [1, 4, 4]");
    ok &= Check(TensorClose(line_covariance.index({0, torch::indexing::Slice(0, 2),
                                                   torch::indexing::Slice(0, 2)}),
                            junction_covariances.index({0, 0})),
                "first endpoint covariance block is incorrect");
    ok &= Check(TensorClose(line_covariance.index({0, torch::indexing::Slice(2, 4),
                                                   torch::indexing::Slice(2, 4)}),
                            junction_covariances.index({0, 1})),
                "second endpoint covariance block is incorrect");
    ok &= Check(TensorClose(line_covariance.index({0, torch::indexing::Slice(0, 2),
                                                   torch::indexing::Slice(2, 4)}),
                            torch::zeros({2, 2})),
                "endpoint cross-covariance block is not zero");

    auto scale_4d = torch::tensor({2.0f, 3.0f, 2.0f, 3.0f}, float_options);
    auto scaled_4d =
        XPLNetTestAccess::ScaleCovariances(line_covariance, scale_4d);
    auto expected_diagonal =
        torch::tensor({8.0f, 27.0f, 16.0f, 45.0f}, float_options);
    ok &= Check(TensorClose(scaled_4d.index({0}).diagonal(), expected_diagonal),
                "4D line endpoint covariance scaling is incorrect");

    auto empty_indices = torch::empty({0}, long_options);
    auto empty_covariances = XPLNetTestAccess::AssembleLineEndpointCovariances(
        junction_covariances, empty_indices, empty_indices, empty_indices);
    ok &= Check(empty_covariances.dim() == 3 &&
                    empty_covariances.size(0) == 0 &&
                    empty_covariances.size(1) == 4 &&
                    empty_covariances.size(2) == 4,
                "empty line covariance does not have shape [0, 4, 4]");

    if (argc == 3 && std::string(argv[1]) == "--gpu") {
        ok &= RunGpuCheck(argv[2]);
    } else if (argc != 1) {
        std::cerr << "Usage: " << argv[0] << " [--gpu CONFIG]" << std::endl;
        return 2;
    }

    return ok ? 0 : 1;
}
