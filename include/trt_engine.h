#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace XPLSLAM {

struct TrtDeleter {
    template <typename T> void operator()(T *p) const noexcept {
        if (p)
            delete p;
    }
};

template <typename T> using TrtUniquePtr = std::unique_ptr<T, TrtDeleter>;

class TrtLogger : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char *msg) noexcept override;
};

/// Per-dimension min/opt/max specification for dynamic shapes.
struct DimProfile {
    std::vector<int64_t> min;
    std::vector<int64_t> opt;
    std::vector<int64_t> max;
};

/// Builder configuration options.
struct TrtBuildConfig {
    /// Optimization profiles for dynamic-shape inputs.
    /// Key = input tensor name, Value = min/opt/max shapes.
    std::unordered_map<std::string, DimProfile> profiles;

    /// Enable FP16 inference.
    bool fp16 = false;

    /// Enable INT8 inference (requires calibrator if no QAT).
    bool int8 = false;

    /// Max workspace size in bytes (default 1 GiB).
    size_t max_workspace_size = 1ULL << 30;

    /// DLA core to use (-1 = no DLA).
    int dla_core = -1;

    /// If non-empty, cache the built engine to this path and reload on
    /// subsequent runs instead of rebuilding.
    std::string engine_cache_path;
};

/// Binding buffer: pre-allocated at max shape, carries actual shape at runtime.
struct BindingBuffer {
    std::string name;
    bool is_input = false;
    nvinfer1::DataType trt_dtype;

    // Max shape from optimization profile (used for allocation)
    std::vector<int64_t> max_shape;

    // Current actual shape (set before each inference)
    std::vector<int64_t> actual_shape;

    // Pre-allocated tensor at max_shape, lives on GPU for the entire lifetime
    torch::Tensor buffer;

    // Convenience: return a view/slice of buffer matching actual_shape
    torch::Tensor view_actual() const;

    int64_t max_volume() const;
    int64_t actual_volume() const;
};

class TrtEngine {
  public:
    TrtEngine() = default;
    ~TrtEngine();

    /// Build engine from ONNX model and pre-allocate all buffers.
    /// @param onnx_path   Path to the .onnx model file.
    /// @param config      Builder / profile configuration.
    /// @param device_id   CUDA device index.
    bool build(const std::string &onnx_path, const TrtBuildConfig &config = {},
               int device_id = 0);

    /// Load a previously serialised engine file directly (skip build).
    bool load(const std::string &engine_path, int device_id = 0,
              const std::unordered_map<std::string, std::vector<int64_t>>
                  &max_shapes = {});

    /// Set actual input shape for this inference call (no realloc).
    void set_input_shape(const std::string &name,
                         const std::vector<int64_t> &shape);

    /// Get the pre-allocated input buffer to write data into directly.
    torch::Tensor get_input_buffer(const std::string &name);

    /// Copy data into pre-allocated input buffer.
    void copy_input(const std::string &name, const torch::Tensor &src);

    /// Run inference on current buffers. No allocation.
    void infer(cudaStream_t stream = nullptr);

    /// Get output view matching actual output shape.
    torch::Tensor get_output(const std::string &name) const;

    /// Get raw pre-allocated buffer (full max-shape).
    torch::Tensor get_raw_buffer(const std::string &name) const;

    std::vector<std::string> input_names() const;
    std::vector<std::string> output_names() const;

  private:
    TrtLogger logger_;
    TrtUniquePtr<nvinfer1::IRuntime> runtime_;
    TrtUniquePtr<nvinfer1::ICudaEngine> engine_;
    TrtUniquePtr<nvinfer1::IExecutionContext> context_;
    int device_id_ = 0;

    std::unordered_map<std::string, BindingBuffer> bindings_;

    torch::ScalarType trt_to_torch_dtype(nvinfer1::DataType dtype) const;
    void
    allocate_buffers(const std::unordered_map<std::string, std::vector<int64_t>>
                         &max_shapes);

    /// Internal: build engine from ONNX, return serialised plan.
    std::vector<char> build_engine_from_onnx(const std::string &onnx_path,
                                             const TrtBuildConfig &config);

    /// Internal: deserialise plan blob into engine_ + context_.
    bool deserialize_plan(const std::vector<char> &plan);
};

} // namespace XPLSLAM
