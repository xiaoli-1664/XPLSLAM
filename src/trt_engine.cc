#include "trt_engine.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>

namespace XPLSLAM {

// ---- Logger ----
void TrtLogger::log(Severity severity, const char *msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cerr << "[TRT] " << msg << std::endl;
    }
}

// ---- BindingBuffer helpers ----
int64_t BindingBuffer::max_volume() const {
    return std::accumulate(max_shape.begin(), max_shape.end(), int64_t(1),
                           std::multiplies<int64_t>());
}

int64_t BindingBuffer::actual_volume() const {
    return std::accumulate(actual_shape.begin(), actual_shape.end(), int64_t(1),
                           std::multiplies<int64_t>());
}

torch::Tensor BindingBuffer::view_actual() const {
    return buffer.view({-1}).slice(0, 0, actual_volume()).view(actual_shape);
}

// ---- Dtype conversion ----
torch::ScalarType
TrtEngine::trt_to_torch_dtype(nvinfer1::DataType dtype) const {
    switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
        return torch::kFloat32;
    case nvinfer1::DataType::kHALF:
        return torch::kFloat16;
    case nvinfer1::DataType::kINT32:
        return torch::kInt32;
    case nvinfer1::DataType::kINT8:
        return torch::kInt8;
    default:
        return torch::kFloat32;
    }
}

// ---- Destructor ----
TrtEngine::~TrtEngine() {}

// ---- Build engine from ONNX ----
std::vector<char>
TrtEngine::build_engine_from_onnx(const std::string &onnx_path,
                                  const TrtBuildConfig &config) {
    // Builder
    TrtUniquePtr<nvinfer1::IBuilder> builder(
        nvinfer1::createInferBuilder(logger_));
    if (!builder) {
        throw std::runtime_error("Failed to create TRT builder");
    }

    // Network
    TrtUniquePtr<nvinfer1::INetworkDefinition> network(
        builder->createNetworkV2(0));
    if (!network) {
        throw std::runtime_error("Failed to create TRT network");
    }

    // ONNX parser
    TrtUniquePtr<nvonnxparser::IParser> parser(
        nvonnxparser::createParser(*network, logger_));
    if (!parser) {
        throw std::runtime_error("Failed to create ONNX parser");
    }

    if (!parser->parseFromFile(
            onnx_path.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::string errs;
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            errs += std::string(parser->getError(i)->desc()) + "\n";
        }
        throw std::runtime_error("Failed to parse ONNX model: " + errs);
    }
    std::cout << "[TRT] Parsed ONNX: " << onnx_path
              << "  inputs=" << network->getNbInputs()
              << " outputs=" << network->getNbOutputs() << std::endl;

    // Builder config
    TrtUniquePtr<nvinfer1::IBuilderConfig> builder_config(
        builder->createBuilderConfig());
    if (!builder_config) {
        throw std::runtime_error("Failed to create builder config");
    }

    builder_config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                                       config.max_workspace_size);

    if (config.fp16) {
        builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "[TRT] FP16 enabled" << std::endl;
    }
    if (config.int8) {
        builder_config->setFlag(nvinfer1::BuilderFlag::kINT8);
        std::cout << "[TRT] INT8 enabled" << std::endl;
    }

    // DLA
    if (config.dla_core >= 0) {
        if (config.dla_core < builder->getNbDLACores()) {
            builder_config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
            builder_config->setDLACore(config.dla_core);
            builder_config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
            std::cout << "[TRT] Using DLA core " << config.dla_core
                      << std::endl;
        } else {
            std::cerr << "[TRT] Requested DLA core " << config.dla_core
                      << " not available (max " << builder->getNbDLACores() - 1
                      << ")" << std::endl;
        }
    }

    // Optimization profiles
    if (!config.profiles.empty()) {
        nvinfer1::IOptimizationProfile *profile =
            builder->createOptimizationProfile();
        for (const auto &[name, dim] : config.profiles) {
            nvinfer1::Dims min_dims, opt_dims, max_dims;
            min_dims.nbDims = static_cast<int>(dim.min.size());
            opt_dims.nbDims = static_cast<int>(dim.opt.size());
            max_dims.nbDims = static_cast<int>(dim.max.size());
            for (int d = 0; d < min_dims.nbDims; ++d) {
                min_dims.d[d] = dim.min[d];
            }
            for (int d = 0; d < opt_dims.nbDims; ++d) {
                opt_dims.d[d] = dim.opt[d];
            }
            for (int d = 0; d < max_dims.nbDims; ++d) {
                max_dims.d[d] = dim.max[d];
            }
            profile->setDimensions(
                name.c_str(), nvinfer1::OptProfileSelector::kMIN, min_dims);
            profile->setDimensions(
                name.c_str(), nvinfer1::OptProfileSelector::kOPT, opt_dims);
            profile->setDimensions(
                name.c_str(), nvinfer1::OptProfileSelector::kMAX, max_dims);
            std::cout << "[TRT] Profile '" << name << "' min=[";
            for (size_t i = 0; i < dim.min.size(); ++i)
                std::cout << dim.min[i] << (i + 1 < dim.min.size() ? "," : "");
            std::cout << "] opt=[";
            for (size_t i = 0; i < dim.opt.size(); ++i)
                std::cout << dim.opt[i] << (i + 1 < dim.opt.size() ? "," : "");
            std::cout << "] max=[";
            for (size_t i = 0; i < dim.max.size(); ++i)
                std::cout << dim.max[i] << (i + 1 < dim.max.size() ? "," : "");
            std::cout << "]" << std::endl;
        }
        builder_config->addOptimizationProfile(profile);
    }

    // Build serialised engine
    std::cout << "[TRT] Building engine (this may take a while)..."
              << std::endl;
    TrtUniquePtr<nvinfer1::IHostMemory> plan(
        builder->buildSerializedNetwork(*network, *builder_config));
    if (!plan) {
        throw std::runtime_error("Failed to build TRT engine from ONNX");
    }

    std::vector<char> blob(static_cast<const char *>(plan->data()),
                           static_cast<const char *>(plan->data()) +
                               plan->size());
    std::cout << "[TRT] Engine built successfully ("
              << blob.size() / (1024.0 * 1024.0) << " MB)" << std::endl;
    return blob;
}

// ---- Deserialize plan blob ----
bool TrtEngine::deserialize_plan(const std::vector<char> &plan) {
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(plan.data(), plan.size()));
    if (!engine_) {
        std::cerr << "Failed to deserialize engine" << std::endl;
        return false;
    }
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        std::cerr << "Failed to create execution context" << std::endl;
        return false;
    }
    return true;
}

// ---- Build from ONNX (main entry) ----
bool TrtEngine::build(const std::string &onnx_path,
                      const TrtBuildConfig &config, int device_id) {
    device_id_ = device_id;
    cudaSetDevice(device_id_);

    std::vector<char> plan;

    // Try loading cached engine
    if (!config.engine_cache_path.empty()) {
        std::ifstream cache(config.engine_cache_path,
                            std::ios::binary | std::ios::ate);
        if (cache.is_open()) {
            auto size = cache.tellg();
            cache.seekg(0, std::ios::beg);
            plan.resize(size);
            cache.read(plan.data(), size);
            cache.close();
            std::cout << "[TRT] Loaded cached engine: "
                      << config.engine_cache_path << std::endl;
        }
    }

    // Build if no cache
    if (plan.empty()) {
        plan = build_engine_from_onnx(onnx_path, config);

        // Save cache
        if (!config.engine_cache_path.empty()) {
            std::ofstream out(config.engine_cache_path, std::ios::binary);
            if (out.is_open()) {
                out.write(plan.data(), plan.size());
                out.close();
                std::cout << "[TRT] Cached engine to: "
                          << config.engine_cache_path << std::endl;
            }
        }
    }

    if (!deserialize_plan(plan)) {
        return false;
    }

    // Collect max shapes from profiles config for buffer allocation
    std::unordered_map<std::string, std::vector<int64_t>> max_shapes;
    for (const auto &[name, dim] : config.profiles) {
        max_shapes[name] = dim.max;
    }
    allocate_buffers(max_shapes);

    std::cout << "Loaded TRT engine from ONNX: " << onnx_path << " ("
              << bindings_.size() << " bindings, pre-allocated)" << std::endl;
    return true;
}

// ---- Load pre-built engine ----
bool TrtEngine::load(
    const std::string &engine_path, int device_id,
    const std::unordered_map<std::string, std::vector<int64_t>> &max_shapes) {
    device_id_ = device_id;
    cudaSetDevice(device_id_);

    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Cannot open engine file: " << engine_path << std::endl;
        return false;
    }
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> blob(size);
    file.read(blob.data(), size);
    file.close();

    if (!deserialize_plan(blob)) {
        return false;
    }

    allocate_buffers(max_shapes);

    std::cout << "Loaded TRT engine: " << engine_path << " ("
              << bindings_.size() << " bindings, pre-allocated)" << std::endl;
    return true;
}

// ---- Pre-allocate all buffers ----
void TrtEngine::allocate_buffers(
    const std::unordered_map<std::string, std::vector<int64_t>> &max_shapes) {
    bindings_.clear();
    int nb = engine_->getNbIOTensors();
    bool has_profile = (engine_->getNbOptimizationProfiles() > 0);

    // First pass: determine max shapes for all bindings
    for (int i = 0; i < nb; ++i) {
        const char *tensor_name = engine_->getIOTensorName(i);
        std::string name(tensor_name);

        BindingBuffer buf;
        buf.name = name;
        buf.is_input = (engine_->getTensorIOMode(tensor_name) ==
                        nvinfer1::TensorIOMode::kINPUT);
        buf.trt_dtype = engine_->getTensorDataType(tensor_name);

        if (buf.is_input) {
            // Determine max shape for inputs
            auto it = max_shapes.find(name);
            if (it != max_shapes.end()) {
                buf.max_shape = it->second;
            } else if (has_profile) {
                // Query max shape from optimization profile 0
                auto profile_dims = engine_->getProfileShape(
                    tensor_name, 0, nvinfer1::OptProfileSelector::kMAX);
                buf.max_shape.resize(profile_dims.nbDims);
                for (int d = 0; d < profile_dims.nbDims; ++d) {
                    buf.max_shape[d] = profile_dims.d[d];
                }
            } else {
                // Static input shape
                auto dims = engine_->getTensorShape(tensor_name);
                buf.max_shape.resize(dims.nbDims);
                for (int d = 0; d < dims.nbDims; ++d) {
                    buf.max_shape[d] = dims.d[d];
                }
            }
            buf.actual_shape = buf.max_shape;
        }
        // Output max_shape will be resolved in third pass after setting input

        bindings_[name] = std::move(buf);
    }

    // Second pass: set all input bindings to max shape in context,
    // then resolve true output shapes
    for (auto &[name, buf] : bindings_) {
        if (!buf.is_input)
            continue;
        nvinfer1::Dims dims;
        dims.nbDims = static_cast<int>(buf.max_shape.size());
        for (int d = 0; d < dims.nbDims; ++d) {
            dims.d[d] = buf.max_shape[d];
        }
        context_->setInputShape(name.c_str(), dims);
    }

    // Now query output max shapes from context
    for (auto &[name, buf] : bindings_) {
        if (buf.is_input)
            continue;
        auto dims = context_->getTensorShape(name.c_str());
        buf.max_shape.resize(dims.nbDims);
        for (int d = 0; d < dims.nbDims; ++d) {
            buf.max_shape[d] = dims.d[d];
        }
        buf.actual_shape = buf.max_shape;
    }

    // Allocate tensors and register addresses with context
    for (auto &[name, buf] : bindings_) {
        auto dtype = trt_to_torch_dtype(buf.trt_dtype);
        buf.buffer =
            torch::empty(buf.max_shape, torch::TensorOptions()
                                            .device(torch::kCUDA, device_id_)
                                            .dtype(dtype));

        // Register persistent address with the execution context
        context_->setTensorAddress(name.c_str(), buf.buffer.data_ptr());

        std::cout << "  [" << (buf.is_input ? "IN " : "OUT") << "] " << name
                  << " max_shape=[";
        for (size_t d = 0; d < buf.max_shape.size(); ++d) {
            std::cout << buf.max_shape[d]
                      << (d + 1 < buf.max_shape.size() ? "," : "");
        }
        std::cout << "] " << buf.buffer.nbytes() / (1024.0 * 1024.0) << " MB"
                  << std::endl;
    }
}

// ---- Set input shape (no realloc) ----
void TrtEngine::set_input_shape(const std::string &name,
                                const std::vector<int64_t> &shape) {
    auto it = bindings_.find(name);
    if (it == bindings_.end()) {
        throw std::runtime_error("Binding not found: " + name);
    }
    auto &buf = it->second;
    if (!buf.is_input) {
        throw std::runtime_error("Not an input binding: " + name);
    }

    if (shape.size() != buf.max_shape.size()) {
        throw std::runtime_error("Shape rank mismatch for " + name);
    }
    for (size_t d = 0; d < shape.size(); ++d) {
        if (shape[d] > buf.max_shape[d]) {
            throw std::runtime_error(
                "Shape dim " + std::to_string(d) + " (" +
                std::to_string(shape[d]) + ") exceeds max (" +
                std::to_string(buf.max_shape[d]) + ") for " + name);
        }
    }

    buf.actual_shape = shape;

    nvinfer1::Dims dims;
    dims.nbDims = static_cast<int>(shape.size());
    for (int d = 0; d < dims.nbDims; ++d) {
        dims.d[d] = shape[d];
    }
    context_->setInputShape(name.c_str(), dims);
}

// ---- Get pre-allocated input buffer ----
torch::Tensor TrtEngine::get_input_buffer(const std::string &name) {
    auto it = bindings_.find(name);
    if (it == bindings_.end()) {
        throw std::runtime_error("Binding not found: " + name);
    }
    return it->second.view_actual();
}

// ---- Copy external tensor into pre-allocated buffer ----
void TrtEngine::copy_input(const std::string &name, const torch::Tensor &src) {
    auto it = bindings_.find(name);
    if (it == bindings_.end()) {
        throw std::runtime_error("Binding not found: " + name);
    }
    auto &buf = it->second;

    std::vector<int64_t> shape(src.sizes().begin(), src.sizes().end());
    set_input_shape(name, shape);

    auto dst_view = buf.view_actual();
    auto src_contig = src.contiguous();
    if (src_contig.scalar_type() != dst_view.scalar_type()) {
        src_contig = src_contig.to(dst_view.scalar_type());
    }
    dst_view.copy_(src_contig);
}

// ---- Inference (zero allocation) ----
void TrtEngine::infer(cudaStream_t stream) {
    bool own_stream = (stream == nullptr);
    if (own_stream) {
        cudaStreamCreate(&stream);
    }

    // Update output actual shapes from context after input shapes are set
    for (auto &[name, buf] : bindings_) {
        if (buf.is_input)
            continue;
        auto dims = context_->getTensorShape(name.c_str());
        buf.actual_shape.resize(dims.nbDims);
        for (int d = 0; d < dims.nbDims; ++d) {
            buf.actual_shape[d] = dims.d[d];
        }
    }

    // Tensor addresses are already registered via setTensorAddress in
    // allocate_buffers. The pointers remain valid since we never reallocate.
    bool ok = context_->enqueueV3(stream);
    if (!ok) {
        throw std::runtime_error("TensorRT enqueueV3 failed");
    }

    if (own_stream) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
}

// ---- Get output (view, no copy) ----
torch::Tensor TrtEngine::get_output(const std::string &name) const {
    auto it = bindings_.find(name);
    if (it == bindings_.end()) {
        throw std::runtime_error("Output binding not found: " + name);
    }
    auto &buf = it->second;
    if (buf.is_input) {
        throw std::runtime_error("Not an output binding: " + name);
    }

    auto result = buf.view_actual();
    if (result.scalar_type() == torch::kFloat16) {
        result = result.to(torch::kFloat32);
    }
    return result;
}

torch::Tensor TrtEngine::get_raw_buffer(const std::string &name) const {
    auto it = bindings_.find(name);
    if (it == bindings_.end()) {
        throw std::runtime_error("Binding not found: " + name);
    }
    return it->second.buffer;
}

std::vector<std::string> TrtEngine::input_names() const {
    std::vector<std::string> names;
    for (auto &[name, buf] : bindings_) {
        if (buf.is_input)
            names.push_back(name);
    }
    return names;
}

std::vector<std::string> TrtEngine::output_names() const {
    std::vector<std::string> names;
    for (auto &[name, buf] : bindings_) {
        if (!buf.is_input)
            names.push_back(name);
    }
    return names;
}

} // namespace XPLSLAM
