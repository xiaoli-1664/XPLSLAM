#include "runtime_csv.h"
#include "system.h"

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/imgcodecs.hpp>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR "."
#endif

namespace fs = std::filesystem;

namespace {

constexpr size_t kCameraCount = 4;
constexpr char kStreamMagic[] = "XPLNCB01";
constexpr std::uint32_t kMaxCompressedImageBytes = 64U * 1024U * 1024U;

struct Options {
    fs::path bag = "";
    fs::path settings = fs::path(PROJECT_SOURCE_DIR) / "config/new_college.yaml";
    fs::path point_vocab =
        fs::path(PROJECT_SOURCE_DIR) / "vocabulary/point_voc.dbow3bin";
    fs::path line_vocab =
        fs::path(PROJECT_SOURCE_DIR) / "vocabulary/line_voc.dbow3bin";
    fs::path output_dir;
    fs::path stream_helper =
        fs::path(PROJECT_SOURCE_DIR) / "examples/new_college_bag_stream.py";
    std::string python = "python3";
    int start = 0;
    int max_frames = 0;
    int step = 1;
    int log_every = 10;
    std::int64_t sync_tolerance_ns = 20000;
    bool use_viewer = false;
    bool check_data_only = false;
    bool realtime = false;
    bool runtime_timing = false;
};

struct CameraInput {
    std::string topic;
    int width = 0;
    int height = 0;
};

struct StreamFrame {
    std::int64_t timestamp_ns = 0;
    std::vector<std::vector<std::uint8_t>> encoded_images;
    std::vector<cv::Mat> images;
};

std::string ReadOptionValue(const std::string &arg, int &index, int argc,
                            char **argv) {
    const size_t equals = arg.find('=');
    if (equals != std::string::npos) {
        return arg.substr(equals + 1);
    }
    if (index + 1 >= argc) {
        throw std::runtime_error("Missing value for option: " + arg);
    }
    return argv[++index];
}

void PrintUsage(const char *program) {
    std::cout
        << "Usage: " << program << " --bag PATH [options]\n"
        << "Options:\n"
        << "  --bag PATH           NewCollege ROS1 bag (required)\n"
        << "  --settings PATH      XPLSLAM NewCollege settings YAML\n"
        << "  --point-vocab PATH   Point DBoW3 vocabulary\n"
        << "  --line-vocab PATH    Line DBoW3 vocabulary\n"
        << "  --output-dir PATH    Output directory\n"
        << "  --stream-helper PATH Python rosbags stream helper\n"
        << "  --python PATH        Python interpreter (default: python3)\n"
        << "  --start N            First synchronized frame index\n"
        << "  --start-frame N      First synchronized frame number (1-based)\n"
        << "  --max-frames N       Number of selected frames, 0 means all\n"
        << "  --step N             Process every Nth synchronized frame\n"
        << "  --sync-tolerance-ns N  Maximum four-camera timestamp span\n"
        << "  --log-every N        Progress print interval\n"
        << "  --viewer             Enable visualization\n"
        << "  --no-viewer          Disable visualization (default)\n"
        << "  --runtime-timing     Write buffered runtime CSV/JSON logs\n"
        << "  --check-data-only    Decode frames without starting XPLSLAM\n"
        << "  --realtime           Pace input according to dataset timestamps\n"
        << "  --no-realtime        Process as fast as possible\n";
}

Options ParseArgs(int argc, char **argv) {
    Options options;
    bool output_explicit = false;
    bool realtime_explicit = false;

    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else if (arg == "--bag" || arg.rfind("--bag=", 0) == 0) {
            options.bag = ReadOptionValue(arg, index, argc, argv);
        } else if (arg == "--settings" ||
                   arg.rfind("--settings=", 0) == 0) {
            options.settings = ReadOptionValue(arg, index, argc, argv);
        } else if (arg == "--point-vocab" ||
                   arg.rfind("--point-vocab=", 0) == 0) {
            options.point_vocab = ReadOptionValue(arg, index, argc, argv);
        } else if (arg == "--line-vocab" ||
                   arg.rfind("--line-vocab=", 0) == 0) {
            options.line_vocab = ReadOptionValue(arg, index, argc, argv);
        } else if (arg == "--output-dir" ||
                   arg.rfind("--output-dir=", 0) == 0) {
            options.output_dir = ReadOptionValue(arg, index, argc, argv);
            output_explicit = true;
        } else if (arg == "--stream-helper" ||
                   arg.rfind("--stream-helper=", 0) == 0) {
            options.stream_helper = ReadOptionValue(arg, index, argc, argv);
        } else if (arg == "--python" ||
                   arg.rfind("--python=", 0) == 0) {
            options.python = ReadOptionValue(arg, index, argc, argv);
        } else if (arg == "--start" || arg.rfind("--start=", 0) == 0) {
            options.start = std::stoi(ReadOptionValue(arg, index, argc, argv));
        } else if (arg == "--start-frame" ||
                   arg.rfind("--start-frame=", 0) == 0) {
            const int frame = std::stoi(ReadOptionValue(arg, index, argc, argv));
            if (frame <= 0) {
                throw std::runtime_error("--start-frame must be >= 1");
            }
            options.start = frame - 1;
        } else if (arg == "--max-frames" ||
                   arg.rfind("--max-frames=", 0) == 0) {
            options.max_frames =
                std::stoi(ReadOptionValue(arg, index, argc, argv));
        } else if (arg == "--step" || arg.rfind("--step=", 0) == 0) {
            options.step = std::stoi(ReadOptionValue(arg, index, argc, argv));
        } else if (arg == "--sync-tolerance-ns" ||
                   arg.rfind("--sync-tolerance-ns=", 0) == 0) {
            options.sync_tolerance_ns =
                std::stoll(ReadOptionValue(arg, index, argc, argv));
        } else if (arg == "--log-every" ||
                   arg.rfind("--log-every=", 0) == 0) {
            options.log_every =
                std::stoi(ReadOptionValue(arg, index, argc, argv));
        } else if (arg == "--viewer") {
            options.use_viewer = true;
        } else if (arg == "--no-viewer") {
            options.use_viewer = false;
        } else if (arg == "--runtime-timing") {
            options.runtime_timing = true;
        } else if (arg == "--check-data-only") {
            options.check_data_only = true;
        } else if (arg == "--realtime") {
            options.realtime = true;
            realtime_explicit = true;
        } else if (arg == "--no-realtime") {
            options.realtime = false;
            realtime_explicit = true;
        } else {
            throw std::runtime_error("Unknown option: " + arg);
        }
    }

    if (options.bag.empty()) {
        throw std::runtime_error("--bag is required; use --help for usage");
    }
    if (!output_explicit) {
        options.output_dir = fs::path(PROJECT_SOURCE_DIR) / "output/new_college" /
                             options.bag.stem();
    }
    if (options.start < 0 || options.max_frames < 0 || options.step <= 0 ||
        options.log_every <= 0 || options.sync_tolerance_ns < 0) {
        throw std::runtime_error(
            "start/max-frames must be non-negative; step/log-every must be "
            "positive; sync-tolerance-ns must be non-negative");
    }
    if (options.use_viewer && !realtime_explicit) {
        options.realtime = true;
    }
    return options;
}

void RequireFile(const fs::path &path, const std::string &description) {
    if (!fs::is_regular_file(path)) {
        throw std::runtime_error(description + " does not exist: " +
                                 path.string());
    }
}

std::vector<CameraInput> ReadCameraInputs(const fs::path &settings_path) {
    cv::FileStorage storage(settings_path.string(), cv::FileStorage::READ);
    if (!storage.isOpened()) {
        throw std::runtime_error("Failed to open settings: " +
                                 settings_path.string());
    }

    const int camera_count = static_cast<int>(storage["Camera.nCams"]);
    if (camera_count != static_cast<int>(kCameraCount)) {
        throw std::runtime_error("NewCollege example requires exactly four cameras");
    }

    std::vector<CameraInput> cameras(kCameraCount);
    std::set<std::string> topics;
    for (size_t camera = 0; camera < kCameraCount; ++camera) {
        const std::string prefix = "Camera" + std::to_string(camera);
        const cv::FileNode topic_node = storage[prefix + ".topic"];
        if (topic_node.empty() || !topic_node.isString()) {
            throw std::runtime_error(prefix + ".topic must be a string");
        }
        cameras[camera].topic = static_cast<std::string>(topic_node);
        if (cameras[camera].topic.empty() ||
            !topics.insert(cameras[camera].topic).second) {
            throw std::runtime_error("Camera topics must be nonempty and unique");
        }
        cameras[camera].width =
            static_cast<int>(storage[prefix + ".width"]);
        cameras[camera].height =
            static_cast<int>(storage[prefix + ".height"]);
        if (cameras[camera].width <= 0 || cameras[camera].height <= 0) {
            throw std::runtime_error(prefix + " has invalid image dimensions");
        }
    }
    return cameras;
}

class BagStreamProcess {
  public:
    BagStreamProcess(const Options &options,
                     const std::vector<CameraInput> &cameras) {
        int pipe_fds[2];
        if (pipe(pipe_fds) != 0) {
            throw std::runtime_error("pipe failed: " +
                                     std::string(std::strerror(errno)));
        }

        std::vector<std::string> arguments = {
            options.python,
            options.stream_helper.string(),
            "--bag",
            options.bag.string(),
            "--start",
            std::to_string(options.start),
            "--max-frames",
            std::to_string(options.max_frames),
            "--step",
            std::to_string(options.step),
            "--sync-tolerance-ns",
            std::to_string(options.sync_tolerance_ns),
        };
        for (const CameraInput &camera : cameras) {
            arguments.push_back("--topic");
            arguments.push_back(camera.topic);
        }

        child_pid_ = fork();
        if (child_pid_ < 0) {
            close(pipe_fds[0]);
            close(pipe_fds[1]);
            throw std::runtime_error("fork failed: " +
                                     std::string(std::strerror(errno)));
        }
        if (child_pid_ == 0) {
            close(pipe_fds[0]);
            if (dup2(pipe_fds[1], STDOUT_FILENO) < 0) {
                std::perror("dup2");
                _exit(127);
            }
            close(pipe_fds[1]);

            std::vector<char *> argv;
            argv.reserve(arguments.size() + 1);
            for (std::string &argument : arguments) {
                argv.push_back(argument.data());
            }
            argv.push_back(nullptr);
            execvp(argv[0], argv.data());
            std::perror("execvp");
            _exit(127);
        }

        close(pipe_fds[1]);
        read_fd_ = pipe_fds[0];
    }

    BagStreamProcess(const BagStreamProcess &) = delete;
    BagStreamProcess &operator=(const BagStreamProcess &) = delete;

    ~BagStreamProcess() {
        if (read_fd_ >= 0) {
            close(read_fd_);
        }
        if (child_pid_ > 0 && !waited_) {
            kill(child_pid_, SIGTERM);
            int status = 0;
            while (waitpid(child_pid_, &status, 0) < 0 && errno == EINTR) {
            }
        }
    }

    int ReadFd() const { return read_fd_; }

    void Finish() {
        if (read_fd_ >= 0) {
            close(read_fd_);
            read_fd_ = -1;
        }

        int status = 0;
        pid_t result;
        do {
            result = waitpid(child_pid_, &status, 0);
        } while (result < 0 && errno == EINTR);
        waited_ = true;
        if (result < 0) {
            throw std::runtime_error("waitpid failed: " +
                                     std::string(std::strerror(errno)));
        }
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            throw std::runtime_error("NewCollege bag stream helper failed");
        }
    }

  private:
    pid_t child_pid_ = -1;
    int read_fd_ = -1;
    bool waited_ = false;
};

bool ReadExact(int fd, void *destination, size_t size, bool allow_clean_eof) {
    size_t total = 0;
    auto *bytes = static_cast<std::uint8_t *>(destination);
    while (total < size) {
        const ssize_t result = read(fd, bytes + total, size - total);
        if (result > 0) {
            total += static_cast<size_t>(result);
            continue;
        }
        if (result == 0) {
            if (allow_clean_eof && total == 0) {
                return false;
            }
            throw std::runtime_error("Unexpected end of NewCollege frame stream");
        }
        if (errno == EINTR) {
            continue;
        }
        throw std::runtime_error("Failed reading NewCollege frame stream: " +
                                 std::string(std::strerror(errno)));
    }
    return true;
}

std::uint32_t DecodeU32(const std::uint8_t *bytes) {
    return static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8) |
           (static_cast<std::uint32_t>(bytes[2]) << 16) |
           (static_cast<std::uint32_t>(bytes[3]) << 24);
}

std::uint64_t DecodeU64(const std::uint8_t *bytes) {
    std::uint64_t value = 0;
    for (int index = 7; index >= 0; --index) {
        value = (value << 8) | bytes[index];
    }
    return value;
}

std::uint32_t ReadU32(int fd) {
    std::uint8_t bytes[4];
    ReadExact(fd, bytes, sizeof(bytes), false);
    return DecodeU32(bytes);
}

void ValidateStreamHeader(int fd) {
    char magic[sizeof(kStreamMagic) - 1];
    ReadExact(fd, magic, sizeof(magic), false);
    if (std::memcmp(magic, kStreamMagic, sizeof(magic)) != 0) {
        throw std::runtime_error("Invalid NewCollege frame stream magic");
    }
    if (ReadU32(fd) != kCameraCount) {
        throw std::runtime_error("NewCollege stream camera count mismatch");
    }
}

bool ReadFrame(int fd, const std::vector<CameraInput> &cameras,
               StreamFrame &frame) {
    std::uint8_t timestamp_bytes[8];
    if (!ReadExact(fd, timestamp_bytes, sizeof(timestamp_bytes), true)) {
        return false;
    }
    frame.timestamp_ns =
        static_cast<std::int64_t>(DecodeU64(timestamp_bytes));
    frame.encoded_images.resize(kCameraCount);
    frame.images.resize(kCameraCount);

    for (size_t camera = 0; camera < kCameraCount; ++camera) {
        const std::uint32_t payload_size = ReadU32(fd);
        if (payload_size == 0 || payload_size > kMaxCompressedImageBytes) {
            throw std::runtime_error("Camera" + std::to_string(camera) +
                                     " has invalid compressed payload size");
        }

        std::vector<std::uint8_t> &buffer = frame.encoded_images[camera];
        buffer.resize(payload_size);
        ReadExact(fd, buffer.data(), payload_size, false);
        frame.images[camera] = cv::imdecode(buffer, cv::IMREAD_GRAYSCALE);

        const CameraInput &expected = cameras[camera];
        const cv::Mat &image = frame.images[camera];
        if (image.empty() || image.type() != CV_8UC1 ||
            image.cols != expected.width || image.rows != expected.height) {
            throw std::runtime_error("Camera" + std::to_string(camera) +
                                     " decoded image geometry/type mismatch");
        }
    }
    return true;
}

int RunDataCheck(BagStreamProcess &stream,
                 const std::vector<CameraInput> &cameras) {
    StreamFrame frame;
    int frame_count = 0;
    std::int64_t first_timestamp = 0;
    std::int64_t last_timestamp = 0;
    while (ReadFrame(stream.ReadFd(), cameras, frame)) {
        if (frame_count == 0) {
            first_timestamp = frame.timestamp_ns;
        }
        last_timestamp = frame.timestamp_ns;
        ++frame_count;
    }
    stream.Finish();
    if (frame_count == 0) {
        throw std::runtime_error("No synchronized NewCollege frames were streamed");
    }
    std::cout << "Data check passed: frames=" << frame_count
              << ", first_timestamp_ns=" << first_timestamp
              << ", last_timestamp_ns=" << last_timestamp << "\n";
    return EXIT_SUCCESS;
}

int RunSlam(const Options &options, BagStreamProcess &stream,
            const std::vector<CameraInput> &cameras) {
    fs::create_directories(options.output_dir);
    const std::string sequence = options.bag.stem().string();
    const fs::path trajectory_path = options.output_dir / "trajectory_tum.txt";

    std::cout << "Loading XPLSLAM for NewCollege\n"
              << "  bag:      " << options.bag << "\n"
              << "  settings: " << options.settings << "\n"
              << "  output:   " << trajectory_path << "\n"
              << "  viewer:   "
              << (options.use_viewer ? "enabled" : "disabled") << "\n"
              << "  playback: "
              << (options.realtime ? "dataset time" : "as fast as possible")
              << "\n";

    StreamFrame frame;
    const auto first_read_begin = std::chrono::steady_clock::now();
    if (!ReadFrame(stream.ReadFd(), cameras, frame)) {
        stream.Finish();
        throw std::runtime_error("No synchronized NewCollege frames were streamed");
    }
    double read_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - first_read_begin).count();
    XPLSLAM::RuntimeCsv runtime_csv(options.runtime_timing ? options.output_dir
                                                         : fs::path());
    const auto startup_begin = std::chrono::steady_clock::now();
    XPLSLAM::System slam(options.point_vocab.string(),
                         options.line_vocab.string(), options.settings.string(),
                         options.use_viewer, 0, "new_college_" + sequence);

    const double startup_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - startup_begin).count();
    int processed = 0;
    std::int64_t first_timestamp_ns = 0;
    std::chrono::steady_clock::time_point playback_begin;
    const auto run_begin = std::chrono::steady_clock::now();

    do {
        const auto wait_begin = std::chrono::steady_clock::now();
        if (processed == 0) {
            first_timestamp_ns = frame.timestamp_ns;
            playback_begin = std::chrono::steady_clock::now();
        }
        if (options.realtime) {
            const double offset_seconds =
                static_cast<double>(frame.timestamp_ns - first_timestamp_ns) *
                1e-9;
            std::this_thread::sleep_until(
                playback_begin + std::chrono::duration_cast<
                                     std::chrono::steady_clock::duration>(
                                     std::chrono::duration<double>(
                                         offset_seconds)));
        }

        const double wait_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wait_begin).count();
        const double timestamp = static_cast<double>(frame.timestamp_ns) * 1e-9;
        XPLSLAM::RuntimeFrame runtime_frame;
        XPLSLAM::RuntimeCapture capture(options.runtime_timing ? &runtime_frame : nullptr);
        const auto frame_begin = std::chrono::steady_clock::now();
        capture.Begin(frame_begin);
        slam.TrackMultiCamera(frame.images, timestamp,
                              std::to_string(frame.timestamp_ns));
        const auto frame_end = std::chrono::steady_clock::now();
        capture.End(frame_end);
        const double track_ms = std::chrono::duration<double, std::milli>(
            frame_end - frame_begin).count();
        runtime_csv.Append(options.start + static_cast<size_t>(processed) * options.step,
                           timestamp, runtime_frame, track_ms, read_ms, wait_ms);
        ++processed;

        if (processed == 1 || processed % options.log_every == 0) {
            const double milliseconds =
                std::chrono::duration<double, std::milli>(frame_end -
                                                           frame_begin)
                    .count();
            std::cout << "Processed " << processed << " frames, timestamp "
                      << std::fixed << std::setprecision(6) << timestamp
                      << ", track=" << std::setprecision(2) << milliseconds
                      << " ms\n";
        }
        const auto read_begin = std::chrono::steady_clock::now();
        const bool has_next = ReadFrame(stream.ReadFd(), cameras, frame);
        read_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - read_begin).count();
        if (!has_next) break;
    } while (true);
    stream.Finish();

    const double run_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      run_begin)
            .count();
    const auto shutdown_begin = std::chrono::steady_clock::now();
    slam.Shutdown();
    const auto shutdown_end = std::chrono::steady_clock::now();
    slam.SaveTrajectoryTUM(trajectory_path.string());
    const auto save_end = std::chrono::steady_clock::now();
    runtime_csv.Finish(processed, processed, startup_s, run_seconds,
        std::chrono::duration<double>(shutdown_end - shutdown_begin).count(),
        std::chrono::duration<double>(save_end - shutdown_end).count());

    std::cout << "NewCollege run complete: frames=" << processed
              << ", elapsed_s=" << std::fixed
              << std::setprecision(2) << run_seconds
              << "\nTrajectory: " << trajectory_path << "\n";
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Options options = ParseArgs(argc, argv);
        RequireFile(options.bag, "NewCollege bag");
        RequireFile(options.settings, "Settings file");
        RequireFile(options.stream_helper, "NewCollege stream helper");
        if (!options.check_data_only) {
            RequireFile(options.point_vocab, "Point vocabulary");
            RequireFile(options.line_vocab, "Line vocabulary");
        }

        const std::vector<CameraInput> cameras =
            ReadCameraInputs(options.settings);
        BagStreamProcess stream(options, cameras);
        ValidateStreamHeader(stream.ReadFd());

        if (options.check_data_only) {
            return RunDataCheck(stream, cameras);
        }
        return RunSlam(options, stream, cameras);
    } catch (const std::exception &error) {
        std::cerr << "NewCollege example failed: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
