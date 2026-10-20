#include "runtime_csv.h"
#include "system.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core/persistence.hpp>
#include <opencv2/imgcodecs.hpp>

#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR "."
#endif

namespace fs = std::filesystem;

namespace {

struct Options {
    fs::path dataset_root;
    fs::path settings_file =
        fs::path(PROJECT_SOURCE_DIR) / "config/new_college.yaml";
    fs::path point_vocab =
        fs::path(PROJECT_SOURCE_DIR) / "vocabulary/point_voc.dbow3bin";
    fs::path line_vocab =
        fs::path(PROJECT_SOURCE_DIR) / "vocabulary/line_voc.dbow3bin";
    fs::path timestamp_file;
    fs::path output_dir = fs::path(PROJECT_SOURCE_DIR) / "output/new_college";
    std::string sequence_name = "new_college";
    int start = 0;
    int max_frames = 0;
    int step = 1;
    int log_every = 10;
    bool use_viewer = false;
    bool wait_key = false;
    bool runtime_timing = false;
};

std::string ToLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

std::string Trim(const std::string &s) {
    const auto begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
}

std::vector<std::string> SplitComma(const std::string &line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        fields.push_back(Trim(field));
    }
    return fields;
}

bool LooksLikeHeader(const std::string &line) {
    return std::any_of(line.begin(), line.end(), [](unsigned char c) {
        return std::isalpha(c) || c == '_';
    });
}

bool IsImageFile(const fs::path &path) {
    const std::string ext = ToLower(path.extension().string());
    return ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" ||
           ext == ".tif" || ext == ".tiff";
}

std::vector<fs::path> CollectImages(const fs::path &dir) {
    if (!fs::is_directory(dir)) {
        throw std::runtime_error("Image directory does not exist: " +
                                 dir.string());
    }

    std::vector<fs::path> paths;
    for (const auto &entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file() && IsImageFile(entry.path())) {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

std::vector<double> LoadTimestamps(const fs::path &path,
                                   const std::string &column_name) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open timestamp file: " +
                                 path.string());
    }

    std::vector<double> timestamps;
    std::string line;
    int column = 0;
    bool first_data_line = true;

    while (std::getline(file, line)) {
        line = Trim(line);
        if (line.empty()) {
            continue;
        }

        if (first_data_line && LooksLikeHeader(line)) {
            const auto headers = SplitComma(line);
            for (int i = 0; i < static_cast<int>(headers.size()); ++i) {
                if (headers[i] == column_name) {
                    column = i;
                    break;
                }
            }
            first_data_line = false;
            continue;
        }

        const auto fields = SplitComma(line);
        if (column >= static_cast<int>(fields.size())) {
            throw std::runtime_error("Timestamp column out of range in: " +
                                     line);
        }
        timestamps.push_back(std::stod(fields[column]));
        first_data_line = false;
    }

    return timestamps;
}

std::string ReadOptionValue(const std::string &arg, int &index, int argc,
                            char **argv) {
    const auto eq = arg.find('=');
    if (eq != std::string::npos) {
        return arg.substr(eq + 1);
    }
    if (index + 1 >= argc) {
        throw std::runtime_error("Missing value for option: " + arg);
    }
    return argv[++index];
}

void PrintUsage(const char *program) {
    std::cout
        << "Usage: " << program << " --dataset PATH [options]\n"
        << "Options:\n"
        << "  --dataset PATH       NewCollege dataset root (required)\n"
        << "  --settings PATH      XPLSLAM settings YAML\n"
        << "  --point-vocab PATH   Point DBoW3 vocabulary\n"
        << "  --line-vocab PATH    Line DBoW3 vocabulary\n"
        << "  --timestamps PATH    Timestamp file, default "
           "all_camera_info/cam.txt\n"
        << "  --output-dir PATH    Output directory\n"
        << "  --start N            First frame index\n"
        << "  --start-frame N      First frame number, 1 means the first "
           "frame\n"
        << "  --max-frames N       Number of frames to process, 0 means all\n"
        << "  --step N             Process every Nth frame\n"
        << "  --log-every N        Progress print interval\n"
        << "  --runtime-timing     Write buffered frames.csv/calls.csv and run_totals.json\n"
        << "  --viewer             Enable Pangolin/OpenCV visualization\n"
        << "  --no-viewer          Disable visualization (default)\n"
        << "  --wait-key           Wait for Enter before loading the next "
           "frame\n"
        << "  --no-wait-key        Use dataset timestamps for playback timing "
           "(default)\n";
}

Options ParseArgs(int argc, char **argv) {
    Options options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else if (arg == "--dataset" || arg.rfind("--dataset=", 0) == 0) {
            options.dataset_root = ReadOptionValue(arg, i, argc, argv);
        } else if (arg == "--settings" || arg.rfind("--settings=", 0) == 0) {
            options.settings_file = ReadOptionValue(arg, i, argc, argv);
        } else if (arg == "--point-vocab" ||
                   arg.rfind("--point-vocab=", 0) == 0) {
            options.point_vocab = ReadOptionValue(arg, i, argc, argv);
        } else if (arg == "--line-vocab" ||
                   arg.rfind("--line-vocab=", 0) == 0) {
            options.line_vocab = ReadOptionValue(arg, i, argc, argv);
        } else if (arg == "--timestamps" ||
                   arg.rfind("--timestamps=", 0) == 0) {
            options.timestamp_file = ReadOptionValue(arg, i, argc, argv);
        } else if (arg == "--output-dir" ||
                   arg.rfind("--output-dir=", 0) == 0) {
            options.output_dir = ReadOptionValue(arg, i, argc, argv);
        } else if (arg == "--start" || arg.rfind("--start=", 0) == 0) {
            options.start = std::stoi(ReadOptionValue(arg, i, argc, argv));
        } else if (arg == "--start-frame" ||
                   arg.rfind("--start-frame=", 0) == 0) {
            const int start_frame =
                std::stoi(ReadOptionValue(arg, i, argc, argv));
            if (start_frame <= 0) {
                throw std::runtime_error("--start-frame must be >= 1");
            }
            options.start = start_frame - 1;
        } else if (arg == "--max-frames" ||
                   arg.rfind("--max-frames=", 0) == 0) {
            options.max_frames = std::stoi(ReadOptionValue(arg, i, argc, argv));
        } else if (arg == "--step" || arg.rfind("--step=", 0) == 0) {
            options.step = std::stoi(ReadOptionValue(arg, i, argc, argv));
        } else if (arg == "--log-every" || arg.rfind("--log-every=", 0) == 0) {
            options.log_every = std::stoi(ReadOptionValue(arg, i, argc, argv));
        } else if (arg == "--runtime-timing") {
            options.runtime_timing = true;
        } else if (arg == "--viewer") {
            options.use_viewer = true;
        } else if (arg == "--no-viewer") {
            options.use_viewer = false;
        } else if (arg == "--wait-key") {
            options.wait_key = true;
        } else if (arg == "--no-wait-key") {
            options.wait_key = false;
        } else {
            throw std::runtime_error("Unknown option: " + arg);
        }
    }

    if (options.dataset_root.empty()) {
        throw std::runtime_error("--dataset is required; use --help for usage");
    }
    if (options.timestamp_file.empty()) {
        options.timestamp_file =
            options.dataset_root / "all_camera_info/cam.txt";
    }
    if (options.start < 0) {
        throw std::runtime_error("--start must be >= 0");
    }
    if (options.max_frames < 0) {
        throw std::runtime_error("--max-frames must be >= 0");
    }
    if (options.step <= 0) {
        throw std::runtime_error("--step must be > 0");
    }
    if (options.log_every <= 0) {
        throw std::runtime_error("--log-every must be > 0");
    }

    return options;
}

void RequireFile(const fs::path &path, const std::string &name) {
    if (!fs::is_regular_file(path)) {
        throw std::runtime_error(name + " does not exist: " + path.string());
    }
}

int LoadConfiguredCameraCount(const fs::path &settings_file,
                              size_t max_supported_cameras) {
    cv::FileStorage settings(settings_file.string(), cv::FileStorage::READ);
    if (!settings.isOpened()) {
        throw std::runtime_error("Failed to open settings file: " +
                                 settings_file.string());
    }

    const cv::FileNode node = settings["Camera.nCams"];
    if (node.empty()) {
        throw std::runtime_error("Missing Camera.nCams in settings file: " +
                                 settings_file.string());
    }

    int camera_count = 0;
    node >> camera_count;
    if (camera_count <= 0) {
        throw std::runtime_error("Camera.nCams must be > 0");
    }
    if (static_cast<size_t>(camera_count) > max_supported_cameras) {
        throw std::runtime_error(
            "Camera.nCams exceeds the New College camera directories handled "
            "by this example");
    }

    return camera_count;
}

std::string JoinCameraNames(const std::vector<std::string> &camera_dirs) {
    std::ostringstream oss;
    for (size_t i = 0; i < camera_dirs.size(); ++i) {
        if (i > 0) {
            oss << " ";
        }
        oss << camera_dirs[i];
    }
    return oss.str();
}

bool WaitForNextFrame(size_t frame_index) {
    std::cout << "Processed frame " << frame_index
              << ". Press Enter for next frame, or q + Enter to stop: "
              << std::flush;
    std::string command;
    if (!std::getline(std::cin, command)) {
        return false;
    }
    command = Trim(command);
    return command != "q" && command != "Q" && command != "quit" &&
           command != "QUIT";
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Options options = ParseArgs(argc, argv);

        RequireFile(options.settings_file, "Settings file");
        RequireFile(options.point_vocab, "Point vocabulary");
        RequireFile(options.line_vocab, "Line vocabulary");
        RequireFile(options.timestamp_file, "Timestamp file");

        const fs::path image_root = options.dataset_root / "all_camera_images";
        const std::vector<std::string> dataset_camera_dirs = {"cam0", "cam1",
                                                              "cam3", "cam4"};
        const int configured_ncams = LoadConfiguredCameraCount(
            options.settings_file, dataset_camera_dirs.size());
        const std::vector<std::string> camera_dirs(dataset_camera_dirs.begin(),
                                                   dataset_camera_dirs.begin() +
                                                       configured_ncams);
        std::vector<std::vector<fs::path>> camera_images;
        camera_images.reserve(camera_dirs.size());
        for (const std::string &camera_dir : camera_dirs) {
            camera_images.push_back(CollectImages(image_root / camera_dir));
        }
        const std::vector<double> timestamps =
            LoadTimestamps(options.timestamp_file, "cam0");

        for (size_t cam = 0; cam < camera_images.size(); ++cam) {
            if (camera_images[cam].empty()) {
                throw std::runtime_error("No input images found for " +
                                         camera_dirs[cam]);
            }
        }

        size_t frame_count = timestamps.size();
        for (const auto &images : camera_images) {
            frame_count = std::min(frame_count, images.size());
        }
        if (frame_count == 0) {
            throw std::runtime_error("No synchronized frames found");
        }
        bool count_mismatch = timestamps.size() != frame_count;
        for (const auto &images : camera_images) {
            count_mismatch = count_mismatch || images.size() != frame_count;
        }
        if (count_mismatch) {
            std::cerr << "Warning: input counts differ, using first "
                      << frame_count << " synchronized frames.";
            for (size_t cam = 0; cam < camera_images.size(); ++cam) {
                std::cerr << " " << camera_dirs[cam] << "="
                          << camera_images[cam].size();
            }
            std::cerr << " timestamps=" << timestamps.size() << "\n";
        }
        if (static_cast<size_t>(options.start) >= frame_count) {
            throw std::runtime_error("--start is outside the frame range");
        }

        fs::create_directories(options.output_dir);
        const fs::path trajectory_path =
            options.output_dir / "trajectory_tum.txt";

        std::cout << "Loading XPLSLAM for " << options.sequence_name << "\n"
                  << "  settings: " << options.settings_file << "\n"
                  << "  dataset:  " << options.dataset_root << "\n"
                  << "  cameras:  " << JoinCameraNames(camera_dirs) << "\n"
                  << "  frames:   " << frame_count << "\n"
                  << "  viewer:   "
                  << (options.use_viewer ? "enabled" : "disabled") << "\n"
                  << "  playback: "
                  << (options.wait_key ? "wait key" : "dataset timing") << "\n";

        XPLSLAM::RuntimeCsv runtime_csv(options.runtime_timing ? options.output_dir : fs::path());
        const auto startup_begin = std::chrono::steady_clock::now();
        XPLSLAM::System slam(options.point_vocab.string(),
                             options.line_vocab.string(),
                             options.settings_file.string(), options.use_viewer,
                             0, options.sequence_name);

        const double startup_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - startup_begin).count();
        int processed = 0;
        const auto all_begin = std::chrono::steady_clock::now();
        for (size_t i = static_cast<size_t>(options.start); i < frame_count;
             i += static_cast<size_t>(options.step)) {
            if (options.max_frames > 0 && processed >= options.max_frames) {
                break;
            }

            const auto read_begin = std::chrono::steady_clock::now();
            std::vector<cv::Mat> images;
            images.reserve(camera_images.size());
            for (size_t cam = 0; cam < camera_images.size(); ++cam) {
                cv::Mat image = cv::imread(camera_images[cam][i].string(),
                                           cv::IMREAD_GRAYSCALE);
                if (image.empty()) {
                    throw std::runtime_error("Failed to read " +
                                             camera_dirs[cam] + " frame " +
                                             std::to_string(i));
                }
                images.push_back(image);
            }

            const double read_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - read_begin).count();
            const std::string frame_name = camera_images[0][i].stem().string();
            XPLSLAM::RuntimeFrame runtime_frame;
            XPLSLAM::RuntimeCapture runtime_capture(options.runtime_timing ? &runtime_frame : nullptr);
            const auto frame_begin = std::chrono::steady_clock::now();
            runtime_capture.Begin(frame_begin);
            const Sophus::SE3f Tbw = slam.TrackMultiCamera(
                images, timestamps[i], frame_name);
            const auto frame_end = std::chrono::steady_clock::now();
            runtime_capture.End(frame_end);
            (void)Tbw;

            ++processed;
            const double track_s =
                std::chrono::duration<double>(frame_end - frame_begin).count();
#ifdef XPLSLAM_ENABLE_PROFILING
            if (processed == 1 || processed % options.log_every == 0) {
                const XPLSLAM::ProcessedFrameTiming timing =
                    slam.GetLastProcessedFrameTiming();
                const auto &xplnet = timing.tracker.frame.xplnet;
                const auto &postprocess = xplnet.postprocess;
                const auto &line = postprocess.line;
                const auto &tracker = timing.tracker;
                const auto &frame = tracker.frame;
                const auto &track = tracker.track;
                const auto &local_map = track.local_map;
                const auto &local_search = local_map.search;
                const double processed_other_ms = std::max(
                    0.0, timing.total_ms - timing.image_preparation_ms -
                             timing.mode_control_ms - timing.reset_control_ms -
                             tracker.total_ms - timing.state_update_ms);
                const double tracker_other_ms = std::max(
                    0.0, tracker.total_ms - tracker.input_setup_ms -
                             tracker.frame_construction_ms -
                             tracker.track.total_ms - tracker.output_pose_ms);
                const double track_other_ms = std::max(
                    0.0, track.total_ms - track.step_wait_ms -
                             track.preprocess_ms - track.map_lock_wait_ms -
                             track.map_update_check_ms -
                             track.initialization_ms - track.pose_tracking_ms -
                             track.local_map_tracking_ms -
                             track.post_tracking_ms -
                             track.trajectory_record_ms);
                const double local_map_other_ms = std::max(
                    0.0, local_map.total_ms -
                             local_map.reference_map_update_ms -
                             local_map.local_keyframes_update_ms -
                             local_map.local_points_update_ms -
                             local_map.local_lines_update_ms -
                             local_map.search_local_features_ms -
                             local_map.pose_optimization_ms -
                             local_map.inlier_update_ms -
                             local_map.acceptance_check_ms);
                const double local_search_other_ms = std::max(
                    0.0, local_search.total_ms -
                             local_search.current_feature_cleanup_ms -
                             local_search.point_frustum_check_ms -
                             local_search.line_frustum_check_ms -
                             local_search.projection_matching_ms);
                const double frame_other_ms = std::max(
                    0.0, frame.total_ms - frame.setup_ms -
                             frame.xplnet_detect_ms -
                             frame.feature_materialization_ms -
                             frame.initial_computations_ms -
                             frame.descriptor_flatten_ms -
                             frame.grid_assignment_ms -
                             frame.multicamera_matching_ms);
                const double xplnet_other_ms = std::max(
                    0.0, xplnet.total_ms - xplnet.preprocess_ms -
                             xplnet.frontend_trt_enqueue_ms -
                             xplnet.postprocess_ms -
                             xplnet.final_stream_sync_ms);
                const double postprocess_other_ms = std::max(
                    0.0, postprocess.total_ms - postprocess.output_access_ms -
                             postprocess.point_junction_decode_ms -
                             postprocess.line_processing_ms -
                             postprocess.point_descriptor_sampling_ms -
                             postprocess.result_packaging_ms);
                const double line_other_ms = std::max(
                    0.0, line.total_ms - line.candidate_generation_ms -
                             line.verification_preprocess_ms -
                             line.verification_trt_enqueue_ms -
                             line.filtering_dedup_ms -
                             line.line_descriptor_preprocess_ms -
                             line.line_descriptor_trt_enqueue_ms -
                             line.result_packaging_ms);

                std::cout << "Processed frame " << i << " timestamp "
                          << std::fixed << std::setprecision(6) << timestamps[i]
                          << " in " << std::setprecision(2) << track_s * 1000.0
                          << " ms\n"
                          << "  Processed frame detail: image_preparation="
                          << timing.image_preparation_ms
                          << " ms, mode_control=" << timing.mode_control_ms
                          << " ms, reset_control=" << timing.reset_control_ms
                          << " ms, tracker=" << tracker.total_ms
                          << " ms, state_update=" << timing.state_update_ms
                          << " ms, other=" << processed_other_ms << " ms\n"
                          << "  Tracker detail: input_setup="
                          << tracker.input_setup_ms
                          << " ms, frame_construction="
                          << tracker.frame_construction_ms
                          << " ms, track=" << tracker.track.total_ms
                          << " ms, output_pose=" << tracker.output_pose_ms
                          << " ms, other=" << tracker_other_ms << " ms\n"
                          << "  Track detail: step_wait=" << track.step_wait_ms
                          << " ms, preprocess=" << track.preprocess_ms
                          << " ms, map_lock_wait=" << track.map_lock_wait_ms
                          << " ms, map_update_check="
                          << track.map_update_check_ms
                          << " ms, initialization=" << track.initialization_ms
                          << " ms, pose_tracking=" << track.pose_tracking_ms
                          << " ms, local_map_tracking="
                          << track.local_map_tracking_ms
                          << " ms, post_tracking=" << track.post_tracking_ms
                          << " ms, trajectory_record="
                          << track.trajectory_record_ms
                          << " ms, other=" << track_other_ms
                          << " ms, total=" << track.total_ms << " ms\n"
                          << "  Adaptive line selection: point_evaluation="
                          << track.point_constraint_evaluation_ms
                          << " ms, selection="
                          << track.line_selection_ms << " ms\n"
                          << "  Local map detail: reference_map_update="
                          << local_map.reference_map_update_ms
                          << " ms, local_keyframes_update="
                          << local_map.local_keyframes_update_ms
                          << " ms, local_points_update="
                          << local_map.local_points_update_ms
                          << " ms, local_lines_update="
                          << local_map.local_lines_update_ms
                          << " ms, search_local_features="
                          << local_map.search_local_features_ms
                          << " ms, pose_optimization="
                          << local_map.pose_optimization_ms
                          << " ms, inlier_update="
                          << local_map.inlier_update_ms
                          << " ms, acceptance_check="
                          << local_map.acceptance_check_ms
                          << " ms, other=" << local_map_other_ms
                          << " ms, total=" << local_map.total_ms << " ms\n"
                          << "  Local map search detail: current_feature_cleanup="
                          << local_search.current_feature_cleanup_ms
                          << " ms, point_frustum_check="
                          << local_search.point_frustum_check_ms
                          << " ms, line_frustum_check="
                          << local_search.line_frustum_check_ms
                          << " ms, projection_matching="
                          << local_search.projection_matching_ms
                          << " ms, other=" << local_search_other_ms
                          << " ms, total=" << local_search.total_ms << " ms\n"
                          << "  Frame construction detail: setup="
                          << frame.setup_ms
                          << " ms, xplnet_detect=" << frame.xplnet_detect_ms
                          << " ms, feature_materialization="
                          << frame.feature_materialization_ms
                          << " ms, initial_computations="
                          << frame.initial_computations_ms
                          << " ms, descriptor_flatten="
                          << frame.descriptor_flatten_ms
                          << " ms, grid_assignment="
                          << frame.grid_assignment_ms
                          << " ms, multicamera_matching="
                          << frame.multicamera_matching_ms
                          << " ms, other=" << frame_other_ms << " ms\n"
                          << "  XPLNet detail: preprocess="
                          << xplnet.preprocess_ms
                          << " ms, frontend_trt_enqueue="
                          << xplnet.frontend_trt_enqueue_ms
                          << " ms, postprocess=" << xplnet.postprocess_ms
                          << " ms, final_stream_sync="
                          << xplnet.final_stream_sync_ms
                          << " ms, other=" << xplnet_other_ms << " ms\n"
                          << "  XPLNet postprocess detail: output_access="
                          << postprocess.output_access_ms
                          << " ms, point_junction_decode="
                          << postprocess.point_junction_decode_ms
                          << " ms, line_processing="
                          << postprocess.line_processing_ms
                          << " ms, point_descriptor_sampling="
                          << postprocess.point_descriptor_sampling_ms
                          << " ms, result_packaging="
                          << postprocess.result_packaging_ms
                          << " ms, other=" << postprocess_other_ms << " ms\n"
                          << "  XPLNet line detail: candidate_generation="
                          << line.candidate_generation_ms
                          << " ms, verification_preprocess="
                          << line.verification_preprocess_ms
                          << " ms, verification_trt_enqueue="
                          << line.verification_trt_enqueue_ms
                          << " ms, filtering_dedup="
                          << line.filtering_dedup_ms
                          << " ms, line_descriptor_preprocess="
                          << line.line_descriptor_preprocess_ms
                          << " ms, line_descriptor_trt_enqueue="
                          << line.line_descriptor_trt_enqueue_ms
                          << " ms, result_packaging="
                          << line.result_packaging_ms
                          << " ms, other=" << line_other_ms
                          << " ms, junction_slots="
                          << line.junction_slots_per_image
                          << ", candidate_pairs_per_image="
                          << line.candidate_pairs_per_image
                          << ", accepted_before_dedup="
                          << line.accepted_before_dedup
                          << ", accepted_after_dedup="
                          << line.accepted_after_dedup << "\n";
            }
#endif

            double playback_wait_ms = 0.0;
            bool playback_stopped = false;
            const size_t next_i = i + static_cast<size_t>(options.step);
            const bool max_frames_done =
                options.max_frames > 0 && processed >= options.max_frames;
            if (!max_frames_done && next_i < frame_count) {
                if (options.wait_key) {
                    const auto wait_begin = std::chrono::steady_clock::now();
                    playback_stopped = !WaitForNextFrame(i);
                    playback_wait_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - wait_begin).count();
                } else {
                    const double frame_interval_s =
                        timestamps[next_i] - timestamps[i];
                    if (frame_interval_s > track_s) {
                        const auto wait_begin = std::chrono::steady_clock::now();
                        std::this_thread::sleep_for(
                            std::chrono::duration<double>(frame_interval_s -
                                                          track_s));
                        playback_wait_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - wait_begin).count();
                    }
                }
            }
            runtime_csv.Append(i, timestamps[i], runtime_frame, track_s * 1000.0,
                               read_ms, playback_wait_ms);
            if (playback_stopped) break;
        }

        const auto all_end = std::chrono::steady_clock::now();
        const double total_s =
            std::chrono::duration<double>(all_end - all_begin).count();
        const auto shutdown_begin = std::chrono::steady_clock::now();
        slam.Shutdown();
        const auto shutdown_end = std::chrono::steady_clock::now();
        slam.SaveTrajectoryTUM(trajectory_path.string());
        const auto save_end = std::chrono::steady_clock::now();
        runtime_csv.Finish(frame_count, processed, startup_s, total_s,
            std::chrono::duration<double>(shutdown_end-shutdown_begin).count(),
            std::chrono::duration<double>(save_end-shutdown_end).count());

        std::cout << "Done. Processed " << processed << " frames in "
                  << std::fixed << std::setprecision(2) << total_s
                  << " s\nTrajectory: " << trajectory_path << "\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &e) {
        std::cerr << "new_college failed: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
