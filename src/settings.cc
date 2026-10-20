#include "settings.h"

#include "camera_models/pinhole.h"

#include "system.h"
#include "xplnet.h"

#include <cmath>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/persistence.hpp>

#include <filesystem>
#include <iostream>

using namespace std;

namespace XPLSLAM {

template <>
float Settings::readParameter<float>(cv::FileStorage &fSettings,
                                     const std::string &name, bool &found,
                                     const bool required) {
    cv::FileNode node = fSettings[name];
    if (node.empty()) {
        if (required) {
            std::cerr << name
                      << " required parameter does not exist, aborting..."
                      << std::endl;
            exit(-1);
        } else {
            std::cerr << name << " optional parameter does not exist..."
                      << std::endl;
            found = false;
            return 0.0f;
        }
    } else if (!node.isReal()) {
        std::cerr << name << " parameter must be a real number, aborting..."
                  << std::endl;
        exit(-1);
    } else {
        found = true;
        return node.real();
    }
}

template <>
int Settings::readParameter<int>(cv::FileStorage &fSettings,
                                 const std::string &name, bool &found,
                                 const bool required) {
    cv::FileNode node = fSettings[name];
    if (node.empty()) {
        if (required) {
            std::cerr << name
                      << " required parameter does not exist, aborting..."
                      << std::endl;
            exit(-1);
        } else {
            std::cerr << name << " optional parameter does not exist..."
                      << std::endl;
            found = false;
            return 0;
        }
    } else if (!node.isInt()) {
        std::cerr << name << " parameter must be an integer number, aborting..."
                  << std::endl;
        exit(-1);
    } else {
        found = true;
        return node.operator int();
    }
}

template <>
string Settings::readParameter<string>(cv::FileStorage &fSettings,
                                       const std::string &name, bool &found,
                                       const bool required) {
    cv::FileNode node = fSettings[name];
    if (node.empty()) {
        if (required) {
            std::cerr << name
                      << " required parameter does not exist, aborting..."
                      << std::endl;
            exit(-1);
        } else {
            std::cerr << name << " optional parameter does not exist..."
                      << std::endl;
            found = false;
            return string();
        }
    } else if (!node.isString()) {
        std::cerr << name << " parameter must be a string, aborting..."
                  << std::endl;
        exit(-1);
    } else {
        found = true;
        return node.string();
    }
}

template <>
cv::Mat Settings::readParameter<cv::Mat>(cv::FileStorage &fSettings,
                                         const std::string &name, bool &found,
                                         const bool required) {
    cv::FileNode node = fSettings[name];
    if (node.empty()) {
        if (required) {
            std::cerr << name
                      << " required parameter does not exist, aborting..."
                      << std::endl;
            exit(-1);
        } else {
            std::cerr << name << " optional parameter does not exist..."
                      << std::endl;
            found = false;
            return cv::Mat();
        }
    } else {
        found = true;
        return node.mat();
    }
}

Settings::Settings(const std::string &configFile) {
    cv::FileStorage fSettings(configFile, cv::FileStorage::READ);
    if (!fSettings.isOpened()) {
        std::cerr << "Failed to open settings file at: " << configFile
                  << std::endl;
        exit(-1);
    } else {
        cout << "Loading settings from: " << configFile << std::endl;
    }

    readCamera(fSettings);
    cout << "Camera parameters loaded" << endl;

    readMultiCamera(fSettings);
    cout << "Multi-camera parameters loaded" << endl;

    readImageInfo(fSettings);
    cout << "Image info loaded" << endl;

    readXPLNet(fSettings);
    cout << "XPLNet parameters loaded" << endl;

    readLoadAndSave(fSettings);
    cout << "Load and save parameters loaded" << endl;

    readMapSparsification(fSettings);
    cout << "Map sparsification parameters loaded" << endl;

    readOtherParameters(fSettings);
    cout << "Other parameters loaded" << endl;

    for (int i = 0; i < nCams_; i++) {
        if (vbNeedToUndistort_[i]) {
            precomputeRectificationMaps(i);
            cout << "Rectification maps precomputed for camera " << i << endl;
        }
    }
}

void Settings::readCamera(cv::FileStorage &fSettings) {
    bool found;

    nCams_ = fSettings["Camera.nCams"];

    vcalibrations_.resize(nCams_);
    vDistortionModels_.resize(nCams_);
    vDistortionParameters_.resize(nCams_);
    vRectifiedIntrinsics_.resize(nCams_);
    vbNeedToUndistort_.resize(nCams_);
    vTbc_.resize(nCams_);

    for (int i = 0; i < nCams_; i++) {
        string cam_name = "Camera" + to_string(i);

        float fx = readParameter<float>(fSettings, cam_name + ".fx", found);
        float fy = readParameter<float>(fSettings, cam_name + ".fy", found);
        float cx = readParameter<float>(fSettings, cam_name + ".cx", found);
        float cy = readParameter<float>(fSettings, cam_name + ".cy", found);

        vector<float> vCalibration = {fx, fy, cx, cy};

        vcalibrations_[i] = new Pinhole(vCalibration);

        const cv::FileNode rectified_node =
            fSettings[cam_name + ".rectified_intrinsics"];
        if (!rectified_node.empty()) {
            if (!rectified_node.isSeq() || rectified_node.size() != 4) {
                throw runtime_error(cam_name +
                    ".rectified_intrinsics must be [fx, fy, cx, cy]");
            }
            vector<float> intrinsics;
            for (const cv::FileNode &value : rectified_node) {
                if ((!value.isReal() && !value.isInt()) ||
                    !std::isfinite(static_cast<float>(value.real()))) {
                    throw runtime_error(cam_name +
                        ".rectified_intrinsics must contain finite numbers");
                }
                intrinsics.push_back(static_cast<float>(value.real()));
            }
            if (intrinsics[0] <= 0 || intrinsics[1] <= 0) {
                throw runtime_error(cam_name +
                    ".rectified_intrinsics focal lengths must be positive");
            }
            vRectifiedIntrinsics_[i] = (cv::Mat_<float>(3, 3) <<
                intrinsics[0], 0, intrinsics[2],
                0, intrinsics[1], intrinsics[3],
                0, 0, 1);
        }

        const string distortion_model = readParameter<string>(
            fSettings, cam_name + ".distortion_model", found);
        if (distortion_model == "radtan") {
            vDistortionModels_[i] = DistortionModel::RADTAN;
        } else if (distortion_model == "equidistant") {
            vDistortionModels_[i] = DistortionModel::EQUIDISTANT;
        } else {
            throw runtime_error("Unsupported distortion model '" +
                                distortion_model + "' for " + cam_name);
        }

        const cv::FileNode distortion_node =
            fSettings[cam_name + ".distortion_parameters"];
        if (distortion_node.empty() || !distortion_node.isSeq()) {
            throw runtime_error(cam_name +
                                ".distortion_parameters must be a sequence");
        }

        vector<float> &distortion_parameters = vDistortionParameters_[i];
        distortion_parameters.clear();
        distortion_parameters.reserve(distortion_node.size());
        for (const cv::FileNode &parameter : distortion_node) {
            if (!parameter.isReal() && !parameter.isInt()) {
                throw runtime_error(cam_name +
                                    ".distortion_parameters must contain only "
                                    "numbers");
            }
            distortion_parameters.push_back(
                static_cast<float>(static_cast<double>(parameter)));
        }

        if (vDistortionModels_[i] == DistortionModel::RADTAN &&
            distortion_parameters.size() != 4 &&
            distortion_parameters.size() != 5) {
            throw runtime_error(cam_name +
                                " radtan distortion_parameters must contain "
                                "[k1, k2, p1, p2] or [k1, k2, p1, p2, k3]");
        }
        if (vDistortionModels_[i] == DistortionModel::EQUIDISTANT &&
            distortion_parameters.size() != 4) {
            throw runtime_error(cam_name +
                                " equidistant distortion_parameters must "
                                "contain [k1, k2, k3, k4]");
        }

        vbNeedToUndistort_[i] =
            vDistortionModels_[i] == DistortionModel::EQUIDISTANT ||
            !vRectifiedIntrinsics_[i].empty();
        if (!vbNeedToUndistort_[i]) {
            for (const float distortion : distortion_parameters) {
                if (std::abs(distortion) > 1e-8f) {
                    vbNeedToUndistort_[i] = true;
                    break;
                }
            }
        }

        cv::Mat Tbc =
            readParameter<cv::Mat>(fSettings, cam_name + ".Tbc", found);
        vTbc_[i] = Converter::toSophus(Tbc);
    }

    thDepth_ = readParameter<float>(fSettings, "thDepth", found);
}

void Settings::readMultiCamera(cv::FileStorage &fSettings) {
    bool found = false;

    stereo_pairs_.clear();

    point_min_dot_ = readParameter<float>(
        fSettings, "MultiCamera.point_min_dot", found, false);
    if (!found) {
        point_min_dot_ = 0.0f;
    }

    line_min_dot_ = readParameter<float>(fSettings, "MultiCamera.line_min_dot",
                                         found, false);
    if (!found) {
        line_min_dot_ = 0.0f;
    }

    cv::FileNode pairs_node = fSettings["MultiCamera.stereo_pairs"];
    if (pairs_node.empty()) {
        std::cerr << "MultiCamera.stereo_pairs optional parameter does not "
                     "exist..."
                  << std::endl;
        return;
    }

    if (!pairs_node.isSeq()) {
        std::cerr << "MultiCamera.stereo_pairs must be a sequence, aborting..."
                  << std::endl;
        exit(-1);
    }

    stereo_pairs_.reserve(pairs_node.size());
    for (const cv::FileNode &pair_node : pairs_node) {
        if (!pair_node.isSeq() || pair_node.size() != 2) {
            std::cerr << "Each MultiCamera.stereo_pairs entry must be a pair "
                         "[first_cam, second_cam], aborting..."
                      << std::endl;
            exit(-1);
        }

        const int first_cam = static_cast<int>(pair_node[0]);
        const int second_cam = static_cast<int>(pair_node[1]);
        if (first_cam < 0 || first_cam >= nCams_ || second_cam < 0 ||
            second_cam >= nCams_ || first_cam == second_cam) {
            std::cerr << "Invalid MultiCamera.stereo_pairs entry [" << first_cam
                      << ", " << second_cam << "], aborting..." << std::endl;
            exit(-1);
        }

        stereo_pairs_.emplace_back(first_cam, second_cam);
    }
}

void Settings::readImageInfo(cv::FileStorage &fSettings) {
    bool found;

    vImSize_.resize(nCams_);
    for (int i = 0; i < nCams_; i++) {
        string cam_name = "Camera" + to_string(i);

        int originalRows =
            readParameter<int>(fSettings, cam_name + ".height", found);
        int originalCols =
            readParameter<int>(fSettings, cam_name + ".width", found);
        vImSize_[i].width = originalCols;
        vImSize_[i].height = originalRows;
    }

    fps_ = readParameter<int>(fSettings, "fps", found);
    bRGB_ = readParameter<int>(fSettings, "RGB", found);
}

void Settings::readXPLNet(cv::FileStorage &fSettings) {
    bool found;

    xplnet_config_.xplnet_onnx =
        readParameter<string>(fSettings, "XPLNet.onnx", found);
    xplnet_config_.verify_onnx =
        readParameter<string>(fSettings, "XPLNet.verify_onnx", found);
    xplnet_config_.line_description_onnx =
        readParameter<string>(fSettings, "XPLNet.line_description_onnx", found);
    xplnet_config_.resize_width =
        readParameter<int>(fSettings, "XPLNet.resize_width", found);
    xplnet_config_.resize_height =
        readParameter<int>(fSettings, "XPLNet.resize_height", found);
    xplnet_config_.samples =
        readParameter<int>(fSettings, "XPLNet.samples", found);
    xplnet_config_.max_verify_segments =
        readParameter<int>(fSettings, "XPLNet.max_verify_segments", found);
    xplnet_config_.max_ldesc_lines =
        readParameter<int>(fSettings, "XPLNet.max_ldesc_lines", found);
    xplnet_config_.top_k_kp =
        readParameter<int>(fSettings, "XPLNet.top_k_kp", found);
    xplnet_config_.kp_th =
        readParameter<float>(fSettings, "XPLNet.kp_th", found);
    xplnet_config_.top_k_junc =
        readParameter<int>(fSettings, "XPLNet.top_k_junc", found);
    xplnet_config_.junc_th =
        readParameter<float>(fSettings, "XPLNet.junc_th", found);
    xplnet_config_.line_th =
        readParameter<float>(fSettings, "XPLNet.line_th", found);
    xplnet_config_.dedup_lines =
        readParameter<int>(fSettings, "XPLNet.dedup_lines", found);
    xplnet_config_.dedup_angle_deg =
        readParameter<float>(fSettings, "XPLNet.dedup_angle_deg", found);
    xplnet_config_.dedup_midpoint_cell =
        readParameter<float>(fSettings, "XPLNet.dedup_midpoint_cell", found);
    xplnet_config_.dedup_overlap_ratio =
        readParameter<float>(fSettings, "XPLNet.dedup_overlap_ratio", found);
    xplnet_config_.xplnet_build_config.fp16 =
        readParameter<int>(fSettings, "XPLNet.xplnet_build_config.fp16", found);
    xplnet_config_.xplnet_build_config.int8 =
        readParameter<int>(fSettings, "XPLNet.xplnet_build_config.int8", found);
    xplnet_config_.xplnet_build_config.dla_core = readParameter<int>(
        fSettings, "XPLNet.xplnet_build_config.dla_core", found);
    xplnet_config_.verify_build_config.fp16 =
        readParameter<int>(fSettings, "XPLNet.verify_build_config.fp16", found);
    xplnet_config_.verify_build_config.int8 =
        readParameter<int>(fSettings, "XPLNet.verify_build_config.int8", found);
    xplnet_config_.verify_build_config.dla_core = readParameter<int>(
        fSettings, "XPLNet.verify_build_config.dla_core", found);
    xplnet_config_.line_description_build_config.fp16 = readParameter<int>(
        fSettings, "XPLNet.line_description_build_config.fp16", found);
    xplnet_config_.line_description_build_config.int8 = readParameter<int>(
        fSettings, "XPLNet.line_description_build_config.int8", found);
    xplnet_config_.line_description_build_config.dla_core = readParameter<int>(
        fSettings, "XPLNet.line_description_build_config.dla_core", found);

    const std::filesystem::path cache_root = "output";
    std::filesystem::create_directories(cache_root);
    xplnet_config_.xplnet_build_config =
        MakeImageBuildConfig(xplnet_config_, nCams_, cache_root);
    const cv::FileNode image_engine_cache =
        fSettings["XPLNet.engine_cache_path"];
    if (!image_engine_cache.empty()) {
        if (!image_engine_cache.isString() ||
            image_engine_cache.string().empty()) {
            throw runtime_error(
                "XPLNet.engine_cache_path must be a nonempty string");
        }
        xplnet_config_.xplnet_build_config.engine_cache_path =
            image_engine_cache.string();
    }
    xplnet_config_.verify_build_config =
        MakeVerifyBuildConfig(xplnet_config_, nCams_, cache_root);
    xplnet_config_.line_description_build_config =
        MakeLineDescBuildConfig(xplnet_config_, nCams_, cache_root);
}

TrtBuildConfig
Settings::MakeImageBuildConfig(const XPLNet::Config &config, int batch_size,
                               const std::filesystem::path &cache_root) {
    TrtBuildConfig build;
    build.fp16 = config.xplnet_build_config.fp16;
    build.int8 = config.xplnet_build_config.int8;
    build.dla_core = config.xplnet_build_config.dla_core;
    build.engine_cache_path =
        (cache_root / ("xplnet_b" + std::to_string(batch_size) + ".engine"))
            .string();
    build.profiles["image"] = {
        {batch_size, 1, config.resize_height, config.resize_width},
        {batch_size, 1, config.resize_height, config.resize_width},
        {batch_size, 1, config.resize_height, config.resize_width},
    };
    return build;
}

TrtBuildConfig
Settings::MakeVerifyBuildConfig(const XPLNet::Config &config, int batch_size,
                                const std::filesystem::path &cache_root) {
    TrtBuildConfig build;
    build.fp16 = config.verify_build_config.fp16;
    build.int8 = config.verify_build_config.int8;
    build.dla_core = config.verify_build_config.dla_core;
    build.engine_cache_path =
        (cache_root /
         ("verification_coarse_b" + std::to_string(batch_size) + ".engine"))
            .string();
    build.profiles["line_features"] = {
        {batch_size, 2 * config.verify_dim + (config.samples - 2) * 4, 1},
        {batch_size, 2 * config.verify_dim + (config.samples - 2) * 4,
         std::min(config.max_verify_segments, 12500)},
        {batch_size, 2 * config.verify_dim + (config.samples - 2) * 4,
         config.max_verify_segments},
    };
    return build;
}

TrtBuildConfig
Settings::MakeLineDescBuildConfig(const XPLNet::Config &config, int batch_size,
                                  const std::filesystem::path &cache_root) {
    TrtBuildConfig build;
    build.fp16 = config.line_description_build_config.fp16;
    build.int8 = config.line_description_build_config.int8;
    build.dla_core = config.line_description_build_config.dla_core;
    build.engine_cache_path =
        (cache_root /
         ("line_description_b" + std::to_string(batch_size) + ".engine"))
            .string();
    build.profiles["line_samples"] = {
        {batch_size, config.ldesc_dim, 1, config.samples},
        {batch_size, config.ldesc_dim, std::min(config.max_ldesc_lines, 256),
         config.samples},
        {batch_size, config.ldesc_dim, config.max_ldesc_lines, config.samples},
    };
    return build;
}

void Settings::readLoadAndSave(cv::FileStorage &fSettings) {
    bool found;

    sLoadFrom_ = readParameter<string>(fSettings, "System.LoadAtlasFromFile",
                                       found, false);
    sSaveto_ = readParameter<string>(fSettings, "System.SaveAtlasToFile", found,
                                     false);
}

void Settings::readMapSparsification(cv::FileStorage &fSettings) {
    auto read_optional_int = [&fSettings](const string &name, int fallback) {
        const cv::FileNode node = fSettings[name];
        return node.empty() ? fallback : static_cast<int>(node);
    };
    auto read_optional_double = [&fSettings](const string &name,
                                             double fallback) {
        const cv::FileNode node = fSettings[name];
        return node.empty() ? fallback : static_cast<double>(node.real());
    };

    map_sparsification_.enabled =
        read_optional_int("Sparsification.Enabled", 0) != 0;
    map_sparsification_.min_points_per_keyframe =
        read_optional_int("Sparsification.N", 100);
    map_sparsification_.lambda = static_cast<float>(
        read_optional_double("Sparsification.Lambda", 500.0));
    map_sparsification_.grid_lambda = static_cast<float>(
        read_optional_double("Sparsification.GridLambda", 10.0));
    map_sparsification_.window_length =
        read_optional_int("Sparsification.WindowLength", 30);
    map_sparsification_.non_local_keyframes =
        read_optional_int("Sparsification.NonLocalKF", 30);
    map_sparsification_.mip_gap =
        read_optional_double("Sparsification.MIPGap", 0.002);
    map_sparsification_.solver_threads =
        read_optional_int("Sparsification.SolverThreads", 1);

    if (map_sparsification_.min_points_per_keyframe <= 0 ||
        map_sparsification_.window_length <= 0 ||
        map_sparsification_.non_local_keyframes <= 0 ||
        map_sparsification_.lambda < 0.0f ||
        map_sparsification_.grid_lambda < 0.0f ||
        map_sparsification_.mip_gap < 0.0 ||
        map_sparsification_.solver_threads <= 0) {
        throw runtime_error("Invalid map sparsification configuration");
    }
}

void Settings::readOtherParameters(cv::FileStorage &fSettings) {
    bool found;

    thFarPoints_ =
        readParameter<float>(fSettings, "System.thFarPoints", found, false);
    thFarLines_ =
        readParameter<float>(fSettings, "System.thFarLines", found, false);

    auto optional_int = [&fSettings](const char *name, int fallback) {
        const cv::FileNode node = fSettings[name];
        return node.empty() ? fallback : static_cast<int>(node);
    };
    auto optional_double = [&fSettings](const char *name, double fallback) {
        const cv::FileNode node = fSettings[name];
        return node.empty() ? fallback : node.real();
    };
    adaptive_lines_.enabled =
        optional_int("AdaptiveLines.Enabled", 1) != 0;
    adaptive_lines_.min_point_matches =
        optional_int("AdaptiveLines.MinPointMatches",
                     optional_int("AdaptiveLines.MinPointInliers", 50));
    adaptive_lines_.coverage_rows =
        optional_int("AdaptiveLines.CoverageRows", 4);
    adaptive_lines_.coverage_cols =
        optional_int("AdaptiveLines.CoverageCols", 4);
    adaptive_lines_.min_coverage_ratio =
        optional_double("AdaptiveLines.MinCoverageRatio", 0.25);
    adaptive_lines_.min_normalized_eigenvalue =
        optional_double("AdaptiveLines.MinNormalizedEigenvalue", 0.05);
    adaptive_lines_.line_precheck_chi2 =
        optional_double("AdaptiveLines.LinePrecheckChi2", 9.21);
    adaptive_lines_.min_line_gain =
        optional_double("AdaptiveLines.MinLineGain", 1e-3);
    adaptive_lines_.max_selected_lines =
        optional_int("AdaptiveLines.MaxSelectedLines", 50);
    adaptive_lines_.target_observations =
        optional_int("AdaptiveLines.TargetObservations", 50);

    tracking_robustness_.max_frame_step_m = static_cast<float>(
        optional_double("Tracking.MaxFrameStepMeters", 0.0));

    if (tracking_robustness_.max_frame_step_m < 0.0f) {
        throw runtime_error("Invalid tracking robustness configuration");
    }

    if (adaptive_lines_.min_point_matches < 0 ||
        adaptive_lines_.coverage_rows <= 0 ||
        adaptive_lines_.coverage_cols <= 0 ||
        adaptive_lines_.min_coverage_ratio < 0.0 ||
        adaptive_lines_.min_coverage_ratio > 1.0 ||
        adaptive_lines_.min_normalized_eigenvalue < 0.0 ||
        adaptive_lines_.line_precheck_chi2 <= 0.0 ||
        adaptive_lines_.min_line_gain < 0.0 ||
        adaptive_lines_.max_selected_lines < 0 ||
        adaptive_lines_.target_observations < 0) {
        throw runtime_error("Invalid adaptive line configuration");
    }
}

void Settings::precomputeRectificationMaps(int cam_id) {
    if (cam_id < 0 || cam_id >= nCams_) {
        std::cerr << "Invalid camera id " << cam_id
                  << " when precomputing rectification maps" << std::endl;
        return;
    }

    vM1_.resize(nCams_);
    vM2_.resize(nCams_);

    cv::Mat K = vcalibrations_[cam_id]->toK();
    cv::Mat D(vDistortionParameters_[cam_id].size(), 1, CV_32F,
              vDistortionParameters_[cam_id].data());

    cv::Mat R = cv::Mat::eye(3, 3, CV_32F);
    cv::Mat newK = vRectifiedIntrinsics_[cam_id].empty()
        ? K.clone() : vRectifiedIntrinsics_[cam_id].clone();

    if (vDistortionModels_[cam_id] == DistortionModel::EQUIDISTANT) {
        cv::fisheye::initUndistortRectifyMap(
            K, D, R, newK, vImSize_[cam_id], CV_32F, vM1_[cam_id],
            vM2_[cam_id]);
    } else {
        cv::initUndistortRectifyMap(K, D, R, newK, vImSize_[cam_id], CV_32F,
                                    vM1_[cam_id], vM2_[cam_id]);
    }

    vcalibrations_[cam_id]->setParameter(newK.at<float>(0, 0), 0);
    vcalibrations_[cam_id]->setParameter(newK.at<float>(1, 1), 1);
    vcalibrations_[cam_id]->setParameter(newK.at<float>(0, 2), 2);
    vcalibrations_[cam_id]->setParameter(newK.at<float>(1, 2), 3);
}

}; // namespace XPLSLAM
