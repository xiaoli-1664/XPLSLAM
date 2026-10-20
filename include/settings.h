#pragma once

#include "camera_models/geometric_camera.h"
#include "adaptive_line.h"
#include "xplnet.h"

#include <filesystem>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>
#include <utility>

namespace XPLSLAM {

class System;

struct MapSparsificationConfig {
    bool enabled = false;
    int min_points_per_keyframe = 100;
    float lambda = 500.0f;
    float grid_lambda = 10.0f;
    int window_length = 30;
    int non_local_keyframes = 30;
    double mip_gap = 0.002;
    int solver_threads = 1;
};

struct TrackingRobustnessConfig {
    float max_frame_step_m = 0.0f;
};

class Settings {
  public:
    /*
     * Delete default constructor
     */
    Settings() = delete;

    /*
     * Constructor from file
     */
    Settings(const std::string &configFile);

    /*
     * Ostream operator overloading to dump settings to the terminal
     */
    friend std::ostream &operator<<(std::ostream &output, const Settings &s);

    /*
     * Getter methods
     */
    GeometricCamera *camera(int cam_id) { return vcalibrations_[cam_id]; }
    std::vector<GeometricCamera *> cameras() { return vcalibrations_; }

    cv::Mat cameraDistortionCoef(int cam_id) {
        return cv::Mat(vDistortionParameters_[cam_id].size(), 1, CV_32F,
                       vDistortionParameters_[cam_id].data());
    }

    std::vector<Sophus::SE3f> Tbc() { return vTbc_; }
    std::vector<std::pair<int, int>> stereoPairs() { return stereo_pairs_; }
    float thDepth() { return thDepth_; }
    float pointMinDot() { return point_min_dot_; }
    float lineMinDot() { return line_min_dot_; }
    XPLNet::Config xplnetConfig() { return xplnet_config_; }

    bool needToUndistort(int cam_id) { return vbNeedToUndistort_[cam_id]; }

    cv::Size imSize(int cam_id) { return vImSize_[cam_id]; }
    float fps() { return fps_; }
    bool rgb() { return bRGB_; }

    std::string atlasLoadFile() { return sLoadFrom_; }
    std::string atlasSaveFile() { return sSaveto_; }

    float thFarPoints() { return thFarPoints_; }
    float thFarLines() { return thFarLines_; }

    cv::Mat M1(int cam_id) { return vM1_[cam_id]; }
    cv::Mat M2(int cam_id) { return vM2_[cam_id]; }

    int nCams() { return nCams_; }
    const MapSparsificationConfig &mapSparsification() const {
        return map_sparsification_;
    }
    const AdaptiveLineConfig &adaptiveLines() const { return adaptive_lines_; }
    const TrackingRobustnessConfig &trackingRobustness() const {
        return tracking_robustness_;
    }

  private:
    enum class DistortionModel {
        RADTAN,
        EQUIDISTANT,
    };

    template <typename T>
    T readParameter(cv::FileStorage &fSettings, const std::string &name,
                    bool &found, const bool required = true) {
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
                return T();
            }

        } else {
            found = true;
            return (T)node;
        }
    }

    void readCamera(cv::FileStorage &fSettings);
    void readImageInfo(cv::FileStorage &fSettings);
    void readLoadAndSave(cv::FileStorage &fSettings);
    void readXPLNet(cv::FileStorage &fSettings);
    void readMultiCamera(cv::FileStorage &fSettings);
    void readMapSparsification(cv::FileStorage &fSettings);
    void readOtherParameters(cv::FileStorage &fSettings);
    TrtBuildConfig
    MakeVerifyBuildConfig(const XPLNet::Config &config, int batch_size,
                          const std::filesystem::path &cache_root);

    TrtBuildConfig
    MakeImageBuildConfig(const XPLNet::Config &config, int batch_size,
                         const std::filesystem::path &cache_root);

    TrtBuildConfig
    MakeLineDescBuildConfig(const XPLNet::Config &config, int batch_size,
                            const std::filesystem::path &cache_root);

    void precomputeRectificationMaps(int cam_id);

    int nCams_; // Number of cameras

    /*
     * Visual stuff
     */
    std::vector<GeometricCamera *> vcalibrations_;
    std::vector<DistortionModel> vDistortionModels_;
    std::vector<std::vector<float>> vDistortionParameters_;
    std::vector<cv::Mat> vRectifiedIntrinsics_;

    std::vector<cv::Size> vImSize_;

    float fps_;
    bool bRGB_;

    XPLNet::Config xplnet_config_;

    std::vector<bool> vbNeedToUndistort_;

    std::vector<Sophus::SE3f> vTbc_;
    std::vector<std::pair<int, int>> stereo_pairs_;
    float thDepth_;
    float point_min_dot_;
    float line_min_dot_;

    std::vector<cv::Mat> vM1_, vM2_;
    /*
     * Save & load maps
     */
    std::string sLoadFrom_, sSaveto_;

    /*
     * Other stuff
     */
    float thFarPoints_;
    float thFarLines_;
    MapSparsificationConfig map_sparsification_;
    AdaptiveLineConfig adaptive_lines_;
    TrackingRobustnessConfig tracking_robustness_;
};
}; // namespace XPLSLAM
