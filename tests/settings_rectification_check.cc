#include "settings.h"

#include <opencv2/calib3d.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {

void Require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

void CheckGeometry(XPLSLAM::Settings &settings, cv::FileStorage &source,
                   bool use_rectified) {
    for (int cam = 0; cam < settings.nCams(); ++cam) {
        const std::string prefix = "Camera" + std::to_string(cam);
        const cv::Mat raw_K = (cv::Mat_<float>(3, 3) <<
            source[prefix + ".fx"].real(), 0, source[prefix + ".cx"].real(),
            0, source[prefix + ".fy"].real(), source[prefix + ".cy"].real(),
            0, 0, 1);
        const cv::Mat output_K = settings.camera(cam)->toK();
        const cv::FileNode target = source[prefix + ".rectified_intrinsics"];
        const float expected_fx = use_rectified
            ? static_cast<float>(target[0].real()) : raw_K.at<float>(0, 0);
        Require(std::abs(output_K.at<float>(0, 0) - expected_fx) < 1e-4f,
                "Incorrect output focal length");
        Require(settings.needToUndistort(cam), "Fisheye remapping is disabled");

        // Project output pixels back through the raw fisheye camera. The map
        // must sample these source pixels using the SAME K exposed to tracking.
        const std::vector<cv::Point2f> pixels = {
            {100, 100}, {350, 270}, {600, 400}};
        std::vector<cv::Point2f> rays;
        for (const auto &p : pixels) {
            rays.emplace_back(
                (p.x - output_K.at<float>(0, 2)) / output_K.at<float>(0, 0),
                (p.y - output_K.at<float>(1, 2)) / output_K.at<float>(1, 1));
        }
        std::vector<cv::Point2f> distorted;
        cv::fisheye::distortPoints(rays, distorted, raw_K,
                                  settings.cameraDistortionCoef(cam));
        for (size_t i = 0; i < pixels.size(); ++i) {
            const int x = static_cast<int>(pixels[i].x);
            const int y = static_cast<int>(pixels[i].y);
            Require(std::abs(settings.M1(cam).at<float>(y, x) -
                             distorted[i].x) < 1e-3f &&
                    std::abs(settings.M2(cam).at<float>(y, x) -
                             distorted[i].y) < 1e-3f,
                    "Remap and tracking intrinsics disagree");
        }
    }
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 2) return 1;
    const auto temporary = std::filesystem::temp_directory_path() /
        ("xplslam_rectification_" + std::to_string(getpid()) + ".yaml");
    try {
        cv::FileStorage source(argv[1], cv::FileStorage::READ);
        Require(source.isOpened(), "Cannot open test calibration");
        XPLSLAM::Settings configured(argv[1]);
        CheckGeometry(configured, source, true);

        std::ifstream input(argv[1]);
        std::ostringstream fallback;
        std::string line;
        while (std::getline(input, line)) {
            if (line.find(".rectified_intrinsics:") == std::string::npos) {
                fallback << line << '\n';
            }
        }
        {
            std::ofstream output(temporary);
            output << fallback.str();
        }
        XPLSLAM::Settings defaults(temporary.string());
        CheckGeometry(defaults, source, false);
        {
            std::ofstream output(temporary);
            output << fallback.str()
                   << "\nCamera0.rectified_intrinsics: [-1.0, 270.0, 350.0, 270.0]\n";
        }
        bool rejected = false;
        try {
            XPLSLAM::Settings invalid(temporary.string());
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        Require(rejected, "Invalid output focal length was accepted");
        std::filesystem::remove(temporary);
        return 0;
    } catch (const std::exception &error) {
        std::filesystem::remove(temporary);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
