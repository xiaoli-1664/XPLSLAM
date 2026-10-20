#pragma once

#include "sophus/sim3.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

#include "frame.h"
#include "keyframe.h"
#include "map_point.h"

namespace XPLSLAM {

class Matcher {
  public:
    struct TriangulationMatch {
        int cam1 = -1;
        int idx1 = -1;
        int cam2 = -1;
        int idx2 = -1;
    };

    struct TriangulationMatches {
        KeyFrame::PointFeatureSnapshot point_snapshot1;
        KeyFrame::PointFeatureSnapshot point_snapshot2;
        Sophus::SE3f body_pose1;
        Sophus::SE3f body_pose2;
        std::vector<Sophus::SE3f> camera_poses1;
        std::vector<Sophus::SE3f> camera_poses2;
        std::vector<TriangulationMatch> point_matches;
        std::vector<TriangulationMatch> line_matches;
    };

    Matcher(float nnratio = 0.6);

    int SearchByBow(KeyFramePtr pKF, Frame &F,
                    std::vector<std::vector<MapPointPtr>> &vpMapPointMatches,
                    std::vector<std::vector<MapLinePtr>> &vpMapLineMatches);

    int SearchLinesByBow(
        KeyFramePtr pKF, Frame &F,
        std::vector<std::vector<MapLinePtr>> &vpMapLineMatches);

    int SearchByBow(const Frame &LastFrame, Frame &CurrentFrame,
                    std::vector<std::vector<MapPointPtr>> &vpMapPointMatches,
                    std::vector<std::vector<MapLinePtr>> &vpMapLineMatches);

    int SearchByBow(KeyFramePtr pKF1, KeyFramePtr pKF2,
                    std::vector<std::vector<MapPointPtr>> &vpMapPointMatches12,
                    std::vector<std::vector<MapLinePtr>> &vpMapLineMatches12,
                    float point_high_threshold = TH_P_HIGH,
                    float line_high_threshold = TH_L_HIGH);

    int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame,
                           const float th_point, const float th_line_d,
                           const float th_line_theta);

    int SearchByProjection(Frame &F, KeyFramePtr pKF,
                           const std::set<MapPointPtr> &sAlreadyFoundPoint,
                           const std::set<MapLinePtr> &sAlreadyFoundLine,
                           const float th_point, const float th_line_d,
                           const float th_line_theta);

    int SearchByProjection(Frame &F, const std::vector<MapPointPtr> &vpMapPoints,
                           const std::vector<MapLinePtr> &vpMapLines,
                           const float th_point, const float th_line,
                           const bool bFarPoints, const float thFarPoints,
                           const bool bFarLines, const float thFarLines);
    int
    SearchByProjection(KeyFramePtr pKF, Sophus::SE3f &Tcw,
                       const std::vector<MapPointPtr> &vpPoints,
                       const std::vector<MapLinePtr> &vpLines,
                       std::vector<std::vector<MapPointPtr>> &vpMapPointMatches,
                       std::vector<std::vector<MapLinePtr>> &vpMapLineMatches,
                       const float th_point, const float th_line_d,
                       const float th_line_theta, float ratio_factor,
                       float point_high_threshold = TH_P_HIGH,
                       float line_high_threshold = TH_L_HIGH);

    int SearchForTriangulation(KeyFramePtr pKF1, KeyFramePtr pKF2,
                               TriangulationMatches &matches);

    int Fuse(KeyFramePtr pKF,
             const std::vector<std::vector<MapPointPtr>> &vpMapPoints_cam,
             const std::vector<std::vector<MapLinePtr>> &vpMapLines_cam,
             const float th_point, const float th_line_d,
             const float th_line_theta);

    int Fuse(KeyFramePtr pKF1, Sophus::SE3f &Tbw,
             const std::vector<MapPointPtr> &vpPoints,
             std::vector<MapLinePtr> &vpLines, const float th_point,
             const float th_line_d, const float th_line_theta,
             std::vector<MapPointPtr> &vpReplacedPoints,
             std::vector<MapLinePtr> &vpReplacedLines,
             float point_high_threshold = TH_P_HIGH,
             float line_high_threshold = TH_L_HIGH);

  public:
    static const float TH_L_LOW;
    static const float TH_P_LOW;
    static const float TH_L_HIGH;
    static const float TH_P_HIGH;

  protected:
    float mfNNratio;
};

}; // namespace XPLSLAM
