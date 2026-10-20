#pragma once

#include "adaptive_line.h"
#include "frame.h"
#include "keyframe.h"
#include "loop_closing.h"
#include "map.h"
#include "map_line.h"
#include "map_point.h"
#include "se3quat.h"

namespace XPLSLAM {

class Optimizer {
  public:
    int static PoseOptimization(Frame *pFrame);

    // Assess incoming matches without updating the pose; disable all line edges.
    static PoseConstraintSummary
    EvaluatePointConstraints(Frame *pFrame, const AdaptiveLineConfig &config);

    static PoseConstraintSummary
    SelectLineConstraints(Frame *pFrame, const AdaptiveLineConfig &config,
                          const PoseConstraintSummary &point_summary);

    static int PoseOptimizationSelectedLines(
        Frame *pFrame, const AdaptiveLineConfig &config,
        PoseConstraintSummary *summary = nullptr);

    static int ScreenLineAssociationsAtFixedPose(Frame *pFrame,
                                                 double chi2_threshold,
                                                 bool only_unselected = false);

    void static LocalBundleAdjustment(KeyFramePtr pKF, bool *pbStopFlag,
                                      Map *pMap, int &num_fixedKF,
                                      int &num_OptKF, int &num_MPs,
                                      int &num_edges);

    void static LocalBundleAdjustment(KeyFramePtr pMainKF,
                                      std::vector<KeyFramePtr> vpAdjustKF,
                                      std::vector<KeyFramePtr> vpFixedKF,
                                      bool *pbStopFlag);

    int static OptimizeSE3(
        KeyFramePtr pKF1, KeyFramePtr pKF2,
        std::vector<std::vector<MapPointPtr>> &vpMapPointMatches,
        std::vector<std::vector<MapLinePtr>> &vpMapLineMatches,
        g2o::SE3Quat &g2oT12, const float th2);

    void static OptimizeEssentialGraph(
        KeyFramePtr pCurKF, std::vector<KeyFramePtr> &vpFixedKFs,
        std::vector<KeyFramePtr> &vpFixedCorrectedKFs,
        std::vector<KeyFramePtr> &vpNonFixedKFs,
        std::vector<MapPointPtr> &vpNonCorrectedMPs,
        std::vector<MapLinePtr> &vpNonCorrectedMLs);

    void static OptimizeEssentialGraph(
        Map *pMap, KeyFramePtr pLoopKF, KeyFramePtr pCurKF,
        const LoopClosing::KeyFrameAndPose &NonCorrectedSE3,
        const LoopClosing::KeyFrameAndPose &CorrectedSE3,
        const std::map<KeyFramePtr, std::set<KeyFramePtr>> &LoopConnections,
        const bool bFixScale = true);

    void static GlobalBundleAdjustemnt(Map *pMap, int nIterations = 5,
                                       bool *pbStopFlag = NULL,
                                       const unsigned long nLoopKF = 0,
                                       const bool bRobust = true);

    void static BundleAdjustment(const std::vector<KeyFramePtr> &vpKFs,
                                 const std::vector<MapPointPtr> &vpMP,
                                 const std::vector<MapLinePtr> &vpML,
                                 int nIterations = 5, bool *pbStopFlag = NULL,
                                 const unsigned long nLoopKF = 0,
                                 const bool bRobust = true);
};

}; // namespace XPLSLAM
