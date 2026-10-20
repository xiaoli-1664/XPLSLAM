#pragma once
#include "UPnPL.h"
#include "frame.h"

namespace XPLSLAM {

class UPnPLSolver {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    UPnPLSolver(const Frame &F,
                const std::vector<std::vector<MapPointPtr>> &vpMapPointMatches,
                const std::vector<std::vector<MapLinePtr>> &vpMapLineMatches);

    ~UPnPLSolver();

    void SetRansacParameters(double probability = 0.99, int minInliers = 8,
                             int maxIterations = 30, int minSet = 6,
                             float epsilon = 0.4, float th2 = 5.991);

    bool iterate(int nIterations, bool &bNoMore,
                 std::vector<std::vector<bool>> &vbPointInliers,
                 std::vector<std::vector<bool>> &vbLineInliers, int &nInliers,
                 Eigen::Matrix4f &Tout);

  private:
    void CheckInliers();
    bool Refine();

    int nCam;

    std::vector<Eigen::Matrix3d> mvRbc;
    std::vector<Eigen::Vector3d> mvtbc;

    std::vector<std::vector<MapPointPtr>> mvpMapPointMatches;
    std::vector<std::vector<MapLinePtr>> mvpMapLineMatches;

    std::vector<cv::Point2f> mvP2D;
    std::vector<Eigen::Vector4d> mvL2D;

    std::vector<Eigen::Vector3d> mvP2Dh;
    std::vector<Eigen::VectorXd> mvL2Dh;

    std::vector<Eigen::Vector3d> mvP3Dw;
    std::vector<Eigen::VectorXd> mvL3Dw;
    std::vector<Eigen::Matrix<float, 6, 1>> mvL3Dw_plucker;

    std::vector<int> mvPCamId;
    std::vector<int> mvLCamId;

    std::vector<size_t> mvPIndices;
    std::vector<size_t> mvLIndices;

    Eigen::Matrix3d mR;
    Eigen::Vector3d mt;

    std::vector<bool> mvbInliersi;
    int mnInliersi;

    int mnIterations;
    std::vector<bool> mvbBestInliers;
    int mnBestInliers;
    Eigen::Matrix4f mBestTbw;

    Eigen::Matrix4f mRefinedTbw;
    std::vector<bool> mvbRefinedInliers;
    int mnRefinedInliers;

    int N;
    int Np;
    int Nl;

    float mth2;

    std::vector<size_t> mvAllIndices;

    // RANSAC probability
    double mRansacProb;

    // RANSAC min inliers
    int mRansacMinInliers;

    // RANSAC max iterations
    int mRansacMaxIts;

    // RANSAC expected inliers/total ratio
    float mRansacEpsilon;

    // RANSAC Threshold inlier/outlier. Max error e = dist(P1,T_12*P2)^2
    float mRansacTh;

    // RANSAC Minimun Set used at each iteration
    int mRansacMinSet;

    std::vector<GeometricCamera *> mvpCamera;
};

}; // namespace XPLSLAM
