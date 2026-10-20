#pragma once

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "keyframe.h"
#include <vector>

namespace XPLSLAM {

class SE3Solver {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SE3Solver(KeyFramePtr pKF1, KeyFramePtr pKF2,
              const std::vector<std::vector<MapPointPtr>> &vpMatchedMP12,
              const std::vector<std::vector<MapLinePtr>> &vpMatchedML12,
              std::vector<std::vector<KeyFramePtr>> vpKeyFrameMatchedMP,
              std::vector<std::vector<KeyFramePtr>> vpKeyFrameMatchedML);

#ifdef XPLSLAM_ENABLE_TESTING
    SE3Solver()
        : mnCam(0), mpKF1(nullptr), mpKF2(nullptr), N(0), Np(0), Nl(0),
          mR12i(Eigen::Matrix3f::Identity()), mt12i(Eigen::Vector3f::Zero()),
          ms12i(1.0f), mT12i(Eigen::Matrix4f::Identity()),
          mT21i(Eigen::Matrix4f::Identity()), mnInliersi(0), mnIterations(0),
          mnBestInliers(0), mBestT12(Eigen::Matrix4f::Identity()),
          mBestRotation(Eigen::Matrix3f::Identity()),
          mBestTranslation(Eigen::Vector3f::Zero()), mRansacProb(0.99),
          mRansacMinInliers(3), mRansacMaxIts(100), mTh(9.210f) {}
#endif

    void FromCameraToImage(const std::vector<Eigen::Vector3f> &vP3Db,
                           const std::vector<Eigen::Matrix<float, 6, 1>> &vL3Db,
                           std::vector<Eigen::Vector2f> &vP2D,
                           std::vector<Eigen::Vector3f> &vL2D,
                           std::vector<GeometricCamera *> &vpCamera,
                           const std::vector<Sophus::SE3f> &vTcb,
                           const std::vector<std::pair<int, int>> &vMPCam,
                           const std::vector<std::pair<int, int>> &vMLCam,
                           int frame);

    void SetRansacParameters(double probability = 0.99, int minInliers = 6,
                             int maxIterations = 300);

    Eigen::Matrix4f iterate(int nIterations, bool &bNoMore, int &nInliers,
                            bool &bConverge);

    Eigen::Matrix3f GetEstimatedRotation();

    Eigen::Vector3f GetEstimatedTranslation();

  protected:
    void ComputeSE3(const Eigen::Matrix3f &P3Db1i,
                    const Eigen::Matrix3f &P3Db2i);

    void CheckInliers();

    int mnCam;
    KeyFramePtr mpKF1;
    KeyFramePtr mpKF2;

    std::vector<Sophus::SE3f> mvTcb;
    std::vector<Eigen::Vector3f> mvX3Db1;
    std::vector<Eigen::Vector3f> mvX3Db2;
    std::vector<std::pair<int, int>> mvMPCam;
    std::vector<MapPointPtr> mvpMapPoints1;
    std::vector<MapPointPtr> mvpMapPoints2;
    std::vector<std::vector<MapPointPtr>> mvpMatchedMPs12;
    std::vector<Eigen::Matrix<float, 6, 1>> mvL3Db1;
    std::vector<Eigen::Matrix<float, 6, 1>> mvL3Db2;
    std::vector<std::pair<int, int>> mvMLCam;
    std::vector<MapLinePtr> mvpMapLines1;
    std::vector<MapLinePtr> mvpMapLines2;
    std::vector<std::vector<MapLinePtr>> mvpMatchedMLs12;

    int N;
    int Np;
    int Nl;

    Eigen::Matrix3f mR12i;
    Eigen::Vector3f mt12i;
    float ms12i;
    Eigen::Matrix4f mT12i;
    Eigen::Matrix4f mT21i;
    std::vector<bool> mvbInliersi;
    int mnInliersi;

    // Current Ransac State
    int mnIterations;
    std::vector<bool> mvbBestInliers;
    int mnBestInliers;
    Eigen::Matrix4f mBestT12;
    Eigen::Matrix3f mBestRotation;
    Eigen::Vector3f mBestTranslation;

    // Indices for random selection
    std::vector<size_t> mvAllIndices;

    std::vector<Eigen::Vector2f> mvP1im1;
    std::vector<Eigen::Vector2f> mvP2im2;
    std::vector<Eigen::Vector3f> mvL1im1;
    std::vector<Eigen::Vector3f> mvL2im2;

    // RANSAC probability
    double mRansacProb;

    // RANSAC min inliers
    int mRansacMinInliers;

    // RANSAC max iterations
    int mRansacMaxIts;

    // Threshold inlier/outlier. e = dist(Pi,T_ij*Pj)^2 < 5.991*mSigma2
    float mTh;

    std::vector<GeometricCamera *> vpCamera1;
    std::vector<GeometricCamera *> vpCamera2;
};
}; // namespace XPLSLAM
