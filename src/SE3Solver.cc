#include <Eigen/Eigenvalues>
#include <algorithm>
#include <random>

#include "SE3Solver.h"
#include "camera_models/geometric_camera.h"
#include "geometric_tools.h"
#include "map_point.h"

namespace XPLSLAM {

using namespace std;

namespace {

void ComputeCentroid(const Eigen::Matrix3f &P, Eigen::Matrix3f &Pr,
                     Eigen::Vector3f &C) {
    C = P.rowwise().sum() / P.cols();
    for (int i = 0; i < P.cols(); ++i) {
        Pr.col(i) = P.col(i) - C;
    }
}

} // namespace

SE3Solver::SE3Solver(KeyFramePtr pKF1, KeyFramePtr pKF2,
                     const std::vector<std::vector<MapPointPtr>> &vpMatchedMP12,
                     const std::vector<std::vector<MapLinePtr>> &vpMatchedML12,
                     std::vector<std::vector<KeyFramePtr>> vpKeyFrameMatchedMP,
                     std::vector<std::vector<KeyFramePtr>> vpKeyFrameMatchedML)
    : mnCam(pKF1->mnCam), mnIterations(0), mnBestInliers(0),
      vpCamera1(pKF1->mvpCameras), vpCamera2(pKF2->mvpCameras) {
    mpKF1 = pKF1;
    mpKF2 = pKF2;

    vector<vector<MapPointPtr>> vpKFMP1 = mpKF1->GetMapPointMatches();
    vector<vector<MapLinePtr>> vpKFML1 = mpKF1->GetMapLineMatches();

    mvpMapPoints1.reserve(mpKF1->mnTotalPoints);
    mvpMapPoints2.reserve(mpKF1->mnTotalPoints);
    mvMPCam.reserve(mpKF1->mnTotalPoints);
    mvpMapLines1.reserve(mpKF1->mnTotalLines);
    mvpMapLines2.reserve(mpKF1->mnTotalLines);
    mvMLCam.reserve(mpKF1->mnTotalLines);
    mvpMatchedMPs12 = vpMatchedMP12;
    mvpMatchedMLs12 = vpMatchedML12;
    mvX3Db1.reserve(pKF1->mnTotalPoints);
    mvX3Db2.reserve(pKF1->mnTotalPoints);
    mvL3Db1.reserve(pKF1->mnTotalLines);
    mvL3Db2.reserve(pKF1->mnTotalLines);

    Eigen::Matrix3f Rbw1 = pKF1->GetRotation();
    Eigen::Vector3f tbw1 = pKF1->GetTranslation();
    Eigen::Matrix3f Rbw2 = pKF2->GetRotation();
    Eigen::Vector3f tbw2 = pKF2->GetTranslation();

    mvAllIndices.reserve(pKF1->mnTotalPoints + pKF1->mnTotalLines);

    size_t idx = 0;

    KeyFramePtr pKFm = pKF2;

    for (int c = 0; c < mnCam; ++c) {
        for (size_t i1 = 0; i1 < vpMatchedMP12[c].size(); ++i1) {
            MapPointPtr pMP1 = vpKFMP1[c][i1];
            MapPointPtr pMP2 = vpMatchedMP12[c][i1];

            if (!pMP1 || !pMP2)
                continue;

            if (pMP1->isBad() || pMP2->isBad())
                continue;

            int indexKF1 = pMP1->GetIndexInKeyFrame(pKF1, c);
            int indexKF2 = -1;
            int cam2 = -1;

            vector<int> vIndicesKF2 = pMP2->GetIndexInKeyFrame(pKFm);

            for (int i = 0; i < mnCam; ++i) {
                if (vIndicesKF2[i] >= 0) {
                    indexKF2 = vIndicesKF2[i];
                    cam2 = i;
                    break;
                }
            }

            if (indexKF1 < 0 || indexKF2 < 0)
                continue;

            const cv::Point2f &kp1 = pKF1->mvKeysUn_cams[c][indexKF1];
            const cv::Point2f &kp2 = pKFm->mvKeysUn_cams[cam2][indexKF2];

            mvpMapPoints1.push_back(pMP1);
            mvpMapPoints2.push_back(pMP2);

            mvMPCam.emplace_back(c, cam2);

            Eigen::Vector3f X3D1w = pMP1->GetWorldPos();
            mvX3Db1.push_back(Rbw1 * X3D1w + tbw1);

            Eigen::Vector3f X3D2w = pMP2->GetWorldPos();
            mvX3Db2.push_back(Rbw2 * X3D2w + tbw2);

            mvAllIndices.push_back(idx++);
        }
    }

    for (int c = 0; c < mnCam; ++c) {
        for (size_t i1 = 0; i1 < vpMatchedML12[c].size(); ++i1) {
            MapLinePtr pML1 = vpKFML1[c][i1];
            MapLinePtr pML2 = vpMatchedML12[c][i1];

            if (!pML1 || !pML2)
                continue;

            if (pML1->isBad() || pML2->isBad())
                continue;

            int indexKF1 = pML1->GetIndexInKeyFrame(pKF1, c);
            int indexKF2 = -1;
            int cam2 = -1;

            vector<int> vIndicesKF2 = pML2->GetIndexInKeyFrame(pKFm);

            for (int i = 0; i < mnCam; ++i) {
                if (vIndicesKF2[i] >= 0) {
                    indexKF2 = vIndicesKF2[i];
                    cam2 = i;
                    break;
                }
            }

            if (indexKF1 < 0 || indexKF2 < 0)
                continue;

            const Eigen::Vector4f &line1 =
                mpKF1->mvLineKeysUn_cams[c][indexKF1];
            const Eigen::Vector4f &line2 =
                mpKF2->mvLineKeysUn_cams[cam2][indexKF2];

            mvpMapLines1.push_back(pML1);
            mvpMapLines2.push_back(pML2);

            mvMLCam.emplace_back(c, cam2);

            Eigen::Matrix<float, 6, 1> L3D1w = pML1->GetWorldPos();
            Eigen::Matrix<float, 6, 1> L3D1b;
            TransformLineToCamera(L3D1w, Rbw1, tbw1, L3D1b);
            mvL3Db1.push_back(L3D1b);

            Eigen::Matrix<float, 6, 1> L3D2w = pML2->GetWorldPos();
            Eigen::Matrix<float, 6, 1> L3D2b;
            TransformLineToCamera(L3D2w, Rbw2, tbw2, L3D2b);
            mvL3Db2.push_back(L3D2b);

            // mvAllIndices.push_back(idx++);
        }
    }

    vector<Sophus::SE3f> vTbc = pKF1->GetCameraExtrinsics();
    for (size_t cam_id = 0; cam_id < vTbc.size(); ++cam_id) {
        mvTcb.push_back(vTbc[cam_id].inverse());
    }

    FromCameraToImage(mvX3Db1, mvL3Db1, mvP1im1, mvL1im1, vpCamera1, mvTcb,
                      mvMPCam, mvMLCam, 1);
    FromCameraToImage(mvX3Db2, mvL3Db2, mvP2im2, mvL2im2, vpCamera2, mvTcb,
                      mvMPCam, mvMLCam, 2);

    SetRansacParameters();
}

void SE3Solver::FromCameraToImage(
    const std::vector<Eigen::Vector3f> &vP3Db,
    const std::vector<Eigen::Matrix<float, 6, 1>> &vL3Db,
    std::vector<Eigen::Vector2f> &vP2D, std::vector<Eigen::Vector3f> &vL2D,
    std::vector<GeometricCamera *> &vpCamera,
    const std::vector<Sophus::SE3f> &vTcb,
    const std::vector<std::pair<int, int>> &vMPCam,
    const std::vector<std::pair<int, int>> &vMLCam, int frame) {
    vP2D.clear();
    vL2D.clear();
    vP2D.reserve(vP3Db.size());
    vL2D.reserve(vL3Db.size());

    for (size_t i = 0; i < vP3Db.size(); ++i) {
        const int cam_id = frame == 1 ? vMPCam[i].first : vMPCam[i].second;
        const Eigen::Vector3f x3Dc = vTcb[cam_id] * vP3Db[i];

        vP2D.push_back(vpCamera[cam_id]->project(x3Dc));
    }

    for (size_t i = 0; i < vL3Db.size(); ++i) {
        const int cam_id = frame == 1 ? vMLCam[i].first : vMLCam[i].second;

        Eigen::Matrix<float, 6, 1> l3Dc;
        TransformLineToCamera(vL3Db[i], vTcb[cam_id], l3Dc);

        vL2D.push_back(vpCamera[cam_id]->projectLine(l3Dc));
    }
}

void SE3Solver::SetRansacParameters(double probability, int minInliers,
                                    int maxIterations) {
    mRansacProb = probability;
    mRansacMinInliers = minInliers;
    mRansacMaxIts = maxIterations;

    Np = mvpMapPoints1.size();
    Nl = mvpMapLines1.size();
    // N = Np + Nl;
    N = Np; // Only use points for RANSAC

    mvbInliersi.resize(N);

    // Adjust Parameters according to number of correspondences
    float epsilon = (float)mRansacMinInliers / N;

    // Set RANSAC iterations according to probability, epsilon, and max
    // iterations
    int nIterations;

    if (mRansacMinInliers == N)
        nIterations = 1;
    else
        nIterations = ceil(log(1 - mRansacProb) / log(1 - pow(epsilon, 3)));

    mRansacMaxIts = max(1, min(nIterations, mRansacMaxIts));

    mnIterations = 0;
}

Eigen::Matrix4f SE3Solver::iterate(int nIterations, bool &bNoMore,
                                   int &nInliers, bool &bConverge) {
    bNoMore = false;
    bConverge = false;
    nInliers = 0;

    std::random_device rd;
    std::mt19937 g(rd());

    if (N < mRansacMinInliers) {
        bNoMore = true;
        return Eigen::Matrix4f::Identity();
    }

    vector<size_t> vAvailableIndices;

    // vector<Eigen::Vector3f> P3Db1i;
    // vector<Eigen::Vector3f> P3Db2i;
    // vector<Eigen::Matrix<float, 6, 1>> L3Db1i;
    // vector<Eigen::Matrix<float, 6, 1>> L3Db2i;
    Eigen::Matrix3f P3Db1i;
    Eigen::Matrix3f P3Db2i;

    // P3Db1i.reserve(3);
    // L3Db1i.reserve(3);
    // P3Db2i.reserve(3);
    // L3Db2i.reserve(3);

    int nCurrentIterations = 0;

    Eigen::Matrix4f bestSE3;

    while (mnIterations < mRansacMaxIts && nCurrentIterations < nIterations) {
        nCurrentIterations++;
        mnIterations++;

        vAvailableIndices = mvAllIndices;

        std::shuffle(vAvailableIndices.begin(), vAvailableIndices.end(), g);

        for (short i = 0; i < 3; ++i) {
            int idx = vAvailableIndices[i];

            // if (idx < Np) {
            //     P3Db1i.push_back(mvX3Db1[idx]);
            //     P3Db2i.push_back(mvX3Db2[idx]);
            // } else {
            //     int line_idx = idx - Np;
            //     L3Db1i.push_back(mvL3Db1[line_idx]);
            //     L3Db2i.push_back(mvL3Db2[line_idx]);
            // }

            P3Db1i.col(i) = mvX3Db1[idx];
            P3Db2i.col(i) = mvX3Db2[idx];
        }

        ComputeSE3(P3Db1i, P3Db2i);

        CheckInliers();

        if (mnInliersi >= mnBestInliers) {
            mvbBestInliers = mvbInliersi;
            mnBestInliers = mnInliersi;
            mBestT12 = mT12i;
            mBestRotation = mR12i;
            mBestTranslation = mt12i;

            if (mnInliersi > mRansacMinInliers) {
                nInliers = mnInliersi;
                bConverge = true;
                return mBestT12;
            } else {
                bestSE3 = mBestT12;
            }
        }
    }

    if (mnIterations >= mRansacMaxIts) {
        bNoMore = true;
    }

    return bestSE3;
}

void SE3Solver::ComputeSE3(const Eigen::Matrix3f &P3Db1i,
                           const Eigen::Matrix3f &P3Db2i) {
    Eigen::Matrix3f Pr1;
    Eigen::Matrix3f Pr2;
    Eigen::Vector3f O1;
    Eigen::Vector3f O2;

    ComputeCentroid(P3Db1i, Pr1, O1);
    ComputeCentroid(P3Db2i, Pr2, O2);

    const Eigen::Matrix3f M = Pr2 * Pr1.transpose();

    const float N11 = M(0, 0) + M(1, 1) + M(2, 2);
    const float N12 = M(1, 2) - M(2, 1);
    const float N13 = M(2, 0) - M(0, 2);
    const float N14 = M(0, 1) - M(1, 0);
    const float N22 = M(0, 0) - M(1, 1) - M(2, 2);
    const float N23 = M(0, 1) + M(1, 0);
    const float N24 = M(2, 0) + M(0, 2);
    const float N33 = -M(0, 0) + M(1, 1) - M(2, 2);
    const float N34 = M(1, 2) + M(2, 1);
    const float N44 = -M(0, 0) - M(1, 1) + M(2, 2);

    Eigen::Matrix4f N;
    N << N11, N12, N13, N14, N12, N22, N23, N24, N13, N23, N33, N34, N14, N24,
        N34, N44;

    Eigen::EigenSolver<Eigen::Matrix4f> eigSolver;
    eigSolver.compute(N);

    Eigen::Vector4f eval = eigSolver.eigenvalues().real();
    Eigen::Matrix4f evec = eigSolver.eigenvectors().real();

    int maxIndex;
    eval.maxCoeff(&maxIndex);

    Eigen::Vector3f vec = evec.block<3, 1>(1, maxIndex);
    const float vecNorm = vec.norm();
    if (vecNorm < 1e-5f) {
        mR12i = Eigen::Matrix3f::Identity();
    } else {
        const float ang = atan2(vecNorm, evec(0, maxIndex));
        vec = 2.0f * ang * vec / vecNorm;
        mR12i = Sophus::SO3f::exp(vec).matrix();
    }

    ms12i = 1.0f;
    mt12i = O1 - mR12i * O2;

    mT12i = Eigen::Matrix4f::Identity();
    mT12i.block<3, 3>(0, 0) = mR12i;
    mT12i.block<3, 1>(0, 3) = mt12i;

    mT21i = Eigen::Matrix4f::Identity();
    mT21i.block<3, 3>(0, 0) = mR12i.transpose();
    mT21i.block<3, 1>(0, 3) = -mR12i.transpose() * mt12i;
}

void SE3Solver::CheckInliers() {
    constexpr float kPointReprojectionChi2 = 9.210f;

    mnInliersi = 0;

    for (int i = 0; i < Np; ++i) {
        const int cam1 = mvMPCam[i].first;
        const int cam2 = mvMPCam[i].second;

        const Eigen::Vector3f x3Db2_est =
            mT21i.block<3, 3>(0, 0) * mvX3Db1[i] + mT21i.block<3, 1>(0, 3);
        const Eigen::Vector3f x3Dc2 = mvTcb[cam2] * x3Db2_est;

        const Eigen::Vector2f p1im2 = vpCamera2[cam2]->project(x3Dc2);
        const float err2 = (p1im2 - mvP2im2[i]).squaredNorm();

        const Eigen::Vector3f x3Db1_est =
            mT12i.block<3, 3>(0, 0) * mvX3Db2[i] + mT12i.block<3, 1>(0, 3);
        const Eigen::Vector3f x3Dc1 = mvTcb[cam1] * x3Db1_est;

        const Eigen::Vector2f p2im1 = vpCamera1[cam1]->project(x3Dc1);
        const float err1 = (p2im1 - mvP1im1[i]).squaredNorm();

        if (err1 < kPointReprojectionChi2 && err2 < kPointReprojectionChi2) {
            mvbInliersi[i] = true;
            mnInliersi++;
        } else {
            mvbInliersi[i] = false;
        }
    }
}

Eigen::Matrix3f SE3Solver::GetEstimatedRotation() { return mBestRotation; }

Eigen::Vector3f SE3Solver::GetEstimatedTranslation() {
    return mBestTranslation;
}

}; // namespace XPLSLAM
