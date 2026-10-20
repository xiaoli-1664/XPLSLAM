#include "runtime_timing.h"
#include "upnpl_solver.h"
#include "UPnPL.h"
#include "g2o_types.h"
#include "geometric_camera.h"
#include "geometric_tools.h"
#include "map_point.h"

#include <algorithm>
#include <random>

namespace XPLSLAM {

using namespace std;

UPnPLSolver::UPnPLSolver(
    const Frame &F,
    const std::vector<std::vector<MapPointPtr>> &vpMapPointMatches,
    const std::vector<std::vector<MapLinePtr>> &vpMapLineMatches)
    : mnInliersi(0), mnIterations(0), mnBestInliers(0), N(0),
      mvpCamera(F.mvpCamera) {
    std::vector<Sophus::SE3f> vTbc = F.GetCameraExtrinsics();
    nCam = vTbc.size();
    mvRbc.resize(nCam);
    mvtbc.resize(nCam);
    for (int cam_id = 0; cam_id < nCam; ++cam_id) {
        mvRbc[cam_id] = vTbc[cam_id].rotationMatrix().cast<double>();
        mvRbc[cam_id] = NormalizeRotation(mvRbc[cam_id]);
        mvtbc[cam_id] = vTbc[cam_id].translation().cast<double>();
    }

    mvpMapPointMatches = vpMapPointMatches;
    mvpMapLineMatches = vpMapLineMatches;

    mvP2D.reserve(F.mnTotalPoints);
    mvP2Dh.reserve(F.mnTotalPoints);
    mvP3Dw.reserve(F.mnTotalPoints);
    mvPIndices.reserve(F.mnTotalPoints);
    mvPCamId.reserve(F.mnTotalPoints);
    mvL2D.reserve(F.mnTotalLines);
    mvL2Dh.reserve(F.mnTotalLines);
    mvL3Dw.reserve(F.mnTotalLines);
    mvL3Dw_plucker.reserve(F.mnTotalLines);
    mvLIndices.reserve(F.mnTotalLines);
    mvLCamId.reserve(F.mnTotalLines);

    mvAllIndices.reserve(F.mnTotalPoints + F.mnTotalLines);

    int idx = 0;
    for (int c = 0; c < nCam; ++c) {
        for (size_t i = 0, iend = vpMapPointMatches[c].size(); i < iend; ++i) {
            MapPointPtr pMP = vpMapPointMatches[c][i];

            if (!pMP)
                continue;

            if (pMP->isBad())
                continue;

            const cv::Point2f &pt = F.mvKeysUn_cams[c][i];
            mvP2D.push_back(pt);

            cv::Point3f cv_br = mvpCamera[c]->unproject(pt);
            Eigen::Vector3d P2Dh(cv_br.x, cv_br.y, cv_br.z);
            mvP2Dh.push_back(P2Dh);

            Eigen::Vector3d P3Dw = pMP->GetWorldPos().cast<double>();
            mvP3Dw.push_back(P3Dw);

            mvPIndices.push_back(i);
            mvPCamId.push_back(c);
            mvAllIndices.push_back(idx);
            idx++;
        }
    }
    for (int c = 0; c < nCam; ++c) {
        for (size_t i = 0, iend = vpMapLineMatches[c].size(); i < iend; ++i) {
            MapLinePtr pML = vpMapLineMatches[c][i];

            if (!pML)
                continue;

            if (pML->isBad())
                continue;

            const Eigen::Vector4d &line =
                F.mvLineKeysUn_cams[c][i].cast<double>();
            mvL2D.push_back(line);
            Eigen::VectorXd L2Dh(6);
            L2Dh.head<3>() = mvpCamera[c]->unprojectEig(line.head<2>());
            L2Dh.tail<3>() = mvpCamera[c]->unprojectEig(line.tail<2>());
            mvL2Dh.push_back(L2Dh);

            Eigen::Matrix<float, 6, 1> L3Dw = pML->GetWorldPos();
            mvL3Dw_plucker.push_back(L3Dw);
            Eigen::VectorXd L3DwEnd(6);
            const Eigen::Vector3d moment = L3Dw.head<3>().cast<double>();
            const Eigen::Vector3d direction = L3Dw.tail<3>().cast<double>();
            const Eigen::Vector3d point =
                direction.cross(moment) / direction.squaredNorm();
            L3DwEnd.head<3>() = point + direction;
            L3DwEnd.tail<3>() = point - direction;
            mvL3Dw.push_back(L3DwEnd);

            mvLIndices.push_back(i);
            mvLCamId.push_back(c);
            mvAllIndices.push_back(idx);
            idx++;
        }
    }

    SetRansacParameters();
}

UPnPLSolver::~UPnPLSolver() = default;

void UPnPLSolver::SetRansacParameters(double probability, int minInliers,
                                      int maxIterations, int minSet,
                                      float epsilon, float th2) {
    mRansacProb = probability;
    mRansacMinInliers = minInliers;
    mRansacMaxIts = maxIterations;
    mRansacEpsilon = epsilon;
    mRansacMinSet = minSet;

    Np = mvP2Dh.size();
    Nl = mvL2Dh.size();
    N = Np + Nl;

    mvbInliersi.resize(N);
    int nMinInliers = N * mRansacEpsilon;
    if (nMinInliers < mRansacMinInliers)
        nMinInliers = mRansacMinInliers;
    if (nMinInliers < minSet)
        nMinInliers = minSet;
    mRansacMinInliers = nMinInliers;

    if (mRansacEpsilon < (float)mRansacMinInliers / N)
        mRansacEpsilon = (float)mRansacMinInliers / N;

    // Set RANSAC iterations according to probability, epsilon, and max
    // iterations
    int nIterations;

    if (mRansacMinInliers == N)
        nIterations = 1;
    else
        nIterations =
            ceil(log(1 - mRansacProb) / log(1 - pow(mRansacEpsilon, 3)));

    mRansacMaxIts = max(1, min(nIterations, mRansacMaxIts));

    mth2 = th2;
}

void UPnPLSolver::CheckInliers() {
    mnInliersi = 0;
    for (int i = 0; i < N; ++i) {
        if (i < Np) {
            int cam = mvPCamId[i];
            cv::Point2f p2D = mvP2D[i];
            Eigen::Vector3d p3Dw = mvP3Dw[i];
            Eigen::Vector3d p3Dc =
                mvRbc[cam].inverse() * ((mR * p3Dw + mt) - mvtbc[cam]);

            Eigen::Vector2d p2D_proj = mvpCamera[cam]->project(p3Dc);

            float distX = p2D_proj.x() - p2D.x;
            float distY = p2D_proj.y() - p2D.y;

            float error2 = distX * distX + distY * distY;

            if (error2 < mth2) {
                mvbInliersi[i] = true;
                mnInliersi++;
            } else {
                mvbInliersi[i] = false;
            }
        } else {
            int line_idx = i - Np;
            int cam = mvLCamId[line_idx];
            Eigen::Vector4d l2D = mvL2D[line_idx];
            Eigen::Matrix<float, 6, 1> l3Dw = mvL3Dw_plucker[line_idx];

            Eigen::Vector2f error;

            Sophus::SE3d Tbc(mvRbc[cam], mvtbc[cam]);

            Sophus::SE3d Tbw = Sophus::SE3d(mR, mt);
            Sophus::SE3f Tcw = (Tbc.inverse() * Tbw).cast<float>();
            bool valid = GeometricTools::ComputeLineReprojectionError(
                l3Dw, mvpCamera[cam], Tcw, l2D.cast<float>(), error);
            const float chi2 = error.squaredNorm();
            if (valid && chi2 < mth2) {
                mvbInliersi[i] = true;
                mnInliersi++;
            } else {
                mvbInliersi[i] = false;
            }
        }
    }
}

bool UPnPLSolver::Refine() {
    vector<int> vIndices;
    vIndices.reserve(mvbBestInliers.size());

    for (size_t i = 0; i < mvbBestInliers.size(); ++i) {
        if (mvbBestInliers[i]) {
            vIndices.push_back(i);
        }
    }

    vector<Eigen::Vector3d> vP2DhInliers;
    vector<Eigen::VectorXd> vL2DhInliers;
    vector<Eigen::Vector3d> vP3DwInliers;
    vector<Eigen::VectorXd> vL3DwInliers;
    vector<int> vPCamIdInliers;
    vector<int> vLCamIdInliers;

    for (int idx : vIndices) {
        if (idx < Np) {
            vP2DhInliers.push_back(mvP2Dh[idx]);
            vP3DwInliers.push_back(mvP3Dw[idx]);
            vPCamIdInliers.push_back(mvPCamId[idx]);
        } else {
            int line_idx = idx - Np;
            vL2DhInliers.push_back(mvL2Dh[line_idx]);
            vL3DwInliers.push_back(mvL3Dw[line_idx]);
            vLCamIdInliers.push_back(mvLCamId[line_idx]);
        }
    }

    RuntimeCallScope runtime_call("upnpl_refine");
    runtime_call.call.selected = vL3DwInliers.size();
    UPnPL::UPnPL solver_;
    solver_.solveMain(vP3DwInliers, vL3DwInliers, vP2DhInliers, vL2DhInliers,
                      vPCamIdInliers, vLCamIdInliers, mvRbc, mvtbc, mR, mt);
    mR = NormalizeRotation(mR);

    CheckInliers();

    mnRefinedInliers = mnInliersi;
    mvbRefinedInliers = mvbInliersi;

    if (mnInliersi > mRansacMinInliers) {
        mRefinedTbw.setIdentity();
        mRefinedTbw.block<3, 3>(0, 0) = mR.cast<float>();
        mRefinedTbw.block<3, 1>(0, 3) = mt.cast<float>();
        return true;
    }

    return false;
}

bool UPnPLSolver::iterate(int nIterations, bool &bNoMore,
                          std::vector<std::vector<bool>> &vbPointInliers,
                          std::vector<std::vector<bool>> &vbLineInliers,
                          int &nInliers, Eigen::Matrix4f &Tout) {
    RuntimePhaseScope runtime_phase(RuntimePhase::Pose);
    std::random_device rd;
    std::mt19937 g(rd());

    Tout.setIdentity();
    bNoMore = false;
    vbPointInliers.clear();
    vbLineInliers.clear();
    nInliers = 0;

    if (N < mRansacMinInliers) {
        bNoMore = true;
        return false;
    }

    vector<size_t> vAvailableIndices;

    int nCurrentIterations = 0;

    while (mnIterations < mRansacMaxIts || nCurrentIterations < nIterations) {
        nCurrentIterations++;
        mnIterations++;

        vAvailableIndices = mvAllIndices;
        std::shuffle(vAvailableIndices.begin(), vAvailableIndices.end(), g);

        vector<Eigen::Vector3d> vP2DhSampled;
        vector<Eigen::VectorXd> vL2DhSampled;
        vector<Eigen::Vector3d> vP3DwSampled;
        vector<Eigen::VectorXd> vL3DwSampled;
        vector<int> vPCamIdSampled;
        vector<int> vLCamIdSampled;
        vP2DhSampled.reserve(mRansacMinSet);
        vL2DhSampled.reserve(mRansacMinSet);
        vP3DwSampled.reserve(mRansacMinSet);
        vL3DwSampled.reserve(mRansacMinSet);
        vPCamIdSampled.reserve(mRansacMinSet);
        vLCamIdSampled.reserve(mRansacMinSet);

        for (int i = 0; i < mRansacMinSet; ++i) {
            size_t idx = vAvailableIndices[i];
            if (idx < Np) {
                vP2DhSampled.push_back(mvP2Dh[idx]);
                vP3DwSampled.push_back(mvP3Dw[idx]);
                vPCamIdSampled.push_back(mvPCamId[idx]);
            } else {
                int line_idx = idx - Np;
                vL2DhSampled.push_back(mvL2Dh[line_idx]);
                vL3DwSampled.push_back(mvL3Dw[line_idx]);
                vLCamIdSampled.push_back(mvLCamId[line_idx]);
            }
        }

        RuntimeCallScope runtime_call("upnpl_sample");
        runtime_call.call.selected = vL3DwSampled.size();
        UPnPL::UPnPL solver_;
        solver_.solveMain(vP3DwSampled, vL3DwSampled, vP2DhSampled,
                          vL2DhSampled, vPCamIdSampled, vLCamIdSampled, mvRbc,
                          mvtbc, mR, mt);
        mR = NormalizeRotation(mR);

        CheckInliers();

        if (mnInliersi >= mRansacMinInliers) {
            if (mnInliersi > mnBestInliers) {
                mvbBestInliers = mvbInliersi;
                mnBestInliers = mnInliersi;
                mBestTbw.setIdentity();
                mBestTbw.block<3, 3>(0, 0) = mR.cast<float>();
                mBestTbw.block<3, 1>(0, 3) = mt.cast<float>();
            }

            if (Refine()) {
                nInliers = mnRefinedInliers;
                vbPointInliers.resize(nCam);
                vbLineInliers.resize(nCam);
                for (int cam_id = 0; cam_id < nCam; ++cam_id) {
                    vbPointInliers[cam_id].resize(
                        mvpMapPointMatches[cam_id].size(), false);
                    vbLineInliers[cam_id].resize(
                        mvpMapLineMatches[cam_id].size(), false);
                }
                for (size_t i = 0; i < N; ++i) {
                    if (mvbInliersi[i]) {
                        if (i < Np) {
                            int cam = mvPCamId[i];
                            int idx = mvPIndices[i];
                            vbPointInliers[cam][idx] = true;
                        } else {
                            int line_idx = i - Np;
                            int cam = mvLCamId[line_idx];
                            int idx = mvLIndices[line_idx];
                            vbLineInliers[cam][idx] = true;
                        }
                    }
                }
                Tout = mRefinedTbw;
                return true;
            }
        }
    }

    if (mnIterations >= mRansacMaxIts) {
        bNoMore = true;
        if (mnBestInliers >= mRansacMinInliers) {
            nInliers = mnBestInliers;
            vbPointInliers.resize(nCam);
            vbLineInliers.resize(nCam);
            for (int cam_id = 0; cam_id < nCam; ++cam_id) {
                vbPointInliers[cam_id].resize(mvpMapPointMatches[cam_id].size(),
                                              false);
                vbLineInliers[cam_id].resize(mvpMapLineMatches[cam_id].size(),
                                             false);
            }
            for (size_t i = 0; i < N; ++i) {
                if (mvbBestInliers[i]) {
                    if (i < Np) {
                        int cam = mvPCamId[i];
                        int idx = mvPIndices[i];
                        vbPointInliers[cam][idx] = true;
                    } else {
                        int line_idx = i - Np;
                        int cam = mvLCamId[line_idx];
                        int idx = mvLIndices[line_idx];
                        vbLineInliers[cam][idx] = true;
                    }
                }
            }
        }
    }

    return false;
}

}; // namespace XPLSLAM
