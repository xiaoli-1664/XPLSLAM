#include "loop_closing.h"
#include "SE3Solver.h"
#include "Vocabulary.h"
#include "converter.h"
#include "geometric_tools.h"
#include "map_sparsification.h"
#include "matcher.h"
#include "optimizer.h"
#include "se3quat.h"

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <thread>
#include <unistd.h>

using namespace std;

namespace XPLSLAM {

namespace {

constexpr float kLoopPointHighThreshold = 0.85f;
constexpr float kLoopLineHighThreshold = 0.85f;

} // namespace

LoopClosing::LoopClosing(Atlas *pAtlas, KeyFrameDatabase *pDB,
                         DBoW3::Vocabulary *pVocPoint,
                         DBoW3::Vocabulary *pVocLine, const bool bFixScale,
                         const bool bActiveLC)
    : mbResetRequested(false), mbResetActiveMapRequested(false),
      mbFinishRequested(false), mbFinished(true), mpAtlas(pAtlas),
      mpKeyFrameDB(pDB), mpVocPoint(pVocPoint), mpVocLine(pVocLine),
      mpMatchedKF(NULL), mLastLoopKFid(0), mbRunningGBA(false),
      mbFinishedGBA(true), mbStopGBA(false), mpThreadGBA(NULL),
      mbFixScale(bFixScale), mnFullBAIdx(0), mnLoopNumCoincidences(0),
      mnMergeNumCoincidences(0), mbLoopDetected(false), mbMergeDetected(false),
      mnLoopNumNotFound(0), mnMergeNumNotFound(0), mbActiveLC(bActiveLC) {
    mnCovisibilityConsistencyTh = 3;
    mpLastCurrentKF = nullptr;
    mpCurrentKF = nullptr;

#ifdef REGISTER_TIMES

    vdDataQuery_ms.clear();
    vdEstSim3_ms.clear();
    vdPRTotal_ms.clear();

    vdMergeMaps_ms.clear();
    vdWeldingBA_ms.clear();
    vdMergeOptEss_ms.clear();
    vdMergeTotal_ms.clear();
    vnMergeKFs.clear();
    vnMergeMPs.clear();
    nMerges = 0;

    vdLoopFusion_ms.clear();
    vdLoopOptEss_ms.clear();
    vdLoopTotal_ms.clear();
    vnLoopKFs.clear();
    nLoop = 0;

    vdGBA_ms.clear();
    vdUpdateMap_ms.clear();
    vdFGBATotal_ms.clear();
    vnGBAKFs.clear();
    vnGBAMPs.clear();
    nFGBA_exec = 0;
    nFGBA_abort = 0;

#endif

    mstrFolderSubTraj = "SubTrajectories/";
    mnNumCorrection = 0;
    mnCorrectionGBA = 0;
}

void LoopClosing::SetTracker(Tracking *pTracker) { mpTracker = pTracker; }

void LoopClosing::SetLocalMapper(LocalMapping *pLocalMapper) {
    mpLocalMapper = pLocalMapper;
}

void LoopClosing::SetMapSparsification(MapSparsification *map_sparsification) {
    mpMapSparsification = map_sparsification;
}

void LoopClosing::RequestReset() {
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while (1) {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if (!mbResetRequested)
                break;
        }
        usleep(5000);
    }
}

void LoopClosing::RequestResetActiveMap(Map *pMap) {
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetActiveMapRequested = true;
        mpMapToReset = pMap;
    }

    while (1) {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if (!mbResetActiveMapRequested)
                break;
        }
        usleep(3000);
    }
}

bool LoopClosing::CheckNewKeyFrames() {
    unique_lock<mutex> lock(mMutexLoopQueue);
    return (!mlpLoopKeyFrameQueue.empty());
}

void LoopClosing::Run() {
    mbFinished = false;

    while (1) {
        ProcessSparsifiedKeyFrames();

        if (CheckNewKeyFrames()) {
            if (mpLastCurrentKF) {
                mpLastCurrentKF->mvpLoopCandKFs.clear();
                mpLastCurrentKF->mvpMergeCandKFs.clear();
            }

            bool bFindedRegion = NewDetectCommonRegions();
            // cout << "bFindedRegion: " << bFindedRegion << endl;
            // cout << "mbLoopDetected: " << mbLoopDetected << endl;
            // cout << "mbMergeDetected: " << mbMergeDetected << endl;

            if (bFindedRegion) {
                if (mbMergeDetected) {
                    Sophus::SE3d mTmw =
                        mpMergeMatchedKF->GetPose().cast<double>();
                    g2o::SE3Quat gTmw2(mTmw.rotationMatrix(),
                                       mTmw.translation());
                    Sophus::SE3d mTcw = mpCurrentKF->GetPose().cast<double>();
                    g2o::SE3Quat gTcw1(mTcw.rotationMatrix(),
                                       mTcw.translation());
                    g2o::SE3Quat gTw2c = mg2oMergeTlw.inverse();

                    mSold_new = gTw2c * gTcw1;

                    mg2oMergeTmw = gTmw2 * gTw2c * gTcw1;

                    mg2oMergeTcw = mg2oMergeTlw;

                    cout << "PR: Merge detected with SE3" << endl;

                    MergeLocal();

                    vdPR_CurrentTime.push_back(mpCurrentKF->mTimeStamp);
                    vdPR_MatchedTime.push_back(mpMergeMatchedKF->mTimeStamp);
                    vnPR_TypeRecogn.push_back(1);

                    // Reset all variables
                    mpMergeLastCurrentKF->SetErase();
                    mpMergeMatchedKF->SetErase();
                    mnMergeNumCoincidences = 0;
                    mvpMergeMatchedMPs.clear();
                    mvpMergeMatchedMLs.clear();
                    mvpMergeMPs.clear();
                    mvpMergeMLs.clear();
                    mnMergeNumNotFound = 0;
                    mbMergeDetected = false;

                    if (mbLoopDetected) {
                        // Reset Loop variables
                        mpLoopLastCurrentKF->SetErase();
                        mpLoopMatchedKF->SetErase();
                        mnLoopNumCoincidences = 0;
                        mvpLoopMatchedMPs.clear();
                        mvpLoopMatchedMLs.clear();
                        mvpLoopMPs.clear();
                        mvpLoopMLs.clear();
                        mnLoopNumNotFound = 0;
                        mbLoopDetected = false;
                    }
                }

                if (mbLoopDetected) {
                    vdPR_CurrentTime.push_back(mpCurrentKF->mTimeStamp);
                    vdPR_MatchedTime.push_back(mpLoopMatchedKF->mTimeStamp);
                    vnPR_TypeRecogn.push_back(0);

                    cout << "PR: Loop detected with SE3" << endl;

                    mg2oLoopTbw = mg2oLoopTlw;

                    mvpLoopMapPoints = mvpLoopMPs;
                    mvpLoopMapLines = mvpLoopMLs;

                    CorrectLoop();

                    mnNumCorrection++;

                    mpLoopLastCurrentKF->SetErase();
                    mpLoopMatchedKF->SetErase();
                    mnLoopNumCoincidences = 0;
                    mvpLoopMatchedMPs.clear();
                    mvpLoopMatchedMLs.clear();
                    mvpLoopMPs.clear();
                    mvpLoopMLs.clear();
                    mnLoopNumNotFound = 0;
                    mbLoopDetected = false;
                }
            }
            mpLastCurrentKF = mpCurrentKF;
        }

        ResetIfRequested();

        if (CheckFinish()) {
            break;
        }

        usleep(5000);
    }
    SetFinish();
}

bool LoopClosing::NewDetectCommonRegions() {
    if (!mbActiveLC)
        return false;

    {
        unique_lock<mutex> lock(mMutexLoopQueue);
        mpCurrentKF = mlpLoopKeyFrameQueue.front();
        mlpLoopKeyFrameQueue.pop_front();

        mpCurrentKF->SetNotErase();
        mpCurrentKF->mbCurrentPlaceRecognition = true;

        mpLastMap = mpCurrentKF->GetMap();
    }

    if (mpLastMap->GetAllKeyFrames().size() < 12) {
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }

    bool bLoopDetectedInKF = false;

    if (mnLoopNumCoincidences > 0) {
        Sophus::SE3d mTcl =
            (mpCurrentKF->GetPose() * mpLoopLastCurrentKF->GetPoseInverse())
                .cast<double>();
        g2o::SE3Quat gTcl(mTcl.rotationMatrix(), mTcl.translation());
        g2o::SE3Quat gTcw = gTcl * mg2oLoopTlw;
        int numProjMatches = 0;
        vector<vector<MapPointPtr>> vpMatchedMPs;
        vector<vector<MapLinePtr>> vpMatchedMLs;
        bool bCommonRegion = DetectAndRefinePoseFromLastKF(
            mpCurrentKF, mpLoopMatchedKF, gTcw, numProjMatches, mvpLoopMPs,
            mvpLoopMLs, vpMatchedMPs, vpMatchedMLs);

        if (bCommonRegion) {
            bLoopDetectedInKF = true;

            mnLoopNumCoincidences++;
            mpLoopLastCurrentKF->SetErase();
            mpLoopLastCurrentKF = mpCurrentKF;
            mg2oLoopTlw = gTcw;
            mvpLoopMatchedMPs = vpMatchedMPs;
            mvpLoopMatchedMLs = vpMatchedMLs;

            mbLoopDetected = mnLoopNumCoincidences >= 3;
            mnLoopNumNotFound = 0;

            if (!mbLoopDetected) {
                cout << "PR: Loop detected with SE3" << endl;
            }
        } else {
            bLoopDetectedInKF = false;

            mnLoopNumNotFound++;
            if (mnLoopNumNotFound >= 2) {
                mpLoopLastCurrentKF->SetErase();
                mpLoopMatchedKF->SetErase();
                mnLoopNumCoincidences = 0;
                mvpLoopMatchedMPs.clear();
                mvpLoopMatchedMLs.clear();
                mvpLoopMPs.clear();
                mvpLoopMLs.clear();
                mnLoopNumNotFound = 0;
            }
        }
    }

    bool bMergeDetectedInKF = false;
    if (mnMergeNumCoincidences > 0) {
        Sophus::SE3d mTcl =
            (mpCurrentKF->GetPose() * mpMergeLastCurrentKF->GetPoseInverse())
                .cast<double>();

        g2o::SE3Quat gTcl(mTcl.rotationMatrix(), mTcl.translation());
        g2o::SE3Quat gTcw = gTcl * mg2oMergeTlw;
        int numProjMatches = 0;
        vector<vector<MapPointPtr>> vpMatchedMPs;
        vector<vector<MapLinePtr>> vpMatchedMLs;
        bool bCommonRegion = DetectAndRefinePoseFromLastKF(
            mpCurrentKF, mpMergeMatchedKF, gTcw, numProjMatches, mvpMergeMPs,
            mvpMergeMLs, vpMatchedMPs, vpMatchedMLs);
        if (bCommonRegion) {
            bMergeDetectedInKF = true;

            mnMergeNumCoincidences++;
            mpMergeLastCurrentKF->SetErase();
            mpMergeLastCurrentKF = mpCurrentKF;
            mg2oMergeTlw = gTcw;
            mvpMergeMatchedMPs = vpMatchedMPs;
            mvpMergeMatchedMLs = vpMatchedMLs;

            mbMergeDetected = mnMergeNumCoincidences >= 3;
        } else {
            mbMergeDetected = false;
            bMergeDetectedInKF = false;

            mnMergeNumNotFound++;
            if (mnMergeNumNotFound >= 2) {
                mpMergeLastCurrentKF->SetErase();
                mpMergeMatchedKF->SetErase();
                mnMergeNumCoincidences = 0;
                mvpMergeMatchedMPs.clear();
                mvpMergeMatchedMLs.clear();
                mvpMergeMPs.clear();
                mvpMergeMLs.clear();
                mnMergeNumNotFound = 0;
            }
        }
    }

    if (mbMergeDetected || mbLoopDetected) {
        mpKeyFrameDB->add(mpCurrentKF);
        return true;
    }

    const vector<KeyFramePtr> vpConnectedKeyFrames =
        mpCurrentKF->GetVectorCovisibleKeyFrames();

    vector<KeyFramePtr> vpMergeBowCand, vpLoopBowCand;
    if (!bMergeDetectedInKF || !bLoopDetectedInKF) {
        mpKeyFrameDB->DetectNBestCandidates(mpCurrentKF, vpLoopBowCand,
                                            vpMergeBowCand, 3);
    }

    if (!bLoopDetectedInKF && !vpLoopBowCand.empty()) {
        mbLoopDetected = DetectCommonRegionsFromBoW(
            vpLoopBowCand, mpLoopMatchedKF, mpLoopLastCurrentKF, mg2oLoopTlw,
            mnLoopNumCoincidences, mvpLoopMPs, mvpLoopMLs, mvpLoopMatchedMPs,
            mvpLoopMatchedMLs);
    }

    if (!bMergeDetectedInKF && !vpMergeBowCand.empty()) {
        mbMergeDetected = DetectCommonRegionsFromBoW(
            vpMergeBowCand, mpMergeMatchedKF, mpMergeLastCurrentKF,
            mg2oMergeTlw, mnMergeNumCoincidences, mvpMergeMPs, mvpMergeMLs,
            mvpMergeMatchedMPs, mvpMergeMatchedMLs);
    }

    mpKeyFrameDB->add(mpCurrentKF);

    if (mbLoopDetected || mbMergeDetected) {
        return true;
    }

    mpCurrentKF->SetErase();
    mpCurrentKF->mbCurrentPlaceRecognition = false;

    return false;
}

bool LoopClosing::DetectAndRefinePoseFromLastKF(
    KeyFramePtr pCurrentKF, KeyFramePtr pMatchedKF, g2o::SE3Quat &gTcw,
    int &nNumProjMatches, vector<MapPointPtr> &vpMPs, vector<MapLinePtr> &vpMLs,
    vector<vector<MapPointPtr>> &vpMatchedMPs,
    vector<vector<MapLinePtr>> &vpMatchedMLs) {
    unordered_set<MapPointPtr> spAlreadyMatchedMPs;
    unordered_set<MapLinePtr> spAlreadyMatchedMLs;

    nNumProjMatches = FindMatchesByProjection(
        pCurrentKF, pMatchedKF, gTcw, spAlreadyMatchedMPs, spAlreadyMatchedMLs,
        vpMPs, vpMLs, vpMatchedMPs, vpMatchedMLs);

    int nProjMatches = 30;
    int nProjOptMatches = 50;
    int nProjMatchesRep = 100;

    if (nNumProjMatches >= nProjMatches) {
        Sophus::SE3d mTwm = pMatchedKF->GetPoseInverse().cast<double>();
        g2o::SE3Quat gTwm(mTwm.rotationMatrix(), mTwm.translation());
        g2o::SE3Quat gTcm = gTcw * gTwm;

        int numOptMatches = Optimizer::OptimizeSE3(
            mpCurrentKF, pMatchedKF, vpMatchedMPs, vpMatchedMLs, gTcm, 10);

        if (numOptMatches >= nProjOptMatches) {
            g2o::SE3Quat gTcw_estimation = gTcm * gTwm.inverse();

            nNumProjMatches = FindMatchesByProjection(
                pCurrentKF, pMatchedKF, gTcw_estimation, spAlreadyMatchedMPs,
                spAlreadyMatchedMLs, vpMPs, vpMLs, vpMatchedMPs, vpMatchedMLs);

            if (nNumProjMatches >= nProjMatchesRep) {
                gTcw = gTcw_estimation;
                return true;
            }
        }
    }
    return false;
}

bool LoopClosing::DetectCommonRegionsFromLastKF(
    KeyFramePtr pCurrentKF, KeyFramePtr pMatchedKF, g2o::SE3Quat &gTcw,
    int &nNumProjMatches, std::vector<MapPointPtr> &vpMPs,
    std::vector<MapLinePtr> &vpMLs,
    std::vector<std::vector<MapPointPtr>> &vpMatchedMPs,
    std::vector<std::vector<MapLinePtr>> &vpMatchedMLs) {
    unordered_set<MapPointPtr> spAlreadyMatchedMPs;
    unordered_set<MapLinePtr> spAlreadyMatchedMLs;

    nNumProjMatches = FindMatchesByProjection(
        pCurrentKF, pMatchedKF, gTcw, spAlreadyMatchedMPs, spAlreadyMatchedMLs,
        vpMPs, vpMLs, vpMatchedMPs, vpMatchedMLs);

    int nProjMatches = 30;
    if (nNumProjMatches >= nProjMatches) {
        return true;
    }
    return false;
}

int LoopClosing::FindMatchesByProjection(
    KeyFramePtr pCurrentKF, KeyFramePtr pMatchedKFw, g2o::SE3Quat &gTcw,
    unordered_set<MapPointPtr> &spMatchedMPsOrigin,
    unordered_set<MapLinePtr> &spMatchedMLsOrigin, vector<MapPointPtr> &vpMPs,
    vector<MapLinePtr> &vpMLs, vector<vector<MapPointPtr>> &vpMatchedMPs,
    vector<vector<MapLinePtr>> &vpMatchedMLs) {
    int nNumCovisibles = 10;
    vector<KeyFramePtr> vpCovKFm =
        pMatchedKFw->GetBestCovisibilityKeyFrames(nNumCovisibles);
    int nInitialCov = vpCovKFm.size();
    vpCovKFm.push_back(pMatchedKFw);
    set<KeyFramePtr> spCheckKFs(vpCovKFm.begin(), vpCovKFm.end());
    set<KeyFramePtr> spCurrentCovisibles = pCurrentKF->GetConnectedKeyFrames();
    if (nInitialCov < nNumCovisibles) {
        for (int i = 0; i < nInitialCov; ++i) {
            vector<KeyFramePtr> vpKFs =
                vpCovKFm[i]->GetBestCovisibilityKeyFrames(nNumCovisibles);
            int nInserted = 0;
            int j = 0;
            while (j < vpKFs.size() && nInserted < nNumCovisibles) {
                if (spCheckKFs.find(vpKFs[j]) == spCheckKFs.end() &&
                    spCurrentCovisibles.find(vpKFs[j]) ==
                        spCurrentCovisibles.end()) {
                    spCheckKFs.insert(vpKFs[j]);
                    ++nInserted;
                }
                ++j;
            }
            vpCovKFm.insert(vpCovKFm.end(), vpKFs.begin(), vpKFs.end());
        }
    }

    unordered_set<MapPointPtr> spMapPoints;
    unordered_set<MapLinePtr> spMapLines;
    vpMPs.clear();
    vpMLs.clear();
    vpMatchedMPs.clear();
    vpMatchedMLs.clear();
    for (KeyFramePtr pKF : vpCovKFm) {
        const vector<vector<MapPointPtr>> &vpMPsKF = pKF->GetMapPointMatches();
        const vector<vector<MapLinePtr>> &vpMLsKF = pKF->GetMapLineMatches();

        for (size_t cam_id = 0; cam_id < vpMPsKF.size(); ++cam_id) {
            for (MapPointPtr pMP : vpMPsKF[cam_id]) {
                if (!pMP || pMP->isBad())
                    continue;
                if (spMapPoints.find(pMP) != spMapPoints.end())
                    continue;
                spMapPoints.insert(pMP);
                vpMPs.push_back(pMP);
            }
        }

        for (size_t cam_id = 0; cam_id < vpMLsKF.size(); ++cam_id) {
            for (MapLinePtr pML : vpMLsKF[cam_id]) {
                if (!pML || pML->isBad())
                    continue;
                if (spMapLines.find(pML) != spMapLines.end())
                    continue;
                spMapLines.insert(pML);
                vpMLs.push_back(pML);
            }
        }
    }

    Sophus::SE3f Tcw(gTcw.rotation().cast<float>(),
                     gTcw.translation().cast<float>());
    Matcher matcher(0.9);

    int nCam = pCurrentKF->mnCam;
    vpMatchedMPs.resize(nCam);
    vpMatchedMLs.resize(nCam);
    for (size_t cam_id = 0; cam_id < nCam; ++cam_id) {
        vpMatchedMPs[cam_id].resize(pCurrentKF->Np[cam_id], nullptr);
        vpMatchedMLs[cam_id].resize(pCurrentKF->Nl[cam_id], nullptr);
    }
    int num_matches = matcher.SearchByProjection(
        pCurrentKF, Tcw, vpMPs, vpMLs, vpMatchedMPs, vpMatchedMLs, 3, 3, 0.08,
        0.9, kLoopPointHighThreshold, kLoopLineHighThreshold);
    return num_matches;
}

bool LoopClosing::DetectCommonRegionsFromBoW(
    std::vector<KeyFramePtr> &vpBowCand, KeyFramePtr &pMatchedKF,
    KeyFramePtr &pLastCurrentKF, g2o::SE3Quat &g2oTcw, int &nNumCoincidences,
    std::vector<MapPointPtr> &vpMPs, std::vector<MapLinePtr> &vpMLs,
    std::vector<std::vector<MapPointPtr>> &vpMatchedMPs,
    std::vector<std::vector<MapLinePtr>> &vpMatchedMLs) {
    int nBoWMatches = 20;
    int nBoWInliers = 15;
    int nSE3Inliers = 20;
    int nProjMatches = 50;
    int nProjOptMatches = 80;

    int nCam = mpCurrentKF->mnCam;

    set<KeyFramePtr> spConnectedKeyFrames =
        mpCurrentKF->GetConnectedKeyFrames();

    int nNumCovisibles = 10;

    Matcher matcherBoW(0.9);
    Matcher matcher(0.75);

    KeyFramePtr pBestMatchedKF;
    int nBestMatchesReproj = 0;
    int nBestNumCoindicendes = 0;
    g2o::SE3Quat g2oBestTcw;
    vector<MapPointPtr> vpBestMapPoints;
    vector<MapLinePtr> vpBestMapLines;
    vector<vector<MapPointPtr>> vpBestMatchedMPs;
    vector<vector<MapLinePtr>> vpBestMatchedMLs;

    int numCandidates = vpBowCand.size();
    vector<int> vnStatge(numCandidates, 0);
    vector<int> vnMatchesStage(numCandidates, 0);

    int index = 0;

    for (KeyFramePtr pKFi : vpBowCand) {
        if (!pKFi || pKFi->isBad())
            continue;

        vector<KeyFramePtr> vpCovKFi =
            pKFi->GetBestCovisibilityKeyFrames(nNumCovisibles);
        if (vpCovKFi.empty()) {
            vpCovKFi.push_back(pKFi);
        } else {
            vpCovKFi.push_back(vpCovKFi[0]);
            vpCovKFi[0] = pKFi;
        }

        bool bAbortByNearKF = false;
        for (int j = 0; j < vpCovKFi.size(); ++j) {
            if (spConnectedKeyFrames.find(vpCovKFi[j]) !=
                spConnectedKeyFrames.end()) {
                bAbortByNearKF = true;
                break;
            }
        }
        if (bAbortByNearKF) {
            continue;
        }

        vector<vector<vector<MapPointPtr>>> vvpMatchedMPs;
        vector<vector<vector<MapLinePtr>>> vvpMatchedMLs;
        vvpMatchedMPs.resize(vpCovKFi.size());
        vvpMatchedMLs.resize(vpCovKFi.size());
        set<MapPointPtr> spMatchedMPi;
        set<MapLinePtr> spMatchedMLi;
        int numBoWPointMatches = 0;

        KeyFramePtr pMostBoWMatchesKF = pKFi;
        int nMostBoWNumMatches = 0;

        vector<vector<MapPointPtr>> vpMatchedMPs;
        vector<vector<KeyFramePtr>> vpKeyFrameMatchedMP;
        vector<vector<MapLinePtr>> vpMatchedMLs;
        vector<vector<KeyFramePtr>> vpKeyFrameMatchedML;
        vpMatchedMPs.resize(nCam);
        vpKeyFrameMatchedMP.resize(nCam);
        vpMatchedMLs.resize(nCam);
        vpKeyFrameMatchedML.resize(nCam);
        for (size_t cam_id = 0; cam_id < nCam; ++cam_id) {
            vpMatchedMPs[cam_id].resize(mpCurrentKF->Np[cam_id], nullptr);
            vpKeyFrameMatchedMP[cam_id].resize(mpCurrentKF->Np[cam_id],
                                               nullptr);
            vpMatchedMLs[cam_id].resize(mpCurrentKF->Nl[cam_id], nullptr);
            vpKeyFrameMatchedML[cam_id].resize(mpCurrentKF->Nl[cam_id],
                                               nullptr);
        }

        for (int j = 0; j < vpCovKFi.size(); ++j) {
            if (!vpCovKFi[j] || vpCovKFi[j]->isBad())
                continue;

            int num = matcherBoW.SearchByBow(
                mpCurrentKF, vpCovKFi[j], vvpMatchedMPs[j], vvpMatchedMLs[j],
                kLoopPointHighThreshold, kLoopLineHighThreshold);
            if (num > nMostBoWNumMatches) {
                nMostBoWNumMatches = num;
                // pMostBoWMatchesKF = vpCovKFi[j];
            }
        }

        for (int j = 0; j < vpCovKFi.size(); ++j) {
            for (int c = 0; c < nCam; ++c) {
                for (size_t k = 0; k < vvpMatchedMPs[j][c].size(); ++k) {
                    MapPointPtr pMP = vvpMatchedMPs[j][c][k];
                    if (!pMP || pMP->isBad())
                        continue;

                    if (spMatchedMPi.find(pMP) == spMatchedMPi.end()) {
                        spMatchedMPi.insert(pMP);
                        numBoWPointMatches++;
                        vpMatchedMPs[c][k] = pMP;
                        vpKeyFrameMatchedMP[c][k] = vpCovKFi[j];
                    }
                }

                for (size_t k = 0; k < vvpMatchedMLs[j][c].size(); ++k) {
                    MapLinePtr pML = vvpMatchedMLs[j][c][k];
                    if (!pML || pML->isBad())
                        continue;

                    if (spMatchedMLi.find(pML) == spMatchedMLi.end()) {
                        spMatchedMLi.insert(pML);
                        vpMatchedMLs[c][k] = pML;
                        vpKeyFrameMatchedML[c][k] = vpCovKFi[j];
                    }
                }
            }
        }

        if (numBoWPointMatches >= nBoWMatches) {
            SE3Solver solver = SE3Solver(
                mpCurrentKF, pMostBoWMatchesKF, vpMatchedMPs, vpMatchedMLs,
                vpKeyFrameMatchedMP, vpKeyFrameMatchedML);

            solver.SetRansacParameters(0.99, nBoWInliers, 300);

            bool bNoMore = false;
            int nInliers;
            bool bConverged = false;
            Eigen::Matrix4f mTcm;
            while (!bConverged && !bNoMore) {
                mTcm = solver.iterate(20, bNoMore, nInliers, bConverged);
            }

            if (bConverged) {
                vpCovKFi.clear();
                vpCovKFi = pMostBoWMatchesKF->GetBestCovisibilityKeyFrames(
                    nNumCovisibles);
                vpCovKFi.push_back(pMostBoWMatchesKF);
                set<KeyFramePtr> spCheckKFs(vpCovKFi.begin(), vpCovKFi.end());

                set<MapPointPtr> spMapPoints;
                set<MapLinePtr> spMapLines;
                vector<MapPointPtr> vpMapPoints;
                vector<MapLinePtr> vpMapLines;

                for (KeyFramePtr pKF : vpCovKFi) {
                    const vector<vector<MapPointPtr>> &vpMPsKF =
                        pKF->GetMapPointMatches();
                    const vector<vector<MapLinePtr>> &vpMLsKF =
                        pKF->GetMapLineMatches();

                    for (size_t cam_id = 0; cam_id < vpMPsKF.size(); ++cam_id) {
                        for (MapPointPtr pMP : vpMPsKF[cam_id]) {
                            if (!pMP || pMP->isBad())
                                continue;
                            if (spMapPoints.find(pMP) != spMapPoints.end())
                                continue;
                            spMapPoints.insert(pMP);
                            vpMapPoints.push_back(pMP);
                        }
                    }

                    for (size_t cam_id = 0; cam_id < vpMLsKF.size(); ++cam_id) {
                        for (MapLinePtr pML : vpMLsKF[cam_id]) {
                            if (!pML || pML->isBad())
                                continue;
                            if (spMapLines.find(pML) != spMapLines.end())
                                continue;
                            spMapLines.insert(pML);
                            vpMapLines.push_back(pML);
                        }
                    }
                }

                g2o::SE3Quat gTcm(
                    solver.GetEstimatedRotation().cast<double>(),
                    solver.GetEstimatedTranslation().cast<double>());
                g2o::SE3Quat gTmw(
                    pMostBoWMatchesKF->GetRotation().cast<double>(),
                    pMostBoWMatchesKF->GetTranslation().cast<double>());
                g2o::SE3Quat gTcw = gTcm * gTmw;

                Sophus::SE3f mTcw(gTcw.rotation().cast<float>(),
                                  gTcw.translation().cast<float>());

                vector<vector<MapPointPtr>> vpMatchedMP;
                vector<vector<MapLinePtr>> vpMatchedML;
                vpMatchedMP.resize(nCam);
                vpMatchedML.resize(nCam);
                for (size_t cam_id = 0; cam_id < nCam; ++cam_id) {
                    vpMatchedMP[cam_id].resize(mpCurrentKF->Np[cam_id],
                                               nullptr);
                    vpMatchedML[cam_id].resize(mpCurrentKF->Nl[cam_id],
                                               nullptr);
                }
                int numProjMatches = matcher.SearchByProjection(
                    mpCurrentKF, mTcw, vpMapPoints, vpMapLines, vpMatchedMP,
                    vpMatchedML, 8, 8, 0.12, 0.9, kLoopPointHighThreshold,
                    kLoopLineHighThreshold);

                if (numProjMatches >= nProjMatches) {

                    int numOptMatches = Optimizer::OptimizeSE3(
                        mpCurrentKF, pKFi, vpMatchedMP, vpMatchedML, gTcm, 10);
                    cout << "numOptMatches: " << numOptMatches << endl;

                    if (numOptMatches >= nSE3Inliers) {
                        g2o::SE3Quat gTmw(
                            pMostBoWMatchesKF->GetRotation().cast<double>(),
                            pMostBoWMatchesKF->GetTranslation().cast<double>());
                        g2o::SE3Quat gTcw = gTcm * gTmw;
                        Sophus::SE3f mTcw(gTcw.rotation().cast<float>(),
                                          gTcw.translation().cast<float>());

                        vector<vector<MapPointPtr>> vpMatchedMP;
                        vector<vector<MapLinePtr>> vpMatchedML;
                        vpMatchedMP.resize(nCam);
                        vpMatchedML.resize(nCam);
                        for (size_t cam_id = 0; cam_id < nCam; ++cam_id) {
                            vpMatchedMP[cam_id].resize(mpCurrentKF->Np[cam_id],
                                                       nullptr);
                            vpMatchedML[cam_id].resize(mpCurrentKF->Nl[cam_id],
                                                       nullptr);
                        }
                        int numProjOptMatches = matcher.SearchByProjection(
                            mpCurrentKF, mTcw, vpMapPoints, vpMapLines,
                            vpMatchedMP, vpMatchedML, 5, 5, 0.08, 1.0,
                            kLoopPointHighThreshold, kLoopLineHighThreshold);

                        if (numProjOptMatches >= nProjOptMatches) {
                            int nNumKFs = 0;

                            vector<KeyFramePtr> vpCurrentCovKFs =
                                mpCurrentKF->GetBestCovisibilityKeyFrames(
                                    nNumCovisibles);

                            int j = 0;
                            while (nNumKFs < 3 && j < vpCurrentCovKFs.size()) {
                                KeyFramePtr pKFj = vpCurrentCovKFs[j];
                                Sophus::SE3d mTjc =
                                    (pKFj->GetPose() *
                                     mpCurrentKF->GetPoseInverse())
                                        .cast<double>();
                                g2o::SE3Quat gTjc(mTjc.rotationMatrix(),
                                                  mTjc.translation());
                                g2o::SE3Quat gTjw = gTjc * gTcw;
                                int numProjMatches_j = 0;
                                vector<vector<MapPointPtr>> vpMatchedMP_j;
                                vector<vector<MapLinePtr>> vpMatchedML_j;
                                bool bValid = DetectCommonRegionsFromLastKF(
                                    pKFj, pMostBoWMatchesKF, gTjw,
                                    numProjMatches_j, vpMapPoints, vpMapLines,
                                    vpMatchedMP_j, vpMatchedML_j);

                                if (bValid) {
                                    Sophus::SE3f Tc_w = mpCurrentKF->GetPose();
                                    Sophus::SE3f Tw_cj = pKFj->GetPoseInverse();
                                    Sophus::SE3f Tc_cj = Tc_w * Tw_cj;
                                    Eigen::Vector3f vector_dist =
                                        Tc_cj.translation();
                                    nNumKFs++;
                                }
                                j++;
                            }

                            if (nNumKFs < 3) {
                                vnStatge[index] = 8;
                                vnMatchesStage[index] = nNumKFs;
                            }

                            if (nBestMatchesReproj < numProjOptMatches) {
                                nBestMatchesReproj = numProjOptMatches;
                                nBestNumCoindicendes = nNumKFs;
                                pBestMatchedKF = pMostBoWMatchesKF;
                                g2oBestTcw = gTcw;
                                vpBestMapPoints = vpMapPoints;
                                vpBestMapLines = vpMapLines;
                                vpBestMatchedMPs = vpMatchedMP;
                                vpBestMatchedMLs = vpMatchedML;
                            }
                        }
                    }
                }
            }
        }
        index++;
    }
    if (nBestMatchesReproj > 0) {
        pLastCurrentKF = mpCurrentKF;
        nNumCoincidences = nBestNumCoindicendes;
        pMatchedKF = pBestMatchedKF;
        pMatchedKF->SetNotErase();
        g2oTcw = g2oBestTcw;
        vpMPs = vpBestMapPoints;
        vpMLs = vpBestMapLines;
        vpMatchedMPs = vpBestMatchedMPs;
        vpMatchedMLs = vpBestMatchedMLs;

        return nNumCoincidences >= 3;
    } else {
        int maxStage = -1;
        for (int i = 0; i < vnStatge.size(); ++i) {
            if (vnStatge[i] > maxStage) {
                maxStage = vnStatge[i];
            }
        }
    }

    return false;
}

void LoopClosing::MergeLocal() {
    int numTemporalKFs =
        25; // Temporal KFs in the local window if the map is inertial.

    // Relationship to rebuild the essential graph, it is used two times, first
    // in the local window and later in the rest of the map
    KeyFramePtr pNewChild;
    KeyFramePtr pNewParent;

    vector<KeyFramePtr> vpLocalCurrentWindowKFs;
    vector<KeyFramePtr> vpMergeConnectedKFs;

    // Flag that is true only when we stopped a running BA, in this case we need
    // relaunch at the end of the merge
    bool bRelaunchBA = false;

    // Verbose::PrintMess("MERGE-VISUAL: Check Full Bundle Adjustment",
    // Verbose::VERBOSITY_DEBUG);
    //  If a Global Bundle Adjustment is running, abort it
    if (isRunningGBA()) {
        unique_lock<mutex> lock(mMutexGBA);
        mbStopGBA = true;

        mnFullBAIdx++;

        if (mpThreadGBA) {
            mpThreadGBA->detach();
            delete mpThreadGBA;
        }
        bRelaunchBA = true;
    }

    // Verbose::PrintMess("MERGE-VISUAL: Request Stop Local Mapping",
    // Verbose::VERBOSITY_DEBUG); cout << "Request Stop Local Mapping" << endl;
    StopMapSparsification();
    mpLocalMapper->RequestStop();
    // Wait until Local Mapping has effectively stopped
    while (!mpLocalMapper->isStopped()) {
        usleep(1000);
    }
    // cout << "Local Map stopped" << endl;

    mpLocalMapper->EmptyQueue();

    // Merge map will become in the new active map with the local window of KFs
    // and MPs from the current map. Later, the elements of the current map will
    // be transform to the new active map reference, in order to keep real time
    // tracking
    Map *pCurrentMap = mpCurrentKF->GetMap();
    Map *pMergeMap = mpMergeMatchedKF->GetMap();
    const KeyFramePtr pCurrentOriginKF = pCurrentMap->GetOriginKF();

    // Ensure current keyframe is updated
    mpCurrentKF->UpdateConnections();

    // Get the current KF and its neighbors(visual->covisibles;
    // inertial->temporal+covisibles)
    set<KeyFramePtr> spLocalWindowKFs;
    // Get MPs in the welding area from the current map
    set<MapPointPtr> spLocalWindowMPs;
    set<MapLinePtr> spLocalWindowMLs;

    spLocalWindowKFs.insert(mpCurrentKF);

    vector<KeyFramePtr> vpCovisibleKFs =
        mpCurrentKF->GetBestCovisibilityKeyFrames(numTemporalKFs);
    spLocalWindowKFs.insert(vpCovisibleKFs.begin(), vpCovisibleKFs.end());
    spLocalWindowKFs.insert(mpCurrentKF);
    const int nMaxTries = 5;
    int nNumTries = 0;
    while (spLocalWindowKFs.size() < numTemporalKFs && nNumTries < nMaxTries) {
        vector<KeyFramePtr> vpNewCovKFs;
        for (KeyFramePtr pKFi : spLocalWindowKFs) {
            vector<KeyFramePtr> vpKFiCov =
                pKFi->GetBestCovisibilityKeyFrames(numTemporalKFs / 2);
            for (KeyFramePtr pKFcov : vpKFiCov) {
                if (pKFcov && !pKFcov->isBad() &&
                    spLocalWindowKFs.find(pKFcov) == spLocalWindowKFs.end()) {
                    vpNewCovKFs.push_back(pKFcov);
                }
            }
        }

        spLocalWindowKFs.insert(vpNewCovKFs.begin(), vpNewCovKFs.end());
        nNumTries++;
    }

    for (KeyFramePtr pKFi : spLocalWindowKFs) {
        if (!pKFi || pKFi->isBad())
            continue;

        set<MapPointPtr> spMPs = pKFi->GetMapPoints();
        set<MapLinePtr> spMLs = pKFi->GetMapLines();
        spLocalWindowMPs.insert(spMPs.begin(), spMPs.end());
        spLocalWindowMLs.insert(spMLs.begin(), spMLs.end());
    }

    set<KeyFramePtr> spMergeConnectedKFs;
    spMergeConnectedKFs.insert(mpMergeMatchedKF);
    vpCovisibleKFs =
        mpMergeMatchedKF->GetBestCovisibilityKeyFrames(numTemporalKFs);
    spMergeConnectedKFs.insert(vpCovisibleKFs.begin(), vpCovisibleKFs.end());
    spMergeConnectedKFs.insert(mpMergeMatchedKF);
    nNumTries = 0;
    while (spMergeConnectedKFs.size() < numTemporalKFs &&
           nNumTries < nMaxTries) {
        vector<KeyFramePtr> vpNewCovKFs;
        for (KeyFramePtr pKFi : spMergeConnectedKFs) {
            vector<KeyFramePtr> vpKFiCov =
                pKFi->GetBestCovisibilityKeyFrames(numTemporalKFs / 2);
            for (KeyFramePtr pKFcov : vpKFiCov) {
                if (pKFcov && !pKFcov->isBad() &&
                    spMergeConnectedKFs.find(pKFcov) ==
                        spMergeConnectedKFs.end()) {
                    vpNewCovKFs.push_back(pKFcov);
                }
            }
        }

        spMergeConnectedKFs.insert(vpNewCovKFs.begin(), vpNewCovKFs.end());
        nNumTries++;
    }

    set<MapPointPtr> spMapPointMerge;
    set<MapLinePtr> spMapLineMerge;
    for (KeyFramePtr pKFi : spMergeConnectedKFs) {
        set<MapPointPtr> vpMPs = pKFi->GetMapPoints();
        set<MapLinePtr> vpMLs = pKFi->GetMapLines();
        spMapPointMerge.insert(vpMPs.begin(), vpMPs.end());
        spMapLineMerge.insert(vpMLs.begin(), vpMLs.end());
    }

    vector<MapPointPtr> vpCheckFuseMapPoint;
    vpCheckFuseMapPoint.reserve(spMapPointMerge.size());
    vector<MapLinePtr> vpCheckFuseMapLine;
    vpCheckFuseMapLine.reserve(spMapLineMerge.size());
    copy(spMapPointMerge.begin(), spMapPointMerge.end(),
         back_inserter(vpCheckFuseMapPoint));
    copy(spMapLineMerge.begin(), spMapLineMerge.end(),
         back_inserter(vpCheckFuseMapLine));

    Sophus::SE3d Twc = mpCurrentKF->GetPoseInverse().cast<double>();
    g2o::SE3Quat g2oNonCorrectedTwc(Twc.unit_quaternion(), Twc.translation());
    g2o::SE3Quat g2oNonCorrectedTcw = g2oNonCorrectedTwc.inverse();
    g2o::SE3Quat g2oCorrectedTcw = mg2oMergeTcw;

    KeyFrameAndPose vCorrectedSE3, vNonCorrectedSE3;
    vCorrectedSE3[mpCurrentKF] = g2oCorrectedTcw;
    vNonCorrectedSE3[mpCurrentKF] = g2oNonCorrectedTcw;

    for (KeyFramePtr pKFi : spLocalWindowKFs) {
        if (!pKFi || pKFi->isBad())
            continue;

        g2o::SE3Quat g2oCorrectedTiw;

        if (pKFi != mpCurrentKF) {
            Sophus::SE3d Tiw = pKFi->GetPose().cast<double>();
            g2o::SE3Quat g2oTiw(Tiw.unit_quaternion(), Tiw.translation());
            vNonCorrectedSE3[pKFi] = g2oTiw;

            Sophus::SE3d Tic = Tiw * Twc;
            g2o::SE3Quat g2oTic(Tic.unit_quaternion(), Tic.translation());
            g2oCorrectedTiw = g2oTic * mg2oMergeTcw;
            vCorrectedSE3[pKFi] = g2oCorrectedTiw;
        } else {
            g2oCorrectedTiw = g2oCorrectedTcw;
        }
        pKFi->mTbwMerge = pKFi->GetPose();

        Sophus::SE3d correctedTiw(g2oCorrectedTiw.rotation(),
                                  g2oCorrectedTiw.translation());

        pKFi->mTbwMerge = correctedTiw.cast<float>();
    }

    set<MapPointPtr>::iterator itMP = spLocalWindowMPs.begin();
    while (itMP != spLocalWindowMPs.end()) {
        MapPointPtr pMPi = *itMP;
        if (!pMPi || pMPi->isBad()) {
            itMP = spLocalWindowMPs.erase(itMP);
            continue;
        }

        KeyFramePtr pRefKF = pMPi->GetReferenceKeyFrame();
        if (vCorrectedSE3.find(pRefKF) == vCorrectedSE3.end()) {
            itMP = spLocalWindowMPs.erase(itMP);
            continue;
        }
        g2o::SE3Quat g2oCorrectedTwi = vCorrectedSE3[pRefKF].inverse();
        g2o::SE3Quat g2oNonCorrectedTiw = vNonCorrectedSE3[pRefKF];

        Eigen::Vector3d P3Dw = pMPi->GetWorldPos().cast<double>();
        Eigen::Vector3d eigCorrectedP3Dw =
            g2oCorrectedTwi.map(g2oNonCorrectedTiw.map(P3Dw));
        pMPi->mPosMerge = eigCorrectedP3Dw.cast<float>();

        itMP++;
    }

    set<MapLinePtr>::iterator itML = spLocalWindowMLs.begin();
    while (itML != spLocalWindowMLs.end()) {
        MapLinePtr pMLi = *itML;
        if (!pMLi || pMLi->isBad()) {
            itML = spLocalWindowMLs.erase(itML);
            continue;
        }

        KeyFramePtr pRefKF = pMLi->GetReferenceKeyFrame();
        if (vCorrectedSE3.find(pRefKF) == vCorrectedSE3.end()) {
            itML = spLocalWindowMLs.erase(itML);
            continue;
        }
        g2o::SE3Quat g2oCorrectedTwi = vCorrectedSE3[pRefKF].inverse();
        g2o::SE3Quat g2oNonCorrectedTiw = vNonCorrectedSE3[pRefKF];

        Eigen::Matrix<double, 6, 1> L3Dw = pMLi->GetWorldPos().cast<double>();
        Eigen::Matrix<double, 6, 1> eigCorrectedL3Dw, eigCorrectedL3Di;
        TransformLineToCamera(
            L3Dw, g2oNonCorrectedTiw.rotation().toRotationMatrix(),
            g2oNonCorrectedTiw.translation(), eigCorrectedL3Di);
        TransformLineToCamera(eigCorrectedL3Di,
                              g2oCorrectedTwi.rotation().toRotationMatrix(),
                              g2oCorrectedTwi.translation(), eigCorrectedL3Dw);

        pMLi->mPosMerge = eigCorrectedL3Dw.cast<float>();

        itML++;
    }

    {
        unique_lock<mutex> currentLock(
            pCurrentMap->mMutexMapUpdate); // We update the current map with the
                                           // Merge information
        unique_lock<mutex> mergeLock(
            pMergeMap->mMutexMapUpdate); // We remove the Kfs and MPs in the
                                         // merged area from the old map

        for (KeyFramePtr pKFi : spLocalWindowKFs) {
            if (!pKFi || pKFi->isBad()) {
                continue;
            }

            pKFi->mTbwBefMerge = pKFi->GetPose();
            pKFi->mTwbBefMerge = pKFi->GetPoseInverse();
            pKFi->SetPose(pKFi->mTbwMerge);

            // Make sure connections are updated
            pKFi->UpdateMap(pMergeMap);
            pKFi->mnMergeCorrectedForKF = mpCurrentKF->mnId;
            pMergeMap->AddKeyFrame(pKFi);
            pCurrentMap->EraseKeyFrame(pKFi);
        }

        for (MapPointPtr pMPi : spLocalWindowMPs) {
            if (!pMPi || pMPi->isBad())
                continue;

            pMPi->SetWorldPos(pMPi->mPosMerge);
            pMPi->UpdateMap(pMergeMap);
            pMergeMap->AddMapPoint(pMPi);
            pCurrentMap->EraseMapPoint(pMPi);
        }

        for (MapLinePtr pMLi : spLocalWindowMLs) {
            if (!pMLi || pMLi->isBad())
                continue;

            pMLi->SetWorldPos(pMLi->mPosMerge);
            pMLi->UpdateMap(pMergeMap);
            pMergeMap->AddMapLine(pMLi);
            pCurrentMap->EraseMapLine(pMLi);
        }

        mpAtlas->ChangeMap(pMergeMap);
        mpAtlas->SetMapBad(pCurrentMap);
        pMergeMap->IncreaseChangeIndex();
        // TODO for debug
        pMergeMap->ChangeId(pCurrentMap->GetId());
    }

    // Rebuild the essential graph in the local window
    // EraseKeyFrame clears the map's origin when the merge moves that KF.
    if (pCurrentOriginKF) {
        pCurrentOriginKF->SetFirstConnection(false);
    }
    pNewChild =
        mpCurrentKF
            ->GetParent();    // Old parent, it will be the new child of this KF
    pNewParent = mpCurrentKF; // Old child, now it will be the parent of its own
                              // parent(we need eliminate this KF from children
                              // list in its old parent)
    mpCurrentKF->ChangeParent(mpMergeMatchedKF);
    while (pNewChild) {
        pNewChild->EraseChild(
            pNewParent); // We remove the relation between the old parent and
                         // the new for avoid loop
        KeyFramePtr pOldParent = pNewChild->GetParent();

        pNewChild->ChangeParent(pNewParent);

        pNewParent = pNewChild;
        pNewChild = pOldParent;
    }

    // Update the connections between the local window
    mpMergeMatchedKF->UpdateConnections();

    vpMergeConnectedKFs = mpMergeMatchedKF->GetVectorCovisibleKeyFrames();
    vpMergeConnectedKFs.push_back(mpMergeMatchedKF);

    SearchAndFuse(vCorrectedSE3, vpCheckFuseMapPoint, vpCheckFuseMapLine);

    for (KeyFramePtr pKFi : spLocalWindowKFs) {
        if (!pKFi || pKFi->isBad())
            continue;

        pKFi->UpdateConnections();
    }
    for (KeyFramePtr pKFi : spMergeConnectedKFs) {
        if (!pKFi || pKFi->isBad())
            continue;

        pKFi->UpdateConnections();
    }

    bool bStop = false;
    vpLocalCurrentWindowKFs.clear();
    vpMergeConnectedKFs.clear();
    std::copy(spLocalWindowKFs.begin(), spLocalWindowKFs.end(),
              std::back_inserter(vpLocalCurrentWindowKFs));
    std::copy(spMergeConnectedKFs.begin(), spMergeConnectedKFs.end(),
              std::back_inserter(vpMergeConnectedKFs));
    Optimizer::LocalBundleAdjustment(mpCurrentKF, vpLocalCurrentWindowKFs,
                                     vpMergeConnectedKFs, &bStop);

    mpLocalMapper->Release();

    vector<KeyFramePtr> vpCurrentMapKFs = pCurrentMap->GetAllKeyFrames();
    vector<MapPointPtr> vpCurrentMapMPs = pCurrentMap->GetAllMapPoints();
    vector<MapLinePtr> vpCurrentMapMLs = pCurrentMap->GetAllMapLines();

    if (vpCurrentMapKFs.size() != 0) {
        mpLocalMapper->RequestStop();
        while (!mpLocalMapper->isStopped()) {
            usleep(1000);
        }
        Optimizer::OptimizeEssentialGraph(
            mpCurrentKF, vpMergeConnectedKFs, vpLocalCurrentWindowKFs,
            vpCurrentMapKFs, vpCurrentMapMPs, vpCurrentMapMLs);

        {
            unique_lock<mutex> currentLock(pCurrentMap->mMutexMapUpdate);
            unique_lock<mutex> mergeLock(pMergeMap->mMutexMapUpdate);

            for (KeyFramePtr pKFi : vpCurrentMapKFs) {
                if (!pKFi || pKFi->isBad() || pKFi->GetMap() != pCurrentMap) {
                    continue;
                }
                // Make sure connections are updated
                pKFi->UpdateMap(pMergeMap);
                pMergeMap->AddKeyFrame(pKFi);
                pCurrentMap->EraseKeyFrame(pKFi);
            }

            for (MapPointPtr pMPi : vpCurrentMapMPs) {
                if (!pMPi || pMPi->isBad())
                    continue;

                pMPi->UpdateMap(pMergeMap);
                pMergeMap->AddMapPoint(pMPi);
                pCurrentMap->EraseMapPoint(pMPi);
            }

            for (MapLinePtr pMLi : vpCurrentMapMLs) {
                if (!pMLi || pMLi->isBad())
                    continue;

                pMLi->UpdateMap(pMergeMap);
                pMergeMap->AddMapLine(pMLi);
                pCurrentMap->EraseMapLine(pMLi);
            }
        }
    }

    mpLocalMapper->Release();
    ReleaseMapSparsification();

    if (bRelaunchBA &&
        (pCurrentMap->KeyFramesInMap() < 200 && mpAtlas->CountMaps() == 1)) {
        mbRunningGBA = true;
        mbFinishedGBA = false;
        mbStopGBA = false;
        mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment, this,
                                 pMergeMap, mpCurrentKF->mnId);
    }

    mpMergeMatchedKF->AddMergeEdge(mpCurrentKF);
    mpCurrentKF->AddMergeEdge(mpMergeMatchedKF);

    pCurrentMap->IncreaseChangeIndex();
    pMergeMap->IncreaseChangeIndex();

    mpAtlas->RemoveBadMaps();
}

void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap,
                                vector<MapPointPtr> &vpMapPoints,
                                vector<MapLinePtr> &vpMapLines) {
    Matcher matcher(0.8);

    for (auto mit = CorrectedPosesMap.begin(); mit != CorrectedPosesMap.end();
         ++mit) {
        KeyFramePtr pKFi = mit->first;
        Map *pMap = pKFi->GetMap();

        g2o::SE3Quat g2oTbw = mit->second;
        Sophus::SE3f Tbw(g2oTbw.rotation().cast<float>(),
                         g2oTbw.translation().cast<float>());

        vector<MapPointPtr> vpReplacePoints(vpMapPoints.size(), nullptr);
        vector<MapLinePtr> vpReplaceLines(vpMapLines.size(), nullptr);

        int numFused = matcher.Fuse(
            pKFi, Tbw, vpMapPoints, vpMapLines, 4, 4, 0.1, vpReplacePoints,
            vpReplaceLines, kLoopPointHighThreshold, kLoopLineHighThreshold);

        unique_lock<mutex> lock(pMap->mMutexMapUpdate);
        const int nLP = vpMapPoints.size();
        const int nLL = vpMapLines.size();
        for (int i = 0; i < nLP; ++i) {
            MapPointPtr pRep = vpReplacePoints[i];
            if (pRep) {
                pRep->Replace(vpMapPoints[i]);
            }
        }
        for (int i = 0; i < nLL; ++i) {
            MapLinePtr pRep = vpReplaceLines[i];
            if (pRep) {
                pRep->Replace(vpMapLines[i]);
            }
        }
    }
}

void LoopClosing::ResetIfRequested() {
    unique_lock<mutex> lock(mMutexReset);
    if (mbResetRequested) {
        cout << "Loop closer reset requested..." << endl;
        mlpLoopKeyFrameQueue.clear();
        {
            lock_guard<mutex> sparse_lock(mMutexSparsifiedQueue);
            mlpSparsifiedKeyFrameQueue.clear();
        }
        mLastLoopKFid =
            0; // TODO old variable, it is not use in the new algorithm
        mbResetRequested = false;
        mbResetActiveMapRequested = false;
    } else if (mbResetActiveMapRequested) {

        for (list<KeyFramePtr>::const_iterator it =
                 mlpLoopKeyFrameQueue.begin();
             it != mlpLoopKeyFrameQueue.end();) {
            KeyFramePtr pKFi = *it;
            if (pKFi->GetMap() == mpMapToReset) {
                it = mlpLoopKeyFrameQueue.erase(it);
            } else
                ++it;
        }

        {
            lock_guard<mutex> sparse_lock(mMutexSparsifiedQueue);
            for (auto it = mlpSparsifiedKeyFrameQueue.begin();
                 it != mlpSparsifiedKeyFrameQueue.end();) {
                if (!*it || (*it)->GetMap() == mpMapToReset)
                    it = mlpSparsifiedKeyFrameQueue.erase(it);
                else
                    ++it;
            }
        }

        mLastLoopKFid =
            mpAtlas->GetLastInitKFid(); // TODO old variable, it is not use in
                                        // the new algorithm
        mbResetActiveMapRequested = false;
    }
}

bool LoopClosing::CheckFinish() {
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LoopClosing::SetFinish() {
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

void LoopClosing::CorrectLoop() {
    // Send a stop signal to Local Mapping
    // Avoid new keyframes are inserted while correcting the loop
    StopMapSparsification();
    mpLocalMapper->RequestStop();
    mpLocalMapper->EmptyQueue(); // Proccess keyframes in the queue

    int nCam = mpCurrentKF->mnCam;

    // If a Global Bundle Adjustment is running, abort it
    if (isRunningGBA()) {
        cout << "Stoping Global Bundle Adjustment...";
        unique_lock<mutex> lock(mMutexGBA);
        mbStopGBA = true;

        mnFullBAIdx++;

        if (mpThreadGBA) {
            mpThreadGBA->detach();
            delete mpThreadGBA;
        }
        cout << "  Done!!" << endl;
    }

    // Wait until Local Mapping has effectively stopped
    while (!mpLocalMapper->isStopped()) {
        usleep(1000);
    }

    // Ensure current keyframe is updated
    // cout << "Start updating connections" << endl;
    // assert(mpCurrentKF->GetMap()->CheckEssentialGraph());
    mpCurrentKF->UpdateConnections();
    // assert(mpCurrentKF->GetMap()->CheckEssentialGraph());

    // Retrive keyframes connected to the current keyframe and compute corrected
    // Sim3 pose by propagation
    mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
    mvpCurrentConnectedKFs.push_back(mpCurrentKF);

    KeyFrameAndPose CorrectedSE3, NonCorrectedSE3;
    CorrectedSE3[mpCurrentKF] = mg2oLoopTbw;
    Sophus::SE3f Twb = mpCurrentKF->GetPoseInverse();
    Sophus::SE3f Tbw = mpCurrentKF->GetPose();
    g2o::SE3Quat g2oTbw(Tbw.unit_quaternion().cast<double>(),
                        Tbw.translation().cast<double>());
    NonCorrectedSE3[mpCurrentKF] = g2oTbw;

    Sophus::SE3d correctedTbw(mg2oLoopTbw.rotation(),
                              mg2oLoopTbw.translation());
    mpCurrentKF->SetPose(correctedTbw.cast<float>());

    Map *pLoopMap = mpCurrentKF->GetMap();

    {
        unique_lock<mutex> lock(pLoopMap->mMutexMapUpdate);
        for (vector<KeyFramePtr>::iterator vit = mvpCurrentConnectedKFs.begin(),
                                           vend = mvpCurrentConnectedKFs.end();
             vit != vend; vit++) {
            KeyFramePtr pKFi = *vit;

            if (pKFi != mpCurrentKF) {
                Sophus::SE3f Tiw = pKFi->GetPose();
                Sophus::SE3d Tic = (Tiw * Twb).cast<double>();
                g2o::SE3Quat g2oTic(Tic.unit_quaternion(), Tic.translation());
                g2o::SE3Quat g2oCorrectedTiw = g2oTic * mg2oLoopTbw;
                // Pose corrected with the Sim3 of the loop closure
                CorrectedSE3[pKFi] = g2oCorrectedTiw;

                // Update keyframe pose with corrected Sim3. First transform
                // Sim3 to SE3 (scale translation)
                Sophus::SE3d correctedTiw(g2oCorrectedTiw.rotation(),
                                          g2oCorrectedTiw.translation());
                pKFi->SetPose(correctedTiw.cast<float>());

                // Pose without correction
                g2o::SE3Quat g2oTiw(Tiw.unit_quaternion().cast<double>(),
                                    Tiw.translation().cast<double>());
                NonCorrectedSE3[pKFi] = g2oTiw;
            }
        }

        for (auto mit = CorrectedSE3.begin(); mit != CorrectedSE3.end();
             ++mit) {
            KeyFramePtr pKFi = mit->first;
            g2o::SE3Quat g2oCorrectedTiw = mit->second;
            g2o::SE3Quat g2oCorrectedTwi = g2oCorrectedTiw.inverse();

            g2o::SE3Quat g2oTiw = NonCorrectedSE3[pKFi];
            vector<vector<MapPointPtr>> vpMPsi = pKFi->GetMapPointMatches();
            vector<vector<MapLinePtr>> vpMLi = pKFi->GetMapLineMatches();

            for (size_t cam_id = 0; cam_id < vpMPsi.size(); ++cam_id) {
                for (MapPointPtr pMPi : vpMPsi[cam_id]) {
                    if (!pMPi || pMPi->isBad() ||
                        pMPi->mnCorrectedByKF == mpCurrentKF->mnId)
                        continue;

                    Eigen::Vector3d P3Dw = pMPi->GetWorldPos().cast<double>();
                    Eigen::Vector3d eigCorrectedP3Dw =
                        g2oCorrectedTwi.map(g2oTiw.map(P3Dw));

                    pMPi->SetWorldPos(eigCorrectedP3Dw.cast<float>());
                    pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
                    pMPi->mnCorrectedReference = pKFi->mnId;
                }

                for (MapLinePtr pMLi : vpMLi[cam_id]) {
                    if (!pMLi || pMLi->isBad() ||
                        pMLi->mnCorrectedByKF == mpCurrentKF->mnId)
                        continue;

                    Eigen::Matrix<double, 6, 1> L3Dw =
                        pMLi->GetWorldPos().cast<double>();
                    Eigen::Matrix<double, 6, 1> eigCorrectedL3Di;
                    Eigen::Matrix<double, 6, 1> eigCorrectedL3Dw;
                    TransformLineToCamera(
                        L3Dw, g2oTiw.rotation().toRotationMatrix(),
                        g2oTiw.translation(), eigCorrectedL3Di);
                    TransformLineToCamera(
                        eigCorrectedL3Di,
                        g2oCorrectedTwi.rotation().toRotationMatrix(),
                        g2oCorrectedTwi.translation(), eigCorrectedL3Dw);

                    pMLi->SetWorldPos(eigCorrectedL3Dw.cast<float>());
                    pMLi->mnCorrectedByKF = mpCurrentKF->mnId;
                    pMLi->mnCorrectedReference = pKFi->mnId;
                }
            }
            pKFi->UpdateConnections();
        }

        pLoopMap->IncreaseChangeIndex();

        for (size_t c = 0; c < nCam; ++c) {
            for (size_t i = 0; i < mvpLoopMatchedMPs[c].size(); ++i) {
                MapPointPtr pLoopMP = mvpLoopMatchedMPs[c][i];
                if (!pLoopMP)
                    continue;

                MapPointPtr pCurMP = mpCurrentKF->GetMapPoint(i, c);

                if (pCurMP)
                    pCurMP->Replace(pLoopMP);
                else {
                    mpCurrentKF->AddMapPoint(pLoopMP, i, c);
                    pLoopMP->AddObservation(mpCurrentKF, i, c);
                    pLoopMP->ComputeDistinctiveDescriptors();
                }
            }

            for (size_t i = 0; i < mvpLoopMatchedMLs[c].size(); ++i) {
                MapLinePtr pLoopML = mvpLoopMatchedMLs[c][i];
                if (!pLoopML)
                    continue;

                MapLinePtr pCurML = mpCurrentKF->GetMapLine(i, c);

                if (pCurML)
                    pCurML->Replace(pLoopML);
                else {
                    mpCurrentKF->AddMapLine(pLoopML, i, c);
                    pLoopML->AddObservation(mpCurrentKF, i, c);
                    pLoopML->ComputeDistinctiveDescriptors();
                }
            }
        }
    }

    SearchAndFuse(CorrectedSE3, mvpLoopMapPoints, mvpLoopMapLines);

    map<KeyFramePtr, set<KeyFramePtr>> LoopConnections;

    for (vector<KeyFramePtr>::iterator vit = mvpCurrentConnectedKFs.begin(),
                                       vend = mvpCurrentConnectedKFs.end();
         vit != vend; vit++) {
        KeyFramePtr pKFi = *vit;
        vector<KeyFramePtr> vpPreviousNeighbors =
            pKFi->GetVectorCovisibleKeyFrames();

        // Update connections. Detect new links.
        pKFi->UpdateConnections();
        LoopConnections[pKFi] = pKFi->GetConnectedKeyFrames();
        for (vector<KeyFramePtr>::iterator
                 vit_prev = vpPreviousNeighbors.begin(),
                 vend_prev = vpPreviousNeighbors.end();
             vit_prev != vend_prev; vit_prev++) {
            LoopConnections[pKFi].erase(*vit_prev);
        }
        for (vector<KeyFramePtr>::iterator
                 vit2 = mvpCurrentConnectedKFs.begin(),
                 vend2 = mvpCurrentConnectedKFs.end();
             vit2 != vend2; vit2++) {
            LoopConnections[pKFi].erase(*vit2);
        }
    }

#ifdef XPLSLAM_ENABLE_PROFILING
    cout << "Loop connections: " << LoopConnections.size() << endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
#endif
    Optimizer::OptimizeEssentialGraph(pLoopMap, mpLoopMatchedKF, mpCurrentKF,
                                      NonCorrectedSE3, CorrectedSE3,
                                      LoopConnections, mbFixScale);
#ifdef XPLSLAM_ENABLE_PROFILING
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used =
        chrono::duration_cast<chrono::duration<double>>(t2 - t1) * 1000;
    cout << "OptimizeEssentialGraph time cost: " << time_used.count() << " ms"
         << endl;
#endif

    cout << "loop frame id: " << mpCurrentKF->mnId
         << ", loop matched frame id: " << mpLoopMatchedKF->mnId << endl;

    mpAtlas->InformNewBigChange();

    mpLoopMatchedKF->AddLoopEdge(mpCurrentKF);
    mpCurrentKF->AddLoopEdge(mpLoopMatchedKF);

    if (pLoopMap->KeyFramesInMap() < 200 && mpAtlas->CountMaps() == 1) {
        mbRunningGBA = true;
        mbFinishedGBA = false;
        mbStopGBA = false;
        mnCorrectionGBA = mnNumCorrection;

        mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment, this,
                                 pLoopMap, mpCurrentKF->mnId);
    }

    mpLocalMapper->Release();
    ReleaseMapSparsification();

    mLastLoopKFid = mpCurrentKF->mnId;
    cout << "Loop closed!" << endl;
}

void LoopClosing::RunGlobalBundleAdjustment(Map *pActiveMap,
                                            unsigned long nLoopKF) {
    cout << "Starting Global Bundle Adjustment" << endl;

    Optimizer::GlobalBundleAdjustemnt(pActiveMap, 10, &mbStopGBA, nLoopKF,
                                      false);

    int idx = mnFullBAIdx;

    {
        unique_lock<mutex> lock(mMutexGBA);
        if (idx != mnFullBAIdx)
            return;

        if (!mbStopGBA) {
            cout << "Global Bundle Adjustment finished" << endl;
            StopMapSparsification();
            mpLocalMapper->RequestStop();
            // Wait until Local Mapping has effectively stopped

            while (!mpLocalMapper->isStopped() &&
                   !mpLocalMapper->isFinished()) {
                usleep(1000);
            }

            unique_lock<mutex> lock(pActiveMap->mMutexMapUpdate);

            list<KeyFramePtr> lpKFtoCheck(
                pActiveMap->mvpKeyFrameOrigins.begin(),
                pActiveMap->mvpKeyFrameOrigins.end());

            while (!lpKFtoCheck.empty()) {
                KeyFramePtr pKF = lpKFtoCheck.front();
                const set<KeyFramePtr> sChilds = pKF->GetChilds();

                Sophus::SE3f Twb = pKF->GetPoseInverse();

                for (set<KeyFramePtr>::const_iterator sit = sChilds.begin();
                     sit != sChilds.end(); sit++) {
                    KeyFramePtr pChild = *sit;
                    if (!pChild || pChild->isBad())
                        continue;

                    if (pChild->mnBAGlobalForKF != nLoopKF) {

                        Sophus::SE3f Tchildb = pChild->GetPose() * Twb;

                        pChild->mTbwGBA =
                            Tchildb * pKF->mTbwGBA; //*Tcorc*pKF->mTcwGBA;

                        Sophus::SO3f Rcor = pChild->mTbwGBA.so3().inverse() *
                                            pChild->GetPose().so3();

                        pChild->mnBAGlobalForKF = nLoopKF;
                    }
                    lpKFtoCheck.push_back(pChild);
                }

                // cout << "-------Update pose" << endl;
                pKF->mTbwBefGBA = pKF->GetPose();
                // cout << "pKF->mTcwBefGBA: " << pKF->mTcwBefGBA << endl;
                pKF->SetPose(pKF->mTbwGBA);

                lpKFtoCheck.pop_front();
            }

            const vector<MapPointPtr> vpMPs = pActiveMap->GetAllMapPoints();
            const vector<MapLinePtr> vpMLs = pActiveMap->GetAllMapLines();

            for (size_t i = 0; i < vpMPs.size(); ++i) {
                MapPointPtr pMP = vpMPs[i];

                if (pMP->isBad())
                    continue;

                if (pMP->mnBAGlobalForKF == nLoopKF) {
                    // If optimized by Global BA, just update
                    pMP->SetWorldPos(pMP->mPosGBA);
                } else {
                    // Update according to the correction of its reference
                    // keyframe
                    KeyFramePtr pRefKF = pMP->GetReferenceKeyFrame();

                    if (pRefKF->mnBAGlobalForKF != nLoopKF)
                        continue;

                    Eigen::Vector3f Xb =
                        pRefKF->mTbwBefGBA * pMP->GetWorldPos();

                    // Backproject using corrected camera
                    pMP->SetWorldPos(pRefKF->GetPoseInverse() * Xb);
                }
            }

            for (size_t i = 0; i < vpMLs.size(); ++i) {
                MapLinePtr pML = vpMLs[i];

                if (pML->isBad())
                    continue;

                if (pML->mnBAGlobalForKF == nLoopKF) {
                    // If optimized by Global BA, just update
                    pML->SetWorldPos(pML->mPosGBA);
                } else {
                    // Update according to the correction of its reference
                    // keyframe
                    KeyFramePtr pRefKF = pML->GetReferenceKeyFrame();

                    if (pRefKF->mnBAGlobalForKF != nLoopKF)
                        continue;

                    Eigen::Matrix<float, 6, 1> Lb, Lw;
                    TransformLineToCamera(pML->GetWorldPos(),
                                          pRefKF->mTbwBefGBA, Lb);
                    TransformLineToCamera(Lb, pRefKF->GetPoseInverse(), Lw);
                    pML->SetWorldPos(Lw);
                }
            }

            pActiveMap->InformNewBigChange();
            pActiveMap->IncreaseChangeIndex();

            mpLocalMapper->Release();
            ReleaseMapSparsification();
        }

        mbFinishedGBA = true;
        mbRunningGBA = false;
    }
}

void LoopClosing::InsertKeyFrame(KeyFramePtr pKF) {
    unique_lock<mutex> lock(mMutexLoopQueue);
    if (pKF->mnId != 0)
        mlpLoopKeyFrameQueue.push_back(pKF);
}

void LoopClosing::InsertSparsifiedKeyFrame(KeyFramePtr pKF) {
    lock_guard<mutex> lock(mMutexSparsifiedQueue);
    mlpSparsifiedKeyFrameQueue.push_back(pKF);
}

void LoopClosing::ProcessSparsifiedKeyFrames() {
    list<KeyFramePtr> pending;
    {
        lock_guard<mutex> lock(mMutexSparsifiedQueue);
        pending.swap(mlpSparsifiedKeyFrameQueue);
    }

    for (const KeyFramePtr &keyframe : pending) {
        if (!keyframe || keyframe->isBad())
            continue;

        bool waiting_for_place_recognition = false;
        {
            lock_guard<mutex> lock(mMutexLoopQueue);
            waiting_for_place_recognition =
                std::find(mlpLoopKeyFrameQueue.begin(),
                          mlpLoopKeyFrameQueue.end(),
                          keyframe) != mlpLoopKeyFrameQueue.end();
        }

        bool compacted = false;
        while (!compacted) {
            KeyFrame::PointFeatureCompaction compaction =
                keyframe->PreparePointFeatureCompaction();
            if (!compaction.valid)
                break;

            Map *map = keyframe->GetMap();
            if (!map)
                break;
            unique_lock<mutex> map_update_lock(map->mMutexMapUpdate);
            if (keyframe->isBad() || keyframe->GetMap() != map)
                break;

            if (!waiting_for_place_recognition)
                mpKeyFrameDB->erase(keyframe);
            compacted = keyframe->CommitPointFeatureCompaction(
                std::move(compaction));
            if (!waiting_for_place_recognition)
                mpKeyFrameDB->add(keyframe);
        }
        if (compacted && mpMapSparsification)
            mpMapSparsification->NotifyKeyFrameCompacted();
    }
}

void LoopClosing::StopMapSparsification() {
    if (!mpMapSparsification)
        return;
    mpMapSparsification->RequestStop();
    while (!mpMapSparsification->isStopped())
        usleep(1000);
}

void LoopClosing::ReleaseMapSparsification() {
    if (mpMapSparsification)
        mpMapSparsification->Release();
}

void LoopClosing::RequestFinish() {
    unique_lock<mutex> lock(mMutexFinish);
    // cout << "LC: Finish requested" << endl;
    mbFinishRequested = true;
}

}; // namespace XPLSLAM
