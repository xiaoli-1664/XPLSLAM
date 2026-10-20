#pragma once

#include "Vocabulary.h"
#include "atlas.h"
#include "keyframe.h"
#include "local_mapping.h"
#include "se3quat.h"
#include "tracking.h"

#include "keyframe_database.h"

#include <boost/algorithm/string.hpp>
#include <mutex>
#include <thread>

namespace XPLSLAM {

class Tracking;
class LocalMapping;
class KeyFrameDatabase;
class Map;
class MapSparsification;

class LoopClosing {
  public:
    typedef std::pair<std::set<KeyFramePtr>, int> ConsistentGroup;
    typedef std::map<
        KeyFramePtr, g2o::SE3Quat, std::less<KeyFramePtr>,
        Eigen::aligned_allocator<std::pair<const KeyFramePtr, g2o::SE3Quat>>>
        KeyFrameAndPose;

  public:
    LoopClosing(Atlas *pAtlas, KeyFrameDatabase *pDB,
                DBoW3::Vocabulary *pVocPoint, DBoW3::Vocabulary *pVocLine,
                const bool bFixScale, const bool bActiveLC);

    void SetTracker(Tracking *pTracker);

    void SetLocalMapper(LocalMapping *pLocalMapper);
    void SetMapSparsification(MapSparsification *map_sparsification);

    // Main function
    void Run();

    void InsertKeyFrame(KeyFramePtr pKF);
    void InsertSparsifiedKeyFrame(KeyFramePtr pKF);

    void RequestReset();
    void RequestResetActiveMap(Map *pMap);

    // This function will run in a separate thread
    void RunGlobalBundleAdjustment(Map *pActiveMap, unsigned long nLoopKF);

    bool isRunningGBA() {
        std::unique_lock<std::mutex> lock(mMutexGBA);
        return mbRunningGBA;
    }
    bool isFinishedGBA() {
        std::unique_lock<std::mutex> lock(mMutexGBA);
        return mbFinishedGBA;
    }

    void RequestFinish();

    bool isFinished();

    Viewer *mpViewer;

#ifdef REGISTER_TIMES

    vector<double> vdDataQuery_ms;
    vector<double> vdEstSim3_ms;
    vector<double> vdPRTotal_ms;

    vector<double> vdMergeMaps_ms;
    vector<double> vdWeldingBA_ms;
    vector<double> vdMergeOptEss_ms;
    vector<double> vdMergeTotal_ms;
    vector<int> vnMergeKFs;
    vector<int> vnMergeMPs;
    int nMerges;

    vector<double> vdLoopFusion_ms;
    vector<double> vdLoopOptEss_ms;
    vector<double> vdLoopTotal_ms;
    vector<int> vnLoopKFs;
    int nLoop;

    vector<double> vdGBA_ms;
    vector<double> vdUpdateMap_ms;
    vector<double> vdFGBATotal_ms;
    vector<int> vnGBAKFs;
    vector<int> vnGBAMPs;
    int nFGBA_exec;
    int nFGBA_abort;

#endif

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  protected:
    bool CheckNewKeyFrames();
    void ProcessSparsifiedKeyFrames();
    void StopMapSparsification();
    void ReleaseMapSparsification();

    // Methods to implement the new place recognition algorithm
    bool NewDetectCommonRegions();
    bool DetectAndRefinePoseFromLastKF(
        KeyFramePtr pCurrentKF, KeyFramePtr pMatchedKF, g2o::SE3Quat &gTcw,
        int &nNumProjMatches, std::vector<MapPointPtr> &vpMPs,
        std::vector<MapLinePtr> &vpMLs,
        std::vector<std::vector<MapPointPtr>> &vpMatchedMPs,
        std::vector<std::vector<MapLinePtr>> &vpMatchedMLs);
    bool DetectCommonRegionsFromBoW(
        std::vector<KeyFramePtr> &vpBowCand, KeyFramePtr&pMatchedKF,
        KeyFramePtr&pLastCurrentKF, g2o::SE3Quat &g2oTcw, int &nNumCoincidences,
        std::vector<MapPointPtr> &vpMPs, std::vector<MapLinePtr> &vpMLs,
        std::vector<std::vector<MapPointPtr>> &vpMatchedMPs,
        std::vector<std::vector<MapLinePtr>> &vpMatchedMLs);
    bool DetectCommonRegionsFromLastKF(
        KeyFramePtr pCurrentKF, KeyFramePtr pMatchedKF, g2o::SE3Quat &gTcw,
        int &nNumProjMatches, std::vector<MapPointPtr> &vpMPs,
        std::vector<MapLinePtr> &vpMLs,
        std::vector<std::vector<MapPointPtr>> &vpMatchedMPs,
        std::vector<std::vector<MapLinePtr>> &vpMatchedMLs);

    int FindMatchesByProjection(
        KeyFramePtr pCurrentKF, KeyFramePtr pMatchedKFw, g2o::SE3Quat &g2oTcw,
        std::unordered_set<MapPointPtr> &spMatchedMPinOrigin,
        std::unordered_set<MapLinePtr> &spMatchedMLinOrigin,
        std::vector<MapPointPtr> &vpMapPoints,
        std::vector<MapLinePtr> &vpMapLines,
        std::vector<std::vector<MapPointPtr>> &vpMatchedMapPoints,
        std::vector<std::vector<MapLinePtr>> &vpMatchedMapLines);

    void SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap,
                       std::vector<MapPointPtr> &vpMapPoints,
                       std::vector<MapLinePtr> &vpMapLines);
    void SearchAndFuse(const std::vector<KeyFramePtr> &vConectedKFs,
                       std::vector<MapPointPtr> &vpMapPoints,
                       std::vector<MapLinePtr> &vpMapLines);

    void CorrectLoop();

    void MergeLocal();

    void CheckObservations(std::set<KeyFramePtr> &spKFsMap1,
                           std::set<KeyFramePtr> &spKFsMap2);

    void ResetIfRequested();
    bool mbResetRequested;
    bool mbResetActiveMapRequested;
    Map *mpMapToReset;
    std::mutex mMutexReset;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    Atlas *mpAtlas;
    Tracking *mpTracker;

    KeyFrameDatabase *mpKeyFrameDB;
    DBoW3::Vocabulary *mpVocPoint;
    DBoW3::Vocabulary *mpVocLine;

    LocalMapping *mpLocalMapper;

    std::list<KeyFramePtr> mlpLoopKeyFrameQueue;
    std::list<KeyFramePtr> mlpSparsifiedKeyFrameQueue;

    std::mutex mMutexLoopQueue;
    std::mutex mMutexSparsifiedQueue;
    MapSparsification *mpMapSparsification = nullptr;

    // Loop detector parameters
    float mnCovisibilityConsistencyTh;

    // Loop detector variables
    KeyFramePtr mpCurrentKF;
    KeyFramePtr mpLastCurrentKF;
    KeyFramePtr mpMatchedKF;
    std::vector<ConsistentGroup> mvConsistentGroups;
    std::vector<KeyFramePtr> mvpEnoughConsistentCandidates;
    std::vector<KeyFramePtr> mvpCurrentConnectedKFs;
    std::vector<MapPointPtr> mvpCurrentMatchedPoints;
    std::vector<MapPointPtr> mvpLoopMapPoints;
    std::vector<MapLinePtr> mvpLoopMapLines;
    cv::Mat mScw;
    g2o::Sim3 mg2oScw;

    //-------
    Map *mpLastMap;

    bool mbLoopDetected;
    int mnLoopNumCoincidences;
    int mnLoopNumNotFound;
    KeyFramePtr mpLoopLastCurrentKF;
    g2o::SE3Quat mg2oLoopTlw;
    g2o::SE3Quat mg2oLoopTbw;
    KeyFramePtr mpLoopMatchedKF;
    std::vector<MapPointPtr> mvpLoopMPs;
    std::vector<MapLinePtr> mvpLoopMLs;
    std::vector<std::vector<MapPointPtr>> mvpLoopMatchedMPs;
    std::vector<std::vector<MapLinePtr>> mvpLoopMatchedMLs;

    bool mbMergeDetected;
    int mnMergeNumCoincidences;
    int mnMergeNumNotFound;
    KeyFramePtr mpMergeLastCurrentKF;
    g2o::SE3Quat mg2oMergeTlw;
    g2o::SE3Quat mg2oMergeTmw;
    g2o::SE3Quat mg2oMergeTcw;
    KeyFramePtr mpMergeMatchedKF;
    std::vector<MapPointPtr> mvpMergeMPs;
    std::vector<MapLinePtr> mvpMergeMLs;
    std::vector<std::vector<MapPointPtr>> mvpMergeMatchedMPs;
    std::vector<std::vector<MapLinePtr>> mvpMergeMatchedMLs;
    std::vector<KeyFramePtr> mvpMergeConnectedKFs;

    g2o::SE3Quat mSold_new;
    //-------

    long unsigned int mLastLoopKFid;

    // Variables related to Global Bundle Adjustment
    bool mbRunningGBA;
    bool mbFinishedGBA;
    bool mbStopGBA;
    std::mutex mMutexGBA;
    std::thread *mpThreadGBA;

    // Fix scale in the stereo/RGB-D case
    bool mbFixScale;

    int mnFullBAIdx;

    std::vector<double> vdPR_CurrentTime;
    std::vector<double> vdPR_MatchedTime;
    std::vector<int> vnPR_TypeRecogn;

    // DEBUG
    std::string mstrFolderSubTraj;
    int mnNumCorrection;
    int mnCorrectionGBA;

    // To (de)activate LC
    bool mbActiveLC = true;

#ifdef REGISTER_LOOP
    string mstrFolderLoop;
#endif
};

} // namespace XPLSLAM
