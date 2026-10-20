#pragma once

#include "atlas.h"
#include "keyframe.h"
#include "keyframe_database.h"
#include "settings.h"
#include "tracking.h"

#include <mutex>

namespace XPLSLAM {

class System;
class Tracking;
class LoopClosing;
class Atlas;
class MapSparsification;

class LocalMapping {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LocalMapping(System *pSys, Atlas *pAtlas,
                 const std::string &_strSeqName = std::string());

    void SetLoopCloser(LoopClosing *pLoopCloser);

    void SetTracker(Tracking *pTracker);
    void SetMapSparsification(MapSparsification *map_sparsification,
                              int non_local_threshold);

    // Main function
    void Run();

    void InsertKeyFrame(KeyFramePtr pKF);
    void EmptyQueue();

    // Thread Synch
    void RequestStop();
    void RequestReset();
    void RequestResetActiveMap(Map *pMap);
    bool Stop();
    void Release();
    bool isStopped();
    bool stopRequested();
    bool AcceptKeyFrames();
    void SetAcceptKeyFrames(bool flag);
    bool SetNotStop(bool flag);

    void InterruptBA();

    void RequestFinish();
    bool isFinished();

    int KeyframesInQueue() {
        std::unique_lock<std::mutex> lock(mMutexNewKFs);
        return mlNewKeyFrames.size();
    }

    double GetCurrKFTime();
    KeyFramePtr GetCurrKF();

    Eigen::Matrix3d mRwg;
    Eigen::Vector3d mbg;
    Eigen::Vector3d mba;
    double mScale;
    double mInitTime;
    double mCostTime;

    unsigned int mInitSect;
    unsigned int mIdxInit;
    unsigned int mnKFs;
    double mFirstTs;
    int mnMatchesInliers;

    // For debugging (erase in normal mode)
    int mInitFr;
    int mIdxIteration;
    std::string strSequence;

    bool mbNotBA1;
    bool mbNotBA2;

    bool mbWriteStats;

    // not consider far points (clouds)
    bool mbFarPoints;
    bool mbFarLines;
    float mThFarPoints;
    float mThFarLines;

#ifdef REGISTER_TIMES
    vector<double> vdKFInsert_ms;
    vector<double> vdMPCulling_ms;
    vector<double> vdMPCreation_ms;
    vector<double> vdLBA_ms;
    vector<double> vdKFCulling_ms;
    vector<double> vdLMTotal_ms;

    vector<double> vdLBASync_ms;
    vector<double> vdKFCullingSync_ms;
    vector<int> vnLBA_edges;
    vector<int> vnLBA_KFopt;
    vector<int> vnLBA_KFfixed;
    vector<int> vnLBA_MPs;
    int nLBA_exec;
    int nLBA_abort;
#endif
  protected:
    bool CheckNewKeyFrames();
    void ProcessNewKeyFrame();
    void CreateNewMapFeatures();

    void MapFeatureCulling();
    void SearchInNeighbors();
    void KeyFrameCulling();

    System *mpSystem;

    void ResetIfRequested();
    bool mbResetRequested;
    bool mbResetRequestedActiveMap;
    Map *mpMapToReset;
    std::mutex mMutexReset;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    Atlas *mpAtlas;

    LoopClosing *mpLoopCloser;
    Tracking *mpTracker;

    std::list<KeyFramePtr> mlNewKeyFrames;

    KeyFramePtr mpCurrentKeyFrame;

    std::list<MapPointPtr> mlpRecentAddedMapPoints;
    std::list<MapLinePtr> mlpRecentAddedMapLines;
    std::list<KeyFramePtr> mlpKeyFramesToSparsification;

    std::mutex mMutexNewKFs;

    bool mbAbortBA;

    bool mbStopped;
    bool mbStopRequested;
    bool mbNotStop;
    std::mutex mMutexStop;

    bool mbAcceptKeyFrames;
    std::mutex mMutexAccept;

    void ScaleRefinement();

    int mNumLM;
    int mNumKFCulling;

    float mTinit;

    int countRefinement;

    MapSparsification *mpMapSparsification = nullptr;
    int mnSparsificationNonLocalThreshold = 30;

    // DEBUG
    std::ofstream f_lm;
};

} // namespace XPLSLAM
