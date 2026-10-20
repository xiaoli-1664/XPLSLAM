#pragma once

#include "converter.h"
#include "frame.h"
#include "keyframe.h"
#include "map.h"
#include "slam_types.h"

#include <opencv2/core/core.hpp>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/serialization.hpp>

namespace XPLSLAM {

class KeyFrame;
class Map;
class Frame;

class MapPoint : public std::enable_shared_from_this<MapPoint> {

    // friend class boost::serialization::access;
    // template <class Archive>
    // void serialize(Archive &ar, const unsigned int version) {
    //     ar & mnId;
    //     ar & mnFirstKFid;
    //     ar & mnFirstFrame;
    //     ar & nObs;
    //     // Variables used by the tracking
    //     // ar & mTrackProjX;
    //     // ar & mTrackProjY;
    //     // ar & mTrackDepth;
    //     // ar & mTrackDepthR;
    //     // ar & mTrackProjXR;
    //     // ar & mTrackProjYR;
    //     // ar & mbTrackInView;
    //     // ar & mbTrackInViewR;
    //     // ar & mnTrackScaleLevel;
    //     // ar & mnTrackScaleLevelR;
    //     // ar & mnTrackReferenceForFrame;
    //     // ar & mvnLastFrameSeen;
    //
    //     // Variables used by local mapping
    //     // ar & mnBALocalForKF;
    //     // ar & mnFuseCandidateForKF;
    //
    //     // Variables used by loop closing and merging
    //     // ar & mnLoopPointForKF;
    //     // ar & mnCorrectedByKF;
    //     // ar & mnCorrectedReference;
    //     // serializeMatrix(ar,mPosGBA,version);
    //     // ar & mnBAGlobalForKF;
    //     // ar & mnBALocalForMerge;
    //     // serializeMatrix(ar,mPosMerge,version);
    //
    //     // Protected variables
    //     ar &boost::serialization::make_array(mWorldPos.data(),
    //                                          mWorldPos.size());
    //     // ar & BOOST_SERIALIZATION_NVP(mBackupObservationsId);
    //     // ar & mObservations;
    //     ar & mBackupObservationsId1;
    //     ar & mBackupObservationsId2;
    //     serializeMatrix(ar, mDescriptor, version);
    //     ar & mBackupRefKFId;
    //     // ar & mnVisible;
    //     // ar & mnFound;
    //
    //     ar & mbBad;
    //     ar & mBackupReplacedId;
    //
    //     ar & mfMinDistance;
    //     ar & mfMaxDistance;
    // }

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MapPoint();

    MapPoint(const Eigen::Vector3f &Pos, const KeyFramePtr &pRefKF, Map *pMap);
    MapPoint(const Eigen::Vector3f &Pos, Map *pMap, Frame *pFrame,
             const int &idxF, const int &cam_id);

    void SetWorldPos(const Eigen::Vector3f &Pos);
    Eigen::Vector3f GetWorldPos();

    KeyFramePtr GetReferenceKeyFrame();

    std::map<KeyFramePtr, std::vector<int>>
    GetObservations();
    void GetObservations(
        std::vector<std::pair<KeyFramePtr, std::vector<int>>> &observations);
    int Observations();
    int RecentCullingStatus(long int currentKFid, int minObservations,
                            float minFoundRatio);
    bool HasEnoughBetterObservations(const KeyFramePtr &pExcludedKF, float score,
                                     int minObservations);

    void AddObservation(const KeyFramePtr &pKF, int idx, int cam_id);
    void UpdateObservation(const KeyFramePtr &pKF, int idx, int cam_id);
    void EraseObservation(const KeyFramePtr &pKF, int cam_id);
    void EraseObservation(const KeyFramePtr &pKF);
    bool HasObservation(const KeyFramePtr &pKF, int cam_id);

    std::vector<int> GetIndexInKeyFrame(const KeyFramePtr &pKF);
    int GetIndexInKeyFrame(const KeyFramePtr &pKF, int cam_id);
    bool IsInKeyFrame(const KeyFramePtr &pKF);

    void SetBadFlag();
    bool isBad();

    void Replace(const MapPointPtr &pMP);
    MapPointPtr GetReplaced();

    void IncreaseVisible(int n = 1);
    void IncreaseFound(int n = 1);
    float GetFoundRatio();
    inline int GetFound() { return mnFound; }

    void ComputeDistinctiveDescriptors();

    cv::Mat GetDescriptor();
    bool GetDescriptorForMatching(cv::Mat &descriptor);

    float GetMinDistanceInvariance();
    float GetMaxDistanceInvariance();
    int PredictScale(const float &currentDist, const KeyFramePtr &pKF);
    int PredictScale(const float &currentDist, Frame *pF);

    Map *GetMap();
    void UpdateMap(Map *pMap);

    void PrintObservations();

    void PreSave(std::set<KeyFramePtr> &spKF,
                 std::set<MapPointPtr> &spMP);
    void PostLoad(std::map<long unsigned int, KeyFramePtr> &mpKFid,
                  std::map<long unsigned int, MapPointPtr> &mpMPid);
    void RemoveObservationReference(const KeyFramePtr &pKF, int cam_id);

  public:
    long unsigned int mnId;
    static long unsigned int nNextId;
    static int nCam;
    long int mnFirstKFid;
    long int mnFirstFrame;
    int nObs;

    // Variables used by the tracking
    std::vector<float> mvTrackProjX;
    std::vector<float> mvTrackProjY;
    std::vector<float> mvTrackDepth;
    std::vector<bool> mvbTrackInView;
    long unsigned int mnTrackReferenceForFrame;
    std::vector<long unsigned int> mvnLastFrameSeen;

    // Variables used by local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnFuseCandidateForKF;

    // Variables used by loop closing
    long unsigned int mnLoopPointForKF;
    long unsigned int mnCorrectedByKF;
    long unsigned int mnCorrectedReference;
    Eigen::Vector3f mPosGBA;
    long unsigned int mnBAGlobalForKF;
    long unsigned int mnBALocalForMerge;

    // Variable used by merging
    Eigen::Vector3f mPosMerge;

    static std::mutex mGlobalMutex;

    unsigned int mnOriginMapId;

  protected:
    // Position in absolute coordinates
    Eigen::Vector3f mWorldPos;

    // Keyframes observing the point and associated index in keyframe
    std::map<KeyFrameWeakPtr, std::vector<int>, std::owner_less<KeyFrameWeakPtr>>
        mObservations;
    // For save relation without pointer, this is necessary for save/load
    // function
    std::map<long unsigned int, std::vector<int>> mBackupObservationsId;

    // Best descriptor to fast matching
    cv::Mat mDescriptor;

    // Reference KeyFrame
    KeyFrameWeakPtr mpRefKF;
    long unsigned int mBackupRefKFId;

    // Tracking counters
    int mnVisible;
    int mnFound;

    // Bad flag (we do not currently erase MapPoint from memory)
    bool mbBad;
    MapPointWeakPtr mpReplaced;
    // For save relation without pointer, this is necessary for save/load
    // function
    long long int mBackupReplacedId;

    Map *mpMap;

    // Mutex
    std::mutex mMutexPos;
    std::mutex mMutexFeatures;
    std::mutex mMutexMap;
};

} // namespace XPLSLAM
