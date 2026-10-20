#pragma once

#include "converter.h"
#include "frame.h"
#include "keyframe.h"
#include "map.h"
#include "slam_types.h"

#include <map>
#include <mutex>
#include <set>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core/core.hpp>

#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/serialization.hpp>

namespace XPLSLAM {

class KeyFrame;
class Map;
class Frame;

using PluckerLine = Eigen::Matrix<float, 6, 1>;

class MapLine : public std::enable_shared_from_this<MapLine> {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MapLine();

    MapLine(const PluckerLine &line_w, const KeyFramePtr &pRefKF, Map *pMap);
    MapLine(const PluckerLine &line_w, Map *pMap, Frame *pFrame,
            const int &idxF, const int &cam_id);

    void SetWorldPos(const PluckerLine &line_w);
    PluckerLine GetWorldPos();

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
    void EraseObservation(const KeyFramePtr &pKF, int cam_id);
    void EraseObservation(const KeyFramePtr &pKF);
    bool HasObservation(const KeyFramePtr &pKF, int cam_id);

    std::vector<int> GetIndexInKeyFrame(const KeyFramePtr &pKF);
    int GetIndexInKeyFrame(const KeyFramePtr &pKF, int cam_id);
    bool IsInKeyFrame(const KeyFramePtr &pKF);

    void SetBadFlag();
    bool isBad();

    void Replace(const MapLinePtr &pML);
    MapLinePtr GetReplaced();

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
                 std::set<MapLinePtr> &spML);
    void PostLoad(std::map<long unsigned int, KeyFramePtr> &mpKFid,
                  std::map<long unsigned int, MapLinePtr> &mpMLid);

  public:
    long unsigned int mnId;
    static long unsigned int nNextId;
    static int nCam;
    long int mnFirstKFid;
    long int mnFirstFrame;
    int nObs;

    // Variables used by tracking
    std::vector<float> mvTrackProjD;
    std::vector<float> mvTrackProjTheta;
    std::vector<bool> mvbTrackInView;
    std::vector<float> mvTrackDepth;
    long unsigned int mnTrackReferenceForFrame;
    std::vector<long unsigned int> mvnLastFrameSeen;

    // Variables used by local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnFuseCandidateForKF;

    // Variables used by loop closing and merging
    long unsigned int mnLoopLineForKF;
    long unsigned int mnCorrectedByKF;
    long unsigned int mnCorrectedReference;
    PluckerLine mPosGBA;
    long unsigned int mnBAGlobalForKF;
    long unsigned int mnBALocalForMerge;

    // Variables used by merging
    PluckerLine mPosMerge;

    static std::mutex mGlobalMutex;

    unsigned int mnOriginMapId;

  protected:
    // Line in world coordinates represented with Plucker coordinates
    PluckerLine mWorldPos;

    // Keyframes observing the line and associated {feature index, camera id}
    std::map<KeyFrameWeakPtr, std::vector<int>, std::owner_less<KeyFrameWeakPtr>>
        mObservations;

    // For save/load without pointers
    std::map<long unsigned int, std::vector<int>> mBackupObservationsId;

    // Best descriptor to speed up matching
    cv::Mat mDescriptor;

    // Reference KeyFrame
    KeyFrameWeakPtr mpRefKF;
    long unsigned int mBackupRefKFId;

    // Tracking counters
    int mnVisible;
    int mnFound;

    // Bad flag (we do not currently erase MapLine from memory)
    bool mbBad;
    MapLineWeakPtr mpReplaced;
    long long int mBackupReplacedId;

    // Scale invariance distances
    float mfMinDistance;
    float mfMaxDistance;

    Map *mpMap;

    // Mutex
    std::mutex mMutexPos;
    std::mutex mMutexFeatures;
    std::mutex mMutexMap;
};

} // namespace XPLSLAM
