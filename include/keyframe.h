#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "BowVector.h"
#include "Eigen/Core"
#include "FeatureVector.h"
#include "Vocabulary.h"
#include "sophus/geometry.hpp"
#include "sophus/se3.hpp"

#include "converter.h"
#include "slam_types.h"

namespace XPLSLAM {

class Map;
class MapPoint;
class MapLine;
class Frame;
class KeyFrameDatabase;
class GeometricCamera;

class KeyFrame : public std::enable_shared_from_this<KeyFrame> {

    using GridType = std::vector<std::vector<std::vector<size_t>>>;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    KeyFrame();
    KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB);

    // Pose functions
    void SetPose(const Sophus::SE3f &Tbw);
    void SetVelocity(const Eigen::Vector3f &Vw_);

    Sophus::SE3f GetPose() const;
    Sophus::SE3f GetCameraPose(int cam_id) const;
    std::vector<Sophus::SE3f> GetCameraPose() const;

    Sophus::SE3f GetPoseInverse() const;
    Sophus::SE3f GetCameraPoseInverse(int cam_id) const;
    std::vector<Sophus::SE3f> GetCameraPoseInverse() const;
    Eigen::Vector3f GetCenter();
    Eigen::Vector3f GetCameraCenter(int cam_id) const;
    std::vector<Eigen::Vector3f> GetCameraCenter() const;
    std::vector<Sophus::SE3f> GetCameraExtrinsics() const;

    Eigen::Matrix3f GetRotation() const;
    Eigen::Vector3f GetTranslation() const;
    Eigen::Vector3f GetVelocity();
    bool isVelocitySet();

    void PostLoad(std::map<long unsigned int, KeyFramePtr> &mpKFid,
                  std::map<long unsigned int, MapPointPtr> &mpMPid,
                  std::map<long unsigned int, MapLinePtr> &mpMLid,
                  std::map<unsigned int, GeometricCamera *> &mpCamId);

    // Bag of Words Representation
    void ComputeBoW();

    // Covisibility graph functions
    void AddConnection(const KeyFramePtr &pKF, const int &weight);
    void EraseConnection(const KeyFramePtr &pKF);

    void UpdateConnections(bool upParent = true);
    void UpdateBestCovisibles();
    std::set<KeyFramePtr>
    GetConnectedKeyFrames();
    std::vector<KeyFramePtr> GetVectorCovisibleKeyFrames();
    std::vector<KeyFramePtr>
    GetBestCovisibilityKeyFrames(const int &N) const;
    std::vector<KeyFramePtr> GetCovisiblesByWeight(const int &w);
    int GetWeight(const KeyFramePtr &pKF);

    // Spanning tree functions
    void AddChild(const KeyFramePtr &pKF);
    void EraseChild(const KeyFramePtr &pKF);
    void ChangeParent(const KeyFramePtr &pKF);
    std::set<KeyFramePtr> GetChilds() const;
    KeyFramePtr GetParent() const;
    bool hasChild(const KeyFramePtr &pKF);
    void SetFirstConnection(bool bFirst);

    // Loop Edges
    void AddLoopEdge(const KeyFramePtr &pKF);
    std::set<KeyFramePtr> GetLoopEdges();

    // Merge Edges
    void AddMergeEdge(const KeyFramePtr &pKF);
    std::set<KeyFramePtr> GetMergeEdges();

    // MapPoint observation functions
    enum class PointObservationDescriptorStatus {
        Found,
        Missing,
        InvalidDescriptor
    };

    struct PointFeatureSnapshot {
        int mnCam = 0;
        int mnGridColsPoint = 0;
        int mnGridRowsPoint = 0;
        uint64_t mnFeatureRevision = 0;
        bool mbSparsified = false;
        std::vector<int> Np;
        int mnTotalPoints = 0;
        std::vector<float> mvfGridElementWidthInvPoint;
        std::vector<float> mvfGridElementHeightInvPoint;
        std::vector<int> mvnMinX;
        std::vector<int> mvnMinY;
        std::vector<int> mvnMaxX;
        std::vector<int> mvnMaxY;
        std::vector<std::vector<cv::Point2f>> mvKeysUn_cams;
        std::vector<std::vector<float>> mvKeyScores_cams;
        std::vector<cv::Mat> mvDescriptors_point;
        cv::Mat mDescriptors_point;
        std::vector<std::pair<int, int>> mvPointDescriptorRowToCamRow;
        DBoW3::FeatureVector mFeatVec_point;
        std::vector<std::vector<MapPointPtr>> mvpMapPoints_cams;
        std::vector<std::vector<std::vector<std::vector<size_t>>>> mvGrid_point;

        bool IsPointInImage(float x, float y, int cam_id) const;
        std::vector<size_t> GetPointFeaturesInArea(float x, float y, float r,
                                                   int cam_id) const;
    };

    struct PointFeatureCompaction {
        bool valid = false;
        uint64_t source_feature_revision = 0;
        std::vector<std::vector<cv::Point2f>> keys;
        std::vector<std::vector<float>> scores;
        std::vector<std::vector<MapPointPtr>> points;
        std::vector<GridType> grids;
        std::vector<int> descriptor_rows_per_camera;
        std::vector<std::pair<int, int>> row_mapping;
        cv::Mat descriptors;
        DBoW3::BowVector bow;
        DBoW3::FeatureVector feature_vector;
        std::vector<std::tuple<MapPointPtr, int, int>> observation_updates;
    };

    int GetNumberMPs();
    int GetNumberMLs();
    std::vector<GridType> GetPointFeatureGrids() const;
    PointFeatureSnapshot
    GetPointFeatureSnapshot(bool include_grid = true, bool include_bow = true,
                            bool include_associations = true) const;
    bool IsPointFeatureSnapshotCurrent(
        const PointFeatureSnapshot &snapshot) const;
    bool GetPointDescriptor(int cam, int idx, cv::Mat &descriptor) const;
    PointObservationDescriptorStatus GetPointObservationDescriptor(
        const MapPointPtr &pMP, int cam, int idx, int &resolved_idx,
        cv::Mat &descriptor) const;
    PointFeatureCompaction PreparePointFeatureCompaction() const;
    bool CommitPointFeatureCompaction(PointFeatureCompaction &&compaction);
    void CompactPointFeatures();
    bool UpdateCountInLocalMapping(bool is_local, int non_local_threshold);
    bool isSparsified() const;

    void AddMapPoint(const MapPointPtr &pMP, const size_t &idx, int cam);
    void AddMapLine(const MapLinePtr &pML, const size_t &idx, int cam);
    void EraseMapPointMatch(const int &idx, int cam);
    void EraseMapPointMatch(const MapPointPtr &pMP, int cam);
    void EraseMapLineMatch(const int &idx, int cam);
    void EraseMapLineMatch(const MapLinePtr &pML, int cam);
    void ReplaceMapPointMatch(const int &idx, const MapPointPtr &pMP, int cam);
    void ReplaceMapLineMatch(const int &idx, const MapLinePtr &pML, int cam);

    std::set<MapPointPtr> GetMapPoints();
    std::set<MapLinePtr> GetMapLines();
    std::vector<std::vector<MapPointPtr>> GetMapPointMatches() const;
    std::vector<std::vector<MapLinePtr>> GetMapLineMatches() const;
    void AppendMapPoints(std::vector<MapPointPtr> &vpMapPoints,
                         std::unordered_set<MapPointPtr> &spMapPoints) const;
    void AppendMapLines(std::vector<MapLinePtr> &vpMapLines,
                        std::unordered_set<MapLinePtr> &spMapLines) const;
    void AppendMapPointAssociations(
        std::vector<std::tuple<MapPointPtr, size_t, int>> &vAssociations) const;
    void AppendMapLineAssociations(
        std::vector<std::tuple<MapLinePtr, size_t, int>> &vAssociations) const;

    int TrackedMapPoints(const int &minObs) const;
    int TrackedMapLines(const int &minObs) const;
    MapPointPtr GetMapPoint(const size_t &idx, int cam_id);
    MapLinePtr GetMapLine(const size_t &idx, int cam_id);

    // Point feature functions
    std::vector<size_t> GetPointFeaturesInArea(const float &x, const float &y,
                                               const float &r,
                                               const int cam_id) const;
    std::vector<size_t>
    GetLineFeaturesInArea(const float &d, const float &theta, const float &r_d,
                          const float &r_theta, const int cam_id) const;

    // Image
    bool IsPointInImage(const float &x, const float &y, int cam_id) const;
    bool IsLineInImage(const Eigen::Vector3f &line, int cam_id) const;

    // Enable/Disable bad flag changes
    void SetNotErase();
    void SetErase();

    // Set/check bad flag
    void SetBadFlag();
    bool isBad() const;

    static bool weightComp(int a, int b) { return a > b; }

    static bool lId(const KeyFramePtr &pKF1, const KeyFramePtr &pKF2) {
        return pKF1->mnId < pKF2->mnId;
    }

    Map *GetMap() const;
    void UpdateMap(Map *pMap);

    bool ProjectPointDistort(const MapPointPtr &pMP, int cam_id,
                             cv::Point2f &kp,
                             float &u, float &v);
    bool ProjectLineDistort(const MapLinePtr &pML, int cam_id, float &d,
                            float &theta);
    bool ProjectPointUnDistort(const MapPointPtr &pMP, int cam_id,
                               cv::Point2f &kp,
                               float &u, float &v);
    bool ProjectLineUnDistort(const MapLinePtr &pML, int cam_id, float &d,
                              float &theta);

    void SetPointVocabulary(DBoW3::Vocabulary *pVoc);
    void SetLineVocabulary(DBoW3::Vocabulary *pVoc);
    void SetKeyFrameDatabase(KeyFrameDatabase *pKFDB);

  public:
    static long unsigned int nNextId;
    int mnCam;
    long unsigned int mnId;
    const long unsigned int mnFrameId;

    const double mTimeStamp;

    // Grid (to speed up feature matching)
    const int mnGridColsPoint;
    const int mnGridRowsPoint;
    const int mnGridColsLine;
    const int mnGridRowsLine;
    std::vector<float> mvfGridElementWidthInvPoint;
    std::vector<float> mvfGridElementHeightInvPoint;
    const std::vector<float> mvfGridElementWidthInvLine;
    const std::vector<float> mvfGridElementHeightInvLine;

    // Variables used by the tracking
    long unsigned int mnTrackReferenceForFrame;
    long unsigned int mnFuseTargetForKF;

    // Variables used by the local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnBAFixedForKF;

    // Number of optimizations by BA(amount of iterations in BA)
    long unsigned int mnNumberOfOpt;

    // Variables used by the keyframe database
    long unsigned int mnLoopQuery;
    int mnLoopWords;
    float mLoopScore;
    long unsigned int mnRelocQuery;
    int mnRelocWords;
    float mRelocScore;
    long unsigned int mnMergeQuery;
    int mnMergeWords;
    float mMergeScore;
    long unsigned int mnPlaceRecognitionQuery;
    int mnPlaceRecognitionWords;
    float mPlaceRecognitionScore;

    bool mbCurrentPlaceRecognition;

    // Variables used by loop closing
    Sophus::SE3f mTbwGBA;
    Sophus::SE3f mTbwBefGBA;
    long unsigned int mnBAGlobalForKF;

    // Variables used by merging
    Sophus::SE3f mTbwMerge;
    Sophus::SE3f mTbwBefMerge;
    Sophus::SE3f mTwbBefMerge;
    long unsigned int mnMergeCorrectedForKF;
    long unsigned int mnMergeForKF;
    float mfScaleMerge;
    long unsigned int mnBALocalForMerge;

    float mfScale;

    // Calibration parameters
    const std::vector<float> vfx, vfy, vcx, vcy, vinvfx, vinvfy;
    const float mThDepth;

    // Number of detected point and line features
    std::vector<int> Np;
    const std::vector<int> Nl;
    int mnTotalPoints;
    const int mnTotalLines;

    // Point features, stereo coordinate and descriptors (all associated by an
    // index)
    std::vector<std::vector<cv::Point2f>> mvKeysUn_cams;
    const std::vector<std::vector<Eigen::Vector4f>> mvLineKeysUn_cams;
    std::vector<std::vector<float>> mvKeyScores_cams;
    const std::vector<std::vector<float>> mvLineScores_cams;
    const std::vector<std::vector<Eigen::Matrix4f>>
        mvLineEndpointCovariances_cams;
    const std::vector<std::vector<Eigen::Vector2f>> mvLinePolarUn_cams;
    std::vector<cv::Mat> mvDescriptors_point;
    cv::Mat mDescriptors_point;
    std::vector<std::pair<int, int>> mvPointDescriptorRowToCamRow;
    cv::Mat mDescriptors_line;
    std::vector<cv::Mat> mvDescriptors_line;
    std::vector<std::pair<int, int>> mvLineDescriptorRowToCamRow;

    // BoW
    DBoW3::BowVector mBowVec_point;
    DBoW3::FeatureVector mFeatVec_point;

    DBoW3::BowVector mBowVec_line;
    DBoW3::FeatureVector mFeatVec_line;

    // Pose relative to parent (this is computed when bad flag is activated)
    Sophus::SE3f mTbp;

    // Image bounds and calibration
    std::vector<int> mvnMinX;
    std::vector<int> mvnMinY;
    std::vector<int> mvnMaxX;
    std::vector<int> mvnMaxY;
    const std::vector<float> mvLineDMax;

    // Preintegrated IMU measurements from previous keyframe
    KeyFrameWeakPtr mPrevKF;
    KeyFrameWeakPtr mNextKF;

    unsigned int mnOriginMapId;

    std::string mNameFile;

    int mnDataset;

    std::vector<KeyFrameWeakPtr> mvpLoopCandKFs;
    std::vector<KeyFrameWeakPtr> mvpMergeCandKFs;

    long unsigned int mnMapSparsificationId = 0;
    bool mbSparsified = false;

    // bool mbHasHessian;
    // cv::Mat mHessianPose;

    // The following variables need to be accessed trough a mutex to be thread
    // safe.
  protected:
    // sophus poses
    Sophus::SE3f mTbw;
    Eigen::Matrix3f mRbw;
    Sophus::SE3f mTwb;
    Eigen::Matrix3f mRwb;

    Eigen::Vector3f mOwb;
    Eigen::Vector3f mVw;
    bool mbHasVelocity;

    std::vector<Sophus::SE3f> mvTbc;

    // MapPoints associated to keypoints
    std::vector<std::vector<MapPointPtr>> mvpMapPoints_cams;
    std::vector<std::vector<MapLinePtr>> mvpMapLines_cams;
    // For save relation without pointer, this is necessary for save/load
    // function
    std::vector<long long int> mvBackupMapPointsId;
    std::vector<long long int> mvBackupMapLinesId;

    // BoW
    KeyFrameDatabase *mpKeyFrameDB;
    DBoW3::Vocabulary *mpVocabulary_point;
    DBoW3::Vocabulary *mpVocabulary_line;

    // Grid over the image to speed up feature matching
    std::vector<GridType> mvGrid_point;
    std::vector<GridType> mvGrid_line;

    std::map<KeyFrameWeakPtr, int, std::owner_less<KeyFrameWeakPtr>>
        mConnectedKeyFrameWeights;
    std::vector<KeyFrameWeakPtr> mvpOrderedConnectedKeyFrames;
    std::vector<int> mvOrderedWeights;
    // For save relation without pointer, this is necessary for save/load
    // function
    std::map<long unsigned int, int> mBackupConnectedKeyFrameIdWeights;

    // Spanning Tree and Loop Edges
    bool mbFirstConnection;
    KeyFramePtr mpParent;
    std::set<KeyFrameWeakPtr, std::owner_less<KeyFrameWeakPtr>> mspChildrens;
    std::set<KeyFrameWeakPtr, std::owner_less<KeyFrameWeakPtr>> mspLoopEdges;
    std::set<KeyFrameWeakPtr, std::owner_less<KeyFrameWeakPtr>> mspMergeEdges;

    // Bad flags
    bool mbNotErase;
    bool mbToBeErased;
    bool mbBad;
    int mnCountInLocal = 0;

    Map *mpMap;

    // Mutex
    mutable std::mutex mMutexPose; // for pose, velocity and biases
    mutable std::mutex mMutexConnections;
    mutable std::mutex mMutexFeatures;
    // Monotonic revision of feature layout and map-feature associations.
    // Optimistic readers use it to detect any concurrent change before
    // publishing results computed from a PointFeatureSnapshot.
    uint64_t mnFeatureRevision = 0;
    mutable std::mutex mMutexMap;

  public:
    std::vector<GeometricCamera *> mvpCameras;
};

} // namespace XPLSLAM
