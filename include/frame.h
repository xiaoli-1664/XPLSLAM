#pragma once

#include <mutex>
#include <opencv2/opencv.hpp>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Vocabulary.h"
#include "sophus/se3.hpp"
#include "xplnet.h"

#include "BowVector.h"
#include "FeatureVector.h"

#include "converter.h"
#include "slam_types.h"

namespace XPLSLAM {
#define FRAME_GRID_ROWS_POINT 48
#define FRAME_GRID_COLS_POINT 64

#define FRAME_GRID_ROWS_LINE 48
#define FRAME_GRID_COLS_LINE 64

class MapPoint;
class MapLine;
class KeyFrame;
class GeometricCamera;
class XPLNet;

class Frame {
    friend struct FrameTestAccess;

    using GridType = std::vector<std::vector<std::vector<size_t>>>;
    using StereoPointIndexGrid =
        std::vector<std::vector<std::vector<std::unordered_set<size_t>>>>;

  public:
    enum class LineFeatureState { None = 0, TrackingReady = 1, StereoReady = 2 };

    Frame();

    // Copy constructor.
    Frame(const Frame &frame);

    Frame(
        const std::vector<cv::Mat> &imgs, const double &timeStamp,
        XPLNet *extractor, DBoW3::Vocabulary *voc_point,
        DBoW3::Vocabulary *voc_line, std::vector<cv::Mat> &vK,
        std::vector<cv::Mat> &vdistCoef, const std::vector<Sophus::SE3f> &vTbc,
        const std::vector<std::pair<int, int>> &vStereoPair,
        const StereoPointIndexGrid *pStereoPointIndexInGrid,
        const float &pointStereoMinDot, const float &lineStereoMinDot,
        const float &thDepth, std::vector<GeometricCamera *> vpCamera,
        Frame *pPrevF = static_cast<Frame *>(NULL));

    const FrameConstructionTiming &GetConstructionTiming() const {
        return mConstructionTiming;
    }

    // Destructor
    // ~Frame();

    void ExtractFeatures(const std::vector<cv::Mat> &imgs);

    bool EnsureLineFeatures(LineFeatureState target =
                                LineFeatureState::TrackingReady);
    bool HasLineFeatures() const {
        return mLineFeatureState != LineFeatureState::None;
    }
    bool HasStereoLineFeatures() const {
        return mLineFeatureState == LineFeatureState::StereoReady;
    }

    void cvMatToGPUTorch(const std::vector<cv::Mat> &imgs,
                         torch::Tensor &tensor);

    // Compute point and available line Bag of Words representations.
    void ComputeBoW();

    // Compute only the BoW of already extracted lines; leave point BoW lazy.
    void ComputeLineBoW();

    void ComputeMulticameraMatches();
    void ComputePointMulticameraMatches();
    void ComputeLineMulticameraMatches();

    // Set the camera pose. (Imu pose is not modified!)
    void SetPose(const Sophus::SE3<float> &Tbw);

    // Set IMU velocity
    void SetVelocity(Eigen::Vector3f Vw);

    Eigen::Vector3f GetVelocity() const;

    Sophus::SE3f GetCameraPose(int cam_id) const;

    std::vector<Sophus::SE3f> GetCameraExtrinsics() const;

    Sophus::SE3f GetCameraExtrinsic(int cam_id) const;

    // Check if a MapPoint is in the frustum of the camera
    // and fill variables of the MapPoint to be used by the tracking
    bool isPointInFrustum(MapPointPtr pMP, float viewingCosLimit, int cam_id);
    bool isLineInFrustum(MapLinePtr pML, int cam_id);

    bool IsPointInImage(const float &x, const float &y, int cam_id) const;
    bool IsLineInImage(const Eigen::Vector3f &line, int cam_id) const;

    bool ProjectPointDistort(MapPointPtr pMP, cv::Point2f &kp, float &u,
                             float &v);

    Eigen::Vector3f inRefCoordinates(Eigen::Vector3f pCw, int cam_id);

    // Compute the cell of a point feature (return false if outside the grid)
    bool PointPosInGrid(const cv::Point2f &kp, int &posX, int &posY);
    bool LinePosInGrid(const float &d, const float &theta, int &posX,
                       int &posY);

    std::vector<size_t> GetPointFeaturesInArea(const float &x, const float &y,
                                               const float &r,
                                               int cam_id) const;

    std::vector<size_t>
    GetLineFeaturesInArea(const float &d, const float &theta, const float &r_d,
                          const float &r_theta, int cam_id) const;

    // Backprojects a keypoint (if stereo/depth info available) into 3D world
    // coordinates.
    void UnprojectStereoPoint(const int &pair_id, const int &i,
                              Eigen::Vector3f &x3D);
    void UnprojectStereoLine(const int &pair_id, const int &i,
                             Eigen::Matrix<float, 6, 1> &line3D);

    bool isSet() const;

    // Computes rotation, translation and camera center matrices from the camera
    // pose.
    void UpdatePoseMatrices();

    // Returns the camera center.
    inline Eigen::Vector3f GetCenter() { return mOw; }

    // Returns inverse of rotation
    inline Eigen::Matrix3f GetRotationInverse() { return mRwb; }

    inline Sophus::SE3<float> GetPose() const { return mTbw; }

    inline Eigen::Vector3f GetTranslation() const { return mTbw.translation(); }

    inline Eigen::Matrix3f GetRotation() const { return mTbw.rotationMatrix(); }

    inline Eigen::Matrix3f GetRwb() const { return mRwb; }

    inline Eigen::Vector3f GetOw() const { return mOw; }

    inline bool HasPose() const { return mbHasPose; }

    inline bool HasVelocity() const { return mbHasVelocity; }

  private:
    // Sophus/Eigen migration
    Sophus::SE3<float> mTbw;
    Eigen::Matrix<float, 3, 3> mRwb;
    Eigen::Matrix<float, 3, 1> mOw;
    Eigen::Matrix<float, 3, 3> mRbw;
    Eigen::Matrix<float, 3, 1> mtbw;
    bool mbHasPose;

    // Rcw_ not necessary as Sophus has a method for extracting the rotation
    // matrix: Tcw_.rotationMatrix() tcw_ not necessary as Sophus has a method
    // for extracting the translation vector: Tcw_.translation() Twc_ not
    // necessary as Sophus has a method for easily computing the inverse pose:
    // Tcw_.inverse()

    std::vector<Sophus::SE3f> mvTbc;
    std::vector<Sophus::SE3f> mvTcb;
    std::vector<Eigen::Matrix3f> mvRbc;
    std::vector<Eigen::Vector3f> mvtbc;

    Eigen::Vector3f mVw;
    bool mbHasVelocity;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Vocabulary used for relocalization.
    DBoW3::Vocabulary *mpvoc_point;
    DBoW3::Vocabulary *mpvoc_line;

    int mnCam;

    // Feature extractor.
    XPLNet *mpxplnet;

    // Frame timestamp.
    double mTimeStamp;

    // Calibration matrix and OpenCV distortion parameters.
    std::vector<cv::Mat> mvK;
    std::vector<Eigen::Matrix3f> mvK_;
    static std::vector<float> vfx;
    static std::vector<float> vfy;
    static std::vector<float> vcx;
    static std::vector<float> vcy;
    static std::vector<float> vinvfx;
    static std::vector<float> vinvfy;
    std::vector<cv::Mat> mDistCoef;

    // Threshold close/far points. Close points are inserted from 1 view.
    // Far points are inserted as in the monocular case from 2 views.
    float mThDepth;

    // Number of detected point and line features.
    std::vector<int> Np;
    std::vector<int> Nl;
    int mnTotalPoints;
    int mnTotalLines;

    // Vector of point features (original for visualization) and undistorted
    // (actually used by the system). In the stereo case, mvKeysUn is
    // redundant as images must be rectified. In the RGB-D case, RGB images
    // can be distorted.
    std::vector<std::vector<cv::Point2f>> mvKeys_cams;
    std::vector<std::vector<Eigen::Vector4f>> mvLineKeys_cams;
    std::vector<std::vector<cv::Point2f>> mvKeysUn_cams;
    std::vector<std::vector<Eigen::Vector4f>> mvLineKeysUn_cams;
    std::vector<std::vector<float>> mvKeyScores_cams;
    std::vector<std::vector<float>> mvLineScores_cams;
    std::vector<std::vector<Eigen::Matrix4f>>
        mvLineEndpointCovariances_cams;
    std::vector<std::vector<Eigen::Vector2f>> mvLinePolarUn_cams;

    // Corresponding stereo coordinate and depth for each point feature.
    std::vector<std::vector<MapPointPtr>> mvpMapPoints_cams;
    std::vector<std::vector<MapLinePtr>> mvpMapLines_cams;

    DBoW3::BowVector mBowVec_point;
    DBoW3::FeatureVector mFeatVec_point;

    DBoW3::BowVector mBowVec_line;
    DBoW3::FeatureVector mFeatVec_line;

    // Each descriptor row is associated to one feature.
    std::vector<cv::Mat> mvDescriptors_point;
    cv::Mat mDescriptors_point;
    std::vector<std::pair<int, int>> mvPointDescriptorRowToCamRow;
    std::vector<cv::Mat> mvDescriptors_line;
    cv::Mat mDescriptors_line;
    std::vector<std::pair<int, int>> mvLineDescriptorRowToCamRow;

    // MapPoints associated to point features, NULL pointer if no association.
    // Flag to identify outlier associations.
    std::vector<std::vector<bool>> mvbOutlierPoint_cams;
    std::vector<std::vector<bool>> mvbOutlierLine_cams;
    // A map-line association may be useful for mapping without being selected
    // as a pose constraint. Empty means legacy behavior: use every matched
    // line. Adaptive tracking explicitly sizes and fills this mask.
    std::vector<std::vector<bool>> mvbUseLineConstraint_cams;
    int mnCloseMPs;
    int mnCloseMLs;

    // Keypoints are assigned to cells in a grid to reduce matching complexity
    // when projecting MapPoints.
    static std::vector<float> mvfGridElementWidthInvPoint;
    static std::vector<float> mvfGridElementHeightInvPoint;

    static std::vector<float> mvfGridElementWidthInvLine;
    static std::vector<float> mvfGridElementHeightInvLine;

    std::vector<GridType> mvGrid_point;
    std::vector<GridType> mvGrid_line;

    KeyFramePtr mpLastKeyFrame;

    // Pointer to previous frame
    Frame *mpPrevFrame;

    // Current and Next Frame id.
    static long unsigned int nNextId;
    long unsigned int mnId;

    // Reference Keyframe.
    KeyFramePtr mpReferenceKF;

    // Undistorted Image Bounds (computed once).
    static std::vector<float> mvnMinX;
    static std::vector<float> mvnMaxX;
    static std::vector<float> mvnMinY;
    static std::vector<float> mvnMaxY;
    static std::vector<float> mvLineDMax;

    static bool mbInitialComputations;

    std::string mNameFile;

    int mnDataset;

#ifdef REGISTER_TIMES
    double mTimeORB_Ext;
    double mTimeStereoMatch;
#endif

  private:
    void UndistortKeyPoints();

    // Computes image bounds for the undistorted image (called in the
    // constructor).
    void ComputeImageBounds(const std::vector<cv::Mat> &img);

    // Assign keypoints to the grid for speed up feature matching (called in the
    // constructor).
    void AssignFeaturesToGrid();
    void AssignPointFeaturesToGrid();
    void AssignLineFeaturesToGrid();
    void MaterializeLines(const std::vector<DetectionResult> &results);
    void ComputeMulticameraMatchesImpl(bool include_points,
                                       bool include_lines);

    bool mbIsSet;
    FrameConstructionTiming mConstructionTiming;
    LineFeatureState mLineFeatureState = LineFeatureState::None;
    uint64_t mDeferredLineToken = 0;

  public:
    std::vector<GeometricCamera *> mvpCamera;

    std::vector<cv::Mat> mvImgs;

    // overlap pair match
    std::vector<std::pair<int, int>> mvStereoPair;
    float mPointStereoMinDot;
    float mLineStereoMinDot;

    // overlap point match
    std::vector<std::vector<float>> mvDepthStereoPoint;
    std::vector<std::vector<int>> mvFirstToSecondPointMatch;
    std::vector<std::vector<Eigen::Vector3f>> mvStereo3DPoints;
    const StereoPointIndexGrid *mpStereoPointIndexInGrid;

    // overlap line match
    std::vector<std::vector<float>> mvDepthStereoLine;
    std::vector<std::vector<int>> mvFirstToSecondLineMatch;
    std::vector<std::vector<Eigen::VectorXf>> mvStereo3DLines;
};

} // namespace XPLSLAM
