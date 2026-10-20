#include "keyframe.h"
#include "converter.h"
#include "frame.h"
#include "geometric_tools.h"
#include "map.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <tuple>

using namespace std;

namespace XPLSLAM {

namespace {

constexpr int kPointDescriptorCols = 64;

int ComputeFeatureVectorLevelsup(const DBoW3::Vocabulary *vocabulary) {
    return std::max(vocabulary ? vocabulary->getDepthLevels() - 2 : 0, 0);
}

bool IsValidPointDescriptorRow(const cv::Mat &descriptors, int row) {
    return !descriptors.empty() && row >= 0 && row < descriptors.rows &&
           descriptors.cols == kPointDescriptorCols &&
           descriptors.type() == CV_32F;
}

void BuildCameraDescriptorViews(const cv::Mat &descriptors,
                                const vector<cv::Mat> &camera_descriptors,
                                vector<cv::Mat> &camera_descriptor_views) {
    camera_descriptor_views.assign(camera_descriptors.size(), cv::Mat());

    int total_rows = 0;
    for (const cv::Mat &camera_descriptor : camera_descriptors) {
        total_rows += std::max(camera_descriptor.rows, 0);
    }

    if (total_rows != descriptors.rows) {
        for (size_t cam = 0; cam < camera_descriptors.size(); ++cam) {
            camera_descriptor_views[cam] = camera_descriptors[cam].clone();
        }
        return;
    }

    int first_row = 0;
    for (size_t cam = 0; cam < camera_descriptors.size(); ++cam) {
        const int rows = std::max(camera_descriptors[cam].rows, 0);
        if (rows > 0) {
            camera_descriptor_views[cam] =
                descriptors.rowRange(first_row, first_row + rows);
            first_row += rows;
        }
    }
}

void BuildCameraDescriptorViews(const cv::Mat &descriptors,
                                const vector<int> &rows_per_camera,
                                vector<cv::Mat> &camera_descriptor_views) {
    camera_descriptor_views.assign(rows_per_camera.size(), cv::Mat());
    const int total_rows =
        accumulate(rows_per_camera.begin(), rows_per_camera.end(), 0);
    if (total_rows != descriptors.rows) {
        throw runtime_error(
            "Cannot build KeyFrame point descriptor views: row count mismatch");
    }

    int first_row = 0;
    for (size_t cam = 0; cam < rows_per_camera.size(); ++cam) {
        const int rows = rows_per_camera[cam];
        if (rows < 0) {
            throw runtime_error(
                "Cannot build KeyFrame point descriptor views: negative row "
                "count");
        }
        if (rows > 0) {
            camera_descriptor_views[cam] =
                descriptors.rowRange(first_row, first_row + rows);
            first_row += rows;
        }
    }
}

} // namespace

long unsigned int KeyFrame::nNextId = 0;

KeyFrame::KeyFrame()
    : mnFrameId(0), mTimeStamp(0), mnGridColsPoint(FRAME_GRID_COLS_POINT),
      mnGridRowsPoint(FRAME_GRID_ROWS_POINT),
      mnGridColsLine(FRAME_GRID_COLS_LINE),
      mnGridRowsLine(FRAME_GRID_ROWS_LINE), mvfGridElementHeightInvPoint(0),
      mvfGridElementWidthInvPoint(0), mvfGridElementHeightInvLine(0),
      mvfGridElementWidthInvLine(0), mnTrackReferenceForFrame(0),
      mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0),
      mnBALocalForMerge(0), mnLoopQuery(0), mnLoopWords(0), mLoopScore(0),
      mnRelocQuery(0), mnRelocWords(0), mRelocScore(0), mnMergeQuery(0),
      mnMergeWords(0), mMergeScore(0), mnPlaceRecognitionQuery(0),
      mnPlaceRecognitionWords(0), mPlaceRecognitionScore(0),
      mbCurrentPlaceRecognition(false), mnBAGlobalForKF(0),
      mnMergeCorrectedForKF(0), mnMergeForKF(0), mThDepth(0), mnTotalPoints(0),
      mnTotalLines(0) {}

KeyFrame::KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB)
    : mnFrameId(F.mnId), mTimeStamp(F.mTimeStamp),
      mnGridColsPoint(FRAME_GRID_COLS_POINT),
      mnGridRowsPoint(FRAME_GRID_ROWS_POINT),
      mnGridColsLine(FRAME_GRID_COLS_LINE),
      mnGridRowsLine(FRAME_GRID_ROWS_LINE),
      mvfGridElementHeightInvPoint(F.mvfGridElementHeightInvPoint),
      mvfGridElementWidthInvPoint(F.mvfGridElementWidthInvPoint),
      mvfGridElementHeightInvLine(F.mvfGridElementHeightInvLine),
      mvfGridElementWidthInvLine(F.mvfGridElementWidthInvLine),
      mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0),
      mnBAFixedForKF(0), mnBALocalForMerge(0), mnLoopQuery(0), mnLoopWords(0),
      mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0),
      mnPlaceRecognitionQuery(0), mnPlaceRecognitionWords(0),
      mDescriptors_point(F.mDescriptors_point.clone()),
      mvPointDescriptorRowToCamRow(F.mvPointDescriptorRowToCamRow),
      mDescriptors_line(F.mDescriptors_line.clone()),
      mvLineDescriptorRowToCamRow(F.mvLineDescriptorRowToCamRow),
      mPlaceRecognitionScore(0), vfx(F.vfx), vfy(F.vfy), vcx(F.vcx), vcy(F.vcy),
      vinvfx(F.vinvfx), vinvfy(F.vinvfy), mThDepth(F.mThDepth), Np(F.Np),
      Nl(F.Nl), mnTotalPoints(F.mnTotalPoints), mnTotalLines(F.mnTotalLines),
      mvKeysUn_cams(F.mvKeysUn_cams), mvLineKeysUn_cams(F.mvLineKeysUn_cams),
      mvKeyScores_cams(F.mvKeyScores_cams),
      mvLineScores_cams(F.mvLineScores_cams),
      mvLineEndpointCovariances_cams(F.mvLineEndpointCovariances_cams),
      mvLinePolarUn_cams(F.mvLinePolarUn_cams), mBowVec_point(F.mBowVec_point),
      mFeatVec_point(F.mFeatVec_point), mBowVec_line(F.mBowVec_line),
      mFeatVec_line(F.mFeatVec_line), mvLineDMax(F.mvLineDMax), mPrevKF(),
      mNextKF(), mvpMapPoints_cams(F.mvpMapPoints_cams),
      mvpMapLines_cams(F.mvpMapLines_cams), mpKeyFrameDB(pKFDB),
      mpVocabulary_point(F.mpvoc_point), mpVocabulary_line(F.mpvoc_line),
      mbFirstConnection(true), mpParent(nullptr), mbNotErase(false),
      mnDataset(F.mnDataset), mbToBeErased(false), mbBad(false), mpMap(pMap),
      mbCurrentPlaceRecognition(false), mNameFile(F.mNameFile),
      mnMergeCorrectedForKF(0), mvpCameras(F.mvpCamera),
      mvTbc(F.GetCameraExtrinsics()), mnNumberOfOpt(0), mbHasVelocity(false) {
    mnId = nNextId++;

    mnCam = F.mnCam;

    BuildCameraDescriptorViews(mDescriptors_point, F.mvDescriptors_point,
                               mvDescriptors_point);
    BuildCameraDescriptorViews(mDescriptors_line, F.mvDescriptors_line,
                               mvDescriptors_line);

    mvnMaxY.resize(mnCam);
    mvnMinY.resize(mnCam);
    mvnMaxX.resize(mnCam);
    mvnMinX.resize(mnCam);

    mvGrid_point = F.mvGrid_point;
    mvGrid_line = F.mvGrid_line;

    for (int c = 0; c < mnCam; ++c) {
        mvnMaxY[c] = F.mvnMaxY[c];
        mvnMinY[c] = F.mvnMinY[c];
        mvnMaxX[c] = F.mvnMaxX[c];
        mvnMinX[c] = F.mvnMinX[c];
    }

    if (!F.HasVelocity()) {
        mVw.setZero();
        mbHasVelocity = false;
    } else {
        mVw = F.GetVelocity();
        mbHasVelocity = true;
    }

    SetPose(F.GetPose());

    mnOriginMapId = pMap->GetId();
}

bool KeyFrame::isBad() const {
    unique_lock<mutex> lock(mMutexConnections);
    return mbBad;
}

void KeyFrame::UpdateMap(Map *pMap) {
    unique_lock<mutex> lock(mMutexMap);
    mpMap = pMap;
}

void KeyFrame::SetPointVocabulary(DBoW3::Vocabulary *pVoc) {
    mpVocabulary_point = pVoc;
}

void KeyFrame::SetLineVocabulary(DBoW3::Vocabulary *pVoc) {
    mpVocabulary_line = pVoc;
}

void KeyFrame::SetKeyFrameDatabase(KeyFrameDatabase *pKFDB) {
    mpKeyFrameDB = pKFDB;
}

void KeyFrame::PostLoad(map<long unsigned int, KeyFramePtr> &mpKFid,
                        map<long unsigned int, MapPointPtr> &mpMPid,
                        map<long unsigned int, MapLinePtr> &mpMLid,
                        map<unsigned int, GeometricCamera *> &mpCamId) {
    // Rebuild the empty variables

    // Pose
    SetPose(mTbw);

    // Reference reconstruction
    // Each MapPoint sight from this KeyFrame
    mvpMapPoints_cams.clear();
    mvpMapLines_cams.clear();

    // TODO: complete the postload function
}

void KeyFrame::SetPose(const Sophus::SE3f &Tbw) {
    unique_lock<mutex> lock(mMutexPose);

    mTbw = Tbw;
    mRbw = mTbw.rotationMatrix();
    mTwb = mTbw.inverse();
    mRwb = mTwb.rotationMatrix();

    mOwb = mTwb.translation();
}

Map *KeyFrame::GetMap() const {
    unique_lock<mutex> lock(mMutexMap);
    return mpMap;
}

void KeyFrame::AddMapPoint(const MapPointPtr &pMP, const size_t &idx, int cam) {
    unique_lock<mutex> lock(mMutexFeatures);
    if (mvpMapPoints_cams[cam][idx] != pMP) {
        mvpMapPoints_cams[cam][idx] = pMP;
        ++mnFeatureRevision;
    }
}

void KeyFrame::AddMapLine(const MapLinePtr &pML, const size_t &idx, int cam) {
    unique_lock<mutex> lock(mMutexFeatures);
    if (mvpMapLines_cams[cam][idx] != pML) {
        mvpMapLines_cams[cam][idx] = pML;
        ++mnFeatureRevision;
    }
}

vector<vector<MapPointPtr>> KeyFrame::GetMapPointMatches() const {
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints_cams;
}

vector<vector<MapLinePtr>> KeyFrame::GetMapLineMatches() const {
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapLines_cams;
}

int KeyFrame::GetNumberMPs() {
    vector<MapPointPtr> candidates;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        for (const auto &camera_points : mvpMapPoints_cams) {
            candidates.insert(candidates.end(), camera_points.begin(),
                              camera_points.end());
        }
    }

    unordered_set<MapPointPtr> unique_points;
    for (const MapPointPtr &point : candidates) {
        if (point && !point->isBad())
            unique_points.insert(point);
    }
    return static_cast<int>(unique_points.size());
}

int KeyFrame::GetNumberMLs() {
    vector<MapLinePtr> candidates;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        for (const auto &camera_lines : mvpMapLines_cams) {
            candidates.insert(candidates.end(), camera_lines.begin(),
                              camera_lines.end());
        }
    }

    unordered_set<MapLinePtr> unique_lines;
    for (const MapLinePtr &line : candidates) {
        if (line && !line->isBad())
            unique_lines.insert(line);
    }
    return static_cast<int>(unique_lines.size());
}

vector<KeyFrame::GridType> KeyFrame::GetPointFeatureGrids() const {
    unique_lock<mutex> lock(mMutexFeatures);
    return mvGrid_point;
}

bool KeyFrame::PointFeatureSnapshot::IsPointInImage(float x, float y,
                                                    int cam_id) const {
    return cam_id >= 0 && cam_id < static_cast<int>(mvnMinX.size()) &&
           cam_id < static_cast<int>(mvnMaxX.size()) &&
           cam_id < static_cast<int>(mvnMinY.size()) &&
           cam_id < static_cast<int>(mvnMaxY.size()) && x >= mvnMinX[cam_id] &&
           x < mvnMaxX[cam_id] && y >= mvnMinY[cam_id] && y < mvnMaxY[cam_id];
}

vector<size_t> KeyFrame::PointFeatureSnapshot::GetPointFeaturesInArea(
    float x, float y, float r, int cam_id) const {
    vector<size_t> indices;
    if (cam_id < 0 || cam_id >= static_cast<int>(Np.size()) ||
        cam_id >= static_cast<int>(mvGrid_point.size()) ||
        cam_id >= static_cast<int>(mvKeysUn_cams.size()) ||
        cam_id >= static_cast<int>(mvfGridElementWidthInvPoint.size()) ||
        cam_id >= static_cast<int>(mvfGridElementHeightInvPoint.size()) ||
        cam_id >= static_cast<int>(mvnMinX.size()) ||
        cam_id >= static_cast<int>(mvnMinY.size()) || mnGridColsPoint <= 0 ||
        mnGridRowsPoint <= 0) {
        return indices;
    }

    indices.reserve(Np[cam_id]);

    const int nMinCellX =
        max(0, (int)floor((x - mvnMinX[cam_id] - r) *
                          mvfGridElementWidthInvPoint[cam_id]));
    if (nMinCellX >= mnGridColsPoint)
        return indices;

    const int nMaxCellX = min(mnGridColsPoint - 1,
                              (int)ceil((x - mvnMinX[cam_id] + r) *
                                        mvfGridElementWidthInvPoint[cam_id]));
    if (nMaxCellX < 0)
        return indices;

    const int nMinCellY =
        max(0, (int)floor((y - mvnMinY[cam_id] - r) *
                          mvfGridElementHeightInvPoint[cam_id]));
    if (nMinCellY >= mnGridRowsPoint)
        return indices;

    const int nMaxCellY = min(mnGridRowsPoint - 1,
                              (int)ceil((y - mvnMinY[cam_id] + r) *
                                        mvfGridElementHeightInvPoint[cam_id]));
    if (nMaxCellY < 0)
        return indices;

    const auto &camera_grid = mvGrid_point[cam_id];
    const auto &camera_keys = mvKeysUn_cams[cam_id];
    for (int ix = nMinCellX; ix <= nMaxCellX; ++ix) {
        for (int iy = nMinCellY; iy <= nMaxCellY; ++iy) {
            if (iy < 0 || iy >= static_cast<int>(camera_grid.size()) ||
                ix < 0 || ix >= static_cast<int>(camera_grid[iy].size())) {
                throw runtime_error(
                    "Inconsistent KeyFrame point feature snapshot grid cell");
            }
            const vector<size_t> &cell_indices = camera_grid[iy][ix];
            for (size_t idx : cell_indices) {
                if (idx >= camera_keys.size()) {
                    throw runtime_error(
                        "Inconsistent KeyFrame point feature snapshot grid "
                        "index");
                }
                const cv::Point2f &kp = camera_keys[idx];
                const float dist_x = kp.x - x;
                const float dist_y = kp.y - y;
                if (fabs(dist_x) < r && fabs(dist_y) < r)
                    indices.push_back(idx);
            }
        }
    }

    return indices;
}

KeyFrame::PointFeatureSnapshot
KeyFrame::GetPointFeatureSnapshot(bool include_grid, bool include_bow,
                                  bool include_associations) const {
    unique_lock<mutex> lock(mMutexFeatures);
    PointFeatureSnapshot snapshot;
    snapshot.mnCam = mnCam;
    snapshot.mnGridColsPoint = mnGridColsPoint;
    snapshot.mnGridRowsPoint = mnGridRowsPoint;
    snapshot.mnFeatureRevision = mnFeatureRevision;
    snapshot.mbSparsified = mbSparsified;
    snapshot.Np = Np;
    snapshot.mnTotalPoints = mnTotalPoints;
    snapshot.mvfGridElementWidthInvPoint = mvfGridElementWidthInvPoint;
    snapshot.mvfGridElementHeightInvPoint = mvfGridElementHeightInvPoint;
    snapshot.mvnMinX = mvnMinX;
    snapshot.mvnMinY = mvnMinY;
    snapshot.mvnMaxX = mvnMaxX;
    snapshot.mvnMaxY = mvnMaxY;
    snapshot.mvKeysUn_cams = mvKeysUn_cams;
    snapshot.mvKeyScores_cams = mvKeyScores_cams;
    snapshot.mvDescriptors_point = mvDescriptors_point;
    if (include_bow) {
        snapshot.mDescriptors_point = mDescriptors_point;
        snapshot.mvPointDescriptorRowToCamRow =
            mvPointDescriptorRowToCamRow;
        snapshot.mFeatVec_point = mFeatVec_point;
    }
    if (include_associations)
        snapshot.mvpMapPoints_cams = mvpMapPoints_cams;
    if (include_grid)
        snapshot.mvGrid_point = mvGrid_point;
    return snapshot;
}

bool KeyFrame::IsPointFeatureSnapshotCurrent(
    const PointFeatureSnapshot &snapshot) const {
    unique_lock<mutex> lock(mMutexFeatures);
    return snapshot.mnFeatureRevision == mnFeatureRevision;
}

bool KeyFrame::GetPointDescriptor(int cam, int idx, cv::Mat &descriptor) const {
    unique_lock<mutex> lock(mMutexFeatures);
    descriptor.release();

    if (cam < 0 || cam >= static_cast<int>(mvDescriptors_point.size()) ||
        idx < 0) {
        return false;
    }

    const cv::Mat &descriptors = mvDescriptors_point[cam];
    if (!IsValidPointDescriptorRow(descriptors, idx)) {
        return false;
    }

    descriptors.row(idx).copyTo(descriptor);
    return true;
}

KeyFrame::PointObservationDescriptorStatus
KeyFrame::GetPointObservationDescriptor(const MapPointPtr &pMP, int cam,
                                        int idx, int &resolved_idx,
                                        cv::Mat &descriptor) const {
    unique_lock<mutex> lock(mMutexFeatures);
    descriptor.release();
    resolved_idx = -1;

    if (!pMP || cam < 0 || cam >= static_cast<int>(mvpMapPoints_cams.size()) ||
        cam >= static_cast<int>(mvDescriptors_point.size())) {
        return PointObservationDescriptorStatus::InvalidDescriptor;
    }

    const cv::Mat &descriptors = mvDescriptors_point[cam];
    auto copy_descriptor = [&](int row) {
        if (!IsValidPointDescriptorRow(descriptors, row)) {
            return PointObservationDescriptorStatus::InvalidDescriptor;
        }
        descriptors.row(row).copyTo(descriptor);
        resolved_idx = row;
        return PointObservationDescriptorStatus::Found;
    };

    const vector<MapPointPtr> &points = mvpMapPoints_cams[cam];

    // Before sparsification the feature rows are still the original Frame rows.
    // Some construction paths add the MapPoint observation before the KeyFrame
    // row is filled, so preserve the historical index-based behaviour there.
    if (!mbSparsified) {
        return copy_descriptor(idx);
    }

    if (idx >= 0 && idx < static_cast<int>(points.size()) &&
        points[idx] == pMP) {
        return copy_descriptor(idx);
    }

    for (size_t candidate_idx = 0; candidate_idx < points.size();
         ++candidate_idx) {
        if (points[candidate_idx] == pMP) {
            return copy_descriptor(static_cast<int>(candidate_idx));
        }
    }

    return PointObservationDescriptorStatus::Missing;
}

bool KeyFrame::UpdateCountInLocalMapping(bool is_local,
                                         int non_local_threshold) {
    unique_lock<mutex> lock(mMutexFeatures);
    if (is_local) {
        mnCountInLocal = 0;
        return false;
    }
    if (mbSparsified)
        return false;
    ++mnCountInLocal;
    return mnCountInLocal >= non_local_threshold;
}

bool KeyFrame::isSparsified() const {
    unique_lock<mutex> lock(mMutexFeatures);
    return mbSparsified;
}

void KeyFrame::CompactPointFeatures() {
    while (true) {
        PointFeatureCompaction compaction = PreparePointFeatureCompaction();
        if (!compaction.valid ||
            CommitPointFeatureCompaction(std::move(compaction))) {
            return;
        }
    }
}

KeyFrame::PointFeatureCompaction
KeyFrame::PreparePointFeatureCompaction() const {
    const PointFeatureSnapshot snapshot =
        GetPointFeatureSnapshot(false, false, true);
    PointFeatureCompaction compaction;
    if (snapshot.mbSparsified)
        return compaction;

    compaction.source_feature_revision = snapshot.mnFeatureRevision;
    compaction.keys.resize(snapshot.mnCam);
    compaction.scores.resize(snapshot.mnCam);
    compaction.points.resize(snapshot.mnCam);
    compaction.grids.resize(snapshot.mnCam);
    compaction.descriptor_rows_per_camera.assign(snapshot.mnCam, 0);
    vector<cv::Mat> descriptors_for_bow;

    for (int cam = 0; cam < snapshot.mnCam; ++cam) {
        compaction.grids[cam].assign(
            snapshot.mnGridRowsPoint,
            vector<vector<size_t>>(snapshot.mnGridColsPoint));

        const size_t point_count =
            min({snapshot.mvpMapPoints_cams[cam].size(),
                 snapshot.mvKeysUn_cams[cam].size(),
                 snapshot.mvKeyScores_cams[cam].size()});
        compaction.keys[cam].reserve(point_count);
        compaction.scores[cam].reserve(point_count);
        compaction.points[cam].reserve(point_count);

        for (size_t old_index = 0; old_index < point_count; ++old_index) {
            const MapPointPtr &point =
                snapshot.mvpMapPoints_cams[cam][old_index];
            if (!point || point->isBad())
                continue;

            const size_t new_index = compaction.points[cam].size();
            compaction.points[cam].push_back(point);
            compaction.keys[cam].push_back(
                snapshot.mvKeysUn_cams[cam][old_index]);
            compaction.scores[cam].push_back(
                snapshot.mvKeyScores_cams[cam][old_index]);

            if (cam >= static_cast<int>(
                           snapshot.mvDescriptors_point.size()) ||
                old_index >=
                    static_cast<size_t>(
                        snapshot.mvDescriptors_point[cam].rows) ||
                snapshot.mvDescriptors_point[cam].type() != CV_32F) {
                throw runtime_error(
                    "Cannot compact KeyFrame point features: retained "
                    "MapPoint has no valid CV_32F descriptor");
            }

            const cv::Mat descriptor = snapshot.mvDescriptors_point[cam].row(
                static_cast<int>(old_index));
            compaction.descriptors.push_back(descriptor);
            ++compaction.descriptor_rows_per_camera[cam];
            descriptors_for_bow.push_back(descriptor);
            compaction.row_mapping.emplace_back(cam,
                                                static_cast<int>(new_index));

            int grid_x = 0;
            int grid_y = 0;
            if (GeometricTools::ImagePointToGridCell(
                    compaction.keys[cam].back(), snapshot.mvnMinX[cam],
                    snapshot.mvnMinY[cam],
                    snapshot.mvfGridElementWidthInvPoint[cam],
                    snapshot.mvfGridElementHeightInvPoint[cam],
                    snapshot.mnGridColsPoint, snapshot.mnGridRowsPoint, grid_x,
                    grid_y)) {
                compaction.grids[cam][grid_y][grid_x].push_back(new_index);
            }

            compaction.observation_updates.emplace_back(
                point, static_cast<int>(new_index), cam);
        }
    }

    if (mpVocabulary_point && !descriptors_for_bow.empty()) {
        const int levelsup_point =
            ComputeFeatureVectorLevelsup(mpVocabulary_point);
        mpVocabulary_point->transform(descriptors_for_bow, compaction.bow,
                                      compaction.feature_vector,
                                      levelsup_point);
    }
    compaction.valid = true;
    return compaction;
}

bool KeyFrame::CommitPointFeatureCompaction(
    PointFeatureCompaction &&compaction) {
    const KeyFramePtr self = shared_from_this();
    unique_lock<mutex> lock(mMutexFeatures);
    if (!compaction.valid || mbSparsified ||
        compaction.source_feature_revision != mnFeatureRevision) {
        return false;
    }

    mvKeysUn_cams.swap(compaction.keys);
    mvKeyScores_cams.swap(compaction.scores);
    mvpMapPoints_cams.swap(compaction.points);
    mvGrid_point.swap(compaction.grids);
    mvPointDescriptorRowToCamRow.swap(compaction.row_mapping);
    mDescriptors_point = std::move(compaction.descriptors);
    BuildCameraDescriptorViews(mDescriptors_point,
                               compaction.descriptor_rows_per_camera,
                               mvDescriptors_point);

    Np.resize(mnCam);
    mnTotalPoints = 0;
    for (int cam = 0; cam < mnCam; ++cam) {
        Np[cam] = static_cast<int>(mvpMapPoints_cams[cam].size());
        mnTotalPoints += Np[cam];
    }

    mBowVec_point = std::move(compaction.bow);
    mFeatVec_point = std::move(compaction.feature_vector);
    ++mnFeatureRevision;

    for (const auto &[point, new_index, cam] :
         compaction.observation_updates) {
        point->UpdateObservation(self, new_index, cam);
    }

    mbSparsified = true;
    return true;
}

void KeyFrame::AppendMapPoints(vector<MapPointPtr> &vpMapPoints,
                               unordered_set<MapPointPtr> &spMapPoints) const {
    vector<MapPointPtr> vpCandidates;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        for (const auto &vpCameraPoints : mvpMapPoints_cams) {
            for (MapPointPtr pMP : vpCameraPoints) {
                if (pMP && spMapPoints.insert(pMP).second)
                    vpCandidates.push_back(pMP);
            }
        }
    }
    for (MapPointPtr pMP : vpCandidates) {
        if (!pMP->isBad())
            vpMapPoints.push_back(pMP);
    }
}

void KeyFrame::AppendMapLines(vector<MapLinePtr> &vpMapLines,
                              unordered_set<MapLinePtr> &spMapLines) const {
    vector<MapLinePtr> vpCandidates;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        for (const auto &vpCameraLines : mvpMapLines_cams) {
            for (MapLinePtr pML : vpCameraLines) {
                if (pML && spMapLines.insert(pML).second)
                    vpCandidates.push_back(pML);
            }
        }
    }
    for (MapLinePtr pML : vpCandidates) {
        if (!pML->isBad())
            vpMapLines.push_back(pML);
    }
}

void KeyFrame::AppendMapPointAssociations(
    vector<tuple<MapPointPtr, size_t, int>> &vAssociations) const {
    unique_lock<mutex> lock(mMutexFeatures);
    for (int cam = 0; cam < mnCam; ++cam) {
        for (size_t idx = 0; idx < mvpMapPoints_cams[cam].size(); ++idx) {
            MapPointPtr pMP = mvpMapPoints_cams[cam][idx];
            if (pMP)
                vAssociations.emplace_back(pMP, idx, cam);
        }
    }
}

void KeyFrame::AppendMapLineAssociations(
    vector<tuple<MapLinePtr, size_t, int>> &vAssociations) const {
    unique_lock<mutex> lock(mMutexFeatures);
    for (int cam = 0; cam < mnCam; ++cam) {
        for (size_t idx = 0; idx < mvpMapLines_cams[cam].size(); ++idx) {
            MapLinePtr pML = mvpMapLines_cams[cam][idx];
            if (pML)
                vAssociations.emplace_back(pML, idx, cam);
        }
    }
}

vector<size_t> KeyFrame::GetPointFeaturesInArea(const float &x, const float &y,
                                                const float &r,
                                                int cam_id) const {
    vector<size_t> indices;
    indices.reserve(Np[cam_id]);

    const int nMinCellX =
        max(0, (int)floor((x - mvnMinX[cam_id] - r) *
                          mvfGridElementWidthInvPoint[cam_id]));
    if (nMinCellX >= FRAME_GRID_COLS_POINT) {
        return indices;
    }

    const int nMaxCellX = min(static_cast<int>(FRAME_GRID_COLS_POINT) - 1,
                              (int)ceil((x - mvnMinX[cam_id] + r) *
                                        mvfGridElementWidthInvPoint[cam_id]));
    if (nMaxCellX < 0) {
        return indices;
    }

    const int nMinCellY =
        max(0, (int)floor((y - mvnMinY[cam_id] - r) *
                          mvfGridElementHeightInvPoint[cam_id]));
    if (nMinCellY >= FRAME_GRID_ROWS_POINT) {
        return indices;
    }

    const int nMaxCellY = min(static_cast<int>(FRAME_GRID_ROWS_POINT) - 1,
                              (int)ceil((y - mvnMinY[cam_id] + r) *
                                        mvfGridElementHeightInvPoint[cam_id]));
    if (nMaxCellY < 0) {
        return indices;
    }

    for (int ix = nMinCellX; ix <= nMaxCellX; ++ix) {
        for (int iy = nMinCellY; iy <= nMaxCellY; ++iy) {
            const vector<size_t> &cell_indices = mvGrid_point[cam_id][iy][ix];
            if (cell_indices.empty()) {
                continue;
            }

            for (size_t idx : cell_indices) {
                const cv::Point2f &kp = mvKeysUn_cams[cam_id][idx];

                const float dist_x = kp.x - x;
                const float dist_y = kp.y - y;

                if (fabs(dist_x) < r && fabs(dist_y) < r) {
                    indices.push_back(idx);
                }
            }
        }
    }

    return indices;
}

vector<size_t> KeyFrame::GetLineFeaturesInArea(const float &d,
                                               const float &theta,
                                               const float &r_d,
                                               const float &r_theta,
                                               const int cam_id) const {
    vector<size_t> indices;

    indices.reserve(Nl[cam_id]);

    float query_d = d;
    float query_theta = theta;
    if (!GeometricTools::NormalizeLinePolar(query_d, query_theta)) {
        return indices;
    }

    auto appendLineCells = [&](float center_d, float min_theta,
                               float max_theta) {
        min_theta = std::max(0.0f, min_theta);
        max_theta = std::min(GeometricTools::kPi, max_theta);
        if (min_theta > max_theta) {
            return;
        }

        const int nMinCellX =
            max(0, static_cast<int>(
                       floor(min_theta * mvfGridElementWidthInvLine[cam_id])));
        if (nMinCellX >= mnGridColsLine) {
            return;
        }

        const int nMaxCellX =
            min(mnGridColsLine - 1,
                static_cast<int>(
                    ceil(max_theta * mvfGridElementWidthInvLine[cam_id])));
        if (nMaxCellX < 0) {
            return;
        }

        const int nMinCellY = max(
            0, static_cast<int>(floor((center_d + mvLineDMax[cam_id] - r_d) *
                                      mvfGridElementHeightInvLine[cam_id])));
        if (nMinCellY >= mnGridRowsLine) {
            return;
        }

        const int nMaxCellY =
            min(mnGridRowsLine - 1,
                static_cast<int>(ceil((center_d + mvLineDMax[cam_id] + r_d) *
                                      mvfGridElementHeightInvLine[cam_id])));
        if (nMaxCellY < 0) {
            return;
        }

        for (int ix = nMinCellX; ix <= nMaxCellX; ++ix) {
            for (int iy = nMinCellY; iy <= nMaxCellY; ++iy) {
                const vector<size_t> &cell_indices =
                    mvGrid_line[cam_id][iy][ix];
                if (cell_indices.empty()) {
                    continue;
                }

                for (size_t idx : cell_indices) {
                    if (idx >= mvLinePolarUn_cams[cam_id].size()) {
                        continue;
                    }
                    const Eigen::Vector2f &line_polar =
                        mvLinePolarUn_cams[cam_id][idx];
                    if (!line_polar.allFinite()) {
                        continue;
                    }

                    const float direct_theta =
                        fabs(line_polar(1) - query_theta);
                    const float direct_d = fabs(line_polar(0) - query_d);
                    const float flipped_theta =
                        GeometricTools::kPi - direct_theta;
                    const float flipped_d = fabs(line_polar(0) + query_d);
                    if ((direct_d < r_d && direct_theta < r_theta) ||
                        (flipped_d < r_d && flipped_theta < r_theta)) {
                        indices.push_back(idx);
                    }
                }
            }
        }
    };

    appendLineCells(query_d, query_theta - r_theta, query_theta + r_theta);
    if (query_theta - r_theta < 0.0f) {
        appendLineCells(-query_d, GeometricTools::kPi + query_theta - r_theta,
                        GeometricTools::kPi);
    }
    if (query_theta + r_theta >= GeometricTools::kPi) {
        appendLineCells(-query_d, 0.0f,
                        query_theta + r_theta - GeometricTools::kPi);
    }

    return indices;
}

Eigen::Vector3f KeyFrame::GetCameraCenter(int cam_id) const {
    unique_lock<mutex> lock(mMutexPose);
    return (mTwb * mvTbc[cam_id]).translation();
}

vector<Eigen::Vector3f> KeyFrame::GetCameraCenter() const {
    unique_lock<mutex> lock(mMutexPose);
    vector<Eigen::Vector3f> centers(mnCam);
    for (int cam_id = 0; cam_id < mnCam; ++cam_id) {
        centers[cam_id] = (mTwb * mvTbc[cam_id]).translation();
    }
    return centers;
}

Eigen::Vector3f KeyFrame::GetTranslation() const {
    unique_lock<mutex> lock(mMutexPose);
    return mTbw.translation();
}

Eigen::Matrix3f KeyFrame::GetRotation() const {
    unique_lock<mutex> lock(mMutexPose);
    return mTbw.rotationMatrix();
}

vector<Sophus::SE3f> KeyFrame::GetCameraExtrinsics() const {
    unique_lock<mutex> lock(mMutexPose);
    return mvTbc;
}

vector<KeyFramePtr> KeyFrame::GetBestCovisibilityKeyFrames(const int &N) const {
    unique_lock<mutex> lock(mMutexConnections);
    vector<KeyFramePtr> keyframes;
    keyframes.reserve(
        min(static_cast<size_t>(N), mvpOrderedConnectedKeyFrames.size()));
    for (const KeyFrameWeakPtr &weakKF : mvpOrderedConnectedKeyFrames) {
        if (auto pKF = weakKF.lock()) {
            keyframes.push_back(pKF);
            if (static_cast<int>(keyframes.size()) == N)
                break;
        }
    }
    return keyframes;
}

set<KeyFramePtr> KeyFrame::GetConnectedKeyFrames() {
    unique_lock<mutex> lock(mMutexConnections);
    set<KeyFramePtr> connected;
    for (auto it = mConnectedKeyFrameWeights.begin();
         it != mConnectedKeyFrameWeights.end();) {
        if (auto pKF = it->first.lock()) {
            connected.insert(pKF);
            ++it;
        } else {
            it = mConnectedKeyFrameWeights.erase(it);
        }
    }
    return connected;
}

vector<KeyFramePtr> KeyFrame::GetCovisiblesByWeight(const int &w) {
    unique_lock<mutex> lock(mMutexConnections);

    if (mvpOrderedConnectedKeyFrames.empty()) {
        return {};
    }

    vector<int>::iterator it =
        upper_bound(mvOrderedWeights.begin(), mvOrderedWeights.end(), w,
                    KeyFrame::weightComp);

    if (it == mvOrderedWeights.end() && mvOrderedWeights.back() < w) {
        return {};
    } else {
        int n = it - mvOrderedWeights.begin();
        vector<KeyFramePtr> keyframes;
        keyframes.reserve(n);
        for (int i = 0; i < n; ++i) {
            if (auto pKF = mvpOrderedConnectedKeyFrames[i].lock())
                keyframes.push_back(pKF);
        }
        return keyframes;
    }
}

set<KeyFramePtr> KeyFrame::GetChilds() const {
    unique_lock<mutex> lockCon(mMutexConnections);
    set<KeyFramePtr> children;
    for (const KeyFrameWeakPtr &weakKF : mspChildrens) {
        if (auto pKF = weakKF.lock())
            children.insert(pKF);
    }
    return children;
}

KeyFramePtr KeyFrame::GetParent() const {
    unique_lock<mutex> lockCon(mMutexConnections);
    return mpParent;
}

void KeyFrame::AddLoopEdge(const KeyFramePtr &pKF) {
    unique_lock<mutex> lock(mMutexConnections);
    mbNotErase = true;
    mspLoopEdges.insert(pKF);
}

set<KeyFramePtr> KeyFrame::GetLoopEdges() {
    unique_lock<mutex> lock(mMutexConnections);
    set<KeyFramePtr> edges;
    for (const KeyFrameWeakPtr &weakKF : mspLoopEdges) {
        if (auto pKF = weakKF.lock())
            edges.insert(pKF);
    }
    return edges;
}

int KeyFrame::TrackedMapPoints(const int &minObs) const {
    unique_lock<mutex> lock(mMutexFeatures);

    int nPoints = 0;
    const bool bCheckObs = minObs > 0;
    for (int cam_id = 0; cam_id < mnCam; ++cam_id) {
        for (int i = 0; i < Np[cam_id]; ++i) {
            MapPointPtr pMP = mvpMapPoints_cams[cam_id][i];
            if (pMP) {
                if (!pMP->isBad()) {
                    if (bCheckObs) {
                        if (pMP->Observations() >= minObs) {
                            nPoints++;
                        }
                    } else {
                        nPoints++;
                    }
                }
            }
        }
    }
    return nPoints;
}

int KeyFrame::TrackedMapLines(const int &minObs) const {
    unique_lock<mutex> lock(mMutexFeatures);

    int nLines = 0;
    const bool bCheckObs = minObs > 0;
    for (int cam_id = 0; cam_id < mnCam; ++cam_id) {
        for (int i = 0; i < Nl[cam_id]; ++i) {
            MapLinePtr pML = mvpMapLines_cams[cam_id][i];
            if (pML) {
                if (!pML->isBad()) {
                    if (bCheckObs) {
                        if (pML->Observations() >= minObs) {
                            nLines++;
                        }
                    } else {
                        nLines++;
                    }
                }
            }
        }
    }
    return nLines;
}

Sophus::SE3f KeyFrame::GetPoseInverse() const {
    unique_lock<mutex> lock(mMutexPose);
    return mTwb;
}

Sophus::SE3f KeyFrame::GetPose() const {
    unique_lock<mutex> lock(mMutexPose);
    return mTbw;
}

Sophus::SE3f KeyFrame::GetCameraPose(int cam_id) const {
    unique_lock<mutex> lock(mMutexPose);
    return mvTbc[cam_id].inverse() * mTbw;
}

vector<Sophus::SE3f> KeyFrame::GetCameraPose() const {
    unique_lock<mutex> lock(mMutexPose);
    vector<Sophus::SE3f> camera_poses(mnCam);
    for (int cam_id = 0; cam_id < mnCam; ++cam_id) {
        camera_poses[cam_id] = mvTbc[cam_id].inverse() * mTbw;
    }
    return camera_poses;
}

Sophus::SE3f KeyFrame::GetCameraPoseInverse(int cam_id) const {
    unique_lock<mutex> lock(mMutexPose);
    return mTwb * mvTbc[cam_id];
}

vector<Sophus::SE3f> KeyFrame::GetCameraPoseInverse() const {
    unique_lock<mutex> lock(mMutexPose);
    vector<Sophus::SE3f> camera_poses_inverse(mnCam);
    for (int cam_id = 0; cam_id < mnCam; ++cam_id) {
        camera_poses_inverse[cam_id] = mTwb * mvTbc[cam_id];
    }
    return camera_poses_inverse;
}

void KeyFrame::ComputeBoW() {
    if (mBowVec_point.empty() || mFeatVec_point.empty()) {
        vector<cv::Mat> vCurrentDesc =
            Converter::toDescriptorVector(mDescriptors_point);
        const int levelsup_point =
            ComputeFeatureVectorLevelsup(mpVocabulary_point);
        mpVocabulary_point->transform(vCurrentDesc, mBowVec_point,
                                      mFeatVec_point, levelsup_point);
    }

    if (mBowVec_line.empty() || mFeatVec_line.empty()) {
        vector<cv::Mat> vCurrentDesc =
            Converter::toDescriptorVector(mDescriptors_line);
        const int levelsup_line =
            ComputeFeatureVectorLevelsup(mpVocabulary_line);
        mpVocabulary_line->transform(vCurrentDesc, mBowVec_line, mFeatVec_line,
                                     levelsup_line);
    }
}

void KeyFrame::UpdateConnections(bool upParent) {
    map<KeyFramePtr, int> KFcounter;

    vector<vector<MapPointPtr>> vpMPs;
    vector<vector<MapLinePtr>> vpMLs;

    {
        unique_lock<mutex> lock(mMutexFeatures);
        vpMPs = mvpMapPoints_cams;
        vpMLs = mvpMapLines_cams;
    }

    unordered_set<MapPointPtr> spMPs;
    unordered_set<MapLinePtr> spMLs;
    for (int cam_id = 0; cam_id < mnCam; ++cam_id) {
        for (size_t i = 0; i < vpMPs[cam_id].size(); ++i) {
            MapPointPtr pMP = vpMPs[cam_id][i];
            if (!pMP)
                continue;
            if (pMP->isBad())
                continue;

            if (spMPs.count(pMP)) {
                continue;
            }
            spMPs.insert(pMP);

            auto observations = pMP->GetObservations();

            for (auto &obs : observations) {
                if (obs.first->mnId == mnId || obs.first->isBad() ||
                    obs.first->GetMap() != mpMap) {
                    continue;
                }
                KFcounter[obs.first]++;
            }
        }

        for (size_t i = 0; i < vpMLs[cam_id].size(); ++i) {
            MapLinePtr pML = vpMLs[cam_id][i];
            if (!pML)
                continue;
            if (pML->isBad())
                continue;

            if (spMLs.count(pML))
                continue;
            spMLs.insert(pML);

            auto observations = pML->GetObservations();

            for (auto &obs : observations) {
                if (obs.first->mnId == mnId || obs.first->isBad() ||
                    obs.first->GetMap() != mpMap) {
                    continue;
                }
                KFcounter[obs.first]++;
            }
        }
    }

    if (KFcounter.empty()) {
        return;
    }

    const bool this_keyframe_sparsified = isSparsified();
    if (this_keyframe_sparsified) {
        unique_lock<mutex> lockCon(mMutexConnections);
        for (const auto &[weak_kf, weight] : mConnectedKeyFrameWeights) {
            KeyFramePtr connected_keyframe = weak_kf.lock();
            if (!connected_keyframe || connected_keyframe->isBad() ||
                connected_keyframe->GetMap() != mpMap) {
                continue;
            }
            KFcounter[connected_keyframe] = weight;
        }
    }

    int nmax = 0;
    KeyFramePtr pKFmax = nullptr;
    const int default_threshold = 15;
    const int sparsified_threshold = 7;

    vector<pair<int, KeyFramePtr>> vPairs;
    vPairs.reserve(KFcounter.size());
    if (!upParent)
        cout << "UPDATE_CONN: current KF " << mnId << endl;
    for (map<KeyFramePtr, int>::iterator mit = KFcounter.begin(),
                                         mend = KFcounter.end();
         mit != mend; mit++) {
        if (!upParent)
            cout << "  UPDATE_CONN: KF " << mit->first->mnId
                 << " ; num matches: " << mit->second << endl;
        if (mit->second > nmax) {
            nmax = mit->second;
            pKFmax = mit->first;
        }
        const int threshold =
            (this_keyframe_sparsified || mit->first->isSparsified())
                ? sparsified_threshold
                : default_threshold;
        if (mit->second >= threshold) {
            vPairs.push_back(make_pair(mit->second, mit->first));
            (mit->first)->AddConnection(shared_from_this(), mit->second);
        }
    }

    if (vPairs.empty()) {
        vPairs.push_back(make_pair(nmax, pKFmax));
        pKFmax->AddConnection(shared_from_this(), nmax);
    }

    sort(vPairs.begin(), vPairs.end());
    list<KeyFramePtr> lKFs;
    list<int> lWs;
    for (size_t i = 0; i < vPairs.size(); i++) {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    {
        unique_lock<mutex> lockCon(mMutexConnections);

        mConnectedKeyFrameWeights.clear();
        for (const auto &[pKF, weight] : KFcounter)
            mConnectedKeyFrameWeights.emplace(KeyFrameWeakPtr(pKF), weight);
        mvpOrderedConnectedKeyFrames.assign(lKFs.begin(), lKFs.end());
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());

        if (mbFirstConnection && mnId != mpMap->GetInitKFid()) {
            mpParent = mvpOrderedConnectedKeyFrames.front().lock();
            if (mpParent)
                mpParent->AddChild(shared_from_this());
            mbFirstConnection = false;
        }
    }
}

void KeyFrame::AddConnection(const KeyFramePtr &pKF, const int &weight) {
    {
        unique_lock<mutex> lock(mMutexConnections);
        const KeyFrameWeakPtr key(pKF);
        const auto it = mConnectedKeyFrameWeights.find(key);
        if (it == mConnectedKeyFrameWeights.end())
            mConnectedKeyFrameWeights[key] = weight;
        else if (it->second != weight)
            it->second = weight;
        else
            return;
    }

    UpdateBestCovisibles();
}

void KeyFrame::UpdateBestCovisibles() {
    unique_lock<mutex> lock(mMutexConnections);
    vector<pair<int, KeyFramePtr>> vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());
    for (auto it = mConnectedKeyFrameWeights.begin();
         it != mConnectedKeyFrameWeights.end();) {
        if (auto pKF = it->first.lock()) {
            vPairs.push_back(make_pair(it->second, pKF));
            ++it;
        } else {
            it = mConnectedKeyFrameWeights.erase(it);
        }
    }

    sort(vPairs.begin(), vPairs.end());
    list<KeyFramePtr> lKFs;
    list<int> lWs;
    for (size_t i = 0, iend = vPairs.size(); i < iend; i++) {
        if (!vPairs[i].second->isBad()) {
            lKFs.push_front(vPairs[i].second);
            lWs.push_front(vPairs[i].first);
        }
    }

    mvpOrderedConnectedKeyFrames.assign(lKFs.begin(), lKFs.end());
    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());
}

void KeyFrame::AddChild(const KeyFramePtr &pKF) {
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.insert(pKF);
}

void KeyFrame::EraseMapPointMatch(const int &idx, int cam) {
    unique_lock<mutex> lock(mMutexFeatures);
    if (cam < 0 || cam >= static_cast<int>(mvpMapPoints_cams.size()) ||
        idx < 0 || idx >= static_cast<int>(mvpMapPoints_cams[cam].size())) {
        return;
    }
    if (mvpMapPoints_cams[cam][idx]) {
        mvpMapPoints_cams[cam][idx] = nullptr;
        ++mnFeatureRevision;
    }
}

void KeyFrame::EraseMapPointMatch(const MapPointPtr &pMP, int cam) {
    unique_lock<mutex> lock(mMutexFeatures);
    if (!pMP || cam < 0 || cam >= static_cast<int>(mvpMapPoints_cams.size())) {
        return;
    }

    bool changed = false;
    for (MapPointPtr &point : mvpMapPoints_cams[cam]) {
        if (point == pMP) {
            point = nullptr;
            changed = true;
        }
    }
    if (changed)
        ++mnFeatureRevision;
}

void KeyFrame::EraseMapLineMatch(const int &idx, int cam) {
    unique_lock<mutex> lock(mMutexFeatures);
    if (cam < 0 || cam >= static_cast<int>(mvpMapLines_cams.size()) ||
        idx < 0 || idx >= static_cast<int>(mvpMapLines_cams[cam].size())) {
        return;
    }
    if (mvpMapLines_cams[cam][idx]) {
        mvpMapLines_cams[cam][idx] = nullptr;
        ++mnFeatureRevision;
    }
}

void KeyFrame::EraseMapLineMatch(const MapLinePtr &pML, int cam) {
    int index = pML->GetIndexInKeyFrame(shared_from_this(), cam);
    if (index != -1)
        EraseMapLineMatch(index, cam);
}

MapPointPtr KeyFrame::GetMapPoint(const size_t &idx, int cam) {
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints_cams[cam][idx];
}

MapLinePtr KeyFrame::GetMapLine(const size_t &idx, int cam) {
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapLines_cams[cam][idx];
}

bool KeyFrame::IsPointInImage(const float &x, const float &y,
                              int cam_id) const {
    return (x >= mvnMinX[cam_id] && x < mvnMaxX[cam_id] &&
            y >= mvnMinY[cam_id] && y < mvnMaxY[cam_id]);
}

bool KeyFrame::IsLineInImage(const Eigen::Vector3f &line, int cam_id) const {
    if (cam_id < 0 || cam_id >= mnCam || !line.allFinite()) {
        return false;
    }

    const float a = line(0);
    const float b = line(1);
    const float c = line(2);
    if (std::abs(a) < 1e-6f && std::abs(b) < 1e-6f) {
        return false;
    }

    const float min_x = static_cast<float>(mvnMinX[cam_id]);
    const float max_x = static_cast<float>(mvnMaxX[cam_id]);
    const float min_y = static_cast<float>(mvnMinY[cam_id]);
    const float max_y = static_cast<float>(mvnMaxY[cam_id]);

    const float v1 = a * min_x + b * min_y + c;
    const float v2 = a * max_x + b * min_y + c;
    const float v3 = a * max_x + b * max_y + c;
    const float v4 = a * min_x + b * max_y + c;

    const float min_value = std::min(std::min(v1, v2), std::min(v3, v4));
    const float max_value = std::max(std::max(v1, v2), std::max(v3, v4));
    return min_value <= 0.0f && max_value >= 0.0f;
}

void KeyFrame::ReplaceMapPointMatch(const int &idx, const MapPointPtr &pMP,
                                    int cam) {
    unique_lock<mutex> lock(mMutexFeatures);
    if (cam < 0 || cam >= static_cast<int>(mvpMapPoints_cams.size()) ||
        idx < 0 || idx >= static_cast<int>(mvpMapPoints_cams[cam].size())) {
        return;
    }
    if (mvpMapPoints_cams[cam][idx] != pMP) {
        mvpMapPoints_cams[cam][idx] = pMP;
        ++mnFeatureRevision;
    }
}

void KeyFrame::ReplaceMapLineMatch(const int &idx, const MapLinePtr &pML,
                                   int cam) {
    unique_lock<mutex> lock(mMutexFeatures);
    if (cam < 0 || cam >= static_cast<int>(mvpMapLines_cams.size()) ||
        idx < 0 || idx >= static_cast<int>(mvpMapLines_cams[cam].size())) {
        return;
    }
    if (mvpMapLines_cams[cam][idx] != pML) {
        mvpMapLines_cams[cam][idx] = pML;
        ++mnFeatureRevision;
    }
}

vector<KeyFramePtr> KeyFrame::GetVectorCovisibleKeyFrames() {
    return GetBestCovisibilityKeyFrames(
        static_cast<int>(mvpOrderedConnectedKeyFrames.size()));
}

void KeyFrame::SetBadFlag() {
    const KeyFramePtr keepAlive = shared_from_this();
    {
        unique_lock<mutex> lock(mMutexConnections);
        if (mnId == mpMap->GetInitKFid()) {
            return;
        } else if (mbNotErase) {
            mbToBeErased = true;
            return;
        }
    }

    const auto connected = GetConnectedKeyFrames();
    for (const KeyFramePtr &pKF : connected)
        pKF->EraseConnection(keepAlive);

    for (int c = 0; c < mnCam; ++c) {
        for (size_t i = 0; i < mvpMapPoints_cams[c].size(); i++) {
            if (mvpMapPoints_cams[c][i]) {
                mvpMapPoints_cams[c][i]->EraseObservation(keepAlive, c);
            }
        }
    }

    for (int c = 0; c < mnCam; ++c) {
        for (size_t i = 0; i < mvpMapLines_cams[c].size(); i++) {
            if (mvpMapLines_cams[c][i]) {
                mvpMapLines_cams[c][i]->EraseObservation(keepAlive, c);
            }
        }
    }

    {
        unique_lock<mutex> lock(mMutexConnections);
        unique_lock<mutex> lock1(mMutexFeatures);

        mConnectedKeyFrameWeights.clear();
        mvpOrderedConnectedKeyFrames.clear();

        // Update Spanning Tree
        set<KeyFramePtr> sParentCandidates;
        if (mpParent)
            sParentCandidates.insert(mpParent);

        // Assign at each iteration one children with a parent (the pair with
        // highest covisibility weight) Include that children as new parent
        // candidate for the rest
        while (!mspChildrens.empty()) {
            bool bContinue = false;

            int max = -1;
            KeyFramePtr pC;
            KeyFramePtr pP;

            for (auto sit = mspChildrens.begin(); sit != mspChildrens.end();) {
                KeyFramePtr pKF = sit->lock();
                if (!pKF) {
                    sit = mspChildrens.erase(sit);
                    continue;
                }
                ++sit;
                if (pKF->isBad())
                    continue;

                // Check if a parent candidate is connected to the keyframe
                vector<KeyFramePtr> vpConnected =
                    pKF->GetVectorCovisibleKeyFrames();
                for (size_t i = 0, iend = vpConnected.size(); i < iend; i++) {
                    for (set<KeyFramePtr>::iterator
                             spcit = sParentCandidates.begin(),
                             spcend = sParentCandidates.end();
                         spcit != spcend; spcit++) {
                        if (vpConnected[i]->mnId == (*spcit)->mnId) {
                            int w = pKF->GetWeight(vpConnected[i]);
                            if (w > max) {
                                pC = pKF;
                                pP = vpConnected[i];
                                max = w;
                                bContinue = true;
                            }
                        }
                    }
                }
            }

            if (bContinue) {
                pC->ChangeParent(pP);
                sParentCandidates.insert(pC);
                mspChildrens.erase(KeyFrameWeakPtr(pC));
            } else
                break;
        }

        // If a children has no covisibility links with any parent candidate,
        // assign to the original parent of this KF
        if (!mspChildrens.empty()) {
            for (const KeyFrameWeakPtr &weakChild : mspChildrens) {
                if (auto pChild = weakChild.lock())
                    pChild->ChangeParent(mpParent);
            }
        }

        if (mpParent) {
            mpParent->EraseChild(keepAlive);
            mTbp = mTbw * mpParent->GetPoseInverse();
        }
        mbBad = true;
    }

    mpMap->EraseKeyFrame(keepAlive);
    mpKeyFrameDB->erase(keepAlive);
}

void KeyFrame::EraseConnection(const KeyFramePtr &pKF) {
    bool bUpdate = false;
    {
        unique_lock<mutex> lock(mMutexConnections);
        const KeyFrameWeakPtr key(pKF);
        if (mConnectedKeyFrameWeights.count(key)) {
            mConnectedKeyFrameWeights.erase(key);
            bUpdate = true;
        }
    }

    if (bUpdate)
        UpdateBestCovisibles();
}

int KeyFrame::GetWeight(const KeyFramePtr &pKF) {
    unique_lock<mutex> lock(mMutexConnections);
    const KeyFrameWeakPtr key(pKF);
    if (mConnectedKeyFrameWeights.count(key))
        return mConnectedKeyFrameWeights[key];
    else
        return 0;
}

void KeyFrame::ChangeParent(const KeyFramePtr &pKF) {
    unique_lock<mutex> lockCon(mMutexConnections);
    if (pKF.get() == this) {
        cout << "ERROR: Change parent KF, the parent and child are the same KF"
             << endl;
        throw std::invalid_argument("The parent and child can not be the same");
    }

    mpParent = pKF;
    pKF->AddChild(shared_from_this());
}

void KeyFrame::EraseChild(const KeyFramePtr &pKF) {
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.erase(KeyFrameWeakPtr(pKF));
}

void KeyFrame::SetNotErase() {
    unique_lock<mutex> lock(mMutexConnections);
    mbNotErase = true;
}

void KeyFrame::SetErase() {
    {
        unique_lock<mutex> lock(mMutexConnections);
        for (auto it = mspLoopEdges.begin(); it != mspLoopEdges.end();) {
            if (it->expired())
                it = mspLoopEdges.erase(it);
            else
                ++it;
        }
        if (mspLoopEdges.empty()) {
            mbNotErase = false;
        }
    }

    if (mbToBeErased) {
        SetBadFlag();
    }
}

set<MapPointPtr> KeyFrame::GetMapPoints() {
    unique_lock<mutex> lock(mMutexFeatures);
    set<MapPointPtr> s;
    for (int cam_id = 0; cam_id < mnCam; ++cam_id) {
        for (size_t i = 0; i < mvpMapPoints_cams[cam_id].size(); i++) {
            MapPointPtr pMP = mvpMapPoints_cams[cam_id][i];
            if (pMP && !pMP->isBad()) {
                s.insert(pMP);
            }
        }
    }
    return s;
}

set<MapLinePtr> KeyFrame::GetMapLines() {
    unique_lock<mutex> lock(mMutexFeatures);
    set<MapLinePtr> s;
    for (int cam_id = 0; cam_id < mnCam; ++cam_id) {
        for (size_t i = 0; i < mvpMapLines_cams[cam_id].size(); i++) {
            MapLinePtr pML = mvpMapLines_cams[cam_id][i];
            if (pML && !pML->isBad()) {
                s.insert(pML);
            }
        }
    }
    return s;
}

void KeyFrame::SetFirstConnection(bool bFirst) {
    unique_lock<mutex> lockCon(mMutexConnections);
    mbFirstConnection = bFirst;
}

bool KeyFrame::hasChild(const KeyFramePtr &pKF) {
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens.count(KeyFrameWeakPtr(pKF));
}

void KeyFrame::AddMergeEdge(const KeyFramePtr &pKF) {
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true;
    mspMergeEdges.insert(pKF);
}

set<KeyFramePtr> KeyFrame::GetMergeEdges() {
    unique_lock<mutex> lockCon(mMutexConnections);
    set<KeyFramePtr> edges;
    for (const KeyFrameWeakPtr &weakKF : mspMergeEdges) {
        if (auto pKF = weakKF.lock())
            edges.insert(pKF);
    }
    return edges;
}

}; // namespace XPLSLAM
