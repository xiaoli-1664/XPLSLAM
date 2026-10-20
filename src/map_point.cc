#include "map_point.h"

#include <algorithm>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <tuple>

using namespace std;

namespace XPLSLAM {

namespace {

constexpr int kPointDescriptorCols = 64;

std::string ObservationDescriptorError(const KeyFramePtr &pKF, int cam,
                                       int idx) {
    std::ostringstream oss;
    oss << "Invalid MapPoint observation descriptor";
    if (pKF) {
        oss << " in KeyFrame " << pKF->mnId;
    }
    oss << " camera " << cam << " index " << idx;
    return oss.str();
}

cv::Mat RequirePointObservationDescriptor(const KeyFramePtr &pKF, int cam,
                                          int idx) {
    cv::Mat descriptor;
    if (!pKF || !pKF->GetPointDescriptor(cam, idx, descriptor) ||
        descriptor.rows != 1 || descriptor.cols != kPointDescriptorCols ||
        descriptor.type() != CV_32F) {
        throw std::runtime_error(ObservationDescriptorError(pKF, cam, idx));
    }
    return descriptor;
}

int SelectRepresentativeDescriptorIndex(const vector<cv::Mat> &descriptors) {
    const size_t N = descriptors.size();
    if (N <= 1) {
        return 0;
    }

    vector<vector<float>> similarities(N, vector<float>(N, 0.0f));
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i + 1; j < N; ++j) {
            const float similarity = descriptors[i].dot(descriptors[j]);
            similarities[i][j] = similarity;
            similarities[j][i] = similarity;
        }
    }

    float bestMedian = numeric_limits<float>::lowest();
    int bestIdx = 0;
    for (size_t i = 0; i < N; ++i) {
        vector<float> scores;
        scores.reserve(N - 1);
        for (size_t j = 0; j < N; ++j) {
            if (j != i) {
                scores.push_back(similarities[i][j]);
            }
        }

        sort(scores.begin(), scores.end());
        const float median = scores[(scores.size() - 1) / 2];
        if (median > bestMedian) {
            bestMedian = median;
            bestIdx = static_cast<int>(i);
        }
    }

    return bestIdx;
}

} // namespace

long unsigned int MapPoint::nNextId = 0;
int MapPoint::nCam = 0;
mutex MapPoint::mGlobalMutex;

MapPoint::MapPoint()
    : mnFirstKFid(0), mnFirstFrame(0), nObs(0), mnTrackReferenceForFrame(0),
      mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0),
      mnCorrectedByKF(0), mnCorrectedReference(0), mnBAGlobalForKF(0),
      mnVisible(1), mnFound(1), mbBad(false) {
    mvnLastFrameSeen.assign(std::max(nCam, 0), 0);
}

MapPoint::MapPoint(const Eigen::Vector3f &Pos, const KeyFramePtr &pRefKF,
                   Map *pMap)
    : mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0),
      mnTrackReferenceForFrame(0), mnBALocalForKF(0), mnFuseCandidateForKF(0),
      mnLoopPointForKF(0), mnCorrectedByKF(0), mnCorrectedReference(0),
      mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1),
      mbBad(false), mpMap(pMap),
      mnOriginMapId(pMap->GetId()) {
    SetWorldPos(Pos);

    mvbTrackInView.resize(pRefKF->mvpCameras.size(), false);
    mvTrackProjX.resize(pRefKF->mvpCameras.size(), -1.0f);
    mvTrackProjY.resize(pRefKF->mvpCameras.size(), -1.0f);
    mvTrackDepth.resize(pRefKF->mvpCameras.size(), -1.0f);
    mvnLastFrameSeen.assign(pRefKF->mvpCameras.size(), 0);

    // MapPoints can be created from Tracking and Local Mapping. This mutex
    // avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId = nNextId++;
}

MapPoint::MapPoint(const Eigen::Vector3f &Pos, Map *pMap, Frame *pFrame,
                   const int &idxF, const int &cam_id)
    : mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0),
      mnTrackReferenceForFrame(0), mnBALocalForKF(0), mnFuseCandidateForKF(0),
      mnLoopPointForKF(0), mnCorrectedByKF(0), mnCorrectedReference(0),
      mnBAGlobalForKF(0), mnVisible(1), mnFound(1), mbBad(false), mpMap(pMap),
      mnOriginMapId(pMap->GetId()) {
    SetWorldPos(Pos);

    mvbTrackInView.resize(pFrame->mvpCamera.size(), false);
    mvTrackProjX.resize(pFrame->mvpCamera.size(), -1.0f);
    mvTrackProjY.resize(pFrame->mvpCamera.size(), -1.0f);
    mvTrackDepth.resize(pFrame->mvpCamera.size(), -1.0f);
    mvnLastFrameSeen.assign(pFrame->mvpCamera.size(), 0);

    pFrame->mvDescriptors_point[cam_id].row(idxF).copyTo(mDescriptor);

    // MapPoints can be created from Tracking and Local Mapping. This mutex
    // avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId = nNextId++;
}

void MapPoint::SetWorldPos(const Eigen::Vector3f &Pos) {
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    mWorldPos = Pos;
}

bool MapPoint::isBad() {
    unique_lock<mutex> lock1(mMutexFeatures, std::defer_lock);
    unique_lock<mutex> lock2(mMutexPos, std::defer_lock);
    lock(lock1, lock2);

    return mbBad;
}

void MapPoint::UpdateMap(Map *pMap) {
    unique_lock<mutex> lock(mMutexMap);
    mpMap = pMap;
}

void MapPoint::PostLoad(map<long unsigned int, KeyFramePtr> &mpKFid,
                        map<long unsigned int, MapPointPtr> &mpMPid) {
    mpRefKF = mpKFid[mBackupRefKFId];
    if (mpRefKF.expired()) {
        cout << "ERROR: MP without KF reference " << mBackupRefKFId
             << "; Num obs: " << nObs << endl;
    }
    mpReplaced.reset();
    if (mBackupReplacedId >= 0) {
        auto it = mpMPid.find(mBackupReplacedId);
        if (it != mpMPid.end())
            mpReplaced = it->second;
    }

    mObservations.clear();

    for (map<long unsigned int, vector<int>>::const_iterator
             it = mBackupObservationsId.begin(),
             end = mBackupObservationsId.end();
         it != end; ++it) {
        KeyFramePtr pKFi = mpKFid[it->first];
        vector<int> indexes = vector<int>(it->second);
        if (pKFi) {
            mObservations[pKFi] = indexes;
        }
    }

    mBackupObservationsId.clear();
}

void MapPoint::AddObservation(const KeyFramePtr &pKF, int idx, int cam_id) {
    const cv::Mat observation_descriptor =
        RequirePointObservationDescriptor(pKF, cam_id, idx);

    unique_lock<mutex> lock(mMutexFeatures);
    vector<int> indexes;
    const KeyFrameWeakPtr key(pKF);

    if (auto it = mObservations.find(key); it != mObservations.end()) {
        indexes = it->second;
    } else {
        indexes = vector<int>(nCam, -1);
    }

    indexes[cam_id] = idx;

    mObservations[key] = indexes;

    nObs++;
    if (mDescriptor.empty()) {
        observation_descriptor.copyTo(mDescriptor);
    }
}

void MapPoint::UpdateObservation(const KeyFramePtr &pKF, int idx, int cam_id) {
    unique_lock<mutex> lock(mMutexFeatures);
    const auto it = mObservations.find(KeyFrameWeakPtr(pKF));
    if (it == mObservations.end() ||
        cam_id < 0 || cam_id >= static_cast<int>(it->second.size())) {
        return;
    }
    it->second[cam_id] = idx;
}

bool MapPoint::HasObservation(const KeyFramePtr &pKF, int cam_id) {
    unique_lock<mutex> lock(mMutexFeatures);
    const auto it = mObservations.find(KeyFrameWeakPtr(pKF));
    if (it != mObservations.end()) {
        return it->second[cam_id] != -1;
    }
    return false;
}

void MapPoint::ComputeDistinctiveDescriptors() {
    // Retrieve all observed descriptors
    const MapPointPtr self = shared_from_this();
    vector<cv::Mat> vDescriptors;
    vector<tuple<KeyFramePtr, int, int>> observation_repairs;
    vector<pair<KeyFramePtr, int>> stale_observations;

    auto observations = GetObservations();
    if (isBad())
        return;

    if (observations.empty())
        return;

    vDescriptors.reserve(observations.size() * nCam);

    for (const auto &[pKF, indexes] : observations) {

        if (!pKF->isBad()) {
            for (int c = 0; c < nCam; c++) {
                if (c >= static_cast<int>(indexes.size()))
                    throw std::runtime_error(
                        ObservationDescriptorError(pKF, c, -1));
                int idx = indexes[c];
                if (idx != -1) {
                    int resolved_idx = -1;
                    cv::Mat descriptor;
                    const KeyFrame::PointObservationDescriptorStatus status =
                        pKF->GetPointObservationDescriptor(
                            self, c, idx, resolved_idx, descriptor);

                    if (status ==
                        KeyFrame::PointObservationDescriptorStatus::Found) {
                        if (resolved_idx != idx) {
                            observation_repairs.emplace_back(pKF, c,
                                                             resolved_idx);
                        }
                        vDescriptors.push_back(descriptor);
                    } else if (
                        status ==
                        KeyFrame::PointObservationDescriptorStatus::Missing) {
                        stale_observations.emplace_back(pKF, c);
                    } else {
                        throw std::runtime_error(
                            ObservationDescriptorError(pKF, c, idx));
                    }
                }
            }
        }
    }

    for (const auto &[pKF, cam, idx] : observation_repairs) {
        UpdateObservation(pKF, idx, cam);
    }

    for (const auto &[pKF, cam] : stale_observations) {
        RemoveObservationReference(pKF, cam);
    }

    if (isBad())
        return;

    if (vDescriptors.empty())
        return;

    const int BestIdx = SelectRepresentativeDescriptorIndex(vDescriptors);

    {
        unique_lock<mutex> lock(mMutexFeatures);
        mDescriptor = vDescriptors[BestIdx].clone();
    }
}

Map *MapPoint::GetMap() {
    unique_lock<mutex> lock(mMutexMap);
    return mpMap;
}

MapPointPtr MapPoint::GetReplaced() {
    unique_lock<mutex> lock1(mMutexFeatures, std::defer_lock);
    unique_lock<mutex> lock2(mMutexPos, std::defer_lock);
    lock(lock1, lock2);
    return mpReplaced.lock();
}

Eigen::Vector3f MapPoint::GetWorldPos() {
    unique_lock<mutex> lock(mMutexPos);
    return mWorldPos;
}

int MapPoint::Observations() {
    unique_lock<mutex> lock(mMutexFeatures);
    return nObs;
}

int MapPoint::RecentCullingStatus(long int currentKFid, int minObservations,
                                  float minFoundRatio) {
    unique_lock<mutex> lock(mMutexFeatures);
    if (mbBad)
        return 1;
    const long int age = currentKFid - mnFirstKFid;
    if (static_cast<float>(mnFound) / mnVisible < minFoundRatio ||
        (age >= 2 && nObs <= minObservations))
        return 2;
    return age >= 3 ? 1 : 0;
}

bool MapPoint::HasEnoughBetterObservations(const KeyFramePtr &pExcludedKF,
                                           float score, int minObservations) {
    vector<pair<KeyFramePtr, vector<int>>> observations;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if (mbBad || nObs <= minObservations)
            return false;
        observations.reserve(mObservations.size());
        for (const auto &observation : mObservations) {
            if (auto pKF = observation.first.lock())
                observations.emplace_back(pKF, observation.second);
        }
    }
    int nBetterObservations = 0;
    for (const auto &observation : observations) {
        const KeyFramePtr &pKF = observation.first;
        if (pKF == pExcludedKF || pKF->isBad())
            continue;
        float bestScore = 0.0f;
        const vector<int> &indexes = observation.second;
        for (int cam = 0; cam < static_cast<int>(indexes.size()); ++cam) {
            if (indexes[cam] != -1)
                bestScore =
                    max(bestScore, pKF->mvKeyScores_cams[cam][indexes[cam]]);
        }
        if (bestScore > score && ++nBetterObservations > minObservations)
            return true;
    }
    return false;
}

cv::Mat MapPoint::GetDescriptor() {
    unique_lock<mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}

bool MapPoint::GetDescriptorForMatching(cv::Mat &descriptor) {
    unique_lock<mutex> lock(mMutexFeatures);
    descriptor.release();

    if (mbBad)
        return false;

    if (mDescriptor.empty() || mDescriptor.rows != 1 ||
        mDescriptor.cols != kPointDescriptorCols ||
        mDescriptor.type() != CV_32F) {
        std::ostringstream oss;
        oss << "Trackable MapPoint " << mnId
            << " has an invalid descriptor";
        throw std::runtime_error(oss.str());
    }

    // cv::Mat keeps the descriptor storage alive through reference counting,
    // even if SetBadFlag() releases mDescriptor after this lock is dropped.
    // Matching is read-only, so avoid a heap allocation and 64-float copy for
    // every projected candidate.
    descriptor = mDescriptor;
    return true;
}

vector<int> MapPoint::GetIndexInKeyFrame(const KeyFramePtr &pKF) {
    unique_lock<mutex> lock(mMutexFeatures);
    const auto it = mObservations.find(KeyFrameWeakPtr(pKF));
    if (it != mObservations.end()) {
        return it->second;
    }
    return vector<int>(nCam, -1);
}

int MapPoint::GetIndexInKeyFrame(const KeyFramePtr &pKF, int cam_id) {
    unique_lock<mutex> lock(mMutexFeatures);
    const auto it = mObservations.find(KeyFrameWeakPtr(pKF));
    if (it != mObservations.end()) {
        return it->second[cam_id];
    }
    return -1;
}

std::map<KeyFramePtr, std::vector<int>>
MapPoint::GetObservations() {
    unique_lock<mutex> lock(mMutexFeatures);
    map<KeyFramePtr, vector<int>> observations;
    for (auto it = mObservations.begin(); it != mObservations.end();) {
        if (auto pKF = it->first.lock()) {
            observations.emplace(pKF, it->second);
            ++it;
        } else {
            it = mObservations.erase(it);
        }
    }
    return observations;
}

void MapPoint::GetObservations(
    vector<pair<KeyFramePtr, vector<int>>> &observations) {
    unique_lock<mutex> lock(mMutexFeatures);
    observations.clear();
    observations.reserve(mObservations.size());
    for (auto it = mObservations.begin(); it != mObservations.end();) {
        if (auto pKF = it->first.lock()) {
            observations.emplace_back(pKF, it->second);
            ++it;
        } else {
            it = mObservations.erase(it);
        }
    }
}

void MapPoint::IncreaseVisible(int n) {
    unique_lock<mutex> lock(mMutexFeatures);
    mnVisible += n;
}

void MapPoint::IncreaseFound(int n) {
    unique_lock<mutex> lock(mMutexFeatures);
    mnFound += n;
}

float MapPoint::GetFoundRatio() {
    unique_lock<mutex> lock(mMutexFeatures);
    return static_cast<float>(mnFound) / mnVisible;
}

void MapPoint::SetBadFlag() {
    const MapPointPtr keepAlive = shared_from_this();
    map<KeyFramePtr, vector<int>> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        if (mbBad)
            return;
        mbBad = true;
        for (const auto &[weakKF, indexes] : mObservations) {
            if (auto pKF = weakKF.lock())
                obs.emplace(pKF, indexes);
        }
        mObservations.clear();
        nObs = 0;
        mpRefKF.reset();
        mDescriptor.release();
    }
    for (const auto &[pKF, indexes] : obs) {
        for (int c = 0; c < nCam; c++) {
            int idx = indexes[c];
            if (idx != -1) {
                pKF->EraseMapPointMatch(keepAlive, c);
            }
        }
    }

    mpMap->EraseMapPoint(keepAlive);
}

void MapPoint::Replace(const MapPointPtr &pMP) {
    const MapPointPtr keepAlive = shared_from_this();
    if (pMP->mnId == this->mnId || mpReplaced.lock() == pMP)
        return;

    int nvisible, nfound;
    map<KeyFramePtr, vector<int>> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        for (const auto &[weakKF, indexes] : mObservations) {
            if (auto pKF = weakKF.lock())
                obs.emplace(pKF, indexes);
        }
        mObservations.clear();
        mbBad = true;
        nvisible = mnVisible;
        nfound = mnFound;
        mpReplaced = pMP;
    }

    for (const auto &[pKF, indexes] : obs) {
        // Replace measurement in keyframe
        for (int c = 0; c < nCam; c++) {
            int idx = indexes[c];

            if (idx == -1)
                continue;

            if (!pMP->HasObservation(pKF, c)) {
                int replacement_idx = -1;
                cv::Mat descriptor;
                const KeyFrame::PointObservationDescriptorStatus status =
                    pKF->GetPointObservationDescriptor(
                        keepAlive, c, idx, replacement_idx, descriptor);
                if (status ==
                    KeyFrame::PointObservationDescriptorStatus::Missing) {
                    continue;
                }
                if (status !=
                    KeyFrame::PointObservationDescriptorStatus::Found) {
                    throw std::runtime_error(
                        ObservationDescriptorError(pKF, c, idx));
                }

                pKF->ReplaceMapPointMatch(replacement_idx, pMP, c);
                pMP->AddObservation(pKF, replacement_idx, c);
            } else {
                pKF->EraseMapPointMatch(keepAlive, c);
            }
        }
    }
    pMP->IncreaseFound(nfound);
    pMP->IncreaseVisible(nvisible);
    pMP->ComputeDistinctiveDescriptors();

    mpMap->EraseMapPoint(keepAlive);
}

void MapPoint::EraseObservation(const KeyFramePtr &pKF, int cam) {
    bool bBad = false;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        const auto it = mObservations.find(KeyFrameWeakPtr(pKF));
        if (it != mObservations.end()) {
            vector<int> &indexes = it->second;
            if (indexes[cam] != -1) {
                nObs--;
                indexes[cam] = -1;
            }
            bool bAllBad = true;
            for (int c = 0; c < nCam; c++) {
                if (indexes[c] != -1) {
                    bAllBad = false;
                    break;
                }
            }
            if (bAllBad) {
                mObservations.erase(it);
                if (mpRefKF.lock() == pKF)
                    mpRefKF = mObservations.empty()
                                  ? KeyFrameWeakPtr()
                                  : mObservations.begin()->first;
            }

            if (nObs <= 2)
                bBad = true;
        }
    }

    if (bBad) {
        SetBadFlag();
    }
}

void MapPoint::EraseObservation(const KeyFramePtr &pKF) {
    bool bBad = false;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        const auto it = mObservations.find(KeyFrameWeakPtr(pKF));
        if (it != mObservations.end()) {
            vector<int> &indexes = it->second;
            for (int c = 0; c < nCam; c++) {
                if (indexes[c] != -1) {
                    nObs--;
                    indexes[c] = -1;
                }
            }
            mObservations.erase(it);
            if (mpRefKF.lock() == pKF)
                mpRefKF = mObservations.empty()
                              ? KeyFrameWeakPtr()
                              : mObservations.begin()->first;

            if (nObs <= 2)
                bBad = true;
        }
    }

    if (bBad) {
        SetBadFlag();
    }
}

void MapPoint::RemoveObservationReference(const KeyFramePtr &pKF, int cam_id) {
    bool bBad = false;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        const auto it = mObservations.find(KeyFrameWeakPtr(pKF));
        if (it == mObservations.end() || cam_id < 0 ||
            cam_id >= static_cast<int>(it->second.size())) {
            return;
        }

        vector<int> &indexes = it->second;
        if (indexes[cam_id] == -1) {
            return;
        }

        --nObs;
        indexes[cam_id] = -1;

        bool bAllBad = true;
        for (int c = 0; c < nCam; c++) {
            if (indexes[c] != -1) {
                bAllBad = false;
                break;
            }
        }
        if (bAllBad) {
            mObservations.erase(it);
            if (mpRefKF.lock() == pKF)
                mpRefKF = mObservations.empty()
                              ? KeyFrameWeakPtr()
                              : mObservations.begin()->first;
        }

        if (nObs <= 2)
            bBad = true;
    }

    if (bBad) {
        SetBadFlag();
    }
}

KeyFramePtr MapPoint::GetReferenceKeyFrame() {
    unique_lock<mutex> lock(mMutexFeatures);
    return mpRefKF.lock();
}

}; // namespace XPLSLAM
