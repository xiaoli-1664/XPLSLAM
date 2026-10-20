#include "map_line.h"

#include <algorithm>
#include <limits>
#include <mutex>

using namespace std;

namespace XPLSLAM {

namespace {

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

long unsigned int MapLine::nNextId = 0;
int MapLine::nCam = 0;
mutex MapLine::mGlobalMutex;

MapLine::MapLine()
    : mnFirstKFid(0), mnFirstFrame(0), nObs(0), mnTrackReferenceForFrame(0),
      mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopLineForKF(0),
      mnCorrectedByKF(0), mnCorrectedReference(0), mnBAGlobalForKF(0),
      mnVisible(1), mnFound(1), mbBad(false) {
    mvnLastFrameSeen.assign(std::max(nCam, 0), 0);
}

MapLine::MapLine(const PluckerLine &Pos, const KeyFramePtr &pRefKF, Map *pMap)
    : mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0),
      mnTrackReferenceForFrame(0), mnBALocalForKF(0), mnFuseCandidateForKF(0),
      mnLoopLineForKF(0), mnCorrectedByKF(0), mnCorrectedReference(0),
      mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1),
      mbBad(false), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap),
      mnOriginMapId(pMap->GetId()) {
    SetWorldPos(Pos);

    mvbTrackInView.resize(pRefKF->mvpCameras.size(), false);
    mvTrackProjD.resize(pRefKF->mvpCameras.size(), -1.0f);
    mvTrackProjTheta.resize(pRefKF->mvpCameras.size(), -1.0f);
    mvTrackDepth.resize(pRefKF->mvpCameras.size(), -1.0f);
    mvnLastFrameSeen.assign(pRefKF->mvpCameras.size(), 0);

    // MapLines can be created from Tracking and Local Mapping. This mutex
    // avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexLineCreation);
    mnId = nNextId++;
}

MapLine::MapLine(const PluckerLine &Pos, Map *pMap, Frame *pFrame,
                 const int &idxF, const int &cam_id)
    : mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0),
      mnTrackReferenceForFrame(0), mnBALocalForKF(0), mnFuseCandidateForKF(0),
      mnLoopLineForKF(0), mnCorrectedByKF(0), mnCorrectedReference(0),
      mnBAGlobalForKF(0), mnVisible(1), mnFound(1), mbBad(false), mpMap(pMap),
      mnOriginMapId(pMap->GetId()) {
    SetWorldPos(Pos);

    mvbTrackInView.resize(pFrame->mvpCamera.size(), false);
    mvTrackProjD.resize(pFrame->mvpCamera.size(), -1.0f);
    mvTrackProjTheta.resize(pFrame->mvpCamera.size(), -1.0f);
    mvTrackDepth.resize(pFrame->mvpCamera.size(), -1.0f);
    mvnLastFrameSeen.assign(pFrame->mvpCamera.size(), 0);

    pFrame->mvDescriptors_line[cam_id].row(idxF).copyTo(mDescriptor);

    // MapLines can be created from Tracking and Local Mapping. This mutex
    // avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexLineCreation);
    mnId = nNextId++;
}

void MapLine::SetWorldPos(const PluckerLine &Pos) {
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    mWorldPos = Pos;
}

bool MapLine::isBad() {
    unique_lock<mutex> lock1(mMutexFeatures, std::defer_lock);
    unique_lock<mutex> lock2(mMutexPos, std::defer_lock);
    lock(lock1, lock2);

    return mbBad;
}

void MapLine::UpdateMap(Map *pMap) {
    unique_lock<mutex> lock(mMutexMap);
    mpMap = pMap;
}

void MapLine::PostLoad(map<long unsigned int, KeyFramePtr> &mpKFid,
                       map<long unsigned int, MapLinePtr> &mpMLid) {
    mpRefKF = mpKFid[mBackupRefKFId];
    if (mpRefKF.expired()) {
        cout << "ERROR: MP without KF reference " << mBackupRefKFId
             << "; Num obs: " << nObs << endl;
    }
    mpReplaced.reset();
    if (mBackupReplacedId >= 0) {
        auto it = mpMLid.find(mBackupReplacedId);
        if (it != mpMLid.end())
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

void MapLine::AddObservation(const KeyFramePtr &pKF, int idx, int cam_id) {
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
}

bool MapLine::HasObservation(const KeyFramePtr &pKF, int cam_id) {
    unique_lock<mutex> lock(mMutexFeatures);
    const auto it = mObservations.find(KeyFrameWeakPtr(pKF));
    if (it != mObservations.end()) {
        return it->second[cam_id] != -1;
    }
    return false;
}

void MapLine::ComputeDistinctiveDescriptors() {
    // Retrieve all observed descriptors
    vector<cv::Mat> vDescriptors;

    auto observations = GetObservations();
    if (isBad())
        return;

    if (observations.empty())
        return;

    vDescriptors.reserve(observations.size() * nCam);

    for (const auto &[pKF, indexes] : observations) {
        if (!pKF->isBad()) {
            for (int c = 0; c < nCam; c++) {
                int idx = indexes[c];
                if (idx != -1) {
                    vDescriptors.push_back(pKF->mvDescriptors_line[c].row(idx));
                }
            }
        }
    }

    if (vDescriptors.empty())
        return;

    const int BestIdx = SelectRepresentativeDescriptorIndex(vDescriptors);

    {
        unique_lock<mutex> lock(mMutexFeatures);
        mDescriptor = vDescriptors[BestIdx].clone();
    }
}

Map *MapLine::GetMap() {
    unique_lock<mutex> lock(mMutexMap);
    return mpMap;
}

MapLinePtr MapLine::GetReplaced() {
    unique_lock<mutex> lock1(mMutexFeatures, std::defer_lock);
    unique_lock<mutex> lock2(mMutexPos, std::defer_lock);
    lock(lock1, lock2);
    return mpReplaced.lock();
}

PluckerLine MapLine::GetWorldPos() {
    unique_lock<mutex> lock(mMutexPos);
    return mWorldPos;
}

int MapLine::Observations() {
    unique_lock<mutex> lock(mMutexFeatures);
    return nObs;
}

int MapLine::RecentCullingStatus(long int currentKFid, int minObservations,
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

bool MapLine::HasEnoughBetterObservations(const KeyFramePtr &pExcludedKF,
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
                    max(bestScore, pKF->mvLineScores_cams[cam][indexes[cam]]);
        }
        if (bestScore > score && ++nBetterObservations > minObservations)
            return true;
    }
    return false;
}

std::map<KeyFramePtr, std::vector<int>>
MapLine::GetObservations() {
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

void MapLine::GetObservations(
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

void MapLine::IncreaseVisible(int n) {
    unique_lock<mutex> lock(mMutexFeatures);
    mnVisible += n;
}

void MapLine::IncreaseFound(int n) {
    unique_lock<mutex> lock(mMutexFeatures);
    mnFound += n;
}

cv::Mat MapLine::GetDescriptor() {
    unique_lock<mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}

bool MapLine::GetDescriptorForMatching(cv::Mat &descriptor) {
    unique_lock<mutex> lock(mMutexFeatures);
    descriptor.release();

    if (mbBad || mDescriptor.empty())
        return false;

    // Matching only reads the descriptor. The shallow cv::Mat copy keeps its
    // storage alive if this MapLine is replaced after the lock is released.
    descriptor = mDescriptor;
    return true;
}

vector<int> MapLine::GetIndexInKeyFrame(const KeyFramePtr &pKF) {
    unique_lock<mutex> lock(mMutexFeatures);
    const auto it = mObservations.find(KeyFrameWeakPtr(pKF));
    if (it != mObservations.end()) {
        return it->second;
    }
    return vector<int>(nCam, -1);
}

int MapLine::GetIndexInKeyFrame(const KeyFramePtr &pKF, int cam_id) {
    unique_lock<mutex> lock(mMutexFeatures);
    const auto it = mObservations.find(KeyFrameWeakPtr(pKF));
    if (it != mObservations.end()) {
        return it->second[cam_id];
    }
    return -1;
}

float MapLine::GetFoundRatio() {
    unique_lock<mutex> lock(mMutexFeatures);
    return static_cast<float>(mnFound) / static_cast<float>(mnVisible);
}

void MapLine::SetBadFlag() {
    const MapLinePtr keepAlive = shared_from_this();
    map<KeyFramePtr, vector<int>> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mbBad = true;
        for (const auto &[weakKF, indexes] : mObservations) {
            if (auto pKF = weakKF.lock())
                obs.emplace(pKF, indexes);
        }
        mObservations.clear();
    }
    for (const auto &[pKF, indexes] : obs) {
        for (int c = 0; c < nCam; c++) {
            int idx = indexes[c];
            if (idx != -1) {
                pKF->EraseMapLineMatch(idx, c);
            }
        }
    }

    mpMap->EraseMapLine(keepAlive);
}

void MapLine::Replace(const MapLinePtr &pML) {
    const MapLinePtr keepAlive = shared_from_this();
    if (pML->mnId == this->mnId || mpReplaced.lock() == pML)
        return;

    int nvisible, nfound;
    map<KeyFramePtr, vector<int>> obs;

    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mbBad = true;
        for (const auto &[weakKF, indexes] : mObservations) {
            if (auto pKF = weakKF.lock())
                obs.emplace(pKF, indexes);
        }
        mObservations.clear();
        nvisible = mnVisible;
        nfound = mnFound;
        mpReplaced = pML;
    }

    for (const auto &[pKF, indexes] : obs) {
        for (int c = 0; c < nCam; c++) {
            int idx = indexes[c];
            if (idx == -1)
                continue;
            if (!pML->HasObservation(pKF, c)) {
                pKF->ReplaceMapLineMatch(idx, pML, c);
                pML->AddObservation(pKF, idx, c);
            } else {
                pKF->EraseMapLineMatch(idx, c);
            }
        }
    }

    pML->IncreaseFound(nfound);
    pML->IncreaseVisible(nvisible);
    pML->ComputeDistinctiveDescriptors();

    mpMap->EraseMapLine(keepAlive);
}

void MapLine::EraseObservation(const KeyFramePtr &pKF, int cam) {
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

void MapLine::EraseObservation(const KeyFramePtr &pKF) {
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

KeyFramePtr MapLine::GetReferenceKeyFrame() {
    unique_lock<mutex> lock(mMutexFeatures);
    return mpRefKF.lock();
}

}; // namespace XPLSLAM
