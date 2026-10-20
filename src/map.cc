#include "map.h"
#include "Vocabulary.h"

#include <mutex>
#include <shared_mutex>

using namespace std;

namespace XPLSLAM {

long unsigned int Map::nNextId = 0;

Map::Map()
    : mnMaxKFid(0), mnBigChangeIdx(0), mnMapChange(0),
      mpFirstRegionKF(nullptr), mbFail(false),
      mIsInUse(false), mHasTumbnail(false), mbBad(false),
      mnMapChangeNotified(0) {
    mnId = nNextId++;
}

Map::Map(int initKFid)
    : mnInitKFid(initKFid), mnMaxKFid(initKFid),
      /*mnLastLoopKFid(initKFid),*/ mnBigChangeIdx(0), mIsInUse(false),
      mHasTumbnail(false), mbBad(false),
      mpFirstRegionKF(nullptr), mnMapChange(0),
      mbFail(false), mnMapChangeNotified(0) {
    mnId = nNextId++;
}

Map::~Map() {
    // TODO: erase all the map points, lines and keyframes
    mspMapPoints.clear();
    mspMapLines.clear();

    mspKeyFrames.clear();

    mvpReferenceMapPoints.clear();
    mvpReferenceMapLines.clear();
    mvpKeyFrameOrigins.clear();
}

void Map::clear() {
    unique_lock<shared_mutex> lock(mMutexMap);

    mspMapPoints.clear();
    mspMapLines.clear();
    mspKeyFrames.clear();

    mvpReferenceMapPoints.clear();
    mvpReferenceMapLines.clear();
    mvpKeyFrameOrigins.clear();
    mvpBackupMapPoints.clear();
    mvpBackupMapLines.clear();
    mvpBackupKeyFrames.clear();
    mvBackupKeyFrameOriginsId.clear();

    mpKFinitial.reset();
    mpKFlowerID.reset();
    mpFirstRegionKF.reset();
    mnMaxKFid = mnInitKFid;
    mnBigChangeIdx = 0;
    mnMapChange++;
    mnMapChangeNotified = mnMapChange;
    mbFail = false;
}

long unsigned int Map::GetMaxKFid() {
    shared_lock<shared_mutex> lock(mMutexMap);
    return mnMaxKFid;
}

void Map::SetStoredMap() {
    unique_lock<shared_mutex> lock(mMutexMap);
    mIsInUse = false;
}

long unsigned int Map::GetId() { return mnId; }

void Map::SetCurrentMap() {
    unique_lock<shared_mutex> lock(mMutexMap);
    mIsInUse = true;
}

void Map::PostLoad(KeyFrameDatabase *pKFDB, DBoW3::Vocabulary *pVocPoint,
                   DBoW3::Vocabulary *pVocLine,
                   map<unsigned int, GeometricCamera *> &mpCams) {
    std::copy(mvpBackupMapPoints.begin(), mvpBackupMapPoints.end(),
              std::inserter(mspMapPoints, mspMapPoints.begin()));
    std::copy(mvpBackupMapLines.begin(), mvpBackupMapLines.end(),
              std::inserter(mspMapLines, mspMapLines.begin()));
    std::copy(mvpBackupKeyFrames.begin(), mvpBackupKeyFrames.end(),
              std::inserter(mspKeyFrames, mspKeyFrames.begin()));

    map<long unsigned int, MapPointPtr> mpMapPointId;
    for (const MapPointPtr &pMPi : mspMapPoints) {
        if (!pMPi || pMPi->isBad())
            continue;

        pMPi->UpdateMap(this);
        mpMapPointId[pMPi->mnId] = pMPi;
    }

    map<long unsigned int, MapLinePtr> mpMapLineId;
    for (const MapLinePtr &pMLi : mspMapLines) {
        if (!pMLi || pMLi->isBad())
            continue;

        pMLi->UpdateMap(this);
        mpMapLineId[pMLi->mnId] = pMLi;
    }

    map<long unsigned int, KeyFramePtr> mpKeyFrameId;
    for (const KeyFramePtr &pKFi : mspKeyFrames) {
        if (!pKFi || pKFi->isBad())
            continue;

        pKFi->UpdateMap(this);
        pKFi->SetPointVocabulary(pVocPoint);
        pKFi->SetLineVocabulary(pVocLine);
        pKFi->SetKeyFrameDatabase(pKFDB);
        mpKeyFrameId[pKFi->mnId] = pKFi;
    }

    // References reconstruction between different instances
    for (const MapPointPtr &pMPi : mspMapPoints) {
        if (!pMPi || pMPi->isBad())
            continue;

        pMPi->PostLoad(mpKeyFrameId, mpMapPointId);
    }

    for (const MapLinePtr &pMLi : mspMapLines) {
        if (!pMLi || pMLi->isBad())
            continue;

        pMLi->PostLoad(mpKeyFrameId, mpMapLineId);
    }

    for (const KeyFramePtr &pKFi : mspKeyFrames) {
        if (!pKFi || pKFi->isBad())
            continue;

        pKFi->PostLoad(mpKeyFrameId, mpMapPointId, mpMapLineId, mpCams);
        pKFDB->add(pKFi);
    }

    if (mnBackupKFinitialID != -1) {
        mpKFinitial = mpKeyFrameId[mnBackupKFinitialID];
    }

    if (mnBackupKFlowerID != -1) {
        mpKFlowerID = mpKeyFrameId[mnBackupKFlowerID];
    }

    mvpKeyFrameOrigins.clear();
    mvpKeyFrameOrigins.reserve(mvBackupKeyFrameOriginsId.size());
    for (int i = 0; i < mvBackupKeyFrameOriginsId.size(); ++i) {
        mvpKeyFrameOrigins.push_back(
            mpKeyFrameId[mvBackupKeyFrameOriginsId[i]]);
    }

    mvpBackupMapPoints.clear();
    mvpBackupMapLines.clear();
}

vector<KeyFramePtr> Map::GetAllKeyFrames() {
    shared_lock<shared_mutex> lock(mMutexMap);
    return vector<KeyFramePtr>(mspKeyFrames.begin(), mspKeyFrames.end());
}

vector<MapPointPtr> Map::GetAllMapPoints() {
    shared_lock<shared_mutex> lock(mMutexMap);
    return vector<MapPointPtr>(mspMapPoints.begin(), mspMapPoints.end());
}

vector<MapLinePtr> Map::GetAllMapLines() {
    shared_lock<shared_mutex> lock(mMutexMap);
    return vector<MapLinePtr>(mspMapLines.begin(), mspMapLines.end());
}

vector<MapPointPtr> Map::GetReferenceMapPoints() {
    shared_lock<shared_mutex> lock(mMutexMap);
    return mvpReferenceMapPoints;
}

vector<MapLinePtr> Map::GetReferenceMapLines() {
    shared_lock<shared_mutex> lock(mMutexMap);
    return mvpReferenceMapLines;
}

bool Map::IsBad() {
    shared_lock<shared_mutex> lock(mMutexMap);
    return mbBad;
}

unsigned int Map::GetLowerKFID() {
    shared_lock<shared_mutex> lock(mMutexMap);
    if (mpKFlowerID) {
        return mpKFlowerID->mnId;
    }
    return 0;
}

int Map::GetMapChangeIndex() {
    shared_lock<shared_mutex> lock(mMutexMap);
    return mnMapChange;
}

int Map::GetLastMapChange() {
    shared_lock<shared_mutex> lock(mMutexMap);
    return mnMapChangeNotified;
}

void Map::SetLastMapChange(int currentChangeId) {
    unique_lock<shared_mutex> lock(mMutexMap);
    mnMapChangeNotified = currentChangeId;
}

void Map::AddKeyFrame(const KeyFramePtr &pKF) {
    unique_lock<shared_mutex> lock(mMutexMap);
    if (mspKeyFrames.empty()) {
        cout << "First KF:" << pKF->mnId << "; Map init KF:" << mnInitKFid
             << endl;
        mnInitKFid = pKF->mnId;
        mpKFinitial = pKF;
        mpKFlowerID = pKF;
    }
    mspKeyFrames.insert(pKF);
    if (pKF->mnId > mnMaxKFid) {
        mnMaxKFid = pKF->mnId;
    }
    if (pKF->mnId < mpKFlowerID->mnId) {
        mpKFlowerID = pKF;
    }
}

void Map::AddMapPoint(const MapPointPtr &pMP) {
    unique_lock<shared_mutex> lock(mMutexMap);
    mspMapPoints.insert(pMP);
}

void Map::AddMapLine(const MapLinePtr &pML) {
    unique_lock<shared_mutex> lock(mMutexMap);
    mspMapLines.insert(pML);
}

void Map::SetReferenceMapPoints(const vector<MapPointPtr> &vpMPs) {
    unique_lock<shared_mutex> lock(mMutexMap);
    mvpReferenceMapPoints = vpMPs;
}

void Map::SetReferenceMapLines(const vector<MapLinePtr> &vpMLs) {
    unique_lock<shared_mutex> lock(mMutexMap);
    mvpReferenceMapLines = vpMLs;
}

long unsigned int Map::KeyFramesInMap() {
    shared_lock<shared_mutex> lock(mMutexMap);
    return mspKeyFrames.size();
}

long unsigned int Map::GetInitKFid() {
    shared_lock<shared_mutex> lock(mMutexMap);
    return mnInitKFid;
}

void Map::EraseMapPoint(const MapPointPtr &pMP) {
    unique_lock<shared_mutex> lock(mMutexMap);
    mspMapPoints.erase(pMP);
    mvpReferenceMapPoints.erase(
        remove(mvpReferenceMapPoints.begin(), mvpReferenceMapPoints.end(), pMP),
        mvpReferenceMapPoints.end());
    mvpBackupMapPoints.erase(
        remove(mvpBackupMapPoints.begin(), mvpBackupMapPoints.end(), pMP),
        mvpBackupMapPoints.end());
}

void Map::EraseMapLine(const MapLinePtr &pML) {
    unique_lock<shared_mutex> lock(mMutexMap);
    mspMapLines.erase(pML);
    mvpReferenceMapLines.erase(
        remove(mvpReferenceMapLines.begin(), mvpReferenceMapLines.end(), pML),
        mvpReferenceMapLines.end());
    mvpBackupMapLines.erase(
        remove(mvpBackupMapLines.begin(), mvpBackupMapLines.end(), pML),
        mvpBackupMapLines.end());
}

void Map::IncreaseChangeIndex() {
    unique_lock<shared_mutex> lock(mMutexMap);
    mnMapChange++;
}

void Map::EraseKeyFrame(const KeyFramePtr &pKF) {
    unique_lock<shared_mutex> lock(mMutexMap);
    mspKeyFrames.erase(pKF);
    mvpKeyFrameOrigins.erase(
        remove(mvpKeyFrameOrigins.begin(), mvpKeyFrameOrigins.end(), pKF),
        mvpKeyFrameOrigins.end());
    mvpBackupKeyFrames.erase(
        remove(mvpBackupKeyFrames.begin(), mvpBackupKeyFrames.end(), pKF),
        mvpBackupKeyFrames.end());
    if (mpKFinitial == pKF)
        mpKFinitial.reset();
    if (mpFirstRegionKF == pKF)
        mpFirstRegionKF.reset();
    if (mspKeyFrames.size() > 0) {
        if (!mpKFlowerID || pKF->mnId == mpKFlowerID->mnId) {
            vector<KeyFramePtr> vpKFs(mspKeyFrames.begin(),
                                      mspKeyFrames.end());
            sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);
            mpKFlowerID = vpKFs[0];
        }
    } else {
        mpKFlowerID.reset();
    }

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

void Map::ChangeId(long unsigned int nId) { mnId = nId; }

KeyFramePtr Map::GetOriginKF() { return mpKFinitial; }

void Map::InformNewBigChange() {
    unique_lock<shared_mutex> lock(mMutexMap);
    mnBigChangeIdx++;
}

void Map::SetBad() { mbBad = true; }

long unsigned int Map::MapPointsInMap() {
    shared_lock<shared_mutex> lock(mMutexMap);
    return mspMapPoints.size();
}

}; // namespace XPLSLAM
