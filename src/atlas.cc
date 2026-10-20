#include "atlas.h"

#include "geometric_camera.h"
#include "pinhole.h"

#include <mutex>

using namespace std;

namespace XPLSLAM {

Atlas::Atlas() { mpCurrentMap = static_cast<Map *>(NULL); }

Atlas::Atlas(int initKFid) : mnLastInitKFidMap(initKFid), mHasViewer(false) {
    mpCurrentMap = static_cast<Map *>(NULL);
    CreateNewMap();
}

Atlas::~Atlas() {
    for (std::set<Map *>::iterator it = mspMaps.begin(), end = mspMaps.end();
         it != end;) {
        Map *pMi = *it;

        if (pMi) {
            delete pMi;
            pMi = static_cast<Map *>(NULL);

            it = mspMaps.erase(it);
        } else
            ++it;
    }
    for (Map *pMap : mspBadMaps) {
        if (pMap)
            pMap->clear();
    }
    mspBadMaps.clear();
}

void Atlas::CreateNewMap() {
    unique_lock<mutex> lock(mMutexAtlas);
    cout << "Creation of new map with id: " << Map::nNextId << endl;
    if (mpCurrentMap) {
        if (!mspMaps.empty() && mnLastInitKFidMap < mpCurrentMap->GetMaxKFid())
            mnLastInitKFidMap = mpCurrentMap->GetMaxKFid() +
                                1; // The init KF is the next of current maximum

        mpCurrentMap->SetStoredMap();
        cout << "Stored map with ID: " << mpCurrentMap->GetId() << endl;
    }
    cout << "Creation of new map with last KF id: " << mnLastInitKFidMap
         << endl;

    mpCurrentMap = new Map(mnLastInitKFidMap);
    mpCurrentMap->SetCurrentMap();
    mspMaps.insert(mpCurrentMap);
}

void Atlas::SetKeyFrameDababase(KeyFrameDatabase *pKFDB) {
    mpKeyFrameDB = pKFDB;
}

void Atlas::SetVocabulary(DBoW3::Vocabulary *pvoc_point,
                          DBoW3::Vocabulary *pvoc_line) {
    mpVocabulary_point = pvoc_point;
    mpVocabulary_line = pvoc_line;
}

void Atlas::SetViewer(Viewer *pViewer) {
    unique_lock<mutex> lock(mMutexAtlas);
    mpViewer = pViewer;
    mHasViewer = pViewer != nullptr;
}

void Atlas::PostLoad() {
    map<unsigned int, GeometricCamera *> mpCams;
    for (GeometricCamera *pCam : mvpCameras) {
        mpCams[pCam->GetId()] = pCam;
    }

    mspMaps.clear();
    unsigned long int numKF = 0, numMP = 0, numML = 0;
    for (Map *pMi : mvpBackupMaps) {
        mspMaps.insert(pMi);
        pMi->PostLoad(mpKeyFrameDB, mpVocabulary_point, mpVocabulary_line,
                      mpCams);
        numKF += pMi->GetAllKeyFrames().size();
        numMP += pMi->GetAllMapPoints().size();
        numML += pMi->GetAllMapLines().size();
    }
    mvpBackupMaps.clear();
}

GeometricCamera *Atlas::AddCamera(GeometricCamera *pCam) {
    unique_lock<mutex> lock(mMutexAtlas);

    // Check if the camera already exists
    bool bAlreadyInMap = false;
    int index_cam = -1;
    for (size_t i = 0; i < mvpCameras.size(); ++i) {
        GeometricCamera *pCam_i = mvpCameras[i];
        if (!pCam)
            std::cout << "Not pCam" << std::endl;
        if (!pCam_i)
            std::cout << "Not pCam_i" << std::endl;
        if (pCam->GetType() != pCam_i->GetType())
            continue;

        if (pCam->GetType() == GeometricCamera::CAM_PINHOLE) {
            if (((Pinhole *)pCam_i)->IsEqual(*pCam)) {
                bAlreadyInMap = true;
                index_cam = i;
            }
        }
    }

    if (bAlreadyInMap) {
        return mvpCameras[index_cam];
    } else {
        mvpCameras.push_back(pCam);
        return pCam;
    }
}

void Atlas::clearAtlas() {
    unique_lock<mutex> lock(mMutexAtlas);
    for (Map *pMap : mspMaps) {
        if (pMap)
            pMap->clear();
    }
    for (Map *pMap : mspBadMaps) {
        if (pMap)
            pMap->clear();
    }
    mspMaps.clear();
    mspBadMaps.clear();
    mpCurrentMap = static_cast<Map *>(NULL);
    mnLastInitKFidMap = 0;
}

Map *Atlas::GetCurrentMap() {
    {
        unique_lock<mutex> lock(mMutexAtlas);
        if (mpCurrentMap && !mpCurrentMap->IsBad()) {
            return mpCurrentMap;
        }
    }

    if (!mpCurrentMap) {
        CreateNewMap();
    }

    while (true) {
        unique_lock<mutex> lock(mMutexAtlas);
        if (mpCurrentMap && !mpCurrentMap->IsBad()) {
            return mpCurrentMap;
        }
        lock.unlock();
        usleep(3000);
    }
}

void Atlas::clearMap() {
    unique_lock<mutex> lock(mMutexAtlas);
    mpCurrentMap->clear();
}

vector<Map *> Atlas::GetAllMaps() {
    unique_lock<mutex> lock(mMutexAtlas);
    struct compFunctor {
        inline bool operator()(Map *elem1, Map *elem2) {
            return elem1->GetId() < elem2->GetId();
        }
    };
    vector<Map *> vMaps(mspMaps.begin(), mspMaps.end());
    sort(vMaps.begin(), vMaps.end(), compFunctor());
    return vMaps;
}

void Atlas::AddKeyFrame(const KeyFramePtr &pKF) {
    Map *pMapKF = pKF->GetMap();
    pMapKF->AddKeyFrame(pKF);
}

void Atlas::AddMapPoint(const MapPointPtr &pMP) {
    Map *pMapMP = pMP->GetMap();
    pMapMP->AddMapPoint(pMP);
}

void Atlas::AddMapLine(const MapLinePtr &pML) {
    Map *pMapML = pML->GetMap();
    pMapML->AddMapLine(pML);
}

std::vector<MapPointPtr> Atlas::GetAllMapPoints() {
    unique_lock<mutex> lock(mMutexAtlas);
    return mpCurrentMap->GetAllMapPoints();
}

std::vector<KeyFramePtr> Atlas::GetAllKeyFrames() {
    unique_lock<mutex> lock(mMutexAtlas);
    return mpCurrentMap->GetAllKeyFrames();
}

std::vector<MapLinePtr> Atlas::GetAllMapLines() {
    unique_lock<mutex> lock(mMutexAtlas);
    return mpCurrentMap->GetAllMapLines();
}

std::vector<MapPointPtr> Atlas::GetReferenceMapPoints() {
    unique_lock<mutex> lock(mMutexAtlas);
    return mpCurrentMap->GetReferenceMapPoints();
}

std::vector<MapLinePtr> Atlas::GetReferenceMapLines() {
    unique_lock<mutex> lock(mMutexAtlas);
    return mpCurrentMap->GetReferenceMapLines();
}

void Atlas::SetReferenceMapPoints(const std::vector<MapPointPtr> &vpMPs) {
    unique_lock<mutex> lock(mMutexAtlas);
    mpCurrentMap->SetReferenceMapPoints(vpMPs);
}

void Atlas::SetReferenceMapLines(const std::vector<MapLinePtr> &vpMLs) {
    unique_lock<mutex> lock(mMutexAtlas);
    mpCurrentMap->SetReferenceMapLines(vpMLs);
}

long unsigned int Atlas::MapPointsInMap() {
    unique_lock<mutex> lock(mMutexAtlas);
    return mpCurrentMap ? mpCurrentMap->MapPointsInMap() : 0;
}

long unsigned Atlas::KeyFramesInMap() {
    unique_lock<mutex> lock(mMutexAtlas);
    return mpCurrentMap ? mpCurrentMap->KeyFramesInMap() : 0;
}

void Atlas::ChangeMap(Map *pMap) {
    unique_lock<mutex> lock(mMutexAtlas);
    cout << "Change to map with id: " << pMap->GetId() << endl;
    if (mpCurrentMap) {
        mpCurrentMap->SetStoredMap();
    }

    mpCurrentMap = pMap;
    mpCurrentMap->SetCurrentMap();
}

void Atlas::SetMapBad(Map *pMap) {
    mspMaps.erase(pMap);
    pMap->SetBad();

    mspBadMaps.insert(pMap);
}

void Atlas::RemoveBadMaps() {
    unique_lock<mutex> lock(mMutexAtlas);
    for (Map *pMap : mspBadMaps) {
        if (pMap)
            pMap->clear();
    }
    mspBadMaps.clear();
}

unsigned long int Atlas::GetLastInitKFid() {
    unique_lock<mutex> lock(mMutexAtlas);
    return mnLastInitKFidMap;
}

void Atlas::InformNewBigChange() {
    unique_lock<mutex> lock(mMutexAtlas);
    mpCurrentMap->InformNewBigChange();
}

int Atlas::CountMaps() {
    unique_lock<mutex> lock(mMutexAtlas);
    return mspMaps.size();
}

}; // namespace XPLSLAM
