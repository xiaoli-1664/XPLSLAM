#pragma once

#include "Vocabulary.h"
#include "keyframe.h"
#include "keyframe_database.h"
#include "map_line.h"
#include "map_point.h"
#include "slam_types.h"

#include <list>
#include <set>

#include <mutex>
#include <shared_mutex>

#include <boost/serialization/base_object.hpp>

namespace XPLSLAM {

class MapPoint;
class MapLine;
class KeyFrame;
class Atlas;
class KeyFrameDatabase;

class Map {
    friend class boost::serialization::access;

    template <class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & mnId;
        ar & mnInitKFid;
        ar & mnMaxKFid;
        ar & mnBigChangeIdx;

        // Save/load a set structure, the set structure is broken in
        // libboost 1.58 for ubuntu 16.04, a vector is serializated
        // ar & mspKeyFrames;
        // ar & mspMapPoints;
        ar & mvpBackupKeyFrames;
        ar & mvpBackupMapPoints;

        ar & mvBackupKeyFrameOriginsId;

        ar & mnBackupKFinitialID;
        ar & mnBackupKFlowerID;
    }

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Map();
    Map(int initKFid);
    ~Map();

    void AddKeyFrame(const KeyFramePtr &pKF);
    void AddMapPoint(const MapPointPtr &pMP);
    void AddMapLine(const MapLinePtr &pML);
    void EraseMapPoint(const MapPointPtr &pMP);
    void EraseMapLine(const MapLinePtr &pML);
    void EraseKeyFrame(const KeyFramePtr &pKF);
    void SetReferenceMapPoints(const std::vector<MapPointPtr> &vpMPs);
    void SetReferenceMapLines(const std::vector<MapLinePtr> &vpMLs);
    void InformNewBigChange();
    int GetLastBigChangeIdx();

    std::vector<KeyFramePtr> GetAllKeyFrames();
    std::vector<MapPointPtr> GetAllMapPoints();
    std::vector<MapLinePtr> GetAllMapLines();
    std::vector<MapPointPtr> GetReferenceMapPoints();
    std::vector<MapLinePtr> GetReferenceMapLines();

    long unsigned int MapPointsInMap();
    long unsigned int MapLinesInMap();
    long unsigned KeyFramesInMap();

    long unsigned int GetId();

    long unsigned int GetInitKFid();
    void SetInitKFid(long unsigned int initKFif);
    long unsigned int GetMaxKFid();

    KeyFramePtr GetOriginKF();

    void SetCurrentMap();
    void SetStoredMap();

    bool HasThumbnail();
    bool IsInUse();

    void SetBad();
    bool IsBad();

    void clear();

    int GetMapChangeIndex();
    void IncreaseChangeIndex();
    int GetLastMapChange();
    void SetLastMapChange(int currentChangeId);

    void ApplyScaledRotation(const Sophus::SE3f &T, const float s,
                             const bool bScaledVel = false);

    void PrintEssentialGraph();
    bool CheckEssentialGraph();
    void ChangeId(long unsigned int nId);

    unsigned int GetLowerKFID();

    void PreSave(std::set<GeometricCamera *> &spCams);
    void PostLoad(KeyFrameDatabase *pKFDB, DBoW3::Vocabulary *pVoc_point,
                  DBoW3::Vocabulary *pVoc_line,
                  std::map<unsigned int, GeometricCamera *> &mpCams);

    void printReprojectionError(std::list<KeyFramePtr> &lpLocalWindowKFs,
                                const KeyFramePtr &mpCurrentKF,
                                std::string &name,
                                std::string &name_folder);

    std::vector<KeyFramePtr> mvpKeyFrameOrigins;
    std::vector<unsigned long int> mvBackupKeyFrameOriginsId;
    KeyFramePtr mpFirstRegionKF;
    std::mutex mMutexMapUpdate;

    // This avoid that two points are created simultaneously in separate threads
    // (id conflict)
    std::mutex mMutexPointCreation;
    std::mutex mMutexLineCreation;

    bool mbFail;

    static long unsigned int nNextId;

    // DEBUG: show KFs which are used in LBA
    std::set<long unsigned int> msOptKFs;
    std::set<long unsigned int> msFixedKFs;

  protected:
    long unsigned int mnId;

    std::set<MapPointPtr> mspMapPoints;
    std::set<MapLinePtr> mspMapLines;
    std::set<KeyFramePtr> mspKeyFrames;

    // Save/load, the set structure is broken in libboost 1.58 for ubuntu 16.04,
    // a vector is serializated
    std::vector<MapPointPtr> mvpBackupMapPoints;
    std::vector<MapLinePtr> mvpBackupMapLines;
    std::vector<KeyFramePtr> mvpBackupKeyFrames;

    KeyFramePtr mpKFinitial;
    KeyFramePtr mpKFlowerID;

    unsigned long int mnBackupKFinitialID;
    unsigned long int mnBackupKFlowerID;

    std::vector<MapPointPtr> mvpReferenceMapPoints;
    std::vector<MapLinePtr> mvpReferenceMapLines;

    int mnMapChange;
    int mnMapChangeNotified;

    long unsigned int mnInitKFid;
    long unsigned int mnMaxKFid;
    // long unsigned int mnLastLoopKFid;

    // Index related to a big change in the map (loop closure, global BA)
    int mnBigChangeIdx;

    bool mIsInUse;
    bool mHasTumbnail;
    bool mbBad = false;

    // Mutex
    std::shared_mutex mMutexMap;
};

} // namespace XPLSLAM
