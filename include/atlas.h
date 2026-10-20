#pragma once

#include "Vocabulary.h"
#include "geometric_camera.h"
#include "keyframe.h"
#include "map.h"
#include "map_point.h"
#include "pinhole.h"
#include "slam_types.h"

#include <boost/serialization/export.hpp>
#include <boost/serialization/vector.hpp>
#include <mutex>
#include <set>

namespace XPLSLAM {
class Viewer;
class Map;
class MapPoint;
class MapLine;
class KeyFrame;
class KeyFrameDatabase;
class Frame;
class KannalaBrandt8;
class Pinhole;

// BOOST_CLASS_EXPORT_GUID(Pinhole, "Pinhole")
// BOOST_CLASS_EXPORT_GUID(KannalaBrandt8, "KannalaBrandt8")

class Atlas {
    friend class boost::serialization::access;

    // template <class Archive>
    // void serialize(Archive &ar, const unsigned int version) {
    //     ar.template register_type<Pinhole>();
    //     ar.template register_type<KannalaBrandt8>();
    //
    //     // Save/load a set structure, the set structure is broken in
    //     // libboost 1.58 for ubuntu 16.04, a vector is serializated
    //     // ar & mspMaps;
    //     ar & mvpBackupMaps;
    //     ar & mvpCameras;
    //     // Need to save/load the static Id from Frame, KeyFrame, MapPoint and
    //     // Map
    //     ar &Map::nNextId;
    //     ar &Frame::nNextId;
    //     ar &KeyFrame::nNextId;
    //     ar &MapPoint::nNextId;
    //     ar &GeometricCamera::nNextId;
    //     ar & mnLastInitKFidMap;
    // }

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Atlas();
    Atlas(int initKFid); // When its initialization the first map is created
    ~Atlas();

    void CreateNewMap();
    void ChangeMap(Map *pMap);

    unsigned long int GetLastInitKFid();

    void SetViewer(Viewer *pViewer);

    // Method for change components in the current map
    void AddKeyFrame(const KeyFramePtr &pKF);
    void AddMapPoint(const MapPointPtr &pMP);
    void AddMapLine(const MapLinePtr &pML);
    // void EraseMapPoint(MapPointPtr pMP);
    // void EraseKeyFrame(KeyFramePtr pKF);

    GeometricCamera *AddCamera(GeometricCamera *pCam);
    std::vector<GeometricCamera *> GetAllCameras();

    /* All methods without Map pointer work on current map */
    void SetReferenceMapPoints(const std::vector<MapPointPtr> &vpMPs);
    void SetReferenceMapLines(const std::vector<MapLinePtr> &vpMLs);
    void InformNewBigChange();
    int GetLastBigChangeIdx();

    long unsigned int MapPointsInMap();
    long unsigned int MapLinesInMap();
    long unsigned KeyFramesInMap();

    // Method for get data in current map
    std::vector<KeyFramePtr> GetAllKeyFrames();
    std::vector<MapPointPtr> GetAllMapPoints();
    std::vector<MapLinePtr> GetAllMapLines();
    std::vector<MapPointPtr> GetReferenceMapPoints();
    std::vector<MapLinePtr> GetReferenceMapLines();

    std::vector<Map *> GetAllMaps();

    int CountMaps();

    void clearMap();

    void clearAtlas();

    Map *GetCurrentMap();

    void SetMapBad(Map *pMap);
    void RemoveBadMaps();

    // Function for garantee the correction of serialization of this object
    void PreSave();
    void PostLoad();

    std::map<long unsigned int, KeyFramePtr> GetAtlasKeyframes();

    void SetKeyFrameDababase(KeyFrameDatabase *pKFDB);
    KeyFrameDatabase *GetKeyFrameDatabase();

    void SetVocabulary(DBoW3::Vocabulary *pvoc_point,
                       DBoW3::Vocabulary *pvoc_line);
    DBoW3::Vocabulary *GetVocabularyPoint();
    DBoW3::Vocabulary *GetVocabularyLine();

    long unsigned int GetNumLivedKF();

    long unsigned int GetNumLivedMP();
    long unsigned int GetNumLivedML();

  protected:
    std::set<Map *> mspMaps;
    std::set<Map *> mspBadMaps;
    // Its necessary change the container from set to vector because
    // libboost 1.58 and Ubuntu 16.04 have an error with this cointainer
    std::vector<Map *> mvpBackupMaps;

    Map *mpCurrentMap;

    std::vector<GeometricCamera *> mvpCameras;

    unsigned long int mnLastInitKFidMap;

    Viewer *mpViewer;
    bool mHasViewer;

    // Class references for the map reconstruction from the save file
    KeyFrameDatabase *mpKeyFrameDB;
    DBoW3::Vocabulary *mpVocabulary_point;
    DBoW3::Vocabulary *mpVocabulary_line;

    // Mutex
    std::mutex mMutexAtlas;

}; // class Atlas

} // namespace XPLSLAM
