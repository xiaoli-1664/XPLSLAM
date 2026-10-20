#pragma once

#include <list>
#include <set>
#include <vector>

#include "Vocabulary.h"
#include "frame.h"
#include "keyframe.h"
#include "map.h"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/library_version_type.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/vector.hpp>

#include <mutex>

namespace XPLSLAM {

class KeyFrame;
class Frame;
class Map;

class KeyFrameDatabase {
    // friend class boost::serialization::access;
    //
    // template <class Archive>
    // void serialize(Archive &ar, const unsigned int version) {
    //     ar & mvBackupInvertedFileId;
    // }

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    KeyFrameDatabase() {}
    KeyFrameDatabase(const DBoW3::Vocabulary &voc_point,
                     const DBoW3::Vocabulary &voc_line);

    void add(KeyFramePtr pKF);

    void erase(KeyFramePtr pKF);

    void clear();
    void clearMap(Map *pMap);

    // Loop Detection(DEPRECATED)
    std::vector<KeyFramePtr> DetectLoopCandidates(KeyFramePtr pKF, float minScore);

    // Loop and Merge Detection
    void DetectCandidates(KeyFramePtr pKF, float minScore,
                          std::vector<KeyFramePtr> &vpLoopCand,
                          std::vector<KeyFramePtr> &vpMergeCand);
    void DetectBestCandidates(KeyFramePtr pKF,
                              std::vector<KeyFramePtr> &vpLoopCand,
                              std::vector<KeyFramePtr> &vpMergeCand,
                              int nMinWords);
    void DetectNBestCandidates(KeyFramePtr pKF,
                               std::vector<KeyFramePtr> &vpLoopCand,
                               std::vector<KeyFramePtr> &vpMergeCand,
                               int nNumCandidates);

    // Relocalization
    std::vector<KeyFramePtr> DetectRelocalizationCandidates(
        Frame *F, Map *pMap, bool use_lines = true);

    void PreSave();
    void PostLoad(std::map<long unsigned int, KeyFramePtr> mpKFid);
    void SetVocabulary(const DBoW3::Vocabulary *pvoc_point,
                       const DBoW3::Vocabulary *pvoc_line);

  protected:
    // Associated vocabulary
    const DBoW3::Vocabulary *mpvocabulary_point;
    const DBoW3::Vocabulary *mpvocabulary_line;

    // Inverted file
    std::vector<std::list<KeyFrameWeakPtr>> mvInvertedFilePoint;
    std::vector<std::list<KeyFrameWeakPtr>> mvInvertedFileLine;

    // For save relation without pointer, this is necessary for save/load
    // function
    std::vector<std::list<long unsigned int>> mvBackupInvertedFilePointId;
    std::vector<std::list<long unsigned int>> mvBackupInvertedFileLineId;

    // Mutex
    std::mutex mMutex;
    long unsigned int mnRelocQueryCounter = 0;
};

} // namespace XPLSLAM
