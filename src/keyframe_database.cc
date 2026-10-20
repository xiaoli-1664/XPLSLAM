#include "runtime_timing.h"
#include "keyframe_database.h"
#include "BowVector.h"
#include "keyframe.h"

#include <mutex>

using namespace std;

namespace XPLSLAM {

namespace {

vector<KeyFramePtr> LockKeyFrames(list<KeyFrameWeakPtr> &entries) {
    vector<KeyFramePtr> keyframes;
    keyframes.reserve(entries.size());
    for (auto it = entries.begin(); it != entries.end();) {
        if (auto pKF = it->lock()) {
            keyframes.push_back(pKF);
            ++it;
        } else {
            it = entries.erase(it);
        }
    }
    return keyframes;
}

} // namespace

bool compFirst(const pair<float, KeyFramePtr> &a,
               const pair<float, KeyFramePtr> &b) {
    return a.first > b.first;
}

KeyFrameDatabase::KeyFrameDatabase(const DBoW3::Vocabulary &voc_point,
                                   const DBoW3::Vocabulary &voc_line)
    : mpvocabulary_point(&voc_point), mpvocabulary_line(&voc_line) {
    mvInvertedFilePoint.resize(voc_point.size());
    mvInvertedFileLine.resize(voc_line.size());
}

void KeyFrameDatabase::add(KeyFramePtr pKF) {
    unique_lock<mutex> lock(mMutex);

    for (DBoW3::BowVector::const_iterator vit = pKF->mBowVec_point.begin(),
                                          vend = pKF->mBowVec_point.end();
         vit != vend; vit++) {
        mvInvertedFilePoint[vit->first].push_back(pKF);
    }

    for (DBoW3::BowVector::const_iterator vit = pKF->mBowVec_line.begin(),
                                          vend = pKF->mBowVec_line.end();
         vit != vend; vit++) {
        mvInvertedFileLine[vit->first].push_back(pKF);
    }
}

void KeyFrameDatabase::clear() {
    mvInvertedFilePoint.clear();
    mvInvertedFileLine.clear();
    mvInvertedFilePoint.resize(mpvocabulary_point->size());
    mvInvertedFileLine.resize(mpvocabulary_line->size());
}

void KeyFrameDatabase::clearMap(Map *pMap) {
    unique_lock<mutex> lock(mMutex);

    // Erase elements in the Inverse File for the entry
    for (std::vector<list<KeyFrameWeakPtr>>::iterator
             vit = mvInvertedFilePoint.begin(),
             vend = mvInvertedFilePoint.end();
         vit != vend; vit++) {
        // List of keyframes that share the word
        list<KeyFrameWeakPtr> &lKFs = *vit;

        for (auto lit = lKFs.begin(); lit != lKFs.end();) {
            KeyFramePtr pKFi = lit->lock();
            if (!pKFi) {
                lit = lKFs.erase(lit);
                continue;
            }
            if (pMap == pKFi->GetMap()) {
                lit = lKFs.erase(lit);
                // Dont delete the KF because the class Map clean all the KF
                // when it is destroyed
            } else {
                ++lit;
            }
        }
    }

    for (std::vector<list<KeyFrameWeakPtr>>::iterator
             vit = mvInvertedFileLine.begin(),
             vend = mvInvertedFileLine.end();
         vit != vend; vit++) {
        // List of keyframes that share the word
        list<KeyFrameWeakPtr> &lKFs = *vit;

        for (auto lit = lKFs.begin(); lit != lKFs.end();) {
            KeyFramePtr pKFi = lit->lock();
            if (!pKFi) {
                lit = lKFs.erase(lit);
                continue;
            }
            if (pMap == pKFi->GetMap()) {
                lit = lKFs.erase(lit);
                // Dont delete the KF because the class Map clean all the KF
                // when it is destroyed
            } else {
                ++lit;
            }
        }
    }
}

void KeyFrameDatabase::erase(KeyFramePtr pKF) {
    unique_lock<mutex> lock(mMutex);

    // Erase elements in the Inverse File for the entry
    for (DBoW3::BowVector::const_iterator vit = pKF->mBowVec_point.begin(),
                                          vend = pKF->mBowVec_point.end();
         vit != vend; vit++) {
        // List of keyframes that share the word
        list<KeyFrameWeakPtr> &lKFs = mvInvertedFilePoint[vit->first];

        for (auto lit = lKFs.begin(); lit != lKFs.end();) {
            const KeyFramePtr candidate = lit->lock();
            if (!candidate || pKF == candidate)
                lit = lKFs.erase(lit);
            else
                ++lit;
        }
    }

    for (DBoW3::BowVector::const_iterator vit = pKF->mBowVec_line.begin(),
                                          vend = pKF->mBowVec_line.end();
         vit != vend; vit++) {
        // List of keyframes that share the word
        list<KeyFrameWeakPtr> &lKFs = mvInvertedFileLine[vit->first];

        for (auto lit = lKFs.begin(); lit != lKFs.end();) {
            const KeyFramePtr candidate = lit->lock();
            if (!candidate || pKF == candidate)
                lit = lKFs.erase(lit);
            else
                ++lit;
        }
    }
}

void KeyFrameDatabase::DetectNBestCandidates(KeyFramePtr pKF,
                                             vector<KeyFramePtr> &vpLoopCand,
                                             vector<KeyFramePtr> &vpMergeCand,
                                             int nNumCandidates) {
    list<KeyFramePtr> lKFsSharingWords;
    set<KeyFramePtr> spConnectedKF;

    // Search all keyframes that share a word with current frame
    {
        unique_lock<mutex> lock(mMutex);

        spConnectedKF = pKF->GetConnectedKeyFrames();

        for (DBoW3::BowVector::const_iterator vit = pKF->mBowVec_point.begin(),
                                              vend = pKF->mBowVec_point.end();
             vit != vend; vit++) {
            list<KeyFrameWeakPtr> &lKFs = mvInvertedFilePoint[vit->first];

            for (const KeyFramePtr &pKFi : LockKeyFrames(lKFs)) {

                if (pKFi->mnPlaceRecognitionQuery != pKF->mnId) {
                    pKFi->mnPlaceRecognitionWords = 0;
                    if (!spConnectedKF.count(pKFi)) {

                        pKFi->mnPlaceRecognitionQuery = pKF->mnId;
                        lKFsSharingWords.push_back(pKFi);
                    }
                }
                pKFi->mnPlaceRecognitionWords++;
            }
        }

        for (DBoW3::BowVector::const_iterator vit = pKF->mBowVec_line.begin(),
                                              vend = pKF->mBowVec_line.end();
             vit != vend; vit++) {
            list<KeyFrameWeakPtr> &lKFs = mvInvertedFileLine[vit->first];

            for (const KeyFramePtr &pKFi : LockKeyFrames(lKFs)) {

                if (pKFi->mnPlaceRecognitionQuery != pKF->mnId) {
                    pKFi->mnPlaceRecognitionWords = 0;
                    if (!spConnectedKF.count(pKFi)) {

                        pKFi->mnPlaceRecognitionQuery = pKF->mnId;
                        lKFsSharingWords.push_back(pKFi);
                    }
                }
                pKFi->mnPlaceRecognitionWords++;
            }
        }
    }
    if (lKFsSharingWords.empty())
        return;

    // Only compare against those keyframes that share enough words
    int maxCommonWords = 0;
    for (list<KeyFramePtr>::iterator lit = lKFsSharingWords.begin(),
                                    lend = lKFsSharingWords.end();
         lit != lend; lit++) {
        if ((*lit)->mnPlaceRecognitionWords > maxCommonWords)
            maxCommonWords = (*lit)->mnPlaceRecognitionWords;
    }

    int minCommonWords = maxCommonWords * 0.8f;

    list<pair<float, KeyFramePtr>> lScoreAndMatch;

    int nscores = 0;

    // Compute similarity score.
    for (list<KeyFramePtr>::iterator lit = lKFsSharingWords.begin(),
                                    lend = lKFsSharingWords.end();
         lit != lend; lit++) {
        KeyFramePtr pKFi = *lit;

        if (pKFi->mnPlaceRecognitionWords > minCommonWords) {
            nscores++;
            float si_point = mpvocabulary_point->score(pKF->mBowVec_point,
                                                       pKFi->mBowVec_point);
            float si_line =
                mpvocabulary_line->score(pKF->mBowVec_line, pKFi->mBowVec_line);
            float si = 0.5f * si_point + 0.5f * si_line;
            pKFi->mPlaceRecognitionScore = si;
            lScoreAndMatch.push_back(make_pair(si, pKFi));
        }
    }

    if (lScoreAndMatch.empty())
        return;

    list<pair<float, KeyFramePtr>> lAccScoreAndMatch;
    float bestAccScore = 0;

    // Lets now accumulate score by covisibility
    for (list<pair<float, KeyFramePtr>>::iterator it = lScoreAndMatch.begin(),
                                                 itend = lScoreAndMatch.end();
         it != itend; it++) {
        KeyFramePtr pKFi = it->second;
        vector<KeyFramePtr> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first;
        float accScore = bestScore;
        KeyFramePtr pBestKF = pKFi;
        for (vector<KeyFramePtr>::iterator vit = vpNeighs.begin(),
                                          vend = vpNeighs.end();
             vit != vend; vit++) {
            KeyFramePtr pKF2 = *vit;
            if (pKF2->mnPlaceRecognitionQuery != pKF->mnId)
                continue;

            accScore += pKF2->mPlaceRecognitionScore;
            if (pKF2->mPlaceRecognitionScore > bestScore) {
                pBestKF = pKF2;
                bestScore = pKF2->mPlaceRecognitionScore;
            }
        }
        lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
        if (accScore > bestAccScore)
            bestAccScore = accScore;
    }

    lAccScoreAndMatch.sort(compFirst);

    vpLoopCand.reserve(nNumCandidates);
    vpMergeCand.reserve(nNumCandidates);
    set<KeyFramePtr> spAlreadyAddedKF;
    int i = 0;
    list<pair<float, KeyFramePtr>>::iterator it = lAccScoreAndMatch.begin();
    while (i < lAccScoreAndMatch.size() &&
           (vpLoopCand.size() < nNumCandidates ||
            vpMergeCand.size() < nNumCandidates)) {
        KeyFramePtr pKFi = it->second;
        if (pKFi->isBad()) {
            i++;
            it++;
            continue;
        }

        if (!spAlreadyAddedKF.count(pKFi)) {
            if (pKF->GetMap() == pKFi->GetMap() &&
                vpLoopCand.size() < nNumCandidates) {
                vpLoopCand.push_back(pKFi);
            } else if (pKF->GetMap() != pKFi->GetMap() &&
                       vpMergeCand.size() < nNumCandidates &&
                       !pKFi->GetMap()->IsBad()) {
                vpMergeCand.push_back(pKFi);
            }
            spAlreadyAddedKF.insert(pKFi);
        }
        i++;
        it++;
    }
}

vector<KeyFramePtr> KeyFrameDatabase::DetectRelocalizationCandidates(
    Frame *F, Map *pMap, bool use_lines) {
    RuntimePhaseScope runtime_phase(RuntimePhase::Matching);
    list<KeyFramePtr> lKFsSharingWords;
    long unsigned int query_id = 0;

    {
        unique_lock<mutex> lock(mMutex);
        query_id = ++mnRelocQueryCounter;
        for (auto vit = F->mBowVec_point.begin(), vend = F->mBowVec_point.end();
             vit != vend; vit++) {
            list<KeyFrameWeakPtr> &lKFs = mvInvertedFilePoint[vit->first];

            for (const KeyFramePtr &pKFi : LockKeyFrames(lKFs)) {

                if (pKFi->mnRelocQuery != query_id) {
                    pKFi->mnRelocWords = 0;
                    pKFi->mnRelocQuery = query_id;
                    lKFsSharingWords.push_back(pKFi);
                }
                pKFi->mnRelocWords++;
            }
        }
        if (use_lines) {
            for (auto vit = F->mBowVec_line.begin(),
                      vend = F->mBowVec_line.end();
                 vit != vend; vit++) {
                list<KeyFrameWeakPtr> &lKFs = mvInvertedFileLine[vit->first];

                for (const KeyFramePtr &pKFi : LockKeyFrames(lKFs)) {
                    if (pKFi->mnRelocQuery != query_id) {
                        pKFi->mnRelocWords = 0;
                        pKFi->mnRelocQuery = query_id;
                        lKFsSharingWords.push_back(pKFi);
                    }
                    pKFi->mnRelocWords++;
                }
            }
        }
    }

    if (lKFsSharingWords.empty())
        return vector<KeyFramePtr>();

    // Only compare against those keyframes that share enough words
    int maxCommonWords = 0;
    for (list<KeyFramePtr>::iterator lit = lKFsSharingWords.begin(),
                                    lend = lKFsSharingWords.end();
         lit != lend; lit++) {
        if ((*lit)->mnRelocWords > maxCommonWords)
            maxCommonWords = (*lit)->mnRelocWords;
    }

    int minCommonWords = maxCommonWords * 0.8f;

    list<pair<float, KeyFramePtr>> lScoreAndMatch;

    // Compute similarity score.
    for (list<KeyFramePtr>::iterator lit = lKFsSharingWords.begin(),
                                    lend = lKFsSharingWords.end();
         lit != lend; lit++) {
        KeyFramePtr pKFi = *lit;

        if (pKFi->mnRelocWords > minCommonWords) {
            float si_point = mpvocabulary_point->score(F->mBowVec_point,
                                                       pKFi->mBowVec_point);
            if (use_lines) {
                const float si_line = mpvocabulary_line->score(
                    F->mBowVec_line, pKFi->mBowVec_line);
                pKFi->mRelocScore = 0.5f * si_point + 0.5f * si_line;
            } else {
                pKFi->mRelocScore = si_point;
            }
            lScoreAndMatch.push_back(make_pair(pKFi->mRelocScore, pKFi));
        }
    }

    if (lScoreAndMatch.empty())
        return vector<KeyFramePtr>();

    list<pair<float, KeyFramePtr>> lAccScoreAndMatch;
    float bestAccScore = 0;

    // Lets now accumulate score by covisibility
    for (list<pair<float, KeyFramePtr>>::iterator it = lScoreAndMatch.begin(),
                                                 itend = lScoreAndMatch.end();
         it != itend; it++) {
        KeyFramePtr pKFi = it->second;
        vector<KeyFramePtr> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first;
        float accScore = bestScore;
        KeyFramePtr pBestKF = pKFi;
        for (vector<KeyFramePtr>::iterator vit = vpNeighs.begin(),
                                          vend = vpNeighs.end();
             vit != vend; vit++) {
            KeyFramePtr pKF2 = *vit;
            if (pKF2->mnRelocQuery != query_id)
                continue;

            accScore += pKF2->mRelocScore;
            if (pKF2->mRelocScore > bestScore) {
                pBestKF = pKF2;
                bestScore = pKF2->mRelocScore;
            }
        }
        lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
        if (accScore > bestAccScore)
            bestAccScore = accScore;
    }

    float minScoreToRetain = 0.75f * bestAccScore;
    set<KeyFramePtr> spAlreadyAddedKF;
    vector<KeyFramePtr> vpRelocCandidates;
    vpRelocCandidates.reserve(lAccScoreAndMatch.size());
    for (list<pair<float, KeyFramePtr>>::iterator
             it = lAccScoreAndMatch.begin(),
             itend = lAccScoreAndMatch.end();
         it != itend; it++) {
        const float &si = it->first;
        if (si > minScoreToRetain) {
            KeyFramePtr pKFi = it->second;
            if (pKFi->GetMap() != pMap)
                continue;
            if (!spAlreadyAddedKF.count(pKFi)) {
                vpRelocCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpRelocCandidates;
}

}; // namespace XPLSLAM
