#include "runtime_timing.h"
#include "matcher.h"

#include <Eigen/src/Core/Matrix.h>
#include <algorithm>
#include <limits>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

#include "FeatureVector.h"
#include "geometric_camera.h"
#include "geometric_tools.h"
#include "map.h"
#include "map_line.h"
#include "sophus/se3.hpp"

using namespace std;

namespace XPLSLAM {

const float Matcher::TH_L_LOW = 0.7f;
const float Matcher::TH_P_LOW = 0.7f;
const float Matcher::TH_L_HIGH = 0.8f;
const float Matcher::TH_P_HIGH = 0.8f;

namespace {

float CosineDistanceSquared(float score) {
    score = std::clamp(score, -1.0f, 1.0f);
    return std::max(0.0f, 2.0f - 2.0f * score);
}

bool PassCosineLoweRatio(float bestScore, float secondBestScore, float ratio) {
    if (ratio <= 0.0f) {
        return false;
    }

    if (secondBestScore == std::numeric_limits<float>::lowest()) {
        return true;
    }

    const float bestDist2 = CosineDistanceSquared(bestScore);
    const float secondDist2 = CosineDistanceSquared(secondBestScore);
    if (secondDist2 <= std::numeric_limits<float>::epsilon()) {
        return false;
    }

    // Lowe's ratio is defined on descriptor distance. For unit descriptors,
    // squared Euclidean distance is 2 - 2 * cosine_similarity, so the squared
    // comparison uses ratio^2 and keeps the configured threshold unchanged.
    const float distanceRatio = std::min(ratio, 1.0f);
    return bestDist2 < distanceRatio * distanceRatio * secondDist2;
}

using PointSnapshot = KeyFrame::PointFeatureSnapshot;

std::string PointSnapshotError(const char *context,
                               const PointSnapshot &snapshot, int cam = -1,
                               int idx = -1, int row = -1) {
    std::ostringstream oss;
    oss << "Inconsistent KeyFrame point feature snapshot in " << context;
    if (cam >= 0)
        oss << " camera " << cam;
    if (idx >= 0)
        oss << " index " << idx;
    if (row >= 0)
        oss << " row " << row;
    oss << " total_rows=" << snapshot.mDescriptors_point.rows
        << " mapping_size=" << snapshot.mvPointDescriptorRowToCamRow.size();
    return oss.str();
}

void ResolvePointRow(const PointSnapshot &snapshot, size_t row, int &cam,
                     int &cam_idx, const char *context) {
    if (row >= snapshot.mvPointDescriptorRowToCamRow.size()) {
        throw std::runtime_error(PointSnapshotError(context, snapshot, -1, -1,
                                                    static_cast<int>(row)));
    }
    tie(cam, cam_idx) = snapshot.mvPointDescriptorRowToCamRow[row];
    if (cam < 0 || cam >= static_cast<int>(snapshot.mvpMapPoints_cams.size()) ||
        cam >= static_cast<int>(snapshot.mvKeysUn_cams.size()) ||
        cam >= static_cast<int>(snapshot.mvDescriptors_point.size()) ||
        cam_idx < 0 ||
        cam_idx >= static_cast<int>(snapshot.mvpMapPoints_cams[cam].size()) ||
        cam_idx >= static_cast<int>(snapshot.mvKeysUn_cams[cam].size()) ||
        cam_idx >= snapshot.mvDescriptors_point[cam].rows) {
        throw std::runtime_error(PointSnapshotError(
            context, snapshot, cam, cam_idx, static_cast<int>(row)));
    }
}

cv::Mat PointFlatDescriptorRow(const PointSnapshot &snapshot, size_t row,
                               const char *context) {
    if (row >= static_cast<size_t>(snapshot.mDescriptors_point.rows)) {
        throw std::runtime_error(PointSnapshotError(context, snapshot, -1, -1,
                                                    static_cast<int>(row)));
    }
    return snapshot.mDescriptors_point.row(static_cast<int>(row));
}

cv::Mat PointCameraDescriptorRow(const PointSnapshot &snapshot, int cam,
                                 size_t idx, const char *context) {
    if (cam < 0 ||
        cam >= static_cast<int>(snapshot.mvDescriptors_point.size()) ||
        idx >= static_cast<size_t>(snapshot.mvDescriptors_point[cam].rows)) {
        throw std::runtime_error(
            PointSnapshotError(context, snapshot, cam, static_cast<int>(idx)));
    }
    return snapshot.mvDescriptors_point[cam].row(static_cast<int>(idx));
}

MapPointPtr PointMapPointAt(const PointSnapshot &snapshot, int cam, size_t idx,
                            const char *context) {
    if (cam < 0 || cam >= static_cast<int>(snapshot.mvpMapPoints_cams.size()) ||
        idx >= snapshot.mvpMapPoints_cams[cam].size()) {
        throw std::runtime_error(
            PointSnapshotError(context, snapshot, cam, static_cast<int>(idx)));
    }
    return snapshot.mvpMapPoints_cams[cam][idx];
}

const cv::Point2f &PointKeyAt(const PointSnapshot &snapshot, int cam,
                              size_t idx, const char *context) {
    if (cam < 0 || cam >= static_cast<int>(snapshot.mvKeysUn_cams.size()) ||
        idx >= snapshot.mvKeysUn_cams[cam].size()) {
        throw std::runtime_error(
            PointSnapshotError(context, snapshot, cam, static_cast<int>(idx)));
    }
    return snapshot.mvKeysUn_cams[cam][idx];
}

void EnsurePointMatchStorage(const PointSnapshot &snapshot,
                             vector<vector<MapPointPtr>> &vpMapPointMatches) {
    if (vpMapPointMatches.size() < static_cast<size_t>(snapshot.mnCam))
        vpMapPointMatches.resize(snapshot.mnCam);
    for (int cam = 0; cam < snapshot.mnCam; ++cam) {
        const size_t expected_size = cam < static_cast<int>(snapshot.Np.size())
                                         ? static_cast<size_t>(snapshot.Np[cam])
                                         : size_t{0};
        if (vpMapPointMatches[cam].size() < expected_size)
            vpMapPointMatches[cam].resize(expected_size, nullptr);
    }
}

} // namespace

Matcher::Matcher(float nnratio) : mfNNratio(nnratio) {}

int Matcher::SearchByBow(KeyFramePtr pKF, Frame &F,
                         vector<vector<MapPointPtr>> &vpMapPointMatches,
                         vector<vector<MapLinePtr>> &vpMapLineMatches) {
    RuntimePhaseScope runtime_phase(RuntimePhase::Matching);
    const PointSnapshot point_snapshot =
        pKF->GetPointFeatureSnapshot(false, true);
    const vector<vector<MapPointPtr>> &vpMapPointsKF =
        point_snapshot.mvpMapPoints_cams;
    const vector<vector<MapLinePtr>> &vpMapLinesKF = pKF->GetMapLineMatches();

    const int n_cam = point_snapshot.mnCam;
    vpMapPointMatches.resize(n_cam);
    vpMapLineMatches.resize(n_cam);

    for (int c = 0; c < n_cam; ++c) {
        vpMapPointMatches[c].assign(F.Np[c], nullptr);
        vpMapLineMatches[c].assign(F.Nl[c], nullptr);
    }

    const DBoW3::FeatureVector &vFeatVecPointKF = point_snapshot.mFeatVec_point;

    int nmatches = 0;
    unordered_set<MapPointPtr> spMatchedMapPoints;
    unordered_set<MapLinePtr> spMatchedMapLines;

    DBoW3::FeatureVector::const_iterator KFit = vFeatVecPointKF.begin();
    DBoW3::FeatureVector::const_iterator Fit = F.mFeatVec_point.begin();
    DBoW3::FeatureVector::const_iterator KFend = vFeatVecPointKF.end();
    DBoW3::FeatureVector::const_iterator Fend = F.mFeatVec_point.end();

    int n = 0;
    int n1 = 0;

    while (KFit != KFend && Fit != Fend) {
        if (KFit->first == Fit->first) {
            const vector<unsigned int> &vIndicesKF = KFit->second;
            const vector<unsigned int> &vIndicesF = Fit->second;

            for (size_t iKF = 0; iKF < vIndicesKF.size(); ++iKF) {
                const unsigned int idxKF = vIndicesKF[iKF];
                int camKF, idxKF_cam;
                ResolvePointRow(point_snapshot, idxKF, camKF, idxKF_cam,
                                "SearchByBow(KeyFrame,Frame)");

                MapPointPtr pMP =
                    PointMapPointAt(point_snapshot, camKF, idxKF_cam,
                                    "SearchByBow(KeyFrame,Frame)");

                if (!pMP)
                    continue;

                if (pMP->isBad())
                    continue;

                if (spMatchedMapPoints.count(pMP))
                    continue;

                ++n;

                const cv::Mat dKF = PointFlatDescriptorRow(
                    point_snapshot, idxKF, "SearchByBow(KeyFrame,Frame)");

                vector<float> bestScore(n_cam,
                                        std::numeric_limits<float>::lowest());
                vector<int> bestIdxCamF(n_cam, -1);
                vector<float> bestScore2(n_cam,
                                         std::numeric_limits<float>::lowest());

                for (size_t iF = 0; iF < vIndicesF.size(); ++iF) {
                    const unsigned int idxF = vIndicesF[iF];
                    int camF, idxF_cam;
                    tie(camF, idxF_cam) = F.mvPointDescriptorRowToCamRow[idxF];

                    if (vpMapPointMatches[camF][idxF_cam])
                        continue;

                    const cv::Mat &dF = F.mDescriptors_point.row(idxF);

                    const float score = dKF.dot(dF);

                    if (score > bestScore[camF]) {
                        bestScore2[camF] = bestScore[camF];
                        bestScore[camF] = score;
                        bestIdxCamF[camF] = idxF_cam;
                    } else if (score > bestScore2[camF]) {
                        bestScore2[camF] = score;
                    }
                }

                int bestMatchCam = -1;
                float bestMatchScore = std::numeric_limits<float>::lowest();
                for (int c = 0; c < n_cam; ++c) {
                    if (bestScore[c] >= TH_P_HIGH) {
                        n1++;
                        if (bestScore[c] > bestMatchScore &&
                            PassCosineLoweRatio(bestScore[c], bestScore2[c],
                                                mfNNratio)) {
                            bestMatchScore = bestScore[c];
                            bestMatchCam = c;
                        }
                    }
                }

                if (bestMatchCam >= 0) {
                    vpMapPointMatches[bestMatchCam][bestIdxCamF[bestMatchCam]] =
                        pMP;
                    spMatchedMapPoints.insert(pMP);
                    nmatches++;
                }
            }

            KFit++;
            Fit++;
        } else if (KFit->first < Fit->first) {
            KFit = vFeatVecPointKF.lower_bound(Fit->first);
        } else {
            Fit = F.mFeatVec_point.lower_bound(KFit->first);
        }
    }

    // cout << "SearchByBow: " << nmatches << " matches found for points. " << n
    //      << " candidates " << "passed high threshold: " << n1 << endl;

    const DBoW3::FeatureVector &vFeatVecLineKF = pKF->mFeatVec_line;

    KFit = vFeatVecLineKF.begin();
    Fit = F.mFeatVec_line.begin();
    KFend = vFeatVecLineKF.end();
    Fend = F.mFeatVec_line.end();

    while (KFit != KFend && Fit != Fend) {
        if (KFit->first == Fit->first) {
            const vector<unsigned int> &vIndicesKF = KFit->second;
            const vector<unsigned int> &vIndicesF = Fit->second;

            for (size_t iKF = 0; iKF < vIndicesKF.size(); ++iKF) {
                const unsigned int idxKF = vIndicesKF[iKF];
                int camKF, idxKF_cam;
                tie(camKF, idxKF_cam) = pKF->mvLineDescriptorRowToCamRow[idxKF];

                MapLinePtr pML = vpMapLinesKF[camKF][idxKF_cam];

                if (!pML)
                    continue;

                if (pML->isBad())
                    continue;

                if (spMatchedMapLines.count(pML))
                    continue;

                const cv::Mat &dKF = pKF->mDescriptors_line.row(idxKF);

                vector<float> bestScore(n_cam,
                                        std::numeric_limits<float>::lowest());
                vector<int> bestIdxCamF(n_cam, -1);
                vector<float> bestScore2(n_cam,
                                         std::numeric_limits<float>::lowest());

                for (size_t iF = 0; iF < vIndicesF.size(); ++iF) {
                    const unsigned int idxF = vIndicesF[iF];
                    int camF, idxF_cam;
                    tie(camF, idxF_cam) = F.mvLineDescriptorRowToCamRow[idxF];

                    if (vpMapLineMatches[camF][idxF_cam])
                        continue;

                    const cv::Mat &dF = F.mDescriptors_line.row(idxF);

                    const float score = dKF.dot(dF);

                    if (score > bestScore[camF]) {
                        bestScore2[camF] = bestScore[camF];
                        bestScore[camF] = score;
                        bestIdxCamF[camF] = idxF_cam;
                    } else if (score > bestScore2[camF]) {
                        bestScore2[camF] = score;
                    }
                }

                int bestMatchCam = -1;
                float bestMatchScore = std::numeric_limits<float>::lowest();
                for (int c = 0; c < n_cam; ++c) {
                    if (bestScore[c] >= TH_L_HIGH &&
                        bestScore[c] > bestMatchScore &&
                        PassCosineLoweRatio(bestScore[c], bestScore2[c],
                                            mfNNratio)) {
                        bestMatchScore = bestScore[c];
                        bestMatchCam = c;
                    }
                }

                if (bestMatchCam >= 0) {
                    vpMapLineMatches[bestMatchCam][bestIdxCamF[bestMatchCam]] =
                        pML;
                    spMatchedMapLines.insert(pML);
                    nmatches++;
                }
            }

            KFit++;
            Fit++;
        } else if (KFit->first < Fit->first) {
            KFit = vFeatVecLineKF.lower_bound(Fit->first);
        } else {
            Fit = F.mFeatVec_line.lower_bound(KFit->first);
        }
    }

    return nmatches;
}

int Matcher::SearchLinesByBow(KeyFramePtr pKF, Frame &F,
                              vector<vector<MapLinePtr>> &vpMapLineMatches) {
    RuntimePhaseScope runtime_phase(RuntimePhase::Matching);
    const int n_cam = pKF->mnCam;
    vpMapLineMatches.resize(n_cam);
    for (int cam = 0; cam < n_cam; ++cam) {
        vpMapLineMatches[cam].assign(F.Nl[cam], nullptr);
    }
    if (!F.HasLineFeatures()) {
        return 0;
    }

    const vector<vector<MapLinePtr>> vpMapLinesKF = pKF->GetMapLineMatches();
    const DBoW3::FeatureVector &kf_features = pKF->mFeatVec_line;
    auto kf_it = kf_features.begin();
    auto frame_it = F.mFeatVec_line.begin();
    const auto kf_end = kf_features.end();
    const auto frame_end = F.mFeatVec_line.end();
    unordered_set<MapLinePtr> matched_lines;
    int matches = 0;

    while (kf_it != kf_end && frame_it != frame_end) {
        if (kf_it->first == frame_it->first) {
            for (unsigned int flat_kf_index : kf_it->second) {
                int kf_cam = -1;
                int kf_index = -1;
                tie(kf_cam, kf_index) =
                    pKF->mvLineDescriptorRowToCamRow[flat_kf_index];
                MapLinePtr line = vpMapLinesKF[kf_cam][kf_index];
                if (!line || line->isBad() || matched_lines.count(line) != 0) {
                    continue;
                }

                const cv::Mat descriptor_kf =
                    pKF->mDescriptors_line.row(flat_kf_index);
                vector<float> best_score(n_cam,
                                         std::numeric_limits<float>::lowest());
                vector<float> second_score(
                    n_cam, std::numeric_limits<float>::lowest());
                vector<int> best_index(n_cam, -1);
                for (unsigned int flat_frame_index : frame_it->second) {
                    int frame_cam = -1;
                    int frame_index = -1;
                    tie(frame_cam, frame_index) =
                        F.mvLineDescriptorRowToCamRow[flat_frame_index];
                    if (vpMapLineMatches[frame_cam][frame_index]) {
                        continue;
                    }
                    const float score = descriptor_kf.dot(
                        F.mDescriptors_line.row(flat_frame_index));
                    if (score > best_score[frame_cam]) {
                        second_score[frame_cam] = best_score[frame_cam];
                        best_score[frame_cam] = score;
                        best_index[frame_cam] = frame_index;
                    } else if (score > second_score[frame_cam]) {
                        second_score[frame_cam] = score;
                    }
                }

                int best_cam = -1;
                float best = std::numeric_limits<float>::lowest();
                for (int cam = 0; cam < n_cam; ++cam) {
                    if (best_score[cam] >= TH_L_HIGH &&
                        best_score[cam] > best &&
                        PassCosineLoweRatio(best_score[cam], second_score[cam],
                                            mfNNratio)) {
                        best = best_score[cam];
                        best_cam = cam;
                    }
                }
                if (best_cam >= 0) {
                    vpMapLineMatches[best_cam][best_index[best_cam]] = line;
                    matched_lines.insert(line);
                    ++matches;
                }
            }
            ++kf_it;
            ++frame_it;
        } else if (kf_it->first < frame_it->first) {
            kf_it = kf_features.lower_bound(frame_it->first);
        } else {
            frame_it = F.mFeatVec_line.lower_bound(kf_it->first);
        }
    }
    return matches;
}

int Matcher::SearchByBow(const Frame &LastFrame, Frame &CurrentFrame,
                         vector<vector<MapPointPtr>> &vpMapPointMatches,
                         vector<vector<MapLinePtr>> &vpMapLineMatches) {
    RuntimePhaseScope runtime_phase(RuntimePhase::Matching);
    const int nCam = CurrentFrame.mnCam;
    vpMapPointMatches.resize(nCam);
    vpMapLineMatches.resize(nCam);
    for (int cam = 0; cam < nCam; ++cam) {
        vpMapPointMatches[cam].assign(CurrentFrame.Np[cam], nullptr);
        vpMapLineMatches[cam].assign(CurrentFrame.Nl[cam], nullptr);
    }

    int nmatches = 0;
    unordered_set<MapPointPtr> spMatchedMapPoints;
    unordered_set<MapLinePtr> spMatchedMapLines;

    auto lastPointIt = LastFrame.mFeatVec_point.begin();
    auto currentPointIt = CurrentFrame.mFeatVec_point.begin();
    const auto lastPointEnd = LastFrame.mFeatVec_point.end();
    const auto currentPointEnd = CurrentFrame.mFeatVec_point.end();

    while (lastPointIt != lastPointEnd && currentPointIt != currentPointEnd) {
        if (lastPointIt->first == currentPointIt->first) {
            for (const unsigned int lastIdx : lastPointIt->second) {
                int lastCam, lastCamIdx;
                tie(lastCam, lastCamIdx) =
                    LastFrame.mvPointDescriptorRowToCamRow[lastIdx];

                MapPointPtr pMP =
                    LastFrame.mvpMapPoints_cams[lastCam][lastCamIdx];
                if (!pMP || pMP->isBad() ||
                    LastFrame.mvbOutlierPoint_cams[lastCam][lastCamIdx]) {
                    continue;
                }

                if (spMatchedMapPoints.count(pMP))
                    continue;

                const cv::Mat &lastDescriptor =
                    LastFrame.mDescriptors_point.row(lastIdx);
                vector<float> bestScore(nCam,
                                        std::numeric_limits<float>::lowest());
                vector<float> secondBestScore(
                    nCam, std::numeric_limits<float>::lowest());
                vector<int> bestCurrentCamIdx(nCam, -1);

                for (const unsigned int currentIdx : currentPointIt->second) {
                    int currentCam, currentCamIdx;
                    tie(currentCam, currentCamIdx) =
                        CurrentFrame.mvPointDescriptorRowToCamRow[currentIdx];

                    if (vpMapPointMatches[currentCam][currentCamIdx]) {
                        continue;
                    }

                    const float score = lastDescriptor.dot(
                        CurrentFrame.mDescriptors_point.row(currentIdx));
                    if (score > bestScore[currentCam]) {
                        secondBestScore[currentCam] = bestScore[currentCam];
                        bestScore[currentCam] = score;
                        bestCurrentCamIdx[currentCam] = currentCamIdx;
                    } else if (score > secondBestScore[currentCam]) {
                        secondBestScore[currentCam] = score;
                    }
                }

                int bestMatchCam = -1;
                float bestMatchScore = std::numeric_limits<float>::lowest();
                for (int cam = 0; cam < nCam; ++cam) {
                    if (bestScore[cam] >= TH_P_HIGH &&
                        bestScore[cam] > bestMatchScore &&
                        PassCosineLoweRatio(bestScore[cam],
                                            secondBestScore[cam], mfNNratio)) {
                        bestMatchScore = bestScore[cam];
                        bestMatchCam = cam;
                    }
                }

                if (bestMatchCam >= 0) {
                    vpMapPointMatches[bestMatchCam]
                                     [bestCurrentCamIdx[bestMatchCam]] = pMP;
                    spMatchedMapPoints.insert(pMP);
                    nmatches++;
                }
            }

            ++lastPointIt;
            ++currentPointIt;
        } else if (lastPointIt->first < currentPointIt->first) {
            lastPointIt =
                LastFrame.mFeatVec_point.lower_bound(currentPointIt->first);
        } else {
            currentPointIt =
                CurrentFrame.mFeatVec_point.lower_bound(lastPointIt->first);
        }
    }

    auto lastLineIt = LastFrame.mFeatVec_line.begin();
    auto currentLineIt = CurrentFrame.mFeatVec_line.begin();
    const auto lastLineEnd = LastFrame.mFeatVec_line.end();
    const auto currentLineEnd = CurrentFrame.mFeatVec_line.end();

    while (lastLineIt != lastLineEnd && currentLineIt != currentLineEnd) {
        if (lastLineIt->first == currentLineIt->first) {
            for (const unsigned int lastIdx : lastLineIt->second) {
                int lastCam, lastCamIdx;
                tie(lastCam, lastCamIdx) =
                    LastFrame.mvLineDescriptorRowToCamRow[lastIdx];

                MapLinePtr pML =
                    LastFrame.mvpMapLines_cams[lastCam][lastCamIdx];
                if (!pML || pML->isBad() ||
                    LastFrame.mvbOutlierLine_cams[lastCam][lastCamIdx]) {
                    continue;
                }

                if (spMatchedMapLines.count(pML))
                    continue;

                const cv::Mat &lastDescriptor =
                    LastFrame.mDescriptors_line.row(lastIdx);
                vector<float> bestScore(nCam,
                                        std::numeric_limits<float>::lowest());
                vector<float> secondBestScore(
                    nCam, std::numeric_limits<float>::lowest());
                vector<int> bestCurrentCamIdx(nCam, -1);

                for (const unsigned int currentIdx : currentLineIt->second) {
                    int currentCam, currentCamIdx;
                    tie(currentCam, currentCamIdx) =
                        CurrentFrame.mvLineDescriptorRowToCamRow[currentIdx];

                    if (vpMapLineMatches[currentCam][currentCamIdx]) {
                        continue;
                    }

                    const float score = lastDescriptor.dot(
                        CurrentFrame.mDescriptors_line.row(currentIdx));
                    if (score > bestScore[currentCam]) {
                        secondBestScore[currentCam] = bestScore[currentCam];
                        bestScore[currentCam] = score;
                        bestCurrentCamIdx[currentCam] = currentCamIdx;
                    } else if (score > secondBestScore[currentCam]) {
                        secondBestScore[currentCam] = score;
                    }
                }

                int bestMatchCam = -1;
                float bestMatchScore = std::numeric_limits<float>::lowest();
                for (int cam = 0; cam < nCam; ++cam) {
                    if (bestScore[cam] >= TH_L_HIGH &&
                        bestScore[cam] > bestMatchScore &&
                        PassCosineLoweRatio(bestScore[cam],
                                            secondBestScore[cam], mfNNratio)) {
                        bestMatchScore = bestScore[cam];
                        bestMatchCam = cam;
                    }
                }

                if (bestMatchCam >= 0) {
                    vpMapLineMatches[bestMatchCam]
                                    [bestCurrentCamIdx[bestMatchCam]] = pML;
                    spMatchedMapLines.insert(pML);
                    nmatches++;
                }
            }

            ++lastLineIt;
            ++currentLineIt;
        } else if (lastLineIt->first < currentLineIt->first) {
            lastLineIt =
                LastFrame.mFeatVec_line.lower_bound(currentLineIt->first);
        } else {
            currentLineIt =
                CurrentFrame.mFeatVec_line.lower_bound(lastLineIt->first);
        }
    }

    return nmatches;
}

int Matcher::SearchByBow(
    KeyFramePtr pKF1, KeyFramePtr pKF2,
    std::vector<std::vector<MapPointPtr>> &vpMapPointMatches12,
    std::vector<std::vector<MapLinePtr>> &vpMapLineMatches12,
    float point_high_threshold, float line_high_threshold) {
    RuntimePhaseScope runtime_phase(RuntimePhase::Matching);
    const PointSnapshot point_snapshot1 =
        pKF1->GetPointFeatureSnapshot(false, true);
    const PointSnapshot point_snapshot2 =
        pKF2->GetPointFeatureSnapshot(false, true);
    int nCam = point_snapshot1.mnCam;
    const DBoW3::FeatureVector &vFeatVecPointKF1 =
        point_snapshot1.mFeatVec_point;
    const vector<vector<MapPointPtr>> &vpMapPointsKF1 =
        point_snapshot1.mvpMapPoints_cams;

    const DBoW3::FeatureVector &vFeatVecPointKF2 =
        point_snapshot2.mFeatVec_point;
    const vector<vector<MapPointPtr>> &vpMapPointsKF2 =
        point_snapshot2.mvpMapPoints_cams;

    const DBoW3::FeatureVector &vFeatVecLineKF1 = pKF1->mFeatVec_line;
    const vector<vector<MapLinePtr>> &vpMapLinesKF1 = pKF1->GetMapLineMatches();
    const vector<cv::Mat> &mvDescriptors_line1 = pKF1->mvDescriptors_line;

    const DBoW3::FeatureVector &vFeatVecLineKF2 = pKF2->mFeatVec_line;
    const vector<vector<MapLinePtr>> &vpMapLinesKF2 = pKF2->GetMapLineMatches();
    const vector<cv::Mat> &mvDescriptors_line2 = pKF2->mvDescriptors_line;

    vpMapPointMatches12.resize(nCam);
    vpMapLineMatches12.resize(nCam);
    vector<bool> vbMatchedPoint2(
        point_snapshot2.mvPointDescriptorRowToCamRow.size(), false);
    vector<bool> vbMatchedLine2(pKF2->mnTotalLines, false);
    for (int c = 0; c < nCam; ++c) {
        vpMapPointMatches12[c].assign(point_snapshot1.Np[c], nullptr);
        vpMapLineMatches12[c].assign(pKF1->Nl[c], nullptr);
    }

    int nmatches = 0;
    unordered_set<MapPointPtr> spMatchedMapPoints1;
    unordered_set<MapPointPtr> spMatchedMapPoints2;
    unordered_set<MapLinePtr> spMatchedMapLines1;
    unordered_set<MapLinePtr> spMatchedMapLines2;

    DBoW3::FeatureVector::const_iterator f1it = vFeatVecPointKF1.begin();
    DBoW3::FeatureVector::const_iterator f2it = vFeatVecPointKF2.begin();
    DBoW3::FeatureVector::const_iterator f1end = vFeatVecPointKF1.end();
    DBoW3::FeatureVector::const_iterator f2end = vFeatVecPointKF2.end();

    while (f1it != f1end && f2it != f2end) {
        if (f1it->first == f2it->first) {
            for (size_t i1 = 0; i1 < f1it->second.size(); ++i1) {
                const size_t idx1 = f1it->second[i1];

                int cam1, idx1_cam;
                ResolvePointRow(point_snapshot1, idx1, cam1, idx1_cam,
                                "SearchByBow(KeyFrame,KeyFrame)");

                MapPointPtr pMP1 =
                    PointMapPointAt(point_snapshot1, cam1, idx1_cam,
                                    "SearchByBow(KeyFrame,KeyFrame)");
                if (!pMP1 || pMP1->isBad())
                    continue;

                if (spMatchedMapPoints1.count(pMP1))
                    continue;

                const cv::Mat d1 =
                    PointCameraDescriptorRow(point_snapshot1, cam1, idx1_cam,
                                             "SearchByBow(KeyFrame,KeyFrame)");

                float bestScore = std::numeric_limits<float>::lowest();
                float bestScore2 = std::numeric_limits<float>::lowest();
                int bestIdx2 = -1;
                int bestIdxCam2 = -1;
                int bestCam2 = -1;

                for (size_t i2 = 0; i2 < f2it->second.size(); ++i2) {
                    const size_t idx2 = f2it->second[i2];

                    int cam2, idx2_cam;
                    ResolvePointRow(point_snapshot2, idx2, cam2, idx2_cam,
                                    "SearchByBow(KeyFrame,KeyFrame)");

                    MapPointPtr pMP2 =
                        PointMapPointAt(point_snapshot2, cam2, idx2_cam,
                                        "SearchByBow(KeyFrame,KeyFrame)");
                    if (!pMP2 || pMP2->isBad() || vbMatchedPoint2[idx2] ||
                        spMatchedMapPoints2.count(pMP2))
                        continue;

                    const cv::Mat d2 = PointCameraDescriptorRow(
                        point_snapshot2, cam2, idx2_cam,
                        "SearchByBow(KeyFrame,KeyFrame)");

                    float score = d1.dot(d2);

                    MapPointPtr pBestMP2 =
                        bestCam2 >= 0 ? vpMapPointsKF2[bestCam2][bestIdxCam2]
                                      : nullptr;

                    if (pMP2 == pBestMP2) {
                        if (score > bestScore) {
                            bestScore = score;
                            bestIdxCam2 = idx2_cam;
                            bestCam2 = cam2;
                            bestIdx2 = idx2;
                        }
                        continue;
                    }

                    if (score > bestScore) {
                        bestScore2 = bestScore;
                        bestScore = score;
                        bestIdxCam2 = idx2_cam;
                        bestCam2 = cam2;
                        bestIdx2 = idx2;
                    } else if (score > bestScore2) {
                        bestScore2 = score;
                    }
                }

                if (bestIdx2 >= 0 && bestScore >= point_high_threshold) {
                    if (PassCosineLoweRatio(bestScore, bestScore2, mfNNratio)) {
                        MapPointPtr pMP2 =
                            vpMapPointsKF2[bestCam2][bestIdxCam2];
                        vpMapPointMatches12[cam1][idx1_cam] = pMP2;
                        vbMatchedPoint2[bestIdx2] = true;
                        spMatchedMapPoints1.insert(pMP1);
                        spMatchedMapPoints2.insert(pMP2);
                        nmatches++;
                    }
                }
            }

            f1it++;
            f2it++;
        } else if (f1it->first < f2it->first) {
            f1it = vFeatVecPointKF1.lower_bound(f2it->first);
        } else {
            f2it = vFeatVecPointKF2.lower_bound(f1it->first);
        }
    }

    f1it = vFeatVecLineKF1.begin();
    f2it = vFeatVecLineKF2.begin();
    f1end = vFeatVecLineKF1.end();
    f2end = vFeatVecLineKF2.end();

    while (f1it != f1end && f2it != f2end) {
        if (f1it->first == f2it->first) {
            for (size_t i1 = 0; i1 < f1it->second.size(); ++i1) {
                const size_t idx1 = f1it->second[i1];

                int cam1, idx1_cam;
                tie(cam1, idx1_cam) = pKF1->mvLineDescriptorRowToCamRow[idx1];

                MapLinePtr pML1 = vpMapLinesKF1[cam1][idx1_cam];
                if (!pML1 || pML1->isBad())
                    continue;

                if (spMatchedMapLines1.count(pML1))
                    continue;

                const cv::Mat &d1 = mvDescriptors_line1[cam1].row(idx1_cam);

                float bestScore = std::numeric_limits<float>::lowest();
                float bestScore2 = std::numeric_limits<float>::lowest();
                int bestIdx2 = -1;
                int bestIdxCam2 = -1;
                int bestCam2 = -1;

                for (size_t i2 = 0; i2 < f2it->second.size(); ++i2) {
                    const size_t idx2 = f2it->second[i2];

                    int cam2, idx2_cam;
                    tie(cam2, idx2_cam) =
                        pKF2->mvLineDescriptorRowToCamRow[idx2];

                    MapLinePtr pML2 = vpMapLinesKF2[cam2][idx2_cam];
                    if (!pML2 || pML2->isBad() || vbMatchedLine2[idx2] ||
                        spMatchedMapLines2.count(pML2))
                        continue;

                    const cv::Mat &d2 = mvDescriptors_line2[cam2].row(idx2_cam);

                    const float score = d1.dot(d2);

                    MapLinePtr pBestML2 =
                        bestCam2 >= 0 ? vpMapLinesKF2[bestCam2][bestIdxCam2]
                                      : nullptr;

                    if (pML2 == pBestML2) {
                        if (score > bestScore) {
                            bestScore = score;
                            bestIdxCam2 = idx2_cam;
                            bestCam2 = cam2;
                            bestIdx2 = idx2;
                        }
                        continue;
                    }

                    if (score > bestScore) {
                        bestScore2 = bestScore;
                        bestScore = score;
                        bestIdxCam2 = idx2_cam;
                        bestCam2 = cam2;
                        bestIdx2 = idx2;
                    } else if (score > bestScore2) {
                        bestScore2 = score;
                    }
                }

                if (bestIdx2 >= 0 && bestScore >= line_high_threshold) {
                    if (PassCosineLoweRatio(bestScore, bestScore2, mfNNratio)) {
                        MapLinePtr pML2 = vpMapLinesKF2[bestCam2][bestIdxCam2];
                        vpMapLineMatches12[cam1][idx1_cam] = pML2;
                        vbMatchedLine2[bestIdx2] = true;
                        spMatchedMapLines1.insert(pML1);
                        spMatchedMapLines2.insert(pML2);
                        nmatches++;
                    }
                }
            }

            f1it++;
            f2it++;
        } else if (f1it->first < f2it->first) {
            f1it = vFeatVecLineKF1.lower_bound(f2it->first);
        } else {
            f2it = vFeatVecLineKF2.lower_bound(f1it->first);
        }
    }

    return nmatches;
}

int Matcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame,
                                const float th_point, const float th_line_d,
                                const float th_line_theta) {
    RuntimePhaseScope runtime_phase(RuntimePhase::Matching);
    int nmatches = 0;

    const Sophus::SE3f Tbw = CurrentFrame.GetPose();
    const Eigen::Vector3f twb = Tbw.inverse().translation();

    unordered_set<MapPointPtr> spProjectedMapPoints;
    unordered_set<MapLinePtr> spProjectedMapLines;

    for (int cl = 0; cl < LastFrame.mnCam; ++cl) {
        for (int i = 0; i < LastFrame.Np[cl]; ++i) {
            MapPointPtr pMP = LastFrame.mvpMapPoints_cams[cl][i];
            if (!pMP)
                continue;
            if (pMP->isBad())
                continue;
            if (LastFrame.mvbOutlierPoint_cams[cl][i])
                continue;
            if (spProjectedMapPoints.count(pMP))
                continue;
            spProjectedMapPoints.insert(pMP);

            Eigen::Vector3f x3Dw = pMP->GetWorldPos();

            for (int cc = 0; cc < CurrentFrame.mnCam; ++cc) {
                Eigen::Vector3f x3Dc =
                    CurrentFrame.GetCameraExtrinsic(cc).inverse() * Tbw * x3Dw;

                if (x3Dc.z() < 0)
                    continue;

                Eigen::Vector2f uv = CurrentFrame.mvpCamera[cc]->project(x3Dc);

                if (uv(0) < CurrentFrame.mvnMinX[cc] ||
                    uv(0) > CurrentFrame.mvnMaxX[cc] ||
                    uv(1) < CurrentFrame.mvnMinY[cc] ||
                    uv(1) > CurrentFrame.mvnMaxY[cc]) {
                    continue;
                }

                vector<size_t> vIndices2;

                vIndices2 = CurrentFrame.GetPointFeaturesInArea(uv(0), uv(1),
                                                                th_point, cc);

                if (vIndices2.empty())
                    continue;

                cv::Mat dMP;
                if (!pMP->GetDescriptorForMatching(dMP))
                    continue;

                float bestScore = std::numeric_limits<float>::lowest();
                int bestIdx2 = -1;

                for (vector<size_t>::const_iterator vit = vIndices2.begin(),
                                                    vend = vIndices2.end();
                     vit != vend; ++vit) {
                    const size_t i2 = *vit;

                    if (CurrentFrame.mvpMapPoints_cams[cc][i2]) {
                        if (CurrentFrame.mvpMapPoints_cams[cc][i2]
                                ->Observations() > 0) {
                            continue;
                        }
                    }

                    const cv::Mat &d =
                        CurrentFrame.mvDescriptors_point[cc].row(i2);

                    const float score = dMP.dot(d);

                    if (score > bestScore) {
                        bestScore = score;
                        bestIdx2 = i2;
                    }
                }

                if (bestScore >= TH_P_LOW) {
                    CurrentFrame.mvpMapPoints_cams[cc][bestIdx2] = pMP;
                    ++nmatches;
                }
            }
        }

        if (!CurrentFrame.HasLineFeatures()) {
            continue;
        }
        for (int i = 0; i < LastFrame.Nl[cl]; ++i) {
            MapLinePtr pML = LastFrame.mvpMapLines_cams[cl][i];
            if (!pML)
                continue;
            if (pML->isBad())
                continue;
            if (LastFrame.mvbOutlierLine_cams[cl][i])
                continue;
            if (spProjectedMapLines.count(pML))
                continue;
            spProjectedMapLines.insert(pML);

            Eigen::Matrix<float, 6, 1> l3Dw = pML->GetWorldPos();

            for (int cc = 0; cc < CurrentFrame.mnCam; ++cc) {
                Sophus::SE3f Tcb =
                    CurrentFrame.GetCameraExtrinsic(cc).inverse();
                Sophus::SE3f Tcw = Tcb * Tbw;
                Eigen::Matrix3f Rcw = Tcw.rotationMatrix();
                Eigen::Vector3f tcw = Tcw.translation();

                Eigen::Matrix<float, 6, 1> l3Dc;
                l3Dc.head<3>() =
                    Rcw * l3Dw.head<3>() + tcw.cross(Rcw * l3Dw.tail<3>());
                l3Dc.tail<3>() = Rcw * l3Dw.tail<3>();

                Eigen::Vector3f l2Dc =
                    CurrentFrame.mvpCamera[cc]->projectLine(l3Dc);

                if (!CurrentFrame.IsLineInImage(l2Dc, cc))
                    continue;

                float line_d = 0.0f;
                float line_theta = 0.0f;
                if (!GeometricTools::ImageLineToNormalizedPolar(l2Dc, line_d,
                                                                line_theta)) {
                    continue;
                }

                vector<size_t> vIndices2;

                vIndices2 = CurrentFrame.GetLineFeaturesInArea(
                    line_d, line_theta, th_line_d, th_line_theta, cc);

                if (vIndices2.empty())
                    continue;

                const cv::Mat dML = pML->GetDescriptor();

                float bestScore = std::numeric_limits<float>::lowest();
                int bestIdx2 = -1;

                for (vector<size_t>::const_iterator vit = vIndices2.begin(),
                                                    vend = vIndices2.end();
                     vit != vend; ++vit) {
                    const size_t i2 = *vit;

                    if (CurrentFrame.mvpMapLines_cams[cc][i2]) {
                        if (CurrentFrame.mvpMapLines_cams[cc][i2]
                                ->Observations() > 0) {
                            continue;
                        }
                    }

                    const cv::Mat &d =
                        CurrentFrame.mvDescriptors_line[cc].row(i2);

                    const float score = dML.dot(d);

                    if (score > bestScore) {
                        bestScore = score;
                        bestIdx2 = i2;
                    }
                }

                if (bestScore >= TH_L_LOW) {
                    CurrentFrame.mvpMapLines_cams[cc][bestIdx2] = pML;
                    nmatches++;
                }
            }
        }
    }

    return nmatches;
}

int Matcher::SearchByProjection(Frame &CurrentFrame, KeyFramePtr pKF,
                                const std::set<MapPointPtr> &sAlreadyFoundPoint,
                                const std::set<MapLinePtr> &sAlreadyFoundLine,
                                const float th_point, const float th_line_d,
                                const float th_line_theta) {
    RuntimePhaseScope runtime_phase(RuntimePhase::Matching);
    int nmatches = 0;

    const Sophus::SE3f Tbw = CurrentFrame.GetPose();
    const Eigen::Vector3f twb = Tbw.inverse().translation();

    const PointSnapshot point_snapshot =
        pKF->GetPointFeatureSnapshot(false, false);
    const vector<vector<MapPointPtr>> &vpMPs = point_snapshot.mvpMapPoints_cams;
    const vector<vector<MapLinePtr>> &vpMLs = pKF->GetMapLineMatches();

    set<MapPointPtr> sProjectedMapPoints;
    set<MapLinePtr> sProjectedMapLines;

    int nCam = point_snapshot.mnCam;

    for (int c = 0; c < nCam; ++c) {
        if (c >= static_cast<int>(vpMPs.size()))
            continue;
        for (int i = 0; i < vpMPs[c].size(); ++i) {
            MapPointPtr pMP = vpMPs[c][i];
            if (!pMP)
                continue;
            if (pMP->isBad() || sAlreadyFoundPoint.count(pMP) ||
                sProjectedMapPoints.count(pMP))
                continue;

            sProjectedMapPoints.insert(pMP);

            Eigen::Vector3f x3Dw = pMP->GetWorldPos();

            for (int cc = 0; cc < nCam; ++cc) {
                Eigen::Vector3f x3Dc =
                    CurrentFrame.GetCameraExtrinsic(cc).inverse() * Tbw * x3Dw;

                if (x3Dc.z() < 0)
                    continue;

                Eigen::Vector2f uv = CurrentFrame.mvpCamera[cc]->project(x3Dc);

                if (uv(0) < CurrentFrame.mvnMinX[cc] ||
                    uv(0) > CurrentFrame.mvnMaxX[cc] ||
                    uv(1) < CurrentFrame.mvnMinY[cc] ||
                    uv(1) > CurrentFrame.mvnMaxY[cc]) {
                    continue;
                }

                vector<size_t> vIndices2;

                vIndices2 = CurrentFrame.GetPointFeaturesInArea(uv(0), uv(1),
                                                                th_point, cc);

                if (vIndices2.empty())
                    continue;

                cv::Mat dMP;
                if (!pMP->GetDescriptorForMatching(dMP))
                    continue;

                float bestScore = std::numeric_limits<float>::lowest();
                int bestIdx2 = -1;

                for (vector<size_t>::const_iterator vit = vIndices2.begin(),
                                                    vend = vIndices2.end();
                     vit != vend; ++vit) {
                    const size_t i2 = *vit;

                    if (CurrentFrame.mvpMapPoints_cams[cc][i2]) {
                        continue;
                    }

                    const cv::Mat &d =
                        CurrentFrame.mvDescriptors_point[cc].row(i2);

                    const float score = dMP.dot(d);

                    if (score > bestScore) {
                        bestScore = score;
                        bestIdx2 = i2;
                    }
                }

                if (bestScore >= TH_P_LOW) {
                    CurrentFrame.mvpMapPoints_cams[cc][bestIdx2] = pMP;
                    nmatches++;
                }
            }
        }

        if (!CurrentFrame.HasLineFeatures()) {
            continue;
        }
        for (int i = 0; i < vpMLs[c].size(); ++i) {
            MapLinePtr pML = vpMLs[c][i];
            if (!pML)
                continue;
            if (pML->isBad() || sAlreadyFoundLine.count(pML) ||
                sProjectedMapLines.count(pML))
                continue;

            sProjectedMapLines.insert(pML);

            Eigen::Matrix<float, 6, 1> l3Dw = pML->GetWorldPos();

            for (int cc = 0; cc < CurrentFrame.mnCam; ++cc) {
                Sophus::SE3f Tcb =
                    CurrentFrame.GetCameraExtrinsic(cc).inverse();
                Sophus::SE3f Tcw = Tcb * Tbw;
                Eigen::Matrix3f Rcw = Tcw.rotationMatrix();
                Eigen::Vector3f tcw = Tcw.translation();

                Eigen::Matrix<float, 6, 1> l3Dc;
                l3Dc.head<3>() =
                    Rcw * l3Dw.head<3>() + tcw.cross(Rcw * l3Dw.tail<3>());
                l3Dc.tail<3>() = Rcw * l3Dw.tail<3>();

                Eigen::Vector3f l2Dc =
                    CurrentFrame.mvpCamera[cc]->projectLine(l3Dc);

                if (!CurrentFrame.IsLineInImage(l2Dc, cc))
                    continue;

                float line_d = 0.0f;
                float line_theta = 0.0f;
                if (!GeometricTools::ImageLineToNormalizedPolar(l2Dc, line_d,
                                                                line_theta)) {
                    continue;
                }

                vector<size_t> vIndices2;

                vIndices2 = CurrentFrame.GetLineFeaturesInArea(
                    line_d, line_theta, th_line_d, th_line_theta, cc);

                if (vIndices2.empty())
                    continue;

                const cv::Mat dML = pML->GetDescriptor();

                float bestScore = std::numeric_limits<float>::lowest();
                int bestIdx2 = -1;

                for (vector<size_t>::const_iterator vit = vIndices2.begin(),
                                                    vend = vIndices2.end();
                     vit != vend; ++vit) {
                    const size_t i2 = *vit;

                    if (CurrentFrame.mvpMapLines_cams[cc][i2]) {
                        continue;
                    }

                    const cv::Mat &d =
                        CurrentFrame.mvDescriptors_line[cc].row(i2);

                    const float score = dML.dot(d);

                    if (score > bestScore) {
                        bestScore = score;
                        bestIdx2 = i2;
                    }
                }

                if (bestScore >= TH_L_LOW) {
                    CurrentFrame.mvpMapLines_cams[cc][bestIdx2] = pML;
                    nmatches++;
                }
            }
        }
    }

    return nmatches;
}

int Matcher::SearchByProjection(Frame &F,
                                const vector<MapPointPtr> &vpMapPoints,
                                const vector<MapLinePtr> &vpMapLines,
                                const float th_point, const float th_line,
                                const bool bFarPoints, const float thFarPoints,
                                const bool bFarLines, const float thFarLines) {
    RuntimePhaseScope runtime_phase(RuntimePhase::Matching);
    int nmatches = 0;

    for (size_t iMP = 0; iMP < vpMapPoints.size(); ++iMP) {
        MapPointPtr pMP = vpMapPoints[iMP];

        if (pMP->isBad())
            continue;

        for (int c = 0; c < F.mnCam; ++c) {
            if (!pMP->mvbTrackInView[c])
                continue;

            if (bFarPoints && pMP->mvTrackDepth[c] > thFarPoints)
                continue;

            const float r = 2.5f * th_point;

            const vector<size_t> vIndices = F.GetPointFeaturesInArea(
                pMP->mvTrackProjX[c], pMP->mvTrackProjY[c], r, c);

            if (!vIndices.empty()) {
                cv::Mat MPdescriptor;
                if (!pMP->GetDescriptorForMatching(MPdescriptor))
                    continue;

                float bestScore = std::numeric_limits<float>::lowest();
                int bestIdx = -1;
                float bestScore2 = std::numeric_limits<float>::lowest();

                for (vector<size_t>::const_iterator vit = vIndices.begin(),
                                                    vend = vIndices.end();
                     vit != vend; ++vit) {
                    const size_t idx = *vit;

                    if (F.mvpMapPoints_cams[c][idx]) {
                        if (F.mvpMapPoints_cams[c][idx]->Observations() > 0) {
                            continue;
                        }
                    }

                    const cv::Mat &d = F.mvDescriptors_point[c].row(idx);

                    const float score = MPdescriptor.dot(d);

                    if (score > bestScore) {
                        bestScore2 = bestScore;
                        bestScore = score;
                        bestIdx = idx;
                    } else if (score > bestScore2) {
                        bestScore2 = score;
                    }
                }

                if (bestIdx >= 0 && bestScore >= TH_P_LOW) {
                    if (PassCosineLoweRatio(bestScore, bestScore2, mfNNratio)) {
                        F.mvpMapPoints_cams[c][bestIdx] = pMP;
                        nmatches++;
                    }
                }
            }
        }
    }

    if (!F.HasLineFeatures()) {
        return nmatches;
    }

    for (size_t iML = 0; iML < vpMapLines.size(); ++iML) {
        MapLinePtr pML = vpMapLines[iML];

        if (pML->isBad())
            continue;

        for (int c = 0; c < F.mnCam; ++c) {
            if (!pML->mvbTrackInView[c])
                continue;

            if (bFarLines && pML->mvTrackDepth[c] > thFarLines)
                continue;

            float r_d = th_line * 8;
            float r_theta = th_line * 0.17f;

            const vector<size_t> vIndices = F.GetLineFeaturesInArea(
                pML->mvTrackProjD[c], pML->mvTrackProjTheta[c], r_d, r_theta,
                c);

            if (!vIndices.empty()) {
                const cv::Mat MLdescriptor = pML->GetDescriptor();

                float bestScore = std::numeric_limits<float>::lowest();
                int bestIdx = -1;
                float bestScore2 = std::numeric_limits<float>::lowest();

                for (vector<size_t>::const_iterator vit = vIndices.begin(),
                                                    vend = vIndices.end();
                     vit != vend; ++vit) {
                    const size_t idx = *vit;

                    if (F.mvpMapLines_cams[c][idx]) {
                        if (F.mvpMapLines_cams[c][idx]->Observations() > 0) {
                            continue;
                        }
                    }

                    const cv::Mat &d = F.mvDescriptors_line[c].row(idx);

                    const float score = MLdescriptor.dot(d);

                    if (score > bestScore) {
                        bestScore2 = bestScore;
                        bestScore = score;
                        bestIdx = idx;
                    } else if (score > bestScore2) {
                        bestScore2 = score;
                    }
                }

                if (bestIdx >= 0 && bestScore >= TH_L_LOW) {
                    if (PassCosineLoweRatio(bestScore, bestScore2, mfNNratio)) {
                        F.mvpMapLines_cams[c][bestIdx] = pML;
                        nmatches++;
                    }
                }
            }
        }
    }

    return nmatches;
}

// TODO: thshold ratio
int Matcher::SearchByProjection(KeyFramePtr pKF, Sophus::SE3f &Tcw,
                                const vector<MapPointPtr> &vpPoints,
                                const vector<MapLinePtr> &vpLines,
                                vector<vector<MapPointPtr>> &vpMapPointMatches,
                                vector<vector<MapLinePtr>> &vpMapLineMatches,
                                const float th_point, const float th_line_d,
                                const float th_line_theta, float ratio_factor,
                                float point_high_threshold,
                                float line_high_threshold) {
    RuntimePhaseScope runtime_phase(RuntimePhase::Matching);
    const PointSnapshot point_snapshot =
        pKF->GetPointFeatureSnapshot(true, false, false);
    const int nCam = point_snapshot.mnCam;
    EnsurePointMatchStorage(point_snapshot, vpMapPointMatches);

    const vector<Sophus::SE3f> vTbc = pKF->GetCameraExtrinsics();
    vector<Sophus::SE3f> vTccw(nCam);
    vector<Eigen::Matrix3f> vRccw(nCam);
    vector<Eigen::Vector3f> vtccw(nCam);
    const Sophus::SE3f Tbw = Tcw;
    for (int c = 0; c < nCam; ++c) {
        vTccw[c] = vTbc[c].inverse() * Tbw;
        vRccw[c] = vTccw[c].rotationMatrix();
        vtccw[c] = vTccw[c].translation();
    }

    unordered_set<MapPointPtr> spAlreadyFoundPoints;
    unordered_set<MapLinePtr> spAlreadyFoundLines;

    for (int c = 0; c < nCam; ++c) {
        spAlreadyFoundPoints.insert(vpMapPointMatches[c].begin(),
                                    vpMapPointMatches[c].end());
        spAlreadyFoundLines.insert(vpMapLineMatches[c].begin(),
                                   vpMapLineMatches[c].end());
    }
    spAlreadyFoundPoints.erase(nullptr);
    spAlreadyFoundLines.erase(nullptr);

    int nmatches = 0;

    for (MapPointPtr pMP : vpPoints) {
        if (!pMP || pMP->isBad() || spAlreadyFoundPoints.count(pMP))
            continue;

        const Eigen::Vector3f x3Dw = pMP->GetWorldPos();
        cv::Mat dMP;
        if (!pMP->GetDescriptorForMatching(dMP))
            continue;

        for (int c = 0; c < nCam; ++c) {
            Eigen::Vector3f x3Dc = vTccw[c] * x3Dw;
            if (x3Dc.z() <= 0.0f)
                continue;

            const Eigen::Vector2f uv = pKF->mvpCameras[c]->project(x3Dc);
            if (!point_snapshot.IsPointInImage(uv.x(), uv.y(), c))
                continue;

            const vector<size_t> vIndices =
                point_snapshot.GetPointFeaturesInArea(uv.x(), uv.y(), th_point,
                                                      c);
            if (vIndices.empty())
                continue;

            float bestScore = std::numeric_limits<float>::lowest();
            int bestIdx = -1;

            for (size_t idx : vIndices) {
                if (vpMapPointMatches[c][idx])
                    continue;

                const cv::Mat d = PointCameraDescriptorRow(
                    point_snapshot, c, idx, "SearchByProjection(KeyFrame)");
                const float score = dMP.dot(d);
                if (score > bestScore) {
                    bestScore = score;
                    bestIdx = static_cast<int>(idx);
                }
            }

            if (bestIdx >= 0 &&
                bestScore >= point_high_threshold * ratio_factor) {
                vpMapPointMatches[c][bestIdx] = pMP;
                ++nmatches;
            }
        }
    }

    for (MapLinePtr pML : vpLines) {
        if (!pML || pML->isBad() || spAlreadyFoundLines.count(pML))
            continue;

        const Eigen::Matrix<float, 6, 1> l3Dw = pML->GetWorldPos();
        const cv::Mat dML = pML->GetDescriptor();

        for (int c = 0; c < nCam; ++c) {
            Eigen::Matrix<float, 6, 1> l3Dc;
            l3Dc.tail<3>() = vRccw[c] * l3Dw.tail<3>();
            l3Dc.head<3>() =
                vRccw[c] * l3Dw.head<3>() + vtccw[c].cross(l3Dc.tail<3>());

            const Eigen::Vector3f l2Dc = pKF->mvpCameras[c]->projectLine(l3Dc);
            if (!pKF->IsLineInImage(l2Dc, c))
                continue;

            float line_d = 0.0f;
            float line_theta = 0.0f;
            if (!GeometricTools::ImageLineToNormalizedPolar(l2Dc, line_d,
                                                            line_theta)) {
                continue;
            }

            const vector<size_t> vIndices = pKF->GetLineFeaturesInArea(
                line_d, line_theta, th_line_d, th_line_theta, c);
            if (vIndices.empty())
                continue;

            float bestScore = std::numeric_limits<float>::lowest();
            int bestIdx = -1;

            for (size_t idx : vIndices) {
                if (vpMapLineMatches[c][idx])
                    continue;

                const cv::Mat &d = pKF->mvDescriptors_line[c].row(idx);
                const float score = dML.dot(d);
                if (score > bestScore) {
                    bestScore = score;
                    bestIdx = static_cast<int>(idx);
                }
            }

            if (bestIdx >= 0 &&
                bestScore >= line_high_threshold * ratio_factor) {
                vpMapLineMatches[c][bestIdx] = pML;
                ++nmatches;
            }
        }
    }

    return nmatches;
}

int Matcher::SearchForTriangulation(KeyFramePtr pKF1, KeyFramePtr pKF2,
                                    TriangulationMatches &matches) {
    matches = {};
    matches.point_snapshot1 = pKF1->GetPointFeatureSnapshot(false, true);
    matches.point_snapshot2 = pKF2->GetPointFeatureSnapshot(false, true);
    const PointSnapshot &point_snapshot1 = matches.point_snapshot1;
    const PointSnapshot &point_snapshot2 = matches.point_snapshot2;
    const DBoW3::FeatureVector &vFeatVecPointKF1 =
        point_snapshot1.mFeatVec_point;
    const DBoW3::FeatureVector &vFeatVecPointKF2 =
        point_snapshot2.mFeatVec_point;

    const DBoW3::FeatureVector &vFeatVecLineKF1 = pKF1->mFeatVec_line;
    const DBoW3::FeatureVector &vFeatVecLineKF2 = pKF2->mFeatVec_line;

    matches.body_pose1 = pKF1->GetPose();
    matches.body_pose2 = pKF2->GetPose();
    const vector<Sophus::SE3f> vTbc1 = pKF1->GetCameraExtrinsics();
    const vector<Sophus::SE3f> vTbc2 = pKF2->GetCameraExtrinsics();
    matches.camera_poses1.resize(vTbc1.size());
    matches.camera_poses2.resize(vTbc2.size());
    for (size_t cam = 0; cam < vTbc1.size(); ++cam) {
        matches.camera_poses1[cam] = vTbc1[cam].inverse() * matches.body_pose1;
    }
    for (size_t cam = 0; cam < vTbc2.size(); ++cam) {
        matches.camera_poses2[cam] = vTbc2[cam].inverse() * matches.body_pose2;
    }
    const vector<Sophus::SE3f> &vT1w = matches.camera_poses1;
    const vector<Sophus::SE3f> &vT2w = matches.camera_poses2;
    vector<Sophus::SE3f> vTw2(vT2w.size());
    vector<Eigen::Vector3f> vC1w(vT1w.size());
    for (size_t cam = 0; cam < vT2w.size(); ++cam)
        vTw2[cam] = vT2w[cam].inverse();
    for (size_t cam = 0; cam < vT1w.size(); ++cam)
        vC1w[cam] = vT1w[cam].inverse().translation();

    int nCam = point_snapshot1.mnCam;

    vector<Eigen::Vector2f> vep(nCam * nCam);
    vector<Sophus::SE3f> vT12(nCam * nCam);
    vector<Eigen::Matrix3f> vR12(nCam * nCam);
    vector<Eigen::Vector3f> vt12(nCam * nCam);

    vector<GeometricCamera *> vpCameras1 = pKF1->mvpCameras;
    vector<GeometricCamera *> vpCameras2 = pKF2->mvpCameras;

    for (int c1 = 0; c1 < nCam; ++c1) {
        for (int c2 = 0; c2 < nCam; ++c2) {
            Eigen::Vector3f C2 = vT2w[c2] * vC1w[c1];
            vep[c1 * nCam + c2] = pKF2->mvpCameras[c2]->project(C2);

            vT12[c1 * nCam + c2] = vT1w[c1] * vTw2[c2];
            vR12[c1 * nCam + c2] = vT12[c1 * nCam + c2].rotationMatrix();
            vt12[c1 * nCam + c2] = vT12[c1 * nCam + c2].translation();
        }
    }

    int nmatches = 0;

    vector<bool> vbMatchedPoint2(
        point_snapshot2.mvPointDescriptorRowToCamRow.size(), false);
    vector<bool> vbMatchedLine2(pKF2->mnTotalLines, false);

    vector<int> vMatchesPoint12(
        point_snapshot1.mvPointDescriptorRowToCamRow.size(), -1);
    vector<int> vMatchesLine12(pKF1->mnTotalLines, -1);

    DBoW3::FeatureVector::const_iterator f1it = vFeatVecPointKF1.begin();
    DBoW3::FeatureVector::const_iterator f2it = vFeatVecPointKF2.begin();
    DBoW3::FeatureVector::const_iterator f1end = vFeatVecPointKF1.end();
    DBoW3::FeatureVector::const_iterator f2end = vFeatVecPointKF2.end();

    while (f1it != f1end && f2it != f2end) {
        if (f1it->first == f2it->first) {
            for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++) {
                const unsigned int idx1 = f1it->second[i1];
                int cam1, idx1_cam;
                ResolvePointRow(point_snapshot1, idx1, cam1, idx1_cam,
                                "SearchForTriangulation");

                MapPointPtr pMP1 = PointMapPointAt(
                    point_snapshot1, cam1, idx1_cam, "SearchForTriangulation");

                if (pMP1)
                    continue;

                const cv::Point2f &kp1 = PointKeyAt(
                    point_snapshot1, cam1, idx1_cam, "SearchForTriangulation");

                const cv::Mat d1 = PointFlatDescriptorRow(
                    point_snapshot1, idx1, "SearchForTriangulation");

                float bestScore = std::numeric_limits<float>::lowest();
                int bestIdx2 = -1;

                for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2;
                     i2++) {
                    size_t idx2 = f2it->second[i2];

                    int cam2, idx2_cam;
                    ResolvePointRow(point_snapshot2, idx2, cam2, idx2_cam,
                                    "SearchForTriangulation");

                    int idx12 = cam1 * nCam + cam2;

                    MapPointPtr pMP2 =
                        PointMapPointAt(point_snapshot2, cam2, idx2_cam,
                                        "SearchForTriangulation");

                    if (vbMatchedPoint2[idx2] || pMP2)
                        continue;

                    const cv::Mat d2 = PointFlatDescriptorRow(
                        point_snapshot2, idx2, "SearchForTriangulation");

                    const float score = d1.dot(d2);

                    if (score < TH_P_HIGH)
                        continue;

                    const cv::Point2f &kp2 =
                        PointKeyAt(point_snapshot2, cam2, idx2_cam,
                                   "SearchForTriangulation");

                    const float distex = vep[idx12](0) - kp2.x;
                    const float distey = vep[idx12](1) - kp2.y;

                    if (distex * distex + distey * distey < 100)
                        continue;

                    if (vpCameras1[cam1]->epipolarConstrain(
                            vpCameras2[cam2], kp1, kp2, vR12[idx12],
                            vt12[idx12], 1, 1)) {
                        if (score > bestScore) {
                            bestScore = score;
                            bestIdx2 = idx2;
                        }
                    }
                }

                if (bestIdx2 >= 0) {
                    vMatchesPoint12[idx1] = bestIdx2;
                    vbMatchedPoint2[bestIdx2] = true;
                    nmatches++;
                }
            }

            f1it++;
            f2it++;
        } else if (f1it->first < f2it->first) {
            f1it = vFeatVecPointKF1.lower_bound(f2it->first);
        } else {
            f2it = vFeatVecPointKF2.lower_bound(f1it->first);
        }
    }

    f1it = vFeatVecLineKF1.begin();
    f2it = vFeatVecLineKF2.begin();
    f1end = vFeatVecLineKF1.end();
    f2end = vFeatVecLineKF2.end();

    while (f1it != f1end && f2it != f2end) {
        if (f1it->first == f2it->first) {
            for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++) {
                const unsigned int idx1 = f1it->second[i1];
                int cam1, idx1_cam;
                tie(cam1, idx1_cam) = pKF1->mvLineDescriptorRowToCamRow[idx1];

                MapLinePtr pML1 = pKF1->GetMapLine(idx1_cam, cam1);

                if (pML1)
                    continue;

                const cv::Mat &d1 = pKF1->mDescriptors_line.row(idx1);

                float bestScore = std::numeric_limits<float>::lowest();
                float bestScore2 = std::numeric_limits<float>::lowest();
                int bestIdx2 = -1;

                for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2;
                     i2++) {
                    size_t idx2 = f2it->second[i2];

                    int cam2, idx2_cam;
                    tie(cam2, idx2_cam) =
                        pKF2->mvLineDescriptorRowToCamRow[idx2];

                    MapLinePtr pML2 = pKF2->GetMapLine(idx2_cam, cam2);

                    if (vbMatchedLine2[idx2] || pML2)
                        continue;

                    const cv::Mat &d2 = pKF2->mDescriptors_line.row(idx2);

                    const float score = d1.dot(d2);

                    if (score < TH_L_HIGH)
                        continue;

                    if (score > bestScore) {
                        bestScore2 = bestScore;
                        bestScore = score;
                        bestIdx2 = idx2;
                    } else if (score > bestScore2) {
                        bestScore2 = score;
                    }
                }

                if (bestIdx2 >= 0 &&
                    PassCosineLoweRatio(bestScore, bestScore2, mfNNratio)) {
                    vMatchesLine12[idx1] = bestIdx2;
                    vbMatchedLine2[bestIdx2] = true;
                    nmatches++;
                }
            }

            f1it++;
            f2it++;
        } else if (f1it->first < f2it->first) {
            f1it = vFeatVecLineKF1.lower_bound(f2it->first);
        } else {
            f2it = vFeatVecLineKF2.lower_bound(f1it->first);
        }
    }

    matches.point_matches.clear();
    matches.line_matches.clear();

    for (size_t i1 = 0; i1 < vMatchesPoint12.size(); i1++) {
        if (vMatchesPoint12[i1] >= 0) {
            int cam1 = -1;
            int idx1 = -1;
            int cam2 = -1;
            int idx2 = -1;
            ResolvePointRow(point_snapshot1, i1, cam1, idx1,
                            "SearchForTriangulationResult");
            ResolvePointRow(point_snapshot2, vMatchesPoint12[i1], cam2, idx2,
                            "SearchForTriangulationResult");
            matches.point_matches.push_back({cam1, idx1, cam2, idx2});
        }
    }

    for (size_t i1 = 0; i1 < vMatchesLine12.size(); i1++) {
        if (vMatchesLine12[i1] >= 0) {
            int cam1 = -1;
            int idx1 = -1;
            int cam2 = -1;
            int idx2 = -1;
            tie(cam1, idx1) = pKF1->mvLineDescriptorRowToCamRow[i1];
            tie(cam2, idx2) =
                pKF2->mvLineDescriptorRowToCamRow[vMatchesLine12[i1]];
            matches.line_matches.push_back({cam1, idx1, cam2, idx2});
        }
    }

    return nmatches;
}

int Matcher::Fuse(KeyFramePtr pKF,
                  const std::vector<std::vector<MapPointPtr>> &vpMapPoints_cam,
                  const std::vector<std::vector<MapLinePtr>> &vpMapLines_cam,
                  const float th_point, const float th_line_d,
                  const float th_line_theta) {
    vector<GeometricCamera *> vpCameras = pKF->mvpCameras;

    Map *pMap = pKF->GetMap();
    const PointSnapshot point_snapshot =
        pKF->GetPointFeatureSnapshot(true, false, false);
    int nCam = point_snapshot.mnCam;

    vector<Sophus::SE3f> vTcw = pKF->GetCameraPose();
    vector<Eigen::Matrix3f> vRcw(nCam);
    vector<Eigen::Vector3f> vtcw(nCam);

    for (int c = 0; c < nCam; ++c) {
        vRcw[c] = vTcw[c].rotationMatrix();
        vtcw[c] = vTcw[c].translation();
    }

    struct PendingPointFuse {
        MapPointPtr point;
        int cam;
        int idx;
        float score;
    };
    struct PendingLineFuse {
        MapLinePtr line;
        int cam;
        int idx;
        float score;
    };
    vector<PendingPointFuse> pending_point_fuses;
    vector<PendingLineFuse> pending_line_fuses;

    unordered_set<MapPointPtr> spFusedMapPoints;
    unordered_set<MapLinePtr> spFusedMapLines;

    for (int c2 = 0; c2 < nCam; ++c2) {
        if (c2 >= static_cast<int>(vpMapPoints_cam.size()) ||
            c2 >= static_cast<int>(vpMapLines_cam.size()))
            continue;

        for (int i = 0; i < vpMapPoints_cam[c2].size(); ++i) {
            MapPointPtr pMP = vpMapPoints_cam[c2][i];

            if (!pMP)
                continue;

            if (pMP->isBad())
                continue;

            if (spFusedMapPoints.count(pMP))
                continue;

            spFusedMapPoints.insert(pMP);

            const Eigen::Vector3f x3Dw = pMP->GetWorldPos();
            cv::Mat dMP;

            for (int c1 = 0; c1 < nCam; ++c1) {
                if (pMP->HasObservation(pKF, c1))
                    continue;

                if (pMP->isBad())
                    break;

                Eigen::Vector3f x3Dc = vTcw[c1] * x3Dw;

                if (x3Dc.z() < 0)
                    continue;

                const Eigen::Vector2f uv = vpCameras[c1]->project(x3Dc);

                if (!point_snapshot.IsPointInImage(uv.x(), uv.y(), c1))
                    continue;

                const vector<size_t> vIndices =
                    point_snapshot.GetPointFeaturesInArea(uv.x(), uv.y(),
                                                          th_point, c1);

                if (vIndices.empty())
                    continue;

                if (dMP.empty() && !pMP->GetDescriptorForMatching(dMP))
                    break;

                float bestScore = -1.0;
                int bestIdx = -1;
                bool hasDescriptorCandidate = false;
                for (vector<size_t>::const_iterator vit = vIndices.begin(),
                                                    vend = vIndices.end();
                     vit != vend; ++vit) {
                    size_t idx = *vit;
                    const cv::Point2f &kp =
                        PointKeyAt(point_snapshot, c1, idx, "Fuse");

                    const float ex = uv.x() - kp.x;
                    const float ey = uv.y() - kp.y;
                    if (ex * ex + ey * ey > 5.99)
                        continue;

                    hasDescriptorCandidate = true;
                    const cv::Mat d = PointCameraDescriptorRow(point_snapshot,
                                                               c1, idx, "Fuse");

                    const float score = dMP.dot(d);

                    if (score > bestScore) {
                        bestScore = score;
                        bestIdx = idx;
                    }
                }

                if (hasDescriptorCandidate)
                    pending_point_fuses.push_back(
                        {pMP, c1, bestIdx, bestScore});
            }
        }

        for (int i = 0; i < vpMapLines_cam[c2].size(); ++i) {
            MapLinePtr pML = vpMapLines_cam[c2][i];

            if (!pML)
                continue;

            if (pML->isBad())
                continue;

            if (spFusedMapLines.count(pML))
                continue;

            spFusedMapLines.insert(pML);

            const Eigen::Matrix<float, 6, 1> l3Dw = pML->GetWorldPos();
            cv::Mat dML;

            for (int c1 = 0; c1 < nCam; ++c1) {
                if (pML->HasObservation(pKF, c1))
                    continue;

                if (pML->isBad())
                    break;

                Eigen::Matrix<float, 6, 1> l3Dc;
                l3Dc.tail<3>() = vRcw[c1] * l3Dw.tail<3>();
                l3Dc.head<3>() =
                    vRcw[c1] * l3Dw.head<3>() + vtcw[c1].cross(l3Dc.tail<3>());

                const Eigen::Vector3f l2Dc = vpCameras[c1]->projectLine(l3Dc);

                if (!pKF->IsLineInImage(l2Dc, c1))
                    continue;

                float line_d = 0.0f;
                float line_theta = 0.0f;

                if (!GeometricTools::ImageLineToNormalizedPolar(l2Dc, line_d,
                                                                line_theta)) {
                    continue;
                }

                vector<size_t> vIndices = pKF->GetLineFeaturesInArea(
                    line_d, line_theta, th_line_d, th_line_theta, c1);

                if (vIndices.empty())
                    continue;

                if (dML.empty() && !pML->GetDescriptorForMatching(dML))
                    break;

                float bestScore = -1.0;
                int bestIdx = -1;
                bool hasDescriptorCandidate = false;

                for (vector<size_t>::const_iterator vit = vIndices.begin(),
                                                    vend = vIndices.end();
                     vit != vend; ++vit) {
                    size_t idx = *vit;
                    const Eigen::Vector4f &kl = pKF->mvLineKeysUn_cams[c1][idx];

                    Eigen::Vector2f error;
                    if (!GeometricTools::ComputeLineReprojectionError(
                            l3Dw, vpCameras[c1], vTcw[c1], kl, error))
                        continue;

                    const float chi2 = error.squaredNorm();
                    if (chi2 > 5.99)
                        continue;

                    hasDescriptorCandidate = true;
                    const cv::Mat &d = pKF->mvDescriptors_line[c1].row(idx);

                    const float score = dML.dot(d);

                    if (score > bestScore) {
                        bestScore = score;
                        bestIdx = idx;
                    }
                }

                if (hasDescriptorCandidate)
                    pending_line_fuses.push_back({pML, c1, bestIdx, bestScore});
            }
        }
    }

    unique_lock<mutex> map_update_lock;
    if (pMap) {
        map_update_lock = unique_lock<mutex>(pMap->mMutexMapUpdate);
    }

    if (!pKF->IsPointFeatureSnapshotCurrent(point_snapshot)) {
        if (map_update_lock.owns_lock())
            map_update_lock.unlock();
        return Fuse(pKF, vpMapPoints_cam, vpMapLines_cam, th_point, th_line_d,
                    th_line_theta);
    }

    int nFused = 0;
    unordered_set<MapPointPtr> changed_point_descriptors;
    unordered_set<MapLinePtr> changed_line_descriptors;
    for (const PendingPointFuse &pending : pending_point_fuses) {
        MapPointPtr pMP = pending.point;
        if (!pMP || pMP->isBad() || pMP->HasObservation(pKF, pending.cam))
            continue;

        int bestIdx = pending.idx;
        float bestScore = pending.score;
        if (changed_point_descriptors.count(pMP)) {
            cv::Mat descriptor;
            if (!pMP->GetDescriptorForMatching(descriptor))
                continue;
            bestIdx = -1;
            bestScore = -1.0f;
            const Eigen::Vector3f x3Dw = pMP->GetWorldPos();
            const Eigen::Vector3f x3Dc = vTcw[pending.cam] * x3Dw;
            if (x3Dc.z() < 0)
                continue;
            const Eigen::Vector2f uv = vpCameras[pending.cam]->project(x3Dc);
            if (!point_snapshot.IsPointInImage(uv.x(), uv.y(), pending.cam))
                continue;
            const vector<size_t> candidate_indices =
                point_snapshot.GetPointFeaturesInArea(uv.x(), uv.y(), th_point,
                                                      pending.cam);
            for (size_t idx : candidate_indices) {
                const cv::Point2f &key =
                    PointKeyAt(point_snapshot, pending.cam, idx, "FuseApply");
                const float ex = uv.x() - key.x;
                const float ey = uv.y() - key.y;
                if (ex * ex + ey * ey > 5.99f)
                    continue;
                const cv::Mat candidate = PointCameraDescriptorRow(
                    point_snapshot, pending.cam, idx, "FuseApply");
                const float score = descriptor.dot(candidate);
                if (score > bestScore) {
                    bestScore = score;
                    bestIdx = static_cast<int>(idx);
                }
            }
        }
        if (bestIdx < 0 || bestScore < TH_P_HIGH)
            continue;

        MapPointPtr pMPinKF = pKF->GetMapPoint(bestIdx, pending.cam);
        if (pMPinKF) {
            if (!pMPinKF->isBad()) {
                if (pMPinKF->Observations() > pMP->Observations()) {
                    pMP->Replace(pMPinKF);
                    if (pMPinKF != pMP)
                        changed_point_descriptors.insert(pMPinKF);
                } else {
                    pMPinKF->Replace(pMP);
                    if (pMPinKF != pMP)
                        changed_point_descriptors.insert(pMP);
                }
            }
        } else {
            pMP->AddObservation(pKF, bestIdx, pending.cam);
            pKF->AddMapPoint(pMP, bestIdx, pending.cam);
        }
        ++nFused;
    }

    for (const PendingLineFuse &pending : pending_line_fuses) {
        MapLinePtr pML = pending.line;
        if (!pML || pML->isBad() || pML->HasObservation(pKF, pending.cam))
            continue;

        int bestIdx = pending.idx;
        float bestScore = pending.score;
        if (changed_line_descriptors.count(pML)) {
            cv::Mat descriptor;
            if (!pML->GetDescriptorForMatching(descriptor))
                continue;
            bestIdx = -1;
            bestScore = -1.0f;
            const Eigen::Matrix<float, 6, 1> l3Dw = pML->GetWorldPos();
            Eigen::Matrix<float, 6, 1> l3Dc;
            l3Dc.tail<3>() = vRcw[pending.cam] * l3Dw.tail<3>();
            l3Dc.head<3>() = vRcw[pending.cam] * l3Dw.head<3>() +
                             vtcw[pending.cam].cross(l3Dc.tail<3>());
            const Eigen::Vector3f l2Dc =
                vpCameras[pending.cam]->projectLine(l3Dc);
            if (!pKF->IsLineInImage(l2Dc, pending.cam))
                continue;
            float line_d = 0.0f;
            float line_theta = 0.0f;
            if (!GeometricTools::ImageLineToNormalizedPolar(l2Dc, line_d,
                                                            line_theta))
                continue;
            const vector<size_t> candidate_indices = pKF->GetLineFeaturesInArea(
                line_d, line_theta, th_line_d, th_line_theta, pending.cam);
            for (size_t idx : candidate_indices) {
                const Eigen::Vector4f &key =
                    pKF->mvLineKeysUn_cams[pending.cam][idx];
                Eigen::Vector2f error;
                if (!GeometricTools::ComputeLineReprojectionError(
                        l3Dw, vpCameras[pending.cam], vTcw[pending.cam], key,
                        error) ||
                    error.squaredNorm() > 5.99f)
                    continue;
                const cv::Mat &candidate =
                    pKF->mvDescriptors_line[pending.cam].row(idx);
                const float score = descriptor.dot(candidate);
                if (score > bestScore) {
                    bestScore = score;
                    bestIdx = static_cast<int>(idx);
                }
            }
        }
        if (bestIdx < 0 || bestScore < TH_L_HIGH)
            continue;

        MapLinePtr pMLinKF = pKF->GetMapLine(bestIdx, pending.cam);
        if (pMLinKF) {
            if (!pMLinKF->isBad()) {
                if (pMLinKF->Observations() > pML->Observations()) {
                    pML->Replace(pMLinKF);
                    if (pMLinKF != pML)
                        changed_line_descriptors.insert(pMLinKF);
                } else {
                    pMLinKF->Replace(pML);
                    if (pMLinKF != pML)
                        changed_line_descriptors.insert(pML);
                }
            }
        } else {
            pML->AddObservation(pKF, bestIdx, pending.cam);
            pKF->AddMapLine(pML, bestIdx, pending.cam);
        }
        ++nFused;
    }

    return nFused;
}

int Matcher::Fuse(KeyFramePtr pKF, Sophus::SE3f &Tbw,
                  const std::vector<MapPointPtr> &vpPoints,
                  std::vector<MapLinePtr> &vpLines, const float th_point,
                  const float th_line_d, const float th_line_theta,
                  std::vector<MapPointPtr> &vpReplacedPoints,
                  std::vector<MapLinePtr> &vpReplacedLines,
                  float point_high_threshold, float line_high_threshold) {
    vector<GeometricCamera *> vpCameras = pKF->mvpCameras;

    Map *pMap = pKF->GetMap();
    unique_lock<mutex> map_update_lock;
    if (pMap) {
        map_update_lock = unique_lock<mutex>(pMap->mMutexMapUpdate);
    }

    const PointSnapshot point_snapshot =
        pKF->GetPointFeatureSnapshot(true, false, false);
    int nCam = point_snapshot.mnCam;

    // vector<Sophus::SE3f> vTcw = pKF->GetCameraPose();
    vector<Sophus::SE3f> vTbc = pKF->GetCameraExtrinsics();
    vector<Eigen::Matrix3f> vRcw(nCam);
    vector<Eigen::Vector3f> vtcw(nCam);

    for (int c = 0; c < nCam; ++c) {
        Sophus::SE3f Tcw = vTbc[c].inverse() * Tbw;
        vRcw[c] = Tcw.rotationMatrix();
        vtcw[c] = Tcw.translation();
    }

    int nFused = 0;

    const int nPoints = vpPoints.size();
    const int nLines = vpLines.size();

    for (int iMP = 0; iMP < nPoints; ++iMP) {
        MapPointPtr pMP = vpPoints[iMP];

        if (pMP->isBad())
            continue;

        for (int c1 = 0; c1 < nCam; ++c1) {
            if (pMP->HasObservation(pKF, c1))
                continue;

            if (pMP->isBad())
                break;

            Eigen::Vector3f x3Dw = pMP->GetWorldPos();
            Eigen::Vector3f x3Dc = vRcw[c1] * x3Dw + vtcw[c1];

            if (x3Dc.z() < 0)
                continue;

            const Eigen::Vector2f uv = vpCameras[c1]->project(x3Dc);

            if (!point_snapshot.IsPointInImage(uv.x(), uv.y(), c1))
                continue;

            const vector<size_t> vIndices =
                point_snapshot.GetPointFeaturesInArea(uv.x(), uv.y(), th_point,
                                                      c1);

            if (vIndices.empty())
                continue;

            cv::Mat dMP;
            if (!pMP->GetDescriptorForMatching(dMP))
                continue;

            float bestScore = -1.0;
            int bestIdx = -1;
            for (vector<size_t>::const_iterator vit = vIndices.begin(),
                                                vend = vIndices.end();
                 vit != vend; ++vit) {
                size_t idx = *vit;
                const cv::Mat d = PointCameraDescriptorRow(point_snapshot, c1,
                                                           idx, "FuseSim3");

                const float score = dMP.dot(d);

                if (score > bestScore) {
                    bestScore = score;
                    bestIdx = idx;
                }
            }

            if (bestScore >= point_high_threshold) {
                MapPointPtr pMPinKF = pKF->GetMapPoint(bestIdx, c1);
                if (pMPinKF) {
                    if (!pMPinKF->isBad())
                        vpReplacedPoints[iMP] = pMPinKF;
                } else {
                    pMP->AddObservation(pKF, bestIdx, c1);
                    pKF->AddMapPoint(pMP, bestIdx, c1);
                }
                nFused++;
            }
        }
    }

    for (int iML = 0; iML < nLines; ++iML) {
        MapLinePtr pML = vpLines[iML];

        if (pML->isBad())
            continue;

        for (int c1 = 0; c1 < nCam; ++c1) {
            if (pML->HasObservation(pKF, c1))
                continue;

            if (pML->isBad())
                break;

            Eigen::Matrix<float, 6, 1> l3Dw = pML->GetWorldPos();
            Eigen::Matrix<float, 6, 1> l3Dc;
            l3Dc.tail<3>() = vRcw[c1] * l3Dw.tail<3>();
            l3Dc.head<3>() =
                vRcw[c1] * l3Dw.head<3>() + vtcw[c1].cross(l3Dc.tail<3>());

            const Eigen::Vector3f l2Dc = vpCameras[c1]->projectLine(l3Dc);

            if (!pKF->IsLineInImage(l2Dc, c1))
                continue;

            float line_d = 0.0f;
            float line_theta = 0.0f;

            if (!GeometricTools::ImageLineToNormalizedPolar(l2Dc, line_d,
                                                            line_theta)) {
                continue;
            }

            vector<size_t> vIndices = pKF->GetLineFeaturesInArea(
                line_d, line_theta, th_line_d, th_line_theta, c1);

            if (vIndices.empty())
                continue;

            const cv::Mat dML = pML->GetDescriptor();

            float bestScore = -1.0;
            int bestIdx = -1;

            for (vector<size_t>::const_iterator vit = vIndices.begin(),
                                                vend = vIndices.end();
                 vit != vend; ++vit) {
                size_t idx = *vit;
                const cv::Mat &d = pKF->mvDescriptors_line[c1].row(idx);

                const float score = dML.dot(d);

                if (score > bestScore) {
                    bestScore = score;
                    bestIdx = idx;
                }
            }

            if (bestScore >= line_high_threshold) {
                MapLinePtr pMLinKF = pKF->GetMapLine(bestIdx, c1);
                if (pMLinKF) {
                    if (!pMLinKF->isBad())
                        vpReplacedLines[iML] = pMLinKF;
                } else {
                    pML->AddObservation(pKF, bestIdx, c1);
                    pKF->AddMapLine(pML, bestIdx, c1);
                }
                nFused++;
            }
        }
    }

    return nFused;
}

}; // namespace XPLSLAM
