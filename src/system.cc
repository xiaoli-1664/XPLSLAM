#include "runtime_timing.h"
#include "system.h"
#include "converter.h"
#ifdef XPLSLAM_WITH_PANGOLIN_VIEWER
#include "frame_drawer.h"
#include "map_drawer.h"
#include "viewer.h"
#endif
#include "sophus/se3.hpp"
#include <opencv2/core/persistence.hpp>

#include <algorithm>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/string.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <openssl/md5.h>

using namespace std;

namespace XPLSLAM {

Verbose::eLevel Verbose::th = Verbose::VERBOSITY_NORMAL;

namespace {

// Single choke point for the per-stage timing queries in TrackMultiCamera:
// gating it makes XPLSLAM_ENABLE_PROFILING=OFF a zero-cost switch without
// touching the stage-timing call sites below.
inline chrono::steady_clock::time_point SystemNow() {
#ifdef XPLSLAM_ENABLE_PROFILING
    return chrono::steady_clock::now();
#else
    return chrono::steady_clock::time_point();
#endif
}

constexpr float kMinEpipolarLineNorm = 1e-6f;
constexpr float kMinProjectedDepth = 1e-4f;

bool PointToPointGridCell(const cv::Point2f &point, const cv::Size &image_size,
                          int &grid_x, int &grid_y) {
    if (image_size.width <= 0 || image_size.height <= 0 || point.x < 0.0f ||
        point.x >= static_cast<float>(image_size.width) || point.y < 0.0f ||
        point.y >= static_cast<float>(image_size.height)) {
        return false;
    }

    grid_x = std::min(
        FRAME_GRID_COLS_POINT - 1,
        std::max(0, static_cast<int>(std::floor(
                        point.x * static_cast<float>(FRAME_GRID_COLS_POINT) /
                        static_cast<float>(image_size.width)))));
    grid_y = std::min(
        FRAME_GRID_ROWS_POINT - 1,
        std::max(0, static_cast<int>(std::floor(
                        point.y * static_cast<float>(FRAME_GRID_ROWS_POINT) /
                        static_cast<float>(image_size.height)))));
    return true;
}

void AddGridCellForPoint(const cv::Point2f &point, const cv::Size &image_size,
                         std::unordered_set<size_t> &grid_candidates) {
    int grid_x = 0;
    int grid_y = 0;
    if (!PointToPointGridCell(point, image_size, grid_x, grid_y)) {
        return;
    }

    grid_candidates.insert(
        static_cast<size_t>(grid_y * FRAME_GRID_COLS_POINT + grid_x));
}

void AppendUniquePoint(std::vector<cv::Point2f> &points,
                       const cv::Point2f &candidate) {
    constexpr float kMinDistance2 = 1e-4f;
    for (const cv::Point2f &point : points) {
        const float dx = point.x - candidate.x;
        const float dy = point.y - candidate.y;
        if (dx * dx + dy * dy < kMinDistance2) {
            return;
        }
    }

    points.push_back(candidate);
}

void AddLineRectangleIntersections(const Eigen::Vector3f &line,
                                   const cv::Size &image_size,
                                   std::vector<cv::Point2f> &intersections) {
    if (image_size.width <= 0 || image_size.height <= 0) {
        return;
    }

    const float a = line(0);
    const float b = line(1);
    const float c = line(2);
    const float max_x = static_cast<float>(image_size.width - 1);
    const float max_y = static_cast<float>(image_size.height - 1);

    if (std::abs(b) > kMinEpipolarLineNorm) {
        const float y_left = (-c - a * 0.0f) / b;
        if (y_left >= 0.0f && y_left <= max_y) {
            AppendUniquePoint(intersections, cv::Point2f(0.0f, y_left));
        }

        const float y_right = (-c - a * max_x) / b;
        if (y_right >= 0.0f && y_right <= max_y) {
            AppendUniquePoint(intersections, cv::Point2f(max_x, y_right));
        }
    }

    if (std::abs(a) > kMinEpipolarLineNorm) {
        const float x_top = (-c - b * 0.0f) / a;
        if (x_top >= 0.0f && x_top <= max_x) {
            AppendUniquePoint(intersections, cv::Point2f(x_top, 0.0f));
        }

        const float x_bottom = (-c - b * max_y) / a;
        if (x_bottom >= 0.0f && x_bottom <= max_x) {
            AppendUniquePoint(intersections, cv::Point2f(x_bottom, max_y));
        }
    }
}

void RasterizeSegmentToPointGrid(const cv::Point2f &start,
                                 const cv::Point2f &end,
                                 const cv::Size &image_size,
                                 std::unordered_set<size_t> &grid_candidates) {
    const float cell_width =
        static_cast<float>(image_size.width) / FRAME_GRID_COLS_POINT;
    const float cell_height =
        static_cast<float>(image_size.height) / FRAME_GRID_ROWS_POINT;
    if (cell_width <= 0.0f || cell_height <= 0.0f) {
        return;
    }

    const float dx = end.x - start.x;
    const float dy = end.y - start.y;
    const float steps_f =
        std::max(std::abs(dx) / cell_width, std::abs(dy) / cell_height) * 4.0f;
    const int steps = std::max(1, static_cast<int>(std::ceil(steps_f)));

    for (int step = 0; step <= steps; ++step) {
        const float t = static_cast<float>(step) / static_cast<float>(steps);
        AddGridCellForPoint(start + t * (end - start), image_size,
                            grid_candidates);
    }
}

Eigen::Matrix3f ComputeEpipolarFundamental(const GeometricCamera &first_camera,
                                           const GeometricCamera &second_camera,
                                           const Sophus::SE3f &Tsecond_first) {
    const Eigen::Matrix3f first_K = first_camera.toK_();
    const Eigen::Matrix3f second_K = second_camera.toK_();
    const Eigen::Matrix3f rotation = Tsecond_first.rotationMatrix();
    const Eigen::Vector3f translation = Tsecond_first.translation();
    return second_K.transpose().inverse() * Sophus::SO3f::hat(translation) *
           rotation * first_K.inverse();
}

void AddProjectedFallbackCandidates(
    GeometricCamera &first_camera, GeometricCamera &second_camera,
    const Sophus::SE3f &Tsecond_first, const cv::Point2f &first_center,
    const cv::Size &second_size, std::unordered_set<size_t> &grid_candidates) {
    const Eigen::Vector3f first_ray = first_camera.unprojectEig(first_center);
    for (const float depth : {1.0f, 10.0f, 100.0f}) {
        const Eigen::Vector3f point_first = depth * first_ray;
        const Eigen::Vector3f point_second =
            Tsecond_first.rotationMatrix() * point_first +
            Tsecond_first.translation();
        if (!point_second.allFinite() ||
            point_second.z() <= kMinProjectedDepth) {
            continue;
        }

        const Eigen::Vector2f projected = second_camera.project(point_second);
        AddGridCellForPoint(cv::Point2f(projected.x(), projected.y()),
                            second_size, grid_candidates);
    }

    const Eigen::Vector3f epipole = Tsecond_first.translation();
    if (epipole.z() > kMinProjectedDepth && epipole.allFinite()) {
        const Eigen::Vector2f projected = second_camera.project(epipole);
        AddGridCellForPoint(cv::Point2f(projected.x(), projected.y()),
                            second_size, grid_candidates);
    }
}

} // namespace

System::System(const std::string &strVocFilePoint,
               const std::string &strVocFileLine,
               const std::string &strSettingsFile, const bool bUseViewer,
               const int initFr, const std::string &strSequence)
    : mbReset(false), mbResetActiveMap(false),
      mbActivateLocalizationMode(false), mbDeactivateLocalizationMode(false),
      mbShutDown(false), mpMapSparsification(nullptr),
      mptLocalMapping(nullptr), mptLoopClosing(nullptr),
      mptMapSparsification(nullptr), mptViewer(nullptr),
      mpFrameDrawer(nullptr), mpMapDrawer(nullptr),
      mpViewer(nullptr), mTrackingState(Tracking::NO_IMAGES_YET) {
    cv::FileStorage fsSettings(strSettingsFile, cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
        std::cerr << "Failed to open settings file at: " << strSettingsFile
                  << std::endl;
        exit(-1);
    }

    settings_ = new Settings(strSettingsFile);
    mStrLoadAtlasFromFile = settings_->atlasLoadFile();
    mStrSaveAtlasToFile = settings_->atlasSaveFile();
    PrecomputeStereoPointIndexInGrid();
    std::cout << "Precomputed stereo point index in grid." << std::endl;

    cv::FileNode node = fsSettings["loopClosing"];
    bool activeLC = true;
    if (!node.empty()) {
        activeLC = static_cast<int>(fsSettings["loopClosing"]) != 0;
    }

    mStrVocabularyPointFilePath = strVocFilePoint;
    mStrVocabularyLineFilePath = strVocFileLine;

    bool loadedAtlas = false;

    std::cout << "Loading vocabularies from files: " << strVocFilePoint
              << " and " << strVocFileLine << std::endl;

    if (mStrLoadAtlasFromFile.empty()) {
        std::cout << "No Atlas file specified, initializing an empty Atlas."
                  << std::endl;
        mpVocabulary_point = new DBoW3::Vocabulary(strVocFilePoint);
        mpVocabulary_line = new DBoW3::Vocabulary(strVocFileLine);

        std::cout << "Vocabularies loaded successfully." << std::endl;

        mpKeyFrameDatabase =
            new KeyFrameDatabase(*mpVocabulary_point, *mpVocabulary_line);
        std::cout << "Key Frame Database initialized." << std::endl;

        mpAtlas = new Atlas(0);
    } else {
        mpVocabulary_point = new DBoW3::Vocabulary(strVocFilePoint);
        mpVocabulary_line = new DBoW3::Vocabulary(strVocFileLine);

        mpKeyFrameDatabase =
            new KeyFrameDatabase(*mpVocabulary_point, *mpVocabulary_line);

        cout << "Initialization of Atlas from file: " << mStrLoadAtlasFromFile
             << endl;
        bool isRead = LoadAtlas(FileType::BINARY_FILE);

        if (!isRead) {
            cout << "Error loading the Atlas from file, trying with text file"
                 << endl;
            exit(-1);
        }

        loadedAtlas = true;

        mpAtlas->CreateNewMap();
    }

    cout << "Seq. Name: " << strSequence << endl;
    mpTracker = new Tracking(this, mpVocabulary_point, mpVocabulary_line,
                             mpAtlas, mpKeyFrameDatabase, strSettingsFile,
                             settings_, strSequence);

    mpLocalMapper = new LocalMapping(this, mpAtlas, strSequence);
    mpLocalMapper->mInitFr = initFr;

    mpLocalMapper->mThFarPoints = settings_->thFarPoints();
    mpLocalMapper->mThFarLines = settings_->thFarLines();

    if (mpLocalMapper->mThFarPoints != 0) {
        cout << "Disarding far points with a threshold of "
             << mpLocalMapper->mThFarPoints << " meters" << endl;
        mpLocalMapper->mbFarPoints = true;
    } else {
        mpLocalMapper->mbFarPoints = false;
    }

    if (mpLocalMapper->mThFarLines != 0) {
        cout << "Disarding far lines with a threshold of "
             << mpLocalMapper->mThFarLines << " meters" << endl;
        mpLocalMapper->mbFarLines = true;
    } else {
        mpLocalMapper->mbFarLines = false;
    }

    mpLoopCloser =
        new LoopClosing(mpAtlas, mpKeyFrameDatabase, mpVocabulary_point,
                        mpVocabulary_line, true, activeLC);

#ifdef XPLSLAM_WITH_MAP_SPARSIFICATION
    const MapSparsificationConfig &sparsification =
        settings_->mapSparsification();
    if (sparsification.enabled) {
        mpMapSparsification =
            new MapSparsification(sparsification, mpAtlas);
        mpMapSparsification->SetLoopClosing(mpLoopCloser);
        mpLoopCloser->SetMapSparsification(mpMapSparsification);
        mpLocalMapper->SetMapSparsification(
            mpMapSparsification, sparsification.non_local_keyframes);
    } else {
        mpMapSparsification = nullptr;
    }
#else
    mpMapSparsification = nullptr;
#endif

    mpTracker->SetLocalMapper(mpLocalMapper);
    mpTracker->SetLoopClosing(mpLoopCloser);

    mpLocalMapper->SetTracker(mpTracker);
    mpLocalMapper->SetLoopCloser(mpLoopCloser);

    mpLoopCloser->SetTracker(mpTracker);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);

    mptLocalMapping = new thread(&LocalMapping::Run, mpLocalMapper);
    mptLoopClosing = new thread(&LoopClosing::Run, mpLoopCloser);
    if (mpMapSparsification) {
        mptMapSparsification =
            new thread(&MapSparsification::Run, mpMapSparsification);
    }

#ifdef XPLSLAM_WITH_PANGOLIN_VIEWER
    if (bUseViewer) {
        mpFrameDrawer = new FrameDrawer(mpAtlas);
        mpMapDrawer = new MapDrawer(mpAtlas, settings_, strSettingsFile);
        mpViewer = new Viewer(this, mpFrameDrawer, mpMapDrawer, settings_,
                              strSettingsFile);
        mpAtlas->SetViewer(mpViewer);
        mptViewer = new thread(&Viewer::Run, mpViewer);
    }
#else
    if (bUseViewer) {
        cerr << "XPLSLAM was built without Pangolin viewer support." << endl;
    }
#endif

    Verbose::SetTh(Verbose::VERBOSITY_QUIET);
}

void System::PrecomputeStereoPointIndexInGrid() {
    const std::vector<std::pair<int, int>> stereo_pairs =
        settings_->stereoPairs();
    const std::vector<GeometricCamera *> cameras = settings_->cameras();
    const std::vector<Sophus::SE3f> vTbc = settings_->Tbc();

    mvStereoPointIndexInGrid.assign(
        stereo_pairs.size(),
        std::vector<std::vector<std::unordered_set<size_t>>>(
            FRAME_GRID_ROWS_POINT,
            std::vector<std::unordered_set<size_t>>(FRAME_GRID_COLS_POINT)));

    for (size_t pair_idx = 0; pair_idx < stereo_pairs.size(); ++pair_idx) {
        const int first_cam = stereo_pairs[pair_idx].first;
        const int second_cam = stereo_pairs[pair_idx].second;
        if (first_cam < 0 || second_cam < 0 ||
            first_cam >= static_cast<int>(cameras.size()) ||
            second_cam >= static_cast<int>(cameras.size()) ||
            first_cam >= static_cast<int>(vTbc.size()) ||
            second_cam >= static_cast<int>(vTbc.size()) ||
            cameras[first_cam] == nullptr || cameras[second_cam] == nullptr) {
            continue;
        }

        const cv::Size first_size = settings_->imSize(first_cam);
        const cv::Size second_size = settings_->imSize(second_cam);
        const Sophus::SE3f Tsecond_first =
            vTbc[second_cam].inverse() * vTbc[first_cam];
        const Eigen::Matrix3f fundamental = ComputeEpipolarFundamental(
            *cameras[first_cam], *cameras[second_cam], Tsecond_first);

        for (int grid_y = 0; grid_y < FRAME_GRID_ROWS_POINT; ++grid_y) {
            for (int grid_x = 0; grid_x < FRAME_GRID_COLS_POINT; ++grid_x) {
                auto &grid_candidates =
                    mvStereoPointIndexInGrid[pair_idx][grid_y][grid_x];
                const float center_x =
                    (static_cast<float>(grid_x) + 0.5f) *
                    static_cast<float>(first_size.width) /
                    static_cast<float>(FRAME_GRID_COLS_POINT);
                const float center_y =
                    (static_cast<float>(grid_y) + 0.5f) *
                    static_cast<float>(first_size.height) /
                    static_cast<float>(FRAME_GRID_ROWS_POINT);

                const Eigen::Vector3f line =
                    fundamental * Eigen::Vector3f(center_x, center_y, 1.0f);
                if (line.head<2>().norm() >= kMinEpipolarLineNorm &&
                    line.allFinite()) {
                    std::vector<cv::Point2f> intersections;
                    AddLineRectangleIntersections(line, second_size,
                                                  intersections);
                    if (intersections.size() >= 2) {
                        float best_distance2 = -1.0f;
                        cv::Point2f best_start = intersections[0];
                        cv::Point2f best_end = intersections[1];
                        for (size_t i = 0; i < intersections.size(); ++i) {
                            for (size_t j = i + 1; j < intersections.size();
                                 ++j) {
                                const cv::Point2f diff =
                                    intersections[i] - intersections[j];
                                const float distance2 = diff.dot(diff);
                                if (distance2 > best_distance2) {
                                    best_distance2 = distance2;
                                    best_start = intersections[i];
                                    best_end = intersections[j];
                                }
                            }
                        }
                        RasterizeSegmentToPointGrid(
                            best_start, best_end, second_size, grid_candidates);
                    }
                }

                AddProjectedFallbackCandidates(
                    *cameras[first_cam], *cameras[second_cam], Tsecond_first,
                    cv::Point2f(center_x, center_y), second_size,
                    grid_candidates);
            }
        }
    }
}

bool System::LoadAtlas(int type) {
#ifndef XPLSLAM_ENABLE_ATLAS_SERIALIZATION
    (void)type;
    cerr << "Atlas loading is disabled: atlas serialization is incomplete."
         << endl;
    return false;
#else
    string strFileVocPoint, strFileVocLine, strPointVocChecksum,
        strLineVocChecksum;
    bool isRead = false;

    string pathLoadFileName = "./";
    pathLoadFileName = pathLoadFileName.append(mStrLoadAtlasFromFile);
    pathLoadFileName = pathLoadFileName.append(".osa");

    if (type == TEXT_FILE) // File text
    {
        cout << "Starting to read the save text file " << endl;
        std::ifstream ifs(pathLoadFileName, std::ios::binary);
        if (!ifs.good()) {
            cout << "Load file not found" << endl;
            return false;
        }
        boost::archive::text_iarchive ia(ifs);
        ia >> strFileVocPoint;
        ia >> strPointVocChecksum;
        ia >> strFileVocLine;
        ia >> strLineVocChecksum;
        ia >> mpAtlas;
        cout << "End to load the save text file " << endl;
        isRead = true;
    } else if (type == BINARY_FILE) // File binary
    {
        cout << "Starting to read the save binary file" << endl;
        std::ifstream ifs(pathLoadFileName, std::ios::binary);
        if (!ifs.good()) {
            cout << "Load file not found" << endl;
            return false;
        }
        boost::archive::binary_iarchive ia(ifs);
        ia >> strFileVocPoint;
        ia >> strPointVocChecksum;
        ia >> strFileVocLine;
        ia >> strLineVocChecksum;
        ia >> mpAtlas;
        cout << "End to load the save binary file" << endl;
        isRead = true;
    }

    if (isRead) {
        // Check if the vocabulary is the same
        string strInputPointVocabularyChecksum =
            CalculateCheckSum(mStrVocabularyPointFilePath, TEXT_FILE);
        string strInputLineVocabularyChecksum =
            CalculateCheckSum(mStrVocabularyLineFilePath, TEXT_FILE);

        if (strInputPointVocabularyChecksum.compare(strPointVocChecksum) != 0) {
            cout << "The vocabulary load isn't the same which the load session "
                    "was created "
                 << endl;
            cout << "-Vocabulary name: " << strFileVocPoint << endl;
            return false; // Both are differents
        }

        if (strInputLineVocabularyChecksum.compare(strLineVocChecksum) != 0) {
            cout << "The vocabulary load isn't the same which the load session "
                    "was created "
                 << endl;
            cout << "-Vocabulary name: " << strFileVocLine << endl;
            return false; // Both are differents
        }

        mpAtlas->SetKeyFrameDababase(mpKeyFrameDatabase);
        mpAtlas->SetVocabulary(mpVocabulary_point, mpVocabulary_line);
        mpAtlas->PostLoad();

        return true;
    }
    return false;
#endif
}

string System::CalculateCheckSum(string filename, int type) {
    string checksum = "";

    unsigned char c[MD5_DIGEST_LENGTH];

    std::ios_base::openmode flags = std::ios::in;
    if (type == BINARY_FILE) // Binary file
        flags = std::ios::in | std::ios::binary;

    ifstream f(filename.c_str(), flags);
    if (!f.is_open()) {
        cout << "[E] Unable to open the in file " << filename
             << " for Md5 hash." << endl;
        return checksum;
    }

    MD5_CTX md5Context;
    char buffer[1024];

    MD5_Init(&md5Context);
    while (int count = f.readsome(buffer, sizeof(buffer))) {
        MD5_Update(&md5Context, buffer, count);
    }

    f.close();

    MD5_Final(c, &md5Context);

    for (int i = 0; i < MD5_DIGEST_LENGTH; i++) {
        char aux[10];
        sprintf(aux, "%02x", c[i]);
        checksum = checksum + aux;
    }

    return checksum;
}

Sophus::SE3f System::TrackMultiCamera(const std::vector<cv::Mat> &vIms,
                                      const double &timestamp,
                                      std::string filename) {
    using Clock = chrono::steady_clock;
    const auto total_begin = SystemNow();
    auto stage_begin = total_begin;
    ProcessedFrameTiming timing;

    std::vector<cv::Mat> vImsToFeed(vIms.size());
    {
        RuntimePhaseScope runtime_phase(RuntimePhase::BaseFeatures);
        for (size_t i = 0; i < vIms.size(); i++) {
            if (settings_->needToUndistort(i)) {
                cv::remap(vIms[i], vImsToFeed[i], settings_->M1(i),
                          settings_->M2(i), cv::INTER_LINEAR);
            } else {
                vImsToFeed[i] = vIms[i].clone();
            }
        }
    }
    timing.image_preparation_ms =
        chrono::duration<double, milli>(SystemNow() - stage_begin).count();

    stage_begin = SystemNow();
    {
        unique_lock<mutex> lock(mMutexMode);
        if (mbActivateLocalizationMode) {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while (!mpLocalMapper->isStopped()) {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if (mbDeactivateLocalizationMode) {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }
    timing.mode_control_ms =
        chrono::duration<double, milli>(SystemNow() - stage_begin).count();

    stage_begin = SystemNow();
    {
        unique_lock<mutex> lock(mMutexReset);
        if (mbReset) {
            if (mpMapSparsification)
                mpMapSparsification->RequestReset();
            mpTracker->Reset();
            mbReset = false;
            mbResetActiveMap = false;
        } else if (mbResetActiveMap) {
            if (mpMapSparsification)
                mpMapSparsification->RequestResetActiveMap(
                    mpAtlas->GetCurrentMap());
            mpTracker->ResetActiveMap();
            mbResetActiveMap = false;
        }
    }
    timing.reset_control_ms =
        chrono::duration<double, milli>(SystemNow() - stage_begin).count();

    Sophus::SE3f Tbw =
        mpTracker->GrabMultiCamera(vImsToFeed, timestamp, filename);
    timing.tracker = mpTracker->GetLastMultiCameraTiming();
    timing.frame_id = timing.tracker.frame_id;

    stage_begin = SystemNow();
    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
#ifdef XPLSLAM_ENABLE_PROFILING
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints_cams;
    mTrackedMapLines = mpTracker->mCurrentFrame.mvpMapLines_cams;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn_cams;
    mTrackedKeyLinesUn = mpTracker->mCurrentFrame.mvLineKeysUn_cams;
#endif
#ifdef XPLSLAM_WITH_PANGOLIN_VIEWER
    if (mpFrameDrawer) {
        mpFrameDrawer->Update(mpTracker->mCurrentFrame, mpTracker->mState);
    }
    if (mpMapDrawer && mpTracker->mCurrentFrame.HasPose()) {
        mpMapDrawer->SetCurrentCameraPose(
            Tbw, mpTracker->mCurrentFrame.mpReferenceKF);
        mpMapDrawer->SetCurrentCameraExtrinsics(
            mpTracker->mCurrentFrame.GetCameraExtrinsics());
    }
#endif
    timing.state_update_ms =
        chrono::duration<double, milli>(SystemNow() - stage_begin).count();
    timing.total_ms =
        chrono::duration<double, milli>(SystemNow() - total_begin).count();
    mLastProcessedFrameTiming = timing;

    return Tbw;
}

ProcessedFrameTiming System::GetLastProcessedFrameTiming() {
    unique_lock<mutex> lock(mMutexState);
    return mLastProcessedFrameTiming;
}

void System::ResetActiveMap() {
    unique_lock<mutex> lock(mMutexReset);
    mbResetActiveMap = true;
#ifdef XPLSLAM_WITH_PANGOLIN_VIEWER
    if (mpMapDrawer) {
        mpMapDrawer->ClearTrajectory();
    }
#endif
}

void System::ActivateLocalizationMode() {
    unique_lock<mutex> lock(mMutexMode);
    mbActivateLocalizationMode = true;
}

void System::DeactivateLocalizationMode() {
    unique_lock<mutex> lock(mMutexMode);
    mbDeactivateLocalizationMode = true;
}

bool System::MapChanged() { return false; }

void System::Reset() {
    unique_lock<mutex> lock(mMutexReset);
    mbReset = true;
#ifdef XPLSLAM_WITH_PANGOLIN_VIEWER
    if (mpMapDrawer) {
        mpMapDrawer->ClearTrajectory();
    }
#endif
}

void System::Shutdown() {
    {
        unique_lock<mutex> lock(mMutexReset);
        mbShutDown = true;
    }

    mpLocalMapper->RequestFinish();
#ifdef XPLSLAM_WITH_PANGOLIN_VIEWER
    if (mpViewer) {
        mpViewer->RequestFinish();
    }
#endif

    if (mptLocalMapping && mptLocalMapping->joinable()) {
        mptLocalMapping->join();
    }
    if (mpMapSparsification) {
        mpMapSparsification->RequestFinish();
        if (mptMapSparsification && mptMapSparsification->joinable())
            mptMapSparsification->join();
    }
    mpLoopCloser->RequestFinish();
    if (mptLoopClosing && mptLoopClosing->joinable()) {
        mptLoopClosing->join();
    }
#ifdef XPLSLAM_WITH_PANGOLIN_VIEWER
    if (mptViewer && mptViewer->joinable()) {
        mptViewer->join();
    }
#endif

    if (!mStrSaveAtlasToFile.empty()) {
        Verbose::PrintMess("Atlas saving to file " + mStrSaveAtlasToFile,
                           Verbose::VERBOSITY_NORMAL);
        SaveAtlas(FileType::BINARY_FILE);
    }
}

bool System::isShutDown() { return mbShutDown; }

int System::GetTrackingState() {
    unique_lock<mutex> lock(mMutexState);
    return mTrackingState;
}

std::vector<MapPointPtr> System::GetTrackedMapPoints() {
    unique_lock<mutex> lock(mMutexState);
    std::vector<MapPointPtr> points;
    for (const auto &cam_points : mTrackedMapPoints) {
        points.insert(points.end(), cam_points.begin(), cam_points.end());
    }
    return points;
}

std::vector<MapLinePtr> System::GetTrackedMapLines() {
    unique_lock<mutex> lock(mMutexState);
    std::vector<MapLinePtr> lines;
    for (const auto &cam_lines : mTrackedMapLines) {
        lines.insert(lines.end(), cam_lines.begin(), cam_lines.end());
    }
    return lines;
}

std::vector<std::vector<cv::Point2f>> System::GetTrackedKeyPointsUn() {
    unique_lock<mutex> lock(mMutexState);
    return mTrackedKeyPointsUn;
}

std::vector<std::vector<Eigen::Vector4f>> System::GetTrackedKeyLinesUn() {
    unique_lock<mutex> lock(mMutexState);
    return mTrackedKeyLinesUn;
}

bool System::isLost() {
    unique_lock<mutex> lock(mMutexState);
    return mTrackingState == Tracking::LOST;
}

bool System::isFinished() { return mbShutDown; }

float System::GetImageScale() { return 1.0f; }

long unsigned int System::GetMapPointCount() {
    return mpAtlas ? mpAtlas->MapPointsInMap() : 0;
}

MapSparsification::Statistics System::GetMapSparsificationStatistics() {
    return mpMapSparsification ? mpMapSparsification->GetStatistics()
                               : MapSparsification::Statistics();
}

void System::SaveTrajectoryTUM(const string &filename) {
    cout << endl
         << "Saving camera trajectory to " << filename << " ..." << endl;

    vector<KeyFramePtr> vpKFs = mpAtlas->GetAllKeyFrames();
    if (vpKFs.empty()) {
        cerr << "ERROR: no keyframes in map, trajectory not saved." << endl;
        return;
    }
    sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

    const Sophus::SE3f Two = vpKFs.front()->GetPoseInverse();

    ofstream f(filename.c_str());
    if (!f.is_open()) {
        cerr << "ERROR: failed to open trajectory file " << filename << endl;
        return;
    }
    f << fixed;

    list<KeyFramePtr>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();

    for (list<Sophus::SE3f>::iterator
             lit = mpTracker->mlRelativeFramePoses.begin(),
             lend = mpTracker->mlRelativeFramePoses.end();
         lit != lend && lRit != mpTracker->mlpReferences.end() &&
         lT != mpTracker->mlFrameTimes.end() && lbL != mpTracker->mlbLost.end();
         ++lit, ++lRit, ++lT, ++lbL) {
        if (*lbL || !(*lRit)) {
            continue;
        }

        KeyFramePtr pKF = *lRit;
        Sophus::SE3f Trw;

        while (pKF && pKF->isBad()) {
            Trw = Trw * pKF->mTbp;
            pKF = pKF->GetParent();
        }
        if (!pKF) {
            continue;
        }

        Trw = Trw * pKF->GetPose() * Two;
        const Sophus::SE3f Tcw = (*lit) * Trw;
        const Sophus::SE3f Twc = Tcw.inverse();

        const Eigen::Vector3f twc = Twc.translation();
        const Eigen::Quaternionf q = Twc.unit_quaternion();

        f << setprecision(6) << *lT << " " << setprecision(9) << twc.x() << " "
          << twc.y() << " " << twc.z() << " " << q.x() << " " << q.y() << " "
          << q.z() << " " << q.w() << endl;
    }
    f.close();
}

void System::SaveAtlas(int type) { return; }

}; // namespace XPLSLAM
