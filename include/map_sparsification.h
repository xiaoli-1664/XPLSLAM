#pragma once

#include "settings.h"
#include "slam_types.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <vector>

namespace XPLSLAM {

class Atlas;
class LoopClosing;
class Map;

class MapSparsification {
  public:
    struct Statistics {
        std::uint64_t solved_windows = 0;
        std::uint64_t failed_windows = 0;
        std::uint64_t candidate_points = 0;
        std::uint64_t deleted_points = 0;
        std::uint64_t compacted_keyframes = 0;
        double total_solve_ms = 0.0;
        std::size_t queued_keyframes = 0;
    };

    MapSparsification(const MapSparsificationConfig &config, Atlas *atlas);

    void Run();
    void InsertKeyFrame(const KeyFramePtr &keyframe);
    void SetLoopClosing(LoopClosing *loop_closing);

    void RequestStop();
    void Release();
    bool isStopped();

    void RequestReset();
    void RequestResetActiveMap(Map *map);

    void RequestFinish();
    bool isFinished();

    Statistics GetStatistics() const;
    void NotifyKeyFrameCompacted();

  private:
    struct WindowResult {
        bool solved = false;
        std::size_t candidates = 0;
        std::size_t deleted = 0;
        double solve_ms = 0.0;
    };

    WindowResult Sparsify(const std::vector<KeyFramePtr> &keyframes);
    bool HasWork(bool flush_all) const;
    void SetFinished();

    const MapSparsificationConfig config_;
    Atlas *atlas_;
    LoopClosing *loop_closing_ = nullptr;

    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<KeyFramePtr> keyframes_;
    std::set<KeyFrameWeakPtr, std::owner_less<KeyFrameWeakPtr>>
        submitted_keyframes_;
    int stop_requests_ = 0;
    bool stopped_ = true;
    bool finish_requested_ = false;
    bool finished_ = true;
    Statistics statistics_;
};

} // namespace XPLSLAM
