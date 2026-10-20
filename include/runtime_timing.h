#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace XPLSLAM {

// Optional wall-clock accounting for the thread calling TrackMultiCamera.
// A child phase pauses its parent; asynchronous mapping threads have no sink.
// No CUDA events or additional device synchronization are introduced.
enum class RuntimePhase { BaseFeatures, SparseLines, Matching, Selection, Pose, Other };
enum class RuntimeLineReason { Unspecified, Initialization, WeakPoints, KeyFrame, Relocalization, Always };
inline const char *RuntimeReasonName(RuntimeLineReason reason) {
    static const char *names[] = {"unspecified", "initialization", "weak_points",
                                 "keyframe", "relocalization", "always"};
    return names[static_cast<size_t>(reason)];
}

struct RuntimeCall {
    const char *kind = "";
    long long candidates = 0, selected = 0, points = 0;
    int point_strong = -1;
    long long line_edges_added = 0, line_residuals_used = 0;
    long long active_line_round_sum = 0, solve_rounds = 0, solve_iterations = 0;
    long long failed_solve_rounds = 0;
};

struct RuntimeFrame {
    using Clock = std::chrono::steady_clock;
    std::array<double, 6> phase_ms{};
    unsigned long frame_id = 0;
    int state_before = -1, state_after = -1;
    bool has_pose = false, keyframe_created = false;
    int initialization_calls = 0, relocalization_calls = 0;
    int line_extraction_calls = 0;
    RuntimeLineReason first_line_reason = RuntimeLineReason::Unspecified;
    // Network pair slots include invalid/padded pairs; no extra GPU reduction.
    long long network_pair_slots = 0, network_valid_pairs = 0, network_before_dedup = 0;
    long long detected_lines = 0;
    std::vector<RuntimeCall> calls;
    RuntimePhase phase = RuntimePhase::Other;
    RuntimeLineReason reason = RuntimeLineReason::Unspecified;
    Clock::time_point last;
    void Switch(RuntimePhase next, Clock::time_point now = Clock::now()) {
        phase_ms[static_cast<size_t>(phase)] +=
            std::chrono::duration<double, std::milli>(now - last).count();
        last = now;
        phase = next;
    }
};

inline thread_local RuntimeFrame *runtime_frame = nullptr;

class RuntimeCapture {
public:
    explicit RuntimeCapture(RuntimeFrame *frame) : frame_(frame) {
        if (frame_ && runtime_frame)
            throw std::logic_error("Nested runtime frame capture");
        if (frame_) {
            // Reserve outside the TrackMultiCamera interval.
            frame_->calls.reserve(32);
            runtime_frame = frame_;
        }
    }
    void Begin(RuntimeFrame::Clock::time_point begin) {
        if (frame_) frame_->last = begin;
    }
    void End(RuntimeFrame::Clock::time_point end) {
        if (frame_) frame_->Switch(RuntimePhase::Other, end);
        if (frame_) runtime_frame = nullptr;
    }
    ~RuntimeCapture() { if (frame_) runtime_frame = nullptr; }
    RuntimeCapture(const RuntimeCapture &) = delete;
    RuntimeCapture &operator=(const RuntimeCapture &) = delete;
private:
    RuntimeFrame *frame_;
};

class RuntimePhaseScope {
public:
    explicit RuntimePhaseScope(RuntimePhase phase) : frame_(runtime_frame) {
        if (frame_) { previous_ = frame_->phase; frame_->Switch(phase); }
    }
    ~RuntimePhaseScope() { if (frame_) frame_->Switch(previous_); }
    RuntimePhaseScope(const RuntimePhaseScope &) = delete;
    RuntimePhaseScope &operator=(const RuntimePhaseScope &) = delete;
private:
    RuntimeFrame *frame_;
    RuntimePhase previous_ = RuntimePhase::Other;
};

class RuntimeReasonScope {
public:
    explicit RuntimeReasonScope(RuntimeLineReason reason) : frame_(runtime_frame) {
        if (frame_) {
            previous_ = frame_->reason;
            // Outermost reason wins, e.g. a weak-point fallback during relocalization.
            if (previous_ == RuntimeLineReason::Unspecified) frame_->reason = reason;
        }
    }
    ~RuntimeReasonScope() { if (frame_) frame_->reason = previous_; }
private:
    RuntimeFrame *frame_;
    RuntimeLineReason previous_ = RuntimeLineReason::Unspecified;
};

// Local record survives early returns; slots preserve invocation order, including nesting.
struct RuntimeCallScope {
    explicit RuntimeCallScope(const char *kind) : frame(runtime_frame) {
        call.kind = kind;
        if (frame) { index = frame->calls.size(); frame->calls.emplace_back(); }
    }
    ~RuntimeCallScope() { if (frame) frame->calls[index] = call; }
    size_t index = 0;
    RuntimeFrame *frame;
    RuntimeCall call;
};
} // namespace XPLSLAM
