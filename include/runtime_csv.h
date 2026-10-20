#pragma once

#include "runtime_timing.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <string>

namespace XPLSLAM {
class RuntimeCsv {
public:
    explicit RuntimeCsv(const std::filesystem::path &directory) : directory_(directory) {
        if (directory_.empty()) return;
        for (const char *name : {"frames.csv", "calls.csv", "run_totals.json"})
            if (std::filesystem::exists(directory_ / name))
                throw std::runtime_error(std::string("Runtime output already exists: ") + name);
        frames_.open(directory_ / "frames.csv");
        calls_.open(directory_ / "calls.csv");
        frames_.exceptions(std::ios::failbit | std::ios::badbit);
        calls_.exceptions(std::ios::failbit | std::ios::badbit);
        frames_ << std::setprecision(17)
            << "input_index,timestamp,frame_id,state_before,state_after,has_pose,keyframe_created,initialization_calls,relocalization_calls,track_ms,base_features_ms,sparse_lines_ms,matching_ms,selection_ms,pose_ms,other_ms,partition_error_ms,read_ms,playback_wait_ms,line_extraction_calls,first_line_reason,network_pair_slots,network_valid_pairs,network_before_dedup,detected_lines,evaluation_calls,weak_point_calls,selection_calls,candidates_sum,selected_sum,pose_calls,line_edges_added_sum,line_residuals_used_sum,active_line_round_sum,solve_rounds,solve_iterations,failed_solve_rounds,upnpl_calls,upnpl_line_observations_sum,pose_solver_uses_lines\n";
        calls_ << "input_index,frame_id,call_index,kind,candidates,selected,points,point_strong,line_edges_added,line_residuals_used,active_line_round_sum,solve_rounds,solve_iterations,failed_solve_rounds\n";
        frames_.flush(); calls_.flush(); // Retain schema even if startup aborts.
    }
    void Append(size_t input_index, double timestamp, const RuntimeFrame &frame,
                double track_ms, double read_ms, double wait_ms) {
        if (directory_.empty()) return;
        long long evaluation=0, weak=0, selection=0, candidates=0, selected=0;
        long long pose=0, added=0, used=0, round_sum=0, rounds=0, iterations=0;
        long long upnpl=0, upnpl_lines=0, failed=0;
        for (size_t i=0; i<frame.calls.size(); ++i) {
            const auto &c=frame.calls[i];
            const std::string kind(c.kind);
            if (kind=="evaluation") { ++evaluation; weak += c.point_strong==0; }
            if (kind=="selection") { ++selection; candidates+=c.candidates; selected+=c.selected; }
            if (kind=="pose") {
                ++pose; added+=c.line_edges_added; used+=c.line_residuals_used;
                round_sum+=c.active_line_round_sum; rounds+=c.solve_rounds;
                iterations+=c.solve_iterations; failed+=c.failed_solve_rounds;
            }
            if (kind.rfind("upnpl_",0)==0) { ++upnpl; upnpl_lines+=c.selected; }
            calls_ << input_index << ',' << frame.frame_id << ',' << i << ',' << c.kind
                << ',' << c.candidates << ',' << c.selected << ',' << c.points << ',' << c.point_strong
                << ',' << c.line_edges_added << ',' << c.line_residuals_used
                << ',' << c.active_line_round_sum << ',' << c.solve_rounds << ',' << c.solve_iterations << ',' << c.failed_solve_rounds << '\n';
        }
        const double sum=std::accumulate(frame.phase_ms.begin(),frame.phase_ms.end(),0.0);
        if (!std::isfinite(track_ms) || std::abs(sum-track_ms)>1e-6)
            throw std::runtime_error("Runtime phase accounting does not close");
        for (double ms : frame.phase_ms)
            if (!std::isfinite(ms) || ms<0) throw std::runtime_error("Invalid runtime phase duration");
        frames_ << input_index << ',' << timestamp << ',' << frame.frame_id
            << ',' << frame.state_before << ',' << frame.state_after << ',' << frame.has_pose
            << ',' << frame.keyframe_created << ',' << frame.initialization_calls
            << ',' << frame.relocalization_calls << ',' << track_ms;
        for (double ms : frame.phase_ms) frames_ << ',' << ms;
        frames_ << ',' << sum-track_ms << ',' << read_ms << ',' << wait_ms
            << ',' << frame.line_extraction_calls << ',' << RuntimeReasonName(frame.first_line_reason)
            << ',' << frame.network_pair_slots << ',' << frame.network_valid_pairs << ',' << frame.network_before_dedup << ',' << frame.detected_lines
            << ',' << evaluation << ',' << weak << ',' << selection << ',' << candidates << ',' << selected
            << ',' << pose << ',' << added << ',' << used << ',' << round_sum << ',' << rounds << ',' << iterations
            << ',' << failed << ',' << upnpl << ',' << upnpl_lines << ',' << (used>0 || upnpl_lines>0) << '\n';
        ++rows_;
    }
    void Finish(size_t expected, size_t processed, double startup_s, double loop_s,
                double shutdown_s, double save_s) {
        if (directory_.empty()) return;
        if (rows_!=processed) throw std::runtime_error("Runtime CSV row count mismatch");
        frames_.flush(); calls_.flush();
        std::ofstream out(directory_ / "run_totals.json");
        out.exceptions(std::ios::failbit | std::ios::badbit);
        out << std::setprecision(12) << "{\n  \"available_input_frames\": " << expected
            << ",\n  \"processed_frames\": " << processed
            << ",\n  \"system_startup_s\": " << startup_s
            << ",\n  \"frame_loop_wall_s\": " << loop_s
            << ",\n  \"shutdown_s\": " << shutdown_s
            << ",\n  \"trajectory_save_s\": " << save_s
            << ",\n  \"warmup\": \"none; all input frames retained\"\n}\n";
    }
private:
    std::filesystem::path directory_;
    std::ofstream frames_, calls_;
    size_t rows_=0;
};
} // namespace XPLSLAM
