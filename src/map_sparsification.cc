#include "map_sparsification.h"

#include "atlas.h"
#include "keyframe.h"
#include "loop_closing.h"
#include "map.h"
#include "map_point.h"

#ifdef XPLSLAM_WITH_MAP_SPARSIFICATION
#include "scippp/lin_expr.hpp"
#include "scippp/model.hpp"
#include "scippp/parameters.hpp"
#include "scippp/solution.hpp"
#include "scippp/var.hpp"
#include "scippp/var_type.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>

using namespace std;

namespace XPLSLAM {

namespace {

struct KeyFrameSnapshot {
    KeyFramePtr keyframe;
    vector<vector<MapPointPtr>> occupied_cells;
    vector<MapPointPtr> unique_points;
};

struct SolverSnapshot {
    vector<KeyFrameSnapshot> keyframes;
    vector<MapPointPtr> points;
    vector<int> observations;
    unordered_map<MapPointPtr, size_t> point_indices;
    map<KeyFramePtr, vector<size_t>> external_constraints;
};

SolverSnapshot BuildSnapshot(const vector<KeyFramePtr> &keyframes) {
    SolverSnapshot snapshot;
    unordered_set<KeyFramePtr> window_keyframes;
    window_keyframes.reserve(keyframes.size());
    for (const KeyFramePtr &keyframe : keyframes) {
        if (keyframe && !keyframe->isBad())
            window_keyframes.insert(keyframe);
    }

    for (const KeyFramePtr &keyframe : keyframes) {
        if (!keyframe || keyframe->isBad())
            continue;

        KeyFrameSnapshot keyframe_snapshot;
        keyframe_snapshot.keyframe = keyframe;
        const vector<vector<MapPointPtr>> matches =
            keyframe->GetMapPointMatches();
        const auto grids = keyframe->GetPointFeatureGrids();
        unordered_set<MapPointPtr> unique_points;

        const size_t camera_count = min(matches.size(), grids.size());
        for (size_t cam = 0; cam < camera_count; ++cam) {
            for (const auto &row : grids[cam]) {
                for (const vector<size_t> &cell : row) {
                    vector<MapPointPtr> occupied;
                    unordered_set<MapPointPtr> cell_points;
                    for (size_t index : cell) {
                        if (index >= matches[cam].size())
                            continue;
                        const MapPointPtr &point = matches[cam][index];
                        if (!point || point->isBad() ||
                            !cell_points.insert(point).second) {
                            continue;
                        }
                        occupied.push_back(point);
                        unique_points.insert(point);
                    }
                    if (!occupied.empty())
                        keyframe_snapshot.occupied_cells.push_back(
                            std::move(occupied));
                }
            }
        }
        keyframe_snapshot.unique_points.assign(unique_points.begin(),
                                               unique_points.end());
        snapshot.keyframes.push_back(std::move(keyframe_snapshot));
    }

    for (const KeyFrameSnapshot &keyframe : snapshot.keyframes) {
        for (const MapPointPtr &point : keyframe.unique_points) {
            if (snapshot.point_indices.count(point))
                continue;
            const size_t index = snapshot.points.size();
            snapshot.point_indices.emplace(point, index);
            snapshot.points.push_back(point);
            snapshot.observations.push_back(point->Observations());
        }
    }

    for (size_t point_index = 0; point_index < snapshot.points.size();
         ++point_index) {
        const MapPointPtr &point = snapshot.points[point_index];
        const auto observations = point->GetObservations();
        for (const auto &[keyframe, indices] : observations) {
            (void)indices;
            if (!keyframe || keyframe->isBad() ||
                window_keyframes.count(keyframe) ||
                keyframe->GetMap() != point->GetMap()) {
                continue;
            }
            snapshot.external_constraints[keyframe].push_back(point_index);
        }
    }
    return snapshot;
}

} // namespace

MapSparsification::MapSparsification(
    const MapSparsificationConfig &config, Atlas *atlas)
    : config_(config), atlas_(atlas) {}

void MapSparsification::Run() {
    {
        lock_guard<mutex> lock(mutex_);
        finished_ = false;
        stopped_ = true;
    }

    while (true) {
        vector<KeyFramePtr> window;
        {
            unique_lock<mutex> lock(mutex_);
            condition_.wait(lock, [this] {
                return finish_requested_ || (stop_requests_ == 0 &&
                        keyframes_.size() > static_cast<size_t>(10));
            });

            if (finish_requested_ && keyframes_.empty())
                break;
            if (stop_requests_ > 0 && !finish_requested_) {
                stopped_ = true;
                condition_.notify_all();
                continue;
            }
            if (!HasWork(finish_requested_))
                continue;

            const size_t count =
                finish_requested_
                    ? min(keyframes_.size(),
                          static_cast<size_t>(config_.window_length))
                    : static_cast<size_t>(config_.window_length);
            window.reserve(count);
            for (size_t i = 0; i < count && !keyframes_.empty(); ++i) {
                window.push_back(keyframes_.front());
                keyframes_.pop_front();
            }
            stopped_ = false;
        }

        const WindowResult result = Sparsify(window);
        {
            lock_guard<mutex> lock(mutex_);
            if (result.solved) {
                ++statistics_.solved_windows;
            } else {
                ++statistics_.failed_windows;
                for (const KeyFramePtr &keyframe : window)
                    submitted_keyframes_.erase(KeyFrameWeakPtr(keyframe));
            }
            statistics_.candidate_points += result.candidates;
            statistics_.deleted_points += result.deleted;
            statistics_.total_solve_ms += result.solve_ms;
            stopped_ = true;
            condition_.notify_all();
        }
    }

    SetFinished();
}

void MapSparsification::InsertKeyFrame(const KeyFramePtr &keyframe) {
    if (!keyframe || keyframe->isBad() || keyframe->isSparsified())
        return;
    lock_guard<mutex> lock(mutex_);
    if (finish_requested_)
        return;
    for (auto it = submitted_keyframes_.begin();
         it != submitted_keyframes_.end();) {
        if (it->expired())
            it = submitted_keyframes_.erase(it);
        else
            ++it;
    }
    if (!submitted_keyframes_.insert(KeyFrameWeakPtr(keyframe)).second)
        return;
    keyframes_.push_back(keyframe);
    statistics_.queued_keyframes = keyframes_.size();
    condition_.notify_all();
}

void MapSparsification::SetLoopClosing(LoopClosing *loop_closing) {
    lock_guard<mutex> lock(mutex_);
    loop_closing_ = loop_closing;
}

bool MapSparsification::HasWork(bool flush_all) const {
    return flush_all ? !keyframes_.empty()
                     : keyframes_.size() > static_cast<size_t>(10);
}

MapSparsification::WindowResult
MapSparsification::Sparsify(const vector<KeyFramePtr> &keyframes) {
    WindowResult result;
    if (keyframes.empty())
        return result;

#ifndef XPLSLAM_WITH_MAP_SPARSIFICATION
    return result;
#else
    const SolverSnapshot snapshot = BuildSnapshot(keyframes);
    result.candidates = snapshot.points.size();
    if (snapshot.points.empty()) {
        result.solved = true;
        LoopClosing *loop_closing = nullptr;
        {
            lock_guard<mutex> lock(mutex_);
            loop_closing = loop_closing_;
        }
        if (loop_closing) {
            for (const KeyFramePtr &keyframe : keyframes)
                loop_closing->InsertSparsifiedKeyFrame(keyframe);
        }
        return result;
    }

    const auto solve_begin = chrono::steady_clock::now();
    try {
        scippp::Model model("xplslam_map_sparsification");
        model.setObjsense(scippp::Sense::MINIMIZE);
        model.setParam(scippp::params::DISPLAY::VERBLEVEL, 0);
        model.setParam(scippp::params::LIMITS::GAP, config_.mip_gap);
        model.setParam(scippp::params::PARALLEL::MAXNTHREADS,
                       config_.solver_threads);

        const int max_observations =
            *max_element(snapshot.observations.begin(),
                         snapshot.observations.end());
        vector<scippp::Var> point_variables;
        point_variables.reserve(snapshot.points.size());
        for (size_t i = 0; i < snapshot.points.size(); ++i) {
            const double coefficient =
                max_observations - snapshot.observations[i];
            point_variables.push_back(model.addVar(
                "point_" + to_string(i), coefficient,
                scippp::VarType::BINARY, 0.0, 1.0));
        }

        size_t constraint_index = 0;
        for (const KeyFrameSnapshot &keyframe : snapshot.keyframes) {
            scippp::LinExpr keyframe_expression;
            for (const MapPointPtr &point : keyframe.unique_points) {
                keyframe_expression +=
                    point_variables[snapshot.point_indices.at(point)];
            }
            const scippp::Var keyframe_slack = model.addVar(
                "keyframe_slack_" + to_string(constraint_index),
                config_.lambda, scippp::VarType::INTEGER, 0.0, 1000.0);
            keyframe_expression += keyframe_slack;
            model.addConstr(
                keyframe_expression >= config_.min_points_per_keyframe,
                "keyframe_min_" + to_string(constraint_index));

            size_t cell_index = 0;
            for (const vector<MapPointPtr> &cell :
                 keyframe.occupied_cells) {
                scippp::LinExpr cell_expression;
                for (const MapPointPtr &point : cell) {
                    cell_expression +=
                        point_variables[snapshot.point_indices.at(point)];
                }
                const scippp::Var cell_slack = model.addVar(
                    "grid_slack_" + to_string(constraint_index) + "_" +
                        to_string(cell_index),
                    config_.grid_lambda, scippp::VarType::BINARY, 0.0, 1.0);
                cell_expression += cell_slack;
                model.addConstr(
                    cell_expression >= 1.0,
                    "grid_min_" + to_string(constraint_index) + "_" +
                        to_string(cell_index));
                ++cell_index;
            }
            ++constraint_index;
        }

        size_t external_index = 0;
        for (const auto &[keyframe, point_indices] :
             snapshot.external_constraints) {
            const int total_points = keyframe->GetNumberMPs();
            if (total_points <= 0 || point_indices.empty())
                continue;
            scippp::LinExpr expression;
            for (size_t point_index : point_indices)
                expression += point_variables[point_index];
            const double minimum =
                static_cast<double>(point_indices.size()) /
                static_cast<double>(total_points) *
                config_.min_points_per_keyframe;
            const scippp::Var slack = model.addVar(
                "external_slack_" + to_string(external_index),
                config_.lambda, scippp::VarType::INTEGER, 0.0, 1000.0);
            expression += slack;
            model.addConstr(expression >= minimum,
                            "external_min_" + to_string(external_index));
            ++external_index;
        }

        model.solve();
        result.solve_ms =
            chrono::duration<double, milli>(chrono::steady_clock::now() -
                                            solve_begin)
                .count();
        if (model.getNSols() <= 0)
            return result;

        const scippp::Solution solution = model.getBestSol();
        unordered_map<Map *, vector<MapPointPtr>> points_to_delete_by_map;
        vector<MapPointPtr> points_without_map;
        for (size_t i = 0; i < snapshot.points.size(); ++i) {
            if (!point_variables[i].isZero(solution))
                continue;
            const MapPointPtr &point = snapshot.points[i];
            if (!point || point->isBad())
                continue;
            Map *map = point->GetMap();
            if (map) {
                points_to_delete_by_map[map].push_back(point);
            } else {
                points_without_map.push_back(point);
            }
        }

        for (auto &[map, points] : points_to_delete_by_map) {
            unique_lock<mutex> map_update_lock(map->mMutexMapUpdate);
            for (const MapPointPtr &point : points) {
                if (!point || point->isBad())
                    continue;
                point->SetBadFlag();
                ++result.deleted;
            }
        }
        for (const MapPointPtr &point : points_without_map) {
            if (!point || point->isBad())
                continue;
            point->SetBadFlag();
            ++result.deleted;
        }
        result.solved = true;

        LoopClosing *loop_closing = nullptr;
        {
            lock_guard<mutex> lock(mutex_);
            loop_closing = loop_closing_;
        }
        if (loop_closing) {
            for (const KeyFramePtr &keyframe : keyframes) {
                if (keyframe && !keyframe->isBad())
                    loop_closing->InsertSparsifiedKeyFrame(keyframe);
            }
        }
    } catch (const exception &error) {
        result.solve_ms =
            chrono::duration<double, milli>(chrono::steady_clock::now() -
                                            solve_begin)
                .count();
        cerr << "Map sparsification failed: " << error.what() << endl;
    }
    return result;
#endif
}

void MapSparsification::RequestStop() {
    lock_guard<mutex> lock(mutex_);
    ++stop_requests_;
    condition_.notify_all();
}

void MapSparsification::Release() {
    lock_guard<mutex> lock(mutex_);
    if (stop_requests_ > 0)
        --stop_requests_;
    condition_.notify_all();
}

bool MapSparsification::isStopped() {
    lock_guard<mutex> lock(mutex_);
    return stopped_;
}

void MapSparsification::RequestReset() {
    unique_lock<mutex> lock(mutex_);
    ++stop_requests_;
    condition_.notify_all();
    condition_.wait(lock, [this] { return stopped_; });
    keyframes_.clear();
    submitted_keyframes_.clear();
    --stop_requests_;
    condition_.notify_all();
}

void MapSparsification::RequestResetActiveMap(Map *map) {
    unique_lock<mutex> lock(mutex_);
    ++stop_requests_;
    condition_.notify_all();
    condition_.wait(lock, [this] { return stopped_; });
    for (auto it = keyframes_.begin(); it != keyframes_.end();) {
        if (!*it || (*it)->GetMap() == map)
            it = keyframes_.erase(it);
        else
            ++it;
    }
    for (auto it = submitted_keyframes_.begin();
         it != submitted_keyframes_.end();) {
        const KeyFramePtr keyframe = it->lock();
        if (!keyframe || keyframe->GetMap() == map)
            it = submitted_keyframes_.erase(it);
        else
            ++it;
    }
    --stop_requests_;
    condition_.notify_all();
}

void MapSparsification::RequestFinish() {
    const vector<KeyFramePtr> all_keyframes =
        atlas_ ? atlas_->GetAllKeyFrames() : vector<KeyFramePtr>();
    lock_guard<mutex> lock(mutex_);
    for (const KeyFramePtr &keyframe : all_keyframes) {
        if (keyframe && !keyframe->isBad() && !keyframe->isSparsified() &&
            submitted_keyframes_.insert(KeyFrameWeakPtr(keyframe)).second) {
            keyframes_.push_back(keyframe);
        }
    }
    finish_requested_ = true;
    condition_.notify_all();
}

bool MapSparsification::isFinished() {
    lock_guard<mutex> lock(mutex_);
    return finished_;
}

MapSparsification::Statistics MapSparsification::GetStatistics() const {
    lock_guard<mutex> lock(mutex_);
    Statistics statistics = statistics_;
    statistics.queued_keyframes = keyframes_.size();
    return statistics;
}

void MapSparsification::NotifyKeyFrameCompacted() {
    lock_guard<mutex> lock(mutex_);
    ++statistics_.compacted_keyframes;
}

void MapSparsification::SetFinished() {
    lock_guard<mutex> lock(mutex_);
    stopped_ = true;
    finished_ = true;
    condition_.notify_all();
}

} // namespace XPLSLAM
