#pragma once

namespace XPLSLAM {

struct XPLNetLinePostprocessTiming {
    long long junction_slots_per_image = 0;
    long long candidate_pairs_per_image = 0;
    long long accepted_before_dedup = 0;
    long long accepted_after_dedup = 0;
    double candidate_generation_ms = 0.0;
    double verification_preprocess_ms = 0.0;
    double verification_trt_enqueue_ms = 0.0;
    double filtering_dedup_ms = 0.0;
    double line_descriptor_preprocess_ms = 0.0;
    double line_descriptor_trt_enqueue_ms = 0.0;
    double result_packaging_ms = 0.0;
    double total_ms = 0.0;
};

struct XPLNetPostprocessTiming {
    double output_access_ms = 0.0;
    double point_junction_decode_ms = 0.0;
    double line_processing_ms = 0.0;
    double point_descriptor_sampling_ms = 0.0;
    double result_packaging_ms = 0.0;
    XPLNetLinePostprocessTiming line;
    double total_ms = 0.0;
};

struct XPLNetTimingBreakdown {
    double preprocess_ms = 0.0;
    double frontend_trt_enqueue_ms = 0.0;
    double postprocess_ms = 0.0;
    double final_stream_sync_ms = 0.0;
    XPLNetPostprocessTiming postprocess;
    double total_ms = 0.0;
};

struct FrameConstructionTiming {
    double setup_ms = 0.0;
    double xplnet_detect_ms = 0.0;
    XPLNetTimingBreakdown xplnet;
    double feature_materialization_ms = 0.0;
    double initial_computations_ms = 0.0;
    double descriptor_flatten_ms = 0.0;
    double grid_assignment_ms = 0.0;
    double multicamera_matching_ms = 0.0;
    double deferred_line_ms = 0.0;
    double deferred_line_stereo_ms = 0.0;
    double total_ms = 0.0;
};

struct LocalMapSearchTiming {
    double current_feature_cleanup_ms = 0.0;
    double point_frustum_check_ms = 0.0;
    double line_frustum_check_ms = 0.0;
    double projection_matching_ms = 0.0;
    double total_ms = 0.0;
};

struct LocalMapTrackingTiming {
    double reference_map_update_ms = 0.0;
    double local_keyframes_update_ms = 0.0;
    double local_points_update_ms = 0.0;
    double local_lines_update_ms = 0.0;
    double search_local_features_ms = 0.0;
    LocalMapSearchTiming search;
    double pose_optimization_ms = 0.0;
    double inlier_update_ms = 0.0;
    double acceptance_check_ms = 0.0;
    double total_ms = 0.0;
};

struct TrackTimingBreakdown {
    double step_wait_ms = 0.0;
    double preprocess_ms = 0.0;
    double map_lock_wait_ms = 0.0;
    double map_update_check_ms = 0.0;
    double initialization_ms = 0.0;
    double pose_tracking_ms = 0.0;
    double point_constraint_analysis_ms = 0.0;
    // EvaluatePointConstraints only; excludes pose optimization.
    double point_constraint_evaluation_ms = 0.0;
    double deferred_line_tracking_ms = 0.0;
    double line_selection_optimization_ms = 0.0;
    // SelectLineConstraints only; included in the combined timer above.
    double line_selection_ms = 0.0;
    double local_map_tracking_ms = 0.0;
    LocalMapTrackingTiming local_map;
    double post_tracking_ms = 0.0;
    double trajectory_record_ms = 0.0;
    double total_ms = 0.0;
};

struct MultiCameraTrackingTiming {
    long unsigned int frame_id = 0;
    double input_setup_ms = 0.0;
    double frame_construction_ms = 0.0;
    FrameConstructionTiming frame;
    TrackTimingBreakdown track;
    double output_pose_ms = 0.0;
    double total_ms = 0.0;
};

struct ProcessedFrameTiming {
    long unsigned int frame_id = 0;
    double image_preparation_ms = 0.0;
    double mode_control_ms = 0.0;
    double reset_control_ms = 0.0;
    MultiCameraTrackingTiming tracker;
    double state_update_ms = 0.0;
    double total_ms = 0.0;
};

} // namespace XPLSLAM
