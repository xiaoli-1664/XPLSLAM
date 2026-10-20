cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER /usr/bin/gcc)
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER /usr/bin/g++)
endif()

set(XPLSLAM_CUDA_TOOLKIT_ROOT_DIR
    /usr/local/cuda
    CACHE PATH "CUDA toolkit root used by xplslam")
set(CUDA_TOOLKIT_ROOT_DIR ${XPLSLAM_CUDA_TOOLKIT_ROOT_DIR})
set(CUDA_NVCC_EXECUTABLE ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)
set(CMAKE_CUDA_COMPILER ${CUDA_NVCC_EXECUTABLE})
set(ENV{PATH} "${CUDA_TOOLKIT_ROOT_DIR}/bin:$ENV{PATH}")

project(xplslam)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(GRAPHITE "Build Graphite GPU optimizer backend support" ON)
option(XPLSLAM_ABLATION_UNIT_LINE_INFORMATION
       "Use identity information for every line residual"
       OFF)

set(TENSORRT_ROOT "" CACHE PATH "TensorRT installation prefix")
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT}/include REQUIRED)
find_library(TENSORRT_NVINFER_LIBRARY nvinfer
    HINTS ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib64 REQUIRED)
find_library(TENSORRT_ONNXPARSER_LIBRARY nvonnxparser
    HINTS ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib64 REQUIRED)
get_filename_component(TENSORRT_LIB_DIR "${TENSORRT_NVINFER_LIBRARY}" DIRECTORY)
# TensorRT's versioned builder-resource library is loaded at runtime.
set(TENSORRT_BUILDER_RESOURCE_LIB "")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(GRAPHITE)
    set(XPLSLAM_GRAPHITE_CUDA_ARCHITECTURES
        89
        CACHE STRING "CUDA architectures for xplslam Graphite builds")
    set(XPLSLAM_GRAPHITE_TORCH_CUDA_ARCH_LIST
        "8.9"
        CACHE STRING "Torch CUDA architectures for xplslam Graphite builds")
    if(NOT CMAKE_CUDA_ARCHITECTURES OR
       CMAKE_CUDA_ARCHITECTURES STREQUAL "52" OR
       CMAKE_CUDA_ARCHITECTURES STREQUAL "75")
        set(CMAKE_CUDA_ARCHITECTURES
            ${XPLSLAM_GRAPHITE_CUDA_ARCHITECTURES}
            CACHE STRING "CUDA architectures for xplslam Graphite builds"
            FORCE)
    endif()
    if(NOT TORCH_CUDA_ARCH_LIST OR TORCH_CUDA_ARCH_LIST STREQUAL "7.5")
        set(TORCH_CUDA_ARCH_LIST
            "${XPLSLAM_GRAPHITE_TORCH_CUDA_ARCH_LIST}"
            CACHE STRING "Torch CUDA architectures for xplslam Graphite builds"
            FORCE)
    endif()
    set(ENV{TORCH_CUDA_ARCH_LIST} "${TORCH_CUDA_ARCH_LIST}")
endif()

set(XPLSLAM_DEFAULT_OPENCV_DIR
    /usr/lib/x86_64-linux-gnu/cmake/opencv4
    CACHE PATH "Default OpenCV package directory used by xplslam")
if(NOT OpenCV_DIR AND EXISTS "${XPLSLAM_DEFAULT_OPENCV_DIR}/OpenCVConfig.cmake")
    set(OpenCV_DIR
        "${XPLSLAM_DEFAULT_OPENCV_DIR}"
        CACHE PATH "OpenCV package directory" FORCE)
endif()

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 3.3.9 REQUIRED)

find_package(yaml-cpp REQUIRED)

find_package(OpenSSL REQUIRED)

option(XPLSLAM_WITH_MAP_SPARSIFICATION
       "Build MS-SLAM map point sparsification support"
       ON)
if(XPLSLAM_WITH_MAP_SPARSIFICATION)
    function(xplslam_add_sparsification_solver)
        # Keep generic third-party options in this function scope so they do
        # not alter Pangolin, Graphite, or the project's own test options.
        set(ZLIB OFF)
        set(GMP OFF)
        set(MPFR OFF)
        set(PAPILO OFF)
        set(BOOST OFF)
        set(QUADMATH OFF)
        add_subdirectory(thirdparty/SoPlex EXCLUDE_FROM_ALL)

        set(SOPLEX_DIR "${CMAKE_BINARY_DIR}")
        set(SHARED ON)
        set(READLINE OFF)
        set(ZIMPL OFF)
        set(AMPL OFF)
        set(IPOPT OFF)
        set(LAPACK OFF)
        set(WORHP OFF)
        set(CONOPT OFF)
        set(EXACTSOLVE OFF)
        set(LPSEXACT none)
        set(SYM none)
        set(TPI none)
        set(LPS spx)
        set(BUILD_TESTING OFF)
        add_subdirectory(thirdparty/SCIP EXCLUDE_FROM_ALL)

        set(scip_DIR "${CMAKE_CURRENT_BINARY_DIR}/thirdparty/SCIP")
        set(BUILD_TESTS OFF)
        set(BUILD_UTILS OFF)
        add_subdirectory(thirdparty/SCIPpp EXCLUDE_FROM_ALL)
    endfunction()

    xplslam_add_sparsification_solver()
endif()

execute_process(
    COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_PREFIX_PATH
    RESULT_VARIABLE TORCH_PREFIX_RESULT
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(TORCH_PREFIX_RESULT EQUAL 0 AND TORCH_PREFIX_PATH)
    list(APPEND CMAKE_PREFIX_PATH "${TORCH_PREFIX_PATH}")
endif()

find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)
separate_arguments(XPLSLAM_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
add_compile_options(${XPLSLAM_TORCH_CXX_FLAGS})

include_directories(
${EIGEN3_INCLUDE_DIR}
${OpenCV_INCLUDE_DIRS}
${YAML_CPP_INCLUDE_DIRS}
)

include_directories(${TENSORRT_INCLUDE_DIR}
include
include/camera_models
${PROJECT_SOURCE_DIR}/thirdparty/Sophus
${PROJECT_SOURCE_DIR}/thirdparty/DBow3/src
${PROJECT_SOURCE_DIR}/thirdparty/g2o)

link_directories(${TENSORRT_LIB_DIR})

include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

add_subdirectory(thirdparty/DBow3)

add_subdirectory(thirdparty/upnpl)

option(XPLSLAM_BUILD_G2O_APPS "Build g2o apps" OFF)
option(XPLSLAM_BUILD_G2O_EXAMPLES "Build g2o examples" OFF)
option(XPLSLAM_BUILD_G2O_UNITTESTS "Build g2o unit tests" OFF)
option(XPLSLAM_BUILD_G2O_OPENGL "Build g2o OpenGL helpers" OFF)
set(G2O_BUILD_APPS
    ${XPLSLAM_BUILD_G2O_APPS}
    CACHE BOOL "Build g2o apps" FORCE)
set(G2O_BUILD_EXAMPLES
    ${XPLSLAM_BUILD_G2O_EXAMPLES}
    CACHE BOOL "Build g2o examples" FORCE)
set(BUILD_UNITTESTS
    ${XPLSLAM_BUILD_G2O_UNITTESTS}
    CACHE BOOL "Build g2o unit tests" FORCE)
set(G2O_USE_OPENGL
    ${XPLSLAM_BUILD_G2O_OPENGL}
    CACHE BOOL "Build g2o OpenGL support" FORCE)
set(G2O_BUILD_SLAM2D_TYPES OFF CACHE BOOL "Build g2o slam2d types" FORCE)
set(G2O_BUILD_SLAM2D_ADDON_TYPES
    OFF
    CACHE BOOL "Build g2o slam2d addon types" FORCE)
set(G2O_BUILD_DATA_TYPES OFF CACHE BOOL "Build g2o data types" FORCE)
set(G2O_BUILD_SCLAM2D_TYPES OFF CACHE BOOL "Build g2o sclam2d types" FORCE)
set(G2O_BUILD_SLAM3D_TYPES ON CACHE BOOL "Build g2o slam3d types" FORCE)
set(G2O_BUILD_SLAM3D_ADDON_TYPES
    OFF
    CACHE BOOL "Build g2o slam3d addon types" FORCE)
set(G2O_BUILD_SBA_TYPES ON CACHE BOOL "Build g2o sba types" FORCE)
set(G2O_BUILD_ICP_TYPES OFF CACHE BOOL "Build g2o icp types" FORCE)
set(G2O_BUILD_SIM3_TYPES ON CACHE BOOL "Build g2o sim3 types" FORCE)
# Torch bundles fmt headers; g2o's optional spdlog can select an incompatible
# system fmt version. Keep optimizer logging off without machine-specific paths.
set(G2O_USE_LOGGING OFF CACHE BOOL "Build g2o logging support" FORCE)
add_subdirectory(thirdparty/g2o)

option(XPLSLAM_BUILD_SOPHUS_TESTS "Build Sophus test targets" OFF)
option(XPLSLAM_BUILD_SOPHUS_PYTHON_BINDINGS
       "Build Sophus python bindings"
       OFF)
set(BUILD_SOPHUS_TESTS
    ${XPLSLAM_BUILD_SOPHUS_TESTS}
    CACHE BOOL "Build Sophus tests" FORCE)
set(BUILD_PYTHON_BINDINGS
    ${XPLSLAM_BUILD_SOPHUS_PYTHON_BINDINGS}
    CACHE BOOL "Build Sophus python bindings" FORCE)
add_subdirectory(thirdparty/Sophus)

option(XPLSLAM_ENABLE_PROFILING
       "Enable per-frame timing instrumentation and debug prints"
       ON)

option(XPLSLAM_WITH_PANGOLIN_VIEWER "Build Pangolin viewer support" ON)
if(XPLSLAM_WITH_PANGOLIN_VIEWER)
    set(BUILD_TOOLS OFF CACHE BOOL "Build Pangolin tools" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "Build Pangolin examples" FORCE)
    set(BUILD_TESTS OFF CACHE BOOL "Build Pangolin tests" FORCE)
    set(BUILD_PANGOLIN_PYTHON
        OFF
        CACHE BOOL "Build Pangolin Python support" FORCE)
    add_subdirectory(thirdparty/pangolin)
endif()

if(GRAPHITE)
    if(NOT CMAKE_CUDA_ARCHITECTURES OR
       CMAKE_CUDA_ARCHITECTURES STREQUAL "52" OR
       CMAKE_CUDA_ARCHITECTURES STREQUAL "75")
        set(CMAKE_CUDA_ARCHITECTURES
            ${XPLSLAM_GRAPHITE_CUDA_ARCHITECTURES}
            CACHE STRING "CUDA architectures for xplslam Graphite builds"
            FORCE)
    endif()
    set(XPLSLAM_DEFAULT_CUDSS_LIBRARY_DIR
        /usr/lib/x86_64-linux-gnu/libcudss/12
        CACHE PATH "Default cuDSS CUDA-12 library directory")
    if(NOT cudss_LIBRARY_DIR AND
       EXISTS "${XPLSLAM_DEFAULT_CUDSS_LIBRARY_DIR}/libcudss.so")
        set(cudss_LIBRARY_DIR
            "${XPLSLAM_DEFAULT_CUDSS_LIBRARY_DIR}"
            CACHE PATH "cuDSS library directory" FORCE)
    endif()

    enable_language(CUDA)
    set(CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda --default-stream per-thread")

    set(BUILD_EXAMPLES OFF CACHE BOOL "Build graphite examples" FORCE)
    set(BUILD_TESTS OFF CACHE BOOL "Build graphite tests" FORCE)
    set(BUILD_DOCS OFF CACHE BOOL "Build graphite documentation" FORCE)
    add_subdirectory(thirdparty/graphite)

    # This CUDA installation does not search /usr/include, while adding that
    # directory directly would shadow CUDA 12.6 headers with the distro CUDA
    # headers. Expose only Boost through a build-tree include root.
    set(XPLSLAM_CUDA_BOOST_INCLUDE_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/cuda_system_include")
    file(MAKE_DIRECTORY "${XPLSLAM_CUDA_BOOST_INCLUDE_DIR}")
    if(NOT EXISTS "${XPLSLAM_CUDA_BOOST_INCLUDE_DIR}/boost")
        file(CREATE_LINK
            "${Boost_INCLUDE_DIR}/boost"
            "${XPLSLAM_CUDA_BOOST_INCLUDE_DIR}/boost"
            SYMBOLIC
            COPY_ON_ERROR)
    endif()
    file(GLOB XPLSLAM_CUDSS_HEADERS "/usr/include/cudss*.h")
    foreach(XPLSLAM_CUDSS_HEADER IN LISTS XPLSLAM_CUDSS_HEADERS)
        get_filename_component(XPLSLAM_CUDSS_HEADER_NAME
            "${XPLSLAM_CUDSS_HEADER}" NAME)
        if(NOT EXISTS
           "${XPLSLAM_CUDA_BOOST_INCLUDE_DIR}/${XPLSLAM_CUDSS_HEADER_NAME}")
            file(CREATE_LINK
                "${XPLSLAM_CUDSS_HEADER}"
                "${XPLSLAM_CUDA_BOOST_INCLUDE_DIR}/${XPLSLAM_CUDSS_HEADER_NAME}"
                SYMBOLIC
                COPY_ON_ERROR)
        endif()
    endforeach()
    target_include_directories(graphite INTERFACE
        "$<$<COMPILE_LANGUAGE:CUDA>:${XPLSLAM_CUDA_BOOST_INCLUDE_DIR}>")
endif()

set(PROJECT_RUNTIME_RPATH
    "${TENSORRT_LIB_DIR};${CUDA_TOOLKIT_ROOT_DIR}/lib64;${TORCH_INSTALL_PREFIX}/lib")
set(CMAKE_BUILD_RPATH "${PROJECT_RUNTIME_RPATH}")

set(XPLSLAM_SOURCES
    src/atlas.cc
    src/converter.cc
    src/frame.cc
    src/g2o_types.cc
    src/geometric_tools.cc
    src/upnpl_solver.cc
    src/keyframe.cc
    src/keyframe_database.cc
    src/local_mapping.cc
    src/loop_closing.cc
    src/map_sparsification.cc
    src/map.cc
    src/map_line.cc
    src/map_point.cc
    src/matcher.cc
    src/optimizer.cc
    src/settings.cc
    src/SE3Solver.cc
    src/system.cc
    src/tracking.cc
    src/trt_engine.cc
    src/xplnet.cc
    src/camera_models/pinhole.cc)

if(GRAPHITE)
    list(APPEND XPLSLAM_SOURCES
        src/graphite_slam_backend.cc
        src/graphite_slam_backend_host.cc
        src/graphite_slam_backend.cu)
endif()

if(XPLSLAM_WITH_PANGOLIN_VIEWER)
    list(APPEND XPLSLAM_SOURCES
        src/frame_drawer.cc
        src/map_drawer.cc
        src/viewer.cc)
endif()

add_library(xplslam ${XPLSLAM_SOURCES})

if(XPLSLAM_ABLATION_UNIT_LINE_INFORMATION)
    target_compile_definitions(xplslam PUBLIC
        XPLSLAM_ABLATION_UNIT_LINE_INFORMATION)
endif()

if(XPLSLAM_ENABLE_PROFILING)
    target_compile_definitions(xplslam PUBLIC XPLSLAM_ENABLE_PROFILING)
endif()

if(XPLSLAM_WITH_PANGOLIN_VIEWER)
    target_compile_definitions(xplslam PUBLIC XPLSLAM_WITH_PANGOLIN_VIEWER)
endif()

target_link_libraries(xplslam
DBoW3
UPnPL
Sophus::Sophus
core
types_sba
types_slam3d
types_sim3
${TENSORRT_NVINFER_LIBRARY}
CUDA::cudart
${TENSORRT_ONNXPARSER_LIBRARY}
${TENSORRT_BUILDER_RESOURCE_LIB}
${OpenCV_LIBS}
yaml-cpp
OpenSSL::Crypto
${TORCH_LIBRARIES})

if(XPLSLAM_WITH_MAP_SPARSIFICATION)
    target_link_libraries(xplslam ScipPP)
    target_compile_definitions(xplslam PUBLIC
        XPLSLAM_WITH_MAP_SPARSIFICATION)
endif()

if(GRAPHITE)
    target_link_libraries(xplslam graphite)
    target_compile_definitions(xplslam PUBLIC GRAPHITE)
endif()

if(XPLSLAM_WITH_PANGOLIN_VIEWER)
    target_link_libraries(xplslam pango_display)
endif()

target_link_options(xplslam PRIVATE
-Wl,--disable-new-dtags)

add_executable(xplslam_new_college examples/new_college.cc)

target_compile_definitions(xplslam_new_college PRIVATE
PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

set_target_properties(xplslam_new_college PROPERTIES
BUILD_RPATH "${PROJECT_RUNTIME_RPATH}")

target_link_options(xplslam_new_college PRIVATE
-Wl,--disable-new-dtags)

target_link_libraries(xplslam_new_college
xplslam
${TENSORRT_BUILDER_RESOURCE_LIB})

option(XPLSLAM_BUILD_TESTS "Build SLAM system tests" OFF)
if(XPLSLAM_BUILD_TESTS)
add_executable(g2o_jacobian_check
tests/g2o_jacobian_check.cc
src/g2o_types.cc
src/camera_models/pinhole.cc)

target_compile_options(g2o_jacobian_check PRIVATE
-ffunction-sections
-fdata-sections)

target_link_options(g2o_jacobian_check PRIVATE
-Wl,--gc-sections)

target_include_directories(g2o_jacobian_check SYSTEM PRIVATE
${TORCH_INCLUDE_DIRS})

set_target_properties(g2o_jacobian_check PROPERTIES
BUILD_RPATH "${PROJECT_RUNTIME_RPATH}")

target_link_libraries(g2o_jacobian_check
types_sba
types_slam3d
core
Sophus::Sophus
${OpenCV_LIBS})

enable_testing()
add_executable(runtime_timing_check tests/runtime_timing_check.cc)
add_test(NAME runtime_timing_check COMMAND runtime_timing_check
         "${CMAKE_CURRENT_BINARY_DIR}/testing/runtime_timing")
add_test(NAME g2o_jacobian_check COMMAND g2o_jacobian_check)

add_executable(line_information_policy_check
    tests/line_information_policy_check.cc)
target_link_libraries(line_information_policy_check Eigen3::Eigen)
if(XPLSLAM_ABLATION_UNIT_LINE_INFORMATION)
    target_compile_definitions(line_information_policy_check PRIVATE
        XPLSLAM_ABLATION_UNIT_LINE_INFORMATION)
endif()
add_test(NAME line_information_policy_check
         COMMAND line_information_policy_check)

add_executable(graphite_slam_residual_check
tests/graphite_slam_residual_check.cc
src/g2o_types.cc
src/camera_models/pinhole.cc)

target_compile_options(graphite_slam_residual_check PRIVATE
-ffunction-sections
-fdata-sections)

target_link_options(graphite_slam_residual_check PRIVATE
-Wl,--gc-sections)

target_include_directories(graphite_slam_residual_check SYSTEM PRIVATE
${TORCH_INCLUDE_DIRS})

set_target_properties(graphite_slam_residual_check PROPERTIES
BUILD_RPATH "${PROJECT_RUNTIME_RPATH}")

target_link_libraries(graphite_slam_residual_check
types_sba
types_slam3d
core
Sophus::Sophus
${OpenCV_LIBS})

add_test(NAME graphite_slam_residual_check
         COMMAND graphite_slam_residual_check)

if(GRAPHITE)
    add_executable(graphite_pose_only_optimization_check
    tests/graphite_pose_only_optimization_check.cc
    src/graphite_slam_backend_host.cc
    src/graphite_slam_backend.cu)

    set_target_properties(graphite_pose_only_optimization_check PROPERTIES
    BUILD_RPATH "${PROJECT_RUNTIME_RPATH}")

    target_link_libraries(graphite_pose_only_optimization_check
    graphite
    cudart)

    if(XPLSLAM_ABLATION_UNIT_LINE_INFORMATION)
        target_compile_definitions(graphite_pose_only_optimization_check PRIVATE
            XPLSLAM_ABLATION_UNIT_LINE_INFORMATION)
    endif()

    add_test(NAME graphite_pose_only_optimization_check
             COMMAND graphite_pose_only_optimization_check)
    set_tests_properties(graphite_pose_only_optimization_check PROPERTIES
                         SKIP_RETURN_CODE 77)

    add_executable(graphite_essential_graph_optimization_check
    tests/graphite_essential_graph_optimization_check.cc
    src/graphite_slam_backend.cu)

    set_target_properties(graphite_essential_graph_optimization_check PROPERTIES
    BUILD_RPATH "${PROJECT_RUNTIME_RPATH}")

    target_link_libraries(graphite_essential_graph_optimization_check
    graphite
    cudart)

    add_test(NAME graphite_essential_graph_optimization_check
             COMMAND graphite_essential_graph_optimization_check)
    set_tests_properties(graphite_essential_graph_optimization_check PROPERTIES
                         SKIP_RETURN_CODE 77)

    add_executable(graphite_local_ba_optimization_check
    tests/graphite_local_ba_optimization_check.cc
    src/graphite_slam_backend.cu)

    set_target_properties(graphite_local_ba_optimization_check PROPERTIES
    BUILD_RPATH "${PROJECT_RUNTIME_RPATH}")

    target_link_libraries(graphite_local_ba_optimization_check
    graphite
    cudart)

    if(XPLSLAM_ABLATION_UNIT_LINE_INFORMATION)
        target_compile_definitions(graphite_local_ba_optimization_check PRIVATE
            XPLSLAM_ABLATION_UNIT_LINE_INFORMATION)
    endif()

    add_test(NAME graphite_local_ba_optimization_check
             COMMAND graphite_local_ba_optimization_check)
    set_tests_properties(graphite_local_ba_optimization_check PROPERTIES
                         SKIP_RETURN_CODE 77)
endif()

add_executable(g2o_optimization_check
tests/g2o_optimization_check.cc
src/g2o_types.cc
src/camera_models/pinhole.cc)

target_compile_options(g2o_optimization_check PRIVATE
-ffunction-sections
-fdata-sections)

target_link_options(g2o_optimization_check PRIVATE
-Wl,--gc-sections)

target_include_directories(g2o_optimization_check SYSTEM PRIVATE
${TORCH_INCLUDE_DIRS})

set_target_properties(g2o_optimization_check PROPERTIES
BUILD_RPATH "${PROJECT_RUNTIME_RPATH}")

target_link_libraries(g2o_optimization_check
types_sba
types_slam3d
core
Sophus::Sophus
${OpenCV_LIBS})

add_test(NAME g2o_optimization_check COMMAND g2o_optimization_check)

add_executable(adaptive_line_constraint_check
tests/adaptive_line_constraint_check.cc)

target_include_directories(adaptive_line_constraint_check PRIVATE
${PROJECT_SOURCE_DIR}/include)

add_test(NAME adaptive_line_constraint_check
         COMMAND adaptive_line_constraint_check)

add_executable(se3_solver_check
tests/se3_solver_check.cc)

target_compile_definitions(se3_solver_check PRIVATE XPLSLAM_ENABLE_TESTING)

set_target_properties(se3_solver_check PROPERTIES
BUILD_RPATH "${PROJECT_RUNTIME_RPATH}")

target_link_libraries(se3_solver_check
xplslam
${TENSORRT_BUILDER_RESOURCE_LIB})

add_test(NAME se3_solver_check COMMAND se3_solver_check)

add_executable(stereo_geometry_check tests/stereo_geometry_check.cc)

set_target_properties(stereo_geometry_check PROPERTIES
BUILD_RPATH "${PROJECT_RUNTIME_RPATH}")

target_link_libraries(stereo_geometry_check
xplslam
${TENSORRT_BUILDER_RESOURCE_LIB})

add_test(NAME stereo_geometry_check COMMAND stereo_geometry_check)

add_executable(xplnet_endpoint_covariance_check
tests/xplnet_endpoint_covariance_check.cc)

set_target_properties(xplnet_endpoint_covariance_check PROPERTIES
BUILD_RPATH "${PROJECT_RUNTIME_RPATH}")

target_link_libraries(xplnet_endpoint_covariance_check
xplslam
${TENSORRT_BUILDER_RESOURCE_LIB})

add_test(NAME xplnet_endpoint_covariance_check
         COMMAND xplnet_endpoint_covariance_check)

add_executable(slam_object_lifetime_check
tests/slam_object_lifetime_check.cc)

set_target_properties(slam_object_lifetime_check PROPERTIES
BUILD_RPATH "${PROJECT_RUNTIME_RPATH}")

target_link_libraries(slam_object_lifetime_check
xplslam)

add_test(NAME slam_object_lifetime_check COMMAND slam_object_lifetime_check)

if(XPLSLAM_WITH_MAP_SPARSIFICATION)
    add_executable(map_sparsification_check
    tests/map_sparsification_check.cc)

    target_compile_definitions(map_sparsification_check PRIVATE
    XPLSLAM_ENABLE_TESTING)

    set_target_properties(map_sparsification_check PROPERTIES
    BUILD_RPATH "${PROJECT_RUNTIME_RPATH}")

    target_link_libraries(map_sparsification_check
    xplslam)

    add_test(NAME map_sparsification_check COMMAND map_sparsification_check)
endif()

# Keep assertions active in Release tests and separate our suite from tests
# registered unconditionally by third-party projects such as SoPlex.
get_property(XPLSLAM_TESTS DIRECTORY PROPERTY TESTS)
foreach(XPLSLAM_TEST IN LISTS XPLSLAM_TESTS)
    set_tests_properties(${XPLSLAM_TEST} PROPERTIES LABELS xplslam)
    if(TARGET ${XPLSLAM_TEST})
        target_compile_options(${XPLSLAM_TEST} PRIVATE -UNDEBUG)
    endif()
endforeach()
endif()
