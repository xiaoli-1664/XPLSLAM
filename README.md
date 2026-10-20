# XPLSLAM

XPLSLAM is a multi-camera visual SLAM system using point and line features.
It combines a TensorRT XPLNet frontend with tracking, local mapping, loop
closing, and an optional Graphite GPU optimization backend.

## Dependencies

- Linux, a C++17 compiler, and CMake 3.24 or newer.
- NVIDIA GPU and driver, CUDA Toolkit, TensorRT, and a CUDA-enabled
  PyTorch/LibTorch installation. The development environment uses CUDA 12.6,
  TensorRT 10.8, and PyTorch 2.8 with CUDA 12.6. Use compatible CUDA versions
  and the C++ ABI required by your PyTorch/LibTorch build.
- OpenCV 4, Eigen 3.3.9 or newer, Ceres 2.2 or newer, yaml-cpp, and OpenSSL.
  Use the same Eigen version that your Ceres installation was built with.
- Boost and cuDSS 0.7 or newer for Graphite.
- OpenGL/EGL and the Pangolin prerequisites for visualization. On Ubuntu,
  these include `libgl1-mesa-dev`, `libegl1-mesa-dev`, and `libepoxy-dev`.

Sophus, g2o, Pangolin, SCIP, SCIPpp, and SoPlex are pinned Git submodules.
The required DBoW3, Graphite, and UPnPL sources are included in `thirdparty/`.
See [third-party notices](THIRD_PARTY_NOTICES.md) for provenance and licenses.

## Build

Install the dependencies above, then run:

```bash
git clone --recurse-submodules https://github.com/xiaoli-1664/XPLSLAM.git
cd XPLSLAM

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTENSORRT_ROOT=/path/to/TensorRT \
  -DTorch_DIR=/path/to/torch/share/cmake/Torch \
  -DXPLSLAM_CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
  -DGRAPHITE=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DTORCH_CUDA_ARCH_LIST=8.9
cmake --build build -j12
```

Set `89` / `8.9` to the compute capability of your GPU. CMake can also find
Torch from the active Python environment. For an existing clone, initialize
dependencies with `git submodule update --init --recursive`.

Optional switches:

- `-DGRAPHITE=OFF`: use the g2o optimization backend; the frontend still
  requires CUDA and TensorRT.
- `-DXPLSLAM_WITH_PANGOLIN_VIEWER=OFF`: build without the viewer.
- `-DXPLSLAM_WITH_MAP_SPARSIFICATION=OFF`: omit SCIP/SoPlex map sparsification.
- `-DXPLSLAM_ENABLE_PROFILING=OFF`: disable detailed timing instrumentation.
- `-DXPLSLAM_BUILD_TESTS=ON`: build the SLAM tests; run them with
  `ctest --test-dir build -L xplslam --output-on-failure`. GPU tests require a visible
  CUDA device.

## NewCollege example

Prepare the NewCollege images and timestamps in this layout:

```text
new_college/
├── all_camera_images/
│   ├── cam0/
│   ├── cam1/
│   ├── cam3/
│   └── cam4/
└── all_camera_info/
    └── cam.txt
```

Images are sorted by filename and paired by index across cameras. `cam.txt`
contains timestamps in seconds, either one per line or comma-separated with
a header containing `cam0`. The supplied configuration uses cameras
`cam0`, `cam1`, `cam3`, and `cam4`, in that order. Verify that the images and
calibration match `config/new_college.yaml`.

The model weights and vocabularies are not included in this source release.
Provide these runtime assets separately:

```text
models/xplnet.onnx
models/verification_coarse.onnx
models/line_description.onnx
vocabulary/point_voc.dbow3bin
vocabulary/line_voc.dbow3bin
```

The ONNX models and both DBoW3 vocabularies must match the XPLNet descriptors
used by this version. Update the three `XPLNet.*onnx` paths in
`config/new_college.yaml` if necessary. Relative model paths are resolved
from the current working directory.

Run from the repository root:

```bash
./build/xplslam_new_college \
  --dataset /path/to/new_college \
  --settings config/new_college.yaml \
  --point-vocab vocabulary/point_voc.dbow3bin \
  --line-vocab vocabulary/line_voc.dbow3bin \
  --output-dir output/new_college
```

Add `--viewer` to display tracking and the map, or `--max-frames 300` for a
short run. Use `--help` for all options. The trajectory is written to
`output/new_college/trajectory_tum.txt`; TensorRT engine caches are generated
under `output/` on the first run. Clear the relevant engine caches after
changing models, input dimensions, or GPU/TensorRT versions.

## License

XPLSLAM project code is released under [GPL-3.0](LICENSE). Third-party code
retains its own copyright notices and license terms.
