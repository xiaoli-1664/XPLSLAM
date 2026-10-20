# Third-party notices

The top-level GPL-3.0 license applies to XPLSLAM project code. Third-party
components retain their original licenses and notices.

| Component | Upstream | Source revision / license |
| --- | --- | --- |
| DBoW3 | https://github.com/rmsalinas/DBow3 | `97174a30ced4f21fdb5751ad3691b7d862acc90a`; see `thirdparty/DBow3/LICENSE.txt` |
| Graphite | https://github.com/sfu-rsl/graphite | `34e937b7e4467e19c24e9332970a59d279df956d`; MIT, see `thirdparty/graphite/LICENSE.md` |
| UPnPL | https://github.com/xiaoli-1664/upnpl | Solver sources maintained with this project, GPL-3.0 |
| Sophus | https://github.com/strasdat/Sophus | Pinned submodule; MIT |
| g2o | https://github.com/RainerKuemmerle/g2o | Pinned submodule; see its `doc/` license files and source headers |
| Pangolin | https://github.com/stevenlovegrove/Pangolin | Pinned submodule; MIT |
| SCIP | https://github.com/scipopt/scip | Pinned submodule; Apache-2.0 and bundled component notices |
| SCIPpp | https://github.com/scipopt/SCIPpp | Pinned submodule; Apache-2.0 |
| SoPlex | https://github.com/scipopt/soplex | Pinned submodule; Apache-2.0 and bundled component notices |

The bundled DBoW3 sources include XPLSLAM modifications to vocabulary I/O
and descriptor handling. Its license includes a redistribution notification
condition; retain and consult the complete original license. Its bundled
QuickLZ implementation is copyright Lasse Mikkel Reinhold and is used under
GPL-3.0; see the notices in `src/quicklz.c` and `src/quicklz.h`.

The bundled Graphite sources include XPLSLAM modifications to factor
weighting, active-state handling, GPU operations, optimization, and cuDSS
integration. The DBoW3 and Graphite build files have been limited to the
libraries used by XPLSLAM; upstream data and example programs are omitted.

The scripts in `tools/` generate the SLAM Jacobian headers and require
`wrenfold` only when regenerating those headers. They are not needed for a
normal build.

CUDA, TensorRT, cuDSS, PyTorch/LibTorch, OpenCV, Eigen, Ceres, Boost,
yaml-cpp, and OpenSSL are external dependencies and are not distributed in
this repository. Consult their respective licenses when packaging binaries.
