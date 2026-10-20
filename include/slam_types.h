#pragma once

#include <memory>

namespace XPLSLAM {

class KeyFrame;
class MapPoint;
class MapLine;

using KeyFramePtr = std::shared_ptr<KeyFrame>;
using KeyFrameWeakPtr = std::weak_ptr<KeyFrame>;

using MapPointPtr = std::shared_ptr<MapPoint>;
using MapPointWeakPtr = std::weak_ptr<MapPoint>;

using MapLinePtr = std::shared_ptr<MapLine>;
using MapLineWeakPtr = std::weak_ptr<MapLine>;

} // namespace XPLSLAM
