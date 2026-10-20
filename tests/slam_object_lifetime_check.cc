#include <iostream>
#include <memory>

#define protected public
#include "keyframe.h"
#undef protected

#include "map.h"
#include "map_line.h"
#include "map_point.h"

namespace {

template <typename T>
bool ExpectExpired(const std::weak_ptr<T> &weak, const char *name) {
    if (!weak.expired()) {
        std::cerr << name << " is still alive" << std::endl;
        return false;
    }
    return true;
}

} // namespace

int main() {
    using namespace XPLSLAM;

    MapPoint::nCam = 1;
    MapLine::nCam = 1;

    {
        Map map;
        auto keyframe = std::make_shared<KeyFrame>();
        keyframe->mnId = 1;
        keyframe->mnCam = 1;
        keyframe->mvpMapPoints_cams = {{nullptr}};
        keyframe->mvpMapLines_cams = {{nullptr}};
        keyframe->mvDescriptors_point = {
            cv::Mat(1, 64, CV_32F, cv::Scalar(1.0f))};
        keyframe->mDescriptors_point = keyframe->mvDescriptors_point[0].clone();
        keyframe->mvPointDescriptorRowToCamRow = {{0, 0}};

        auto point =
            std::make_shared<MapPoint>(Eigen::Vector3f::Zero(), keyframe, &map);
        auto line = std::make_shared<MapLine>(PluckerLine::Zero(), keyframe,
                                              &map);

        point->AddObservation(keyframe, 0, 0);
        line->AddObservation(keyframe, 0, 0);
        keyframe->AddMapPoint(point, 0, 0);
        keyframe->AddMapLine(line, 0, 0);

        map.AddKeyFrame(keyframe);
        map.AddMapPoint(point);
        map.AddMapLine(line);

        const MapPointWeakPtr weakPoint = point;
        const MapLineWeakPtr weakLine = line;
        const KeyFrameWeakPtr weakKeyFrame = keyframe;

        map.EraseMapPoint(point);
        map.EraseMapLine(line);
        point.reset();
        line.reset();

        if (weakPoint.expired() || weakLine.expired()) {
            std::cerr << "KeyFrame associations did not retain map features"
                      << std::endl;
            return 1;
        }

        keyframe->EraseMapPointMatch(0, 0);
        keyframe->EraseMapLineMatch(0, 0);

        if (!ExpectExpired(weakPoint, "MapPoint") ||
            !ExpectExpired(weakLine, "MapLine")) {
            return 1;
        }

        map.EraseKeyFrame(keyframe);
        keyframe.reset();
        if (!ExpectExpired(weakKeyFrame, "KeyFrame"))
            return 1;
    }

    {
        Map map;
        auto parent = std::make_shared<KeyFrame>();
        auto child = std::make_shared<KeyFrame>();
        parent->mnId = 2;
        child->mnId = 3;

        map.AddKeyFrame(parent);
        map.AddKeyFrame(child);
        child->ChangeParent(parent);

        const KeyFrameWeakPtr weakParent = parent;
        const KeyFrameWeakPtr weakChild = child;

        map.EraseKeyFrame(child);
        map.EraseKeyFrame(parent);
        child.reset();
        parent.reset();

        if (!ExpectExpired(weakChild, "child KeyFrame") ||
            !ExpectExpired(weakParent, "parent KeyFrame")) {
            return 1;
        }
    }

    {
        Map map;
        auto keyframe = std::make_shared<KeyFrame>();
        keyframe->mnId = 4;

        auto oldPoint =
            std::make_shared<MapPoint>(Eigen::Vector3f::Zero(), keyframe, &map);
        auto replacement = std::make_shared<MapPoint>(
            Eigen::Vector3f::Ones(), keyframe, &map);
        map.AddMapPoint(oldPoint);
        map.AddMapPoint(replacement);

        const MapPointWeakPtr weakReplacement = replacement;
        oldPoint->Replace(replacement);
        map.EraseMapPoint(replacement);
        replacement.reset();

        if (!ExpectExpired(weakReplacement, "replacement MapPoint"))
            return 1;
        if (oldPoint->GetReplaced()) {
            std::cerr << "Expired replacement was not cleared" << std::endl;
            return 1;
        }
    }

    {
        Map map;
        auto keyframe = std::make_shared<KeyFrame>();
        keyframe->mnId = 5;
        auto point =
            std::make_shared<MapPoint>(Eigen::Vector3f::Zero(), keyframe, &map);
        auto line = std::make_shared<MapLine>(PluckerLine::Zero(), keyframe,
                                              &map);

        map.AddKeyFrame(keyframe);
        map.AddMapPoint(point);
        map.AddMapLine(line);

        const KeyFrameWeakPtr weakKeyFrame = keyframe;
        const MapPointWeakPtr weakPoint = point;
        const MapLineWeakPtr weakLine = line;
        keyframe.reset();
        point.reset();
        line.reset();

        map.clear();
        if (!ExpectExpired(weakKeyFrame, "cleared KeyFrame") ||
            !ExpectExpired(weakPoint, "cleared MapPoint") ||
            !ExpectExpired(weakLine, "cleared MapLine")) {
            return 1;
        }
    }

    return 0;
}
