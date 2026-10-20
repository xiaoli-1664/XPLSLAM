#include "map_drawer.h"

#include "keyframe.h"
#include "map.h"
#include "map_line.h"
#include "map_point.h"

#include <opencv2/core/persistence.hpp>
#include <pangolin/gl/gl.h>

using namespace std;

namespace XPLSLAM {

namespace {

float ReadFloat(cv::FileStorage &fs, const string &name,
                const float default_value) {
    cv::FileNode node = fs[name];
    if (node.empty()) {
        return default_value;
    }
    return static_cast<float>(node.real());
}

void FillOpenGlMatrix(const Sophus::SE3f &Twc, pangolin::OpenGlMatrix &M) {
    const Eigen::Matrix4f mat = Twc.matrix();
    for (int i = 0; i < 4; ++i) {
        M.m[4 * i] = mat(0, i);
        M.m[4 * i + 1] = mat(1, i);
        M.m[4 * i + 2] = mat(2, i);
        M.m[4 * i + 3] = mat(3, i);
    }
}

} // namespace

MapDrawer::MapDrawer(Atlas *pAtlas, Settings *settings,
                     const string &strSettingPath)
    : mpAtlas(pAtlas), mKeyFrameSize(0.05f), mKeyFrameLineWidth(1.0f),
      mGraphLineWidth(0.9f), mPointSize(2.0f), mLineWidth(1.0f),
      mCameraSize(0.08f), mCameraLineWidth(2.0f),
      mTrajectoryLineWidth(2.0f), mLineSegmentHalfLength(2.0f),
      mCameraPose(), mvTbc() {
    if (settings) {
        mvTbc = settings->Tbc();
    }
    LoadViewerParameters(strSettingPath);
}

void MapDrawer::LoadViewerParameters(const string &strSettingPath) {
    cv::FileStorage fs(strSettingPath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        return;
    }

    mKeyFrameSize = ReadFloat(fs, "Viewer.KeyFrameSize", mKeyFrameSize);
    mKeyFrameLineWidth =
        ReadFloat(fs, "Viewer.KeyFrameLineWidth", mKeyFrameLineWidth);
    mGraphLineWidth = ReadFloat(fs, "Viewer.GraphLineWidth", mGraphLineWidth);
    mPointSize = ReadFloat(fs, "Viewer.PointSize", mPointSize);
    mLineWidth = ReadFloat(fs, "Viewer.LineWidth", mLineWidth);
    mCameraSize = ReadFloat(fs, "Viewer.CameraSize", mCameraSize);
    mCameraLineWidth =
        ReadFloat(fs, "Viewer.CameraLineWidth", mCameraLineWidth);
    mTrajectoryLineWidth =
        ReadFloat(fs, "Viewer.TrajectoryLineWidth", mTrajectoryLineWidth);
    mLineSegmentHalfLength = ReadFloat(
        fs, "Viewer.LineSegmentHalfLength", mLineSegmentHalfLength);
}

void MapDrawer::DrawMapPoints() {
    Map *pActiveMap = mpAtlas->GetCurrentMap();
    if (!pActiveMap) {
        return;
    }

    const vector<MapPointPtr> points = pActiveMap->GetAllMapPoints();
    const vector<MapPointPtr> ref_points = pActiveMap->GetReferenceMapPoints();
    set<MapPointPtr> ref_set(ref_points.begin(), ref_points.end());

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0f, 0.0f, 0.0f);
    for (MapPointPtr pMP : points) {
        if (!pMP || pMP->isBad() || ref_set.count(pMP)) {
            continue;
        }
        const Eigen::Vector3f pos = pMP->GetWorldPos();
        glVertex3f(pos.x(), pos.y(), pos.z());
    }
    glEnd();

    glPointSize(mPointSize + 1.0f);
    glBegin(GL_POINTS);
    glColor3f(1.0f, 0.0f, 0.0f);
    for (MapPointPtr pMP : ref_set) {
        if (!pMP || pMP->isBad()) {
            continue;
        }
        const Eigen::Vector3f pos = pMP->GetWorldPos();
        glVertex3f(pos.x(), pos.y(), pos.z());
    }
    glEnd();
}

void MapDrawer::DrawMapLines() {
    Map *pActiveMap = mpAtlas->GetCurrentMap();
    if (!pActiveMap) {
        return;
    }

    const vector<MapLinePtr> lines = pActiveMap->GetAllMapLines();
    const vector<MapLinePtr> ref_lines = pActiveMap->GetReferenceMapLines();
    set<MapLinePtr> ref_set(ref_lines.begin(), ref_lines.end());

    glLineWidth(mLineWidth);
    glBegin(GL_LINES);
    for (MapLinePtr pML : lines) {
        if (!pML || pML->isBad()) {
            continue;
        }
        const PluckerLine line = pML->GetWorldPos();
        Eigen::Vector3f moment = line.head<3>();
        Eigen::Vector3f direction = line.tail<3>();
        const float norm = direction.norm();
        if (!line.allFinite() || norm < 1e-6f) {
            continue;
        }
        direction /= norm;
        moment /= norm;
        const Eigen::Vector3f p0 = direction.cross(moment);
        const Eigen::Vector3f p1 = p0 - mLineSegmentHalfLength * direction;
        const Eigen::Vector3f p2 = p0 + mLineSegmentHalfLength * direction;
        if (ref_set.count(pML)) {
            glColor3f(1.0f, 0.55f, 0.0f);
        } else {
            glColor3f(0.0f, 0.55f, 0.85f);
        }
        glVertex3f(p1.x(), p1.y(), p1.z());
        glVertex3f(p2.x(), p2.y(), p2.z());
    }
    glEnd();
}

void MapDrawer::DrawKeyFrames(bool draw_keyframes, bool draw_graph) {
    Map *pActiveMap = mpAtlas->GetCurrentMap();
    if (!pActiveMap) {
        return;
    }

    const vector<KeyFramePtr> keyframes = pActiveMap->GetAllKeyFrames();
    if (draw_keyframes) {
        for (KeyFramePtr pKF : keyframes) {
            if (!pKF || pKF->isBad()) {
                continue;
            }
            DrawCameraRig(pKF->GetPose(), pKF->GetCameraExtrinsics(),
                          mKeyFrameSize, mKeyFrameLineWidth, 0.0f, 0.0f,
                          1.0f);
        }
    }

    if (!draw_graph) {
        return;
    }

    glLineWidth(mGraphLineWidth);
    glColor4f(0.0f, 0.8f, 0.0f, 0.6f);
    glBegin(GL_LINES);
    for (KeyFramePtr pKF : keyframes) {
        if (!pKF || pKF->isBad()) {
            continue;
        }
        const Eigen::Vector3f Ow = pKF->GetPoseInverse().translation();

        const vector<KeyFramePtr> covisible = pKF->GetCovisiblesByWeight(100);
        for (KeyFramePtr pCov : covisible) {
            if (!pCov || pCov->isBad() || pCov->mnId < pKF->mnId) {
                continue;
            }
            const Eigen::Vector3f Ow2 = pCov->GetPoseInverse().translation();
            glVertex3f(Ow.x(), Ow.y(), Ow.z());
            glVertex3f(Ow2.x(), Ow2.y(), Ow2.z());
        }

        KeyFramePtr parent = pKF->GetParent();
        if (parent && !parent->isBad()) {
            const Eigen::Vector3f Owp =
                parent->GetPoseInverse().translation();
            glVertex3f(Ow.x(), Ow.y(), Ow.z());
            glVertex3f(Owp.x(), Owp.y(), Owp.z());
        }

        const set<KeyFramePtr> loops = pKF->GetLoopEdges();
        for (KeyFramePtr pLoop : loops) {
            if (!pLoop || pLoop->isBad() || pLoop->mnId < pKF->mnId) {
                continue;
            }
            const Eigen::Vector3f Owl =
                pLoop->GetPoseInverse().translation();
            glVertex3f(Ow.x(), Ow.y(), Ow.z());
            glVertex3f(Owl.x(), Owl.y(), Owl.z());
        }
    }
    glEnd();
}

void MapDrawer::DrawTrajectory() {
    vector<CameraTrajectoryEntry> trajectory_entries;
    {
        unique_lock<mutex> lock(mMutexCamera);
        trajectory_entries = mvCameraTrajectory;
    }

    vector<Eigen::Vector3f> trajectory;
    trajectory.reserve(trajectory_entries.size());
    for (const CameraTrajectoryEntry &entry : trajectory_entries) {
        Sophus::SE3f Tbw = entry.Tbr;
        KeyFramePtr reference_keyframe = entry.reference_keyframe;
        if (reference_keyframe) {
            Sophus::SE3f Trw;
            while (reference_keyframe && reference_keyframe->isBad()) {
                Trw = Trw * reference_keyframe->mTbp;
                reference_keyframe = reference_keyframe->GetParent();
            }
            if (!reference_keyframe) {
                continue;
            }
            Tbw = entry.Tbr * Trw * reference_keyframe->GetPose();
        }
        trajectory.push_back(Tbw.inverse().translation());
    }

    if (trajectory.size() < 2) {
        return;
    }

    glLineWidth(mTrajectoryLineWidth);
    glColor3f(1.0f, 0.0f, 0.0f);
    glBegin(GL_LINE_STRIP);
    for (const Eigen::Vector3f &position : trajectory) {
        glVertex3f(position.x(), position.y(), position.z());
    }
    glEnd();
}

void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twb) {
    Sophus::SE3f Tbw;
    vector<Sophus::SE3f> extrinsics;
    {
        unique_lock<mutex> lock(mMutexCamera);
        Tbw = mCameraPose;
        extrinsics = mvTbc;
    }
    DrawCameraRig(Tbw, extrinsics, mCameraSize, mCameraLineWidth, 0.0f, 1.0f,
                  0.0f);
    (void)Twb;
}

void MapDrawer::SetCurrentCameraPose(
    const Sophus::SE3f &Tbw, const KeyFramePtr &reference_keyframe) {
    Sophus::SE3f Tbr = Tbw;
    if (reference_keyframe) {
        Tbr = Tbw * reference_keyframe->GetPoseInverse();
    }
    const Eigen::Vector3f current_position = Tbw.inverse().translation();

    unique_lock<mutex> lock(mMutexCamera);
    mCameraPose = Tbw;

    if (mvCameraTrajectory.empty() ||
        (current_position - mvCameraTrajectory.back().recorded_position)
                .squaredNorm() >
            1e-8f) {
        mvCameraTrajectory.push_back(
            {Tbr, reference_keyframe, current_position});
    }
}

void MapDrawer::SetCurrentCameraExtrinsics(const vector<Sophus::SE3f> &vTbc) {
    unique_lock<mutex> lock(mMutexCamera);
    mvTbc = vTbc;
}

void MapDrawer::ClearTrajectory() {
    unique_lock<mutex> lock(mMutexCamera);
    mvCameraTrajectory.clear();
}

void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M,
                                             pangolin::OpenGlMatrix &MOw) {
    Sophus::SE3f Tbw;
    {
        unique_lock<mutex> lock(mMutexCamera);
        Tbw = mCameraPose;
    }
    const Sophus::SE3f Twb = Tbw.inverse();
    FillOpenGlMatrix(Twb, M);

    MOw.SetIdentity();
    const Eigen::Vector3f Ow = Twb.translation();
    MOw.m[12] = Ow.x();
    MOw.m[13] = Ow.y();
    MOw.m[14] = Ow.z();
}

void MapDrawer::DrawFrustum(const float size, const float line_width,
                            const float red, const float green,
                            const float blue) {
    const float w = size;
    const float h = w * 0.75f;
    const float z = w * 0.6f;

    glLineWidth(line_width);
    glColor3f(red, green, blue);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(w, h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, h, z);
    glVertex3f(w, h, z);
    glVertex3f(w, -h, z);
    glVertex3f(-w, h, z);
    glVertex3f(-w, -h, z);
    glVertex3f(-w, h, z);
    glVertex3f(w, h, z);
    glVertex3f(-w, -h, z);
    glVertex3f(w, -h, z);
    glEnd();
}

void MapDrawer::DrawCameraRig(const Sophus::SE3f &Tbw,
                              const vector<Sophus::SE3f> &vTbc,
                              const float size, const float line_width,
                              const float red, const float green,
                              const float blue) {
    pangolin::OpenGlMatrix Twb_gl;
    FillOpenGlMatrix(Tbw.inverse(), Twb_gl);
    glPushMatrix();
#ifdef HAVE_GLES
    glMultMatrixf(Twb_gl.m);
#else
    glMultMatrixd(Twb_gl.m);
#endif
    DrawFrustum(size, line_width, red, green, blue);
    glPopMatrix();

    for (const Sophus::SE3f &Tbc : vTbc) {
        pangolin::OpenGlMatrix Twc_gl;
        FillOpenGlMatrix(Tbw.inverse() * Tbc, Twc_gl);
        glPushMatrix();
#ifdef HAVE_GLES
        glMultMatrixf(Twc_gl.m);
#else
        glMultMatrixd(Twc_gl.m);
#endif
        DrawFrustum(size * 0.75f, line_width, red, green, blue);
        glPopMatrix();
    }
}

} // namespace XPLSLAM
