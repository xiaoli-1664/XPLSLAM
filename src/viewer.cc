#include "viewer.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/persistence.hpp>
#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/display/widgets.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/opengl_render_state.h>
#include <pangolin/handler/handler.h>
#include <pangolin/var/var.h>
#include <unistd.h>

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

} // namespace

Viewer::Viewer(System *pSystem, FrameDrawer *pFrameDrawer,
               MapDrawer *pMapDrawer, Settings *settings,
               const string &strSettingPath)
    : mpSystem(pSystem), mpFrameDrawer(pFrameDrawer),
      mpMapDrawer(pMapDrawer), mT(33.0), mImageViewerScale(1.0f),
      mViewpointX(0.0f), mViewpointY(-3.0f), mViewpointZ(-3.0f),
      mViewpointF(500.0f), mbFinishRequested(false), mbFinished(true) {
    LoadViewerParameters(settings, strSettingPath);
}

void Viewer::LoadViewerParameters(Settings *settings,
                                  const string &strSettingPath) {
    if (settings) {
        const float fps = settings->fps() < 1.0f ? 30.0f : settings->fps();
        mT = 1e3 / fps;
    }

    cv::FileStorage fs(strSettingPath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        return;
    }

    const float fps = ReadFloat(fs, "Camera.fps", settings ? settings->fps()
                                                            : 30.0f);
    mT = 1e3 / (fps < 1.0f ? 30.0f : fps);
    mImageViewerScale =
        ReadFloat(fs, "Viewer.ImageScale",
                  ReadFloat(fs, "Viewer.imageViewScale", mImageViewerScale));
    mViewpointX = ReadFloat(fs, "Viewer.ViewpointX", mViewpointX);
    mViewpointY = ReadFloat(fs, "Viewer.ViewpointY", mViewpointY);
    mViewpointZ = ReadFloat(fs, "Viewer.ViewpointZ", mViewpointZ);
    mViewpointF = ReadFloat(fs, "Viewer.ViewpointF", mViewpointF);
}

void Viewer::Run() {
    {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinished = false;
    }

    pangolin::CreateWindowAndBind("XPLSLAM: Map Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0,
                                            pangolin::Attach::Pix(180));
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", true, true);
    pangolin::Var<bool> menuShowPoints("menu.Show Points", true, true);
    pangolin::Var<bool> menuShowLines("menu.Show Lines", true, true);
    pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames", true, true);
    pangolin::Var<bool> menuShowGraph("menu.Show Graph", false, true);
    pangolin::Var<bool> menuShowTrajectory("menu.Show Trajectory", true, true);
    pangolin::Var<bool> menuShowFrame("menu.Show Frame", true, true);
    pangolin::Var<bool> menuReset("menu.Reset", false, false);
    pangolin::Var<bool> menuFinish("menu.Finish", false, false);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, mViewpointF, mViewpointF, 512,
                                   389, 0.1, 10000),
        pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0,
                                  0, 0.0, -1.0, 0.0));

    pangolin::View &d_cam =
        pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(180), 1.0,
                       -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::OpenGlMatrix Twb;
    pangolin::OpenGlMatrix Ow;
    Twb.SetIdentity();
    Ow.SetIdentity();

    cv::namedWindow("XPLSLAM: Current Frame", cv::WINDOW_NORMAL);
    bool follow = true;

    while (!pangolin::ShouldQuit()) {
        if (CheckFinish()) {
            break;
        }
        if (menuFinish) {
            break;
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        mpMapDrawer->GetCurrentOpenGLCameraMatrix(Twb, Ow);

        if (menuFollowCamera && follow) {
            s_cam.Follow(Twb);
        } else if (menuFollowCamera && !follow) {
            s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(
                1024, 768, mViewpointF, mViewpointF, 512, 389, 0.1, 10000));
            s_cam.SetModelViewMatrix(
                pangolin::ModelViewLookAt(mViewpointX, mViewpointY,
                                          mViewpointZ, 0, 0, 0, 0.0, -1.0,
                                          0.0));
            s_cam.Follow(Twb);
            follow = true;
        } else if (!menuFollowCamera && follow) {
            follow = false;
        }

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        mpMapDrawer->DrawCurrentCamera(Twb);
        if (menuShowKeyFrames || menuShowGraph) {
            mpMapDrawer->DrawKeyFrames(menuShowKeyFrames, menuShowGraph);
        }
        if (menuShowPoints) {
            mpMapDrawer->DrawMapPoints();
        }
        if (menuShowLines) {
            mpMapDrawer->DrawMapLines();
        }
        if (menuShowTrajectory) {
            mpMapDrawer->DrawTrajectory();
        }

        pangolin::FinishFrame();

        if (menuShowFrame) {
            cv::Mat frame = mpFrameDrawer->DrawFrame(mImageViewerScale);
            if (!frame.empty()) {
                cv::imshow("XPLSLAM: Current Frame", frame);
            }
        }
        cv::waitKey(static_cast<int>(mT));

        if (menuReset) {
            menuReset = false;
            mpSystem->ResetActiveMap();
        }
    }

    cv::destroyWindow("XPLSLAM: Current Frame");
    SetFinish();
}

void Viewer::RequestFinish() {
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool Viewer::CheckFinish() {
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void Viewer::SetFinish() {
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool Viewer::isFinished() {
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

} // namespace XPLSLAM
