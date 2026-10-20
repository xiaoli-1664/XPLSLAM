#pragma once

#include <Eigen/Core>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <sophus/se3.hpp>

namespace XPLSLAM {

template <typename Scalar, typename SE3Type>
void TransformLineToCamera(const Eigen::Matrix<Scalar, 6, 1> &line_w,
                           const SE3Type &Tcw,
                           Eigen::Matrix<Scalar, 6, 1> &line_c) {
    const Eigen::Matrix<Scalar, 3, 3> Rcw = Tcw.rotationMatrix();
    const Eigen::Matrix<Scalar, 3, 1> tcw = Tcw.translation();
    line_c.template head<3>() = Rcw * line_w.template head<3>() +
                                tcw.cross(Rcw * line_w.template tail<3>());
    line_c.template tail<3>() = Rcw * line_w.template tail<3>();
}

template <typename Scalar>
void TransformLineToCamera(const Eigen::Matrix<Scalar, 6, 1> &line_w,
                           const Eigen::Matrix<Scalar, 3, 3> &Rcw,
                           const Eigen::Matrix<Scalar, 3, 1> &tcw,
                           Eigen::Matrix<Scalar, 6, 1> &line_c) {
    line_c.template head<3>() = Rcw * line_w.template head<3>() +
                                tcw.cross(Rcw * line_w.template tail<3>());
    line_c.template tail<3>() = Rcw * line_w.template tail<3>();
}

class KeyFrame;
class GeometricCamera;

class GeometricTools {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    static constexpr float kPi = 3.14159265358979323846f;

    // Compute the Fundamental matrix between KF1 and KF2
    static Eigen::Matrix3f ComputeF12(const KeyFrame &kf1, int cam_id1,
                                      const KeyFrame &kf2, int cam_id2);

    // Triangulate point with KF1 and KF2
    static bool Triangulate(Eigen::Vector3f &x_c1, Eigen::Vector3f &x_c2,
                            Eigen::Matrix<float, 3, 4> &Tc1w,
                            Eigen::Matrix<float, 3, 4> &Tc2w,
                            Eigen::Vector3f &x3D);

    static bool ImagePointToGridCell(const cv::Point2f &kp, float min_x,
                                     float min_y, float width_inv,
                                     float height_inv, int cols, int rows,
                                     int &pos_x, int &pos_y);

    static bool NormalizeLinePolar(float &d, float &theta);

    static bool NormalizedLinePolarToGridCell(float d, float theta, float d_max,
                                              float theta_inv, float d_inv,
                                              int cols, int rows, int &pos_x,
                                              int &pos_y);

    static bool LinePolarToGridCell(float d, float theta, float d_max,
                                    float theta_inv, float d_inv, int cols,
                                    int rows, int &pos_x, int &pos_y);

    static bool ImageLineToNormalizedPolar(const Eigen::Vector4f &line,
                                           Eigen::Vector2f &line_polar);

    static bool ImageLineToNormalizedPolar(const Eigen::Vector3f &line,
                                           float &d, float &theta);

    static Eigen::Matrix<float, 3, 4>
    MakeProjectionMatrix(const Sophus::SE3f &Tcw);

    static bool BuildObservationPlane(const Eigen::VectorXf &line,
                                      GeometricCamera *camera,
                                      Eigen::Vector4f &plane);

    static bool
    TriangulateLine(const Eigen::Vector4f &line1, GeometricCamera *camera1,
                    const Sophus::SE3f &Tcw1, const Eigen::Vector4f &line2,
                    GeometricCamera *camera2, const Sophus::SE3f &Tcw2,
                    Eigen::Matrix<float, 6, 1> &line_w);

    static float LineDepthInCamera(const Eigen::Matrix<float, 6, 1> &line_w,
                                   const Sophus::SE3f &Tcw);

    static bool
    LinePointFromBackprojectedRay(const Eigen::Matrix<float, 6, 1> &line_c,
                                  const Eigen::Vector3f &ray_c,
                                  Eigen::Vector3f &point_c);

    static bool
    LinePointFromBackprojectedRay(const Eigen::Matrix<double, 6, 1> &line_c,
                                  const Eigen::Vector3d &ray_c,
                                  Eigen::Vector3d &point_c);

    static bool IsLineDepthPositive(const Eigen::Matrix<float, 6, 1> &line_w,
                                    GeometricCamera *camera,
                                    const Sophus::SE3f &Tcw,
                                    const Eigen::Vector4f &line_obs);

    static bool ComputeLineReprojectionError(
        const Eigen::Matrix<float, 6, 1> &line_w, GeometricCamera *camera,
        const Sophus::SE3f &Tcw, const Eigen::Vector4f &line_obs,
        Eigen::Vector2f &errors);

    template <int rows, int cols>
    static bool CheckMatrices(const cv::Mat &cvMat,
                              const Eigen::Matrix<float, rows, cols> &eigMat) {
        const float epsilon = 1e-3;
        // std::cout << cvMat.cols - cols << cvMat.rows - rows << std::endl;
        if (rows != cvMat.rows || cols != cvMat.cols) {
            std::cout << "wrong cvmat size\n";
            return false;
        }
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                if ((cvMat.at<float>(i, j) > (eigMat(i, j) + epsilon)) ||
                    (cvMat.at<float>(i, j) < (eigMat(i, j) - epsilon))) {
                    std::cout << "cv mat:\n" << cvMat << std::endl;
                    std::cout << "eig mat:\n" << eigMat << std::endl;
                    return false;
                }
        return true;
    }

    template <typename T, int rows, int cols>
    static bool CheckMatrices(const Eigen::Matrix<T, rows, cols> &eigMat1,
                              const Eigen::Matrix<T, rows, cols> &eigMat2) {
        const float epsilon = 1e-3;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                if ((eigMat1(i, j) > (eigMat2(i, j) + epsilon)) ||
                    (eigMat1(i, j) < (eigMat2(i, j) - epsilon))) {
                    std::cout << "eig mat 1:\n" << eigMat1 << std::endl;
                    std::cout << "eig mat 2:\n" << eigMat2 << std::endl;
                    return false;
                }
        return true;
    }
};

} // namespace XPLSLAM
