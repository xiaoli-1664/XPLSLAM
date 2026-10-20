#pragma once

#include <vector>

#include <Eigen/Dense>
#include <ceres/ceres.h>

#include <random>

namespace UPnPL {

using namespace std;

struct LambdaCost {
    LambdaCost(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b, double c)
        : a_(a), b_(b), c_(c) {}

    template <typename T>
    bool operator()(T const *const *lambdas, T *residual) const {
        const T *lambda = lambdas[0];
        int n = a_.rows();

        T result = T(0.0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                result += lambda[i] * T(a_(i, j)) * lambda[j];
            }
        }

        for (int i = 0; i < n; ++i) {
            result += lambda[i] * T(b_(i));
        }

        result += T(c_);

        residual[0] = result;
        return true;
    }

    Eigen::MatrixXd a_;
    Eigen::VectorXd b_;
    double c_;
};

class UPnPL {
  public:
    UPnPL(bool is_normalized = true) : is_normalized_(is_normalized) {}

    void solveMain(const vector<Eigen::Vector3d> &points_w,
                   const vector<Eigen::VectorXd> &lines_w,
                   const vector<Eigen::Vector3d> &uv_c,
                   const vector<Eigen::VectorXd> &lines_c,
                   const vector<int> &points_cam, const vector<int> &lines_cam,
                   const vector<Eigen::Matrix3d> &Rbc,
                   const vector<Eigen::Vector3d> &tbc, Eigen::Matrix3d &R_bw,
                   Eigen::Vector3d &t_bw);

    void solveUPnPL_DLS(const vector<Eigen::Vector3d> &points_w,
                        const vector<Eigen::VectorXd> &lines_w,
                        const vector<Eigen::Vector3d> &uv_c,
                        const vector<Eigen::Vector3d> &normals_c,
                        const vector<int> &points_cam,
                        const vector<int> &lines_cam,
                        const vector<Eigen::Matrix3d> &Rbc,
                        const vector<Eigen::Vector3d> &tbc,
                        Eigen::Matrix3d &R_bw, Eigen::Vector3d &t_bw);

    void solveUPnPL_EPnPL(const vector<Eigen::Vector3d> &points_w,
                          const vector<Eigen::VectorXd> &lines_w,
                          const vector<Eigen::Vector3d> &uv_c,
                          const vector<Eigen::Vector3d> &normals_c,
                          const vector<int> &points_cam,
                          const vector<int> &lines_cam,
                          const vector<Eigen::Matrix3d> &Rbc,
                          const vector<Eigen::Vector3d> &tbc,
                          Eigen::Matrix3d &R_bw, Eigen::Vector3d &t_bw);

    void chooseControlPoints(const vector<Eigen::Vector3d> &points_w,
                             const vector<Eigen::VectorXd> &lines_w,
                             vector<Eigen::Vector3d> &control_points);

    void computeAlpha(const vector<Eigen::Vector3d> &points_w,
                      const vector<Eigen::VectorXd> &lines_w,
                      const vector<Eigen::Vector3d> &control_points,
                      vector<double> &alpha);

    double solveN1(const vector<Eigen::Vector3d> &control_points_w,
                   const Eigen::VectorXd &beta, const Eigen::VectorXd &y,
                   Eigen::Matrix3d &Rbw, Eigen::Vector3d &tbw);

    // double solveN1_reproj(const vector<Eigen::Vector3d> &control_points_w,
    //                       const Eigen::VectorXd &beta, const Eigen::VectorXd
    //                       &y, Eigen::Matrix3d &Rbw, Eigen::Vector3d &tbw);

    double solveN2(const vector<Eigen::Vector3d> &control_points_w,
                   const Eigen::MatrixXd &beta, const Eigen::VectorXd &y,
                   Eigen::Matrix3d &Rbw, Eigen::Vector3d &tbw);

    double solveN3(const vector<Eigen::Vector3d> &control_points_w,
                   const Eigen::MatrixXd &beta, const Eigen::VectorXd &y,
                   Eigen::Matrix3d &Rbw, Eigen::Vector3d &tbw);

    double solveN4(const vector<Eigen::Vector3d> &control_points_w,
                   const Eigen::MatrixXd &beta, const Eigen::VectorXd &y,
                   Eigen::Matrix3d &Rbw, Eigen::Vector3d &tbw);

    bool lambdaRefine(const vector<Eigen::MatrixXd> &a,
                      const vector<Eigen::VectorXd> &b, vector<double> &c,
                      Eigen::VectorXd &lambda);

    void computePose(const vector<Eigen::Vector3d> &control_points_b,
                     const vector<double> &alpha, Eigen::Matrix3d &R_bw,
                     Eigen::Vector3d &t_bw);

    void computePose(const vector<int> &index,
                     const vector<Eigen::Vector3d> &control_points_b,
                     const vector<double> &alpha, Eigen::Matrix3d &R_bw,
                     Eigen::Vector3d &t_bw);

    void computePose(const vector<Eigen::Vector3d> &control_points_b,
                     Eigen::Matrix3d &R_bw, Eigen::Vector3d &t_bw);

    double computeReprojError(const Eigen::Matrix3d &R_bw,
                              const Eigen::Vector3d &t_bw);

    double computeReprojError(const vector<int> &index,
                              const Eigen::Matrix3d &R_bw,
                              const Eigen::Vector3d &t_bw);

    void normalization(const vector<Eigen::Vector3d> &points_w,
                       const vector<Eigen::VectorXd> &lines_w,
                       vector<Eigen::Vector3d> &points_w_n,
                       vector<Eigen::VectorXd> &lines_w_n, double &scale,
                       Eigen::Vector3d &center, bool normalize = false);

    Eigen::Vector2d computeCorrectness(const Eigen::Vector2d &p,
                                       const Eigen::Vector2d &pd,
                                       const Eigen::Vector2d &v,
                                       const Eigen::Vector3d &line_param,
                                       double d);

    Eigen::Vector3d backProjPointToLine(const Eigen::Vector2d &x,
                                        const Eigen::Vector3d &X,
                                        const Eigen::Vector3d &v);

    double pointLineDistance(const Eigen::Vector3d &x, const Eigen::Vector3d &a,
                             const Eigen::Vector3d &b);

    double pointLineDistance2D(const Eigen::Vector2d &x,
                               const Eigen::Vector3d &line_param) {
        double a = line_param(0), b = line_param(1), c = line_param(2);
        return fabs(a * x(0) + b * x(1) + c) / sqrt(a * a + b * b);
    }

    Eigen::Matrix<double, 3, 9> phi(const Eigen::Vector3d &p);

    static void CGR2Rotation(const Eigen::Vector3d &s, Eigen::Matrix3d &R);

    static Eigen::Matrix3d crossProductMatrix(const Eigen::Vector3d &v) {
        Eigen::Matrix3d M;
        M << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
        return M;
    }

  private:
    int n_, m_; // number of points and lines
    bool is_normalized_ = true;

    vector<Eigen::Vector3d> points_w_n_;
    vector<Eigen::VectorXd> lines_w_n_;
    vector<Eigen::Vector3d> control_points_w_;

    vector<Eigen::Vector3d> uv_b_;
    vector<Eigen::Vector3d> normals_b_;

    vector<int> points_cam_;
    vector<int> lines_cam_;

    vector<Eigen::Vector3d> tbc_n_;

    vector<double> alpha_;

    int num_error_ = 20;
};

} // namespace UPnPL
