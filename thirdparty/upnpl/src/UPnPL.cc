#include "UPnPL.h"
#include "constructM.h"
// #include "constructM_N2.h"
#include "constructM_N2_d2.h"
#include "constructM_N3.h"
#include "constructM_N4.h"
#include <chrono>
#include <iostream>

namespace UPnPL {

void UPnPL::solveMain(const vector<Eigen::Vector3d> &points_w,
                      const vector<Eigen::VectorXd> &lines_w,
                      const vector<Eigen::Vector3d> &uv_c,
                      const vector<Eigen::VectorXd> &lines_c,
                      const vector<int> &points_cam,
                      const vector<int> &lines_cam,
                      const vector<Eigen::Matrix3d> &Rbc,
                      const vector<Eigen::Vector3d> &tbc, Eigen::Matrix3d &R_bw,
                      Eigen::Vector3d &t_bw) {
    if (lines_w.size() == 0) {
        is_normalized_ = false;
    }
    vector<Eigen::Vector3d> normals_c(lines_c.size());
    for (size_t i = 0; i < lines_c.size(); ++i) {
        Eigen::Vector3d line_start = lines_c[i].head<3>();
        Eigen::Vector3d line_end = lines_c[i].tail<3>();
        normals_c[i] = (line_start.cross(line_end));
    }
    solveUPnPL_EPnPL(points_w, lines_w, uv_c, normals_c, points_cam, lines_cam,
                     Rbc, tbc, R_bw, t_bw);

    if (!is_normalized_)
        return;

    vector<Eigen::Matrix3d> R_cw_tmp(Rbc.size());
    vector<Eigen::Vector3d> t_cw_tmp(tbc.size());

    for (int i = 0; i < Rbc.size(); ++i) {
        R_cw_tmp[i] = Rbc[i].transpose() * R_bw;
        t_cw_tmp[i] = Rbc[i].transpose() * (t_bw - tbc[i]);
    }

    is_normalized_ = false;

    vector<Eigen::VectorXd> lines_w_corrected(lines_w.size(),
                                              Eigen::VectorXd::Zero(6));
    for (int i = 0; i < lines_w.size(); ++i) {
        int cam = lines_cam[i];
        Eigen::Vector3d Xs_w = lines_w[i].head<3>();
        Eigen::Vector3d Xe_w = lines_w[i].tail<3>();
        const auto &R_cw = R_cw_tmp[cam];
        const auto &t_cw = t_cw_tmp[cam];
        Eigen::Vector3d Xs_c_proj = R_cw * Xs_w + t_cw;
        // Rbc[cam].transpose() * (R_bw * Xs_w + t_bw - tbc[cam]);
        Eigen::Vector3d Xe_c_proj = R_cw * Xe_w + t_cw;
        // Rbc[cam].transpose() * (R_bw * Xe_w + t_bw - tbc[cam]);

        Eigen::Vector3d Xc = Xs_c_proj;
        Eigen::Vector3d V = (Xe_c_proj - Xs_c_proj).normalized();

        Xs_c_proj /= Xs_c_proj(2);
        Xe_c_proj /= Xe_c_proj(2);
        Eigen::Vector3d line_param_proj = Xs_c_proj.cross(Xe_c_proj);

        Eigen::Vector2d Xs_c_proj_plane = Xs_c_proj.head<2>();
        Eigen::Vector2d Xe_c_proj_plane = Xe_c_proj.head<2>();
        Eigen::Vector2d v = (Xe_c_proj_plane - Xs_c_proj_plane).normalized();

        Eigen::Vector3d Xs_c = lines_c[i].head<3>();
        Eigen::Vector3d Xe_c = lines_c[i].tail<3>();
        Xs_c /= Xs_c(2);
        Xe_c /= Xe_c(2);
        Eigen::Vector3d line_param = Xs_c.cross(Xe_c);

        Eigen::Vector2d Xs_c_plane = Xs_c.head<2>();
        Eigen::Vector2d Xe_c_plane = Xe_c.head<2>();
        double ds = pointLineDistance2D(Xs_c_plane, line_param_proj);
        double de = pointLineDistance2D(Xe_c_plane, line_param_proj);

        Xs_c_proj_plane =
            computeCorrectness(Xs_c_proj_plane, Xs_c_plane, v, line_param, ds);
        Xe_c_proj_plane =
            computeCorrectness(Xe_c_proj_plane, Xe_c_plane, v, line_param, de);

        lines_w_corrected[i].head<3>() =
            R_cw.transpose() *
            (backProjPointToLine(Xs_c_proj_plane, Xc, V) - t_cw);
        // R_bw.transpose() *
        // (Rbc[cam] * backProjPointToLine(Xs_c_proj_plane, Xc, V) + tbc[cam] -
        //  t_bw);
        lines_w_corrected[i].tail<3>() =
            R_cw.transpose() *
            (backProjPointToLine(Xe_c_proj_plane, Xc, V) - t_cw);
        // R_bw.transpose() *
        // (Rbc[cam] * backProjPointToLine(Xe_c_proj_plane, Xc, V) + tbc[cam] -
        //  t_bw);
    }

    solveUPnPL_EPnPL(points_w, lines_w_corrected, uv_c, normals_c, points_cam,
                     lines_cam, Rbc, tbc, R_bw, t_bw);
}

void UPnPL::solveUPnPL_DLS(const vector<Eigen::Vector3d> &points_w,
                           const vector<Eigen::VectorXd> &lines_w,
                           const vector<Eigen::Vector3d> &uv_c,
                           const vector<Eigen::Vector3d> &normals_c,
                           const vector<int> &points_cam,
                           const vector<int> &lines_cam,
                           const vector<Eigen::Matrix3d> &Rbc,
                           const vector<Eigen::Vector3d> &tbc,
                           Eigen::Matrix3d &R_bw, Eigen::Vector3d &t_bw) {
    Eigen::Vector3d true_r;
    // true_r << 0.189469, 0.189469, 0;
    // true_r << 0, 0, 0.267949;
    // true_r << -0.11346, -0.0846874, 1.82878;
    true_r << 0.175029, -0.132517, -2.5747;
    // true_r << -0.324333, 0.018495, 0.223566;
    Eigen::VectorXd true_r1(9);
    true_r1 << true_r(0), true_r(1), true_r(2), true_r(0) * true_r(0),
        true_r(1) * true_r(1), true_r(2) * true_r(2), true_r(0) * true_r(1),
        true_r(0) * true_r(2), true_r(1) * true_r(2);

    Eigen::Matrix3d R_true;
    CGR2Rotation(true_r, R_true);

    cout << "True rotation matrix R_true:\n" << R_true << endl;
    Eigen::Vector3d tbw;
    // tbw << 1.82443, 0.985914, -0.971929;
    tbw << 1.69477, 1.16269, -1.18882;
    // tbw << -0.929333, -4.4235, -0.236136;

    int n = points_w.size(), m = lines_w.size();

    double scale;
    Eigen::Vector3d center;

    vector<Eigen::Vector3d> points_w_n;
    vector<Eigen::VectorXd> lines_w_n;

    normalization(points_w, lines_w, points_w_n, lines_w_n, scale, center,
                  false);

    vector<Eigen::Vector3d> tbc_n = tbc;
    for (auto &t : tbc_n) {
        t *= scale;
    }

    vector<Eigen::Vector3d> uv_b(n);
    for (int i = 0; i < n; ++i) {
        uv_b[i] = uv_c[i];
        uv_b[i].normalize();
        uv_b[i] = Rbc[points_cam[i]] * uv_b[i];
    }

    Eigen::Matrix<double, 1, 9> I0;
    I0 << Eigen::Vector3d::Zero().transpose(),
        Eigen::Vector3d::Ones().transpose(),
        Eigen::Vector3d::Zero().transpose();

    vector<Eigen::Vector3d> normals_b(m);
    for (int i = 0; i < m; ++i) {
        normals_b[i] = Rbc[lines_cam[i]] * normals_c[i];
        normals_b[i].normalize();
    }

    double error1 = 0.0;
    for (int i = 0; i < m; ++i) {
        Eigen::Vector3d line_start = lines_w_n[i].head<3>();
        Eigen::Vector3d line_end = lines_w_n[i].tail<3>();
        double e1 =
            normals_b[i].dot(R_true * line_start + tbw - tbc_n[lines_cam[i]]);
        // e1 *= 1 + true_r.squaredNorm();
        double e2 =
            normals_b[i].dot(R_true * line_end + tbw - tbc_n[lines_cam[i]]);
        // e2 *= 1 + true_r.squaredNorm();
        error1 += e1 * e1 + e2 * e2;
    }
    cout << "r snorm: " << (1 + true_r.squaredNorm()) << endl;
    cout << "Initial error: " << error1 << endl;

    Eigen::Matrix3d H4 = n * Eigen::Matrix3d::Identity();
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(3, 2 * m + 3 * n);
    Eigen::Matrix<double, 3, 9> G_phi_i, G_phi_j;
    Eigen::Matrix<double, 3, 1> G_pi, G_pj;
    G_phi_i.setZero();
    G_phi_j.setZero();
    G_pi.setZero();
    G_pj.setZero();
    for (int i = 0; i < m; ++i) {
        Eigen::Matrix3d uuT = normals_b[i] * normals_b[i].transpose();
        H4 += 2 * uuT;
        G.block<3, 1>(0, 2 * i) = -normals_b[i];
        G.block<3, 1>(0, 2 * i + 1) = -normals_b[i];
        int cam = lines_cam[i];
        G_phi_i +=
            -uuT * (phi(lines_w_n[i].head<3>() + lines_w_n[i].tail<3>()) -
                    2 * tbc_n[cam] * I0);
        G_pi += -uuT * (lines_w_n[i].head<3>() + lines_w_n[i].tail<3>() -
                        2 * tbc_n[cam]);
    }
    for (int j = 0; j < n; ++j) {
        Eigen::Matrix3d UV = uv_b[j] * uv_b[j].transpose();
        H4 -= UV;
        Eigen::Matrix3d G_j = UV - Eigen::Matrix3d::Identity();
        G.block<3, 3>(0, 2 * m + 3 * j) = G_j;
        int cam = points_cam[j];
        G_phi_j += G_j * (phi(points_w_n[j]) - tbc_n[cam] * I0);
        G_pj += G_j * (points_w_n[j] - tbc_n[cam]);
    }
    if (H4.determinant() < 1e-6) {
        std::cerr << "H4 is singular, cannot proceed with UPnPL." << std::endl;
        return;
    }
    H4 = H4.inverse().eval();
    G = H4 * G;
    G_phi_i = H4 * G_phi_i;
    G_phi_j = H4 * G_phi_j;
    G_pi = H4 * G_pi;
    G_pj = H4 * G_pj;

    Eigen::Matrix<double, 10, 10> A = Eigen::Matrix<double, 10, 10>::Zero();

    for (int i = 0; i < m; ++i) {
        Eigen::Matrix<double, 1, 9> Ai1, Ai2;
        Ai1.setZero();
        Ai2.setZero();

        Eigen::Matrix<double, 1, 1> bi1, bi2;
        bi1.setZero();
        bi2.setZero();

        Eigen::Vector3d ui_b = normals_b[i];

        int cam = lines_cam[i];
        Ai1 = ui_b.transpose() * (phi(lines_w_n[i].head<3>()) -
                                  tbc_n[cam] * I0 + G_phi_i + G_phi_j);
        Ai2 = ui_b.transpose() * (phi(lines_w_n[i].tail<3>()) -
                                  tbc_n[cam] * I0 + G_phi_i + G_phi_j);
        bi1 = ui_b.transpose() *
              (lines_w_n[i].head<3>() - tbc_n[cam] + G_pi + G_pj);
        bi2 = ui_b.transpose() *
              (lines_w_n[i].tail<3>() - tbc_n[cam] + G_pi + G_pj);

        A.block<9, 9>(0, 0) += Ai1.transpose() * Ai1 + Ai2.transpose() * Ai2;
        A.block<9, 1>(0, 9) += Ai1.transpose() * bi1 + Ai2.transpose() * bi2;
        A.block<1, 9>(9, 0) += bi1 * Ai1 + bi2 * Ai2;
        A.block<1, 1>(9, 9) += bi1 * bi1 + bi2 * bi2;
    }

    for (int j = 0; j < n; ++j) {
        Eigen::Matrix<double, 3, 9> Aj;
        Aj.setZero();

        Eigen::Vector3d bj = Eigen::Vector3d::Zero();
        Eigen::Matrix3d PI =
            uv_b[j] * uv_b[j].transpose() - Eigen::Matrix3d::Identity();

        int cam = points_cam[j];
        Aj = PI * (G_phi_i + G_phi_j + phi(points_w_n[j]) - tbc_n[cam] * I0);
        bj = PI * (points_w_n[j] - tbc_n[cam] + G_pi + G_pj);

        A.block<9, 9>(0, 0) += Aj.transpose() * Aj;
        A.block<9, 1>(0, 9) += Aj.transpose() * bj;
        A.block<1, 9>(9, 0) += bj.transpose() * Aj;
        A.block<1, 1>(9, 9) += bj.transpose() * bj;
    }

    Eigen::Vector4d u;
    // random generation of u
    u.setRandom();
    u *= 100;

    Eigen::Matrix<double, 120, 120> M = Eigen::Matrix<double, 120, 120>::Zero();
    Eigen::Matrix<double, 27, 27> M0 = Eigen::Matrix<double, 27, 27>::Zero();
    constructM(A, u, M);

    // 10ms
    M0 = M.block<27, 27>(0, 0) - M.block<27, 93>(0, 27) *
                                     M.block<93, 93>(27, 27).inverse() *
                                     M.block<93, 27>(27, 0);

    Eigen::EigenSolver<Eigen::Matrix<double, 27, 27>> es(M0);
    es.eigenvectors();

    Eigen::MatrixXcd V = es.eigenvectors();
    // auto D = es.eigenvalues();
    vector<double> errors;
    vector<Eigen::Vector3d> r_cols;
    vector<int> r_indices;
    errors.reserve(27);
    r_cols.reserve(27);
    r_indices.reserve(27);
    int j = 0;
    for (int i = 0; i < 27; ++i) {
        Eigen::VectorXcd Vk = V.col(i);
        if (Vk(0) == complex<double>(0, 0)) {
            continue; // Skip if the first element is zero
        }
        Vk /= Vk(0); // Normalize the first element to 1
        if (Vk(1).imag() == 0.0) {
            // cout << "Vk: " << Vk.transpose() << endl;
            r_indices.push_back(j++);

            Eigen::Vector3d r;
            r << Vk(1).real(), Vk(2).real(), Vk(3).real();
            r_cols.push_back(r);

            Eigen::VectorXd r_dls;
            r_dls.resize(10);
            r_dls << r(0), r(1), r(2), r(0) * r(0), r(1) * r(1), r(2) * r(2),
                r(0) * r(1), r(0) * r(2), r(1) * r(2), 1;
            double error = r_dls.transpose() * A * r_dls;
            errors.push_back(error);
        }
    }

    sort(r_indices.begin(), r_indices.end(),
         [&errors](int a, int b) { return errors[a] < errors[b]; });

    cout << "Real columns: " << r_cols.size() << endl;
    cout << "error: " << errors[r_indices[0]] << endl;

    CGR2Rotation(r_cols[r_indices[0]], R_bw);
    cout << "r: " << r_cols[r_indices[0]].transpose() << endl;

    Eigen::VectorXd true_r_dls(10);
    true_r_dls << true_r(0), true_r(1), true_r(2), true_r(0) * true_r(0),
        true_r(1) * true_r(1), true_r(2) * true_r(2), true_r(0) * true_r(1),
        true_r(0) * true_r(2), true_r(1) * true_r(2), 1;
    double error = true_r_dls.transpose() * A * true_r_dls;
    cout << "True error: " << error << endl;
    // R_bw = R_true;

    t_bw.setZero();
    for (int i = 0; i < m; ++i) {
        Eigen::Vector3d p1 = lines_w_n[i].head<3>();
        Eigen::Vector3d p2 = lines_w_n[i].tail<3>();

        t_bw += G.block<3, 1>(0, 2 * i) * normals_b[i].transpose() *
                (R_bw * (p1 + p2) - 2 * tbc_n[lines_cam[i]]);
    }

    for (int j = 0; j < n; ++j) {
        Eigen::Vector3d p = points_w_n[j];
        t_bw +=
            G.block<3, 3>(0, 2 * m + 3 * j) * (R_bw * p - tbc_n[points_cam[j]]);
    }

    t_bw = t_bw / scale - R_bw * center;
    cout << "tbw: " << t_bw.transpose() << endl;
}

void UPnPL::solveUPnPL_EPnPL(const vector<Eigen::Vector3d> &points_w,
                             const vector<Eigen::VectorXd> &lines_w,
                             const vector<Eigen::Vector3d> &uv_c,
                             const vector<Eigen::Vector3d> &normals_c,
                             const vector<int> &points_cam,
                             const vector<int> &lines_cam,
                             const vector<Eigen::Matrix3d> &Rbc,
                             const vector<Eigen::Vector3d> &tbc,
                             Eigen::Matrix3d &R_bw, Eigen::Vector3d &t_bw) {
    // Eigen::Matrix3d R_bw_true;
    // Eigen::Vector3d t_bw_true;
    //
    // Eigen::Vector3d s;
    // s << 0, 0, 8;
    // CGR2Rotation(s, R_bw_true);
    // t_bw_true << 0, 1, 2;

    points_cam_ = points_cam;
    lines_cam_ = lines_cam;

    n_ = points_w.size(), m_ = lines_w.size();

    double scale;
    Eigen::Vector3d center;

    normalization(points_w, lines_w, points_w_n_, lines_w_n_, scale, center,
                  is_normalized_);

    tbc_n_ = tbc;
    for (auto &t : tbc_n_) {
        t *= scale;
    }

    uv_b_.resize(n_);
    for (int i = 0; i < n_; ++i) {
        uv_b_[i] = Rbc[points_cam[i]] * uv_c[i];
        // cout << "uv_b_[" << i << "]: " << uv_b_[i].transpose() << endl;
    }

    normals_b_.resize(m_);
    for (int i = 0; i < m_; ++i) {
        normals_b_[i] = Rbc[lines_cam[i]] * normals_c[i];
        normals_b_[i].normalize();
        // cout << "normals_b_[" << i << "]: " << normals_b_[i].transpose()
        //      << endl;
    }

    control_points_w_.resize(4);
    chooseControlPoints(points_w_n_, lines_w_n_, control_points_w_);
    computeAlpha(points_w_n_, lines_w_n_, control_points_w_, alpha_);

    // vector<Eigen::Vector3d> control_points_b_true(4);
    // for (int i = 0; i < 4; ++i) {
    //     control_points_b_true[i] = R_bw_true * control_points_w[i] +
    //     t_bw_true; cout << "Control point b " << i << ": "
    //          << control_points_b_true[i].transpose() << endl;
    // }

    Eigen::VectorXd y = Eigen::VectorXd::Zero(12);
    Eigen::MatrixXd AtA = Eigen::MatrixXd::Zero(12, 12);
    for (int i = 0; i < m_; ++i) {
        Eigen::Vector3d p1 = lines_w_n_[i].head<3>();
        Eigen::Vector3d p2 = lines_w_n_[i].tail<3>();

        // Eigen::MatrixXd Ai(1, 12);
        // Ai << (alpha_[i * 8 + 0] - alpha_[i * 8 + 4]) *
        //           normals_b_[i].transpose(),
        //     (alpha_[i * 8 + 1] - alpha_[i * 8 + 5]) *
        //     normals_b_[i].transpose(), (alpha_[i * 8 + 2] - alpha_[i * 8 +
        //     6]) * normals_b_[i].transpose(), (alpha_[i * 8 + 3] - alpha_[i *
        //     8 + 7]) * normals_b_[i].transpose();

        Eigen::MatrixXd Ai(2, 12);
        Ai << alpha_[i * 8 + 0] * normals_b_[i].transpose(),
            alpha_[i * 8 + 1] * normals_b_[i].transpose(),
            alpha_[i * 8 + 2] * normals_b_[i].transpose(),
            alpha_[i * 8 + 3] * normals_b_[i].transpose(),
            alpha_[i * 8 + 4] * normals_b_[i].transpose(),
            alpha_[i * 8 + 5] * normals_b_[i].transpose(),
            alpha_[i * 8 + 6] * normals_b_[i].transpose(),
            alpha_[i * 8 + 7] * normals_b_[i].transpose();

        Eigen::Vector2d bi;
        bi << normals_b_[i].dot(tbc_n_[lines_cam[i]]),
            normals_b_[i].dot(tbc_n_[lines_cam[i]]);

        // double bi = 0;

        AtA += Ai.transpose() * Ai;
        y += Ai.transpose() * bi;
    }

    for (int j = 0; j < n_; ++j) {
        Eigen::MatrixXd Aj(2, 12);
        Eigen::Vector2d bj;
        double u = uv_b_[j](0), v = uv_b_[j](1), h = uv_b_[j](2);
        bj << tbc_n_[points_cam[j]](2) * u - h * tbc_n_[points_cam[j]](0),
            tbc_n_[points_cam[j]](2) * v - h * tbc_n_[points_cam[j]](1);
        Aj << -h * alpha_[m_ * 8 + j * 4], 0, u * alpha_[m_ * 8 + j * 4],
            -h * alpha_[m_ * 8 + j * 4 + 1], 0, u * alpha_[m_ * 8 + j * 4 + 1],
            -h * alpha_[m_ * 8 + j * 4 + 2], 0, u * alpha_[m_ * 8 + j * 4 + 2],
            -h * alpha_[m_ * 8 + j * 4 + 3], 0, u * alpha_[m_ * 8 + j * 4 + 3],
            0, -h * alpha_[m_ * 8 + j * 4], v * alpha_[m_ * 8 + j * 4], 0,
            -h * alpha_[m_ * 8 + j * 4 + 1], v * alpha_[m_ * 8 + j * 4 + 1], 0,
            -h * alpha_[m_ * 8 + j * 4 + 2], v * alpha_[m_ * 8 + j * 4 + 2], 0,
            -h * alpha_[m_ * 8 + j * 4 + 3], v * alpha_[m_ * 8 + j * 4 + 3];
        AtA += Aj.transpose() * Aj;
        y += Aj.transpose() * bj;
    }
    y = AtA.ldlt().solve(y);
    // cout << "y:\n" << y.transpose() << endl;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(AtA);
    if (es.info() != Eigen::Success) {
        cerr << "Eigen decomposition failed AtA." << endl;
        return;
    }

    Eigen::MatrixXd eigenvectors = es.eigenvectors();
    Eigen::VectorXd eigenvalues = es.eigenvalues();
    // cout << "Eigenvalues:\n" << eigenvalues.transpose() << endl;

    double lambda = 0.0;

    Eigen::MatrixXd betas_N2 = Eigen::MatrixXd::Zero(12, 2);
    betas_N2 << eigenvectors.col(0), eigenvectors.col(1);

    Eigen::MatrixXd betas_N3 = Eigen::MatrixXd::Zero(12, 3);
    betas_N3 << eigenvectors.col(0), eigenvectors.col(1), eigenvectors.col(2);

    Eigen::MatrixXd betas_N4 = Eigen::MatrixXd::Zero(12, 4);
    betas_N4 << eigenvectors.col(0), eigenvectors.col(1), eigenvectors.col(2),
        eigenvectors.col(3);

    Eigen::Matrix3d R_N1, R_N2, R_N3, R_N4;
    Eigen::Vector3d t_N1, t_N2, t_N3, t_N4;

    // Eigen::Matrix3d R_N0;
    // Eigen::Vector3d t_N0;
    //
    // vector<Eigen::Vector3d> control_points_b(4);
    // control_points_b[0] << y(0), y(1), y(2);
    // control_points_b[1] << y(3), y(4), y(5);
    // control_points_b[2] << y(6), y(7), y(8);
    // control_points_b[3] << y(9), y(10), y(11);
    //
    // computePose(control_points_b, alpha_, R_N0, t_N0);
    //
    // double error_N0 =
    //     computeReprojError(R_N0, t_N0); // compute reprojection error
    // cout << "error_N0: " << error_N0 << endl;

    double error2 = numeric_limits<double>::max();
    error2 = solveN2(control_points_w_, betas_N2, y, R_N2, t_N2);
    // cout << "error2: " << error2 << endl;

    double error1 = numeric_limits<double>::max();
    error1 = solveN1(control_points_w_, eigenvectors.col(0), y, R_N1, t_N1);
    // cout << "error: " << error1 << endl;

    double error3 = numeric_limits<double>::max();
    error3 = solveN3(control_points_w_, betas_N3, y, R_N3, t_N3);
    // cout << "error3: " << error3 << endl;

    double error4 = numeric_limits<double>::max();
    error4 = solveN4(control_points_w_, betas_N4, y, R_N4, t_N4);
    // cout << "error4: " << error4 << endl;

    // select min error
    if (error1 < error2 && error1 < error3 && error1 < error4) {
        R_bw = R_N1;
        t_bw = t_N1;
    } else if (error2 < error1 && error2 < error3 && error2 < error4) {
        R_bw = R_N2;
        t_bw = t_N2;
    } else if (error3 < error1 && error3 < error2 && error3 < error4) {
        R_bw = R_N3;
        t_bw = t_N3;
    } else {
        R_bw = R_N4;
        t_bw = t_N4;
    }

    t_bw = t_bw / scale - R_bw * center;
}

double UPnPL::solveN1(const vector<Eigen::Vector3d> &control_points_w,
                      const Eigen::VectorXd &beta, const Eigen::VectorXd &y,
                      Eigen::Matrix3d &Rbw, Eigen::Vector3d &tbw) {
    Eigen::VectorXd lambdas(1), lambdas_tmp(1);
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();

    Eigen::MatrixXd Iij;
    vector<Eigen::MatrixXd> as;
    as.reserve(6);
    vector<Eigen::VectorXd> bs;
    bs.reserve(6);
    vector<double> cs;
    cs.reserve(6);
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            Iij = Eigen::MatrixXd::Zero(3, 12);
            Iij.block<3, 3>(0, i * 3) = Eigen::Matrix3d::Identity();
            Iij.block<3, 3>(0, j * 3) = -Eigen::Matrix3d::Identity();

            double a = beta.transpose() * Iij.transpose() * Iij * beta;
            double b = 2 * y.transpose() * Iij.transpose() * Iij * beta;
            double c =
                y.transpose() * Iij.transpose() * Iij * y -
                (control_points_w[i] - control_points_w[j]).squaredNorm();

            Eigen::Vector3d Hij = Eigen::Vector3d::Zero();
            Hij << a, b, c;
            as.emplace_back((Eigen::MatrixXd(1, 1) << a).finished());
            bs.emplace_back((Eigen::VectorXd(1) << b).finished());
            cs.emplace_back(c);

            H += Hij * Hij.transpose();
        }
    }

    Eigen::Matrix4d M = Eigen::Matrix4d::Zero();

    Eigen::Vector2d u;
    // random generation of u
    u.setRandom();
    u *= 100;

    M(0, 0) = u(0);
    M(0, 1) = u(1);
    M(1, 1) = u(0);
    M(1, 2) = u(1);
    M(2, 2) = u(0);
    M(2, 3) = u(1);
    M(3, 0) = 2 * H(2, 1);
    M(3, 1) = 2 * H(1, 1) + 4 * H(2, 0);
    M(3, 2) = 2 * H(0, 1) + 4 * H(1, 0);
    M(3, 3) = 4 * H(0, 0);

    Eigen::Matrix3d M0 = Eigen::Matrix3d::Zero();
    M0 = M.block<3, 3>(0, 0) - M.block<3, 1>(0, 3) *
                                   M.block<1, 1>(3, 3).inverse() *
                                   M.block<1, 3>(3, 0);

    Eigen::EigenSolver<Eigen::Matrix3d> es(M0);
    if (es.info() != Eigen::Success) {
        // cerr << "Eigen decomposition failed." << endl;
        return numeric_limits<double>::max();
    }

    int total = m_ + n_;
    vector<int> index(total);
    iota(index.begin(), index.end(), 0); // Fill index with 0, 1, ..., total-1

    random_device rd;
    mt19937 g(rd());
    shuffle(index.begin(), index.end(), g); // Shuffle the indices
    index.resize(min(total, num_error_));

    Eigen::MatrixXcd V = es.eigenvectors();
    Eigen::Matrix3d Rbw_tmp;
    Eigen::Vector3d tbw_tmp;
    vector<Eigen::Vector3d> control_points_b(4);
    Eigen::VectorXd y1;
    double min_error = numeric_limits<double>::max();
    int j = 0;
    for (int i = 0; i < 3; ++i) {
        Eigen::VectorXcd Vk = V.col(i);
        if (Vk(0) == complex<double>(0, 0)) {
            continue; // Skip if the first element is zero
        }
        Vk /= Vk(0); // Normalize the first element to 1
        if (Vk(1).imag() < 1e-3) {
            lambdas_tmp(0) = Vk(1).real();

            y1 = y + lambdas_tmp(0) * beta;

            control_points_b[0] << y1(0), y1(1), y1(2);
            control_points_b[1] << y1(3), y1(4), y1(5);
            control_points_b[2] << y1(6), y1(7), y1(8);
            control_points_b[3] << y1(9), y1(10), y1(11);

            // computePose(index, control_points_b, alpha_, Rbw_tmp, tbw_tmp);
            computePose(control_points_b, Rbw_tmp, tbw_tmp);

            double error = computeReprojError(index, Rbw_tmp, tbw_tmp);
            // cout << "error: " << error << endl;
            if (error < min_error) {
                min_error = error;
                // Rbw = Rbw_tmp;
                // tbw = tbw_tmp;
                lambdas = lambdas_tmp;
            }
        }
    }

    lambdas_tmp = lambdas;
    if (lambdaRefine(as, bs, cs, lambdas_tmp)) {
        lambdas = lambdas_tmp;
    }

    y1 = y + lambdas(0) * beta;
    control_points_b[0] << y1(0), y1(1), y1(2);
    control_points_b[1] << y1(3), y1(4), y1(5);
    control_points_b[2] << y1(6), y1(7), y1(8);
    control_points_b[3] << y1(9), y1(10), y1(11);

    computePose(control_points_b, alpha_, Rbw, tbw);

    return computeReprojError(Rbw, tbw);
}

// double UPnPL::solveN2(const vector<Eigen::Vector3d> &control_points_w,
//                       const Eigen::MatrixXd &beta, const Eigen::VectorXd &y,
//                       Eigen::Matrix3d &Rbw, Eigen::Vector3d &tbw) {
//     Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
//     Eigen::VectorXd lambdas(2), lambdas_tmp(2);
//
//     Eigen::MatrixXd Iij;
//     vector<Eigen::MatrixXd> as;
//     vector<Eigen::VectorXd> bs;
//     vector<double> cs;
//
//     for (int i = 0; i < 4; ++i) {
//         for (int j = i + 1; j < 4; ++j) {
//             Iij = Eigen::MatrixXd::Zero(3, 12);
//             Iij.block<3, 3>(0, i * 3) = Eigen::Matrix3d::Identity();
//             Iij.block<3, 3>(0, j * 3) = -Eigen::Matrix3d::Identity();
//
//             Eigen::MatrixXd a = beta.transpose() * Iij.transpose() * Iij *
//             beta; Eigen::VectorXd b =
//                 2 * y.transpose() * Iij.transpose() * Iij * beta;
//             double c =
//                 y.transpose() * Iij.transpose() * Iij * y -
//                 (control_points_w[i] - control_points_w[j]).squaredNorm();
//             Eigen::VectorXd Hij(6);
//             Hij << a(0, 0), a(1, 1), a(0, 1) + a(1, 0), b(0), b(1), c;
//             as.emplace_back(a);
//             bs.emplace_back(b);
//             cs.emplace_back(c);
//
//             H += Hij * Hij.transpose();
//         }
//     }
//
//     Eigen::MatrixXd M = Eigen::MatrixXd::Zero(21, 21);
//
//     Eigen::Vector3d u;
//     u.setRandom();
//     u *= 100;
//
//     constructM_N2(H, u, M);
//     Eigen::MatrixXd M0 =
//         M.block<9, 9>(0, 0) - M.block<9, 12>(0, 9) *
//                                   M.block<12, 12>(9, 9).inverse() *
//                                   M.block<12, 9>(9, 0);
//
//     Eigen::EigenSolver<Eigen::MatrixXd> es(M0);
//     if (es.info() != Eigen::Success) {
//         cerr << "Eigen decomposition failed." << endl;
//         return numeric_limits<double>::max();
//     }
//
//     Eigen::MatrixXcd V = es.eigenvectors();
//     Eigen::Matrix3d Rbw_tmp;
//     Eigen::Vector3d tbw_tmp;
//     vector<Eigen::Vector3d> control_points_b(4);
//     Eigen::VectorXd y1;
//     double min_error = numeric_limits<double>::max();
//     int j = 0;
//     for (int i = 0; i < 9; ++i) {
//         Eigen::VectorXcd Vk = V.col(i);
//         if (Vk(0) == complex<double>(0, 0)) {
//             continue; // Skip if the first element is zero
//         }
//         Vk /= Vk(0); // Normalize the first element to 1
//         if (Vk(1).imag() == 0.0) {
//             // cout << "Vk: " << Vk.transpose() << endl;
//
//             lambdas(0) = Vk(1).real();
//             lambdas(1) = Vk(2).real();
//             lambdas_tmp = lambdas;
//             if (lambdaRefine(as, bs, cs, lambdas_tmp))
//                 lambdas = lambdas_tmp;
//             // cout << "lambdas: " << lambdas.transpose() << endl;
//
//             y1 = y + lambdas(0) * beta.col(0) + lambdas(1) * beta.col(1);
//
//             control_points_b[0] << y1(0), y1(1), y1(2);
//             control_points_b[1] << y1(3), y1(4), y1(5);
//             control_points_b[2] << y1(6), y1(7), y1(8);
//             control_points_b[3] << y1(9), y1(10), y1(11);
//
//             computePose(control_points_b, alpha_, Rbw_tmp, tbw_tmp);
//
//             double error = computeReprojError(Rbw_tmp, tbw_tmp);
//             // cout << "error: " << error << endl;
//             if (error < min_error) {
//                 min_error = error;
//                 Rbw = Rbw_tmp;
//                 tbw = tbw_tmp;
//                 lambdas = lambdas;
//             }
//         }
//     }
//
//     return min_error;
// }

double UPnPL::solveN2(const vector<Eigen::Vector3d> &control_points_w,
                      const Eigen::MatrixXd &beta, const Eigen::VectorXd &y,
                      Eigen::Matrix3d &Rbw, Eigen::Vector3d &tbw) {
    Eigen::VectorXd lambdas(2), lambdas_tmp(2);

    Eigen::MatrixXd Iij;
    vector<Eigen::MatrixXd> as, as_M;
    as.reserve(6);
    vector<Eigen::VectorXd> bs, bs_M;
    bs.reserve(6);
    vector<double> cs, cs_M;
    cs.reserve(6);

    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            Iij = Eigen::MatrixXd::Zero(3, 12);
            Iij.block<3, 3>(0, i * 3) = Eigen::Matrix3d::Identity();
            Iij.block<3, 3>(0, j * 3) = -Eigen::Matrix3d::Identity();

            Eigen::MatrixXd a = beta.transpose() * Iij.transpose() * Iij * beta;
            Eigen::VectorXd b =
                2 * y.transpose() * Iij.transpose() * Iij * beta;
            double c =
                y.transpose() * Iij.transpose() * Iij * y -
                (control_points_w[i] - control_points_w[j]).squaredNorm();
            as.emplace_back(a);
            bs.emplace_back(b);
            cs.emplace_back(c);
        }
    }

    as_M.resize(2);
    as_M[0] = as[0] + as[1] + as[2];
    as_M[1] = as[3] + as[4] + as[5];
    bs_M.resize(2);
    bs_M[0] = bs[0] + bs[1] + bs[2];
    bs_M[1] = bs[3] + bs[4] + bs[5];
    cs_M.resize(2);
    cs_M[0] = cs[0] + cs[1] + cs[2];
    cs_M[1] = cs[3] + cs[4] + cs[5];

    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(10, 10);

    Eigen::VectorXd u(3);
    // random generation of u
    u.setRandom();
    u *= 100;

    constructM_N2(as_M, bs_M, cs_M, u, M);
    Eigen::MatrixXd M_inv_block =
        M.block<6, 6>(4, 4).partialPivLu().solve(M.block<6, 4>(4, 0));
    Eigen::MatrixXd M0 =
        M.block<4, 4>(0, 0) - M.block<4, 6>(0, 4) * M_inv_block;

    Eigen::EigenSolver<Eigen::MatrixXd> es(M0);
    if (es.info() != Eigen::Success) {
        // cerr << "Eigen decomposition failed N2." << endl;
        return numeric_limits<double>::max();
    }

    vector<int> index(m_ + n_);
    iota(index.begin(), index.end(), 0); // Fill index with 0, 1, ..., total-1
    random_device rd;
    mt19937 g(rd());
    shuffle(index.begin(), index.end(), g); // Shuffle the indices
    index.resize(min(num_error_, m_ + n_));

    Eigen::MatrixXcd V = es.eigenvectors();
    Eigen::Matrix3d Rbw_tmp;
    Eigen::Vector3d tbw_tmp;
    vector<Eigen::Vector3d> control_points_b(4);
    Eigen::VectorXd y1;
    double min_error = numeric_limits<double>::max();
    int j = 0;
    for (int i = 0; i < 4; ++i) {
        Eigen::VectorXcd Vk = V.col(i);
        if (Vk(0) == complex<double>(0, 0)) {
            continue; // Skip if the first element is zero
        }
        Vk /= Vk(0); // Normalize the first element to 1
        if (Vk(1).imag() < 1e-3) {
            // cout << "Vk: " << Vk.transpose() << endl;

            lambdas_tmp(0) = Vk(1).real();
            lambdas_tmp(1) = Vk(2).real();

            y1 =
                y + lambdas_tmp(0) * beta.col(0) + lambdas_tmp(1) * beta.col(1);
            // cout << "y1: " << y1.transpose() << endl;

            control_points_b[0] << y1(0), y1(1), y1(2);
            control_points_b[1] << y1(3), y1(4), y1(5);
            control_points_b[2] << y1(6), y1(7), y1(8);
            control_points_b[3] << y1(9), y1(10), y1(11);

            // computePose(index, control_points_b, alpha_, Rbw_tmp, tbw_tmp);
            computePose(control_points_b, Rbw_tmp, tbw_tmp);

            double error = computeReprojError(index, Rbw_tmp, tbw_tmp);
            // cout << "error: " << error << endl;
            if (error < min_error) {
                min_error = error;
                // Rbw = Rbw_tmp;
                // tbw = tbw_tmp;
                lambdas = lambdas_tmp;
            }
        }
    }
    lambdas_tmp = lambdas;
    if (lambdaRefine(as, bs, cs, lambdas_tmp)) {
        lambdas = lambdas_tmp;
    }
    y1 = y + lambdas(0) * beta.col(0) + lambdas(1) * beta.col(1);
    control_points_b[0] << y1(0), y1(1), y1(2);
    control_points_b[1] << y1(3), y1(4), y1(5);
    control_points_b[2] << y1(6), y1(7), y1(8);
    control_points_b[3] << y1(9), y1(10), y1(11);
    computePose(control_points_b, alpha_, Rbw, tbw);
    return computeReprojError(Rbw, tbw);
}

double UPnPL::solveN4(const vector<Eigen::Vector3d> &control_points_w,
                      const Eigen::MatrixXd &beta, const Eigen::VectorXd &y,
                      Eigen::Matrix3d &Rbw, Eigen::Vector3d &tbw) {
    Eigen::VectorXd lambdas(4), lambdas_tmp(4);

    Eigen::MatrixXd Iij;
    vector<Eigen::MatrixXd> as, as_M;
    as.reserve(6);
    vector<Eigen::VectorXd> bs, bs_M;
    bs.reserve(6);
    vector<double> cs, cs_M;
    cs.reserve(6);

    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            Iij = Eigen::MatrixXd::Zero(3, 12);
            Iij.block<3, 3>(0, i * 3) = Eigen::Matrix3d::Identity();
            Iij.block<3, 3>(0, j * 3) = -Eigen::Matrix3d::Identity();

            Eigen::MatrixXd a = beta.transpose() * Iij.transpose() * Iij * beta;
            Eigen::VectorXd b =
                2 * y.transpose() * Iij.transpose() * Iij * beta;
            double c =
                y.transpose() * Iij.transpose() * Iij * y -
                (control_points_w[i] - control_points_w[j]).squaredNorm();
            as.emplace_back(a);
            bs.emplace_back(b);
            cs.emplace_back(c);
        }
    }

    as_M.resize(4);
    as_M[0] = as[0];
    as_M[1] = as[1];
    as_M[2] = as[2] + as[3];
    as_M[3] = as[4] + as[5];
    bs_M.resize(4);
    bs_M[0] = bs[0];
    bs_M[1] = bs[1];
    bs_M[2] = bs[2] + bs[3];
    bs_M[3] = bs[4] + bs[5];
    cs_M.resize(4);
    cs_M[0] = cs[0];
    cs_M[1] = cs[1];
    cs_M[2] = cs[2] + cs[3];
    cs_M[3] = cs[4] + cs[5];

    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(126, 126);

    Eigen::VectorXd u(5);
    u.setRandom();
    u *= 100;

    constructM_N4(as_M, bs_M, cs_M, u, M);
    Eigen::MatrixXd M_inv_block =
        M.block<110, 110>(16, 16).partialPivLu().solve(M.block<110, 16>(16, 0));
    Eigen::MatrixXd M0 =
        M.block<16, 16>(0, 0) - M.block<16, 110>(0, 16) * M_inv_block;
    // Eigen::MatrixXd M0 =
    //     M.block<16, 16>(0, 0) - M.block<16, 110>(0, 16) *
    //                                 M.block<110, 110>(16, 16).inverse() *
    //                                 M.block<110, 16>(16, 0);
    Eigen::EigenSolver<Eigen::MatrixXd> es(M0);
    if (es.info() != Eigen::Success) {
        // cerr << "Eigen decomposition failed N4." << endl;
        return numeric_limits<double>::max();
    }

    vector<int> index(m_ + n_);
    iota(index.begin(), index.end(), 0); // Fill index with 0, 1, ..., total-1
    random_device rd;
    mt19937 g(rd());
    shuffle(index.begin(), index.end(), g); // Shuffle the indices
    index.resize(min(num_error_, m_ + n_));

    Eigen::MatrixXcd V = es.eigenvectors();
    Eigen::Matrix3d Rbw_tmp;
    Eigen::Vector3d tbw_tmp;
    vector<Eigen::Vector3d> control_points_b(4);
    Eigen::VectorXd y1;
    double min_error = numeric_limits<double>::max();
    int j = 0;
    for (int i = 0; i < 16; ++i) {
        Eigen::VectorXcd Vk = V.col(i);
        if (Vk(0) == complex<double>(0, 0)) {
            continue; // Skip
        }
        Vk /= Vk(0); // Normalize the first element to 1
        if (Vk(1).imag() < 1e-3) {

            lambdas_tmp(0) = Vk(1).real();
            lambdas_tmp(1) = Vk(2).real();
            lambdas_tmp(2) = Vk(3).real();
            lambdas_tmp(3) = Vk(4).real();
            // cout << "lambdas: " << lambdas.transpose() << endl;

            y1 = y + lambdas_tmp(0) * beta.col(0) +
                 lambdas_tmp(1) * beta.col(1) + lambdas_tmp(2) * beta.col(2) +
                 lambdas_tmp(3) * beta.col(3);
            // cout << "y1: " << y1.transpose() << endl;

            control_points_b[0] << y1(0), y1(1), y1(2);
            control_points_b[1] << y1(3), y1(4), y1(5);
            control_points_b[2] << y1(6), y1(7), y1(8);
            control_points_b[3] << y1(9), y1(10), y1(11);

            computePose(index, control_points_b, alpha_, Rbw_tmp, tbw_tmp);

            double error = computeReprojError(index, Rbw_tmp, tbw_tmp);
            // cout << "error: " << error << endl;
            if (error < min_error) {
                min_error = error;
                // Rbw = Rbw_tmp;
                // tbw = tbw_tmp;
                lambdas = lambdas_tmp;
            }
        }
    }

    lambdas_tmp = lambdas;
    if (lambdaRefine(as, bs, cs, lambdas_tmp)) {
        lambdas = lambdas_tmp;
    }

    y1 = y + lambdas(0) * beta.col(0) + lambdas(1) * beta.col(1) +
         lambdas(2) * beta.col(2) + lambdas(3) * beta.col(3);
    control_points_b[0] << y1(0), y1(1), y1(2);
    control_points_b[1] << y1(3), y1(4), y1(5);
    control_points_b[2] << y1(6), y1(7), y1(8);
    control_points_b[3] << y1(9), y1(10), y1(11);
    computePose(control_points_b, alpha_, Rbw, tbw);
    return computeReprojError(Rbw, tbw);
}

double UPnPL::solveN3(const vector<Eigen::Vector3d> &control_points_w,
                      const Eigen::MatrixXd &beta, const Eigen::VectorXd &y,
                      Eigen::Matrix3d &Rbw, Eigen::Vector3d &tbw) {
    Eigen::VectorXd lambdas(3), lambdas_tmp(3);

    Eigen::MatrixXd Iij;
    vector<Eigen::MatrixXd> as, as_M;
    as.reserve(6);
    vector<Eigen::VectorXd> bs, bs_M;
    bs.reserve(6);
    vector<double> cs, cs_M;
    cs.reserve(6);

    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            Iij = Eigen::MatrixXd::Zero(3, 12);
            Iij.block<3, 3>(0, i * 3) = Eigen::Matrix3d::Identity();
            Iij.block<3, 3>(0, j * 3) = -Eigen::Matrix3d::Identity();

            Eigen::MatrixXd a = beta.transpose() * Iij.transpose() * Iij * beta;
            Eigen::VectorXd b =
                2 * y.transpose() * Iij.transpose() * Iij * beta;
            double c =
                y.transpose() * Iij.transpose() * Iij * y -
                (control_points_w[i] - control_points_w[j]).squaredNorm();
            as.emplace_back(a);
            bs.emplace_back(b);
            cs.emplace_back(c);
        }
    }

    as_M.resize(3);
    as_M[0] = as[0] + as[1];
    as_M[1] = as[2] + as[3];
    as_M[2] = as[4] + as[5];
    bs_M.resize(3);
    bs_M[0] = bs[0] + bs[1];
    bs_M[1] = bs[2] + bs[3];
    bs_M[2] = bs[4] + bs[5];
    cs_M.resize(3);
    cs_M[0] = cs[0] + cs[1];
    cs_M[1] = cs[2] + cs[3];
    cs_M[2] = cs[4] + cs[5];

    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(35, 35);

    Eigen::VectorXd u(4);
    u.setRandom();
    u *= 100;

    constructM_N3(as_M, bs_M, cs_M, u, M);
    Eigen::MatrixXd M_inv_block =
        M.block<27, 27>(8, 8).partialPivLu().solve(M.block<27, 8>(8, 0));
    Eigen::MatrixXd M0 =
        M.block<8, 8>(0, 0) - M.block<8, 27>(0, 8) * M_inv_block;
    // Eigen::MatrixXd M0 =
    //     M.block<8, 8>(0, 0) - M.block<8, 27>(0, 8) *
    //                               M.block<27, 27>(8, 8).inverse() *
    //                               M.block<27, 8>(8, 0);

    Eigen::EigenSolver<Eigen::MatrixXd> es(M0);
    if (es.info() != Eigen::Success) {
        // cerr << "Eigen decomposition failed N3." << endl;
        return numeric_limits<double>::max();
    }

    vector<int> index(m_ + n_);
    iota(index.begin(), index.end(), 0); // Fill index with 0, 1, ..., total-1
    random_device rd;
    mt19937 g(rd());
    shuffle(index.begin(), index.end(), g); // Shuffle the indices
    index.resize(min(num_error_, m_ + n_));

    Eigen::MatrixXcd V = es.eigenvectors();
    Eigen::Matrix3d Rbw_tmp;
    Eigen::Vector3d tbw_tmp;
    vector<Eigen::Vector3d> control_points_b(4);
    Eigen::VectorXd y1;
    double min_error = numeric_limits<double>::max();
    int j = 0;
    for (int i = 0; i < 8; ++i) {
        Eigen::VectorXcd Vk = V.col(i);
        if (Vk(0) == complex<double>(0, 0)) {
            continue; // Skip if the first element is zero
        }
        Vk /= Vk(0); // Normalize the first element to 1
        if (Vk(1).imag() < 1e-3) {
            // cout << "Vk: " << Vk.transpose() << endl;

            lambdas_tmp(0) = Vk(1).real();
            lambdas_tmp(1) = Vk(2).real();
            lambdas_tmp(2) = Vk(3).real();

            y1 = y + lambdas_tmp(0) * beta.col(0) +
                 lambdas_tmp(1) * beta.col(1) + lambdas_tmp(2) * beta.col(2);

            control_points_b[0] << y1(0), y1(1), y1(2);
            control_points_b[1] << y1(3), y1(4), y1(5);
            control_points_b[2] << y1(6), y1(7), y1(8);
            control_points_b[3] << y1(9), y1(10), y1(11);

            // computePose(index, control_points_b, alpha_, Rbw_tmp, tbw_tmp);
            computePose(control_points_b, Rbw_tmp, tbw_tmp);

            double error = computeReprojError(index, Rbw_tmp, tbw_tmp);
            // cout << "error: " << error << endl;
            if (error < min_error) {
                min_error = error;
                // Rbw = Rbw_tmp;
                // tbw = tbw_tmp;
                lambdas = lambdas_tmp;
            }
        }
    }
    lambdas_tmp = lambdas;
    if (lambdaRefine(as, bs, cs, lambdas_tmp)) {
        lambdas = lambdas_tmp;
    }
    y1 = y + lambdas(0) * beta.col(0) + lambdas(1) * beta.col(1) +
         lambdas(2) * beta.col(2);
    control_points_b[0] << y1(0), y1(1), y1(2);
    control_points_b[1] << y1(3), y1(4), y1(5);
    control_points_b[2] << y1(6), y1(7), y1(8);
    control_points_b[3] << y1(9), y1(10), y1(11);
    computePose(control_points_b, alpha_, Rbw, tbw);
    return computeReprojError(Rbw, tbw);
}

bool UPnPL::lambdaRefine(const vector<Eigen::MatrixXd> &a,
                         const vector<Eigen::VectorXd> &b, vector<double> &c,
                         Eigen::VectorXd &lambda) {
    int dim = lambda.size();

    ceres::Problem problem;
    int k = 0;
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            ceres::DynamicAutoDiffCostFunction<LambdaCost> *cost_function =
                new ceres::DynamicAutoDiffCostFunction<LambdaCost>(
                    new LambdaCost(a[k], b[k], c[k]));
            cost_function->AddParameterBlock(dim);
            cost_function->SetNumResiduals(1);
            problem.AddResidualBlock(cost_function, nullptr, lambda.data());
            k++;
        }
    }

    const int max_iter = 10;
    ceres::Solver::Options options;
    options.max_num_iterations = max_iter;
    // options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    return !(summary.iterations.size() == max_iter + 1);
}

double UPnPL::computeReprojError(const Eigen::Matrix3d &R_bw,
                                 const Eigen::Vector3d &t_bw) {
    double reproj_error = 0.0;

    for (int i = 0; i < m_; ++i) {
        Eigen::Vector3d p1 = lines_w_n_[i].head<3>();
        Eigen::Vector3d p2 = lines_w_n_[i].tail<3>();

        Eigen::Vector3d line_start_b = R_bw * p1 + t_bw;
        Eigen::Vector3d line_end_b = R_bw * p2 + t_bw;

        reproj_error +=
            abs(normals_b_[i].dot(line_start_b - tbc_n_[lines_cam_[i]]) *
                normals_b_[i].dot(line_end_b - tbc_n_[lines_cam_[i]]));
    }

    for (int j = 0; j < n_; ++j) {
        Eigen::Vector3d p = points_w_n_[j];
        Eigen::Vector3d point_b = R_bw * p + t_bw;

        reproj_error += (point_b - tbc_n_[points_cam_[j]])
                            .cross(uv_b_[j].normalized())
                            .squaredNorm();
    }

    return reproj_error;
}

double UPnPL::computeReprojError(const vector<int> &index,
                                 const Eigen::Matrix3d &R_bw,
                                 const Eigen::Vector3d &t_bw) {
    double reproj_error = 0.0;

    for (const int &i : index) {
        if (i < m_) {
            Eigen::Vector3d p1 = lines_w_n_[i].head<3>();
            Eigen::Vector3d p2 = lines_w_n_[i].tail<3>();

            Eigen::Vector3d line_start_b = R_bw * p1 + t_bw;
            Eigen::Vector3d line_end_b = R_bw * p2 + t_bw;

            reproj_error +=
                abs(normals_b_[i].dot(line_start_b - tbc_n_[lines_cam_[i]]) *
                    normals_b_[i].dot(line_end_b - tbc_n_[lines_cam_[i]]));
        } else {
            int j = i - m_;
            Eigen::Vector3d p = points_w_n_[j];
            Eigen::Vector3d point_b = R_bw * p + t_bw;

            reproj_error += (point_b - tbc_n_[points_cam_[j]])
                                .cross(uv_b_[j].normalized())
                                .squaredNorm();
        }
    }

    return reproj_error;
}

void UPnPL::computePose(const vector<Eigen::Vector3d> &control_points_b,
                        const vector<double> &alpha, Eigen::Matrix3d &R_bw,
                        Eigen::Vector3d &t_bw) {
    vector<Eigen::Vector3d> points_b(n_);
    vector<Eigen::VectorXd> lines_b(m_);

    Eigen::Vector3d pb0 = Eigen::Vector3d::Zero();
    Eigen::Vector3d pw0 = Eigen::Vector3d::Zero();

    for (int i = 0; i < m_; ++i) {
        Eigen::Vector3d p1 = alpha[i * 8] * control_points_b[0] +
                             alpha[i * 8 + 1] * control_points_b[1] +
                             alpha[i * 8 + 2] * control_points_b[2] +
                             alpha[i * 8 + 3] * control_points_b[3];
        Eigen::Vector3d p2 = alpha[i * 8 + 4] * control_points_b[0] +
                             alpha[i * 8 + 5] * control_points_b[1] +
                             alpha[i * 8 + 6] * control_points_b[2] +
                             alpha[i * 8 + 7] * control_points_b[3];
        lines_b[i].resize(6);
        lines_b[i].head<3>() = p1;
        lines_b[i].tail<3>() = p2;

        Eigen::Vector3d pw1 = lines_w_n_[i].head<3>();
        Eigen::Vector3d pw2 = lines_w_n_[i].tail<3>();

        pb0 += p1;
        pb0 += p2;

        pw0 += pw1;
        pw0 += pw2;
    }

    for (int j = 0; j < n_; ++j) {
        Eigen::Vector3d p = alpha[m_ * 8 + j * 4] * control_points_b[0] +
                            alpha[m_ * 8 + j * 4 + 1] * control_points_b[1] +
                            alpha[m_ * 8 + j * 4 + 2] * control_points_b[2] +
                            alpha[m_ * 8 + j * 4 + 3] * control_points_b[3];
        points_b[j] = p;

        pb0 += p;
        pw0 += points_w_n_[j];
    }

    pb0 /= (n_ + 2 * m_);
    pw0 /= (n_ + 2 * m_);

    Eigen::Matrix3d ABt = Eigen::Matrix3d::Zero();

    for (int i = 0; i < m_; ++i) {
        Eigen::Vector3d pb1 = lines_b[i].head<3>();
        Eigen::Vector3d pb2 = lines_b[i].tail<3>();

        Eigen::Vector3d pw1 = lines_w_n_[i].head<3>();
        Eigen::Vector3d pw2 = lines_w_n_[i].tail<3>();

        ABt += (pb1 - pb0) * (pw1 - pw0).transpose();
        ABt += (pb2 - pb0) * (pw2 - pw0).transpose();
    }

    for (int j = 0; j < n_; ++j) {
        Eigen::Vector3d pb = points_b[j];
        Eigen::Vector3d pw = points_w_n_[j];

        ABt += (pb - pb0) * (pw - pw0).transpose();
    }

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(ABt, Eigen::ComputeFullU |
                                                   Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    R_bw = U * V.transpose();

    if (R_bw.determinant() < 0) {
        V.col(2) *= -1;
        R_bw = U * V.transpose();
    }
    // cout << "R_bw:\n" << R_bw << endl;
    t_bw = pb0 - R_bw * pw0;
}

void UPnPL::computePose(const vector<int> &index,
                        const vector<Eigen::Vector3d> &control_points_b,
                        const vector<double> &alpha, Eigen::Matrix3d &R_bw,
                        Eigen::Vector3d &t_bw) {
    Eigen::Vector3d pb0 = Eigen::Vector3d::Zero();
    Eigen::Vector3d pw0 = Eigen::Vector3d::Zero();

    vector<Eigen::Vector3d> line_endpoints;
    vector<int> lines_index_w;
    vector<Eigen::Vector3d> points;
    vector<int> points_index_w;
    line_endpoints.reserve(2 * index.size());
    lines_index_w.reserve(index.size());
    points.reserve(index.size());
    points_index_w.reserve(index.size());

    for (const int &i : index) {
        if (i < m_) {
            Eigen::Vector3d p1 = alpha[i * 8] * control_points_b[0] +
                                 alpha[i * 8 + 1] * control_points_b[1] +
                                 alpha[i * 8 + 2] * control_points_b[2] +
                                 alpha[i * 8 + 3] * control_points_b[3];
            Eigen::Vector3d p2 = alpha[i * 8 + 4] * control_points_b[0] +
                                 alpha[i * 8 + 5] * control_points_b[1] +
                                 alpha[i * 8 + 6] * control_points_b[2] +
                                 alpha[i * 8 + 7] * control_points_b[3];
            pb0 += p1;
            pb0 += p2;
            line_endpoints.push_back(p1);
            line_endpoints.push_back(p2);

            Eigen::Vector3d pw1 = lines_w_n_[i].head<3>();
            Eigen::Vector3d pw2 = lines_w_n_[i].tail<3>();
            lines_index_w.push_back(i);
            pw0 += pw1;
            pw0 += pw2;
        } else {
            int j = i - m_;
            Eigen::Vector3d p =
                alpha[m_ * 8 + j * 4] * control_points_b[0] +
                alpha[m_ * 8 + j * 4 + 1] * control_points_b[1] +
                alpha[m_ * 8 + j * 4 + 2] * control_points_b[2] +
                alpha[m_ * 8 + j * 4 + 3] * control_points_b[3];
            pb0 += p;
            points.push_back(p);

            Eigen::Vector3d pw = points_w_n_[j];
            points_index_w.push_back(j);
            pw0 += pw;
        }
    }

    int num = line_endpoints.size() + points.size();
    pb0 /= num;
    pw0 /= num;

    Eigen::Matrix3d ABt = Eigen::Matrix3d::Zero();

    for (size_t i = 0; i < line_endpoints.size(); i += 2) {
        Eigen::Vector3d pb1 = line_endpoints[i];
        Eigen::Vector3d pb2 = line_endpoints[i + 1];

        Eigen::Vector3d pw1 = lines_w_n_[lines_index_w[i / 2]].head<3>();
        Eigen::Vector3d pw2 = lines_w_n_[lines_index_w[i / 2]].tail<3>();

        ABt += (pb1 - pb0) * (pw1 - pw0).transpose();
        ABt += (pb2 - pb0) * (pw2 - pw0).transpose();
    }

    for (size_t j = 0; j < points.size(); ++j) {
        Eigen::Vector3d pb = points[j];
        Eigen::Vector3d pw = points_w_n_[points_index_w[j]];

        ABt += (pb - pb0) * (pw - pw0).transpose();
    }

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(ABt, Eigen::ComputeFullU |
                                                   Eigen::ComputeFullV);

    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    R_bw = U * V.transpose();

    if (R_bw.determinant() < 0) {
        V.col(2) *= -1;
        R_bw = U * V.transpose();
    }

    // cout << "R_bw:\n" << R_bw << endl;
    t_bw = pb0 - R_bw * pw0;
}

void UPnPL::computePose(const vector<Eigen::Vector3d> &control_points_b,
                        Eigen::Matrix3d &R_bw, Eigen::Vector3d &t_bw) {
    Eigen::Vector3d pb0 = Eigen::Vector3d::Zero();
    Eigen::Vector3d pw0 = Eigen::Vector3d::Zero();

    for (int i = 0; i < 4; ++i) {
        pb0 += control_points_b[i];
        pw0 += control_points_w_[i];
    }

    pb0 /= 4;
    pw0 /= 4;

    Eigen::Matrix3d ABt = Eigen::Matrix3d::Zero();

    for (int i = 0; i < 4; ++i) {
        Eigen::Vector3d pb = control_points_b[i];
        Eigen::Vector3d pw = control_points_w_[i];

        ABt += (pb - pb0) * (pw - pw0).transpose();
    }

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(ABt, Eigen::ComputeFullU |
                                                   Eigen::ComputeFullV);

    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    R_bw = U * V.transpose();

    if (R_bw.determinant() < 0) {
        V.col(2) *= -1;
        R_bw = U * V.transpose();
    }

    t_bw = pb0 - R_bw * pw0;
}

void UPnPL::chooseControlPoints(const vector<Eigen::Vector3d> &points_w,
                                const vector<Eigen::VectorXd> &lines_w,
                                vector<Eigen::Vector3d> &control_points) {
    control_points.resize(4);
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();

    int n = points_w.size(), m = lines_w.size();

    for (const auto &p : points_w) {
        centroid += p;
    }

    for (const auto &l : lines_w) {
        Eigen::Vector3d p1 = l.head<3>();
        Eigen::Vector3d p2 = l.tail<3>();
        centroid += p1;
        centroid += p2;
    }

    centroid /= (n + 2 * m);
    control_points[0] = centroid;

    Eigen::MatrixXd PW0 = Eigen::MatrixXd::Zero(n + 2 * m, 3);

    for (int i = 0; i < m; ++i) {
        Eigen::Vector3d p1 = lines_w[i].head<3>();
        Eigen::Vector3d p2 = lines_w[i].tail<3>();

        PW0.row(i * 2) = p1 - centroid;
        PW0.row(i * 2 + 1) = p2 - centroid;
    }

    for (int i = 0; i < n; ++i) {
        PW0.row(m * 2 + i) = points_w[i] - centroid;
    }

    Eigen::Matrix3d PW0tPW0 = PW0.transpose() * PW0;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(PW0tPW0, Eigen::ComputeFullV);

    Eigen::VectorXd singular_values = svd.singularValues();

    Eigen::Matrix3d V = svd.matrixV();

    for (int i = 1; i < 4; ++i) {
        double k =
            sqrt(singular_values(i - 1) / static_cast<double>(n + 2 * m));

        Eigen::Vector3d principal_component = V.col(i - 1);

        Eigen::Vector3d control_point = centroid + k * principal_component;

        control_points[i] = control_point;
    }
}

void UPnPL::computeAlpha(const vector<Eigen::Vector3d> &points_w,
                         const vector<Eigen::VectorXd> &lines_w,
                         const vector<Eigen::Vector3d> &control_points,
                         vector<double> &alpha) {
    int n = points_w.size(), m = lines_w.size();

    Eigen::Matrix3d CC = Eigen::Matrix3d::Zero();

    const Eigen::Vector3d &c0 = control_points[0];
    const Eigen::Vector3d &c1 = control_points[1];
    const Eigen::Vector3d &c2 = control_points[2];
    const Eigen::Vector3d &c3 = control_points[3];

    CC.col(0) = c1 - c0;
    CC.col(1) = c2 - c0;
    CC.col(2) = c3 - c0;

    alpha.resize((n + 2 * m) * 4, 0);

    Eigen::Matrix3d CC_inv;
    double det_CC = CC.determinant();
    if (abs(det_CC) > 1e-6) {
        CC_inv = CC.inverse();

        for (int i = 0; i < m; ++i) {
            Eigen::Vector3d p1 = lines_w[i].head<3>();
            Eigen::Vector3d alpha1 = CC_inv * (p1 - c0);

            alpha[i * 8 + 1] = alpha1(0);
            alpha[i * 8 + 2] = alpha1(1);
            alpha[i * 8 + 3] = alpha1(2);

            alpha[i * 8] = 1 - alpha1(0) - alpha1(1) - alpha1(2);

            Eigen::Vector3d p2 = lines_w[i].tail<3>();
            Eigen::Vector3d alpha2 = CC_inv * (p2 - c0);

            alpha[i * 8 + 5] = alpha2(0);
            alpha[i * 8 + 6] = alpha2(1);
            alpha[i * 8 + 7] = alpha2(2);

            alpha[i * 8 + 4] = 1 - alpha2(0) - alpha2(1) - alpha2(2);
        }

        for (int i = 0; i < n; ++i) {
            Eigen::Vector3d p = points_w[i];
            Eigen::Vector3d alpha_p = CC_inv * (p - c0);

            alpha[m * 8 + i * 4 + 1] = alpha_p(0);
            alpha[m * 8 + i * 4 + 2] = alpha_p(1);
            alpha[m * 8 + i * 4 + 3] = alpha_p(2);

            alpha[m * 8 + i * 4] = 1 - alpha_p(0) - alpha_p(1) - alpha_p(2);
        }
    } else {
        vector<pair<int, int>> index_pair = {{1, 2}, {1, 3}, {2, 3}};

        Eigen::Vector3d u, v;
        int i1 = -1, i2 = -1;

        for (const auto &pair : index_pair) {
            Eigen::Vector3d vi = control_points[pair.first] - c0;
            Eigen::Vector3d vj = control_points[pair.second] - c0;
            if ((vi.cross(vj)).norm() > 1e-6) {
                u = vi;
                v = vj;
                i1 = pair.first;
                i2 = pair.second;
                break;
            }
        }

        if (i1 == -1 || i2 == -1) {
            cerr << "All control points are collinear or coincident." << endl;
            return;
        }

        Eigen::Matrix<double, 3, 2> CC2;
        CC2.col(0) = u;
        CC2.col(1) = v;

        Eigen::Matrix2d CC2_T_CC2 = CC2.transpose() * CC2;
        if (abs(CC2_T_CC2.determinant()) < 1e-6) {
            cerr << "Control points are collinear or coincident." << endl;
            return;
        }

        Eigen::Matrix<double, 2, 3> CC2_pinv =
            CC2_T_CC2.inverse() * CC2.transpose();

        for (int i = 0; i < m; ++i) {
            Eigen::Vector3d p1 = lines_w[i].head<3>();
            Eigen::Vector2d alpha1 = CC2_pinv * (p1 - c0);
            alpha[i * 8 + i1] = alpha1(0);
            alpha[i * 8 + i2] = alpha1(1);
            alpha[i * 8] = 1 - alpha1(0) - alpha1(1);

            Eigen::Vector3d p2 = lines_w[i].tail<3>();
            Eigen::Vector2d alpha2 = CC2_pinv * (p2 - c0);
            alpha[i * 8 + 4 + i1] = alpha2(0);
            alpha[i * 8 + 4 + i2] = alpha2(1);
            alpha[i * 8 + 4] = 1 - alpha2(0) - alpha2(1);
        }

        for (int i = 0; i < n; ++i) {
            Eigen::Vector3d p = points_w[i];
            Eigen::Vector2d alpha_p = CC2_pinv * (p - c0);
            alpha[m * 8 + i * 4 + i1] = alpha_p(0);
            alpha[m * 8 + i * 4 + i2] = alpha_p(1);
            alpha[m * 8 + i * 4] = 1 - alpha_p(0) - alpha_p(1);
        }
    }
}

void UPnPL::normalization(const vector<Eigen::Vector3d> &points_w,
                          const vector<Eigen::VectorXd> &lines_w,
                          vector<Eigen::Vector3d> &points_w_n,
                          vector<Eigen::VectorXd> &lines_w_n, double &scale,
                          Eigen::Vector3d &center, bool normalize) {
    int n = points_w.size(), m = lines_w.size();

    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();

    for (const auto &p : points_w) {
        A += Eigen::Matrix3d::Identity();
        b += p;
    }

    for (const auto &l : lines_w) {
        Eigen::Vector3d p1 = l.head<3>();
        Eigen::Vector3d p2 = l.tail<3>();
        Eigen::Vector3d d = p2 - p1;
        d.normalize();

        Eigen::Matrix3d Pi = Eigen::Matrix3d::Identity() - d * d.transpose();
        A += Pi;
        b += Pi * p1;
    }

    center = A.ldlt().solve(b);

    if (!normalize) {
        center.setZero();
    }

    double dist = 0;
    for (const auto &p : points_w) {
        dist += (p - center).norm();
    }

    vector<double> line_dist;
    line_dist.reserve(m);

    for (const auto &l : lines_w) {
        Eigen::Vector3d p1 = l.head<3>();
        Eigen::Vector3d p2 = l.tail<3>();

        line_dist.emplace_back(pointLineDistance(center, p1, p2));
        dist += 2 * sqrt(2) * line_dist.back();
    }

    scale = sqrt(3) / (dist / (n + 2 * m));

    if (!normalize) {
        scale = 1;
    }

    points_w_n.resize(n);
    for (int i = 0; i < n; ++i) {
        points_w_n[i] = scale * (points_w[i] - center);
    }

    lines_w_n.resize(m);
    for (int i = 0; i < m; ++i) {
        Eigen::Vector3d p1 = lines_w[i].head<3>();
        Eigen::Vector3d p2 = lines_w[i].tail<3>();

        Eigen::Vector3d p1_n = scale * (p1 - center);
        Eigen::Vector3d p2_n = scale * (p2 - center);

        Eigen::Vector3d d = p2_n - p1_n;
        d.normalize();

        Eigen::Vector3d p_proj = p1_n + d * (-p1_n).dot(d);
        double dist_l = line_dist[i] * scale;
        dist_l = 0.5;

        lines_w_n[i].resize(6);
        if (normalize) {
            lines_w_n[i].head<3>() = p_proj + d * dist_l;
            lines_w_n[i].tail<3>() = p_proj - d * dist_l;
        } else {
            lines_w_n[i].head<3>() = p1_n;
            lines_w_n[i].tail<3>() = p2_n;
        }
    }
}

Eigen::Vector2d UPnPL::computeCorrectness(const Eigen::Vector2d &p,
                                          const Eigen::Vector2d &pd,
                                          const Eigen::Vector2d &v,
                                          const Eigen::Vector3d &line_param,
                                          double d) {
    double a = line_param(0), b = line_param(1), c = line_param(2);
    double px = p(0), py = p(1);
    double vx = v(0), vy = v(1);

    double lambda1 =
        (d * sqrt(a * a + b * b) - a * px - b * py - c) / (a * vx + b * vy);
    double lambda2 =
        (-d * sqrt(a * a + b * b) - a * px - b * py - c) / (a * vx + b * vy);

    Eigen::Vector2d p1 = p + lambda1 * v;
    double d1 = (p1 - pd).norm();
    Eigen::Vector2d p2 = p + lambda2 * v;
    double d2 = (p2 - pd).norm();

    if (d1 < d2)
        return p1;
    return p2;
}

Eigen::Vector3d UPnPL::backProjPointToLine(const Eigen::Vector2d &x,
                                           const Eigen::Vector3d &X,
                                           const Eigen::Vector3d &v) {
    Eigen::Vector3d r0, r1;
    r0 << X(0), v(0), -x(0);
    r1 << X(1), v(1), -x(1);
    Eigen::Vector3d r3 = r0.cross(r1);
    return X + r3(1) / r3(0) * v;
}

double UPnPL::pointLineDistance(const Eigen::Vector3d &x,
                                const Eigen::Vector3d &a,
                                const Eigen::Vector3d &b) {
    Eigen::Vector3d ab = b - a;
    Eigen::Vector3d ax = x - a;
    return (ab.cross(ax)).norm() / ab.norm();
}

Eigen::Matrix<double, 3, 9> UPnPL::phi(const Eigen::Vector3d &p) {
    Eigen::Matrix<double, 3, 9> J;
    J.setZero();
    double x = p(0), y = p(1), z = p(2);
    J << 0, 2 * z, -2 * y, x, -x, -x, 2 * y, 2 * z, 0, -2 * z, 0, 2 * x, -y, y,
        -y, 2 * x, 0, 2 * z, 2 * y, -2 * x, 0, -z, -z, z, 0, 2 * x, 2 * y;
    return J;
}

void UPnPL::CGR2Rotation(const Eigen::Vector3d &s, Eigen::Matrix3d &R) {
    double norm_s = s.norm();
    if (norm_s < 1e-6) {
        R.setIdentity();
        return;
    }

    R = (1 - s.transpose() * s) * Eigen::Matrix3d::Identity() +
        2 * crossProductMatrix(s) + 2 * s * s.transpose();

    R /= 1 + s.transpose() * s;
}

} // namespace UPnPL
