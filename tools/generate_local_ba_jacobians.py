#!/usr/bin/env python3
import argparse
from pathlib import Path

from pose_codegen import at_zero, perturbed_pose, pose_jacobian

from wrenfold import code_generation, geometry, sym
from wrenfold.type_annotations import (
    FloatScalar,
    Matrix2,
    Matrix3,
    Vector2,
    Vector3,
    Vector4,
    Vector6,
)


EPSILON = 1.0e-8


def _rotation_from_vector(rvec: Vector3):
    return geometry.Quaternion.from_rotation_vector(
        rvec[0], rvec[1], rvec[2], EPSILON).to_rotation_matrix()


def _cross(a: Vector3, b: Vector3):
    return Vector3([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])


def _project_point(point_c: Vector3, fx: FloatScalar, fy: FloatScalar,
                   cx: FloatScalar, cy: FloatScalar):
    return Vector2([
        fx * point_c[0] / point_c[2] + cx,
        fy * point_c[1] / point_c[2] + cy,
    ])


def _project_line(normal_c: Vector3, fx: FloatScalar, fy: FloatScalar,
                  cx: FloatScalar, cy: FloatScalar):
    return Vector3([
        fy * normal_c[0],
        fx * normal_c[1],
        -fy * cx * normal_c[0] - fx * cy * normal_c[1] +
        fx * fy * normal_c[2],
    ])


def _line_error(line_2d: Vector3, point_2d: Vector2):
    a = line_2d[0]
    b = line_2d[1]
    c = line_2d[2]
    x = point_2d[0]
    y = point_2d[1]
    scale = sym.sqrt(a * a + b * b)
    return (a * x + b * y + c) / scale


def _rotation_so2(theta: FloatScalar):
    return Matrix2([
        [sym.cos(theta), -sym.sin(theta)],
        [sym.sin(theta), sym.cos(theta)],
    ])


def _camera_transform(Rwb: Matrix3, twb: Vector3,
                      rcb: Vector3, tcb: Vector3):
    Rcw = _rotation_from_vector(rcb) * Rwb.transpose()
    return Rcw, tcb - Rcw * twb


def local_ba_point_residual(
        Rwb: Matrix3, twb: Vector3, point_w: Vector3, rcb: Vector3, tcb: Vector3,
        fx: FloatScalar, fy: FloatScalar, cx: FloatScalar, cy: FloatScalar,
        measurement: Vector2):
    Rcw, tcw = _camera_transform(Rwb, twb, rcb, tcb)
    point_c = Rcw * point_w + tcw
    residual = measurement - _project_point(point_c, fx, fy, cx, cy)
    return code_generation.OutputArg(residual, "residual")


def local_ba_point_jacobian_pose(
        Rwb: Matrix3, twb: Vector3, point_w: Vector3, rcb: Vector3, tcb: Vector3,
        fx: FloatScalar, fy: FloatScalar, cx: FloatScalar, cy: FloatScalar,
        measurement: Vector2):
    Rwb, twb, delta = perturbed_pose(Rwb, twb)
    Rcw, tcw = _camera_transform(Rwb, twb, rcb, tcb)
    point_c = Rcw * point_w + tcw
    residual = measurement - _project_point(point_c, fx, fy, cx, cy)
    J_pose = pose_jacobian(residual, delta)
    return code_generation.OutputArg(J_pose, "J_pose")


def local_ba_point_jacobian_point(
        Rwb: Matrix3, twb: Vector3, point_w: Vector3, rcb: Vector3, tcb: Vector3,
        fx: FloatScalar, fy: FloatScalar, cx: FloatScalar, cy: FloatScalar,
        measurement: Vector2):
    Rcw, tcw = _camera_transform(Rwb, twb, rcb, tcb)
    point_c = Rcw * point_w + tcw
    residual = measurement - _project_point(point_c, fx, fy, cx, cy)
    J_point = sym.jacobian(residual, [point_w[i] for i in range(3)])
    return code_generation.OutputArg(J_point, "J_point")


def _line_residual(Rwb: Matrix3, twb: Vector3, line_base_U: Matrix3,
                   line_base_W: Matrix2, line_delta: Vector4,
                   rcb: Vector3, tcb: Vector3, fx: FloatScalar,
                   fy: FloatScalar, cx: FloatScalar, cy: FloatScalar,
                   measurement: Vector4):
    line_rotation = _rotation_from_vector(
        Vector3([line_delta[0], line_delta[1], line_delta[2]]))
    line_U = line_base_U * line_rotation
    line_W = line_base_W * _rotation_so2(line_delta[3])
    moment_w = Vector3([
        line_U[0, 0] * line_W[0, 0],
        line_U[1, 0] * line_W[0, 0],
        line_U[2, 0] * line_W[0, 0],
    ])
    direction_w = Vector3([
        line_U[0, 1] * line_W[1, 0],
        line_U[1, 1] * line_W[1, 0],
        line_U[2, 1] * line_W[1, 0],
    ])
    Rcw, tcw = _camera_transform(Rwb, twb, rcb, tcb)
    direction_c = Rcw * direction_w
    normal_c = Rcw * moment_w + _cross(tcw, direction_c)
    line_2d = _project_line(normal_c, fx, fy, cx, cy)
    start = Vector2([measurement[0], measurement[1]])
    end = Vector2([measurement[2], measurement[3]])
    residual = Vector2([
        _line_error(line_2d, start),
        _line_error(line_2d, end),
    ])
    return residual, line_2d


def local_ba_line_residual(
        Rwb: Matrix3, twb: Vector3, line_base_U: Matrix3, line_base_W: Matrix2,
        line_delta: Vector4, rcb: Vector3, tcb: Vector3,
        fx: FloatScalar, fy: FloatScalar, cx: FloatScalar, cy: FloatScalar,
        measurement: Vector4):
    residual, line_2d = _line_residual(
        Rwb, twb, line_base_U, line_base_W, line_delta, rcb, tcb, fx, fy, cx, cy,
        measurement)
    return (code_generation.OutputArg(residual, "residual"),
            code_generation.OutputArg(line_2d, "line_2d"))


def local_ba_line_jacobian_pose(
        Rwb: Matrix3, twb: Vector3, line_base_U: Matrix3, line_base_W: Matrix2,
        line_delta: Vector4, rcb: Vector3, tcb: Vector3,
        fx: FloatScalar, fy: FloatScalar, cx: FloatScalar, cy: FloatScalar,
        measurement: Vector4):
    Rwb, twb, delta = perturbed_pose(Rwb, twb)
    residual, line_2d = _line_residual(
        Rwb, twb, line_base_U, line_base_W, line_delta, rcb, tcb, fx, fy, cx, cy,
        measurement)
    J_pose = pose_jacobian(residual, delta)
    return (code_generation.OutputArg(J_pose, "J_pose"),
            code_generation.OutputArg(at_zero(line_2d, delta), "line_2d"))


def local_ba_line_jacobian_line(
        Rwb: Matrix3, twb: Vector3, line_base_U: Matrix3, line_base_W: Matrix2,
        line_delta: Vector4, rcb: Vector3, tcb: Vector3,
        fx: FloatScalar, fy: FloatScalar, cx: FloatScalar, cy: FloatScalar,
        measurement: Vector4):
    residual, line_2d = _line_residual(
        Rwb, twb, line_base_U, line_base_W, line_delta, rcb, tcb, fx, fy, cx, cy,
        measurement)
    J_line = sym.jacobian(residual, [line_delta[i] for i in range(4)])
    return (code_generation.OutputArg(J_line, "J_line"),
            code_generation.OutputArg(line_2d, "line_2d"))


def pose_only_line_jacobian_pose(
        Rwb: Matrix3, twb: Vector3, line_w: Vector6,
        Rcb: Matrix3, tcb: Vector3, to_pl: Matrix3,
        measurement: Vector4):
    Rwb, twb, delta = perturbed_pose(Rwb, twb)
    Rcw = Rcb * Rwb.transpose()
    tcw = tcb - Rcw * twb
    moment = Vector3([line_w[i] for i in range(3)])
    direction = Vector3([line_w[i + 3] for i in range(3)])
    normal_c = Rcw * moment + _cross(tcw, Rcw * direction)
    line_2d = to_pl * normal_c
    residual = Vector2([
        _line_error(line_2d, Vector2([measurement[0], measurement[1]])),
        _line_error(line_2d, Vector2([measurement[2], measurement[3]])),
    ])
    return (code_generation.OutputArg(pose_jacobian(residual, delta), "J_pose"),
            code_generation.OutputArg(at_zero(line_2d, delta), "line_2d"))


def pose_only_point_jacobian_pose(
        Rwb: Matrix3, twb: Vector3, point_w: Vector3,
        Rcb: Matrix3, tcb: Vector3, fx: FloatScalar, fy: FloatScalar):
    Rwb, twb, delta = perturbed_pose(Rwb, twb)
    point_c = Rcb * Rwb.transpose() * (point_w - twb) + tcb
    residual = -_project_point(point_c, fx, fy, 0, 0)
    return code_generation.OutputArg(pose_jacobian(residual, delta), "J_pose")


def _generate(func):
    generator = code_generation.CppGenerator(
        code_generation.CppMatrixTypeBehavior.Eigen)
    code = code_generation.generate_function(func, generator)
    code = code.replace("void local_ba_",
                        "XPLSLAM_GRAPHITE_HD void local_ba_")
    code = code.replace("void pose_only_",
                        "XPLSLAM_GRAPHITE_HD void pose_only_")
    return "\n".join(line.rstrip() for line in code.splitlines()) + "\n"


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=repo_root / "include" / "graphite_local_ba_jacobians.cuh")
    output = parser.parse_args().output
    body = "\n\n".join([
        _generate(pose_only_point_jacobian_pose),
        _generate(pose_only_line_jacobian_pose),
        _generate(local_ba_point_residual),
        _generate(local_ba_point_jacobian_point),
        _generate(local_ba_point_jacobian_pose),
        _generate(local_ba_line_residual),
        _generate(local_ba_line_jacobian_line),
        _generate(local_ba_line_jacobian_pose),
    ])
    output.write_text(
        "// Generated by tools/generate_local_ba_jacobians.py; do not edit.\n"
        "// Twb: Rwb *= Exp(dtheta), twb += dt_world.\n"
        "#pragma once\n\n"
        "#include <Eigen/Core>\n"
        "#include <cmath>\n\n"
        "#if defined(__CUDACC__)\n"
        "#define XPLSLAM_GRAPHITE_HD __host__ __device__\n"
        "#else\n"
        "#define XPLSLAM_GRAPHITE_HD\n"
        "#endif\n\n"
        "namespace XPLSLAM::graphite_backend::generated {\n\n"
        f"{body}\n\n"
        "} // namespace XPLSLAM::graphite_backend::generated\n\n"
        "#undef XPLSLAM_GRAPHITE_HD\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
