"""Wrenfold helpers for Twb: right SO(3) increments, additive world translation."""
from wrenfold import sym
from wrenfold.type_annotations import Matrix3, Vector3


def perturbed_pose(Rwb: Matrix3, twb: Vector3):
    delta = sym.make_symbols([f"pose_delta_{i}" for i in range(6)])
    x, y, z = delta[:3]
    skew = Matrix3([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    # The first-order retraction gives the exact derivative at delta=0.
    # Runtime updates use ExpSO3, not this first-order approximation.
    return Rwb * (sym.eye(3) + skew), twb + Vector3(delta[3:]), delta


def at_zero(expression, delta):
    return expression.subs([(x, 0) for x in delta])


def pose_jacobian(residual, delta):
    return at_zero(sym.jacobian(residual, delta), delta)
