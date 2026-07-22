"""Impulse-based collision response with linear + angular terms.

Face rest vs edge/vertex tipping
--------------------------------
* **Face support** (body face nearly parallel to the contact plane): normal
  force through the COM, strong settle, may sleep.  Prevents the
  "stop then spin up on the floor" loop caused by single-corner contacts.
* **Edge/vertex support** (no face well aligned with the plane): geometric
  lever arm so gravity tips the body onto a face.  Never sleep.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from engine.types import Vector3

try:
    import os as _os
    if _os.environ.get("PYENGINE_PURE_PYTHON", "0").lower() in ("1", "true", "yes"):
        raise ImportError("Cython disabled via PYENGINE_PURE_PYTHON=1")
    from engine.cython.cy_response_3d import (
        resolve_contact_3d_fast as _cy_resolve_contact,
        face_align_from_matrix_fast as _cy_face_align,
        obb_support_feature_centroid_fast as _cy_obb_support,
    )
    _USE_CYTHON = True
except Exception:
    _USE_CYTHON = False
    _cy_resolve_contact = None
    _cy_face_align = None
    _cy_obb_support = None

RESTITUTION_THRESHOLD = 1.0
IMPACT_BLEND_START = 1.2
IMPACT_BLEND_END = 7.0
# |cos| between body face normal and contact normal → face support.
FACE_ALIGN_THRESHOLD = 0.82
# Must be this aligned with the floor before we freeze as "face rest".
# 0.92 ≈ 23° residual tilt (looked stuck at an angle); 0.985 ≈ 10°.
FACE_REST_ALIGN = 0.985
# In-plane COM offset (m) that marks edge/vertex / off-center support.
UNSTABLE_SUPPORT_OFFSET = 0.06
MAX_NORMAL_TANGENT_ARM = 0.35
RESTING_TANGENTIAL_SPEED = 0.08
MAX_ANGULAR_SPEED = 20.0
GRAVITY = 9.81


def _as_np3(v) -> np.ndarray:
    if isinstance(v, np.ndarray):
        return np.asarray(v, dtype=np.float64).reshape(3)
    if hasattr(v, "to_numpy"):
        return np.asarray(v.to_numpy(), dtype=np.float64).reshape(3)
    if hasattr(v, "x"):
        return np.array(
            [float(v.x), float(v.y), float(getattr(v, "z", 0.0))],
            dtype=np.float64,
        )
    return np.asarray(v, dtype=np.float64).reshape(3)


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ],
        dtype=np.float64,
    )


def _clamp_omega(omega: np.ndarray, limit: float = MAX_ANGULAR_SPEED) -> np.ndarray:
    mag = float(np.linalg.norm(omega))
    if mag > limit and mag > 1e-12:
        return omega * (limit / mag)
    return omega


def _clamp_vec(v: np.ndarray, max_len: float) -> np.ndarray:
    mag = float(np.linalg.norm(v))
    if mag > max_len and mag > 1e-12:
        return v * (max_len / mag)
    return v


def _impact_weight(closing: float) -> float:
    if closing <= IMPACT_BLEND_START:
        return 0.0
    if closing >= IMPACT_BLEND_END:
        return 1.0
    t = (closing - IMPACT_BLEND_START) / (IMPACT_BLEND_END - IMPACT_BLEND_START)
    return t * t * (3.0 - 2.0 * t)


def _support_offset(ra_full: np.ndarray, n: np.ndarray) -> float:
    ra_t = ra_full - n * float(np.dot(ra_full, n))
    return float(np.linalg.norm(ra_t))


def _face_align_from_rotation(game_object, n: np.ndarray) -> float:
    """How well a body face aligns with the contact normal (1 = face support)."""
    if game_object is None:
        return 0.0
    try:
        R = game_object.transform.rotation_matrix
        R = np.ascontiguousarray(R, dtype=np.float64).reshape(3, 3)
        n64 = np.ascontiguousarray(n, dtype=np.float64).reshape(3)
        if _USE_CYTHON and _cy_face_align is not None:
            return float(_cy_face_align(R, n64))
        best = 0.0
        for i in range(3):
            best = max(best, abs(float(np.dot(R[:, i], n64))))
            best = max(best, abs(float(np.dot(R[i, :], n64))))
        return best
    except Exception:
        return 0.0


def _effective_mass(
    direction: np.ndarray,
    inv_mass_a: float,
    inv_mass_b: float,
    ra: np.ndarray,
    rb: np.ndarray,
    i_inv_a: Optional[np.ndarray],
    i_inv_b: Optional[np.ndarray],
) -> float:
    k = inv_mass_a + inv_mass_b
    if i_inv_a is not None and inv_mass_a > 0.0:
        rna = _cross(ra, direction)
        k += float(np.dot(rna, i_inv_a @ rna))
    if i_inv_b is not None and inv_mass_b > 0.0:
        rnb = _cross(rb, direction)
        k += float(np.dot(rnb, i_inv_b @ rnb))
    return k


def _apply_impulse(
    va, oa, vb, ob,
    inv_mass_a, inv_mass_b,
    i_inv_a, i_inv_b,
    ra, rb,
    j_vec,
):
    va = va + j_vec * inv_mass_a
    vb = vb - j_vec * inv_mass_b
    if i_inv_a is not None and inv_mass_a > 0.0:
        oa = oa + i_inv_a @ _cross(ra, j_vec)
    if i_inv_b is not None and inv_mass_b > 0.0:
        ob = ob - i_inv_b @ _cross(rb, j_vec)
    return va, oa, vb, ob


def _normal_lever_arms(ra_full, rb_full, n, weight: float):
    ra_com = n * float(np.dot(ra_full, n))
    rb_com = n * float(np.dot(rb_full, n))
    ra_t = _clamp_vec(ra_full - ra_com, MAX_NORMAL_TANGENT_ARM)
    rb_t = _clamp_vec(rb_full - rb_com, MAX_NORMAL_TANGENT_ARM)
    return ra_com + weight * ra_t, rb_com + weight * rb_t


def resolve_contact_3d(
    *,
    pos_a,
    vel_a,
    omega_a,
    inv_mass_a: float,
    i_inv_a: Optional[np.ndarray],
    pos_b,
    vel_b,
    omega_b,
    inv_mass_b: float,
    i_inv_b: Optional[np.ndarray],
    contact_point,
    normal,
    restitution: float,
    static_friction: float,
    dynamic_friction: float,
    face_align_a: float = 0.0,
    face_align_b: float = 0.0,
    dt: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Resolve one 3D contact.

    Returns
    -------
    (vel_a, omega_a, vel_b, omega_b, unstable_support)
    """
    if _USE_CYTHON and _cy_resolve_contact is not None:
        from engine.component import Time
        if dt is None:
            dt = float(getattr(Time, "delta_time", 0.0) or (1.0 / 60.0))
        pa = np.ascontiguousarray(_as_np3(pos_a), dtype=np.float64)
        pb = np.ascontiguousarray(_as_np3(pos_b), dtype=np.float64)
        va = np.ascontiguousarray(_as_np3(vel_a), dtype=np.float64)
        vb = np.ascontiguousarray(_as_np3(vel_b), dtype=np.float64)
        oa = np.ascontiguousarray(_as_np3(omega_a), dtype=np.float64)
        ob = np.ascontiguousarray(_as_np3(omega_b), dtype=np.float64)
        cp = np.ascontiguousarray(_as_np3(contact_point), dtype=np.float64)
        n = np.ascontiguousarray(_as_np3(normal), dtype=np.float64)
        ia = None if i_inv_a is None else np.ascontiguousarray(i_inv_a, dtype=np.float64)
        ib = None if i_inv_b is None else np.ascontiguousarray(i_inv_b, dtype=np.float64)
        return _cy_resolve_contact(
            pa, va, oa, float(inv_mass_a), ia,
            pb, vb, ob, float(inv_mass_b), ib,
            cp, n,
            float(restitution), float(static_friction), float(dynamic_friction),
            float(face_align_a), float(face_align_b), float(dt),
        )

    n = _as_np3(normal)
    n_len = float(np.linalg.norm(n))
    if n_len < 1e-12:
        return (
            _as_np3(vel_a), _as_np3(omega_a),
            _as_np3(vel_b), _as_np3(omega_b),
            False,
        )
    n = n / n_len

    pa = _as_np3(pos_a)
    pb = _as_np3(pos_b)
    va = _as_np3(vel_a)
    vb = _as_np3(vel_b)
    oa = _as_np3(omega_a)
    ob = _as_np3(omega_b)
    cp = _as_np3(contact_point)

    ra_full = cp - pa
    rb_full = cp - pb

    v_rel0 = (va + _cross(oa, ra_full)) - (vb + _cross(ob, rb_full))
    v_n0 = float(np.dot(v_rel0, n))
    if v_n0 > 0.0:
        return va, oa, vb, ob, False

    closing = -v_n0
    w_impact = _impact_weight(closing)

    off_a = _support_offset(ra_full, n) if inv_mass_a > 0.0 else 0.0
    off_b = _support_offset(rb_full, n) if inv_mass_b > 0.0 else 0.0
    support_off = max(off_a, off_b)

    # Face parallel to contact plane (candidate for stable rest).
    aligned_a = inv_mass_a > 0.0 and face_align_a >= FACE_ALIGN_THRESHOLD
    aligned_b = inv_mass_b > 0.0 and face_align_b >= FACE_ALIGN_THRESHOLD
    face_aligned = aligned_a or aligned_b
    best_align = max(face_align_a if inv_mass_a > 0.0 else 0.0,
                     face_align_b if inv_mass_b > 0.0 else 0.0)

    # Stable face rest: face well aligned AND contact near COM projection.
    face_support = face_aligned and support_off < UNSTABLE_SUPPORT_OFFSET
    unstable = support_off >= UNSTABLE_SUPPORT_OFFSET

    # Floor contacts: settle only when *nearly flat* on a face.
    # Too-low align threshold left cubes frozen at ~15–25° with a corner sunk in.
    ground_like = abs(float(n[1])) > 0.88
    if ground_like and closing < 2.5:
        if best_align >= FACE_REST_ALIGN and support_off < 0.08:
            # Truly face-down on floor → stable rest
            face_support = True
            unstable = False
        elif (
            support_off >= UNSTABLE_SUPPORT_OFFSET
            or best_align < FACE_REST_ALIGN
        ):
            # Edge / vertex / residual tilt → keep tipping under gravity
            face_support = False
            unstable = True
            # Exact 45° knife-edge can put COM above the edge (off≈0).
            # Nudge a virtual offset *perpendicular to the edge* so gravity tips.
            if support_off < 1e-3 and inv_mass_a > 0.0:
                edge = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                edge = edge - n * float(np.dot(edge, n))
                if float(np.linalg.norm(edge)) < 1e-6:
                    edge = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                    edge = edge - n * float(np.dot(edge, n))
                el = float(np.linalg.norm(edge))
                if el > 1e-6:
                    edge /= el
                    tip_dir = _cross(n, edge)
                    tl = float(np.linalg.norm(tip_dir))
                    if tl > 1e-6:
                        tip_dir /= tl
                        ra_full = ra_full + tip_dir * 0.03
                        support_off = 0.03

    if face_support:
        # Resting face: pure COM normal (no phantom floor spin)
        w_n = 0.0 if closing < 1.5 else w_impact * 0.1
    elif unstable:
        tip = min(1.0, max(support_off, 0.02) / 0.25)
        w_n = max(w_impact, 0.75 + 0.25 * tip)
    else:
        w_n = w_impact

    ra_n, rb_n = _normal_lever_arms(ra_full, rb_full, n, w_n)

    v_rel_n = (va + _cross(oa, ra_n)) - (vb + _cross(ob, rb_n))
    v_n = float(np.dot(v_rel_n, n))
    if v_n > 0.0:
        return va, oa, vb, ob, unstable

    kn = _effective_mass(n, inv_mass_a, inv_mass_b, ra_n, rb_n, i_inv_a, i_inv_b)
    if kn < 1e-12:
        return va, oa, vb, ob, unstable

    e = max(0.0, min(1.0, float(restitution)))
    if closing < RESTITUTION_THRESHOLD or face_support and closing < 2.5:
        # Kill micro-bounce on faces so they don't hop and re-hit a corner
        if face_support and closing < 2.5:
            e = 0.0
        elif closing < RESTITUTION_THRESHOLD:
            e = 0.0

    jn = -(1.0 + e) * v_n / kn
    if jn < 0.0:
        jn = 0.0

    jn_vec = n * jn
    va, oa, vb, ob = _apply_impulse(
        va, oa, vb, ob,
        inv_mass_a, inv_mass_b, i_inv_a, i_inv_b,
        ra_n, rb_n, jn_vec,
    )

    # Gravity tipping on edges/vertices: continuous support torque about COM.
    # Discrete jn only cancels g*dt vertically; without this, balanced edges sleep.
    if unstable and ground_like and closing < 3.0 and inv_mass_a > 0.0 and i_inv_a is not None:
        from engine.component import Time
        dt = float(getattr(Time, "delta_time", 0.0) or (1.0 / 60.0))
        dt = max(1e-5, min(dt, 0.05))
        mass_a = 1.0 / inv_mass_a
        # Support ≈ weight along +n (floor pushes up)
        support = n * (mass_a * GRAVITY * max(0.0, float(n[1])))
        if float(np.linalg.norm(support)) < 1e-8:
            support = n * (mass_a * GRAVITY)
        # Use full geometric lever (ra_full), not COM-aligned
        tau_imp = _cross(ra_full, support) * dt
        oa = oa + i_inv_a @ tau_imp
        # Keep a minimum tip rate so residuals aren't stuck below noise floors
        if float(np.linalg.norm(oa)) < 0.15:
            oa = oa + i_inv_a @ tau_imp  # double once if still tiny
    if unstable and ground_like and closing < 3.0 and inv_mass_b > 0.0 and i_inv_b is not None:
        from engine.component import Time
        dt = float(getattr(Time, "delta_time", 0.0) or (1.0 / 60.0))
        dt = max(1e-5, min(dt, 0.05))
        mass_b = 1.0 / inv_mass_b
        support = n * (mass_b * GRAVITY * max(0.0, float(n[1])))
        if float(np.linalg.norm(support)) < 1e-8:
            support = n * (mass_b * GRAVITY)
        tau_imp = _cross(rb_full, support) * dt
        ob = ob + i_inv_b @ tau_imp

    # --- Friction ---
    if face_support:
        # Linear friction only while settling on a face (no spin-up)
        ra_f = ra_n
        rb_f = rb_n
        friction_angular = closing > 2.0  # only on hard scrapes
    elif unstable:
        ra_f, rb_f = ra_full, rb_full
        friction_angular = True
    else:
        ra_f = ra_full * w_n + ra_n * (1.0 - w_n)
        rb_f = rb_full * w_n + rb_n * (1.0 - w_n)
        friction_angular = w_n > 0.2

    v_rel = (va + _cross(oa, ra_f)) - (vb + _cross(ob, rb_f))
    v_t = v_rel - n * float(np.dot(v_rel, n))
    t_mag = float(np.linalg.norm(v_t))
    mu_s = max(0.0, float(static_friction))
    mu_d = max(0.0, float(dynamic_friction))

    if t_mag >= 1e-10:
        t = v_t / t_mag
        if not friction_angular or t_mag < RESTING_TANGENTIAL_SPEED:
            kt_lin = inv_mass_a + inv_mass_b
            if kt_lin > 1e-12:
                # jt < 0 opposes slide when impulse is jt * t
                jt = -t_mag / kt_lin
                max_f = abs(jn) * max(mu_s, mu_d)
                # Clamp magnitude only — keep sign (do NOT double-negate)
                if abs(jt) > max_f:
                    jt = math.copysign(max_f, jt) if max_f > 0.0 else 0.0
                jt_vec = t * jt
                va = va + jt_vec * inv_mass_a
                vb = vb - jt_vec * inv_mass_b
                if face_support:
                    oa = oa * 0.5
                    ob = ob * 0.5
        else:
            kt = _effective_mass(
                t, inv_mass_a, inv_mass_b, ra_f, rb_f, i_inv_a, i_inv_b
            )
            if kt > 1e-12:
                vt_along = float(np.dot(v_rel, t))
                jt = -vt_along / kt
                max_static = abs(jn) * mu_s
                if abs(jt) > max_static:
                    # Keep sign of jt; dynamic friction cone
                    jt = math.copysign(abs(jn) * mu_d, jt)
                if unstable:
                    jt *= 0.45
                # Never reverse past zero (no friction energy injection)
                if vt_along * (vt_along + jt * kt) < 0.0:
                    jt = -vt_along / kt
                jt_vec = t * jt
                va, oa, vb, ob = _apply_impulse(
                    va, oa, vb, ob,
                    inv_mass_a, inv_mass_b, i_inv_a, i_inv_b,
                    ra_f, rb_f, jt_vec,
                )

    # Face / floor settling: kill residual spin. With friction > 0, also kill
    # micro horizontal creep so bodies don't "ice skate" after landing.
    if face_support and closing < 2.0:
        if inv_mass_a > 0.0:
            oa = oa * 0.25
            if float(np.linalg.norm(oa)) < 0.5:
                oa = np.zeros(3)
            speed = float(np.linalg.norm(va))
            if speed < 0.08:
                va = np.zeros(3)
            elif mu_s > 1e-6 and speed < 0.35:
                # Nearly stopped on a rough floor → stick
                va = np.zeros(3)
        if inv_mass_b > 0.0:
            ob = ob * 0.25
            if float(np.linalg.norm(ob)) < 0.5:
                ob = np.zeros(3)

    elif not unstable and w_n < 0.5:
        damp = 0.72 + 0.2 * w_n
        if inv_mass_b < 1e-12 and inv_mass_a > 0.0:
            oa = oa * damp
        elif inv_mass_a < 1e-12 and inv_mass_b > 0.0:
            ob = ob * damp

    return va, _clamp_omega(oa), vb, _clamp_omega(ob), unstable


def estimate_contact_point(pos_a, pos_b, normal, depth: float = 0.0) -> np.ndarray:
    pa = _as_np3(pos_a)
    pb = _as_np3(pos_b)
    n = _as_np3(normal)
    n_len = float(np.linalg.norm(n))
    if n_len > 1e-12:
        n = n / n_len
        mid = 0.5 * (pa + pb)
        return (mid - n * (0.5 * float(depth))).astype(np.float64)
    return (0.5 * (pa + pb)).astype(np.float64)


def stabilize_contact_point(
    pos_a, pos_b, contact_point, normal, depth: float,
    face_align_a: float = 0.0, face_align_b: float = 0.0,
):
    """For face support, snap contact under COM; keep edge features otherwise."""
    pa = _as_np3(pos_a)
    pb = _as_np3(pos_b)
    cp = _as_np3(contact_point)
    n = _as_np3(normal)
    n_len = float(np.linalg.norm(n))
    if n_len < 1e-12:
        return cp
    n = n / n_len

    ca = pa - n * float(np.dot(pa - cp, n))
    cb = pb - n * float(np.dot(pb - cp, n))
    com_ref = ca if face_align_a >= face_align_b else cb

    offset_vec = cp - com_ref
    offset_vec = offset_vec - n * float(np.dot(offset_vec, n))
    offset = float(np.linalg.norm(offset_vec))
    best_align = max(face_align_a, face_align_b)

    # Snap under COM only when nearly flat on a face (not residual tilt).
    if best_align >= FACE_REST_ALIGN and offset < 0.08:
        return com_ref

    # Keep geometric edge/vertex / tilted contacts so gravity can tip
    return cp


def body_state_from_rigidbody(rb, game_object, is_immovable: bool):
    if game_object is None:
        return (
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            0.0,
            None,
        )

    pos = _as_np3(game_object.transform.position)

    if rb is None or is_immovable:
        return (
            pos,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            0.0,
            None,
        )

    vel = _as_np3(rb.velocity)
    omega = _as_np3(rb.angular_velocity)
    mass = float(rb.mass) if rb.mass > 1e-10 else 1e-10
    inv_mass = 1.0 / mass
    try:
        i_inv = np.asarray(rb.get_world_inertia_inv_matrix(), dtype=np.float64)
    except Exception:
        i_inv = np.eye(3, dtype=np.float64) * (inv_mass * 6.0)
    return pos, vel, omega, inv_mass, i_inv


def apply_body_state(
    rb,
    vel: np.ndarray,
    omega: np.ndarray,
    *,
    allow_sleep: bool = True,
) -> None:
    if rb is None:
        return
    from engine.component import Time

    lin = float(np.linalg.norm(vel))
    ang = float(np.linalg.norm(omega))

    # Snap micro residuals only on stable face rest. On edges/vertices
    # (allow_sleep=False) small tip rates must accumulate under gravity.
    if allow_sleep:
        if lin < 0.06:
            vel = np.zeros(3, dtype=np.float64)
            lin = 0.0
        if ang < 0.12:
            omega = np.zeros(3, dtype=np.float64)
            ang = 0.0

    max_w = float(getattr(rb, "max_angular_velocity", MAX_ANGULAR_SPEED))
    omega = _clamp_omega(omega, max_w)

    rb._velocity = Vector3(float(vel[0]), float(vel[1]), float(vel[2]))
    rb._angular_velocity = Vector3(float(omega[0]), float(omega[1]), float(omega[2]))

    if not allow_sleep:
        rb._sleep_timer = 0.0
        rb._is_sleeping = False
        return

    thr = float(getattr(rb, "sleep_threshold", 0.08))
    if lin < thr and ang < thr * 2.0:
        dt = float(getattr(Time, "delta_time", 0.0) or (1.0 / 60.0))
        rb._sleep_timer = float(getattr(rb, "_sleep_timer", 0.0)) + dt
        if rb._sleep_timer >= float(getattr(rb, "sleep_time", 0.35)):
            rb.sleep()
    else:
        rb._sleep_timer = 0.0
        rb._is_sleeping = False
