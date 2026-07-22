"""Impulse-based 2D collision response with linear + angular terms.

Maps the 3D rotational solver (``engine.d3.physics.response``) to the plane:

* Angular velocity is a **scalar** ω about Z (rad/s).
* Inverse inertia is a **scalar** I⁻¹.
* Relative velocity at contact: ``v + ω × r`` with
  ``ω × r = (-ω·ry, ω·rx)`` and torque ``τ = rx·jy − ry·jx``.

Face rest vs edge tipping
-------------------------
* **Face support** (body edge nearly parallel to contact normal): normal
  impulse through COM, strong settle, may sleep.
* **Edge/corner support**: geometric lever arm so gravity tips the body.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from engine.types.vector2 import Vector2

try:
    import os as _os
    if _os.environ.get("PYENGINE_PURE_PYTHON", "0").lower() in ("1", "true", "yes"):
        raise ImportError("Cython disabled via PYENGINE_PURE_PYTHON=1")
    from engine.cython.cy_response_2d import resolve_contact_2d_fast as _cy_resolve_contact
    _USE_CYTHON = True
except Exception:
    _USE_CYTHON = False
    _cy_resolve_contact = None

RESTITUTION_THRESHOLD = 1.0
IMPACT_BLEND_START = 1.2
IMPACT_BLEND_END = 7.0
FACE_ALIGN_THRESHOLD = 0.82
FACE_REST_ALIGN = 0.985
# Edge/corner: small in-plane offset already counts as unstable support.
UNSTABLE_SUPPORT_OFFSET = 0.06
# Face–face / face–ground: only tip when COM projects near the *edge* of the
# support face. 0.06 was far too tight for stacks (2–10 cm misalignment on a
# 1 m box is still stable in real life, but was treated as an edge and spun).
FACE_TIP_OFFSET = 0.40
MAX_NORMAL_TANGENT_ARM = 0.35
RESTING_TANGENTIAL_SPEED = 0.08
MAX_ANGULAR_SPEED = 20.0
GRAVITY = 9.81


def _as_np2(v) -> np.ndarray:
    if isinstance(v, np.ndarray):
        return np.asarray(v, dtype=np.float64).reshape(2)
    if hasattr(v, "to_numpy"):
        arr = np.asarray(v.to_numpy(), dtype=np.float64).ravel()
        return arr[:2].copy()
    if hasattr(v, "x"):
        return np.array([float(v.x), float(v.y)], dtype=np.float64)
    return np.asarray(v, dtype=np.float64).reshape(2)


def _cross_z(rx: float, ry: float, jx: float, jy: float) -> float:
    """2D cross product z-component: r × j."""
    return rx * jy - ry * jx


def _omega_cross_r(omega: float, rx: float, ry: float) -> np.ndarray:
    """ω k̂ × r = (-ω ry, ω rx)."""
    return np.array([-omega * ry, omega * rx], dtype=np.float64)


def _clamp_omega(omega: float, limit: float = MAX_ANGULAR_SPEED) -> float:
    if omega > limit:
        return limit
    if omega < -limit:
        return -limit
    return omega


def _clamp_vec2(v: np.ndarray, max_len: float) -> np.ndarray:
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
    """How well a body face (box edge normal) aligns with contact normal."""
    if game_object is None:
        return 0.0
    try:
        from engine.d2.physics.collider import CircleCollider2D, BoxCollider2D

        # Circles have no preferred face — leave 0 so friction can roll them.
        if game_object.get_component(CircleCollider2D) is not None:
            if game_object.get_component(BoxCollider2D) is None:
                return 0.0

        angle = math.radians(float(game_object.transform.rotation_z))
        c, s = math.cos(angle), math.sin(angle)
        # Local axes of a 2D OBB (face normals)
        axes = (
            np.array([c, s], dtype=np.float64),
            np.array([-s, c], dtype=np.float64),
        )
        n64 = _as_np2(n)
        best = 0.0
        for ax in axes:
            best = max(best, abs(float(np.dot(ax, n64))))
        return best
    except Exception:
        return 0.0


def _effective_mass(
    direction: np.ndarray,
    inv_mass_a: float,
    inv_mass_b: float,
    ra: np.ndarray,
    rb: np.ndarray,
    i_inv_a: Optional[float],
    i_inv_b: Optional[float],
) -> float:
    k = inv_mass_a + inv_mass_b
    dx, dy = float(direction[0]), float(direction[1])
    if i_inv_a is not None and inv_mass_a > 0.0:
        # (r × d)_z^2 * I⁻¹
        rn = _cross_z(float(ra[0]), float(ra[1]), dx, dy)
        k += float(i_inv_a) * rn * rn
    if i_inv_b is not None and inv_mass_b > 0.0:
        rn = _cross_z(float(rb[0]), float(rb[1]), dx, dy)
        k += float(i_inv_b) * rn * rn
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
    jx, jy = float(j_vec[0]), float(j_vec[1])
    if i_inv_a is not None and inv_mass_a > 0.0:
        oa = oa + float(i_inv_a) * _cross_z(float(ra[0]), float(ra[1]), jx, jy)
    if i_inv_b is not None and inv_mass_b > 0.0:
        ob = ob - float(i_inv_b) * _cross_z(float(rb[0]), float(rb[1]), jx, jy)
    return va, oa, vb, ob


def _normal_lever_arms(ra_full, rb_full, n, weight: float):
    ra_com = n * float(np.dot(ra_full, n))
    rb_com = n * float(np.dot(rb_full, n))
    ra_t = _clamp_vec2(ra_full - ra_com, MAX_NORMAL_TANGENT_ARM)
    rb_t = _clamp_vec2(rb_full - rb_com, MAX_NORMAL_TANGENT_ARM)
    return ra_com + weight * ra_t, rb_com + weight * rb_t


def resolve_contact_2d(
    *,
    pos_a,
    vel_a,
    omega_a: float,
    inv_mass_a: float,
    i_inv_a: Optional[float],
    pos_b,
    vel_b,
    omega_b: float,
    inv_mass_b: float,
    i_inv_b: Optional[float],
    contact_point,
    normal,
    restitution: float,
    static_friction: float,
    dynamic_friction: float,
    face_align_a: float = 0.0,
    face_align_b: float = 0.0,
    dt: Optional[float] = None,
) -> Tuple[np.ndarray, float, np.ndarray, float, bool]:
    """Resolve one 2D contact.

    Returns
    -------
    (vel_a, omega_a, vel_b, omega_b, unstable_support)
    """
    if _USE_CYTHON and _cy_resolve_contact is not None:
        from engine.component import Time
        if dt is None:
            dt = float(getattr(Time, "delta_time", 0.0) or (1.0 / 60.0))
        pa = np.ascontiguousarray(_as_np2(pos_a), dtype=np.float64)
        pb = np.ascontiguousarray(_as_np2(pos_b), dtype=np.float64)
        va = np.ascontiguousarray(_as_np2(vel_a), dtype=np.float64)
        vb = np.ascontiguousarray(_as_np2(vel_b), dtype=np.float64)
        cp = np.ascontiguousarray(_as_np2(contact_point), dtype=np.float64)
        n = np.ascontiguousarray(_as_np2(normal), dtype=np.float64)
        ia = -1.0 if i_inv_a is None else float(i_inv_a)
        ib = -1.0 if i_inv_b is None else float(i_inv_b)
        return _cy_resolve_contact(
            pa, va, float(omega_a), float(inv_mass_a), ia,
            pb, vb, float(omega_b), float(inv_mass_b), ib,
            cp, n,
            float(restitution), float(static_friction), float(dynamic_friction),
            float(face_align_a), float(face_align_b), float(dt),
        )

    n = _as_np2(normal)
    n_len = float(np.linalg.norm(n))
    if n_len < 1e-12:
        return _as_np2(vel_a), float(omega_a), _as_np2(vel_b), float(omega_b), False
    n = n / n_len

    pa = _as_np2(pos_a)
    pb = _as_np2(pos_b)
    va = _as_np2(vel_a)
    vb = _as_np2(vel_b)
    oa = float(omega_a)
    ob = float(omega_b)
    cp = _as_np2(contact_point)

    ra_full = cp - pa
    rb_full = cp - pb

    v_rel0 = (va + _omega_cross_r(oa, float(ra_full[0]), float(ra_full[1]))) - (
        vb + _omega_cross_r(ob, float(rb_full[0]), float(rb_full[1]))
    )
    v_n0 = float(np.dot(v_rel0, n))
    if v_n0 > 0.0:
        return va, oa, vb, ob, False

    closing = -v_n0
    w_impact = _impact_weight(closing)

    off_a = _support_offset(ra_full, n) if inv_mass_a > 0.0 else 0.0
    off_b = _support_offset(rb_full, n) if inv_mass_b > 0.0 else 0.0
    support_off = max(off_a, off_b)

    aligned_a = inv_mass_a > 0.0 and face_align_a >= FACE_ALIGN_THRESHOLD
    aligned_b = inv_mass_b > 0.0 and face_align_b >= FACE_ALIGN_THRESHOLD
    face_aligned = aligned_a or aligned_b
    best_align = max(
        face_align_a if inv_mass_a > 0.0 else 0.0,
        face_align_b if inv_mass_b > 0.0 else 0.0,
    )

    # ------------------------------------------------------------------
    # Face vs edge: driven primarily by orientation, not tiny offsets.
    # Flat stacked boxes with a few cm of misalignment must NOT tip.
    # ------------------------------------------------------------------
    ground_like = abs(float(n[1])) > 0.88

    if face_aligned or best_align >= FACE_ALIGN_THRESHOLD:
        # Face-ish contact: stable while COM projects well inside the face.
        if support_off < FACE_TIP_OFFSET:
            face_support = True
            unstable = False
        else:
            # COM near / past the face edge → can tip
            face_support = False
            unstable = True
    else:
        # Edge / corner geometry (no face aligned with the contact normal)
        face_support = False
        unstable = support_off >= UNSTABLE_SUPPORT_OFFSET or support_off > 1e-4

    # Ground: nearly flat + contact near COM → hard face rest (may sleep)
    if ground_like and closing < 2.5:
        if best_align >= FACE_REST_ALIGN and support_off < FACE_TIP_OFFSET:
            face_support = True
            unstable = False
        elif best_align < FACE_ALIGN_THRESHOLD and (
            support_off >= UNSTABLE_SUPPORT_OFFSET or best_align < FACE_REST_ALIGN
        ):
            # True edge/corner on the floor — allow tipping
            face_support = False
            unstable = True
            if support_off < 1e-3 and inv_mass_a > 0.0:
                tang = np.array([-float(n[1]), float(n[0])], dtype=np.float64)
                tl = float(np.linalg.norm(tang))
                if tl > 1e-6:
                    tang /= tl
                    lean = float(np.dot(ra_full, tang))
                    if abs(lean) > 1e-6:
                        ra_full = ra_full + tang * (0.03 if lean > 0.0 else -0.03)
                        support_off = 0.03

    if face_support:
        # Resting / stacked faces: normal force through COM (no phantom spin)
        w_n = 0.0 if closing < 2.0 else w_impact * 0.05
    elif unstable:
        tip = min(1.0, max(support_off, 0.02) / 0.25)
        w_n = max(w_impact, 0.75 + 0.25 * tip)
    else:
        w_n = w_impact

    ra_n, rb_n = _normal_lever_arms(ra_full, rb_full, n, w_n)

    v_rel_n = (va + _omega_cross_r(oa, float(ra_n[0]), float(ra_n[1]))) - (
        vb + _omega_cross_r(ob, float(rb_n[0]), float(rb_n[1]))
    )
    v_n = float(np.dot(v_rel_n, n))
    if v_n > 0.0:
        return va, oa, vb, ob, unstable

    kn = _effective_mass(n, inv_mass_a, inv_mass_b, ra_n, rb_n, i_inv_a, i_inv_b)
    if kn < 1e-12:
        return va, oa, vb, ob, unstable

    e = max(0.0, min(1.0, float(restitution)))
    if closing < RESTITUTION_THRESHOLD or (face_support and closing < 2.5):
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

    # Gravity tipping: only for true edge/corner support (not flat stacks).
    # Face-aligned contacts with a modest COM offset must not invent torque.
    allow_gravity_tip = (
        unstable
        and not face_support
        and best_align < FACE_ALIGN_THRESHOLD
        and ground_like
        and closing < 3.0
    )
    if allow_gravity_tip and inv_mass_a > 0.0 and i_inv_a is not None:
        from engine.component import Time
        step = float(dt if dt is not None else getattr(Time, "delta_time", 0.0) or (1.0 / 60.0))
        step = max(1e-5, min(step, 0.05))
        mass_a = 1.0 / inv_mass_a
        support = n * (mass_a * GRAVITY * max(0.0, float(n[1])))
        if float(np.linalg.norm(support)) < 1e-8:
            support = n * (mass_a * GRAVITY)
        tau_imp = _cross_z(
            float(ra_full[0]), float(ra_full[1]),
            float(support[0]), float(support[1]),
        ) * step
        oa = oa + float(i_inv_a) * tau_imp
    if allow_gravity_tip and inv_mass_b > 0.0 and i_inv_b is not None:
        from engine.component import Time
        step = float(dt if dt is not None else getattr(Time, "delta_time", 0.0) or (1.0 / 60.0))
        step = max(1e-5, min(step, 0.05))
        mass_b = 1.0 / inv_mass_b
        support = n * (mass_b * GRAVITY * max(0.0, float(n[1])))
        if float(np.linalg.norm(support)) < 1e-8:
            support = n * (mass_b * GRAVITY)
        tau_imp = _cross_z(
            float(rb_full[0]), float(rb_full[1]),
            float(support[0]), float(support[1]),
        ) * step
        ob = ob + float(i_inv_b) * tau_imp

    # --- Friction ---
    if face_support:
        ra_f = ra_n
        rb_f = rb_n
        # Stacks / floor rest: linear friction only (no spin-up from scrapes)
        friction_angular = closing > 3.0
    elif unstable:
        ra_f, rb_f = ra_full, rb_full
        friction_angular = True
    else:
        ra_f = ra_full * w_n + ra_n * (1.0 - w_n)
        rb_f = rb_full * w_n + rb_n * (1.0 - w_n)
        friction_angular = w_n > 0.2

    v_rel = (va + _omega_cross_r(oa, float(ra_f[0]), float(ra_f[1]))) - (
        vb + _omega_cross_r(ob, float(rb_f[0]), float(rb_f[1]))
    )
    v_t = v_rel - n * float(np.dot(v_rel, n))
    t_mag = float(np.linalg.norm(v_t))
    mu_s = max(0.0, float(static_friction))
    mu_d = max(0.0, float(dynamic_friction))

    if t_mag >= 1e-10:
        t = v_t / t_mag
        if not friction_angular or t_mag < RESTING_TANGENTIAL_SPEED:
            kt_lin = inv_mass_a + inv_mass_b
            if kt_lin > 1e-12:
                jt = -t_mag / kt_lin
                max_f = abs(jn) * max(mu_s, mu_d)
                if abs(jt) > max_f:
                    jt = math.copysign(max_f, jt) if max_f > 0.0 else 0.0
                jt_vec = t * jt
                va = va + jt_vec * inv_mass_a
                vb = vb - jt_vec * inv_mass_b
                if face_support:
                    oa *= 0.5
                    ob *= 0.5
        else:
            kt = _effective_mass(
                t, inv_mass_a, inv_mass_b, ra_f, rb_f, i_inv_a, i_inv_b
            )
            if kt > 1e-12:
                vt_along = float(np.dot(v_rel, t))
                jt = -vt_along / kt
                max_static = abs(jn) * mu_s
                if abs(jt) > max_static:
                    jt = math.copysign(abs(jn) * mu_d, jt)
                if unstable:
                    jt *= 0.45
                if vt_along * (vt_along + jt * kt) < 0.0:
                    jt = -vt_along / kt
                jt_vec = t * jt
                va, oa, vb, ob = _apply_impulse(
                    va, oa, vb, ob,
                    inv_mass_a, inv_mass_b, i_inv_a, i_inv_b,
                    ra_f, rb_f, jt_vec,
                )

    # Face settle: only full rest snap when contacting *immovable* support.
    # Dynamic–dynamic mid-air contacts used to zero both velocities and then
    # sleep — leaving bodies floating forever.
    a_on_static = inv_mass_b < 1e-12 and inv_mass_a > 0.0
    b_on_static = inv_mass_a < 1e-12 and inv_mass_b > 0.0
    if face_support and closing < 2.5:
        if a_on_static:
            if abs(oa) < 1.5:
                oa = 0.0
            else:
                oa *= 0.15
            speed = float(np.linalg.norm(va))
            if speed < 0.15 or (mu_s > 1e-6 and speed < 0.75):
                va = np.zeros(2, dtype=np.float64)
            elif speed < 1.2 and mu_s > 1e-6:
                va = va * 0.35
        elif inv_mass_a > 0.0:
            # Dynamic partner: kill phantom spin only. If both are already slow
            # on a vertical support contact (stack), gently damp residual so
            # piles settle — but never hard-zero free-fall velocities mid-air.
            if abs(oa) < 0.8:
                oa = 0.0
            else:
                oa *= 0.5
            if (
                ground_like
                and closing < 0.8
                and float(np.linalg.norm(va)) < 0.55
                and float(np.linalg.norm(vb)) < 0.55
            ):
                va = va * 0.4
        if b_on_static:
            if abs(ob) < 1.5:
                ob = 0.0
            else:
                ob *= 0.15
            speed_b = float(np.linalg.norm(vb))
            if speed_b < 0.15 or (mu_s > 1e-6 and speed_b < 0.75):
                vb = np.zeros(2, dtype=np.float64)
            elif speed_b < 1.2 and mu_s > 1e-6:
                vb = vb * 0.35
        elif inv_mass_b > 0.0:
            if abs(ob) < 0.8:
                ob = 0.0
            else:
                ob *= 0.5
            if (
                ground_like
                and closing < 0.8
                and float(np.linalg.norm(va)) < 0.55
                and float(np.linalg.norm(vb)) < 0.55
            ):
                vb = vb * 0.4
    elif not unstable and w_n < 0.5:
        damp = 0.72 + 0.2 * w_n
        if inv_mass_b < 1e-12 and inv_mass_a > 0.0:
            oa *= damp
        elif inv_mass_a < 1e-12 and inv_mass_b > 0.0:
            ob *= damp

    return va, _clamp_omega(oa), vb, _clamp_omega(ob), unstable


def estimate_contact_point(pos_a, pos_b, normal, depth: float = 0.0) -> np.ndarray:
    pa = _as_np2(pos_a)
    pb = _as_np2(pos_b)
    n = _as_np2(normal)
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
    """For face support, snap contact under COM; keep edge features otherwise.

    Stacked flat faces with a few cm of misalignment still count as face
    support — route the normal through the COM so no phantom torque appears.
    """
    pa = _as_np2(pos_a)
    pb = _as_np2(pos_b)
    cp = _as_np2(contact_point)
    n = _as_np2(normal)
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

    # Snap under COM for any face-like contact whose COM still projects inside
    # the support (not only near-perfect flat + tiny offset).
    if best_align >= FACE_ALIGN_THRESHOLD and offset < FACE_TIP_OFFSET:
        return com_ref
    return cp


def body_state_from_rigidbody(rb, game_object, is_immovable: bool):
    if game_object is None:
        return (
            np.zeros(2, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            0.0,
            0.0,
            None,
        )

    pos3 = game_object.transform.position
    pos = np.array([float(pos3.x), float(pos3.y)], dtype=np.float64)

    if rb is None or is_immovable:
        return pos, np.zeros(2, dtype=np.float64), 0.0, 0.0, None

    vel = _as_np2(rb.velocity)
    omega = float(rb.angular_velocity)
    mass = float(rb.mass) if rb.mass > 1e-10 else 1e-10
    inv_mass = 1.0 / mass
    try:
        i_inv = float(rb.get_inertia_inv())
    except Exception:
        i_inv = inv_mass * 6.0
    return pos, vel, omega, inv_mass, i_inv


def apply_body_state(
    rb,
    vel: np.ndarray,
    omega: float,
    *,
    allow_sleep: bool = True,
) -> None:
    if rb is None:
        return
    from engine.component import Time

    lin = float(np.linalg.norm(vel))
    ang = abs(float(omega))

    if allow_sleep:
        # Slightly looser snap so gravity-step residuals (~0.16) still rest
        if lin < 0.20:
            vel = np.zeros(2, dtype=np.float64)
            lin = 0.0
        if ang < 0.25:
            omega = 0.0
            ang = 0.0

    max_w = float(getattr(rb, "max_angular_velocity", MAX_ANGULAR_SPEED))
    omega = _clamp_omega(float(omega), max_w)

    rb._velocity = Vector2(float(vel[0]), float(vel[1]))
    rb._angular_velocity = float(omega)

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
