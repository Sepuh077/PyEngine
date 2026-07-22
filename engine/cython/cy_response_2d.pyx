# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""
Cython-accelerated 2D contact response (linear + angular impulses).

Matches engine.d2.physics.response.resolve_contact_2d semantics
(including face-rest stacks, no mid-air freeze, edge tipping only).
ω is a scalar about Z; I⁻¹ is a scalar (pass < 0 to disable angular for a body).
"""

from libc.math cimport sqrt, fabs, copysign
import numpy as np
cimport numpy as cnp

cnp.import_array()

cdef double RESTITUTION_THRESHOLD = 1.0
cdef double IMPACT_BLEND_START = 1.2
cdef double IMPACT_BLEND_END = 7.0
cdef double FACE_ALIGN_THRESHOLD = 0.82
cdef double FACE_REST_ALIGN = 0.985
cdef double UNSTABLE_SUPPORT_OFFSET = 0.06
cdef double FACE_TIP_OFFSET = 0.40
cdef double MAX_NORMAL_TANGENT_ARM = 0.35
cdef double RESTING_TANGENTIAL_SPEED = 0.08
cdef double MAX_ANGULAR_SPEED = 20.0
cdef double GRAVITY = 9.81


cdef inline double _dot2(double ax, double ay, double bx, double by) noexcept nogil:
    return ax * bx + ay * by


cdef inline double _len2(double x, double y) noexcept nogil:
    return sqrt(x * x + y * y)


cdef inline double _cross_z(double rx, double ry, double jx, double jy) noexcept nogil:
    return rx * jy - ry * jx


cdef inline void _clamp_vec2(double *x, double *y, double max_len) noexcept nogil:
    cdef double mag = _len2(x[0], y[0])
    cdef double s
    if mag > max_len and mag > 1e-12:
        s = max_len / mag
        x[0] *= s
        y[0] *= s


cdef inline double _clamp_omega(double omega, double limit) noexcept nogil:
    if omega > limit:
        return limit
    if omega < -limit:
        return -limit
    return omega


cdef inline double _impact_weight(double closing) noexcept nogil:
    cdef double t
    if closing <= IMPACT_BLEND_START:
        return 0.0
    if closing >= IMPACT_BLEND_END:
        return 1.0
    t = (closing - IMPACT_BLEND_START) / (IMPACT_BLEND_END - IMPACT_BLEND_START)
    return t * t * (3.0 - 2.0 * t)


cdef inline double _support_offset(double rx, double ry, double nx, double ny) noexcept nogil:
    cdef double dn = _dot2(rx, ry, nx, ny)
    cdef double tx = rx - nx * dn
    cdef double ty = ry - ny * dn
    return _len2(tx, ty)


cdef inline double _effective_mass(
    double dx, double dy,
    double inv_mass_a, double inv_mass_b,
    double rax, double ray, double rbx, double rby,
    bint has_ia, double i_inv_a,
    bint has_ib, double i_inv_b,
) noexcept nogil:
    cdef double k = inv_mass_a + inv_mass_b
    cdef double rn
    if has_ia and inv_mass_a > 0.0:
        rn = _cross_z(rax, ray, dx, dy)
        k += i_inv_a * rn * rn
    if has_ib and inv_mass_b > 0.0:
        rn = _cross_z(rbx, rby, dx, dy)
        k += i_inv_b * rn * rn
    return k


cdef inline void _apply_impulse(
    double *vax, double *vay, double *oa,
    double *vbx, double *vby, double *ob,
    double inv_mass_a, double inv_mass_b,
    bint has_ia, double i_inv_a,
    bint has_ib, double i_inv_b,
    double rax, double ray, double rbx, double rby,
    double jx, double jy,
) noexcept nogil:
    vax[0] += jx * inv_mass_a
    vay[0] += jy * inv_mass_a
    vbx[0] -= jx * inv_mass_b
    vby[0] -= jy * inv_mass_b
    if has_ia and inv_mass_a > 0.0:
        oa[0] += i_inv_a * _cross_z(rax, ray, jx, jy)
    if has_ib and inv_mass_b > 0.0:
        ob[0] -= i_inv_b * _cross_z(rbx, rby, jx, jy)


cdef void _resolve_contact_core(
    double pax, double pay,
    double *vax, double *vay, double *oa,
    double inv_mass_a, bint has_ia, double i_inv_a,
    double pbx, double pby,
    double *vbx, double *vby, double *ob,
    double inv_mass_b, bint has_ib, double i_inv_b,
    double cpx, double cpy,
    double nx, double ny,
    double restitution, double static_friction, double dynamic_friction,
    double face_align_a, double face_align_b,
    double dt,
    bint *out_unstable,
) noexcept nogil:
    cdef double n_len = _len2(nx, ny)
    cdef double inv_n
    cdef double rax, ray, rbx, rby
    cdef double vrx, vry, v_n0, closing, w_impact
    cdef double off_a, off_b, support_off
    cdef bint aligned_a, aligned_b, face_aligned, face_support, unstable, ground_like
    cdef bint allow_gravity_tip, a_on_static, b_on_static
    cdef double best_align, w_n, tip
    cdef double tipx, tipy, tl, dn, lean
    cdef double ra_nx, ra_ny, rb_nx, rb_ny
    cdef double ra_tx, ra_ty, rb_tx, rb_ty
    cdef double ra_cx, ra_cy, rb_cx, rb_cy
    cdef double v_n, kn, e, jn, jnx, jny
    cdef double mass, sx, sy, tau
    cdef double ra_fx, ra_fy, rb_fx, rb_fy
    cdef bint friction_angular
    cdef double vtx, vty, t_mag, tx, ty
    cdef double mu_s, mu_d, kt_lin, jt, max_f, max_static, vt_along, kt
    cdef double speed, speed_b, damp

    out_unstable[0] = 0
    if n_len < 1e-12:
        return
    inv_n = 1.0 / n_len
    nx *= inv_n
    ny *= inv_n

    rax = cpx - pax; ray = cpy - pay
    rbx = cpx - pbx; rby = cpy - pby

    # v_rel = (va + ω×r) - (vb + ω×r)
    vrx = vax[0] + (-oa[0] * ray)
    vry = vay[0] + (oa[0] * rax)
    vrx -= vbx[0] + (-ob[0] * rby)
    vry -= vby[0] + (ob[0] * rbx)

    v_n0 = _dot2(vrx, vry, nx, ny)
    if v_n0 > 0.0:
        return

    closing = -v_n0
    w_impact = _impact_weight(closing)

    off_a = _support_offset(rax, ray, nx, ny) if inv_mass_a > 0.0 else 0.0
    off_b = _support_offset(rbx, rby, nx, ny) if inv_mass_b > 0.0 else 0.0
    support_off = off_a if off_a > off_b else off_b

    aligned_a = inv_mass_a > 0.0 and face_align_a >= FACE_ALIGN_THRESHOLD
    aligned_b = inv_mass_b > 0.0 and face_align_b >= FACE_ALIGN_THRESHOLD
    face_aligned = aligned_a or aligned_b
    best_align = 0.0
    if inv_mass_a > 0.0 and face_align_a > best_align:
        best_align = face_align_a
    if inv_mass_b > 0.0 and face_align_b > best_align:
        best_align = face_align_b

    ground_like = fabs(ny) > 0.88

    # Face vs edge (orientation-first, generous face tip offset for stacks)
    if face_aligned or best_align >= FACE_ALIGN_THRESHOLD:
        if support_off < FACE_TIP_OFFSET:
            face_support = 1
            unstable = 0
        else:
            face_support = 0
            unstable = 1
    else:
        face_support = 0
        unstable = support_off >= UNSTABLE_SUPPORT_OFFSET or support_off > 1e-4

    if ground_like and closing < 2.5:
        if best_align >= FACE_REST_ALIGN and support_off < FACE_TIP_OFFSET:
            face_support = 1
            unstable = 0
        elif best_align < FACE_ALIGN_THRESHOLD and (
            support_off >= UNSTABLE_SUPPORT_OFFSET or best_align < FACE_REST_ALIGN
        ):
            face_support = 0
            unstable = 1
            if support_off < 1e-3 and inv_mass_a > 0.0:
                tipx = -ny; tipy = nx
                tl = _len2(tipx, tipy)
                if tl > 1e-6:
                    tipx /= tl; tipy /= tl
                    lean = rax * tipx + ray * tipy
                    if fabs(lean) > 1e-6:
                        if lean < 0.0:
                            tipx = -tipx; tipy = -tipy
                        rax += tipx * 0.03
                        ray += tipy * 0.03
                        support_off = 0.03

    if face_support:
        w_n = 0.0 if closing < 2.0 else w_impact * 0.05
    elif unstable:
        tip = support_off if support_off > 0.02 else 0.02
        if tip > 0.25:
            tip = 0.25
        tip = tip / 0.25
        if tip > 1.0:
            tip = 1.0
        w_n = w_impact if w_impact > (0.75 + 0.25 * tip) else (0.75 + 0.25 * tip)
    else:
        w_n = w_impact

    # lever arms
    dn = _dot2(rax, ray, nx, ny)
    ra_cx = nx * dn; ra_cy = ny * dn
    ra_tx = rax - ra_cx; ra_ty = ray - ra_cy
    _clamp_vec2(&ra_tx, &ra_ty, MAX_NORMAL_TANGENT_ARM)
    ra_nx = ra_cx + w_n * ra_tx
    ra_ny = ra_cy + w_n * ra_ty

    dn = _dot2(rbx, rby, nx, ny)
    rb_cx = nx * dn; rb_cy = ny * dn
    rb_tx = rbx - rb_cx; rb_ty = rby - rb_cy
    _clamp_vec2(&rb_tx, &rb_ty, MAX_NORMAL_TANGENT_ARM)
    rb_nx = rb_cx + w_n * rb_tx
    rb_ny = rb_cy + w_n * rb_ty

    vrx = vax[0] + (-oa[0] * ra_ny)
    vry = vay[0] + (oa[0] * ra_nx)
    vrx -= vbx[0] + (-ob[0] * rb_ny)
    vry -= vby[0] + (ob[0] * rb_nx)
    v_n = _dot2(vrx, vry, nx, ny)
    if v_n > 0.0:
        out_unstable[0] = unstable
        return

    kn = _effective_mass(nx, ny, inv_mass_a, inv_mass_b,
                         ra_nx, ra_ny, rb_nx, rb_ny,
                         has_ia, i_inv_a, has_ib, i_inv_b)
    if kn < 1e-12:
        out_unstable[0] = unstable
        return

    e = restitution
    if e < 0.0:
        e = 0.0
    if e > 1.0:
        e = 1.0
    if face_support and closing < 2.5:
        e = 0.0
    elif closing < RESTITUTION_THRESHOLD:
        e = 0.0

    jn = -(1.0 + e) * v_n / kn
    if jn < 0.0:
        jn = 0.0
    jnx = nx * jn; jny = ny * jn
    _apply_impulse(vax, vay, oa, vbx, vby, ob,
                   inv_mass_a, inv_mass_b, has_ia, i_inv_a, has_ib, i_inv_b,
                   ra_nx, ra_ny, rb_nx, rb_ny, jnx, jny)

    # Gravity tip: only true edge/corner (not face stacks)
    allow_gravity_tip = (
        unstable
        and (not face_support)
        and best_align < FACE_ALIGN_THRESHOLD
        and ground_like
        and closing < 3.0
    )
    if allow_gravity_tip and inv_mass_a > 0.0 and has_ia:
        if dt < 1e-5:
            dt = 1e-5
        if dt > 0.05:
            dt = 0.05
        mass = 1.0 / inv_mass_a
        sx = nx * (mass * GRAVITY * (ny if ny > 0.0 else 0.0))
        sy = ny * (mass * GRAVITY * (ny if ny > 0.0 else 0.0))
        if _len2(sx, sy) < 1e-8:
            sx = nx * (mass * GRAVITY)
            sy = ny * (mass * GRAVITY)
        tau = _cross_z(rax, ray, sx, sy) * dt
        oa[0] += i_inv_a * tau

    if allow_gravity_tip and inv_mass_b > 0.0 and has_ib:
        if dt < 1e-5:
            dt = 1e-5
        if dt > 0.05:
            dt = 0.05
        mass = 1.0 / inv_mass_b
        sx = nx * (mass * GRAVITY * (ny if ny > 0.0 else 0.0))
        sy = ny * (mass * GRAVITY * (ny if ny > 0.0 else 0.0))
        if _len2(sx, sy) < 1e-8:
            sx = nx * (mass * GRAVITY)
            sy = ny * (mass * GRAVITY)
        tau = _cross_z(rbx, rby, sx, sy) * dt
        ob[0] += i_inv_b * tau

    # Friction
    if face_support:
        ra_fx = ra_nx; ra_fy = ra_ny
        rb_fx = rb_nx; rb_fy = rb_ny
        friction_angular = closing > 3.0
    elif unstable:
        ra_fx = rax; ra_fy = ray
        rb_fx = rbx; rb_fy = rby
        friction_angular = 1
    else:
        ra_fx = rax * w_n + ra_nx * (1.0 - w_n)
        ra_fy = ray * w_n + ra_ny * (1.0 - w_n)
        rb_fx = rbx * w_n + rb_nx * (1.0 - w_n)
        rb_fy = rby * w_n + rb_ny * (1.0 - w_n)
        friction_angular = w_n > 0.2

    vrx = vax[0] + (-oa[0] * ra_fy)
    vry = vay[0] + (oa[0] * ra_fx)
    vrx -= vbx[0] + (-ob[0] * rb_fy)
    vry -= vby[0] + (ob[0] * rb_fx)
    dn = _dot2(vrx, vry, nx, ny)
    vtx = vrx - nx * dn; vty = vry - ny * dn
    t_mag = _len2(vtx, vty)
    mu_s = static_friction if static_friction > 0.0 else 0.0
    mu_d = dynamic_friction if dynamic_friction > 0.0 else 0.0

    if t_mag >= 1e-10:
        tx = vtx / t_mag; ty = vty / t_mag
        if (not friction_angular) or t_mag < RESTING_TANGENTIAL_SPEED:
            kt_lin = inv_mass_a + inv_mass_b
            if kt_lin > 1e-12:
                jt = -t_mag / kt_lin
                max_f = fabs(jn) * (mu_s if mu_s > mu_d else mu_d)
                if fabs(jt) > max_f:
                    jt = copysign(max_f, jt) if max_f > 0.0 else 0.0
                vax[0] += tx * jt * inv_mass_a
                vay[0] += ty * jt * inv_mass_a
                vbx[0] -= tx * jt * inv_mass_b
                vby[0] -= ty * jt * inv_mass_b
                if face_support:
                    oa[0] *= 0.5
                    ob[0] *= 0.5
        else:
            kt = _effective_mass(tx, ty, inv_mass_a, inv_mass_b,
                                 ra_fx, ra_fy, rb_fx, rb_fy,
                                 has_ia, i_inv_a, has_ib, i_inv_b)
            if kt > 1e-12:
                vt_along = _dot2(vrx, vry, tx, ty)
                jt = -vt_along / kt
                max_static = fabs(jn) * mu_s
                if fabs(jt) > max_static:
                    jt = copysign(fabs(jn) * mu_d, jt)
                if unstable:
                    jt *= 0.45
                if vt_along * (vt_along + jt * kt) < 0.0:
                    jt = -vt_along / kt
                _apply_impulse(vax, vay, oa, vbx, vby, ob,
                               inv_mass_a, inv_mass_b, has_ia, i_inv_a, has_ib, i_inv_b,
                               ra_fx, ra_fy, rb_fx, rb_fy, tx * jt, ty * jt)

    # Face settle: full snap only on immovable support (no mid-air freeze)
    a_on_static = inv_mass_b < 1e-12 and inv_mass_a > 0.0
    b_on_static = inv_mass_a < 1e-12 and inv_mass_b > 0.0
    if face_support and closing < 2.5:
        if a_on_static:
            if fabs(oa[0]) < 1.5:
                oa[0] = 0.0
            else:
                oa[0] *= 0.15
            speed = _len2(vax[0], vay[0])
            if speed < 0.15 or (mu_s > 1e-6 and speed < 0.75):
                vax[0] = 0.0; vay[0] = 0.0
            elif speed < 1.2 and mu_s > 1e-6:
                vax[0] *= 0.35; vay[0] *= 0.35
        elif inv_mass_a > 0.0:
            if fabs(oa[0]) < 0.8:
                oa[0] = 0.0
            else:
                oa[0] *= 0.5
            if (ground_like and closing < 0.8
                    and _len2(vax[0], vay[0]) < 0.55
                    and _len2(vbx[0], vby[0]) < 0.55):
                vax[0] *= 0.4; vay[0] *= 0.4
        if b_on_static:
            if fabs(ob[0]) < 1.5:
                ob[0] = 0.0
            else:
                ob[0] *= 0.15
            speed_b = _len2(vbx[0], vby[0])
            if speed_b < 0.15 or (mu_s > 1e-6 and speed_b < 0.75):
                vbx[0] = 0.0; vby[0] = 0.0
            elif speed_b < 1.2 and mu_s > 1e-6:
                vbx[0] *= 0.35; vby[0] *= 0.35
        elif inv_mass_b > 0.0:
            if fabs(ob[0]) < 0.8:
                ob[0] = 0.0
            else:
                ob[0] *= 0.5
            if (ground_like and closing < 0.8
                    and _len2(vax[0], vay[0]) < 0.55
                    and _len2(vbx[0], vby[0]) < 0.55):
                vbx[0] *= 0.4; vby[0] *= 0.4
    elif (not unstable) and w_n < 0.5:
        damp = 0.72 + 0.2 * w_n
        if inv_mass_b < 1e-12 and inv_mass_a > 0.0:
            oa[0] *= damp
        elif inv_mass_a < 1e-12 and inv_mass_b > 0.0:
            ob[0] *= damp

    oa[0] = _clamp_omega(oa[0], MAX_ANGULAR_SPEED)
    ob[0] = _clamp_omega(ob[0], MAX_ANGULAR_SPEED)
    out_unstable[0] = unstable


def resolve_contact_2d_fast(
    double[::1] pos_a, double[::1] vel_a, double omega_a,
    double inv_mass_a, double i_inv_a,
    double[::1] pos_b, double[::1] vel_b, double omega_b,
    double inv_mass_b, double i_inv_b,
    double[::1] contact_point, double[::1] normal,
    double restitution, double static_friction, double dynamic_friction,
    double face_align_a=0.0, double face_align_b=0.0,
    double dt=0.016666666666666666,
):
    """
    Fast 2D contact solve.

    i_inv_a / i_inv_b: scalar inverse inertia; pass < 0 to disable angular terms.

    Returns (va, oa, vb, ob, unstable)
    """
    cdef double vax = vel_a[0], vay = vel_a[1]
    cdef double vbx = vel_b[0], vby = vel_b[1]
    cdef double oa = omega_a, ob = omega_b
    cdef bint has_ia = i_inv_a >= 0.0
    cdef bint has_ib = i_inv_b >= 0.0
    cdef bint unstable = 0
    cdef cnp.ndarray[cnp.float64_t, ndim=1] va_out
    cdef cnp.ndarray[cnp.float64_t, ndim=1] vb_out

    if not has_ia:
        i_inv_a = 0.0
    if not has_ib:
        i_inv_b = 0.0

    _resolve_contact_core(
        pos_a[0], pos_a[1],
        &vax, &vay, &oa, inv_mass_a, has_ia, i_inv_a,
        pos_b[0], pos_b[1],
        &vbx, &vby, &ob, inv_mass_b, has_ib, i_inv_b,
        contact_point[0], contact_point[1],
        normal[0], normal[1],
        restitution, static_friction, dynamic_friction,
        face_align_a, face_align_b, dt, &unstable,
    )

    va_out = np.empty(2, dtype=np.float64)
    vb_out = np.empty(2, dtype=np.float64)
    va_out[0] = vax; va_out[1] = vay
    vb_out[0] = vbx; vb_out[1] = vby
    return (va_out, float(oa), vb_out, float(ob), bool(unstable))
