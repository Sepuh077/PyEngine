# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""
Cython-accelerated 3D contact response (linear + angular impulses).

Matches engine.d3.physics.response.resolve_contact_3d semantics.
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
cdef double MAX_NORMAL_TANGENT_ARM = 0.35
cdef double RESTING_TANGENTIAL_SPEED = 0.08
cdef double MAX_ANGULAR_SPEED = 20.0
cdef double GRAVITY = 9.81


cdef inline void _cross3(double ax, double ay, double az,
                         double bx, double by, double bz,
                         double *ox, double *oy, double *oz) noexcept nogil:
    ox[0] = ay * bz - az * by
    oy[0] = az * bx - ax * bz
    oz[0] = ax * by - ay * bx


cdef inline double _dot3(double ax, double ay, double az,
                         double bx, double by, double bz) noexcept nogil:
    return ax * bx + ay * by + az * bz


cdef inline double _len3(double x, double y, double z) noexcept nogil:
    return sqrt(x * x + y * y + z * z)


cdef inline void _clamp_vec3(double *x, double *y, double *z, double max_len) noexcept nogil:
    cdef double mag = _len3(x[0], y[0], z[0])
    cdef double s
    if mag > max_len and mag > 1e-12:
        s = max_len / mag
        x[0] *= s
        y[0] *= s
        z[0] *= s


cdef inline void _clamp_omega3(double *x, double *y, double *z, double limit) noexcept nogil:
    cdef double mag = _len3(x[0], y[0], z[0])
    cdef double s
    if mag > limit and mag > 1e-12:
        s = limit / mag
        x[0] *= s
        y[0] *= s
        z[0] *= s


cdef inline double _impact_weight(double closing) noexcept nogil:
    cdef double t
    if closing <= IMPACT_BLEND_START:
        return 0.0
    if closing >= IMPACT_BLEND_END:
        return 1.0
    t = (closing - IMPACT_BLEND_START) / (IMPACT_BLEND_END - IMPACT_BLEND_START)
    return t * t * (3.0 - 2.0 * t)


cdef inline double _support_offset(double rx, double ry, double rz,
                                   double nx, double ny, double nz) noexcept nogil:
    cdef double dn = _dot3(rx, ry, rz, nx, ny, nz)
    cdef double tx = rx - nx * dn
    cdef double ty = ry - ny * dn
    cdef double tz = rz - nz * dn
    return _len3(tx, ty, tz)


cdef inline double _mat_vec_dot(double[:, ::1] M,
                                double vx, double vy, double vz,
                                double wx, double wy, double wz) noexcept nogil:
    """w · (M @ v) for 3x3 M."""
    cdef double mx = M[0, 0] * vx + M[0, 1] * vy + M[0, 2] * vz
    cdef double my = M[1, 0] * vx + M[1, 1] * vy + M[1, 2] * vz
    cdef double mz = M[2, 0] * vx + M[2, 1] * vy + M[2, 2] * vz
    return mx * wx + my * wy + mz * wz


cdef inline void _mat_mul_vec(double[:, ::1] M,
                              double vx, double vy, double vz,
                              double *ox, double *oy, double *oz) noexcept nogil:
    ox[0] = M[0, 0] * vx + M[0, 1] * vy + M[0, 2] * vz
    oy[0] = M[1, 0] * vx + M[1, 1] * vy + M[1, 2] * vz
    oz[0] = M[2, 0] * vx + M[2, 1] * vy + M[2, 2] * vz


cdef inline double _effective_mass(
    double dx, double dy, double dz,
    double inv_mass_a, double inv_mass_b,
    double rax, double ray, double raz,
    double rbx, double rby, double rbz,
    bint has_ia, double[:, ::1] ia,
    bint has_ib, double[:, ::1] ib,
) noexcept nogil:
    cdef double k = inv_mass_a + inv_mass_b
    cdef double cx, cy, cz
    if has_ia and inv_mass_a > 0.0:
        _cross3(rax, ray, raz, dx, dy, dz, &cx, &cy, &cz)
        k += _mat_vec_dot(ia, cx, cy, cz, cx, cy, cz)
    if has_ib and inv_mass_b > 0.0:
        _cross3(rbx, rby, rbz, dx, dy, dz, &cx, &cy, &cz)
        k += _mat_vec_dot(ib, cx, cy, cz, cx, cy, cz)
    return k


cdef inline void _apply_impulse(
    double *vax, double *vay, double *vaz,
    double *oax, double *oay, double *oaz,
    double *vbx, double *vby, double *vbz,
    double *obx, double *oby, double *obz,
    double inv_mass_a, double inv_mass_b,
    bint has_ia, double[:, ::1] ia,
    bint has_ib, double[:, ::1] ib,
    double rax, double ray, double raz,
    double rbx, double rby, double rbz,
    double jx, double jy, double jz,
) noexcept nogil:
    cdef double cx, cy, cz, tx, ty, tz
    vax[0] += jx * inv_mass_a
    vay[0] += jy * inv_mass_a
    vaz[0] += jz * inv_mass_a
    vbx[0] -= jx * inv_mass_b
    vby[0] -= jy * inv_mass_b
    vbz[0] -= jz * inv_mass_b
    if has_ia and inv_mass_a > 0.0:
        _cross3(rax, ray, raz, jx, jy, jz, &cx, &cy, &cz)
        _mat_mul_vec(ia, cx, cy, cz, &tx, &ty, &tz)
        oax[0] += tx
        oay[0] += ty
        oaz[0] += tz
    if has_ib and inv_mass_b > 0.0:
        _cross3(rbx, rby, rbz, jx, jy, jz, &cx, &cy, &cz)
        _mat_mul_vec(ib, cx, cy, cz, &tx, &ty, &tz)
        obx[0] -= tx
        oby[0] -= ty
        obz[0] -= tz


cdef void _resolve_contact_core(
    double pax, double pay, double paz,
    double *vax, double *vay, double *vaz,
    double *oax, double *oay, double *oaz,
    double inv_mass_a, bint has_ia, double[:, ::1] ia,
    double pbx, double pby, double pbz,
    double *vbx, double *vby, double *vbz,
    double *obx, double *oby, double *obz,
    double inv_mass_b, bint has_ib, double[:, ::1] ib,
    double cpx, double cpy, double cpz,
    double nx, double ny, double nz,
    double restitution, double static_friction, double dynamic_friction,
    double face_align_a, double face_align_b,
    double dt,
    bint *out_unstable,
) noexcept nogil:
    cdef double n_len = _len3(nx, ny, nz)
    cdef double inv_n
    cdef double rax, ray, raz, rbx, rby, rbz
    cdef double vrx, vry, vrz, v_n0, closing, w_impact
    cdef double off_a, off_b, support_off
    cdef bint aligned_a, aligned_b, face_aligned, face_support, unstable, ground_like
    cdef double best_align, w_n, tip
    cdef double edge_x, edge_y, edge_z, el, tip_x, tip_y, tip_z, tl, dn
    cdef double ra_nx, ra_ny, ra_nz, rb_nx, rb_ny, rb_nz
    cdef double ra_tx, ra_ty, ra_tz, rb_tx, rb_ty, rb_tz
    cdef double v_n, kn, e, jn
    cdef double jnx, jny, jnz
    cdef double mass, sx, sy, sz, tau_x, tau_y, tau_z, ox, oy, oz
    cdef double ra_fx, ra_fy, ra_fz, rb_fx, rb_fy, rb_fz
    cdef bint friction_angular
    cdef double vtx, vty, vtz, t_mag, tx, ty, tz
    cdef double mu_s, mu_d, kt_lin, jt, max_f, max_static, vt_along, kt
    cdef double speed, damp
    cdef double ra_cx, ra_cy, ra_cz, rb_cx, rb_cy, rb_cz

    out_unstable[0] = 0
    if n_len < 1e-12:
        return
    inv_n = 1.0 / n_len
    nx *= inv_n
    ny *= inv_n
    nz *= inv_n

    rax = cpx - pax; ray = cpy - pay; raz = cpz - paz
    rbx = cpx - pbx; rby = cpy - pby; rbz = cpz - pbz

    # v_rel at contact
    _cross3(oax[0], oay[0], oaz[0], rax, ray, raz, &ox, &oy, &oz)
    vrx = vax[0] + ox
    vry = vay[0] + oy
    vrz = vaz[0] + oz
    _cross3(obx[0], oby[0], obz[0], rbx, rby, rbz, &ox, &oy, &oz)
    vrx -= vbx[0] + ox
    vry -= vby[0] + oy
    vrz -= vbz[0] + oz

    v_n0 = _dot3(vrx, vry, vrz, nx, ny, nz)
    if v_n0 > 0.0:
        return

    closing = -v_n0
    w_impact = _impact_weight(closing)

    off_a = _support_offset(rax, ray, raz, nx, ny, nz) if inv_mass_a > 0.0 else 0.0
    off_b = _support_offset(rbx, rby, rbz, nx, ny, nz) if inv_mass_b > 0.0 else 0.0
    support_off = off_a if off_a > off_b else off_b

    aligned_a = inv_mass_a > 0.0 and face_align_a >= FACE_ALIGN_THRESHOLD
    aligned_b = inv_mass_b > 0.0 and face_align_b >= FACE_ALIGN_THRESHOLD
    face_aligned = aligned_a or aligned_b
    best_align = 0.0
    if inv_mass_a > 0.0 and face_align_a > best_align:
        best_align = face_align_a
    if inv_mass_b > 0.0 and face_align_b > best_align:
        best_align = face_align_b

    face_support = face_aligned and support_off < UNSTABLE_SUPPORT_OFFSET
    unstable = support_off >= UNSTABLE_SUPPORT_OFFSET

    ground_like = fabs(ny) > 0.88
    if ground_like and closing < 2.5:
        if best_align >= FACE_REST_ALIGN and support_off < 0.08:
            face_support = 1
            unstable = 0
        elif support_off >= UNSTABLE_SUPPORT_OFFSET or best_align < FACE_REST_ALIGN:
            face_support = 0
            unstable = 1
            if support_off < 1e-3 and inv_mass_a > 0.0:
                edge_x = 1.0; edge_y = 0.0; edge_z = 0.0
                dn = _dot3(edge_x, edge_y, edge_z, nx, ny, nz)
                edge_x -= nx * dn; edge_y -= ny * dn; edge_z -= nz * dn
                if _len3(edge_x, edge_y, edge_z) < 1e-6:
                    edge_x = 0.0; edge_y = 0.0; edge_z = 1.0
                    dn = _dot3(edge_x, edge_y, edge_z, nx, ny, nz)
                    edge_x -= nx * dn; edge_y -= ny * dn; edge_z -= nz * dn
                el = _len3(edge_x, edge_y, edge_z)
                if el > 1e-6:
                    edge_x /= el; edge_y /= el; edge_z /= el
                    _cross3(nx, ny, nz, edge_x, edge_y, edge_z, &tip_x, &tip_y, &tip_z)
                    tl = _len3(tip_x, tip_y, tip_z)
                    if tl > 1e-6:
                        tip_x /= tl; tip_y /= tl; tip_z /= tl
                        rax += tip_x * 0.03
                        ray += tip_y * 0.03
                        raz += tip_z * 0.03
                        support_off = 0.03

    if face_support:
        w_n = 0.0 if closing < 1.5 else w_impact * 0.1
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
    dn = _dot3(rax, ray, raz, nx, ny, nz)
    ra_cx = nx * dn; ra_cy = ny * dn; ra_cz = nz * dn
    ra_tx = rax - ra_cx; ra_ty = ray - ra_cy; ra_tz = raz - ra_cz
    _clamp_vec3(&ra_tx, &ra_ty, &ra_tz, MAX_NORMAL_TANGENT_ARM)
    ra_nx = ra_cx + w_n * ra_tx
    ra_ny = ra_cy + w_n * ra_ty
    ra_nz = ra_cz + w_n * ra_tz

    dn = _dot3(rbx, rby, rbz, nx, ny, nz)
    rb_cx = nx * dn; rb_cy = ny * dn; rb_cz = nz * dn
    rb_tx = rbx - rb_cx; rb_ty = rby - rb_cy; rb_tz = rbz - rb_cz
    _clamp_vec3(&rb_tx, &rb_ty, &rb_tz, MAX_NORMAL_TANGENT_ARM)
    rb_nx = rb_cx + w_n * rb_tx
    rb_ny = rb_cy + w_n * rb_ty
    rb_nz = rb_cz + w_n * rb_tz

    _cross3(oax[0], oay[0], oaz[0], ra_nx, ra_ny, ra_nz, &ox, &oy, &oz)
    vrx = vax[0] + ox; vry = vay[0] + oy; vrz = vaz[0] + oz
    _cross3(obx[0], oby[0], obz[0], rb_nx, rb_ny, rb_nz, &ox, &oy, &oz)
    vrx -= vbx[0] + ox; vry -= vby[0] + oy; vrz -= vbz[0] + oz
    v_n = _dot3(vrx, vry, vrz, nx, ny, nz)
    if v_n > 0.0:
        out_unstable[0] = unstable
        return

    kn = _effective_mass(nx, ny, nz, inv_mass_a, inv_mass_b,
                         ra_nx, ra_ny, ra_nz, rb_nx, rb_ny, rb_nz,
                         has_ia, ia, has_ib, ib)
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
    jnx = nx * jn; jny = ny * jn; jnz = nz * jn
    _apply_impulse(vax, vay, vaz, oax, oay, oaz,
                   vbx, vby, vbz, obx, oby, obz,
                   inv_mass_a, inv_mass_b, has_ia, ia, has_ib, ib,
                   ra_nx, ra_ny, ra_nz, rb_nx, rb_ny, rb_nz,
                   jnx, jny, jnz)

    # Gravity tip torque
    if unstable and ground_like and closing < 3.0 and inv_mass_a > 0.0 and has_ia:
        if dt < 1e-5:
            dt = 1e-5
        if dt > 0.05:
            dt = 0.05
        mass = 1.0 / inv_mass_a
        sx = nx * (mass * GRAVITY * (ny if ny > 0.0 else 0.0))
        sy = ny * (mass * GRAVITY * (ny if ny > 0.0 else 0.0))
        sz = nz * (mass * GRAVITY * (ny if ny > 0.0 else 0.0))
        if _len3(sx, sy, sz) < 1e-8:
            sx = nx * (mass * GRAVITY)
            sy = ny * (mass * GRAVITY)
            sz = nz * (mass * GRAVITY)
        _cross3(rax, ray, raz, sx, sy, sz, &tau_x, &tau_y, &tau_z)
        tau_x *= dt; tau_y *= dt; tau_z *= dt
        _mat_mul_vec(ia, tau_x, tau_y, tau_z, &ox, &oy, &oz)
        oax[0] += ox; oay[0] += oy; oaz[0] += oz
        if _len3(oax[0], oay[0], oaz[0]) < 0.15:
            oax[0] += ox; oay[0] += oy; oaz[0] += oz

    if unstable and ground_like and closing < 3.0 and inv_mass_b > 0.0 and has_ib:
        if dt < 1e-5:
            dt = 1e-5
        if dt > 0.05:
            dt = 0.05
        mass = 1.0 / inv_mass_b
        sx = nx * (mass * GRAVITY * (ny if ny > 0.0 else 0.0))
        sy = ny * (mass * GRAVITY * (ny if ny > 0.0 else 0.0))
        sz = nz * (mass * GRAVITY * (ny if ny > 0.0 else 0.0))
        if _len3(sx, sy, sz) < 1e-8:
            sx = nx * (mass * GRAVITY)
            sy = ny * (mass * GRAVITY)
            sz = nz * (mass * GRAVITY)
        _cross3(rbx, rby, rbz, sx, sy, sz, &tau_x, &tau_y, &tau_z)
        tau_x *= dt; tau_y *= dt; tau_z *= dt
        _mat_mul_vec(ib, tau_x, tau_y, tau_z, &ox, &oy, &oz)
        obx[0] += ox; oby[0] += oy; obz[0] += oz

    # Friction
    if face_support:
        ra_fx = ra_nx; ra_fy = ra_ny; ra_fz = ra_nz
        rb_fx = rb_nx; rb_fy = rb_ny; rb_fz = rb_nz
        friction_angular = closing > 2.0
    elif unstable:
        ra_fx = rax; ra_fy = ray; ra_fz = raz
        rb_fx = rbx; rb_fy = rby; rb_fz = rbz
        friction_angular = 1
    else:
        ra_fx = rax * w_n + ra_nx * (1.0 - w_n)
        ra_fy = ray * w_n + ra_ny * (1.0 - w_n)
        ra_fz = raz * w_n + ra_nz * (1.0 - w_n)
        rb_fx = rbx * w_n + rb_nx * (1.0 - w_n)
        rb_fy = rby * w_n + rb_ny * (1.0 - w_n)
        rb_fz = rbz * w_n + rb_nz * (1.0 - w_n)
        friction_angular = w_n > 0.2

    _cross3(oax[0], oay[0], oaz[0], ra_fx, ra_fy, ra_fz, &ox, &oy, &oz)
    vrx = vax[0] + ox; vry = vay[0] + oy; vrz = vaz[0] + oz
    _cross3(obx[0], oby[0], obz[0], rb_fx, rb_fy, rb_fz, &ox, &oy, &oz)
    vrx -= vbx[0] + ox; vry -= vby[0] + oy; vrz -= vbz[0] + oz
    dn = _dot3(vrx, vry, vrz, nx, ny, nz)
    vtx = vrx - nx * dn; vty = vry - ny * dn; vtz = vrz - nz * dn
    t_mag = _len3(vtx, vty, vtz)
    mu_s = static_friction if static_friction > 0.0 else 0.0
    mu_d = dynamic_friction if dynamic_friction > 0.0 else 0.0

    if t_mag >= 1e-10:
        tx = vtx / t_mag; ty = vty / t_mag; tz = vtz / t_mag
        if (not friction_angular) or t_mag < RESTING_TANGENTIAL_SPEED:
            kt_lin = inv_mass_a + inv_mass_b
            if kt_lin > 1e-12:
                jt = -t_mag / kt_lin
                max_f = fabs(jn) * (mu_s if mu_s > mu_d else mu_d)
                if fabs(jt) > max_f:
                    jt = copysign(max_f, jt) if max_f > 0.0 else 0.0
                vax[0] += tx * jt * inv_mass_a
                vay[0] += ty * jt * inv_mass_a
                vaz[0] += tz * jt * inv_mass_a
                vbx[0] -= tx * jt * inv_mass_b
                vby[0] -= ty * jt * inv_mass_b
                vbz[0] -= tz * jt * inv_mass_b
                if face_support:
                    oax[0] *= 0.5; oay[0] *= 0.5; oaz[0] *= 0.5
                    obx[0] *= 0.5; oby[0] *= 0.5; obz[0] *= 0.5
        else:
            kt = _effective_mass(tx, ty, tz, inv_mass_a, inv_mass_b,
                                 ra_fx, ra_fy, ra_fz, rb_fx, rb_fy, rb_fz,
                                 has_ia, ia, has_ib, ib)
            if kt > 1e-12:
                vt_along = _dot3(vrx, vry, vrz, tx, ty, tz)
                jt = -vt_along / kt
                max_static = fabs(jn) * mu_s
                if fabs(jt) > max_static:
                    jt = copysign(fabs(jn) * mu_d, jt)
                if unstable:
                    jt *= 0.45
                if vt_along * (vt_along + jt * kt) < 0.0:
                    jt = -vt_along / kt
                _apply_impulse(vax, vay, vaz, oax, oay, oaz,
                               vbx, vby, vbz, obx, oby, obz,
                               inv_mass_a, inv_mass_b, has_ia, ia, has_ib, ib,
                               ra_fx, ra_fy, ra_fz, rb_fx, rb_fy, rb_fz,
                               tx * jt, ty * jt, tz * jt)

    # Face settle
    if face_support and closing < 2.0:
        if inv_mass_a > 0.0:
            oax[0] *= 0.25; oay[0] *= 0.25; oaz[0] *= 0.25
            if _len3(oax[0], oay[0], oaz[0]) < 0.5:
                oax[0] = 0.0; oay[0] = 0.0; oaz[0] = 0.0
            speed = _len3(vax[0], vay[0], vaz[0])
            if speed < 0.08:
                vax[0] = 0.0; vay[0] = 0.0; vaz[0] = 0.0
            elif mu_s > 1e-6 and speed < 0.35:
                vax[0] = 0.0; vay[0] = 0.0; vaz[0] = 0.0
        if inv_mass_b > 0.0:
            obx[0] *= 0.25; oby[0] *= 0.25; obz[0] *= 0.25
            if _len3(obx[0], oby[0], obz[0]) < 0.5:
                obx[0] = 0.0; oby[0] = 0.0; obz[0] = 0.0
    elif (not unstable) and w_n < 0.5:
        damp = 0.72 + 0.2 * w_n
        if inv_mass_b < 1e-12 and inv_mass_a > 0.0:
            oax[0] *= damp; oay[0] *= damp; oaz[0] *= damp
        elif inv_mass_a < 1e-12 and inv_mass_b > 0.0:
            obx[0] *= damp; oby[0] *= damp; obz[0] *= damp

    _clamp_omega3(oax, oay, oaz, MAX_ANGULAR_SPEED)
    _clamp_omega3(obx, oby, obz, MAX_ANGULAR_SPEED)
    out_unstable[0] = unstable


def resolve_contact_3d_fast(
    double[::1] pos_a, double[::1] vel_a, double[::1] omega_a,
    double inv_mass_a, object i_inv_a,
    double[::1] pos_b, double[::1] vel_b, double[::1] omega_b,
    double inv_mass_b, object i_inv_b,
    double[::1] contact_point, double[::1] normal,
    double restitution, double static_friction, double dynamic_friction,
    double face_align_a=0.0, double face_align_b=0.0,
    double dt=0.016666666666666666,
):
    """
    Fast contact solve. Returns (va, oa, vb, ob, unstable) as float64 arrays / bool.
    i_inv_* may be None or a contiguous (3,3) float64 matrix.
    """
    cdef double vax = vel_a[0], vay = vel_a[1], vaz = vel_a[2]
    cdef double oax = omega_a[0], oay = omega_a[1], oaz = omega_a[2]
    cdef double vbx = vel_b[0], vby = vel_b[1], vbz = vel_b[2]
    cdef double obx = omega_b[0], oby = omega_b[1], obz = omega_b[2]
    cdef bint has_ia = 0, has_ib = 0
    cdef bint unstable = 0
    cdef double[:, ::1] ia
    cdef double[:, ::1] ib
    cdef cnp.ndarray[cnp.float64_t, ndim=2] ia_arr
    cdef cnp.ndarray[cnp.float64_t, ndim=2] ib_arr
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out_va, out_oa, out_vb, out_ob

    # Dummy matrices when absent (never read if has_* is false)
    ia_arr = np.zeros((3, 3), dtype=np.float64)
    ib_arr = np.zeros((3, 3), dtype=np.float64)
    ia = ia_arr
    ib = ib_arr

    if i_inv_a is not None:
        ia_arr = np.ascontiguousarray(i_inv_a, dtype=np.float64)
        if ia_arr.shape[0] == 3 and ia_arr.shape[1] == 3:
            ia = ia_arr
            has_ia = 1
    if i_inv_b is not None:
        ib_arr = np.ascontiguousarray(i_inv_b, dtype=np.float64)
        if ib_arr.shape[0] == 3 and ib_arr.shape[1] == 3:
            ib = ib_arr
            has_ib = 1

    _resolve_contact_core(
        pos_a[0], pos_a[1], pos_a[2],
        &vax, &vay, &vaz, &oax, &oay, &oaz,
        inv_mass_a, has_ia, ia,
        pos_b[0], pos_b[1], pos_b[2],
        &vbx, &vby, &vbz, &obx, &oby, &obz,
        inv_mass_b, has_ib, ib,
        contact_point[0], contact_point[1], contact_point[2],
        normal[0], normal[1], normal[2],
        restitution, static_friction, dynamic_friction,
        face_align_a, face_align_b, dt, &unstable,
    )

    out_va = np.empty(3, dtype=np.float64)
    out_oa = np.empty(3, dtype=np.float64)
    out_vb = np.empty(3, dtype=np.float64)
    out_ob = np.empty(3, dtype=np.float64)
    out_va[0] = vax; out_va[1] = vay; out_va[2] = vaz
    out_oa[0] = oax; out_oa[1] = oay; out_oa[2] = oaz
    out_vb[0] = vbx; out_vb[1] = vby; out_vb[2] = vbz
    out_ob[0] = obx; out_ob[1] = oby; out_ob[2] = obz
    return out_va, out_oa, out_vb, out_ob, bool(unstable)


def face_align_from_matrix_fast(double[:, ::1] R, double[::1] n):
    """Max |dot| of body axes (rows and columns) with contact normal."""
    cdef double nx = n[0], ny = n[1], nz = n[2]
    cdef double nlen = _len3(nx, ny, nz)
    cdef double best = 0.0, d
    cdef int i
    if nlen < 1e-12:
        return 0.0
    nx /= nlen; ny /= nlen; nz /= nlen
    for i in range(3):
        d = fabs(R[0, i] * nx + R[1, i] * ny + R[2, i] * nz)
        if d > best:
            best = d
        d = fabs(R[i, 0] * nx + R[i, 1] * ny + R[i, 2] * nz)
        if d > best:
            best = d
    return best


def obb_support_feature_centroid_fast(
    double[::1] C, double[:, ::1] A, double[::1] E,
    double dx, double dy, double dz,
    double tol=-1.0,
):
    """Centroid of OBB support feature in direction d. Returns float64[3]."""
    cdef double dlen = _len3(dx, dy, dz)
    cdef double best = -1e300
    cdef double sx, sy, sz, vx, vy, vz, s
    cdef double sumx = 0.0, sumy = 0.0, sumz = 0.0
    cdef int count = 0
    cdef double emax, t
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out
    cdef int ix, iy, iz

    out = np.empty(3, dtype=np.float64)
    if dlen < 1e-12:
        out[0] = C[0]; out[1] = C[1]; out[2] = C[2]
        return out
    dx /= dlen; dy /= dlen; dz /= dlen

    emax = E[0]
    if E[1] > emax:
        emax = E[1]
    if E[2] > emax:
        emax = E[2]
    if tol < 0.0:
        t = 0.02 * emax
        if t < 1e-4:
            t = 1e-4
    else:
        t = tol

    for ix in range(2):
        sx = -1.0 if ix == 0 else 1.0
        for iy in range(2):
            sy = -1.0 if iy == 0 else 1.0
            for iz in range(2):
                sz = -1.0 if iz == 0 else 1.0
                # v = C + A @ (s * E)
                vx = C[0] + A[0, 0] * (sx * E[0]) + A[0, 1] * (sy * E[1]) + A[0, 2] * (sz * E[2])
                vy = C[1] + A[1, 0] * (sx * E[0]) + A[1, 1] * (sy * E[1]) + A[1, 2] * (sz * E[2])
                vz = C[2] + A[2, 0] * (sx * E[0]) + A[2, 1] * (sy * E[1]) + A[2, 2] * (sz * E[2])
                s = vx * dx + vy * dy + vz * dz
                if s > best + t:
                    best = s
                    sumx = vx; sumy = vy; sumz = vz
                    count = 1
                elif s >= best - t:
                    sumx += vx; sumy += vy; sumz += vz
                    count += 1

    if count <= 0:
        out[0] = C[0]; out[1] = C[1]; out[2] = C[2]
    else:
        out[0] = sumx / count
        out[1] = sumy / count
        out[2] = sumz / count
    return out
