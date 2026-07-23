"""Shared impulse-solver tuning constants for 2D and 3D response.

Both ``engine.d2.physics.response`` and ``engine.d3.physics.response`` import
these so face-rest / impact-blend behaviour cannot silently diverge.
"""
from __future__ import annotations

# Closing speed (m/s) above which restitution (bounce) is applied fully.
RESTITUTION_THRESHOLD = 1.0
# Blend from settle → bounce over this speed range.
IMPACT_BLEND_START = 1.2
IMPACT_BLEND_END = 7.0

# |cos| between body face normal and contact normal → face support.
FACE_ALIGN_THRESHOLD = 0.82
# Must be this aligned with the floor before we freeze as "face rest".
FACE_REST_ALIGN = 0.985

# In-plane COM offset (m) that marks edge/vertex / off-center support.
UNSTABLE_SUPPORT_OFFSET = 0.06
# 2D face–face: tip only when COM projects near the edge of the support face.
FACE_TIP_OFFSET = 0.40

MAX_NORMAL_TANGENT_ARM = 0.35
RESTING_TANGENTIAL_SPEED = 0.08
MAX_ANGULAR_SPEED = 20.0
GRAVITY = 9.81

# Max contact points solved per 3D manifold.
MAX_MANIFOLD_POINTS = 4
