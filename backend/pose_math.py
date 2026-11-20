"""Pure pose geometry + movement-risk heuristic (unit-tested; mirrors the JS in index.html).

Landmarks from MediaPipe are NORMALIZED (x/width, y/height). Angles computed directly on
normalized values are distorted on non-square video, so every function here takes the
frame size and works in pixel space.
"""
import math

LEAN_LIMIT_DEG = 17.5      # heuristic thresholds, NOT learned from data - calibrate on real clips
SYMMETRY_FLOOR = 82.0
LEAN_ANOMALY_DEG = 20.0
SYMMETRY_ANOMALY = 76.0
BAD_REPS_ANOMALY = 2


def angle_2d(a, b, c, width=1.0, height=1.0):
    """Angle ABC in degrees (0..180) at vertex b. Points are (x, y) normalized coords."""
    v1 = ((a[0] - b[0]) * width, (a[1] - b[1]) * height)
    v2 = ((c[0] - b[0]) * width, (c[1] - b[1]) * height)
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        raise ValueError("degenerate angle: two points coincide")
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def torso_lean_deg(l_sh, r_sh, l_hip, r_hip, width=1.0, height=1.0):
    """Angle (deg) between the hip->shoulder midline and the vertical."""
    sx, sy = (l_sh[0] + r_sh[0]) / 2 * width, (l_sh[1] + r_sh[1]) / 2 * height
    hx, hy = (l_hip[0] + r_hip[0]) / 2 * width, (l_hip[1] + r_hip[1]) / 2 * height
    return math.degrees(math.atan2(abs(sx - hx), max(abs(hy - sy), 1e-6)))


def symmetry_score(left_angle, right_angle, slope=1.1):
    """100 = identical left/right joint angles; clamped to [50, 100]."""
    return max(50.0, min(100.0, 100.0 - abs(left_angle - right_angle) * slope))


def movement_risk(torso_lean, symmetry, bad_reps):
    """Heuristic 0..100 movement-risk score + human-readable flags. Not a trained model."""
    risk, flags = 0.0, []
    if torso_lean > LEAN_LIMIT_DEG:
        risk += (torso_lean - LEAN_LIMIT_DEG) * 2.8
        flags.append(f"torso lean {torso_lean:.1f} deg > {LEAN_LIMIT_DEG}")
    if symmetry < SYMMETRY_FLOOR:
        risk += (SYMMETRY_FLOOR - symmetry) * 1.9
        flags.append(f"left/right symmetry {symmetry:.1f}% < {SYMMETRY_FLOOR}")
    if bad_reps > 0:
        risk += min(45.0, bad_reps * 12.0)
        flags.append(f"{bad_reps} bad rep(s)")
    return round(min(100.0, risk), 1), flags


def movement_anomaly(torso_lean, symmetry, bad_reps):
    return bool(torso_lean > LEAN_ANOMALY_DEG or symmetry < SYMMETRY_ANOMALY
                or bad_reps >= BAD_REPS_ANOMALY)
