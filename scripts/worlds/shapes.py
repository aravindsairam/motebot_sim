"""What a world is made of, as plain data: shapes, rooms, openings and named things.

No Isaac Sim imports: read both by the USD builder (Isaac's Python 3.12) and by ROS-side tools
and tests (system Python 3.10). Lengths in metres, in the world's house frame (x east, y north).
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in the house frame."""

    name: str
    group: str  # Floor | Walls | Furniture | Doors | Dock
    x0: float
    y0: float
    x1: float
    y1: float
    z0: float
    z1: float
    material: str
    collide: bool = True
    floor_type: str = ""
    visible: bool = True


@dataclass(frozen=True)
class Cylinder:
    """Vertical cylinder in the house frame."""

    name: str
    group: str
    x: float
    y: float
    radius: float
    z0: float
    z1: float
    material: str
    collide: bool = True


@dataclass(frozen=True)
class Room:
    """A room: its inside outline, what kind of room it is, and its floor."""

    name: str
    kind: str  # living_room | kitchen | corridor | bedroom | bathroom
    rect: tuple  # (x0, y0, x1, y1), inside the walls
    floor_type: str  # hardwood | tile | carpet

    @property
    def area(self):
        x0, y0, x1, y1 = self.rect
        return (x1 - x0) * (y1 - y0)


@dataclass(frozen=True)
class Opening:
    """A gap in a wall that the robot can drive through."""

    name: str
    rect: tuple  # (x0, y0, x1, y1): the gap, through the whole thickness of the wall
    rooms: tuple  # the two places it joins ("outside" for the front door)
    door: str = ""  # name of the door Thing standing open beside it, if there is one


@dataclass(frozen=True)
class Thing:
    """A named object: its kind, its room, and the shapes it is made of.

    The shapes that collide are the ground truth for every test. A model laid over them later
    only decorates.
    """

    name: str
    kind: str
    room: str
    shapes: tuple


def footprint(shape):
    """(x0, y0, x1, y1) of a shape seen from above."""
    if isinstance(shape, Box):
        return (shape.x0, shape.y0, shape.x1, shape.y1)
    r = shape.radius
    return (shape.x - r, shape.y - r, shape.x + r, shape.y + r)


def overlaps_3d(a, b, eps=1e-9):
    """True if two shapes share volume (touching faces do not count)."""
    if not (a.z0 < b.z1 - eps and b.z0 < a.z1 - eps):
        return False
    if isinstance(a, Cylinder) and isinstance(b, Cylinder):
        return math.hypot(a.x - b.x, a.y - b.y) < a.radius + b.radius - eps
    if isinstance(a, Cylinder):
        a, b = b, a
    if isinstance(b, Cylinder):  # box a, cylinder b: nearest point of the box to the axis
        nx = min(max(b.x, a.x0), a.x1)
        ny = min(max(b.y, a.y0), a.y1)
        return math.hypot(b.x - nx, b.y - ny) < b.radius - eps
    fa, fb = footprint(a), footprint(b)
    return fa[0] < fb[2] - eps and fb[0] < fa[2] - eps and fa[1] < fb[3] - eps and fb[1] < fa[3] - eps


def ray_range(boxes, cyls, ox, oy, angle, max_range):
    """Distance from (ox, oy) along `angle` to the nearest box (x0, y0, x1, y1) or cylinder (x, y, r)."""
    dx, dy = math.cos(angle), math.sin(angle)
    best = max_range
    for x0, y0, x1, y1 in boxes:
        tmin, tmax = 0.0, best
        hit = True
        for o, d, lo, hi in ((ox, dx, x0, x1), (oy, dy, y0, y1)):
            if abs(d) < 1e-12:
                if o < lo or o > hi:
                    hit = False
                    break
            else:
                t1, t2 = (lo - o) / d, (hi - o) / d
                if t1 > t2:
                    t1, t2 = t2, t1
                tmin, tmax = max(tmin, t1), min(tmax, t2)
                if tmin > tmax:
                    hit = False
                    break
        if hit and tmin < best:
            best = tmin
    for cx, cy, r in cyls:
        fx, fy = ox - cx, oy - cy
        b = fx * dx + fy * dy
        c = fx * fx + fy * fy - r * r
        disc = b * b - c
        if disc >= 0.0:
            t = -b - math.sqrt(disc)
            if 0.0 < t < best:
                best = t
    return best


# --- builders for furniture made of several parts --------------------------------------------

def table(name, group, x0, y0, x1, y1, top_z, top_t, leg, inset, top_mat, leg_mat):
    """A top on four square legs. The robot can drive underneath, between the legs."""
    parts = [Box(f"{name}_top", group, x0, y0, x1, y1, top_z - top_t, top_z, top_mat)]
    corners = [
        (x0 + inset, y0 + inset),
        (x1 - inset - leg, y0 + inset),
        (x0 + inset, y1 - inset - leg),
        (x1 - inset - leg, y1 - inset - leg),
    ]
    for i, (lx, ly) in enumerate(corners):
        parts.append(Box(f"{name}_leg_{i}", group, lx, ly, lx + leg, ly + leg, 0.0, top_z - top_t, leg_mat))
    return parts


def chair(name, group, x0, y0, size, back, seat_z, back_top, leg, seat_mat, leg_mat):
    """A square seat on four legs with a backrest along one edge (back: 'n', 's', 'e' or 'w')."""
    x1, y1 = x0 + size, y0 + size
    parts = table(name, group, x0, y0, x1, y1, seat_z, 0.04, leg, 0.01, seat_mat, leg_mat)
    parts[0] = Box(f"{name}_seat", group, x0, y0, x1, y1, seat_z - 0.04, seat_z, seat_mat)
    t = 0.04
    bx0, by0, bx1, by1 = {
        "n": (x0, y1 - t, x1, y1),
        "s": (x0, y0, x1, y0 + t),
        "e": (x1 - t, y0, x1, y1),
        "w": (x0, y0, x0 + t, y1),
    }[back]
    parts.append(Box(f"{name}_back", group, bx0, by0, bx1, by1, seat_z, back_top, seat_mat))
    return parts


def seat_with_back(name, group, x0, y0, x1, y1, back, mat, seat_z=0.42, back_top=0.85, arms=True):
    """A sofa or armchair on a solid base, with a back along one edge and arms on the two sides."""
    parts = [Box(f"{name}_base", group, x0, y0, x1, y1, 0.0, seat_z, mat)]
    bt, at = 0.25, 0.20  # back and arm thickness
    if back == "s":
        parts.append(Box(f"{name}_back", group, x0, y0, x1, y0 + bt, seat_z, back_top, mat))
        arm_rects = [(x0, y0 + bt, x0 + at, y1), (x1 - at, y0 + bt, x1, y1)]
    elif back == "n":
        parts.append(Box(f"{name}_back", group, x0, y1 - bt, x1, y1, seat_z, back_top, mat))
        arm_rects = [(x0, y0, x0 + at, y1 - bt), (x1 - at, y0, x1, y1 - bt)]
    elif back == "w":
        parts.append(Box(f"{name}_back", group, x0, y0, x0 + bt, y1, seat_z, back_top, mat))
        arm_rects = [(x0 + bt, y0, x1, y0 + at), (x0 + bt, y1 - at, x1, y1)]
    else:
        parts.append(Box(f"{name}_back", group, x1 - bt, y0, x1, y1, seat_z, back_top, mat))
        arm_rects = [(x0, y0, x1 - bt, y0 + at), (x0, y1 - at, x1 - bt, y1)]
    if arms:
        for i, (ax0, ay0, ax1, ay1) in enumerate(arm_rects):
            parts.append(Box(f"{name}_arm_{i}", group, ax0, ay0, ax1, ay1, seat_z, seat_z + 0.20, mat))
    return parts
