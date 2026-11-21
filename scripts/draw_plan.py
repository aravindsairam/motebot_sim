"""Draw a world's floor plan, with every room, opening, door and object and its name. No Isaac Sim needed.

Run: python3 sim/scripts/draw_plan.py [--world home] [--out sim/assets/previews/home_plan.png]
(system Python: needs matplotlib, numpy and scipy)
"""

import argparse
import importlib
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from worlds import reach  # noqa: E402
from worlds.shapes import Box, footprint  # noqa: E402

CLEAR_UNDER = 0.20
ROBOT_RADIUS = 0.171
FLOOR_NAMES = {"hardwood": "wood", "tile": "tiles", "carpet": "carpet"}


def _lighter(rgb, amount):
    return tuple(c + (1.0 - c) * amount for c in rgb)


def _rect_patch(rect, **kw):
    from matplotlib.patches import Rectangle

    x0, y0, x1, y1 = rect
    return Rectangle((x0, y0), x1 - x0, y1 - y0, **kw)


def _shape_patch(shape, **kw):
    from matplotlib.patches import Circle

    if isinstance(shape, Box):
        return _rect_patch(footprint(shape), **kw)
    return Circle((shape.x, shape.y), shape.radius, **kw)


def _room_label_spots(world):
    """For each room, the point inside it farthest from anything the robot bumps into."""
    import numpy as np
    from scipy import ndimage

    x0, y0, blocked = reach.blocked_grid(world)
    dist = ndimage.distance_transform_edt(~blocked)
    ny, nx = blocked.shape
    xs = x0 + (np.arange(nx) + 0.5) * reach.RES
    ys = y0 + (np.arange(ny) + 0.5) * reach.RES
    spots = {}
    for room in world.rooms():
        rx0, ry0, rx1, ry1 = room.rect
        cols = (xs >= rx0) & (xs <= rx1)
        rows = (ys >= ry0) & (ys <= ry1)
        sub = np.where(np.outer(rows, cols), dist, -1.0)
        iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
        spots[room.name] = (xs[ix], ys[iy])
    return spots


def _door_swing(ax, leaf, opening):
    """The quarter circle a door leaf sweeps, from where it stands open to where it would close."""
    from matplotlib.patches import Arc

    ox0, oy0, ox1, oy1 = opening.rect
    lx0, ly0, lx1, ly1 = footprint(leaf)
    if (ox1 - ox0) < (oy1 - oy0):  # wall along y: the open leaf runs along x
        hx = ox1 if abs(lx0 - ox1) < 1e-6 else ox0
        hy = oy0 if abs(ly0 - oy0) < 1e-6 else oy1
        free = (lx1 if hx == lx0 else lx0, hy)
        closed = (hx, oy1 if hy == oy0 else oy0)
    else:  # wall along x: the open leaf runs along y
        hy = oy1 if abs(ly0 - oy1) < 1e-6 else oy0
        hx = ox0 if abs(lx0 - ox0) < 1e-6 else ox1
        free = (hx, ly1 if hy == ly0 else ly0)
        closed = (ox1 if hx == ox0 else ox0, hy)
    a1 = math.degrees(math.atan2(free[1] - hy, free[0] - hx))
    a2 = math.degrees(math.atan2(closed[1] - hy, closed[0] - hx))
    if (a2 - a1) % 360.0 > 180.0:
        a1, a2 = a2, a1
    r = math.hypot(free[0] - hx, free[1] - hy)
    ax.add_patch(Arc((hx, hy), 2 * r, 2 * r, theta1=a1, theta2=a2, color="#8a6d3b", lw=0.8, ls=(0, (2, 2)), zorder=6))


def draw(world, out_path, title=None):
    """Draw the plan to out_path (PNG). Returns the list of names written on it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, FancyArrow, Patch

    mats = world.MATERIALS
    labels = []
    fx0, fy0, fx1, fy1 = world.FOOTPRINT
    sx0, sy0, sx1, sy1 = world.STAIRS
    left, right = fx0 - 0.4, fx1 + 0.4
    bottom, top = min(fy0, sy0) - 0.5, fy1 + 0.4
    fig, ax = plt.subplots(figsize=((right - left) * 1.75, (top - bottom) * 1.75 + 1.4))
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal")

    # floors
    for f in world.floors():
        if f.visible:
            ax.add_patch(_rect_patch(footprint(f), fc=_lighter(mats[f.material][0], 0.55), ec="none", zorder=1))
    ax.add_patch(_rect_patch(world.STAIRS, fc="#d9d4cc", ec="#555555", hatch="///", lw=0.6, zorder=1))
    ax.text((sx0 + sx1) / 2, (sy0 + sy1) / 2, f"stairs\n{world.STAIRS_DROP * 100:.0f} cm down",
            ha="center", va="center", fontsize=7, zorder=8,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

    # walls (the lintels above the openings are left out: from above, an opening is a gap)
    for w in world.walls():
        if w.name == "stairs_floor" or w.z0 >= world.DOOR_H - 1e-9:
            continue
        ax.add_patch(_rect_patch(footprint(w), fc="#3a3a3a", ec="#3a3a3a", lw=0.3, zorder=5))
    for g in (world.windows() if hasattr(world, "windows") else []):  # glass above the robot, in the walls
        ax.add_patch(_rect_patch(footprint(g), fc="#9fd0ea", ec="#9fd0ea", lw=1.5, zorder=5.5))
    for o in world.openings():
        x0, y0, x1, y1 = o.rect
        width = max(x1 - x0, y1 - y0)
        along_x = (x1 - x0) > (y1 - y0)
        tx, ty = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(tx, ty, f"{width:.1f} m", ha="center", va="center", fontsize=5.5, color="#3a3a3a", zorder=9,
                rotation=0 if along_x else 90,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="#3a3a3a", lw=0.4, alpha=0.9))

    things = world.things()
    openings_by_door = {o.door: o for o in world.openings() if o.door}

    # objects: flat or overhead parts first, then the parts the robot bumps into on top of them
    def layer(t, s):
        if not s.collide:
            return 0 if s.z1 < 0.05 else 1
        if s.z0 < CLEAR_UNDER - 1e-9:
            return 3
        # things that stand on another thing (the television) are drawn over it
        return 4 if all(p.z0 > 0.0 for p in t.shapes if p.collide) else 2

    for want in (0, 1, 2, 3, 4):
        for t in things:
            for s in t.shapes:
                if layer(t, s) != want:
                    continue
                rgb = mats[s.material][0]
                if want == 0:  # rugs, floor plates: flat, drawn only
                    kw = dict(fc=_lighter(rgb, 0.25), ec=rgb, lw=0.6, ls="--", alpha=0.55, zorder=2)
                elif want == 1:  # leaves, lamp shades: above the floor, drawn only
                    kw = dict(fc=_lighter(rgb, 0.3), ec=rgb, lw=0.6, ls="--", alpha=0.35, zorder=3)
                elif want in (2, 4):  # table tops, the raised bed: the robot passes underneath
                    kw = dict(fc=_lighter(rgb, 0.45), ec="#555555", lw=0.6, ls="--", alpha=0.75, zorder=4)
                else:  # what the robot bumps into, and the lidar sees
                    edge = "#222222" if sum(rgb) > 0.6 else "#888888"
                    kw = dict(fc=rgb, ec=edge, lw=0.6, zorder=6)
                ax.add_patch(_shape_patch(s, **kw))
            if want == 3 and t.kind == "door":
                _door_swing(ax, t.shapes[0], openings_by_door[t.name])

    # names of the objects
    def largest(t):  # a table's top, an L-shaped counter's longest run
        return max((footprint(s) for s in t.shapes), key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

    def contains(outer, inner):
        return outer[0] <= inner[0] and outer[1] <= inner[1] and inner[2] <= outer[2] and inner[3] <= outer[3]

    standing_on = [t for t in things if all(s.z0 > 0.0 for s in t.shapes if s.collide) and any(s.collide for s in t.shapes)]
    carriers = {t.name for t in things for u in standing_on if u is not t and contains(largest(t), largest(u))}
    for t in things:
        bx0, by0, bx1, by1 = largest(t)
        w, h = bx1 - bx0, by1 - by0
        rot = 90 if h > 1.6 * w and h > 0.5 else 0
        if t.kind == "rug":  # in a corner, so the name of what stands on it stays readable
            at, align = (bx0 + 0.05, by0 + 0.05), dict(ha="left", va="bottom")
        elif t in standing_on:  # beside it, clear of what it stands on
            at, align = (bx1 + 0.03, (by0 + by1) / 2), dict(ha="left", va="center")
        elif t.name in carriers:  # low down, clear of what stands on it
            at, align = ((bx0 + bx1) / 2, by0 + 0.04), dict(ha="center", va="bottom")
        else:
            at, align = ((bx0 + bx1) / 2, (by0 + by1) / 2), dict(ha="center", va="center")
        ax.text(*at, t.name, **align, fontsize=5.5, rotation=rot, zorder=10,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
        labels.append(t.name)

    # rooms: name, size and floor where there is most free floor
    for room in world.rooms():
        x, y = _room_label_spots(world)[room.name]
        ax.text(x, y, f"{room.name.replace('_', ' ').upper()}\n{room.area:.1f} m², {FLOOR_NAMES[room.floor_type]}",
                ha="center", va="center", fontsize=9, fontweight="bold", color="#1f3b57", zorder=11,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#1f3b57", lw=0.6, alpha=0.85))
        labels.append(room.name)

    # the robot at its start pose, facing into the room
    rx, ry, ryaw = world.START_POSE_HOUSE
    ax.add_patch(Circle((rx, ry), ROBOT_RADIUS, fc="#9fd3a8", ec="#1d6b2c", lw=1.0, zorder=12))
    ax.add_patch(FancyArrow(rx, ry, 0.13 * math.cos(ryaw), 0.13 * math.sin(ryaw), width=0.03,
                            head_width=0.09, head_length=0.06, color="#1d6b2c", zorder=13))
    ax.text(rx + 0.25, ry - 0.3, "robot at start\n(34 cm)", fontsize=6, color="#1d6b2c", zorder=13)

    total = sum(r.area for r in world.rooms())
    ax.set_title(title or f"{world.__name__.split('.')[-1]}: {len(world.rooms())} rooms, {total:.1f} m², "
                 f"{len(things)} objects (x east, y north, metres)", fontsize=12)
    ax.set_xticks(range(math.ceil(left), math.floor(right) + 1))
    ax.set_yticks(range(math.ceil(bottom), math.floor(top) + 1))
    ax.grid(True, color="#bbbbbb", lw=0.4, zorder=0)
    ax.tick_params(labelsize=8)
    legend = [
        Patch(fc="#6b5a48", ec="#222222", label="solid: the robot bumps into it, the lidar sees it"),
        Patch(fc="#d8cbbb", ec="#555555", ls="--", label="dashed: 20 cm or more up, the robot drives under it"),
        Patch(fc="#e6b8b0", ec="#9e3d35", ls="--", alpha=0.55, label="flat (rugs): drawn only"),
        Patch(fc="#3a3a3a", label="walls; the gaps are openings, with their width"),
        Patch(fc="#9fd0ea", label="windows (above the robot)"),
        Line2D([0], [0], color="#8a6d3b", ls=(0, (2, 2)), label="doors stand open (they do not move)"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--world", default="home", choices=["home"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    world = importlib.import_module(f"worlds.{args.world}")
    out = args.out or os.path.join(HERE, "..", "assets", "previews", f"{args.world}_plan.png")
    draw(world, out)
    print(os.path.normpath(out))


if __name__ == "__main__":
    main()
