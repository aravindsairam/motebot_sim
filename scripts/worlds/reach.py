"""Where the robot's centre can get to in a world, from its start pose, on a 2 cm grid.

Used by the layout tests and the plan picture (system Python only: needs numpy and scipy).
Blocking: every shape that collides and starts lower than 20 cm (the robot passes under anything
higher), and everything off the floor slab (the stairs are a drop).
"""

from dataclasses import dataclass

RES = 0.02
CLEAR_UNDER = 0.20


@dataclass
class Reach:
    x0: float
    y0: float
    res: float
    mask: object  # numpy bool array [iy, ix]: the robot's centre can be here

    def _index(self, x, y):
        return int((y - self.y0) / self.res), int((x - self.x0) / self.res)

    def at(self, point):
        iy, ix = self._index(*point)
        return bool(self.mask[iy, ix])

    def in_rect(self, rect):
        """Number of reachable cells whose centres lie in rect (x0, y0, x1, y1)."""
        import numpy as np

        ny, nx = self.mask.shape
        xs = self.x0 + (np.arange(nx) + 0.5) * self.res
        ys = self.y0 + (np.arange(ny) + 0.5) * self.res
        cols = (xs >= rect[0]) & (xs <= rect[2])
        rows = (ys >= rect[1]) & (ys <= rect[3])
        return int(self.mask[np.ix_(rows, cols)].sum())


def blocked_grid(world, res=RES, margin=0.2):
    """(x0, y0, blocked[iy, ix]) over the world's footprint plus a margin."""
    import numpy as np

    fx0, fy0, fx1, fy1 = world.FOOTPRINT
    x0, y0 = fx0 - margin, fy0 - margin
    nx = int(round((fx1 - fx0 + 2 * margin) / res))
    ny = int(round((fy1 - fy0 + 2 * margin) / res))
    xs = x0 + (np.arange(nx) + 0.5) * res
    ys = y0 + (np.arange(ny) + 0.5) * res
    gx, gy = np.meshgrid(xs, ys)
    blocked = ~((gx >= fx0) & (gx <= fx1) & (gy >= fy0) & (gy <= fy1))
    for s in world.walls() + world.furniture() + world.dock():
        if not s.collide or s.z0 >= CLEAR_UNDER or s.z1 <= 0.0:
            continue
        if hasattr(s, "radius"):  # a cylinder (told apart by shape, so any world's classes work)
            blocked |= (gx - s.x) ** 2 + (gy - s.y) ** 2 <= s.radius ** 2
        else:
            blocked |= (gx >= s.x0) & (gx <= s.x1) & (gy >= s.y0) & (gy <= s.y1)
    return x0, y0, blocked


def reachable(world, radius, res=RES):
    """Cells a disc of `radius` can reach from the start pose without touching anything."""
    from scipy import ndimage

    x0, y0, blocked = blocked_grid(world, res)
    # distance from a free cell's centre to the nearest blocked cell's centre, less half a cell:
    # about the distance to the obstacle's edge
    clearance = ndimage.distance_transform_edt(~blocked) * res - res / 2.0
    fits = clearance >= radius
    labels, _ = ndimage.label(fits)
    sx, sy = world.START_POSE_HOUSE[:2]
    start = labels[int((sy - y0) / res), int((sx - x0) / res)]
    if start == 0:
        raise ValueError(f"the robot does not fit at its start pose with radius {radius}")
    return Reach(x0, y0, res, labels == start)
