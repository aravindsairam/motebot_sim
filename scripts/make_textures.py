"""Write the floor textures of the home world to sim/assets/textures/ (our own pictures, made here).

Run with system Python (numpy and Pillow): python3 sim/scripts/make_textures.py
Each picture repeats seamlessly and covers the floor size given in worlds/home.py (FLOOR_TEXTURES),
so a board or a tile has its true size on the floor. The same seed gives the same pictures.
"""

import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "assets", "textures"))


def smooth_noise(rng, size, cells):
    """Noise that repeats seamlessly: random values on a coarse grid, blended across each cell."""
    grid = rng.random((cells, cells))
    t = np.arange(size) * cells / size
    i0 = np.floor(t).astype(int) % cells
    i1 = (i0 + 1) % cells
    f = t - np.floor(t)
    f = f * f * (3 - 2 * f)
    rows = grid[i0][:, i0] * (1 - f)[None, :] + grid[i0][:, i1] * f[None, :]
    rows2 = grid[i1][:, i0] * (1 - f)[None, :] + grid[i1][:, i1] * f[None, :]
    return rows * (1 - f)[:, None] + rows2 * f[:, None]


def wood(size=1024, metres=1.8, board_w=0.18, board_l=1.2, seed=1):
    """Oak boards running along x: 10 boards across, joints staggered, grain along each board."""
    rng = np.random.default_rng(seed)
    px = size / metres
    y = np.arange(size)[:, None] / px
    x = np.arange(size)[None, :] / px
    board = np.floor(y / board_w).astype(int)
    shift = (board * 0.37 * board_l) % board_l
    piece = np.floor((x + shift) / board_l).astype(int)
    key = (board * 31 + piece * 7) % 97
    tone = 0.85 + 0.3 * ((key * 0.6180339) % 1.0)
    grain = 0.5 + 0.5 * np.sin((y * 2 * np.pi / board_w) * 9 + 6 * smooth_noise(rng, size, 16))
    fine = smooth_noise(rng, size, 128)
    base = np.array([0.62, 0.43, 0.26])
    img = base[None, None, :] * (tone * (0.86 + 0.1 * grain + 0.08 * fine))[:, :, None]
    gap_y = (y % board_w) < 0.0025
    gap_x = ((x + shift) % board_l) < 0.0025
    img[gap_y | gap_x] *= 0.55
    return img


def tiles(size=512, metres=1.2, tile=0.30, grout=0.004, base=(0.86, 0.86, 0.84), seed=2):
    """Square ceramic tiles with grout lines; a slight colour difference from tile to tile."""
    rng = np.random.default_rng(seed)
    px = size / metres
    y = np.arange(size)[:, None] / px
    x = np.arange(size)[None, :] / px
    n = int(round(metres / tile))
    shade = 0.95 + 0.07 * rng.random((n, n))
    tone = shade[np.floor(y / tile).astype(int) % n, np.floor(x / tile).astype(int) % n]
    img = np.array(base)[None, None, :] * (tone * (0.97 + 0.04 * smooth_noise(rng, size, 64)))[:, :, None]
    lines = ((y % tile) < grout) | ((x % tile) < grout)
    img[lines] = np.array([0.62, 0.61, 0.58])
    return img


def carpet(size=512, base=(0.47, 0.50, 0.57), seed=3):
    """A short-pile carpet: fine fibre noise over a gently mottled colour."""
    rng = np.random.default_rng(seed)
    fibres = rng.random((size, size))
    mottle = smooth_noise(rng, size, 8)
    k = 0.88 + 0.12 * mottle + 0.10 * (fibres - 0.5)
    return np.array(base)[None, None, :] * k[:, :, None]


def save(img, name):
    path = os.path.join(OUT, name)
    Image.fromarray((np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)).save(path, optimize=True)
    print(path)


def main():
    os.makedirs(OUT, exist_ok=True)
    save(wood(), "floor_wood.png")
    save(tiles(), "floor_tile.png")
    save(tiles(size=512, metres=0.6, tile=0.15, base=(0.80, 0.86, 0.88), seed=4), "floor_tile_bath.png")
    save(carpet(), "floor_carpet.png")
    save(carpet(base=(0.60, 0.55, 0.47), seed=5), "floor_carpet_beige.png")


if __name__ == "__main__":
    main()
