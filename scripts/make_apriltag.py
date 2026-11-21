#!/usr/bin/env python3
"""Write the image of the dock's AprilTag: sim/assets/textures/apriltag_36h11_<id>.png.

Runs with the system Python (needs OpenCV), not Isaac Sim's:
    python3 sim/scripts/make_apriltag.py

The image is the tag as it would be printed: the black square with its code, plus the white
border one cell wide that the detector needs around it. The size the detector is told
(house_layout.DOCK_TAG_SIZE) is the edge of the black square, which is 8/10 of the image.
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import house_layout as hl  # noqa: E402

CELL_PX = 100  # large and sharp: the renderer scales it down, never up


def main():
    if hl.DOCK_TAG_FAMILY != "36h11":
        sys.exit(f"only the 36h11 family is supported here, not {hl.DOCK_TAG_FAMILY}")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    square = cv2.aruco.generateImageMarker(dictionary, hl.DOCK_TAG_ID, 8 * CELL_PX)  # black border + 6x6 code
    image = np.full((10 * CELL_PX, 10 * CELL_PX), 255, np.uint8)
    image[CELL_PX:-CELL_PX, CELL_PX:-CELL_PX] = square
    out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", hl.DOCK_TAG_TEXTURE))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    print(f"wrote {out} ({image.shape[1]}x{image.shape[0]} px, tag36h11 id {hl.DOCK_TAG_ID})")


if __name__ == "__main__":
    main()
