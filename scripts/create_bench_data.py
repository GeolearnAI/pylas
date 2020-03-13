import sys
from pathlib import Path

import numpy as np

import pylas

CHUNK_SIZE = 50_000

MAX_NUM_PTS = 50 * 10 ** 6

POINT_INCREMENT = 10 ** 6

TIMES_TO_ADD = POINT_INCREMENT // CHUNK_SIZE


def main():
    las = pylas.read(sys.argv[1])

    if len(las.points) < CHUNK_SIZE:
        raise SystemExit(f"The file must have at least {CHUNK_SIZE} points")

    points_in_chunk = las.points[:CHUNK_SIZE]
    increment_points = points_in_chunk
    for _ in range(TIMES_TO_ADD):
        increment_points = np.append(increment_points)
    las = pylas.create(point_format_id=las.point_format.id, file_version=las.header.version)

    while len(las.points) <= MAX_NUM_PTS:
        las.points = np.append(las.points, increment_points)

        out_path = Path(sys.argv[2]) / f"{len(las.points)}.laz"
        las.write(str(out_path))


def main_2():
    las = pylas.read(sys.argv[1])

    if len(las.points) < POINT_INCREMENT:
        raise SystemExit(f"The file must have at least {POINT_INCREMENT} points")

    increment_points = las.points[:POINT_INCREMENT]
    las = pylas.create(point_format_id=las.point_format.id, file_version=las.header.version)

    for i in range(MAX_NUM_PTS // POINT_INCREMENT):
        print(i + 1, MAX_NUM_PTS // POINT_INCREMENT)
        las.points = np.append(las.points, increment_points)

        out_path = Path(sys.argv[2]) / f"{len(las.points)}.laz"
        las.write(str(out_path))

    while len(las.points) <= MAX_NUM_PTS:


if __name__ == '__main__':
    main_2()
