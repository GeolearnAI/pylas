from argparse import ArgumentParser
from pathlib import Path

import numpy as np

import pylas
from pylas.compression import LazBackend


def main(directory):
    all_laz = list(Path(directory).glob("**/*.laz"))
    all_las = list(Path(directory).glob("**/*.las"))
    for backend in [LazBackend.Lazperf, LazBackend.LazrsSingleThreaded, LazBackend.Lazperf, LazBackend.Lazrs]:
        for i, (las_path, laz_path) in enumerate(zip(all_las, all_laz), start=1):
            # if las_path.name != "R1_F_0+200_0+250.las":
            #     continue
            print(i, "/", len(all_las), las_path, laz_path)
            assert las_path.stem == laz_path.stem

            las = pylas.read(str(las_path), laz_backends=[backend])
            laz = pylas.read(str(laz_path), laz_backends=[backend])

            for name in las.point_format.dtype.fields:
                if not np.all(las.points[name] == laz.points[name]):
                    print(f"\tDim '{name}' is not the same", name)


if __name__ == '__main__':
    parser = ArgumentParser(description="Runs the different LAZ backends on LAZ file for decompression"
                                        " and check the results by comparing with the corresponding LAS file")

    parser.add_argument("directory",
                        help="All LAS & LAZ files in the directory or any of its sub directory will be used for"
                             "the  run")

    args = parser.parse_args()
    main(args.directory)
