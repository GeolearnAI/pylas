from argparse import ArgumentParser
from pathlib import Path

import numpy as np

import pylas
from pylas.compression import LazBackend


def main(directory):
    all_las = list(Path(directory).glob("**/*.las"))
    for backend in [LazBackend.Lazperf, LazBackend.Lazrs, LazBackend.LazrsSingleThreaded]:
        for i, las_path in enumerate(all_las, start=1):
            print(i, "/", len(all_las), las_path)
            las = pylas.read(str(las_path))
            laz = pylas.lib.write_then_read_again(las, do_compress=True, laz_backends=[backend])

            for name in las.point_format.dtype.fields:
                if not np.all(las.points[name] == laz.points[name]):
                    print(f"\tDim '{name}' is not the same", name)


if __name__ == '__main__':
    parser = ArgumentParser(description="Runs the different LAZ backends on LAS file for compression"
                                        " and check the results by comparing with the corresponding LAS file")

    parser.add_argument("directory",
                        help="All LAS files in the directory or any of its sub directory will be used for"
                             "the  run")

    args = parser.parse_args()
    main(args.directory)
