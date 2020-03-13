import io
import logging
import time
from pathlib import Path
from argparse import ArgumentParser

import matplotlib.pyplot as plt

import pylas
from pylas.compression import LazBackend

logging.basicConfig(level=logging.DEBUG)

LOAD_IN_MEM = True


def main(laz_dir):
    backends_to_use = [LazBackend.Lazperf, LazBackend.LazrsSingleThreaded, LazBackend.Lazrs]
    all_things = {}
    all_files = list(Path(laz_dir).glob("**/*.laz"))
    for backend in backends_to_use:
        print(f"Using backend: {backend}")
        all_points = []
        for i, path in enumerate(all_files, start=1):
            print(f"{i} / {len(all_files)}")
            if LOAD_IN_MEM:
                with open(str(path), mode="rb") as file:
                    las_source = io.BytesIO(file.read())
            else:
                las_source = open(str(path), mode="rb")

            start_time = time.time()
            las = pylas.read(las_source, laz_backends=[backend])
            time_it_took = time.time() - start_time
            all_points.append((las.header.point_count, time_it_took))

        all_points.sort(key=lambda item: item[0])
        x = [item[0] // 10 ** 6 for item in all_points]
        y = [item[1] for item in all_points]

        all_things[str(backend)] = (x, y)

    fig, ax = plt.subplots()
    ax.set_xlabel('number of points (Millions)')
    ax.set_ylabel('time to read (secs)')  # Add
    ax.set_title("LAZ reading timings")  # Add a title to the axes.
    for name, (x, y) in all_things.items():
        ax.plot(x, y, marker='o', linestyle='--', label=name)
    ax.legend()  # Add a legend.
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser(description="Runs the different LAZ backends on LAZ file for decompression "
                                        "and plots the results")

    parser.add_argument("laz_dir", help="All LAZ files in the directory or any of its sub directory will be used for"
                                        "the  run")

    args = parser.parse_args()
    main(args.laz_dir)
