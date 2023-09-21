import csv
import pathlib
import algorithms
from functools import partial
import numpy as np


def get_image_path(image_name, dir=None):
    return pathlib.Path(dir, image_name)


def read_csv(filename, every=None, choose=0):
    if every is None or every == 1:
        data = {}
        with open(filename, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data[row["filename"]] = float(row["actual"])
        return data
    else:
        data = {}
        i = 0
        with open(filename, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if i % every == choose:
                    data[row["filename"]] = float(row["actual"])
                i += 1
        return data


def squared_error(image_to_volume, actual, image_path):
    measured = image_to_volume(image_path)
    diff = actual - measured
    return diff * diff


def mse(image_to_volume, data, dir=None):
    total = 0
    for image_name, actual in data.items():
        total += squared_error(
            image_to_volume, actual, get_image_path(image_name, dir=dir)
        )
    return total / len(data)


def max_se(image_to_volume, data, dir=None):
    maxerr = float("-inf")
    for image_name, actual in data.items():
        err = squared_error(
            image_to_volume, actual, get_image_path(image_name, dir=dir)
        )
        if err > maxerr:
            maxerr = err
    return maxerr


def get_plot_data(image_to_independent, data, dir=None):
    X = []
    Y = []
    for image_name, actual in data.items():
        x = image_to_independent(get_image_path(image_name, dir=dir))
        y = actual
        X.append(x)
        Y.append(y)

    return X, Y


def find_lowest_error(err_func, tagged_image_to_volumes, data, dir=None):
    smallest_err = float("inf")
    smallest_tag = None

    for tag, image_to_volume in tagged_image_to_volumes:
        err = err_func(image_to_volume, data, dir=dir)
        if err < smallest_err:
            smallest_err = err
            smallest_tag = tag

    return smallest_tag, smallest_err


def find_with_lowest_error_named(
    named_tagged_image_to_volumes, err_func, data, show=False, dir=None
):
    results = []
    for name, tagged_image_to_volumes in named_tagged_image_to_volumes:
        tag, err = find_lowest_error(err_func, tagged_image_to_volumes, data, dir=dir)
        if show:
            print(f"Best tag for {name} t={tag} with lowest error: {err}")
        results.append((tag, err))

    return results


def find_best_fit(csv_path, dir_path):
    data = read_csv(csv_path)

    best_threshold_funcs_tagged = [
        (
            "pixelcount",
            [
                (t, partial(algorithms.pixelcount.image_to_volume, threshold=t))
                for t in range(0, 251, 5)
            ],
        ),
        (
            "convexhull",
            [
                (t, partial(algorithms.convexhull.image_to_volume, threshold=t))
                for t in range(0, 251, 5)
            ],
        ),
        (
            "maxheight",
            [
                (t, partial(algorithms.maxheight.image_to_volume, threshold=t))
                for t in range(0, 251, 5)
            ],
        ),
        (
            "nheights_area",
            [
                (
                    t,
                    partial(
                        algorithms.nheights_area.image_to_volume,
                        threshold=t,
                        nheights=10,
                    ),
                )
                for t in range(0, 251, 5)
            ],
        ),
    ]

    lowest_mse = find_with_lowest_error_named(
        best_threshold_funcs_tagged, mse, data, show=True, dir=dir_path
    )
    lowest_max_se = find_with_lowest_error_named(
        best_threshold_funcs_tagged, max_se, data, show=True, dir=dir_path
    )

    # lowest_mse = [(5, 2357.400676907035), (45, 976.6410854056369), (10, 223.6112424192071)]
    # lowest_max_se = [(5, 3444.9599472611912), (45, 2224.9159656172615), (10, 1086.101405055408)]

    pixelcount = lowest_mse[0], lowest_max_se[0]
    pixelcount = "pixelcount", pixelcount
    convexhull = lowest_mse[1], lowest_max_se[1]
    convexhull = "convexhull", convexhull
    maxheight = lowest_mse[2], lowest_max_se[2]
    maxheight = "maxheight", maxheight
    nheights_area = lowest_mse[3], lowest_max_se[3]
    nheights_area = "nheights_area", nheights_area

    pixelcount = (
        algorithms.pixelcount.image_to_independent,
        algorithms.pixelcount.image_to_volume,
    ), pixelcount
    convexhull = (
        algorithms.convexhull.image_to_independent,
        algorithms.convexhull.image_to_volume,
    ), convexhull
    maxheight = (
        algorithms.maxheight.image_to_independent,
        algorithms.maxheight.image_to_volume,
    ), maxheight
    nheights_area = (
        algorithms.nheights_area.image_to_independent,
        algorithms.nheights_area.image_to_volume,
    ), nheights_area

    stuff = (pixelcount, convexhull, maxheight, nheights_area)
    # stuff = (nheights,)

    for method, (name, bestres) in stuff:
        image_to_independent = method[0]
        image_to_volume = method[1]

        old_mse_run = bestres[0]
        old_max_se_run = bestres[1]

        X, Y = get_plot_data(
            partial(image_to_independent, threshold=old_mse_run[0]), data, dir=dir_path
        )
        reorder = sorted(range(len(X)), key=lambda ii: X[ii])
        X = [X[ii] for ii in reorder]
        Y = [Y[ii] for ii in reorder]
        slope0, intercept0 = np.polyfit(X, Y, 1)
        bestfit_mse = lambda x: slope0 * x + intercept0
        new_mse = mse(
            partial(image_to_volume, threshold=old_mse_run[0], bestfit=bestfit_mse),
            data,
            dir=dir_path,
        )

        X, Y = get_plot_data(
            partial(image_to_independent, threshold=old_max_se_run[0]),
            data,
            dir=dir_path,
        )
        reorder = sorted(range(len(X)), key=lambda ii: X[ii])
        X = [X[ii] for ii in reorder]
        Y = [Y[ii] for ii in reorder]
        slope1, intercept1 = np.polyfit(X, Y, 1)
        bestfit_max_se = lambda x: slope1 * x + intercept1
        new_max_se = max_se(
            partial(
                image_to_volume, threshold=old_max_se_run[0], bestfit=bestfit_max_se
            ),
            data,
            dir=dir_path,
        )

        print(f"For {name}:")
        print(f"    MSE    => vol(x) = {slope0} * x + {intercept0}")
        print(
            f"           => Threshold({old_mse_run[0]}) ... Old({old_mse_run[1]}), New({new_mse})"
        )
        print(f"    MAX_SE => vol(x) = {slope1} * x + {intercept1}")
        print(
            f"           => Threshold({old_max_se_run[0]}) ... Old({old_max_se_run[1]}), New({new_max_se})"
        )


def run_exp14():
    print(" ---------- Running Experiment 14 ---------- ")

    print("\n - White Slides")
    find_best_fit("../exp14/white.csv", "../exp14/images/White")

    print("\n - Black Slides")
    find_best_fit("../exp14/black.csv", "../exp14/images/Black")

    print("\n - Clear Slides")
    find_best_fit("../exp14/clear.csv", "../exp14/images/Unpainted")
    print()


def run_exp19():
    print(" ---------- Running Experiment 19 ---------- ")

    print("\n - Clear Slides")
    find_best_fit("../exp19/clear.csv", "../exp19/images")
    print()


def main():
    run_exp14()
    # run_exp19()


if __name__ == "__main__":
    main()
