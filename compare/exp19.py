from fluid import FluidImage
import numpy as np
from common import read_csv, get_image_path
import matplotlib.pyplot as plt


def reorder(X, Y):
    r = sorted(range(len(X)), key=lambda ii: X[ii])
    X = [X[ii] for ii in r]
    Y = [Y[ii] for ii in r]
    return X, Y


def run_experiment():
    csv_path = "../exp19/clear.csv"
    images_path = "../exp19/images"
    data = read_csv(csv_path)

    crop = (442, 1220, 4260, 1440)

    name1 = "250_1.jpg"
    # path1 = get_image_path(name1, dir=images_path)
    # actual = data[name1]

    methods = {
        "max_height": {},
        "nheights_area": {"n": 10},
        "pixelcount_area": {},
        "convexhull_area": {"largest_only": False},
    }

    Y_orig = [data[n] for n in data]
    results = {}

    for t in range(150, 255, 5):
        print(t)
        Xs = {m: [] for m in list(methods)}

        for name in data:
            fluid = (
                FluidImage(get_image_path(name, dir=images_path))
                .crop(crop)
                .grayscale()
                .threshold(t)
            )
            for method in methods:
                Xs[method].append(fluid.method(method, **methods[method]))

        for method in methods:
            X, Y = reorder(Xs[method], Y_orig)
            slope0, intercept0 = np.polyfit(X, Y, 1)
            bestfit = lambda x: slope0 * x + intercept0
            Y_pred = [bestfit(x) for x in X]
            Y = np.array(Y)
            Y_pred = np.array(Y_pred)
            mse = ((Y - Y_pred) ** 2).mean(axis=0)
            max = ((Y - Y_pred) ** 2).max(axis=0)

            a = results.get(t, None)
            if a is None:
                results[t] = {}

            results[t][method] = {
                "bestfit": bestfit,
                "mse": mse,
                "max": max,
                "slope": slope0,
                "intercept": intercept0,
                "Y_pred": Y_pred,
                "X": X,
                "t": t,
                "Y": Y,
            }

    best = {
        "max_height": {"mse": {}, "max": {}},
        "nheights_area": {"mse": {}, "max": {}},
        "pixelcount_area": {"mse": {}, "max": {}},
        "convexhull_area": {"mse": {}, "max": {}},
    }

    for t in results:
        r = results[t]

        for method in best:
            mse = r[method]["mse"]
            max = r[method]["max"]
            if mse < best[method]["mse"].get("mse", float("inf")):
                best[method]["mse"] = r[method]
            if max < best[method]["max"].get("max", float("inf")):
                best[method]["max"] = r[method]

    plot(best)


def plot(best):
    plot1 = plt.subplot2grid((2, 2), (0, 0))
    plot2 = plt.subplot2grid((2, 2), (0, 1))
    plot3 = plt.subplot2grid((2, 2), (1, 0))
    plot4 = plt.subplot2grid((2, 2), (1, 1))

    plots = {
        "convexhull_area": plot1,
        "max_height": plot2,
        "nheights_area": plot3,
        "pixelcount_area": plot4,
    }

    for m in best:
        res = best[m]["max"]
        # print(res)
        t = res["t"]
        X = res["X"]
        Y = res["Y"]
        Y_pred = res["Y_pred"]
        # bestfit = res["bestfit"] # .predict(m, bestfit=bestfit, **methods[m])
        # X = [FluidImage(get_image_path(n, dir=images_path)).crop(crop).grayscale().threshold(t).method(m, **methods[m]) for n in data]
        # X, Y = reorder(r["X"], Y_orig)
        # Y_pred = [bestfit(d) for d in X]
        plot = plots[m]

        mse = res["mse"]
        max = res["max"]

        plot.plot(X, Y, label="actual")
        plot.plot(X, Y_pred, label=f"regression mse={mse:.1f} max={max:.1f}")

        plot.set_xlabel("X")
        plot.set_ylabel("Volums (uL)")

        plot.set_title(f"{m} (t={t})")
        plot.legend()

    plt.show()

    from pprint import pprint

    pprint(best)


def test_order():
    # csv_path = "../exp19/clear.csv"
    images_path = "../exp19/images"
    # data = read_csv(csv_path)
    crop = (442, 1220, 4260, 1440)

    name1 = "250_1.jpg"
    name2 = "300_1.jpg"

    fluid = (
        FluidImage(get_image_path(name1, dir=images_path))
        .crop(crop)
        .grayscale()
        .threshold(160)
        .show()
    )
    fluid.show(fluid.get_convexhull())
    fluid = (
        FluidImage(get_image_path(name2, dir=images_path))
        .crop(crop)
        .grayscale()
        .threshold(160)
        .show()
    )
    fluid.show(fluid.get_convexhull())


def main():
    run_experiment()
    # test_order()


if __name__ == "__main__":
    main()
