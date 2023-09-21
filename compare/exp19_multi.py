from fluid import FluidImage
import numpy as np
from common import read_csv, get_image_path
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.pipeline import Pipeline


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
    D = 1
    name1 = "250_1.jpg"
    # path1 = get_image_path(name1, dir=images_path)
    # actual = data[name1]

    Y_orig = [data[n] for n in data]
    results = {}

    N = list(range(2, 22, 2))

    for t in range(150, 255, 5):
        print(t)
        Xs = {n: [] for n in N}

        for name in data:
            fluid = (
                FluidImage(get_image_path(name, dir=images_path))
                .crop(crop)
                .grayscale()
                .threshold(t)
            )
            for n in N:
                Xs[n].append(fluid.nheights(n))

        for n in N:
            X, Y = Xs[n], Y_orig
            model = Pipeline(
                [
                    ("poly", PolynomialFeatures(degree=D)),
                    ("linear", linear_model.LinearRegression()),
                ]
            )
            model = model.fit(X, Y)

            Y_pred = model.predict(X)
            # print(Y_pred)

            Y = np.array(Y)
            Y_pred = np.array(Y_pred)
            mse = ((Y - Y_pred) ** 2).mean(axis=0)
            max = ((Y - Y_pred) ** 2).max(axis=0)

            a = results.get(t, None)
            if a is None:
                results[t] = {}

            results[t][n] = {
                "bestfit": model,
                "mse": mse,
                "max": max,
                "slope": None,
                "intercept": None,
                "Y_pred": Y_pred,
                "X": X,
                "t": t,
                "Y": Y,
                "n": n,
            }

    best = {"mse": {}, "max": {}}

    for t in results:
        r = results[t]

        for n in N:
            mse = r[n]["mse"]
            max = r[n]["max"]
            if mse < best["mse"].get("mse", float("inf")):
                best["mse"] = r[n]
            if max < best["max"].get("max", float("inf")):
                best["max"] = r[n]

    plot(best)


def plot(best):
    plot = plt.subplot2grid((1, 1), (0, 0))

    res = best["max"]
    # print(res)
    t = res["t"]
    n = res["n"]
    X = res["X"]
    Y = res["Y"]
    Y_pred = res["Y_pred"]
    # bestfit = res["bestfit"] # .predict(m, bestfit=bestfit, **methods[m])
    # X = [FluidImage(get_image_path(n, dir=images_path)).crop(crop).grayscale().threshold(t).method(m, **methods[m]) for n in data]
    # X, Y = reorder(r["X"], Y_orig)
    # Y_pred = [bestfit(d) for d in X]

    mse = res["mse"]
    max = res["max"]

    plot.plot(X, Y, label="actual")
    plot.plot(X, Y_pred, label=f"regression mse={mse:.1f} max={max:.1f}")

    plot.set_xlabel("X")
    plot.set_ylabel("Volums (uL)")

    plot.set_title(f"nheights (t={t}, n={n})")
    plot.legend()

    plt.show()

    from pprint import pprint

    model = res["bestfit"]
    pprint(best)
    x = (
        FluidImage(get_image_path("300_3_test.jpg", dir="../exp19/images"))
        .crop((442, 1220, 4260, 1440))
        .grayscale()
        .threshold(t)
        .nheights(n)
    )
    print(
        read_csv("../exp19/clear.csv")["300_3.jpg"],
        model.predict(np.array(x).reshape(1, -1)),
    )
    print(model.named_steps["linear"].coef_)


def main():
    run_experiment()


if __name__ == "__main__":
    main()
