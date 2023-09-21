import matplotlib.pyplot as plt
from compare import get_plot_data, read_csv, get_image_path
import algorithms
from functools import partial
import numpy as np
from fluid import FluidImage


def plot(
    plot,
    csv_path,
    images_path,
    method,
    title,
    bestfit,
    threshold,
    xvar="Independent Variable",
    yvar="Volume (uL)",
):
    data = read_csv(csv_path)

    X, Y = get_plot_data(
        partial(method.image_to_independent, threshold=threshold), data, dir=images_path
    )
    reorder = sorted(range(len(X)), key=lambda ii: X[ii])
    X = [X[ii] for ii in reorder]
    Y_actual = [Y[ii] for ii in reorder]
    Y_pred_old = [
        method.image_to_volume(get_image_path(i, dir=images_path), threshold=threshold)
        for i in data
    ]
    Y_pred_new = [
        method.image_to_volume(
            get_image_path(i, dir=images_path), threshold=threshold, bestfit=bestfit
        )
        for i in data
    ]

    # square = lambda x: x * x
    yerr_reg = [2 * abs(y_p - y) for y_p, y in zip(Y_pred_new, Y)]
    yerr_man = [2 * abs(y_p - y) for y_p, y in zip(Y_pred_old, Y)]
    # par = np.polyfit(X, yerr, 2, full=True)

    # yerrUpper = [
    #     (bestfit(xx)) + (par[0][0] * xx**2 + par[0][1] * xx + par[0][2]) for xx in X
    # ]
    # yerrLower = [
    #     (bestfit(xx)) - (par[0][0] * xx**2 + par[0][1] * xx + par[0][2]) for xx in X
    # ]

    err_man_mse = np.square(np.subtract(Y_actual, Y_pred_old)).mean()
    err_reg_mse = np.square(np.subtract(Y_actual, Y_pred_new)).mean()

    err_man_max = np.square(np.subtract(Y_actual, Y_pred_old)).max()
    err_reg_max = np.square(np.subtract(Y_actual, Y_pred_new)).max()

    # err = np.std(Y)

    plot.plot(X, Y, label="actual")
    #  linestyle='None', marker='^',
    plot.errorbar(
        X,
        Y_pred_new,
        yerr=yerr_reg,
        label=f"regression mse={err_reg_mse:.1f} max={err_reg_max:.1f}",
    )  # , yerr=np.std(Y_pred_new)
    # plot.errorbar(
    #     X, Y_pred_old, yerr=yerr_man, label=f"manual mse={err_man_mse:.1f} max={err_man_max:.1f}"
    # )
    # plot.plot(X, yerrLower, ":r")
    # plot.plot(X, yerrUpper, ":r")

    plot.set_xlabel(xvar)
    plot.set_ylabel(yvar)

    plot.set_title(title)
    plot.legend()


def plot_white():
    csv_path = "../exp14/white.csv"
    images_path = "../exp14/images/White"

    plot1 = plt.subplot2grid((2, 2), (0, 0))
    plot2 = plt.subplot2grid((2, 2), (0, 1))
    plot3 = plt.subplot2grid((2, 2), (1, 0))
    plot4 = plt.subplot2grid((2, 2), (1, 1))

    bestfit = lambda x: 0.0006094190355479153 * x + 197.58793148807078
    plot(
        plot4,
        csv_path,
        images_path,
        algorithms.pixelcount,
        "pixelcount_area (t=15)",
        bestfit,
        threshold=15,
        xvar="White Pixel Count",
    )

    bestfit = lambda x: 0.0007030261326851475 * x + 190.1459324074357
    plot(
        plot1,
        csv_path,
        images_path,
        algorithms.convexhull,
        "convexhull_area (t=60)",
        bestfit,
        threshold=60,
        xvar="Area (pixels^2)",
    )

    bestfit = lambda x: 0.0011327907457026455 * x + 187.92509737957403
    plot(
        plot3,
        csv_path,
        images_path,
        algorithms.nheights_area,
        "nheights_area (t=65)",
        bestfit,
        threshold=65,
        xvar="Area (pixels^2)",
    )

    bestfit = lambda x: 1.2547689137474236 * x + 193.68862427632232
    plot(
        plot2,
        csv_path,
        images_path,
        algorithms.maxheight,
        f"max_height (t=20)",
        bestfit,
        threshold=20,
        xvar="Maximum Height (pixels)",
    )

    plt.show()


def plot_black():
    csv_path = "../exp14/black.csv"
    images_path = "../exp14/images/Black"

    plot1 = plt.subplot2grid((2, 2), (0, 0))
    plot2 = plt.subplot2grid((2, 2), (0, 1))
    plot3 = plt.subplot2grid((2, 2), (1, 0))
    plot4 = plt.subplot2grid((2, 2), (1, 1))

    bestfit = lambda x: -0.002693376935068661 * x + 557.0354821544047
    plot(
        plot1,
        csv_path,
        images_path,
        algorithms.pixelcount,
        "pixelcount",
        bestfit,
        threshold=5,
        xvar="White Pixel Count",
    )

    bestfit = lambda x: 0.0009356590614916368 * x + 151.77235096360286
    plot(
        plot2,
        csv_path,
        images_path,
        algorithms.convexhull,
        "convexhull",
        bestfit,
        threshold=45,
        xvar="Area (pixels^2)",
    )

    bestfit = lambda x: 4.821896217264802 * x + -29.819612027159007
    plot(
        plot3,
        csv_path,
        images_path,
        algorithms.maxheight,
        "maxheight1",
        bestfit,
        threshold=10,
        xvar="Maximum Height (pixels)",
    )

    # bestfit = lambda x: 4.821896217264778 * x + -24.997715809892387
    # plot(
    #     plot4,
    #     csv_path,
    #     images_path,
    #     algorithms.nheights,
    #     "maxheight2",
    #     bestfit,
    #     threshold=10,
    #     xvar="Maximum Height (pixels)",
    # )

    # plt.show()

    bestfit = lambda x: 0.0019282779820666218 * x + 126.03154895758811
    plot(
        plot4,
        csv_path,
        images_path,
        algorithms.nheights_area,
        "nheights_area",
        bestfit,
        threshold=50,
        xvar="Area (pixels^2)",
    )

    plt.show()


def plot_clear():
    csv_path = "../exp14/clear.csv"
    images_path = "../exp14/images/Unpainted"

    plot1 = plt.subplot2grid((2, 2), (0, 0))
    plot2 = plt.subplot2grid((2, 2), (0, 1))
    plot3 = plt.subplot2grid((2, 2), (1, 0))
    plot4 = plt.subplot2grid((2, 2), (1, 1))

    bestfit = lambda x: -0.0007888019740469404 * x + 344.14873844585526
    plot(
        plot1,
        csv_path,
        images_path,
        algorithms.pixelcount,
        "pixelcount",
        bestfit,
        threshold=15,
        xvar="White Pixel Count",
    )

    bestfit = lambda x: -0.0006203298425599916 * x + 322.65852843055046
    plot(
        plot2,
        csv_path,
        images_path,
        algorithms.convexhull,
        "convexhull",
        bestfit,
        threshold=35,
        xvar="Area (pixels^2)",
    )

    bestfit = lambda x: -1.5528704939919904 * x + 332.09934579439266
    plot(
        plot3,
        csv_path,
        images_path,
        algorithms.maxheight,
        "maxheight1",
        bestfit,
        threshold=25,
        xvar="Maximum Height (pixels)",
    )

    # bestfit = lambda x: -1.5528704939919917 * x + 330.54647530040074
    # plot(
    #     plot4,
    #     csv_path,
    #     images_path,
    #     algorithms.nheights,
    #     "maxheight2",
    #     bestfit,
    #     threshold=25,
    #     xvar="Maximum Height (pixels)",
    # )

    # plt.show()

    bestfit = lambda x: -0.0013572412280124164 * x + 345.28522834549267
    plot(
        plot4,
        csv_path,
        images_path,
        algorithms.nheights_area,
        "nheights_area",
        bestfit,
        threshold=60,
        xvar="Area (pixels^2)",
    )

    plt.show()


def test_hull():
    # csv_path = "../exp14/clear.csv"
    images_path = "../exp14/images/Unpainted"
    # data = read_csv(csv_path)

    img_name = "test.jpg"
    fluid = (
        FluidImage(get_image_path(img_name, dir=images_path))
        .grayscale()
        .threshold(100, blur=False)
    )
    fluid.show()

    hull = fluid.get_convexhull()
    fluid.show(hull)


def polygon_area(x, y):
    correction = x[-1] * y[0] - y[-1] * x[0]
    main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
    return 0.5 * np.abs(main_area + correction)


def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def test_nheights():
    csv_path = "../exp14/white.csv"
    images_path = "../exp14/images/White"
    data = read_csv(csv_path)

    # img_name = "250uL_1.jpg"
    for img_name in data:
        img_path = get_image_path(img_name, dir=images_path)

        nheights = 10
        # algorithms.nheights.image_to_independent(img_path, threshold=55, nheights=nheights)
        print(img_name)

        print(
            algorithms.nheights_area.image_to_independent(
                img_path, threshold=55, nheights=nheights
            )
        )
        print(algorithms.pixelcount.image_to_independent(img_path, threshold=55))
        print(algorithms.convexhull.image_to_independent(img_path, threshold=55))

        # print(max(algorithms.nheights.image_to_independent(img_path, threshold=55, nheights=200)))
        # print(algorithms.maxheight.image_to_independent(img_path, threshold=55))

        print()


def test_all():
    csv_path = "../exp14/clear.csv"
    images_path = "../exp14/images/Unpainted"
    data = read_csv(csv_path)

    for img_name in data:
        img_path = get_image_path(img_name, dir=images_path)
        fluid = FluidImage(img_path).crop().grayscale().threshold(55)

        # convexhull_area = fluid.convexhull_area(largest_only=True)
        # pixelcount_area = fluid.pixelcount_area()
        # nheights_area = fluid.nheights_area(10)
        convexhull_area = fluid.method("convexhull_area", largest_only=True)
        pixelcount_area = fluid.method("pixelcount_area")
        nheights_area = fluid.method("nheights_area", 10)

        # maxheight1 = fluid.max_height()
        # maxheight2 = max(fluid.nheights(10))
        maxheight1 = fluid.method("max_height")
        maxheight2 = max(fluid.method("nheights", 10))

        print(convexhull_area, pixelcount_area, nheights_area)
        print(maxheight1, maxheight2)
        print(
            data[img_name],
            fluid.predict(
                "max_height", lambda x: -1.5528704939919904 * x + 332.09934579439266
            ),
        )
        print()

        # fluid.show(fluid.get_convexhull(largest_only=True))


def test_copy():
    images_path = "../exp14/images/Unpainted"
    a = FluidImage(images_path + "/" + "250uL_1.jpg")
    b = a.copy()
    c = FluidImage(b)
    a.grayscale()
    b.crop()

    a.show()
    b.show()
    c.show()
    a.copy().show()


def main():
    # plot_white()
    # plot_black()
    # plot_clear()
    # test_hull()
    # test_nheights()
    # test_all()
    test_copy()


if __name__ == "__main__":
    main()
