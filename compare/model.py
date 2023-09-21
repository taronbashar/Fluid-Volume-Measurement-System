import numpy as np
from fluid import FluidImage


class Experiment:
    def __init__(self, measurements):
        self.measurements = measurements
        self.crop = None
        self.threshold = None
        self.convexhull_area_largest_only = False
        self.nheights_n = None

    def set_crop(self, crop):
        self.crop = crop
        return self

    def set_threshold(self, threshold):
        self.threshold = threshold
        return self

    def setup_convexhull_area(self, largest_only=False):
        self.convexhull_area_largest_only = largest_only
        return self

    def setup_nheights(self, n):
        self.nheights_n = n
        return self

    def validate_parameters(self):
        if self.threshold is None:
            raise ValueError("Must give value or values for threshold (set_threshold)")
        if self.nheights_n is None:
            raise ValueError("Must setup n for nheights (setup_nheights)")

    @staticmethod
    def is_iterable(item):
        try:
            iter(item)
            return True
        except:
            return False

    def run(self):
        self.validate_parameters()
        is_crop_iter = (
            False
            if self.crop is None
            else False
            if type(self.crop) is tuple and len(self.crop) == 4
            else Experiment.is_iterable(self.crop)
        )
        is_threshold_iter = Experiment.is_iterable(self.threshold)
        is_nheights_n_iter = Experiment.is_iterable(self.nheights_n)

        if not is_crop_iter:
            self.crop = [self.crop]
        if not is_threshold_iter:
            self.threshold = [self.threshold]
        if not is_nheights_n_iter:
            self.nheights_n = [self.nheights_n]

        result = ExperimentResult()

        for img, actual in self.measurements:
            image = FluidImage(img).grayscale()
            result.add_actual(actual)

            for c in self.crop:
                cropped = image.copy().crop(c)
                for t in self.threshold:
                    thresholded = cropped.copy().threshold(t)
                    convexhull_area = thresholded.convexhull_area(
                        largest_only=self.convexhull_area_largest_only
                    )
                    pixelcount_area = thresholded.pixelcount_area()
                    max_height = thresholded.max_height()
                    result.add_convexhull_area(c, t, convexhull_area)
                    result.add_pixelcount_area(c, t, pixelcount_area)
                    result.add_max_height(c, t, max_height)

                    for n in self.nheights_n:
                        nheights, spacing = thresholded.nheights(n, space=True)
                        nheights_area = FluidImage.polynomial_area(nheights, spacing)

                        result.add_nheights(c, t, n, nheights)
                        result.add_nheights_area(c, t, n, nheights_area)

        return result


class ExperimentResult:
    mse = lambda y, y_pred: ((np.array(y) - np.array(y_pred)) ** 2).mean(axis=0)
    max_se = lambda y, y_pred: ((np.array(y) - np.array(y_pred)) ** 2).max(axis=0)

    def __init__(self):
        self.actual = []
        self.convexhull_area_data = {}
        self.pixelcount_area_data = {}
        self.max_height_data = {}
        self.nheights_area_data = {}
        self.nheights_data = {}

    def add_actual(self, actual):
        self.actual.append(actual)

    def add_convexhull_area(self, crop, threshold, area):
        if self.convexhull_area_data.get(crop, None) is None:
            self.convexhull_area_data[crop] = {}

        if self.convexhull_area_data[crop].get(threshold, None) is None:
            self.convexhull_area_data[crop][threshold] = []

        self.convexhull_area_data[crop][threshold].append(area)

    def add_pixelcount_area(self, crop, threshold, area):
        if self.pixelcount_area_data.get(crop, None) is None:
            self.pixelcount_area_data[crop] = {}

        if self.pixelcount_area_data[crop].get(threshold, None) is None:
            self.pixelcount_area_data[crop][threshold] = []

        self.pixelcount_area_data[crop][threshold].append(area)

    def add_max_height(self, crop, threshold, height):
        if self.max_height_data.get(crop, None) is None:
            self.max_height_data[crop] = {}

        if self.max_height_data[crop].get(threshold, None) is None:
            self.max_height_data[crop][threshold] = []

        self.max_height_data[crop][threshold].append(height)

    def add_nheights_area(self, crop, threshold, n, area):
        if self.nheights_area_data.get(crop, None) is None:
            self.nheights_area_data[crop] = {}

        if self.nheights_area_data[crop].get(threshold, None) is None:
            self.nheights_area_data[crop][threshold] = {}

        if self.nheights_area_data[crop][threshold].get(n, None) is None:
            self.nheights_area_data[crop][threshold][n] = []

        self.nheights_area_data[crop][threshold][n].append(area)

    def add_nheights(self, crop, threshold, n, heights):
        if self.nheights_data.get(crop, None) is None:
            self.nheights_data[crop] = {}

        if self.nheights_data[crop].get(threshold, None) is None:
            self.nheights_data[crop][threshold] = {}

        if self.nheights_data[crop][threshold].get(n, None) is None:
            self.nheights_data[crop][threshold][n] = []

        self.nheights_data[crop][threshold][n].append(heights)

    @staticmethod
    def reorder(X, Y):
        r = sorted(range(len(X)), key=lambda ii: X[ii])
        X = [X[ii] for ii in r]
        Y = [Y[ii] for ii in r]
        return X, Y

    def find_best_polyfit(self, data, error=None):
        if error is None:
            error = ExperimentResult.mse

        lowest_e = {"error": float("inf")}

        for c in data:
            for t in data[c]:
                results = data[c][t]
                
                X, Y = ExperimentResult.reorder(results, self.actual)
                if all([x == 0 for x in X]):
                    continue
                slope0, intercept0 = np.polyfit(X, Y, 1)
                model = lambda x: slope0 * x + intercept0
                Y_pred = [model(x) for x in X]

                e = error(Y, Y_pred)
                if e < lowest_e["error"]:
                    lowest_e = {
                        "c": c,
                        "t": t,
                        "X": X,
                        "Y": Y,
                        "Y_pred": Y_pred,
                        "model": model,
                        "model_slope": slope0,
                        "model_intercept": intercept0,
                        "error": e,
                    }

        return lowest_e

    def find_best_convexhull_area(self, error=None):
        r = self.find_best_polyfit(self.convexhull_area_data, error=error)
        r["method"] = "convexhull_area"
        return r

    def find_best_pixelcount_area(self, error=None):
        r = self.find_best_polyfit(self.pixelcount_area_data, error=error)
        r["method"] = "pixelcount_area"
        return r

    def find_best_max_height(self, error=None):
        r = self.find_best_polyfit(self.max_height_data, error=error)
        r["method"] = "max_height"
        return r

    def find_best_nheights_area(self, error=None):
        if error is None:
            error = ExperimentResult.mse

        lowest_e = {"error": float("inf")}

        for c in self.nheights_area_data:
            for t in self.nheights_area_data[c]:
                for n in self.nheights_area_data[c][t]:
                    results = self.nheights_area_data[c][t][n]

                    X, Y = ExperimentResult.reorder(results, self.actual)
                    if all([x == 0 for x in X]):
                        continue
                    slope0, intercept0 = np.polyfit(X, Y, 1)
                    model = lambda x: slope0 * x + intercept0
                    Y_pred = [model(x) for x in X]

                    e = error(Y, Y_pred)
                    if e < lowest_e["error"]:
                        lowest_e = {
                            "method": "nheights_area",
                            "c": c,
                            "t": t,
                            "n": n,
                            "X": X,
                            "Y": Y,
                            "Y_pred": Y_pred,
                            "model": model,
                            "model_slope": slope0,
                            "model_intercept": intercept0,
                            "error": e,
                        }

        return lowest_e

    def find_best_nheights(self, error=None, regression=None, degree=1):
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn import linear_model
        from sklearn.pipeline import Pipeline

        if error is None:
            error = ExperimentResult.mse

        if regression is None:
            regression = linear_model.LinearRegression()

        lowest_e = {"error": float("inf")}

        for c in self.nheights_data:
            for t in self.nheights_data[c]:
                for n in self.nheights_data[c][t]:
                    results = self.nheights_data[c][t][n]

                    X, Y = ExperimentResult.reorder(results, self.actual)

                    model = Pipeline(
                        [
                            ("poly", PolynomialFeatures(degree=degree)),
                            ("linear", regression),
                        ]
                    )
                    model = model.fit(X, Y)
                    single_model = lambda xs: model.predict([xs])[0]

                    Y_pred = model.predict(X)
                    # Y_pred = [single_model(x) for x in X]

                    Y = np.array(Y)
                    Y_pred = np.array(Y_pred)

                    e = error(Y, Y_pred)
                    if e < lowest_e["error"]:
                        lowest_e = {
                            "method": "nheights",
                            "c": c,
                            "t": t,
                            "n": n,
                            "X": X,
                            "Y": Y,
                            "Y_pred": Y_pred,
                            "model": single_model,
                            "raw_model": model,
                            "error": e,
                        }

        return lowest_e

    def find_best_all(self, error=None):
        res = [
            self.find_best_convexhull_area(error=error),
            self.find_best_pixelcount_area(error=error),
            self.find_best_max_height(error=error),
            self.find_best_nheights_area(error=error),
            self.find_best_nheights(error=error),
        ]
        return min(res, key=lambda r: r["error"])
    
    def find_best_singlevar(self, error=None):
        res = [
            self.find_best_convexhull_area(error=error),
            self.find_best_pixelcount_area(error=error),
            self.find_best_max_height(error=error),
            self.find_best_nheights_area(error=error),
        ]
        return min(res, key=lambda r: r["error"])

import matplotlib.pyplot as plt

def plot(best, errbars=True, xlab=None):
    
    from sklearn.metrics import r2_score
    
    plot = plt.subplot2grid((1, 1), (0, 0))
    
    X = best["X"]
    Y = best["Y"]
    print(Y)
    Y_pred = best["Y_pred"]
    r2 = r2_score(Y, Y_pred)
    error = best["error"]
    errs = 10
    mse = ExperimentResult.mse(Y, Y_pred)
    if errbars:
        plot.errorbar(X, Y, yerr=errs, linestyle='None', marker='.', label="Actual (Measured) Volume")
    else:
        plot.plot(X, Y, linestyle='None', marker='.', label="Actual (Measured) Volume")
    #  linestyle='None', marker='^',
    plot.plot(
        X,
        Y_pred,
        label=f"Regression Model MSE({mse:.1f}) r2({r2:.1f})",
    )
      # , yerr=np.std(Y_pred_new)
    # plot.errorbar(
    #     X, Y_pred_old, yerr=yerr_man, label=f"manual mse={err_man_mse:.1f} max={err_man_max:.1f}"
    # )
    # plot.plot(X, yerrLower, ":r")
    # plot.plot(X, yerrUpper, ":r")

    xlab = xlab if xlab else "X"
    plot.set_xlabel(xlab)
    plot.set_ylabel("Volume (uL)")

    c = best["c"]
    t = best["t"]
    n = best.get("n", None)

    title = f"Generated Model. Method({best['method']}) with threshold({t})" + (f" and N({n})" if n else "")
    plot.set_title(title)
    plot.legend()


def run_exp26(csv_path):
   
    from common import get_image_path, read_csv
    
    images_path = "../exp26/images"
    data = read_csv(csv_path)
    # images_path = "../exp19/images"
    # data = read_csv("../exp19/clear.csv")
    
    measurements = []
    for img_name in data:
        image_path = get_image_path(img_name, dir=images_path)
        actual = data[img_name]
        measurements.append((image_path, actual))

    crop = [(250, 1480, 4260, 1680), (350, 1520, 4300, 1650)]
    threshold = range(150, 255, 5)
    N = 10

    # f = FluidImage(get_image_path("200_1.jpg", dir=images_path))
    # # f.show()
    # f.crop(crop).show()
    # a

    exp = (
        Experiment(measurements)
        .set_crop(crop)
        .set_threshold(threshold)
        .setup_nheights(N)
    )
    results = exp.run()

    # best = results.find_best_singlevar(ExperimentResult.mse)
    
    # # plot(best, errbars=False, xlab="Area (pixels squared)")
    # # # plot(best, errbars=False, xlab="Height (pixels)")
    # # plt.show()
    
    # print(best)
    
    best = results.find_best_convexhull_area(ExperimentResult.mse)
        
    # plot(best, errbars=False, xlab="Area (pixels squared)")
    # # plot(best, errbars=False, xlab="Height (pixels)")
    # plt.show()
    
    print(best)
    
    best = results.find_best_max_height(ExperimentResult.mse)
        
    # plot(best, errbars=False, xlab="Area (pixels squared)")
    # # plot(best, errbars=False, xlab="Height (pixels)")
    # plt.show()
    
    print(best)
    
    best = results.find_best_pixelcount_area(ExperimentResult.mse)
        
    # plot(best, errbars=False, xlab="Area (pixels squared)")
    # # plot(best, errbars=False, xlab="Height (pixels)")
    # plt.show()
    
    print(best)
    
    best = results.find_best_nheights_area(ExperimentResult.mse)
        
    # plot(best, errbars=False, xlab="Area (pixels squared)")
    # # plot(best, errbars=False, xlab="Height (pixels)")
    # plt.show()
    
    print(best)
    
    
def run_exp19(_):
   
    from common import get_image_path, read_csv
    
    images_path = "../exp19/images"
    data = read_csv("../exp19/clear.csv")
    
    measurements = []
    for img_name in data:
        image_path = get_image_path(img_name, dir=images_path)
        actual = data[img_name]
        measurements.append((image_path, actual))

    crop = (442, 1220, 4260, 1440)
    threshold = range(150, 255, 5)
    N = 10

    # f = FluidImage(get_image_path("200_1.jpg", dir=images_path))
    # # f.show()
    # f.crop(crop).show()
    # a

    exp = (
        Experiment(measurements)
        .set_crop(crop)
        .set_threshold(threshold)
        .setup_nheights(N)
    )
    results = exp.run()

    best = results.find_best_singlevar(ExperimentResult.max_se)
    plot(best, errbars=False, xlab="Area (pixels squared)")
    plt.show()
    
    print(best)

def plot_multivar(best, errbars=True, xlab=None):

    from sklearn.metrics import r2_score
    
    plot = plt.subplot2grid((1, 1), (0, 0))
    
    # X = best["X"]
    Y = best["Y"]
    X = [i for i in range(1, len(Y) + 1)]
    print(Y)
    Y_pred = best["Y_pred"]
    r2 = r2_score(Y, Y_pred)
    error = best["error"]
    errs = 10
    mse = ExperimentResult.mse(Y, Y_pred)
    if errbars:
        plot.errorbar(X, Y, yerr=errs, linestyle='None', marker='.', label="Actual (Measured) Volume")
    else:
        plot.scatter(X, Y, label="Actual (Measured) Volume")
    #  linestyle='None', marker='^',
    plot.plot(
        X,
        Y_pred,
        label=f"Regression Model MSE({mse:.1f}) r2({r2:.1f})",
    )
      # , yerr=np.std(Y_pred_new)
    # plot.errorbar(
    #     X, Y_pred_old, yerr=yerr_man, label=f"manual mse={err_man_mse:.1f} max={err_man_max:.1f}"
    # )
    # plot.plot(X, yerrLower, ":r")
    # plot.plot(X, yerrUpper, ":r")

    xlab = xlab if xlab else "X"
    plot.set_xlabel(xlab)
    plot.set_ylabel("Volume (uL)")

    c = best["c"]
    t = best["t"]
    n = best.get("n", None)

    title = f"Generated Model. Method({best['method']}) with threshold({t})" + (f" and N({n})" if n else "")
    plot.set_title(title)
    plot.legend()

def find_multivar(csv_path):
    
    from common import get_image_path, read_csv
    
    images_path = "../exp26/images"
    data = read_csv(csv_path)
    
    measurements = []
    for img_name in data:
        image_path = get_image_path(img_name, dir=images_path)
        actual = data[img_name]
        measurements.append((image_path, actual))

    crop = (250, 1480, 4260, 1680)
    threshold = range(150, 255, 5)
    N = 5

    exp = (
        Experiment(measurements)
        .set_crop(crop)
        .set_threshold(threshold)
        .setup_nheights(N)
    )
    results = exp.run()
    
    
    # best = results.find_best_nheights(ExperimentResult.max_se)
    # print(best)
    # plot_multivar(best, xlab="Image")
    # plt.show()
    
    best = results.find_best_nheights(ExperimentResult.mse, degree=1)
    print(best)
    print(best["raw_model"]["poly"].get_feature_names())
    print(best["raw_model"]["linear"].coef_)
    plot_multivar(best, xlab="Image")
    plt.show()


    # best = results.find_best_nheights(ExperimentResult.max_se, regression=TheilSenRegressor())
    # print(best)
    # plot_multivar(best)
    # plt.show()

if __name__ == "__main__":

    # run_exp19(None)
    # run_exp26("../exp26/clear.csv")
    # run_exp26("../exp26/clear_first.csv")
    # run_exp26("../exp26/clear_all.csv")
    # run_exp26("../exp26/clear_more.csv")
    
    find_multivar("../exp26/clear.csv")