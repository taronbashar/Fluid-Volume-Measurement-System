import pathlib
import csv


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
