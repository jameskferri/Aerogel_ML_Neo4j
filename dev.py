from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

from pandas import read_csv, DataFrame


def get_predictions_from_zip(input_zip):
    input_zip = ZipFile(input_zip)
    data = None
    for name in input_zip.namelist():
        if "predictions" in name:
            data = BytesIO(input_zip.read(name))
            data = read_csv(data, encoding='utf8', sep=",")
    return data


if __name__ == "__main__":

    material_list = []
    for file in Path("output").iterdir():
        df = get_predictions_from_zip(file)
        material_list.extend(df["Final Material"].tolist())

    df = DataFrame()
    df["Final Material"] = material_list
    df.to_csv("dev_1.csv")
