from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

from tqdm import tqdm
from pandas import read_csv


def main(output_dir: Path):

    final_materials = set()

    counter = 0
    for path in output_dir.iterdir():
        if "bulk" in path.stem and ".zip" == path.suffix:
            counter += 1

    # Grab predictions.csv from each zip file in output directory
    # Load zip files in memory
    # for path in tqdm(output_dir.iterdir(), desc="Extracting Data", total=counter):
    for path in output_dir.iterdir():
        if "bulk" in path.stem and ".zip" == path.suffix:
            zip_file = ZipFile(path)
            for file in zip_file.namelist():
                if "predictions" in file:

                    # Read csv files
                    df = zip_file.read(file)
                    df = BytesIO(df)
                    df = read_csv(df)

                    final_materials.update(df["Final Material"])
    print(len(final_materials))


if __name__ == "__main__":
    main(Path("output"))
