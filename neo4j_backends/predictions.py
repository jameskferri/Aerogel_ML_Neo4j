from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

from tqdm import tqdm
from numpy import nan, isnan
from pandas import read_csv, DataFrame
from neo4j import GraphDatabase
import matplotlib.pyplot as plt
import numpy as np


def extract_predictions(output_dir: Path, aerogel_type):

    predictions = []
    final_materials = set()

    counter = 0
    for path in output_dir.iterdir():
        if "bulk" in path.stem and ".zip" == path.suffix and aerogel_type in path.stem:
            counter += 1

    # Grab predictions.csv from each zip file in output directory
    # Load zip files in memory
    for path in tqdm(output_dir.iterdir(), desc="Extracting Data", total=counter):

        if "bulk" in path.stem and ".zip" == path.suffix and aerogel_type in path.stem:
            zip_file = ZipFile(path)
            for file in zip_file.namelist():
                if "predictions" in file:

                    # Read csv files
                    df = zip_file.read(file)
                    df = BytesIO(df)
                    df = read_csv(df)

                    # Calculate error
                    df["error"] = df["pred_avg"] - df["actual"]
                    df["error"] = df["error"].abs()
                    df["error"] = df["error"] / df["actual"]

                    # Organize data
                    df = df[["Final Material", "error"]]

                    # Get final materials from csv file
                    final_materials.update(df["Final Material"])

                    predictions.append(df)

    # DataFrame to combine all data into
    combined_df = DataFrame()
    combined_df["Final Material"] = list(final_materials)

    error_cols = []
    for i, pva in tqdm(enumerate(predictions), desc="Calculating Prediction Errors", total=len(predictions)):

        # Define cols
        error_col = f"error_{i}"

        # Add to running list to take mean and std later
        error_cols.append(error_col)

        # Add NaN column to then loc into
        combined_df[error_col] = nan

        for index, row in pva.iterrows():
            final_material = row["Final Material"]
            error = row["error"]
            combined_df.loc[combined_df["Final Material"] == final_material, error_col] = error

    combined_df["error"] = combined_df[error_cols].mean(axis=1)
    combined_df["error_std"] = combined_df[error_cols].std(axis=1)
    combined_df = combined_df[["Final Material", "error", "error_std"]]

    return combined_df


def insert_paper_error(df, driver, database):

    with driver.session(database=database) as session:

        titles = df["title"].unique()
        for title in tqdm(titles, desc="Inserting paper errors"):
            mean_error = df.loc[df["title"] == title]
            mean_error = mean_error["error"]
            mean_error = mean_error.mean()
            if not isnan(mean_error):
                query = f"""

                MATCH (l:LitInfo)
                WHERE l.title = "{title}"
                SET l.paper_error = {mean_error}

                """

                session.run(query)


