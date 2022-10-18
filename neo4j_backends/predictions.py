from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

from tqdm import tqdm
from numpy import nan, isnan
from pandas import read_csv, DataFrame


def extract_predictions(output_dir: Path, aerogel_type):

    predictions = []
    final_materials = set()

    counter = 0
    for path in output_dir.iterdir():
        if "bulk" in path.stem and ".zip" == path.suffix and aerogel_type in path.stem:
            counter += 1

    # Grab predictions.csv from each zip file in si_no_outliers directory
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
                    df["error"] = df["error"]

                    # Organize data
                    df = df[["Final Material", "error", "pred_avg"]]

                    # Get final materials from csv file
                    final_materials.update(df["Final Material"])

                    predictions.append(df)

    # DataFrame to combine all data into
    combined_df = DataFrame()
    combined_df["Final Material"] = list(final_materials)

    error_cols = []
    pred_cols = []
    for i, pva in tqdm(enumerate(predictions), desc="Calculating Prediction Errors", total=len(predictions)):

        # Define cols
        error_col = f"error_{i}"
        pred_col = f"pred_{i}"

        # Add to running list to take mean and std later
        error_cols.append(error_col)
        pred_cols.append(pred_col)

        # Add NaN column to then loc into
        combined_df[error_col] = nan
        combined_df[pred_col] = nan

        for index, row in pva.iterrows():
            final_material = row["Final Material"]
            error = row["error"]
            pred_avg = row["pred_avg"]
            combined_df.loc[combined_df["Final Material"] == final_material, error_col] = error
            combined_df.loc[combined_df["Final Material"] == final_material, pred_col] = pred_avg

    combined_df["error"] = combined_df[error_cols].mean(axis=1)
    combined_df["error_std"] = combined_df[error_cols].std(axis=1)
    combined_df["pred"] = combined_df[pred_cols].mean(axis=1)
    combined_df["pred_std"] = combined_df[pred_cols].std(axis=1)
    combined_df = combined_df[["Final Material", "error", "error_std", "pred", "pred_std"]]

    return combined_df


def insert_paper_error(df, driver, database, prop_key):

    with driver.session(database=database) as session:

        titles = df["Title"].unique()
        for title in tqdm(titles, desc="Inserting paper errors"):
            mean_error = df.loc[df["Title"] == title]
            mean_error = mean_error["error"]
            mean_error = mean_error.mean()

            title = title.replace('"', '\\"')

            if not isnan(mean_error):
                query = f"""

                MATCH (l:LitInfo)
                WHERE l.title = "{title}"
                SET l.{prop_key} = {mean_error}

                """

                session.run(query)


def insert_errors(df, driver, database, prop_key):

    with driver.session(database=database) as session:

        for index, row in tqdm(df.iterrows(), desc="Inserting Errors"):

            final_material = row["Final Material"]
            error = row["error"]

            query = f"""

            MATCH (l:FinalGel)
            WHERE l.final_material = "{final_material}"
            SET l.{prop_key} = {error}

            """

            session.run(query)


def insert_predicted_values(df, driver, database, prop_key):

    with driver.session(database=database) as session:

        for index, row in tqdm(df.iterrows(), desc="Inserting Predictions"):

            final_material = row["Final Material"]
            pred = row["pred"]

            query = f"""

            MATCH (l:FinalGel)
            WHERE l.final_material = "{final_material}"
            SET l.{prop_key} = {pred}

            """

            session.run(query)


def insert_predicted_std_values(df, driver, database, prop_key):

    with driver.session(database=database) as session:

        for index, row in tqdm(df.iterrows(), desc="Inserting Predictions"):

            final_material = row["Final Material"]
            pred_std = row["pred_std"]

            query = f"""

            MATCH (l:FinalGel)
            WHERE l.final_material = "{final_material}"
            SET l.{prop_key} = {pred_std}

            """

            session.run(query)


def insert_error(df, driver, database, prop_key):

    with driver.session(database=database) as session:
        for index, row in df.iterrows():
            final_material = row["Final Material"]
            error = round(row["error"], 4)

            query = f"""

            MATCH (l:FinalGel)
            WHERE l.final_material = "{final_material}"
            SET l.{prop_key} = {error}

            """

            session.run(query)
