from pathlib import Path
from pandas import DataFrame
from neo4j import GraphDatabase

from backends.data_cleanup import fetch_si_neo4j_dataset
from neo4j_backends.predictions import extract_predictions, insert_paper_error, insert_errors, insert_predicted_values, insert_predicted_std_values
from neo4j_backends.si_insert_base import merge_schema


def main():

    """
    It is recommended that si_ml_run be run before this script, and that the Neo4j Database
    that you wish to target should be active and specified below.

    :return:
    """

    # Grab database driver data
    uri = "neo4j://localhost:7687"
    username = "neo4j"
    password = "password"
    encrypted = False
    trust = "TRUST_ALL_CERTIFICATES"
    driver = GraphDatabase.driver(uri, auth=(username, password), encrypted=encrypted, trust=trust)

    # Define database
    database = "neo4j"

    # Merge main data into Neo4j
    si_dataset = fetch_si_neo4j_dataset()
    schema_file = Path("neo4j_backends/si_schema.txt")
    merge_schema(dataset=si_dataset, schema_file=schema_file, driver=driver, database=database)

    # Merge error data into neo4j
    filter_options = ["control", "no_outliers", "drop"]
    main_df = fetch_si_neo4j_dataset()
    for filter_option in filter_options:
        df = extract_predictions(Path("output") / filter_option, aerogel_type="si")
        new_df = []
        for _, row in df.iterrows():
            title = main_df.loc[main_df["Final Material"] == row["Final Material"]].to_dict('records')[0]
            title = title["Title"]
            row["Title"] = title
            new_df.append(row)
        new_df = DataFrame(new_df)

        insert_paper_error(new_df, driver, database, prop_key=f"{filter_option}_paper_error")
        insert_errors(new_df, driver, database, prop_key=f"{filter_option}_outliers_error")
        insert_predicted_values(new_df, driver, database, prop_key=f"{filter_option}_predicted_surface_area")
        insert_predicted_std_values(new_df, driver, database, prop_key=f"{filter_option}_predicted_surface_area_std")


if __name__ == "__main__":
    main()
