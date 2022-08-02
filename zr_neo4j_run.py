from pathlib import Path
from neo4j import GraphDatabase
from pandas import read_excel, DataFrame

from neo4j_backends.predictions import extract_predictions, insert_paper_error
from neo4j_backends.zr_insert_base import insert_zr_into_neo4j


def main():
    """
    It is recommended that zr_ml_run be run before this script, and that the Neo4j Database
    that you wish to target should be active and specified below.

    :return:
    """

    main_df = read_excel(Path("backends/raw_zr_aerogels.xlsx"))
    df = extract_predictions(Path("output"), aerogel_type="zr")

    new_df = []
    for _, row in df.iterrows():
        title = main_df.loc[main_df["Final Material"] == row["Final Material"]].to_dict('records')[0]
        title = title["Title"]
        row["Title"] = title
        new_df.append(row)
    new_df = DataFrame(new_df)

    uri = "neo4j://localhost:7687"
    username = "neo4j"
    password = "password"
    encrypted = False
    trust = "TRUST_ALL_CERTIFICATES"
    driver = GraphDatabase.driver(uri, auth=(username, password), encrypted=encrypted, trust=trust)

    database = "neo4j"

    # with driver.session(database=database) as session:
    #     insert_zr_into_neo4j(session=session)

    insert_paper_error(new_df, driver, database, prop_key="no_drop_paper_error")


if __name__ == "__main__":
    main()
