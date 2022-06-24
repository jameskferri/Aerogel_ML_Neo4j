from tqdm import tqdm
from pathlib import Path
from neo4j import GraphDatabase
from numpy import isnan

from backends.data_cleanup import fetch_si_neo4j_dataset
from neo4j_backends.predictions import extract_predictions, insert_paper_error
from neo4j_backends.si_insert_base import merge_schema


def main():

    """
    It is recommended that si_ml_run be run before this script, and that the Neo4j Database
    that you wish to target should be active and specified below.

    :return:
    """

    df = extract_predictions(Path("output"), aerogel_type="si")

    uri = "neo4j://localhost:7687"
    username = "neo4j"
    password = "password"
    encrypted = False
    trust = "TRUST_ALL_CERTIFICATES"
    driver = GraphDatabase.driver(uri, auth=(username, password), encrypted=encrypted, trust=trust)

    database = "aerogels"

    si_dataset = fetch_si_neo4j_dataset()
    schema_file = Path("neo4j_backends/si_schema.txt")

    merge_schema(dataset=si_dataset, schema_file=schema_file, driver=driver, database=database)

    insert_paper_error(df, driver, database)


if __name__ == "__main__":
    main()
