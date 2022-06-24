from pathlib import Path
from neo4j import GraphDatabase

from neo4j_backends.predictions import extract_predictions, insert_paper_error
from neo4j_backends.zr_insert_base import insert_zr_into_neo4j


def main():
    """
    It is recommended that si_ml_run be run before this script, and that the Neo4j Database
    that you wish to target should be active and specified below.

    :return:
    """

    df = extract_predictions(Path("output"), aerogel_type="zr")

    uri = "neo4j://localhost:7687"
    username = "neo4j"
    password = "password"
    encrypted = False
    trust = "TRUST_ALL_CERTIFICATES"
    driver = GraphDatabase.driver(uri, auth=(username, password), encrypted=encrypted, trust=trust)

    database = "aerogels"

    with driver.session(database=database) as session:
        insert_zr_into_neo4j(session=session)

    insert_paper_error(df, driver, database)


if __name__ == "__main__":
    main()
