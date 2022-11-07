from pathlib import Path
from os import mkdir
from os.path import exists
from shutil import rmtree

from pandas import DataFrame
from neo4j import GraphDatabase
from sklearn.metrics import mean_squared_error, r2_score
from machine_learning.graph import pva_graph


if __name__ == "__main__":
    uri = "neo4j://localhost:7687"
    username = "neo4j"
    password = "password"
    encrypted = False
    trust = "TRUST_ALL_CERTIFICATES"
    driver = GraphDatabase.driver(uri, auth=(username, password), encrypted=encrypted, trust=trust)

    database = "neo4j"

    if exists(Path("PVA Output")):
        rmtree(Path("PVA Output"))
    mkdir(Path("PVA Output"))

    prop_keys = ["no_outliers", "drop", "control"]

    for prop_key in prop_keys:

        queries = dict(
            Surfactant=f"""
            MATCH (n:FinalGel)
            WHERE n.{prop_key}_single_model_predicted_surface_area IS NOT NULL
            MATCH (n)-[]->(:Surfactant)
            RETURN n
            """,
            Surfactant_Ammon=f"""
            MATCH (n:FinalGel)
            WHERE n.{prop_key}_single_model_predicted_surface_area IS NOT NULL
            MATCH (n)-[]->(:Surfactant)
            MATCH (n)-[]->(b:BaseCatalyst)
            WHERE b.name = "NH4OH"
            RETURN n
            """,
            Base_Catalyst=f"""
            MATCH (n:FinalGel)
            WHERE n.{prop_key}_single_model_predicted_surface_area IS NOT NULL
            MATCH (n)-[]->(:BaseCatalyst)
            RETURN n
            """,
            Supercritical_Drying=f"""
            MATCH (n:FinalGel)
            WHERE n.{prop_key}_single_model_predicted_surface_area IS NOT NULL
            MATCH (n)-[]->(:DryingSteps)-[]->(:DryingMethod {"{"}method: "Supercritical Drying"{"}"})
            RETURN n
            """,
            Ambient_Pressure=f"""
            MATCH (n:FinalGel)
            WHERE n.{prop_key}_single_model_predicted_surface_area IS NOT NULL
            MATCH (n)-[]->(:Surfactant)
            WHERE (n)-[]->(:DryingSteps)-[]->(:DryingMethod {"{"}method: "Ambient Pressure Drying"{"}"})
            RETURN n
            """,
            All=f"""
            MATCH (n:FinalGel)
            WHERE n.{prop_key}_single_model_predicted_surface_area IS NOT NULL
            RETURN n
            """
        )

        for key, query in queries.items():

            with driver.session(database=database) as session:
                data = session.run(query).data()

            pva = []
            for row in data:
                row = list(row.values())[0]

                surface_area = row["surface_area"]
                predicted_sa = row[f"{prop_key}_predicted_surface_area"]
                predicted_sa_std = row[f"{prop_key}_predicted_surface_area_std"]

                pva.append(
                    dict(
                        surface_area=surface_area,
                        predicted_sa=predicted_sa,
                        predicted_sa_std=predicted_sa_std,
                    )
                )

            pva = DataFrame(pva)

            # Verify that at least 5 aerogels are in subregion
            if len(pva) > 2:

                pva["predicted_sa_std"] = pva["predicted_sa_std"].abs()

                max_sa, min_sa = pva["surface_area"].max(), pva["surface_area"].min()
                max_std_sa, min_std_sa = pva["predicted_sa_std"].max(), pva["predicted_sa_std"].min()

                pva["surface_area"] = (pva["surface_area"] - min_sa) / (max_sa - min_sa)
                pva["predicted_sa"] = (pva["predicted_sa"] - min_sa) / (max_sa - min_sa)
                pva["predicted_sa_std"] = (pva["predicted_sa_std"] - min_std_sa) / (max_std_sa - min_std_sa)

                mse = mean_squared_error(pva["surface_area"], pva["predicted_sa"]).mean()
                rmse = mse ** (1 / 2)
                r2 = r2_score(pva["surface_area"], pva["predicted_sa"]).mean()

                pva_graph(pva.rename(columns={"surface_area": "actual",
                                              "predicted_sa": "pred_avg",
                                              "predicted_sa_std": "pred_std"}),
                          r2, mse, rmse, run_name=f"PVA Output/{prop_key}_{key}")
