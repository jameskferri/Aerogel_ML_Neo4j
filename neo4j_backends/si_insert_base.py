from pathlib import Path
from ast import literal_eval

from tqdm import tqdm
from numpy import isnan
from neo4j import GraphDatabase

from backends.data_cleanup import fetch_si_neo4j_dataset


def collect_nodes(schema_file):
    all_nodes = []
    current_node = None
    on_node = False
    for line in schema_file:

        # Remove comments
        line = line.split("#")[0]
        line = line.strip()

        # line is not blank of only a comment
        if line:

            # Working on a node block
            if on_node:

                # If next line is a relationship block, stop collecting properties
                if line[:3].lower() == "rel" or line.strip() == "$end$":
                    all_nodes.append(current_node)
                    on_node = False

                # If working on a node block
                elif line[:4].lower() == "node":
                    all_nodes.append(current_node)
                    _, node_id, node_name = line.split("|")
                    current_node = {"node_id": node_id, "node_name": node_name, "unique_prop": None,
                                    "set_props": [], "props": []}
                    on_node = True

                # Else collect the property on the line
                else:

                    # Remove prop delimiters
                    line = line[1:]
                    if line[0] == "-":
                        line = line[1:]

                    prop_name, prop = line.split(":")
                    prop_name = prop_name.strip()
                    prop = prop.strip()

                    # If property is unique
                    if prop[-1] == "*":
                        prop = prop.split("*")[0]
                        prop = prop.strip()
                        current_node['unique_prop'] = {prop_name: prop}

                    # If property is a constant value
                    elif prop[0] == "{" and prop[-1] == "}":
                        prop = prop[1:-1]
                        prop = literal_eval(prop)
                        current_node["set_props"].append({prop_name: prop})

                    # Else add to general properties
                    else:
                        current_node["props"].append({prop_name: prop})

            # Check and see if started working on node block
            else:
                if line[:4].lower() == "node":
                    _, node_id, node_name = line.split("|")
                    current_node = {"node_id": node_id, "node_name": node_name, "unique_prop": None,
                                    "set_props": [], "props": []}
                    on_node = True

    return all_nodes


def collect_rels(schema_file, nodes):
    all_rels = []
    current_rel = None
    on_rel = False
    for line in schema_file:

        # Remove comments
        line = line.split("#")[0]
        line = line.strip()

        # line is not blank and not only a comment
        if line:

            # Working on a rel block
            if on_rel:

                # If next line is a node block, stop collecting properties
                if line[:4].lower() == "node" or line.strip() == "$end$":
                    all_rels.append(current_rel)
                    on_rel = False

                # If working on a new rel block
                elif line[:3].lower() == "rel":
                    all_rels.append(current_rel)
                    _, node_ids, rel_name = line.split("|")
                    node_id_1, node_id_2 = node_ids.split("->")
                    current_rel = {"node_1": node_id_1, "node_2": node_id_2, "rel_name": rel_name,
                                   "set_props": [], "props": []}
                    on_rel = True

                # Else collect the property on the line
                else:

                    # Remove prop delimiters
                    line = line[1:]
                    if line[0] == "-":
                        line = line[1:]

                    prop_name, prop = line.split(":")
                    prop_name = prop_name.strip()
                    prop = prop.strip()

                    # If property is a constant value
                    if prop[0] == "{" and prop[-1] == "}":
                        prop = prop[1:-1]
                        prop = literal_eval(prop)
                        current_rel["set_props"].append({prop_name: prop})

                    # Else add to general properties
                    else:
                        current_rel["props"].append({prop_name: prop})

            # Check and see if started working on node block
            else:
                if line[:3].lower() == "rel":
                    _, node_ids, rel_name = line.split("|")
                    node_id_1, node_id_2 = node_ids.split("->")
                    current_rel = {"node_1": node_id_1, "node_2": node_id_2, "rel_name": rel_name,
                                   "set_props": [], "props": []}
                    on_rel = True

    new_rels = []
    for rel in all_rels:

        for node in nodes:
            if node["node_id"] == rel["node_1"]:
                rel["node_1"] = (node["node_name"], node["unique_prop"])
            if node["node_id"] == rel["node_2"]:
                rel["node_2"] = (node["node_name"], node["unique_prop"])

        new_rels.append(rel)

    return all_rels


def format_prop(prop):
    if isinstance(prop, float) and isnan(prop):
        return None
    if isinstance(prop, str):
        prop = prop.replace('"', '\\"')
        prop = prop.replace("'", "\\'")
        prop = f"'{prop}'"
    return prop


def merge_schema(dataset, schema_file, driver, database):

    with open(schema_file, "r") as f:
        schema = f.readlines()

    nodes = collect_nodes(schema)
    rels = collect_rels(schema, nodes)

    with driver.session(database=database) as session:

        for _, row in tqdm(dataset.iterrows(), desc="Inserting Data", total=len(dataset)):
            row = dict(row)
            queries = []

            for node in nodes:

                query = ""

                unique_prop_name, unique_prop = tuple(node["unique_prop"].items())[0]
                unique_prop = row[unique_prop]
                unique_prop = format_prop(unique_prop)
                if unique_prop is not None:
                    query += "MERGE (%s:%s {%s:%s})\n" % (node["node_id"], node["node_name"],
                                                          unique_prop_name, unique_prop)

                    for set_prop in node["set_props"]:
                        set_prop_name, set_prop = tuple(set_prop.items())[0]
                        set_prop = format_prop(set_prop)
                        if set_prop is not None:
                            query += "SET %s.%s=%s\n" % (node["node_id"], set_prop_name, set_prop)

                    for prop in node["props"]:
                        prop_name, prop = tuple(prop.items())[0]
                        prop = row[prop]
                        prop = format_prop(prop)
                        if prop is not None:
                            query += "SET %s.%s=%s\n" % (node["node_id"], prop_name, prop)

                    queries.append(query)

            for rel in rels:

                query = ""

                node_1_name = rel["node_1"][0]
                node_1_prop = rel["node_1"][1]
                node_1_prop_name, node_1_prop = tuple(node_1_prop.items())[0]
                node_1_prop = row[node_1_prop]
                node_1_prop = format_prop(node_1_prop)

                node_2_name = rel["node_2"][0]
                node_2_prop = rel["node_2"][1]
                node_2_prop_name, node_2_prop = tuple(node_2_prop.items())[0]
                node_2_prop = row[node_2_prop]
                node_2_prop = format_prop(node_2_prop)

                if node_1_prop is not None and node_2_prop is not None:

                    query += "MATCH (a:%s {%s:%s})\n" % (node_1_name, node_1_prop_name, node_1_prop)
                    query += "MATCH (b:%s {%s:%s})\n" % (node_2_name, node_2_prop_name, node_2_prop)
                    query += "MERGE (a)-[r:%s]->(b)\n" % (rel["rel_name"])

                    for set_prop in rel["set_props"]:
                        set_prop_name, set_prop = tuple(set_prop.items())[0]
                        set_prop = format_prop(set_prop)
                        if set_prop is not None:
                            query += "SET r.%s=%s\n" % (set_prop_name, set_prop)

                    for prop in rel["props"]:
                        prop_name, prop = tuple(prop.items())[0]
                        prop = row[prop]
                        prop = format_prop(prop)
                        if prop is not None:
                            query += "SET r.%s=%s\n" % (prop_name, prop)

                queries.append(query)

            for query in queries:
                if query:
                    session.run(query)


def main():

    si_dataset = fetch_si_neo4j_dataset()
    schema_file = Path("si_schema.txt")

    # Collect driver information
    uri = "neo4j://localhost:7687"
    username = "neo4j"
    password = "password"
    encrypted = False
    trust = "TRUST_ALL_CERTIFICATES"
    driver = GraphDatabase.driver(uri, auth=(username, password), encrypted=encrypted, trust=trust)

    # Specify which database to target
    database = "neo4j"

    merge_schema(dataset=si_dataset, schema_file=schema_file, driver=driver, database=database)


if __name__ == "__main__":
    main()
