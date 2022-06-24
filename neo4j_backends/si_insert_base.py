from pathlib import Path
from ast import literal_eval


def collect_nodes(schema_file):

    all_nodes = []
    current_node = None
    on_node = False
    for line in schema_file:

        # Remove comments
        line = line.split("#")[-1]
        line = line.strip()

        # line is not blank of only a comment
        if line:

            # Working on a node block
            if on_node:

                # If next line is a relationship block, stop collecting properties
                if line[:3].lower() == "rel":
                    all_nodes.append(current_node)
                    on_node = False

                # If working on a node block
                elif line[:3].lower() == "node":
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


def collect_rels(schema_file):

    all_nodes = []
    current_rel = None
    on_rel = False
    for line in schema_file:

        # Remove comments
        line = line.split("#")[-1]
        line = line.strip()

        # line is not blank of only a comment
        if line:

            # Working on a node block
            if on_rel:

                # If next line is a relationship block, stop collecting properties
                if line[:4].lower() == "node":
                    all_nodes.append(current_rel)
                    on_rel = False

                # If working on a node block
                elif line[:3].lower() == "rel":
                    all_nodes.append(current_rel)
                    _, node_id, node_name = line.split("|")
                    current_rel = {"node_id": node_id, "node_name": node_name, "unique_prop": None,
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

                    # If property is unique
                    if prop[-1] == "*":
                        prop = prop.split("*")[0]
                        current_rel['unique_prop'] = {prop_name: prop}

                    # If property is a constant value
                    elif prop[0] == "{" and prop[-1] == "}":
                        prop = prop[1:-1]
                        prop = literal_eval(prop)
                        current_rel["set_props"].append({prop_name: prop})

                    # Else add to general properties
                    else:
                        current_rel["props"].append({prop_name: prop})

            # Check and see if started working on node block
            else:
                if line[:3].lower() == "rel":
                    _, node_id, node_name = line.split("|")
                    current_rel = {"node_id": node_id, "node_name": node_name, "unique_prop": None,
                                    "set_props": [], "props": []}
                    on_rel = True


if __name__ == "__main__":

    with open(Path("si_schema.txt"), "r") as f:
        schema = f.readlines()

    collect_nodes(schema)
