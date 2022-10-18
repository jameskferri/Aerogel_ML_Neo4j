# Aerogel_ML_Neo4j
This repo is the code base for the paper <paper>.
For a brief summary, goal of this code base use machine learning
(Neural Networks from keras) to predict the surface area of Si
Aerogels.
But is hard to determine the quality of data from Si data 
from peer reviewed journals.
So this pipeline serves as a data cleaning method.
Where machine learning data and errors are generated,
and the data is feed to Neo4j for further analyzing.

## Environment
Conda is the preferred environment for this codebase.
With RDkit being the main package requiring conda.
In anaconda prompt (Windows) or terminal (MacOS/Linux),
navigate to the environment directory in this codebase.
Once in the environment directory, run

`conda env create -f far.yml`

This will create a conda virtual environment named
`far` that can be used to run the code.

## Workflow

This repo has three main files:
- si_ml_run.py : Generate ml results
- si_neo4j_run.py : Insert data into Neo4j
- pva_from_neo4j.py : Grab PVA graphs from Neo4j

Where the files should be run in the order

si_ml_run.py -> si_neo4j_run.py -> pva_from_neo4j.py

## Neo4j Python Driver

In si_neo4j_run.py and pva_from_neo4j.py,
the driver for Neo4j must be configured.
Before running either file, 
ensure that the Neo4j database is running
(Recommended to use Neo4j Desktop).
Inside both files there are specs for configuring
the driver. 

Directions for configuring the driver can be found at
https://neo4j.com/docs/api/python-driver/current/api.html