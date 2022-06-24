from os import urandom
from datetime import datetime
from json import dump
from math import ceil
from time import sleep

from numpy import arange
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from backends.data_cleanup import fetch_si_ml_dataset, fetch_zr_ml_dataset
from machine_learning.featurize import featurize
from machine_learning.keras_nn import tune, build_estimator
from machine_learning.graph import pva_graph
from machine_learning.misc import zip_run_name_files


def run_params(base_df, aerogel_type, seed, y_column, num_of_trials, train_percent, validation_percent):
    # Featurize DataFrame
    material_col = base_df["Final Material"]

    base_df = base_df.drop(columns=["Final Material"])
    base_df = featurize(base_df, paper_id_column=None, bit_size=128)

    test_percent = 1 - train_percent

    # TODO think of a better way to do this
    # Calculate start and stop indexes
    num_test_files = int(len(base_df) * test_percent)
    groups = []
    for i in range(len(base_df)):
        if i % num_test_files == 0:
            groups.append(i)
    sections = []
    for i in range(len(groups) - 1):
        sections.append([groups[i], groups[i + 1]])
    if groups[-1] != len(base_df):
        sections.append([groups[-1], len(base_df)])

    for i, section in enumerate(sections):

        start_split = section[0]
        end_split = section[1]

        if end_split - start_split <= 1:
            break

        # Spilt up data
        # train_df = df.sample(frac=train_percent, random_state=seed)
        # test_df = df.drop(train_df.index)
        test_df = base_df.iloc[start_split:end_split]
        train_df = base_df.drop(test_df.index)
        val_df = train_df.sample(frac=validation_percent, random_state=seed)
        train_df = train_df.drop(val_df.index)

        # Grab Final Materials from test set
        test_materials = material_col.loc[test_df.index].tolist()

        # Get Feature Columns
        feature_list = list(train_df.columns)

        # Separate Features and Target
        train_target = train_df[y_column].to_numpy()
        test_target = test_df[y_column]
        val_target = val_df[y_column].to_numpy()
        train_features = train_df.drop(columns=[y_column]).to_numpy()
        test_features = test_df.drop(columns=[y_column]).to_numpy()
        val_features = val_df.drop(columns=[y_column]).to_numpy()

        # Scale Features
        feature_scaler = StandardScaler()
        train_features = feature_scaler.fit_transform(train_features)
        test_features = feature_scaler.transform(test_features)
        val_features = feature_scaler.transform(val_features)

        # Scale Target
        target_scaler = StandardScaler()
        train_target = target_scaler.fit_transform(train_target.reshape(-1, 1))
        val_target = target_scaler.transform(val_target.reshape(-1, 1))

        # Keras Parameters
        n_hidden = [1, 2, 3, 4, 5]
        neurons = list(range(10, 200, 10))
        drop = list(arange(0.2, 0.25, 0.02))
        epochs = [100]
        param_grid = {
            'n_hidden': n_hidden,
            "neurons": neurons,
            'drop': drop,
        }

        # Hyper-Tune the model and fetch the best parameters
        best_params = tune(train_features, train_target, val_features, val_target, epochs=epochs,
                           n_hidden_layers=n_hidden, neurons=neurons, dropouts=drop,
                           num_of_trials=num_of_trials)

        # Gather predicted values on estimator
        predictions = DataFrame()
        for j in range(5):
            estimator = build_estimator(best_params)
            estimator.fit(train_features, train_target, epochs=best_params["epochs"])
            predictions_j = estimator.predict(test_features)
            predictions_j = target_scaler.inverse_transform(predictions_j.reshape(-1, 1)).reshape(-1, )
            predictions[f"predictions_{j}"] = predictions_j

        # # ### DEV SECTION ### #
        # predictions = DataFrame()
        # for j in range(5):
        #     predictions_j = test_target.tolist()
        #     predictions[f"predictions_{j}"] = predictions_j
        # # ### END DEV SECTION ### #

        # Gather PVA data
        pva = DataFrame()
        pva["actual"] = test_target.to_numpy()
        pva["pred_avg"] = predictions.mean(axis=1)
        pva["pred_std"] = predictions.std(axis=1)

        # Scale PVA for stats
        scaled_pva = pva.copy()
        for col in scaled_pva:
            if pva[col].max() - pva[col].min() == 0:
                scaled_pva[col] = 0
            else:
                scaled_pva[col] = (pva[col] - pva[col].min()) / (pva[col].max() - pva[col].min())
        mse = mean_squared_error(scaled_pva["actual"], scaled_pva["pred_avg"]).mean()
        rmse = mse ** (1 / 2)
        r2 = r2_score(scaled_pva["actual"], scaled_pva["pred_avg"]).mean()

        predictions = predictions.join(pva)
        predictions["Index"] = test_target.index.tolist()
        predictions["Final Material"] = test_materials

        # Dump Information about run
        date_string = datetime.now().strftime('%Y_%m_%d %H_%M_%S')
        run_name = f"{aerogel_type}_bulk_{date_string}".replace("/", "_")

        run_info = {
            "seed": seed,
            "y_column": y_column,
            "epochs": epochs,
            "num_of_trials": num_of_trials,
            "param_gird": param_grid,
            "keras_best_params": best_params,
            "dataset": aerogel_type
        }
        with open(f"{run_name}_run_params.json", "w") as f:
            dump(run_info, f)
        with open(f"{run_name}_feature_list.json", "w") as f:
            dump(feature_list, f)
        predictions.to_csv(f"{run_name}_predictions.csv")
        pva_graph(scaled_pva, r2, mse, rmse, run_name)
        zip_run_name_files(run_name)

        sleep(1)


def run(aerogel_type, cycles, num_of_trials, train_percent, validation_percent, drop_papers=None):

    y_column = 'Surface Area (m2/g)'

    seed = int.from_bytes(urandom(3), "big")  # Generate an actual random number

    if aerogel_type == "si":
        si_drop_columns = ['Porosity', 'Porosity (%)', 'Pore Volume (cm3/g)', 'Average Pore Diameter (nm)',
                           'Bulk Density (g/cm3)', 'Young Modulus (MPa)', 'Crystalline Phase',
                           'Average Pore Size (nm)', 'Thermal Conductivity (W/mK)', 'Gelation Time (mins)']
        raw_data = fetch_si_ml_dataset(additional_drop_columns=si_drop_columns)
    elif aerogel_type == "zr":
        zr_drop_columns = ['Porosity', 'Porosity (%)', 'Pore Volume (cm3/g)', 'Pore Size (nm)',
                           'Nanoparticle Size (nm)', 'Density (g/cm3)',
                           'Thermal Conductivity (W/mK)', 'Crystalline Phase', 'Gelation Time (mins)']
        raw_data = fetch_zr_ml_dataset(additional_drop_columns=zr_drop_columns)
    else:
        raise TypeError()

    if drop_papers:
        for drop_paper in drop_papers:
            raw_data = raw_data.loc[raw_data["Title"] != drop_paper]

    # Remove any rows that do not have surface area specified
    raw_data = raw_data.dropna(subset=[y_column])

    # Train and predict only on Aerogels
    raw_data = raw_data.loc[raw_data['Final Gel Type'] == "Aerogel"]
    raw_data = raw_data.drop(columns=['Final Gel Type'])

    # # Fetch upper and lower threshold to filter data by, looking for top and bottom 3 percent
    # p = 0.05
    # upper_threshold = raw_data[y_column].sort_values(ascending=True)[:-int(p * len(raw_data))].max()
    # lower_threshold = raw_data[y_column].sort_values(ascending=True)[int(p * len(raw_data)):].min()
    #
    # # Remove top and bottom 3 percent
    # raw_data = raw_data.loc[raw_data[y_column] >= lower_threshold]
    # raw_data = raw_data.loc[raw_data[y_column] <= upper_threshold]

    for _ in range(cycles):
        # Shuffle DataFrame
        raw_data = raw_data.sample(frac=1)
        raw_data = raw_data.reset_index(drop=True)

        run_params(
            base_df=raw_data,
            aerogel_type=aerogel_type,
            seed=seed,
            y_column=y_column,
            num_of_trials=num_of_trials,
            train_percent=train_percent,
            validation_percent=validation_percent,
        )