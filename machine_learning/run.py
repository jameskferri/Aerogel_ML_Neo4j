from os import mkdir
from os.path import exists
from os import urandom
from datetime import datetime
from json import dump
from math import ceil
from time import sleep

from numpy import arange
from pandas import DataFrame, read_csv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from backends.data_cleanup import fetch_si_ml_dataset
from machine_learning.featurize import featurize
from machine_learning.keras_nn import tune, build_estimator
from machine_learning.graph import pva_graph
from machine_learning.misc import zip_run_name_files
from neo4j_backends.predictions import extract_predictions


def run_params(base_df, aerogel_type, seed, y_column, num_of_trials, train_percent, validation_percent, working_dir):

    # Featurize DataFrame
    material_col = base_df["Final Material"]

    base_df = base_df.drop(columns=["Final Material"])
    base_df = featurize(base_df, paper_id_column=None, bit_size=128)

    test_percent = 1 - train_percent

    # Calculate start and stop indexes for each test set
    sections = []
    n_total = len(base_df)
    n_test = int(n_total * test_percent)
    for i in range(ceil(n_total / n_test)):
        left_test_index = i * n_test
        right_test_index = (i + 1) * n_test

        # Ensure no overflow of indexes
        if right_test_index > n_total:
            right_test_index = n_total

        # If right_test_index - left_test_index != 0
        # Otherwise there are no samples in spilt
        if left_test_index != right_test_index:
            sections.append([left_test_index, right_test_index])

    for i, section in enumerate(sections):

        start_split = section[0]
        end_split = section[1]

        if end_split - start_split <= 1:
            break

        # Spilt up data
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
        n_hidden = list(range(0, 10))
        neurons = list(range(10, 200, 10))
        drop = list(arange(0.15, 0.4, 0.02))
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
        zip_run_name_files(run_name, working_dir)

        sleep(1)


def run(aerogel_type, cycles, num_of_trials, train_percent, validation_percent, filter_options, output_dir):

    if exists(output_dir):
        raise OSError("output directory already exists")
    mkdir(output_dir)

    y_column = 'Surface Area (m2/g)'

    seed = int.from_bytes(urandom(3), "big")  # Generate an actual random number

    if aerogel_type == "si":
        si_drop_columns = ['Porosity', 'Porosity (%)', 'Pore Volume (cm3/g)', 'Average Pore Diameter (nm)',
                           'Bulk Density (g/cm3)', 'Young Modulus (MPa)', 'Crystalline Phase',
                           'Average Pore Size (nm)', 'Thermal Conductivity (W/mK)', 'Gelation Time (mins)']
        raw_data = fetch_si_ml_dataset(additional_drop_columns=si_drop_columns)
    else:
        raise TypeError()

    # Remove any rows that do not have surface area specified
    raw_data = raw_data.dropna(subset=[y_column])

    # Train and predict only on Aerogels
    raw_data = raw_data.loc[raw_data['Final Gel Type'] == "Aerogel"]
    raw_data = raw_data.drop(columns=['Final Gel Type'])

    if "no_outliers" in filter_options and "control" not in filter_options:
        raise ValueError("control must be included in filter options if no_outliers is used")
    if "no_outliers" in filter_options:
        filter_options.remove("no_outliers")
        filter_options.append("no_outliers")

    valid_filter_options = ["control", "drop", "no_outliers"]
    for filter_option in filter_options:
        if filter_option not in valid_filter_options:
            raise ValueError(f"Filter option of {filter_option} is not valid")

    for filter_option in filter_options:

        working_dir = output_dir / filter_option
        mkdir(working_dir)

        if filter_option == "control":
            # Just take copy of original data for a control group
            filtered_data = raw_data.copy()
        elif filter_option == "drop" or filter_option == "no_outliers":
            # Remove paper that are 2 std above or below the average
            y_col_std = raw_data[y_column].std()
            y_col_mean = raw_data[y_column].mean()
            filtered_data = raw_data.loc[raw_data[y_column] < y_col_mean + 2 * y_col_std]
            filtered_data = filtered_data.loc[filtered_data[y_column] > y_col_mean - 2 * y_col_std]
            if filter_option == "no_outliers":
                # Fetch aerogels with low prediction error
                drop_dir = output_dir / "drop"
                drop_predictions = extract_predictions(output_dir=drop_dir, aerogel_type="si")
                drop_predictions = drop_predictions[["Final Material", "error"]]
                error_std = drop_predictions["error"].std()
                error_mean = drop_predictions["error"].mean()
                low_error_aerogels = drop_predictions.loc[drop_predictions["error"] < error_mean + 2 * error_std]
                low_error_aerogels = low_error_aerogels.loc[low_error_aerogels["error"] > error_mean - 2 * error_std]
                low_error_aerogels = low_error_aerogels["Final Material"].tolist()
                # Filter raw data to only keep low error aerogels
                filtered_data = filtered_data.loc[filtered_data["Final Material"].isin(low_error_aerogels)]
        else:
            raise ValueError(f"Filter option of {filter_option} is not valid")

        for _ in range(cycles):

            # Shuffle DataFrame
            filtered_data = filtered_data.sample(frac=1)
            filtered_data = filtered_data.reset_index(drop=True)

            run_params(
                base_df=filtered_data,
                aerogel_type=aerogel_type,
                seed=seed,
                y_column=y_column,
                num_of_trials=num_of_trials,
                train_percent=train_percent,
                validation_percent=validation_percent,
                working_dir=working_dir,
            )
