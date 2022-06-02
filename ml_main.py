from os import urandom
from datetime import datetime
from json import dump
from math import ceil

from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from backends.data_cleanup import fetch_si_ml_dataset
from machine_learning.featurize import featurize
from machine_learning.keras_nn import tune, build_estimator
from machine_learning.graph import pva_graph
from machine_learning.misc import zip_run_name_files


def run_params(base_df, seed, y_column, num_of_trials, train_percent, validation_percent,
               drop_columns, paper_id_column):

    # Drop columns to perform featurization
    base_df = base_df.drop(columns=drop_columns)

    # Featurize DataFrame
    base_df = featurize(base_df, paper_id_column, bit_size=128, drop_final_material_col=False)

    # Reset the Index
    base_df = base_df.reset_index(drop=True)

    # Grab material list and index
    material_col = base_df["Final Material"]
    base_df = base_df.drop(columns=["Final Material"])


    test_percent = 1 - train_percent - validation_percent
    num_of_sections = ceil(1 / test_percent) + 1
    for i in range(num_of_sections):

        # Gather what section running for test set
        start_split = round(i * test_percent * len(base_df))
        end_split = round((i + 1) * test_percent * len(base_df))
        end_split = min(len(base_df), end_split)

        if end_split - start_split <= 0:
            break

        # Spilt up data
        # train_df = df.sample(frac=train_percent, random_state=seed)
        # test_df = df.drop(train_df.index)
        test_df = base_df.iloc[start_split:end_split]
        train_df = base_df.drop(test_df.index)
        val_df = train_df.sample(frac=validation_percent, random_state=seed)
        train_df = train_df.drop(val_df.index)

        # Grab Test df material list
        test_df_index = test_df.index.tolist()
        material_list = material_col.loc[material_col.index[test_df_index]].tolist()

        # Remove grouping column
        train_df = train_df.drop(columns=[paper_id_column])
        test_df = test_df.drop(columns=[paper_id_column])
        val_df = val_df.drop(columns=[paper_id_column])

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
        n_hidden = list(range(1, 5, 1))
        n_neuron = list(range(20, 300, 20))
        drop = [0.20, 0.22, 0.24, 0.26, 0.28, 0.30]
        epochs = [100]
        param_grid = {
            'n_hidden': n_hidden,
            "n_neuron": n_neuron,
            'drop': drop,
        }

        # Hyper-Tune the model and fetch the best parameters
        best_params = tune(train_features, train_target, val_features, val_target, epochs=epochs,
                           n_hidden_layers=n_hidden, n_neurons=n_neuron, dropout=drop,
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
        scaled_pva = (pva - pva.min().min()) / (pva.max().max() - pva.min().min())
        mse = mean_squared_error(scaled_pva["actual"], scaled_pva["pred_avg"]).mean()
        rmse = mse ** (1 / 2)
        r2 = r2_score(scaled_pva["actual"], scaled_pva["pred_avg"]).mean()

        predictions = predictions.join(pva)
        predictions["Index"] = test_target.index.tolist()

        predictions["Final Material"] = material_list

        # Dump Information about run
        date_string = datetime.now().strftime('%Y_%m_%d %H_%M_%S')
        run_name = f"bulk_{date_string}".replace("/", "_")

        run_info = {
            "seed": seed,
            "y_column": y_column,
            "epochs": epochs,
            "num_of_trials": num_of_trials,
            "param_gird": param_grid,
            "keras_best_params": best_params,
        }
        with open(f"{run_name}_run_params.json", "w") as f:
            dump(run_info, f)
        with open(f"{run_name}_feature_list.json", "w") as f:
            dump(feature_list, f)
        predictions.to_csv(f"{run_name}_predictions.csv")
        pva_graph(scaled_pva, r2, mse, rmse, run_name)
        zip_run_name_files(run_name)


def main():
    seed = int.from_bytes(urandom(3), "big")  # Generate an actual random number
    raw_data = fetch_si_ml_dataset()
    raw_data.to_csv('dev.csv')

    # General Properties for Machine Learning
    num_of_trials = 100
    train_percent = 0.8
    validation_percent = 0.1  # Note, this is the percent of the train set used for validation

    # General Columns
    drop_columns = ['Porosity', 'Porosity %', 'Pore Volume cm3/g', 'Average Pore Diameter nm',
                    'Bulk Density g/cm3', 'Young Modulus MPa', 'Crystalline Phase',
                    'Average Pore Size nm', 'Thermal Conductivity W/mK', 'Gelation Time mins']
    paper_id_column = 'Title'

    # Parameters to cycle
    y_column = 'Surface Area m2/g'

    # Drop rows that have NaN in y_column
    raw_data = raw_data.dropna(subset=[y_column])

    # Train and predict only on Aerogels
    raw_data = raw_data.loc[raw_data['Final Gel Type'] == "Aerogel"]
    raw_data = raw_data.drop(columns=['Final Gel Type'])

    # Fetch upper and lower threshold to filter data by, looking for top and bottom 3 percent
    p = 0.03
    upper_threshold = raw_data[y_column].sort_values(ascending=True)[:-int(p * len(raw_data))].max()
    lower_threshold = raw_data[y_column].sort_values(ascending=True)[int(p * len(raw_data)):].min()

    # Grab part of DataFrame that is within the specified surface area range
    raw_data = raw_data.loc[raw_data[y_column] >= lower_threshold]
    raw_data = raw_data.loc[raw_data[y_column] <= upper_threshold]

    for i in range(10):

        # Shuffle df based on seed
        df = raw_data.sample(frac=1, random_state=seed+i)

        # Run parameters
        run_params(
            base_df=df,
            seed=seed+i,
            y_column=y_column,
            num_of_trials=num_of_trials,
            train_percent=train_percent,
            validation_percent=validation_percent,
            drop_columns=drop_columns,
            paper_id_column=paper_id_column,
        )


if __name__ == "__main__":
    main()
