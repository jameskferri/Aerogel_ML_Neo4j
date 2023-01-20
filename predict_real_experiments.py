from pathlib import Path

from numpy import arange
from pandas import read_excel, read_csv, concat
from sklearn.preprocessing import StandardScaler

from backends.data_cleanup import fetch_si_ml_dataset
from machine_learning.featurize import featurize, DataFrame
from machine_learning.keras_nn import tune, build_estimator
from neo4j_backends.predictions import extract_predictions


def filter_data(df, y_col):

    # Fetch aerogels with low prediction error
    y_col_std = df[y_col].std()
    y_col_mean = df[y_col].mean()
    filtered_data = df.loc[df[y_col] < y_col_mean + 2 * y_col_std]
    filtered_data = filtered_data.loc[filtered_data[y_col] > y_col_mean - 2 * y_col_std]

    drop_dir = Path("output/drop")
    drop_predictions = extract_predictions(output_dir=drop_dir, aerogel_type="si")
    drop_predictions = drop_predictions[["Final Material", "error"]]
    error_std = drop_predictions["error"].std()
    error_mean = drop_predictions["error"].mean()
    low_error_aerogels = drop_predictions.loc[drop_predictions["error"] < error_mean + 2 * error_std]
    low_error_aerogels = low_error_aerogels.loc[low_error_aerogels["error"] > error_mean - 2 * error_std]
    low_error_aerogels = low_error_aerogels["Final Material"].tolist()

    # Filter raw data to only keep low error aerogels
    filtered_data = filtered_data.loc[filtered_data["Final Material"].isin(low_error_aerogels)]

    return filtered_data


if __name__ == "__main__":

    # Verify in test_data that y_column is set to 0
    training_data = read_excel(Path("backends/raw_si_aerogels.xlsx"), sheet_name="Comprehensive")
    test_data = read_csv(Path("backends/Si Aerogel Expt Recipe Trial 2_01.19.23.csv"))

    y_column = 'Surface Area (m2/g)'
    material_col = "Final Material"

    num_of_trials = 1
    validation_percent = 0.1
    n_hidden = list(range(0, 10))
    neurons = list(range(10, 200, 10))
    drop = list(arange(0.15, 0.4, 0.02))
    epochs = [100]
    param_grid = {
        'n_hidden': n_hidden,
        "neurons": neurons,
        'drop': drop,
    }

    drop_columns = ['Porosity', 'Porosity (%)', 'Pore Volume (cm3/g)', 'Average Pore Diameter (nm)',
                    'Bulk Density (g/cm3)', 'Young Modulus (MPa)', 'Crystalline Phase',
                    'Average Pore Size (nm)', 'Thermal Conductivity (W/mK)', 'Gelation Time (mins)']

    all_data = concat([training_data, test_data], ignore_index=True, axis=0)
    all_data = fetch_si_ml_dataset(additional_drop_columns=drop_columns, input_data=all_data)

    all_data = all_data.dropna(subset=[y_column])
    all_data = all_data.loc[all_data['Final Gel Type'] == "Aerogel"]
    all_data = all_data.drop(columns=['Final Gel Type', "Title"])

    # Filter data, remove outliers
    test_data = all_data.tail(len(test_data))
    training_data = all_data.drop(all_data.tail(len(test_data)).index)
    training_data = filter_data(training_data, y_column)

    test_materials = test_data["Final Material"]

    training_data = training_data.drop(columns=["Final Material"])
    test_data = test_data.drop(columns=["Final Material"])

    all_data = concat([training_data, test_data], ignore_index=True, axis=0)
    all_data = featurize(all_data, paper_id_column=None, bit_size=128)

    test_data = all_data.tail(len(test_data))
    test_features = test_data.drop(columns=[y_column]).to_numpy()
    training_data = all_data.drop(all_data.tail(len(test_data)).index)

    # Spilt up data
    val_df = training_data.sample(frac=validation_percent)
    training_data = training_data.drop(val_df.index)

    # Separate Features and Target
    train_target = training_data[y_column].to_numpy()
    val_target = val_df[y_column].to_numpy()
    train_features = training_data.drop(columns=[y_column]).to_numpy()
    val_features = val_df.drop(columns=[y_column]).to_numpy()

    # Scale Features
    feature_scaler = StandardScaler()
    train_features = feature_scaler.fit_transform(train_features)
    val_features = feature_scaler.transform(val_features)

    # Scale Target
    target_scaler = StandardScaler()
    train_target = target_scaler.fit_transform(train_target.reshape(-1, 1))
    val_target = target_scaler.transform(val_target.reshape(-1, 1))

    best_params = tune(train_features, train_target, val_features, val_target, epochs=epochs,
                       n_hidden_layers=n_hidden, neurons=neurons, dropouts=drop,
                       num_of_trials=num_of_trials)

    predictions = DataFrame()
    for j in range(5):
        estimator = build_estimator(best_params)
        estimator.fit(train_features, train_target, epochs=best_params["epochs"])
        predictions_j = estimator.predict(test_features)
        predictions_j = target_scaler.inverse_transform(predictions_j.reshape(-1, 1)).reshape(-1, )
        predictions[f"predictions_{j}"] = predictions_j

    predictions["Final Material"] = test_materials

    predictions.to_csv("output.csv")
