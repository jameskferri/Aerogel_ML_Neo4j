from pathlib import Path
from os import urandom
from os import mkdir
from os.path import exists

from machine_learning.run import train_model
from machine_learning.featurize import featurize
from backends.data_cleanup import fetch_si_ml_dataset
from neo4j_backends.predictions import extract_predictions


if __name__ == "__main__":

    # Model dir to save models
    model_dir = Path("model")
    trials = 30

    if exists(model_dir):
        raise OSError("model directory already exists")
    mkdir(model_dir)

    # grab raw data, remove columns that are known after sample is made
    si_drop_columns = ['Porosity', 'Porosity (%)', 'Pore Volume (cm3/g)', 'Average Pore Diameter (nm)',
                       'Bulk Density (g/cm3)', 'Young Modulus (MPa)', 'Crystalline Phase',
                       'Average Pore Size (nm)', 'Thermal Conductivity (W/mK)', 'Gelation Time (mins)']
    raw_data = fetch_si_ml_dataset(additional_drop_columns=si_drop_columns)

    y_column = 'Surface Area (m2/g)'
    material_col = "Final Material"
    seed = None
    test_split = 0.1

    # Generating training data
    raw_data = raw_data.dropna(subset=[y_column])
    raw_data = raw_data.loc[raw_data['Final Gel Type'] == "Aerogel"]
    raw_data = raw_data.drop(columns=['Final Gel Type'])

    raw_data = raw_data.sample(frac=1, random_state=seed)
    raw_data = raw_data.reset_index(drop=True)
    end_split = int(len(raw_data) * test_split)

    # Train the control model
    mkdir(model_dir / "control")
    train_model(raw_data, y_column, material_col, aerogel_type="si", num_of_trials=10,
                start_test_split=0, end_test_split=end_split, validation_percent=0.1,
                seed=seed, working_dir=model_dir / "control", save_model=True, zip_dir=False)

    raw_data = raw_data.sample(frac=1, random_state=seed)
    raw_data = raw_data.reset_index(drop=True)
    end_split = int(len(raw_data) * test_split)

    # Fetch aerogels that are in no_outliers dataset
    no_out_dir = Path("output/no_outliers")
    no_out_predictions = extract_predictions(output_dir=no_out_dir, aerogel_type="si")
    no_out_gels = no_out_predictions["Final Material"].tolist()
    no_out_df = raw_data.loc[raw_data["Final Material"].isin(no_out_gels)]

    # Train no outliers model
    mkdir(model_dir / "no_outliers")
    train_model(no_out_df, y_column, material_col, aerogel_type="si", num_of_trials=10,
                start_test_split=0, end_test_split=end_split, validation_percent=0.1,
                seed=seed, working_dir=model_dir / "no_outliers", save_model=True, zip_dir=False)

    raw_data = raw_data.sample(frac=1, random_state=seed)
    raw_data = raw_data.reset_index(drop=True)
    end_split = int(len(raw_data) * test_split)

    # Fetch aerogels that are in no_outliers dataset
    drop_dir = Path("output/drop")
    drop_predictions = extract_predictions(output_dir=drop_dir, aerogel_type="si")
    drop_gels = drop_predictions["Final Material"].tolist()
    drop_df = raw_data.loc[raw_data["Final Material"].isin(drop_gels)]

    # Train drop model
    mkdir(model_dir / "drop")
    train_model(drop_df, y_column, material_col, aerogel_type="si", num_of_trials=10,
                start_test_split=0, end_test_split=end_split, validation_percent=0.1,
                seed=seed, working_dir=model_dir / "drop", save_model=True, zip_dir=False)


