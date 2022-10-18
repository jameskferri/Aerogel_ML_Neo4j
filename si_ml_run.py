from pathlib import Path

from machine_learning.run import run


if __name__ == "__main__":

    # General Properties for Machine Learning
    num_of_trials = 30  # For hyper-tuning
    train_percent = 0.8
    validation_percent = 0.1

    # Number of times to cycle through all test sets for each filter option
    # This number should be equal to or greater than 2
    cycles = 10

    # Options to filter papers out
    # Control - All papers
    # Drop - Remove based on statistical outliers
    # No Outliers - Remove high error based on results from drop
    filter_options = ["control", "no_outliers", "drop"]

    output_dir = Path("output")

    run(aerogel_type="si",
        cycles=cycles,
        num_of_trials=num_of_trials,
        train_percent=train_percent,
        validation_percent=validation_percent,
        filter_options=filter_options,
        output_dir=output_dir)
