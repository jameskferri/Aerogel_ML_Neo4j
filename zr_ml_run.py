from machine_learning.run import run


if __name__ == "__main__":

    # General Properties for Machine Learning
    num_of_trials = 30  # For hyper-tuning
    train_percent = 0.8
    validation_percent = 0.1
    cycles = 10  # Number of times to cycle through all test set

    run(aerogel_type="zr",
        cycles=cycles,
        num_of_trials=num_of_trials,
        train_percent=train_percent,
        validation_percent=validation_percent)
