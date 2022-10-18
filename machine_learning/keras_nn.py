from random import choices, choice

from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense


def build_estimator(params):

    """
    Build a model object from keras from a dictionary of parameters formatted as

    params = dict(
        epochs=epochs_i,
        n_hidden_layers=n_hidden_layers_i,
        n_neurons=neurons_i,
        dropout=dropouts_i,
        )

    Where the variable types in the param dict values are
    epochs_i: int
    n_hidden_layers_i: int
    neurons_i: list[int]    |   Same length as n_hidden_layers_i
    dropouts_i: list[int]   |   Same length as n_hidden_layers_i

    Defaults in model builder
    hidden layer activation = relu
    output layer activation = linear
    loss = mse
    metrics = mse, mae
    optimizer = Adam

    :param params:
    :return:
    """

    model = Sequential()
    for i in range(params["n_hidden_layers"]):
        model.add(
            Dense(
                kernel_initializer='normal',
                units=params["n_neurons"][i],
                activation='relu',
            )
        )
        if i < 4:
            model.add(
                Dropout(params["dropout"][i])
            )

    model.add(Dense(1))
    model.compile(
        optimizer="Adam",
        loss="mse",
        metrics=["mse", "mae"],
    )
    return model


def tune(train_features, train_target, val_features, val_target, epochs,
         n_hidden_layers, neurons, dropouts, num_of_trials):

    models = {}
    val_mses = []
    for i in range(num_of_trials):

        if val_mses:
            min_val_mse = min(val_mses)
        else:
            min_val_mse = None
        print(f"Trial #{i} | Best Validation MSE: {min_val_mse}")

        epochs_i = choice(epochs)
        n_hidden_layers_i = choice(n_hidden_layers)

        neurons_i = []
        dropouts_i = []
        for _ in range(n_hidden_layers_i):
            neurons_i.append(choice(neurons))
            dropouts_i.append(choice(dropouts))

        params = dict(
            epochs=epochs_i,
            n_hidden_layers=n_hidden_layers_i,
            n_neurons=neurons_i,
            dropout=dropouts_i,
        )

        estimator = build_estimator(params)
        estimator.fit(train_features, train_target, epochs=epochs_i, validation_data=(val_features, val_target))
        val_predictions_i = estimator.predict(val_features)
        val_mse = mean_squared_error(val_target.reshape(-1, ), val_predictions_i.reshape(-1, ))
        val_mses.append(val_mse)

        models[val_mse] = params

    # Return best model
    return models[min(models.keys())]
