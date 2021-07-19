""" Visualiser for our Predictor class """


def visualise(model, x_test, y_test, scaler, plt):
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_facecolor('#000041')
    ax.plot(y_test_scaled, color='red', label='Original price')
    plt.plot(predictions, color='cyan', label='Predicted price')
    plt.legend()

