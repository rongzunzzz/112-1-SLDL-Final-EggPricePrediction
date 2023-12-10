import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def train(model, X_train, y_train):
    model.fit(X_train, y_train)

    return model 

    
def evaluate(model, X_test, y_test, show_best=False):
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)

    rmse = np.sqrt(mse)

    mspe = np.mean(np.square((y_test - predictions) / y_test)) * 100

    if show_best: # used when fine-tuning, only show the result of the best params(model)
        print(f"Mean Squared Error: {mse}")
        print(f"Root Mean Squared Error: {rmse}")
        print(f"Mean Squared Percentage Error: {mspe}%")

        # Plotting the line chart for the best predictions
        x = np.array([i+1 for i in range(len(y_test))])
        plt.figure(figsize=(10, 6))
        plt.plot(x, predictions, label='Predictions')
        plt.plot(x, y_test, label='True Prices')
        plt.title('Best Predictions vs True Prices')
        plt.xlabel('Dates from Earliest to Latest')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    return mse, rmse, mspe


def save_model(year, model, model_name):
    model_folder_path = f'./models'
    model_filename = f'{model_folder_path}/{year}/{model_name}.joblib'
    joblib.dump(model, model_filename)
