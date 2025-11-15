"""
Metrics for evaluating machine learning models.

- Regression Metrics: MAE, MSE, RMSE, MAPE
"""

import numpy as np


# Regression metrics

def mean_absolute_error(a: np.ndarray, b: np.ndarray) -> np.float64:

    return np.mean(np.abs(a - b), dtype=np.float64)


def mean_squared_error(a: np.ndarray, b: np.ndarray) -> np.float64:

    return np.mean(np.square(a - b), dtype=np.float64)


def root_mean_squared_error(a: np.ndarray, b: np.ndarray) -> np.float64:

    return np.sqrt(np.mean(np.square(a - b)), dtype=np.float64)


def mean_abs_percentage_error(a: np.ndarray, b: np.ndarray) -> np.float64:

    return np.mean(
        np.multiply(np.divide(np.abs(a - b), a), 100), dtype=np.float64
    )


# Example usage:
if __name__ == "__main__":
    # Generate some sample data of 1d points
    a = np.random.random(10)
    b = np.random.random(10)

    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(a, b)

    # Mean Squared Error (MSE)
    mse = mean_squared_error(a, b)

    # Root Mean Squared Error (RMSE)
    rmse = root_mean_squared_error(a, b)

    # Mean Absolute Percentage Error (MAPE)
    mape = mean_abs_percentage_error(a, b)

    print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, MAPE: {mape}")
