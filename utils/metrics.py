import numpy as np


def RSE(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Calculates the Root Relative Squared Error (RSE).

    RSE measures the error of the forecast relative to a simple predictor
    (the mean of the true values). A value < 1 indicates the model is better
    than the mean predictor.

    Formula: sqrt(sum((true - pred)^2)) / sqrt(sum((true - mean(true))^2))

    Args:
        pred (np.ndarray): Predicted values.
        true (np.ndarray): Ground truth values.

    Returns:
        float: The RSE value.
    """
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Calculates the Pearson Correlation Coefficient.

    CORR measures the linear relationship between predicted and true values.
    A value of 1 indicates a perfect positive linear relationship, -1 a perfect
    negative relationship, and 0 no linear relationship.

    Args:
        pred (np.ndarray): Predicted values.
        true (np.ndarray): Ground truth values.

    Returns:
        float: The Pearson correlation coefficient.
    """
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    # Add a small epsilon to the denominator to prevent division by zero
    return (u / (d + 1e-9)).mean()


def MAE(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Calculates the Mean Absolute Error (MAE).

    Formula: mean(abs(true - pred))

    Args:
        pred (np.ndarray): Predicted values.
        true (np.ndarray): Ground truth values.

    Returns:
        float: The MAE value.
    """
    return np.mean(np.abs(true - pred))


def MSE(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Calculates the Mean Squared Error (MSE).

    Formula: mean((true - pred)^2)

    Args:
        pred (np.ndarray): Predicted values.
        true (np.ndarray): Ground truth values.

    Returns:
        float: The MSE value.
    """
    return np.mean((true - pred) ** 2)


def RMSE(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Calculates the Root Mean Squared Error (RMSE).

    Formula: sqrt(mean((true - pred)^2))

    Args:
        pred (np.ndarray): Predicted values.
        true (np.ndarray): Ground truth values.

    Returns:
        float: The RMSE value.
    """
    return np.sqrt(MSE(pred, true))


def MAPE(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Calculates the Mean Absolute Percentage Error (MAPE).

    Formula: mean(abs((true - pred) / true))

    Note: This metric can be sensitive to zero or near-zero true values,
          which can lead to 'inf' or very large results.

    Args:
        pred (np.ndarray): Predicted values.
        true (np.ndarray): Ground truth values.

    Returns:
        float: The MAPE value.
    """
    # Add a small epsilon to the denominator to prevent division by zero
    return np.mean(np.abs((pred - true) / (true + 1e-9)))


def MSPE(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Calculates the Mean Squared Percentage Error (MSPE).

    Formula: mean(((true - pred) / true)^2)

    Note: This metric can be sensitive to zero or near-zero true values,
          which can lead to 'inf' or very large results.

    Args:
        pred (np.ndarray): Predicted values.
        true (np.ndarray): Ground truth values.

    Returns:
        float: The MSPE value.
    """
    # Add a small epsilon to the denominator to prevent division by zero
    return np.mean(np.square((pred - true) / (true + 1e-9)))


def metric(pred: np.ndarray, true: np.ndarray) -> tuple[float, float, float, float, float]:
    """
    A convenience function that computes and returns multiple standard evaluation metrics.

    Args:
        pred (np.ndarray): Predicted values.
        true (np.ndarray): Ground truth values.

    Returns:
        tuple[float, float, float, float, float]: A tuple containing the following metrics in order:
                                                   (MAE, MSE, RMSE, MAPE, MSPE).
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe