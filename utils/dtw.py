__author__ = 'Brian Iwana' # Original author of the code

import numpy as np
import math
import sys
from typing import Tuple, Optional, List

# --- Constants for return_flag parameter ---
RETURN_VALUE: int = 0  # Return only the DTW distance value
RETURN_PATH: int = 1   # Return only the optimal alignment path
RETURN_ALL: int = -1   # Return DTW distance, cost matrix, cumulative matrix, and path


def _traceback(DTW: np.ndarray, slope_constraint: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstructs the optimal alignment path from the cumulative cost matrix.

    Args:
        DTW (np.ndarray): The cumulative cost matrix.
        slope_constraint (str): The slope constraint used ('asymmetric' or 'symmetric').

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays,
                                        representing the row and column indices of the optimal path.
    """
    i, j = np.array(DTW.shape) - 1 # Start from the bottom-right corner of the cumulative matrix
    p, q = [i - 1], [j - 1]       # Store path indices (adjusted for 0-based original series)

    if slope_constraint == "asymmetric":
        # Asymmetric constraint allows steps (1,1), (1,0), (1,2) in the cost matrix
        # (meaning from DTW[i-1, j-1], DTW[i-1, j], DTW[i-1, j+1] going backwards from the query (prototype))
        # This implementation uses the equivalent:
        # (i-1, j-1), (i-1, j), (i-1, j-2) going backward from target (sample)
        while i > 1: # Loop until the start of the prototype (row 1 in DTW matrix)
            # Find the minimum of the three possible predecessors
            tb = np.argmin((DTW[i - 1, j], DTW[i - 1, j - 1], DTW[i - 1, j - 2]))

            if tb == 0: # Came from (i-1, j) -> step (1,0) backwards
                i = i - 1
            elif tb == 1: # Came from (i-1, j-1) -> step (1,1) backwards
                i = i - 1
                j = j - 1
            elif tb == 2: # Came from (i-1, j-2) -> step (1,2) backwards
                i = i - 1
                j = j - 2

            p.insert(0, i - 1) # Insert at the beginning to maintain correct order
            q.insert(0, j - 1)
    elif slope_constraint == "symmetric":
        # Symmetric constraint allows steps (1,1), (1,0), (0,1)
        while i > 1 or j > 1: # Loop until the start of either series (row/col 1 in DTW matrix)
            # Find the minimum of the three possible predecessors
            tb = np.argmin((DTW[i - 1, j - 1], DTW[i - 1, j], DTW[i, j - 1]))

            if tb == 0: # Came from (i-1, j-1) -> step (1,1) backwards
                i = i - 1
                j = j - 1
            elif tb == 1: # Came from (i-1, j) -> step (1,0) backwards
                i = i - 1
            elif tb == 2: # Came from (i, j-1) -> step (0,1) backwards
                j = j - 1

            p.insert(0, i - 1)
            q.insert(0, j - 1)
    else:
        sys.exit(f"Error: Unknown slope constraint '{slope_constraint}'")

    return np.array(p), np.array(q)


def dtw(prototype: np.ndarray, sample: np.ndarray, return_flag: int = RETURN_VALUE,
        slope_constraint: str = "asymmetric", window: Optional[int] = None) -> \
        Tuple[np.ndarray, np.ndarray] or np.ndarray or float:
    """
    Computes the Dynamic Time Warping (DTW) distance and optionally the optimal alignment path
    between two time series.

    Args:
        prototype (np.ndarray): The first time series (query), shape (seq_len_p, num_features).
        sample (np.ndarray): The second time series (target), shape (seq_len_s, num_features).
        return_flag (int): Determines what to return (RETURN_VALUE, RETURN_PATH, RETURN_ALL).
        slope_constraint (str): The slope constraint to apply ('asymmetric' or 'symmetric').
        window (Optional[int]): The Sakoe-Chiba band window size. If None, no window constraint is applied.

    Returns:
        Union[float, Tuple[np.ndarray, np.ndarray], Tuple[float, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
            The DTW distance, or the path, or all calculated matrices and path.
    """
    p_len = prototype.shape[0]
    assert p_len != 0, "Prototype time series is empty!"
    s_len = sample.shape[0]
    assert s_len != 0, "Sample time series is empty!"

    if window is None:
        window = s_len # Default to no window constraint

    # Calculate local cost matrix
    # cost[i,j] stores the distance between prototype[i] and sample[j]
    cost = np.full((p_len, s_len), np.inf)
    for i in range(p_len):
        # Apply Sakoe-Chiba band window constraint
        start = max(0, i - window)
        end = min(s_len, i + window + 1)
        # Compute Euclidean distance (L2 norm) between slices of series
        cost[i, start:end] = np.linalg.norm(sample[start:end] - prototype[i], axis=1)

    # Compute the cumulative cost matrix
    DTW_cumulative = _cummulative_matrix(cost, slope_constraint, window)

    if return_flag == RETURN_ALL:
        # Return total distance, local cost matrix, cumulative DTW matrix (excluding padding), and path
        return DTW_cumulative[-1, -1], cost, DTW_cumulative[1:, 1:], _traceback(DTW_cumulative, slope_constraint)
    elif return_flag == RETURN_PATH:
        # Return only the optimal alignment path
        return _traceback(DTW_cumulative, slope_constraint)
    else:
        # Default: return only the total DTW distance
        return DTW_cumulative[-1, -1]


def _cummulative_matrix(cost: np.ndarray, slope_constraint: str, window: Optional[int]) -> np.ndarray:
    """
    Computes the cumulative cost (DTW) matrix based on the local cost matrix and slope constraint.

    Args:
        cost (np.ndarray): The local cost matrix, shape (seq_len_p, seq_len_s).
        slope_constraint (str): The slope constraint ('asymmetric' or 'symmetric').
        window (Optional[int]): The Sakoe-Chiba band window size.

    Returns:
        np.ndarray: The cumulative cost matrix, shape (seq_len_p + 1, seq_len_s + 1).
    """
    p_len, s_len = cost.shape
    
    # Initialize cumulative DTW matrix with infinity, and DTW[0,0] to 0.0
    # The matrix is one larger than the cost matrix in each dimension for padding.
    DTW_cumulative = np.full((p_len + 1, s_len + 1), np.inf)
    DTW_cumulative[0, 0] = 0.0

    if slope_constraint == "asymmetric":
        # Recurrence relation for asymmetric constraint:
        # DTW[i,j] = cost[i-1,j-1] + min(DTW[i-1,j], DTW[i-1,j-1], DTW[i-1,j-2])
        for i in range(1, p_len + 1):
            # Apply Sakoe-Chiba band: iterate only within the window
            for j in range(max(1, i - window), min(s_len + 1, i + window + 1)):
                if j == 1 and i <= window: # Special case for the first column within window
                    DTW_cumulative[i, j] = cost[i - 1, j - 1] + min(DTW_cumulative[i - 1, j], DTW_cumulative[i - 1, j - 1])
                else:
                    DTW_cumulative[i, j] = cost[i - 1, j - 1] + min(DTW_cumulative[i - 1, j - 2] if j - 2 >= 0 else np.inf, # (1,2) step
                                                                   DTW_cumulative[i - 1, j - 1],                       # (1,1) step
                                                                   DTW_cumulative[i - 1, j])                           # (1,0) step
    elif slope_constraint == "symmetric":
        # Recurrence relation for symmetric constraint:
        # DTW[i,j] = cost[i-1,j-1] + min(DTW[i-1,j-1], DTW[i,j-1], DTW[i-1,j])
        for i in range(1, p_len + 1):
            # Apply Sakoe-Chiba band: iterate only within the window
            for j in range(max(1, i - window), min(s_len + 1, i + window + 1)):
                DTW_cumulative[i, j] = cost[i - 1, j - 1] + min(DTW_cumulative[i - 1, j - 1],  # Diagonal step
                                                               DTW_cumulative[i, j - 1],      # Horizontal step
                                                               DTW_cumulative[i - 1, j])       # Vertical step
    else:
        sys.exit(f"Error: Unknown slope constraint '{slope_constraint}'")

    return DTW_cumulative


def shape_dtw(prototype: np.ndarray, sample: np.ndarray, return_flag: int = RETURN_VALUE,
              slope_constraint: str = "asymmetric", window: Optional[int] = None, descr_ratio: float = 0.05) -> \
        Tuple[np.ndarray, np.ndarray] or np.ndarray or float:
    """
    Computes the Shape Dynamic Time Warping (ShapeDTW) distance and optionally the optimal alignment path
    between two time series.

    ShapeDTW aligns sequences based on shape similarity rather than point-wise values.
    It does this by comparing short "descriptor" windows extracted from each time point.
    Based on: https://www.sciencedirect.com/science/article/pii/S0031320317303710

    Args:
        prototype (np.ndarray): The first time series (query), shape (seq_len_p, num_features).
        sample (np.ndarray): The second time series (target), shape (seq_len_s, num_features).
        return_flag (int): Determines what to return (RETURN_VALUE, RETURN_PATH, RETURN_ALL).
        slope_constraint (str): The slope constraint to apply ('asymmetric' or 'symmetric').
        window (Optional[int]): The Sakoe-Chiba band window size. If None, no window constraint is applied.
        descr_ratio (float): The ratio of sequence length to determine the descriptor window size.

    Returns:
        Union[float, Tuple[np.ndarray, np.ndarray], Tuple[float, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
            The ShapeDTW distance, or the path, or all calculated matrices and path.
    """
    p_len = prototype.shape[0]
    assert p_len != 0, "Prototype time series is empty!"
    s_len = sample.shape[0]
    assert s_len != 0, "Sample time series is empty!"

    if window is None:
        window = s_len # Default to no window constraint
        
    # Determine descriptor window lengths for prototype and sample
    p_feature_len = np.clip(np.round(p_len * descr_ratio), 5, 100).astype(int)
    s_feature_len = np.clip(np.round(s_len * descr_ratio), 5, 100).astype(int)
    
    # Pad both time series to handle descriptor windows at the beginning/end
    p_pad_front = (np.ceil(p_feature_len / 2.)).astype(int)
    p_pad_back = (np.floor(p_feature_len / 2.)).astype(int)
    s_pad_front = (np.ceil(s_feature_len / 2.)).astype(int)
    s_pad_back = (np.floor(s_feature_len / 2.)).astype(int)
    
    # Apply edge padding to effectively create windows for every point
    prototype_pad = np.pad(prototype, ((p_pad_front, p_pad_back), (0, 0)), mode="edge")
    sample_pad = np.pad(sample, ((s_pad_front, s_pad_back), (0, 0)), mode="edge")
    
    # Calculate local cost matrix by comparing shape descriptors (windows)
    cost = np.full((p_len, s_len), np.inf)
    for i in range(p_len):
        # Apply Sakoe-Chiba band window constraint for efficiency
        for j in range(max(0, i - window), min(s_len, i + window + 1)):
            # Local cost is the Euclidean distance between shape descriptors (windows)
            cost[i, j] = np.linalg.norm(sample_pad[j : j + s_feature_len] - prototype_pad[i : i + p_feature_len])
            
    # Compute the cumulative cost matrix
    DTW_cumulative = _cummulative_matrix(cost, slope_constraint=slope_constraint, window=window)
    
    if return_flag == RETURN_ALL:
        # Return total distance, local cost matrix, cumulative DTW matrix (excluding padding), and path
        return DTW_cumulative[-1, -1], cost, DTW_cumulative[1:, 1:], _traceback(DTW_cumulative, slope_constraint)
    elif return_flag == RETURN_PATH:
        # Return only the optimal alignment path
        return _traceback(DTW_cumulative, slope_constraint)
    else:
        # Default: return only the total ShapeDTW distance
        return DTW_cumulative[-1, -1]
    
# --- Visualization Helper Functions (requires matplotlib) ---

def draw_graph2d(cost: np.ndarray, DTW: np.ndarray, path: Tuple[np.ndarray, np.ndarray],
                 prototype: np.ndarray, sample: np.ndarray):
    """
    Visualizes the DTW alignment for 2-dimensional time series.

    Args:
        cost (np.ndarray): Local cost matrix.
        DTW (np.ndarray): Cumulative DTW matrix.
        path (Tuple[np.ndarray, np.ndarray]): Optimal alignment path (row_indices, col_indices).
        prototype (np.ndarray): The first time series (2D).
        sample (np.ndarray): The second time series (2D).
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))

    # Subplot 1: Local Cost Matrix with optimal path
    plt.subplot(2, 3, 1)
    plt.imshow(cost.T, cmap=plt.cm.gray, interpolation='none', origin='lower')
    plt.plot(path[0], path[1], 'y')
    plt.title('Local Cost Matrix & Path')
    plt.xlim((-0.5, cost.shape[0] - 0.5))
    plt.ylim((-0.5, cost.shape[1] - 0.5))

    # Subplot 2: Cumulative DTW Matrix with optimal path
    plt.subplot(2, 3, 2)
    plt.imshow(DTW.T, cmap=plt.cm.gray, interpolation='none', origin='lower')
    # path indices need to be adjusted for 1-based indexing of DTW matrix
    plt.plot(path[0] + 1, path[1] + 1, 'y')
    plt.title('Cumulative DTW Matrix & Path')
    plt.xlim((-0.5, DTW.shape[0] - 0.5))
    plt.ylim((-0.5, DTW.shape[1] - 0.5))

    # Subplot 3 (position 4): Prototype series
    plt.subplot(2, 3, 4)
    plt.plot(prototype[:, 0], prototype[:, 1], 'b-o')
    plt.title('Prototype (Blue)')

    # Subplot 4 (position 5): Alignment connection
    plt.subplot(2, 3, 5)
    for i in range(path[0].shape[0]):
        plt.plot([prototype[path[0][i], 0], sample[path[1][i], 0]],
                 [prototype[path[0][i], 1], sample[path[1][i], 1]], 'y-') # Yellow lines connecting aligned points
    plt.plot(sample[:, 0], sample[:, 1], 'g-o') # Sample series
    plt.plot(prototype[:, 0], prototype[:, 1], 'b-o') # Prototype series
    plt.title('Alignment')

    # Subplot 5 (position 6): Sample series
    plt.subplot(2, 3, 6)
    plt.plot(sample[:, 0], sample[:, 1], 'g-o')
    plt.title('Sample (Green)')

    plt.tight_layout()
    plt.show()


def draw_graph1d(cost: np.ndarray, DTW: np.ndarray, path: Tuple[np.ndarray, np.ndarray],
                 prototype: np.ndarray, sample: np.ndarray):
    """
    Visualizes the DTW alignment for 1-dimensional time series.

    Args:
        cost (np.ndarray): Local cost matrix.
        DTW (np.ndarray): Cumulative DTW matrix.
        path (Tuple[np.ndarray, np.ndarray]): Optimal alignment path (row_indices, col_indices).
        prototype (np.ndarray): The first time series (1D or 2D where second dim is 1).
        sample (np.ndarray): The second time series (1D or 2D where second dim is 1).
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    p_steps = np.arange(prototype.shape[0])
    s_steps = np.arange(sample.shape[0])

    # Subplot 1: Local Cost Matrix with optimal path
    plt.subplot(2, 3, 1)
    plt.imshow(cost.T, cmap=plt.cm.gray, interpolation='none', origin='lower')
    plt.plot(path[0], path[1], 'y')
    plt.title('Local Cost Matrix & Path')
    plt.xlim((-0.5, cost.shape[0] - 0.5))
    plt.ylim((-0.5, cost.shape[1] - 0.5))

    # Subplot 2: Cumulative DTW Matrix with optimal path
    plt.subplot(2, 3, 2)
    plt.imshow(DTW.T, cmap=plt.cm.gray, interpolation='none', origin='lower')
    # path indices need to be adjusted for 1-based indexing of DTW matrix
    plt.plot(path[0] + 1, path[1] + 1, 'y')
    plt.title('Cumulative DTW Matrix & Path')
    plt.xlim((-0.5, DTW.shape[0] - 0.5))
    plt.ylim((-0.5, DTW.shape[1] - 0.5))

    # Subplot 3 (position 4): Prototype series
    plt.subplot(2, 3, 4)
    plt.plot(p_steps, prototype[:, 0], 'b-o')
    plt.title('Prototype (Blue)')

    # Subplot 4 (position 5): Alignment connection
    plt.subplot(2, 3, 5)
    for i in range(path[0].shape[0]):
        plt.plot([path[0][i], path[1][i]], # X-coordinates: prototype index, sample index
                 [prototype[path[0][i], 0], sample[path[1][i], 0]], 'y-') # Y-coordinates: prototype value, sample value
    plt.plot(s_steps, sample[:, 0], 'g-o') # Sample series
    plt.plot(p_steps, prototype[:, 0], 'b-o') # Prototype series
    plt.title('Alignment')

    # Subplot 5 (position 6): Sample series
    plt.subplot(2, 3, 6)
    plt.plot(s_steps, sample[:, 0], 'g-o')
    plt.title('Sample (Green)')

    plt.tight_layout()
    plt.show()
