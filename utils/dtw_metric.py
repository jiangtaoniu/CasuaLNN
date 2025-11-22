from numpy import array, zeros, full, argmin, inf, ndim
from scipy.spatial.distance import cdist
from math import isinf
from typing import Callable, Union, Tuple, List, Optional


def _traceback(D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstructs the optimal alignment path from the cumulative cost matrix D.

    This internal traceback function is specific to the DTW implementations in this file.
    It assumes D is a cumulative cost matrix with an extra row/column of infinity padding
    at the top/left, and traces back from D[-1,-1] to D[0,0].

    Args:
        D (np.ndarray): The cumulative cost matrix, typically (r+1, c+1) where r, c are lengths of sequences.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays,
                                        representing the row and column indices of the optimal path.
    """
    # Start from the bottom-right corner of the cumulative matrix (excluding padding row/col)
    i, j = array(D.shape) - 2
    p, q = [i], [j] # Lists to store path indices

    # Trace back until the start of either sequence
    while i > 0 or j > 0:
        # Find the minimum among the three possible predecessors:
        # 1. (i, j) - diagonal step from (i-1, j-1)
        # 2. (i, j+1) - vertical step from (i-1, j)
        # 3. (i+1, j) - horizontal step from (i, j-1)
        # Note: The indices are relative to the D matrix where D[i, j] refers to cost at (i-1, j-1) in actual sequences
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:  # Came from D[i, j] (diagonal)
            i -= 1
            j -= 1
        elif tb == 1:  # Came from D[i, j+1] (vertical)
            i -= 1
        else:  # tb == 2 (Came from D[i+1, j]) (horizontal)
            j -= 1
        p.insert(0, i) # Insert at the beginning to maintain correct order
        q.insert(0, j)
    return array(p), array(q)


def dtw(x: np.ndarray, y: np.ndarray, dist: Callable[[np.ndarray, np.ndarray], float],
        warp: int = 1, w: Union[float, int] = inf, s: float = 1.0) -> \
        Tuple[float, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    This implementation allows for a custom distance function for local cost,
    a 'warp' parameter for controlling step size, a Sakoe-Chiba band window 'w',
    and a slope weight 's' to bias the path towards the diagonal.

    Args:
        x (np.ndarray): First sequence, shape (N1, M) where N1 is length and M is feature dimension.
        y (np.ndarray): Second sequence, shape (N2, M) where N2 is length and M is feature dimension.
        dist (Callable): Distance function used as cost measure between two points (e.g., Euclidean, Manhattan).
                          It should take two 1D arrays (points) and return a scalar distance.
        warp (int): Controls the maximum "warp step" size (number of consecutive diagonal, horizontal, or vertical steps).
                    A value of 1 implies standard DTW with unit steps.
        w (Union[float, int]): Window size limiting the maximal distance between indices of matched entries |i,j|.
                                If `inf`, no window constraint is applied.
        s (float): Weight applied on off-diagonal moves of the path. As 's' gets larger,
                   the warping path is increasingly biased towards the diagonal.

    Returns:
        Tuple[float, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
            - minimum_distance (float): The final DTW distance.
            - cost_matrix (np.ndarray): The local cost matrix.
            - accumulated_cost_matrix (np.ndarray): The accumulated cost matrix (D1).
            - wrap_path (Tuple[np.ndarray, np.ndarray]): The optimal alignment path.
    """
    assert len(x) > 0 and len(y) > 0, "Input sequences cannot be empty."
    assert isinf(w) or (w >= abs(len(x) - len(y))), "Window size 'w' must be >= abs(len(x) - len(y)) or inf."
    assert s > 0, "Slope weight 's' must be positive."

    r, c = len(x), len(y) # Lengths of sequences x and y

    # D0: Global cost matrix with padding for initialization (r+1, c+1)
    if not isinf(w):
        # Initialize D0 with infinity, then set a band of zeros based on window 'w'
        D0 = full((r + 1, c + 1), inf)
        for i in range(1, r + 1):
            D0[i, max(1, i - w):min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
    else:
        # No window constraint: standard initialization for global cost matrix
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf # First row (padding) except D0[0,0]
        D0[1:, 0] = inf # First column (padding) except D0[0,0]
    
    # D1: View of D0 that corresponds to the actual accumulated cost matrix (r, c)
    D1 = D0[1:, 1:]
    
    # C: Local cost matrix, storing point-wise distances
    C = zeros((r, c)) # Initialize local cost matrix
    for i in range(r):
        for j in range(c):
            # Calculate local cost only within the defined window 'w'
            if (isinf(w) or (max(0, i - w) <= j <= min(c, i + w))):
                C[i, j] = dist(x[i], y[j]) # Use provided distance function

    # Populate D1 (accumulated cost matrix)
    jrange_base = range(c)
    for i in range(r):
        jrange = jrange_base # Default to full range
        if not isinf(w):
            # Limit computation to the Sakoe-Chiba band
            jrange = range(max(0, i - w), min(c, i + w + 1))
        for j in jrange:
            min_list = [D0[i, j]] # Start with diagonal predecessor cost
            # Consider warp-step predecessors
            for k in range(1, warp + 1):
                i_k = min(i + k, r) # Ensure index doesn't exceed bounds
                j_k = min(j + k, c) # Ensure index doesn't exceed bounds
                min_list += [D0[i_k, j] * s, D0[i, j_k] * s] # Add scaled horizontal/vertical predecessors
            
            # Apply recurrence relation: D1[i,j] = C[i,j] + min(predecessors)
            D1[i, j] = C[i, j] + min(min_list)
    
    # Handle edge cases for path traceback with very short sequences
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0) # Call internal traceback function

    return D1[-1, -1], C, D1, path


def accelerated_dtw(x: np.ndarray, y: np.ndarray, dist: Union[str, Callable], warp: int = 1) -> \
        Tuple[float, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Computes Dynamic Time Warping (DTW) of two sequences using `scipy.spatial.distance.cdist`
    for faster local cost matrix calculation.

    Args:
        x (np.ndarray): First sequence, shape (N1, M) or (N1,).
        y (np.ndarray): Second sequence, shape (N2, M) or (N2,).
        dist (Union[str, Callable]): Distance metric for `cdist`. Can be a string (e.g., 'euclidean', 'sqeuclidean')
                                     or a callable distance function.
        warp (int): Controls the maximum "warp step" size (number of consecutive diagonal, horizontal, or vertical steps).

    Returns:
        Tuple[float, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
            - minimum_distance (float): The final DTW distance.
            - cost_matrix (np.ndarray): The local cost matrix (C).
            - accumulated_cost_matrix (np.ndarray): The accumulated cost matrix (D1).
            - wrap_path (Tuple[np.ndarray, np.ndarray]): The optimal alignment path.
    """
    assert len(x) > 0 and len(y) > 0, "Input sequences cannot be empty."

    # Reshape 1D inputs to 2D for cdist compatibility
    if ndim(x) == 1:
        x = x.reshape(-1, 1)
    if ndim(y) == 1:
        y = y.reshape(-1, 1)

    r, c = len(x), len(y) # Lengths of sequences x and y

    # D0: Global cost matrix with padding for initialization
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf # First row (padding) except D0[0,0]
    D0[1:, 0] = inf # First column (padding) except D0[0,0]
    
    # Calculate local cost matrix (C) using cdist for acceleration
    C = cdist(x, y, dist)
    
    # D1: View of D0 corresponding to actual sequence costs
    D1 = D0[1:, 1:]
    D1[:, :] = C # Copy local costs into D1

    # Populate D1 (accumulated cost matrix)
    for i in range(r):
        for j in range(c):
            min_list = [D0[i, j]] # Start with diagonal predecessor cost
            # Consider warp-step predecessors
            for k in range(1, warp + 1):
                min_list += [D0[min(i + k, r), j], # Vertical step with warp
                             D0[i, min(j + k, c)]]  # Horizontal step with warp
            
            # Apply recurrence relation: D1[i,j] = C[i,j] + min(predecessors)
            D1[i, j] += min(min_list)
    
    # Handle edge cases for path traceback with very short sequences
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0) # Call internal traceback function

    return D1[-1, -1], C, D1, path


if __name__ == '__main__':
    """
    Demonstration block for DTW usage and visualization.
    """
    w_val = inf # Window size
    s_val = 1.0 # Slope weight

    # --- Example 1: 1-D numeric sequences with Manhattan distance ---
    if 1:
        from sklearn.metrics.pairwise import manhattan_distances
        x_seq = np.array([0, 0, 1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
        y_seq = np.array([1, 1, 1, 2, 2, 2, 2, 3, 2, 0]).reshape(-1, 1)
        dist_func = lambda a, b: manhattan_distances(a.reshape(1, -1), b.reshape(1, -1))[0, 0]
        w_val = 1 # Small window for demonstration
        
        print("Running DTW with Manhattan Distance (1D):")
        dist_val, cost_mat, acc_mat, path_val = dtw(x_seq, y_seq, dist_func, w=w_val, s=s_val)

        # Visualize
        from matplotlib import pyplot as plt
        plt.imshow(cost_mat.T, origin='lower', cmap=plt.cm.Reds, interpolation='nearest')
        plt.plot(path_val[0], path_val[1], '-o')  # Optimal alignment path
        plt.xticks(range(len(x_seq)), x_seq.flatten())
        plt.yticks(range(len(y_seq)), y_seq.flatten())
        plt.xlabel('Sequence X')
        plt.ylabel('Sequence Y')
        plt.axis('tight')
        if isinf(w_val):
            plt.title(f'DTW: {dist_val:.2f}, slope weight: {s_val}')
        else:
            plt.title(f'DTW: {dist_val:.2f}, window width: {w_val}, slope weight: {s_val}')
        plt.show()

    # --- Other commented examples for different data types ---
    # elif 0:  # 2-D numeric
    #     from sklearn.metrics.pairwise import euclidean_distances
    #     x_seq = np.array([[0, 0], [0, 1], [1, 1], [1, 2], [2, 2], [4, 3], [2, 3], [1, 1], [2, 2], [0, 1]])
    #     y_seq = np.array([[1, 0], [1, 1], [1, 1], [2, 1], [4, 3], [4, 3], [2, 3], [3, 1], [1, 2], [1, 0]])
    #     dist_func = lambda a, b: euclidean_distances(a.reshape(1, -1), b.reshape(1, -1))[0, 0]
    #     dist_val, cost_mat, acc_mat, path_val = dtw(x_seq, y_seq, dist_func, w=w_val, s=s_val)
    #     # Visualization would be similar

    # else:  # 1-D list of strings with edit distance (requires NLTK)
    #     from nltk.metrics.distance import edit_distance
    #     x_seq = ['i', 'soon', 'found', 'myself', 'muttering', 'to', 'the', 'walls']
    #     y_seq = ['see', 'drown', 'himself']
    #     dist_func = edit_distance # edit_distance works on strings
    #     dist_val, cost_mat, acc_mat, path_val = dtw(x_seq, y_seq, dist_func, w=w_val, s=s_val)
    #     # Visualization would need adaptation for string data
