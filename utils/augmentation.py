import numpy as np
from tqdm import tqdm
from scipy.interpolate import CubicSpline # Imported here to be available for magnitude_warp and time_warp


def jitter(x: np.ndarray, sigma: float = 0.03) -> np.ndarray:
    """
    Applies random noise (jittering) to the time series.

    This augmentation method adds a small amount of Gaussian noise to each data point,
    which can help improve model robustness to measurement noise.
    Based on: https://arxiv.org/pdf/1706.00527.pdf

    Args:
        x (np.ndarray): Input time series data, expected shape (batch_size, seq_len, num_channels).
        sigma (float): Standard deviation of the Gaussian noise.

    Returns:
        np.ndarray: Jittered time series data.
    """
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    """
    Applies random scaling to the time series.

    This method multiplies the entire time series (per channel) by a random scaling factor,
    which can help improve robustness to amplitude variations.
    Based on: https://arxiv.org/pdf/1706.00527.pdf

    Args:
        x (np.ndarray): Input time series data, expected shape (batch_size, seq_len, num_channels).
        sigma (float): Standard deviation of the scaling factor, centered at 1.0.

    Returns:
        np.ndarray: Scaled time series data.
    """
    # Generate a scaling factor for each batch sample and channel
    # Factor shape: (batch_size, num_channels)
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0], x.shape[2]))
    # Apply scaling factor, broadcasting over sequence length
    return np.multiply(x, factor[:, np.newaxis, :])


def rotation(x: np.ndarray) -> np.ndarray:
    """
    Applies random rotation to the time series.

    In this context, "rotation" involves randomly flipping the sign of features
    and permuting the order of features across the time series.

    Args:
        x (np.ndarray): Input time series data, expected shape (batch_size, seq_len, num_channels).

    Returns:
        np.ndarray: Rotated (flipped and permuted features) time series data.
    """
    # Randomly choose between -1 and 1 for each channel and batch sample
    flip = np.random.choice([-1, 1], size=(x.shape[0], x.shape[2]))
    # Randomly permute channel indices for each batch sample
    rotate_axis = np.arange(x.shape[2])
    np.random.shuffle(rotate_axis) # Shuffle in-place for permutation
    # Apply flipping and permutation
    return flip[:, np.newaxis, :] * x[:, :, rotate_axis]


def permutation(x: np.ndarray, max_segments: int = 5, seg_mode: str = "equal") -> np.ndarray:
    """
    Permutes segments of the time series.

    This method divides each time series into segments and then randomly shuffles
    the order of these segments.

    Args:
        x (np.ndarray): Input time series data, expected shape (batch_size, seq_len, num_channels).
        max_segments (int): Maximum number of segments to divide the time series into.
        seg_mode (str): Mode for segment splitting. "equal" for equal-sized segments,
                        "random" for randomly chosen split points.

    Returns:
        np.ndarray: Time series with permuted segments.
    """
    orig_steps = np.arange(x.shape[1]) # Original time steps (0 to seq_len-1)
    
    # Randomly determine number of segments for each sample
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x): # Iterate over each sample in the batch
        if num_segs[i] > 1:
            if seg_mode == "random":
                # Choose random split points for segments
                split_points = np.random.choice(x.shape[1] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else: # "equal" mode
                # Divide into approximately equal-sized segments
                splits = np.array_split(orig_steps, num_segs[i])
            
            # Permute the order of these segments
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[warp] # Apply the permutation
        else:
            ret[i] = pat # If only one segment, no permutation needed
    return ret


def magnitude_warp(x: np.ndarray, sigma: float = 0.2, knot: int = 4) -> np.ndarray:
    """
    Applies magnitude warping to the time series using cubic splines.

    This method warps the amplitude of the time series by generating a random
    smooth curve that is then applied as a scaling factor over time.

    Args:
        x (np.ndarray): Input time series data, expected shape (batch_size, seq_len, num_channels).
        sigma (float): Standard deviation for generating random warping factors.
        knot (int): Number of knots for the cubic spline interpolation.

    Returns:
        np.ndarray: Magnitude-warped time series data.
    """
    orig_steps = np.arange(x.shape[1])
    
    # Generate random warping factors for each knot point and channel
    # shape: (batch_size, knot+2, num_channels)
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
    # Define time steps for spline interpolation
    warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1., num=knot + 2))).T
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x): # Iterate over each sample in the batch
        # Create a cubic spline for each channel based on random warps
        # Apply the spline to original time steps to get continuous warping factors
        warper = np.array([CubicSpline(warp_steps[:, dim], random_warps[i, :, dim])(orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper # Apply magnitude warping by multiplication

    return ret


def time_warp(x: np.ndarray, sigma: float = 0.2, knot: int = 4) -> np.ndarray:
    """
    Applies time warping to the time series using cubic splines.

    This method warps the time axis of the time series by generating a random
    smooth curve that stretches or compresses parts of the time series.

    Args:
        x (np.ndarray): Input time series data, expected shape (batch_size, seq_len, num_channels).
        sigma (float): Standard deviation for generating random warping factors for time.
        knot (int): Number of knots for the cubic spline interpolation.

    Returns:
        np.ndarray: Time-warped time series data.
    """
    orig_steps = np.arange(x.shape[1])
    
    # Generate random warping factors for each knot point and channel
    # shape: (batch_size, knot+2, num_channels)
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
    # Define time steps for spline interpolation
    warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1., num=knot + 2))).T
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x): # Iterate over each sample in the batch
        for dim in range(x.shape[2]): # Apply warping per channel
            # Create a cubic spline for time warping
            time_warp_spline = CubicSpline(warp_steps[:, dim], warp_steps[:, dim] * random_warps[i, :, dim])(orig_steps)
            # Rescale the warped time to fit the original sequence length
            scale = (x.shape[1] - 1) / time_warp_spline[-1]
            # Interpolate the original pattern onto the warped time axis
            ret[i, :, dim] = np.interp(orig_steps, np.clip(scale * time_warp_spline, 0, x.shape[1] - 1), pat[:, dim])
    return ret


def window_slice(x: np.ndarray, reduce_ratio: float = 0.9) -> np.ndarray:
    """
    Applies window slicing to the time series, resampling to the original length.

    This method extracts a random sub-window of the time series and then
    resamples it back to the original sequence length.
    Based on: https://halshs.archives-ouvertes.fr/halshs-01357973/document

    Args:
        x (np.ndarray): Input time series data, expected shape (batch_size, seq_len, num_channels).
        reduce_ratio (float): The ratio by which to reduce the window size (e.g., 0.9 means 90% of original length).

    Returns:
        np.ndarray: Window-sliced and resampled time series data.
    """
    target_len = np.ceil(reduce_ratio * x.shape[1]).astype(int)
    if target_len >= x.shape[1]: # If target length is not smaller, return original
        return x
    
    # Randomly select start and end points for the slice
    starts = np.random.randint(low=0, high=x.shape[1] - target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x): # Iterate over each sample in the batch
        for dim in range(x.shape[2]): # Apply slicing per channel
            # Resample the sliced window back to the original sequence length
            ret[i, :, dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len), pat[starts[i]:ends[i], dim])
    return ret


def window_warp(x: np.ndarray, window_ratio: float = 0.1, scales: List[float] = [0.5, 2.]) -> np.ndarray:
    """
    Applies window warping to the time series, locally stretching or compressing a random window.

    This method selects a random window, warps its duration by a scaling factor,
    and then resamples the entire series back to the original length.
    Based on: https://halshs.archives-ouvertes.fr/halshs-01357973/document

    Args:
        x (np.ndarray): Input time series data, expected shape (batch_size, seq_len, num_channels).
        window_ratio (float): The proportion of the sequence length to be used as the window size.
        scales (List[float]): Possible scaling factors for warping the window.

    Returns:
        np.ndarray: Window-warped time series data.
    """
    # Randomly choose a scaling factor for each batch sample
    warp_scales = np.random.choice(scales, x.shape[0])
    # Determine the size of the window to be warped
    warp_size = np.ceil(window_ratio * x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)
        
    # Randomly choose start and end points for the window
    window_starts = np.random.randint(low=1, high=x.shape[1] - warp_size - 1, size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)
            
    ret = np.zeros_like(x)
    for i, pat in enumerate(x): # Iterate over each sample in the batch
        for dim in range(x.shape[2]): # Apply warping per channel
            # Extract segments: start, window, end
            start_seg = pat[:window_starts[i], dim]
            # Resample the window segment with the chosen warp scale
            window_seg = np.interp(np.linspace(0, warp_size - 1, num=int(warp_size * warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i], dim])
            end_seg = pat[window_ends[i]:, dim]
            
            # Concatenate the warped segments and resample the entire series to original length
            warped = np.concatenate((start_seg, window_seg, end_seg))                
            ret[i, :, dim] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1] - 1., num=warped.size), warped)
    return ret


def spawner(x: np.ndarray, labels: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    """
    Applies the "SPAWNER" (Synthetic Pattern Generation with Aligned Warping) augmentation.

    This method generates new patterns by averaging two intra-class time series
    after aligning them using Dynamic Time Warping (DTW).
    Based on: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6983028/

    Args:
        x (np.ndarray): Input time series data, expected shape (batch_size, seq_len, num_channels).
        labels (np.ndarray): Corresponding labels for `x`.
        sigma (float): Standard deviation for jittering the final averaged pattern.

    Returns:
        np.ndarray: Augmented time series data generated by SPAWNER.
    """
    import utils.dtw as dtw # Dynamically import dtw for this function
    
    # Randomly select a split point for DTW path division
    random_points = np.random.randint(low=1, high=x.shape[1] - 1, size=x.shape[0])
    window = np.ceil(x.shape[1] / 10.).astype(int) # Window size for DTW
    orig_steps = np.arange(x.shape[1])
    
    # Convert labels to class indices if one-hot encoded
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x): # Iterate over each sample in the batch
        # Find other samples belonging to the same class
        choices = np.delete(np.arange(x.shape[0]), i) # Exclude self
        choices = np.where(l[choices] == l[i])[0] # Filter by class
        
        if choices.size > 0:     
            random_sample = x[np.random.choice(choices)] # Pick a random intra-class sample
            
            # SPAWNER splits the path into two parts and performs DTW on each part
            path1 = dtw.dtw(pat[:random_points[i]], random_sample[:random_points[i]], dtw.RETURN_PATH, slope_constraint="symmetric", window=window)
            path2 = dtw.dtw(pat[random_points[i]:], random_sample[random_points[i]:], dtw.RETURN_PATH, slope_constraint="symmetric", window=window)
            
            # Combine the two DTW paths
            # path1 is (seq_len_part1, 2), path2 is (seq_len_part2, 2)
            # The second part of the path needs to be offset by random_points[i]
            combined_path = np.concatenate((path1[0], path2[0] + random_points[i]), axis=1) # Original has axis=1, assuming it refers to cols
            
            # Calculate the mean of the two patterns along the combined DTW path
            mean_pattern = np.mean([pat[combined_path[0]], random_sample[combined_path[1]]], axis=0) # Assuming pat[path] gives correct indexing for all dimensions
            
            # Interpolate the averaged pattern back to the original sequence length
            for dim in range(x.shape[2]):
                ret[i, :, dim] = np.interp(orig_steps, np.linspace(0, x.shape[1] - 1., num=mean_pattern.shape[0]), mean_pattern[:, dim])
        else:
            # If no other patterns of the same class exist, keep original pattern
            ret[i, :] = pat
    
    return jitter(ret, sigma=sigma) # Apply jittering to the final pattern


def wdba(x: np.ndarray, labels: np.ndarray, batch_size: int = 6, slope_constraint: str = "symmetric", use_window: bool = True) -> np.ndarray:
    """
    Applies Weighted Dynamic Barycenter Averaging (WDBA) augmentation.

    This method finds a representative barycenter (average) pattern within a class
    using DTW.
    Based on: https://ieeexplore.ieee.org/document/8215569

    Args:
        x (np.ndarray): Input time series data, expected shape (batch_size, seq_len, num_channels).
        labels (np.ndarray): Corresponding labels for `x`.
        batch_size (int): Number of intra-class patterns to sample for DBA.
        slope_constraint (str): Slope constraint for DTW ("symmetric" or "asymmetric").
        use_window (bool): Whether to use a window constraint for DTW.

    Returns:
        np.ndarray: Augmented time series data generated by WDBA.
    """
    import utils.dtw as dtw # Dynamically import dtw for this function
    
    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int) # Window size for DTW
    else:
        window = None
    orig_steps = np.arange(x.shape[1])
    
    # Convert labels to class indices if one-hot encoded
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
        
    ret = np.zeros_like(x)
    for i in range(ret.shape[0]): # Iterate over each sample
        # Get samples from the same class
        choices = np.where(l == l[i])[0]
        
        if choices.size > 0:        
            # Pick `batch_size` random intra-class patterns (prototypes)
            k = min(choices.size, batch_size)
            random_prototypes = x[np.random.choice(choices, k, replace=False)]
            
            # Calculate DTW distances between all selected prototypes
            dtw_matrix = np.zeros((k, k))
            for p_idx, prototype in enumerate(random_prototypes):
                for s_idx, sample in enumerate(random_prototypes):
                    if p_idx != s_idx: # No need to calculate distance to self
                        dtw_matrix[p_idx, s_idx] = dtw.dtw(prototype, sample, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                        
            # Find the medoid pattern (the one with the smallest sum of distances to others)
            medoid_id = np.argsort(np.sum(dtw_matrix, axis=1))[0]
            # Order other patterns by their distance to the medoid
            nearest_order = np.argsort(dtw_matrix[medoid_id])
            medoid_pattern = random_prototypes[medoid_id]
            
            # Start Weighted DBA
            average_pattern = np.zeros_like(medoid_pattern)
            weighted_sums = np.zeros(medoid_pattern.shape[0]) # For normalization
            
            # Sum up patterns weighted by their similarity to the medoid
            for nid in nearest_order:
                if nid == medoid_id or (k > 1 and dtw_matrix[medoid_id, nearest_order[1]] == 0.): # Handle case of single prototype or zero distance
                    average_pattern += medoid_pattern
                    weighted_sums += np.ones_like(weighted_sums) 
                else:
                    # Perform DTW to align prototype with the medoid pattern
                    path = dtw.dtw(medoid_pattern, random_prototypes[nid], dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
                    dtw_value = dtw_matrix[medoid_id, nid]
                    warped_pattern = random_prototypes[nid][path[1]] # Warp current pattern to medoid's time axis
                    
                    # Calculate weight based on DTW distance
                    # Weighting scheme inspired by DBA: closer patterns have higher weight
                    weight = np.exp(np.log(0.5) * dtw_value / dtw_matrix[medoid_id, nearest_order[1]])
                    
                    average_pattern[path[0]] += weight * warped_pattern # Accumulate weighted warped pattern
                    weighted_sums[path[0]] += weight # Accumulate weights
            
            # Normalize by weighted sums to get the barycenter
            # np.newaxis is used for broadcasting division over channels
            ret[i, :] = average_pattern / weighted_sums[:, np.newaxis]
        else:
            # If no other patterns of the same class exist, keep original pattern
            ret[i, :] = x[i]
    return ret


def random_guided_warp(x: np.ndarray, labels: np.ndarray, slope_constraint: str = "symmetric", use_window: bool = True, dtw_type: str = "normal") -> np.ndarray:
    """
    Applies random guided time warping using DTW.

    This method warps a time series to align with a randomly selected
    intra-class prototype using DTW.

    Args:
        x (np.ndarray): Input time series data, expected shape (batch_size, seq_len, num_channels).
        labels (np.ndarray): Corresponding labels for `x`.
        slope_constraint (str): Slope constraint for DTW ("symmetric" or "asymmetric").
        use_window (bool): Whether to use a window constraint for DTW.
        dtw_type (str): Type of DTW to use ("normal" for standard DTW, "shape" for ShapeDTW).

    Returns:
        np.ndarray: Augmented time series data with random guided time warping.
    """
    import utils.dtw as dtw # Dynamically import dtw for this function
    
    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int) # Window size for DTW
    else:
        window = None
    orig_steps = np.arange(x.shape[1])
    
    # Convert labels to class indices if one-hot encoded
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x): # Iterate over each sample in the batch
        # Find other samples belonging to the same class
        choices = np.delete(np.arange(x.shape[0]), i) # Exclude self
        choices = np.where(l[choices] == l[i])[0] # Filter by class
        
        if choices.size > 0:        
            random_prototype = x[np.random.choice(choices)] # Pick a random intra-class prototype
            
            # Perform DTW (normal or shape) to find optimal alignment path
            if dtw_type == "shape":
                path = dtw.shape_dtw(random_prototype, pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
            else:
                path = dtw.dtw(random_prototype, pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
                            
            # Time warp: apply the alignment path from the prototype to the current pattern
            warped_pattern = pat[path[1]] # Use path[1] (indices of `pat`) to warp `pat`
            
            # Interpolate the warped pattern back to the original sequence length
            for dim in range(x.shape[2]):
                ret[i, :, dim] = np.interp(orig_steps, np.linspace(0, x.shape[1] - 1., num=warped_pattern.shape[0]), warped_pattern[:, dim])
        else:
            # If no other patterns of the same class exist, keep original pattern
            ret[i, :] = pat
    return ret


def random_guided_warp_shape(x: np.ndarray, labels: np.ndarray, slope_constraint: str = "symmetric", use_window: bool = True) -> np.ndarray:
    """
    Wrapper for `random_guided_warp` using ShapeDTW.
    """
    return random_guided_warp(x, labels, slope_constraint, use_window, dtw_type="shape")


def discriminative_guided_warp(x: np.ndarray, labels: np.ndarray, batch_size: int = 6, slope_constraint: str = "symmetric", use_window: bool = True, dtw_type: str = "normal", use_variable_slice: bool = True) -> np.ndarray:
    """
    Applies discriminative guided time warping using DTW.

    This method warps a time series to align with an intra-class prototype
    that is "most discriminative" (i.e., furthest from negative class prototypes).
    It aims to stretch the pattern towards a good intra-class example and away
    from inter-class examples.

    Args:
        x (np.ndarray): Input time series data, expected shape (batch_size, seq_len, num_channels).
        labels (np.ndarray): Corresponding labels for `x`.
        batch_size (int): Number of prototypes (positive and negative) to sample.
        slope_constraint (str): Slope constraint for DTW ("symmetric" or "asymmetric").
        use_window (bool): Whether to use a window constraint for DTW.
        dtw_type (str): Type of DTW to use ("normal" for standard DTW, "shape" for ShapeDTW).
        use_variable_slice (bool): Whether to apply variable window slicing after warping.

    Returns:
        np.ndarray: Augmented time series data with discriminative guided time warping.
    """
    import utils.dtw as dtw # Dynamically import dtw for this function
    
    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int) # Window size for DTW
    else:
        window = None
    orig_steps = np.arange(x.shape[1])
    
    # Convert labels to class indices if one-hot encoded
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
    
    # Split batch_size for positive and negative prototypes
    positive_batch = np.ceil(batch_size / 2).astype(int)
    negative_batch = np.floor(batch_size / 2).astype(int)
        
    ret = np.zeros_like(x)
    warp_amount = np.zeros(x.shape[0]) # Store warp amount for optional variable slicing
    
    for i, pat in enumerate(x): # Iterate over each sample in the batch
        # Find other samples, excluding self
        choices = np.delete(np.arange(x.shape[0]), i)
        
        # Separate into positive (same class) and negative (different class) choices
        positive_choices = np.where(l[choices] == l[i])[0]
        negative_choices = np.where(l[choices] != l[i])[0]
        
        if positive_choices.size > 0 and negative_choices.size > 0:
            # Sample positive and negative prototypes
            pos_k = min(positive_choices.size, positive_batch)
            neg_k = min(negative_choices.size, negative_batch)
            positive_prototypes = x[np.random.choice(positive_choices, pos_k, replace=False)]
            negative_prototypes = x[np.random.choice(negative_choices, neg_k, replace=False)]
                        
            # Calculate average distance from each positive prototype to its own class
            # and to the negative class. The most discriminative one is chosen.
            pos_aves = np.zeros(pos_k)
            neg_aves = np.zeros(pos_k)
            
            for p_idx, pos_prot in enumerate(positive_prototypes):
                # Average distance to other positive prototypes
                for ps_idx, pos_samp in enumerate(positive_prototypes):
                    if p_idx != ps_idx:
                        if dtw_type == "shape":
                            pos_aves[p_idx] += (1. / (pos_k - 1.)) * dtw.shape_dtw(pos_prot, pos_samp, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                        else:
                            pos_aves[p_idx] += (1. / (pos_k - 1.)) * dtw.dtw(pos_prot, pos_samp, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                # Average distance to negative prototypes
                for ns_idx, neg_samp in enumerate(negative_prototypes):
                    if dtw_type == "shape":
                        neg_aves[p_idx] += (1. / neg_k) * dtw.shape_dtw(pos_prot, neg_samp, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                    else:
                        neg_aves[p_idx] += (1. / neg_k) * dtw.dtw(pos_prot, neg_samp, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
            
            # Select the prototype that maximizes (distance to negative - distance to positive)
            selected_id = np.argmax(neg_aves - pos_aves)
            
            # Perform DTW with the selected discriminative prototype
            if dtw_type == "shape":
                path = dtw.shape_dtw(positive_prototypes[selected_id], pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
            else:
                path = dtw.dtw(positive_prototypes[selected_id], pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
                   
            # Time warp
            warped_pattern = pat[path[1]]
            # Calculate warp amount for variable slicing: sum of absolute difference between original steps and warped steps
            warp_path_interp = np.interp(orig_steps, np.linspace(0, x.shape[1] - 1., num=warped_pattern.shape[0]), path[1])
            warp_amount[i] = np.sum(np.abs(orig_steps - warp_path_interp))
            
            # Interpolate the warped pattern back to the original sequence length
            for dim in range(x.shape[2]):
                ret[i, :, dim] = np.interp(orig_steps, np.linspace(0, x.shape[1] - 1., num=warped_pattern.shape[0]), warped_pattern[:, dim])
        else:
            # If insufficient positive/negative samples, keep original pattern
            ret[i, :] = pat
            warp_amount[i] = 0.
            
    # Apply variable slicing based on the warp amount
    if use_variable_slice:
        max_warp = np.max(warp_amount)
        if max_warp == 0: # If no warping occurred, apply a default slice
            ret = window_slice(ret, reduce_ratio=0.9)
        else:
            for i, pat in enumerate(ret):
                # Adjust reduce_ratio based on the calculated warp_amount
                # More warping -> less reduction to compensate
                ret[i] = window_slice(pat[np.newaxis, :, :], reduce_ratio=0.9 + 0.1 * warp_amount[i] / max_warp)[0]
    return ret


def discriminative_guided_warp_shape(x: np.ndarray, labels: np.ndarray, batch_size: int = 6, slope_constraint: str = "symmetric", use_window: bool = True) -> np.ndarray:
    """
    Wrapper for `discriminative_guided_warp` using ShapeDTW.
    """
    return discriminative_guided_warp(x, labels, batch_size, slope_constraint, use_window, dtw_type="shape")


def run_augmentation(x: np.ndarray, y: np.ndarray, args) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Applies multiple rounds of data augmentation to the input data.

    Args:
        x (np.ndarray): Original input time series data.
        y (np.ndarray): Original labels corresponding to `x`.
        args: Configuration arguments containing augmentation flags and ratio.

    Returns:
        tuple[np.ndarray, np.ndarray, str]: Augmented `x` data, augmented `y` data, and a string tag
                                            indicating applied augmentations.
    """
    print(f"Augmenting data for {args.data}")
    np.random.seed(args.seed) # Ensure reproducibility of augmentation
    
    x_aug = x
    y_aug = y
    augmentation_tags = ""

    if args.augmentation_ratio > 0:
        augmentation_tags = f"{args.augmentation_ratio}"
        for n in range(args.augmentation_ratio):
            # Augment a copy of the original data in each round
            x_temp, current_aug_tags = augment(x, y, args)
            x_aug = np.append(x_aug, x_temp, axis=0) # Concatenate augmented data
            y_aug = np.append(y_aug, y, axis=0)     # Labels are duplicated for augmented data
            # print(f"Round {n+1}: {current_aug_tags} done") # Debug print removed
        if args.extra_tag:
            augmentation_tags += "_" + args.extra_tag
    else:
        augmentation_tags = args.extra_tag if args.extra_tag else "no_augmentation"
    
    return x_aug, y_aug, augmentation_tags


def run_augmentation_single(x: np.ndarray, y: np.ndarray, args) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Applies a single round of data augmentation to a batch or an individual time series.

    Handles reshaping for single-sample inputs to fit augmentation functions
    which typically expect a batch dimension.

    Args:
        x (np.ndarray): Input time series data, shape can be (seq_len, num_channels)
                        for a single series, or (batch_size, seq_len, num_channels).
        y (np.ndarray): Labels corresponding to `x`.
        args: Configuration arguments containing augmentation flags and ratio.

    Returns:
        tuple[np.ndarray, np.ndarray, str]: Augmented `x` data, augmented `y` data, and a string tag
                                            indicating applied augmentations.
    """
    np.random.seed(args.seed) # Ensure reproducibility of augmentation

    x_input = x
    y_aug = y # Labels are often just duplicated, not truly augmented
    
    # Add batch dimension if input is a single time series
    if len(x.shape) == 2: # (sequence_length, num_channels)
        x_input = x[np.newaxis, :, :] # -> (1, sequence_length, num_channels)
    elif len(x.shape) != 3: # Expected batch_size, seq_len, num_channels
        raise ValueError("Input `x` must be 2D (single series) or 3D (batch of series).")

    augmentation_tags = ""
    if args.augmentation_ratio > 0:
        augmentation_tags = f"{args.augmentation_ratio}"
        # Apply augmentation for args.augmentation_ratio times
        # Here, `augment` is called once and `x_aug` is the result.
        # The loop logic of `run_augmentation` is simplified here to a single call to `augment` per run.
        x_aug, current_aug_tags = augment(x_input, y, args)
        # print(f"Round {n+1}: {current_aug_tags} done") # Debug print removed
        if args.extra_tag:
            augmentation_tags += "_" + args.extra_tag
    else:
        x_aug = x # If no augmentation, return original data
        augmentation_tags = args.extra_tag if args.extra_tag else "no_augmentation"

    # Remove the added batch dimension if the input was originally a single time series
    if len(x.shape) == 2:
        x_aug = x_aug.squeeze(0) # -> (sequence_length, num_channels)
    
    return x_aug, y_aug, augmentation_tags


def augment(x: np.ndarray, y: np.ndarray, args) -> tuple[np.ndarray, str]:
    """
    Applies a selected set of augmentation techniques to the input time series data.

    Args:
        x (np.ndarray): Input time series data, expected shape (batch_size, seq_len, num_channels).
        y (np.ndarray): Corresponding labels for `x`. Used by some augmentation methods.
        args: Configuration arguments with boolean flags for each augmentation method.

    Returns:
        tuple[np.ndarray, str]: Augmented `x` data and a string tag indicating applied augmentations.
    """
    augmentation_tags = ""
    # Each 'if' block checks a boolean flag from `args` to decide which augmentation to apply.
    if hasattr(args, 'jitter') and args.jitter:
        x = jitter(x, sigma=args.jitter_sigma if hasattr(args, 'jitter_sigma') else 0.03)
        augmentation_tags += "_jitter"
    if hasattr(args, 'scaling') and args.scaling:
        x = scaling(x, sigma=args.scaling_sigma if hasattr(args, 'scaling_sigma') else 0.1)
        augmentation_tags += "_scaling"
    if hasattr(args, 'rotation') and args.rotation:
        x = rotation(x)
        augmentation_tags += "_rotation"
    if hasattr(args, 'permutation') and args.permutation:
        x = permutation(x)
        augmentation_tags += "_permutation"
    if hasattr(args, 'randompermutation') and args.randompermutation:
        x = permutation(x, seg_mode="random")
        augmentation_tags += "_randomperm"
    if hasattr(args, 'magwarp') and args.magwarp:
        x = magnitude_warp(x, sigma=args.magwarp_sigma if hasattr(args, 'magwarp_sigma') else 0.2,
                           knot=args.magwarp_knot if hasattr(args, 'magwarp_knot') else 4)
        augmentation_tags += "_magwarp"
    if hasattr(args, 'timewarp') and args.timewarp:
        x = time_warp(x, sigma=args.timewarp_sigma if hasattr(args, 'timewarp_sigma') else 0.2,
                      knot=args.timewarp_knot if hasattr(args, 'timewarp_knot') else 4)
        augmentation_tags += "_timewarp"
    if hasattr(args, 'windowslice') and args.windowslice:
        x = window_slice(x, reduce_ratio=args.windowslice_ratio if hasattr(args, 'windowslice_ratio') else 0.9)
        augmentation_tags += "_windowslice"
    if hasattr(args, 'windowwarp') and args.windowwarp:
        x = window_warp(x, window_ratio=args.windowwarp_ratio if hasattr(args, 'windowwarp_ratio') else 0.1,
                        scales=args.windowwarp_scales if hasattr(args, 'windowwarp_scales') else [0.5, 2.])
        augmentation_tags += "_windowwarp"
    if hasattr(args, 'spawner') and args.spawner:
        x = spawner(x, y, sigma=args.spawner_sigma if hasattr(args, 'spawner_sigma') else 0.05)
        augmentation_tags += "_spawner"
    if hasattr(args, 'dtwwarp') and args.dtwwarp:
        x = random_guided_warp(x, y, slope_constraint=args.dtwwarp_slope if hasattr(args, 'dtwwarp_slope') else "symmetric",
                               use_window=args.dtwwarp_window if hasattr(args, 'dtwwarp_window') else True)
        augmentation_tags += "_rgw"
    if hasattr(args, 'shapedtwwarp') and args.shapedtwwarp:
        x = random_guided_warp_shape(x, y, slope_constraint=args.shapedtwwarp_slope if hasattr(args, 'shapedtwwarp_slope') else "symmetric",
                                     use_window=args.shapedtwwarp_window if hasattr(args, 'shapedtwwarp_window') else True)
        augmentation_tags += "_rgws"
    if hasattr(args, 'wdba') and args.wdba:
        x = wdba(x, y, batch_size=args.wdba_batch_size if hasattr(args, 'wdba_batch_size') else 6,
                 slope_constraint=args.wdba_slope if hasattr(args, 'wdba_slope') else "symmetric",
                 use_window=args.wdba_window if hasattr(args, 'wdba_window') else True)
        augmentation_tags += "_wdba"
    if hasattr(args, 'discdtw') and args.discdtw:
        x = discriminative_guided_warp(x, y, batch_size=args.discdtw_batch_size if hasattr(args, 'discdtw_batch_size') else 6,
                                     slope_constraint=args.discdtw_slope if hasattr(args, 'discdtw_slope') else "symmetric",
                                     use_window=args.discdtw_window if hasattr(args, 'discdtw_window') else True,
                                     use_variable_slice=args.discdtw_var_slice if hasattr(args, 'discdtw_var_slice') else True)
        augmentation_tags += "_dgw"
    if hasattr(args, 'discsdtw') and args.discsdtw:
        x = discriminative_guided_warp_shape(x, y, batch_size=args.discsdtw_batch_size if hasattr(args, 'discsdtw_batch_size') else 6,
                                           slope_constraint=args.discsdtw_slope if hasattr(args, 'discsdtw_slope') else "symmetric",
                                           use_window=args.discsdtw_window if hasattr(args, 'discsdtw_window') else True)
        augmentation_tags += "_dgws"
    
    return x, augmentation_tags