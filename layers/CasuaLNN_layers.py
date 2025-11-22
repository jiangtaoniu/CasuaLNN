import torch
import torch.nn as nn
import torch.fft
from layers.Autoformer_EncDec import series_decomp


class DFT_series_decomp(nn.Module):
    """
    Decomposes a time series into trend and seasonal components using the
    Discrete Fourier Transform (DFT). The seasonal component is synthesized
    from the most significant frequency components.
    """

    def __init__(self, top_k: int = 5):
        """
        Args:
            top_k (int): The number of top frequency components to retain for the seasonal part.
        """
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        """
        Decomposes the input time series.

        Args:
            x (torch.Tensor): The input time series of shape `(Batch, Seq_Len, D_Model)`.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the seasonal
            and trend components.
        """
        xf = torch.fft.rfft(x, dim=1)
        freq = torch.abs(xf)

        # Zero out the DC component's magnitude for top-k selection
        freq[:, 0, :] = 0

        # Determine the actual number of frequencies to select
        freq_dim_size = freq.shape[1]
        k = min(self.top_k, freq_dim_size)
        if freq_dim_size > 0 and k < 1 and self.top_k > 0:
            k = 1 # Ensure at least one frequency is selected if possible
        elif freq_dim_size == 0:
            k = 0 # Handle T=0 or T=1 case resulting in 0-sized frequency dimension

        # Filter frequencies based on top-k magnitudes
        if k > 0:
            top_k_freq, _ = torch.topk(freq, k, dim=1)
            threshold = top_k_freq[:, -1:, :]
            mask = freq < threshold
            xf[mask] = 0
        elif freq_dim_size > 0:
            # If k is 0 but frequencies exist, zero out all non-DC components
            xf[:, 1:, :] = 0

        # Reconstruct the seasonal component via inverse FFT
        x_season = torch.fft.irfft(xf, n=x.shape[1], dim=1)
        x_trend = x - x_season
        return x_season, x_trend


def get_num_patches(seq_len, patch_len, stride):
    """
    Calculates the number of patches that will be extracted from a sequence.

    The calculation is based on the formula used by `nn.Unfold`.
    A padding equal to the stride is implicitly assumed at the end of the sequence.

    Args:
        seq_len (int): The length of the input sequence.
        patch_len (int): The length of each patch.
        stride (int): The step size between consecutive patches.

    Returns:
        int: The number of patches.
    """
    if seq_len <= 0:
        return 0
    # The formula floor((L + padding - kernel_size) / stride + 1) is used.
    # Here, we assume a one-sided padding equal to the stride.
    padding = stride
    num_patches = (seq_len + padding - patch_len) // stride + 1
    return max(0, num_patches)


class MultiScaleSeasonMixing(nn.Module):
    """
    Performs bottom-up mixing of seasonal components across multiple scales.

    This module fuses information from finer-resolution scales into
    coarser-resolution scales.
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList()
        self.num_patches_list = []
        current_seq_len = configs.seq_len

        # Pre-calculate the number of patches for each scale
        for i in range(configs.down_sampling_layers + 1):
            num_patches = get_num_patches(current_seq_len, configs.patch_len, configs.stride)
            self.num_patches_list.append(num_patches)

            # Define the down-sampling projection layers
            if i < configs.down_sampling_layers:
                num_patches_fine = num_patches
                len_coarse = current_seq_len // configs.down_sampling_window
                num_patches_coarse = get_num_patches(len_coarse, configs.patch_len, configs.stride)

                if num_patches_fine <= 0 or num_patches_coarse <= 0:
                    raise ValueError(f"Calculated zero or negative patches at scale {i}. Fine: {num_patches_fine}, Coarse: {num_patches_coarse}.")

                self.down_sampling_layers.append(
                    nn.Sequential(
                        torch.nn.Linear(num_patches_fine, num_patches_coarse),
                        nn.GELU(),
                        torch.nn.Linear(num_patches_coarse, num_patches_coarse),
                    )
                )
            current_seq_len = max(1, current_seq_len // configs.down_sampling_window)

    def forward(self, season_list):
        if len(season_list) < 2:
            return season_list

        high_res_season = season_list[0]
        out_season_list = [high_res_season]

        for i in range(len(season_list) - 1):
            low_res_season = season_list[i + 1]
            high_res_season = high_res_season.permute(0, 2, 3, 1)  # B, N, D, T
            B, N, D, T_high = high_res_season.shape

            # Project from fine scale to coarse scale
            res = self.down_sampling_layers[i](high_res_season.reshape(B * N * D, T_high))
            target_T_low = self.down_sampling_layers[i][-1].out_features
            res = res.reshape(B, N, D, target_T_low)

            # Additive fusion
            low_res_season = low_res_season.permute(0, 2, 3, 1) + res
            high_res_season = low_res_season.permute(0, 3, 1, 2)  # Update for next iteration
            out_season_list.append(high_res_season)

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Performs top-down mixing of trend components across multiple scales.

    This module fuses information from coarser-resolution scales into
    finer-resolution scales.
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList()
        self.num_patches_list = []
        current_seq_len = configs.seq_len

        # Pre-calculate the number of patches for each scale
        for i in range(configs.down_sampling_layers + 1):
            num_patches = get_num_patches(current_seq_len, configs.patch_len, configs.stride)
            self.num_patches_list.append(num_patches)
            current_seq_len = max(1, current_seq_len // configs.down_sampling_window)

        # Define the up-sampling projection layers
        for i in reversed(range(configs.down_sampling_layers)):
            num_patches_coarse = self.num_patches_list[i + 1]
            num_patches_fine = self.num_patches_list[i]

            if num_patches_coarse <= 0 or num_patches_fine <= 0:
                raise ValueError(f"Calculated zero or negative patches for upsampling from scale {i+1} to {i}. Coarse: {num_patches_coarse}, Fine: {num_patches_fine}.")

            self.up_sampling_layers.append(
                nn.Sequential(
                    torch.nn.Linear(num_patches_coarse, num_patches_fine),
                    nn.GELU(),
                    torch.nn.Linear(num_patches_fine, num_patches_fine),
                )
            )

    def forward(self, trend_list):
        if len(trend_list) < 2:
            return trend_list

        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_trend_list_rev = [trend_list_reverse[0]]
        current_low_res_trend = trend_list_reverse[0]

        for i in range(len(trend_list_reverse) - 1):
            high_res_trend = trend_list_reverse[i + 1]
            current_low_res_trend = current_low_res_trend.permute(0, 2, 3, 1) # B, N, D, T
            B, N, D, T_low = current_low_res_trend.shape

            # Project from coarse scale to fine scale
            upsample_layer = self.up_sampling_layers[i]
            res = upsample_layer(current_low_res_trend.reshape(B * N * D, T_low))
            target_T_high = upsample_layer[-1].out_features
            res = res.reshape(B, N, D, target_T_high)

            # Additive fusion
            high_res_trend = high_res_trend.permute(0, 2, 3, 1) + res
            current_low_res_trend = high_res_trend.permute(0, 3, 1, 2) # Update for next iteration
            out_trend_list_rev.append(current_low_res_trend)

        out_trend_list_rev.reverse()
        return out_trend_list_rev


class PDM_Deep_Block(nn.Module):
    """
    Past Decomposable Mixing (PDM) Block for deep features.

    This block orchestrates the cross-scale interaction. It first decomposes
    the features from each scale into seasonal and trend components. Then, it
    mixes these components across scales using specialized mixing modules.
    Finally, it fuses the mixed components and adds them back to the original
    features via a residual connection.
    """

    def __init__(self, configs):
        super(PDM_Deep_Block, self).__init__()
        if configs.decomp_method == 'moving_avg':
            self.decomposition = series_decomp(configs.moving_avg)
            print("PDM_Deep_Block using: moving_avg decomposition")
        elif configs.decomp_method == 'dft_decomp':
            self.decomposition = DFT_series_decomp(configs.top_k)
            print(f"PDM_Deep_Block using: dft_decomp decomposition with top_k={configs.top_k}")
        else:
            raise ValueError(f'Unsupported decomposition method: {configs.decomp_method}')

        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)
        self.out_projection = nn.ModuleList([
            nn.Linear(configs.d_model, configs.d_model) for _ in range(configs.down_sampling_layers + 1)
        ])

    def forward(self, feature_list):
        season_list = []
        trend_list = []

        # 1. Decompose features at each scale
        for i, feature in enumerate(feature_list):
            B, T, N, D = feature.shape
            # Handle edge case where sequence length is too short for decomposition
            if T <= 1:
                season = torch.zeros_like(feature)
                trend = feature
            else:
                feature_reshaped = feature.permute(0, 2, 1, 3).reshape(B * N, T, D)
                season_r, trend_r = self.decomposition(feature_reshaped)
                season = season_r.reshape(B, N, T, D).permute(0, 2, 1, 3)
                trend = trend_r.reshape(B, N, T, D).permute(0, 2, 1, 3)
            season_list.append(season)
            trend_list.append(trend)

        # 2. Mix components across scales
        mixed_season_list = self.mixing_multi_scale_season(season_list)
        mixed_trend_list = self.mixing_multi_scale_trend(trend_list)

        # 3. Fuse and apply residual connection
        out_list = []
        for i, (ori_feature, season, trend) in enumerate(zip(feature_list, mixed_season_list, mixed_trend_list)):
            fused_feature = self.out_projection[i](season + trend)
            out_list.append(ori_feature + fused_feature)

        return out_list