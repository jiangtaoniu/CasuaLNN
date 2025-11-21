# layers/MS_IPM_layers_backup3_inOutPatch.py

import torch
import torch.nn as nn
import torch.fft
from layers.Transformer_EncDec import Encoder as TransformerEncoder, EncoderLayer as TransformerEncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from layers.Autoformer_EncDec import series_decomp # 仍然需要导入 moving_avg 的实现

# ######################################################################
# ## --- DFT_series_decomp 类定义 (包含 topk 修正) ---
# ######################################################################
class DFT_series_decomp(nn.Module):
    """
    Series decomposition block using Discrete Fourier Transform
    """
    def __init__(self, top_k: int = 5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x, dim=1)
        freq = torch.abs(xf)
        freq[:, 0, :] = 0

        freq_dim_size = freq.shape[1]
        k = min(self.top_k, freq_dim_size)
        # Ensure k is at least 1 if top_k > 0 and dimension allows
        if freq_dim_size > 0 and k < 1 and self.top_k > 0:
            k = 1
            # print(f"Warning: DFT_series_decomp adapting k to 1 due to small frequency dimension size ({freq_dim_size})")
        elif freq_dim_size == 0: # Handle T=0 or T=1 case resulting in 0 freq dim size (excluding DC)
             k = 0 # No frequencies to select


        if k > 0: # Only perform topk if k is valid
            top_k_freq, top_list = torch.topk(freq, k, dim=1)
            threshold = top_k_freq[:, -1:, :]
            mask = freq < threshold
            mask[:, 0, :] = False # Always keep DC component
            xf[mask] = 0
        elif freq_dim_size > 0: # If k became 0 but freq dim exists (means only DC was present)
             # Zero out everything except DC (index 0)
             xf[:, 1:, :] = 0
        # Else: if freq_dim_size was 0, xf is likely empty or just DC, no action needed.


        x_season = torch.fft.irfft(xf, n=x.shape[1], dim=1)
        x_trend = x - x_season
        return x_season, x_trend
# ######################################################################
# ## --- DFT_series_decomp 结束 ---
# ######################################################################


# ######################################################################
# ## --- get_num_patches 函数 (确认 return 语句存在) ---
# ######################################################################
def get_num_patches(seq_len, patch_len, stride):
    """
    辅助函数，根据序列长度、Patch长度和步长计算Patch的数量。
    """
    padding = stride
    # Basic formula based on how many strides fit after the first patch
    num_strides = (seq_len + padding - patch_len) // stride
    # Total patches = first patch + number of strides
    num_patches = num_strides + 1
    # Ensure non-negative and handle edge cases where seq_len might be small
    # If padded length is less than patch_len, result should be 1 if seq_len > 0?
    if seq_len <= 0:
        return 0
    if seq_len + padding < patch_len:
         # Even if shorter than a patch, unfolding with padding might produce one patch
         # Example: seq=5, patch=16, stride=8, pad=8 -> padded=13. (13-16)//8+1 = -1+1=0 -> incorrect?
         # Let's consider nn.Unfold directly: it produces floor((L + 2*pad - k)/s + 1) patches.
         # Our case: floor((seq_len + padding - patch_len)/stride + 1)
         # If seq_len=5, patch=16, stride=8, pad=8 -> floor((5+8-16)/8 + 1) = floor(-3/8 + 1) = 0.
         # This seems mathematically correct, maybe 0 patches *is* the right answer if padded length < patch_len.
         # Let's stick to the formula and ensure it's non-negative.
         return max(0, (seq_len + padding - patch_len) // stride + 1)

    return max(0, num_patches) # Return the calculated number, ensuring >= 0
# ######################################################################
# ## --- get_num_patches 结束 ---
# ######################################################################


class MultiScaleSeasonMixing(nn.Module):
    """
    适配于深度特征的自下而上季节性混合模块 (已修复维度问题 + None检查)
    """
    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

        # --- 添加断言检查 configs ---
        assert hasattr(configs, 'seq_len') and isinstance(configs.seq_len, int) and configs.seq_len > 0
        assert hasattr(configs, 'patch_len') and isinstance(configs.patch_len, int) and configs.patch_len > 0
        assert hasattr(configs, 'stride') and isinstance(configs.stride, int) and configs.stride > 0
        assert hasattr(configs, 'down_sampling_layers') and isinstance(configs.down_sampling_layers, int) and configs.down_sampling_layers >= 0
        assert hasattr(configs, 'down_sampling_window') and isinstance(configs.down_sampling_window, int) and configs.down_sampling_window > 0
        # --- 断言结束 ---

        self.down_sampling_layers = torch.nn.ModuleList()
        self.num_patches_list = []
        current_seq_len = configs.seq_len

        for i in range(configs.down_sampling_layers + 1):
            num_patches = get_num_patches(current_seq_len, configs.patch_len, configs.stride)

            # --- 添加对 get_num_patches 返回值的显式检查 ---
            if num_patches is None:
                raise TypeError(f"get_num_patches returned None for scale {i} with seq_len={current_seq_len}, patch_len={configs.patch_len}, stride={configs.stride}")
            if not isinstance(num_patches, int):
                 raise TypeError(f"get_num_patches returned non-integer ({type(num_patches)}) for scale {i}")
            # --- 检查结束 ---

            self.num_patches_list.append(num_patches)

            if i < configs.down_sampling_layers:
                num_patches_fine = num_patches
                len_coarse = current_seq_len // configs.down_sampling_window
                num_patches_coarse = get_num_patches(len_coarse, configs.patch_len, configs.stride)

                # --- 添加对 get_num_patches 返回值的显式检查 ---
                if num_patches_coarse is None:
                    raise TypeError(f"get_num_patches returned None for coarse scale {i+1} with len_coarse={len_coarse}")
                if not isinstance(num_patches_coarse, int):
                    raise TypeError(f"get_num_patches returned non-integer ({type(num_patches_coarse)}) for coarse scale {i+1}")
                # --- 检查结束 ---

                # Check for non-positive num_patches *after* ensuring they are integers
                if num_patches_fine <= 0 or num_patches_coarse <= 0:
                     # This check should now compare int <= int
                     raise ValueError(f"Calculated zero or negative patches at scale {i}. Fine: {num_patches_fine}, Coarse: {num_patches_coarse}. Check seq_len, patch_len, stride settings.")

                self.down_sampling_layers.append(
                    nn.Sequential(
                        torch.nn.Linear(num_patches_fine, num_patches_coarse),
                        nn.GELU(),
                        torch.nn.Linear(num_patches_coarse, num_patches_coarse),
                    )
                )
            # Update sequence length for the next scale calculation
            current_seq_len = max(1, current_seq_len // configs.down_sampling_window) # Ensure seq_len doesn't go below 1

    def forward(self, season_list):
        # ... (forward method remains the same as previous version,
        #      including dimension checks which are still useful) ...
        if len(season_list) != len(self.num_patches_list):
             raise ValueError(f"Number of season inputs ({len(season_list)}) doesn't match expected scales ({len(self.num_patches_list)}).")
        for i, season in enumerate(season_list):
            B, T, N, D = season.shape
            expected_T = self.num_patches_list[i]
            if T != expected_T:
                 raise ValueError(f"Input season at scale {i} has incorrect patch dimension: Expected {expected_T}, got {T}.")
        if len(season_list) < 2: return season_list
        high_res_season = season_list[0]
        out_season_list = [high_res_season]
        for i in range(len(season_list) - 1):
            low_res_season = season_list[i + 1]
            high_res_season = high_res_season.permute(0, 2, 3, 1)
            B, N, D, T_high = high_res_season.shape
            expected_T_high = self.down_sampling_layers[i][0].in_features
            if T_high != expected_T_high: raise ValueError(f"Mismatched dimension for down_sampling_layer {i} input: Expected {expected_T_high}, got {T_high}")
            res = self.down_sampling_layers[i](high_res_season.reshape(B * N * D, T_high))
            target_T_low = self.down_sampling_layers[i][-1].out_features
            res = res.reshape(B, N, D, target_T_low)
            expected_T_low_input = self.num_patches_list[i+1]
            if low_res_season.shape[1] != expected_T_low_input: raise ValueError(f"Mismatched dimension for low_res_season input at index {i+1}: Expected {expected_T_low_input}, got {low_res_season.shape[1]}")
            if target_T_low != expected_T_low_input: raise ValueError(f"Calculated target patch dim ({target_T_low}) doesn't match expected input patch dim ({expected_T_low_input}) for scale {i+1}")
            low_res_season = low_res_season.permute(0, 2, 3, 1) + res
            high_res_season = low_res_season.permute(0, 3, 1, 2)
            out_season_list.append(high_res_season)
        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    适配于深度特征的自上而下趋势性混合模块 (已修复维度问题 + None检查)
    """
    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        # --- 添加断言检查 configs (同上) ---
        assert hasattr(configs, 'seq_len') and isinstance(configs.seq_len, int) and configs.seq_len > 0
        assert hasattr(configs, 'patch_len') and isinstance(configs.patch_len, int) and configs.patch_len > 0
        assert hasattr(configs, 'stride') and isinstance(configs.stride, int) and configs.stride > 0
        assert hasattr(configs, 'down_sampling_layers') and isinstance(configs.down_sampling_layers, int) and configs.down_sampling_layers >= 0
        assert hasattr(configs, 'down_sampling_window') and isinstance(configs.down_sampling_window, int) and configs.down_sampling_window > 0
        # --- 断言结束 ---

        self.up_sampling_layers = torch.nn.ModuleList()
        self.num_patches_list = []
        current_seq_len = configs.seq_len
        temp_num_patches = []
        for i in range(configs.down_sampling_layers + 1):
             num_patches = get_num_patches(current_seq_len, configs.patch_len, configs.stride)
             # --- 添加对 get_num_patches 返回值的显式检查 ---
             if num_patches is None:
                 raise TypeError(f"get_num_patches returned None for scale {i} (TrendMixing init) with seq_len={current_seq_len}")
             if not isinstance(num_patches, int):
                 raise TypeError(f"get_num_patches returned non-integer ({type(num_patches)}) for scale {i} (TrendMixing init)")
             # --- 检查结束 ---
             temp_num_patches.append(num_patches)
             current_seq_len = max(1, current_seq_len // configs.down_sampling_window) # Ensure >= 1
        self.num_patches_list = temp_num_patches

        for i in reversed(range(configs.down_sampling_layers)): # i = 1, 0 (if layers=2)
            # coarse scale index = i+1, fine scale index = i
            num_patches_coarse = self.num_patches_list[i + 1]
            num_patches_fine = self.num_patches_list[i]

            # Check for non-positive num_patches *after* ensuring they are integers
            if num_patches_coarse <= 0 or num_patches_fine <= 0:
                 # This check should now compare int <= int
                 raise ValueError(f"Calculated zero or negative patches for upsampling from scale {i+1} to {i}. Coarse: {num_patches_coarse}, Fine: {num_patches_fine}.")

            self.up_sampling_layers.append( # Appends layer for 2->1, then layer for 1->0
                nn.Sequential(
                    torch.nn.Linear(num_patches_coarse, num_patches_fine),
                    nn.GELU(),
                    torch.nn.Linear(num_patches_fine, num_patches_fine),
                )
            )

    def forward(self, trend_list):
        # ... (forward method remains the same as previous version,
        #      including dimension checks which are still useful) ...
        if len(trend_list) != len(self.num_patches_list): raise ValueError(f"Number of trend inputs ({len(trend_list)}) doesn't match expected scales ({len(self.num_patches_list)}).")
        for i, trend in enumerate(trend_list):
            B, T, N, D = trend.shape
            expected_T = self.num_patches_list[i]
            if T != expected_T: raise ValueError(f"Input trend at scale {i} has incorrect patch dimension: Expected {expected_T}, got {T}.")
        if len(trend_list) < 2: return trend_list
        trend_list_reverse = trend_list.copy(); trend_list_reverse.reverse()
        out_trend_list_rev = [trend_list_reverse[0]]
        current_low_res_trend = trend_list_reverse[0]
        for i in range(len(trend_list_reverse) - 1):
            high_res_trend = trend_list_reverse[i + 1]
            current_low_res_trend = current_low_res_trend.permute(0, 2, 3, 1)
            B, N, D, T_low = current_low_res_trend.shape
            upsample_layer = self.up_sampling_layers[i]
            expected_T_low = upsample_layer[0].in_features
            if T_low != expected_T_low: raise ValueError(f"Mismatched dimension for up_sampling_layer {i} input: Expected {expected_T_low}, got {T_low}")
            res = upsample_layer(current_low_res_trend.reshape(B * N * D, T_low))
            target_T_high = upsample_layer[-1].out_features
            res = res.reshape(B, N, D, target_T_high)
            original_high_res_scale_index = len(trend_list) - 1 - (i + 1)
            expected_T_high_input = self.num_patches_list[original_high_res_scale_index]
            if high_res_trend.shape[1] != expected_T_high_input: raise ValueError(f"Mismatched dimension for high_res_trend input at index {i+1} (scale {original_high_res_scale_index}): Expected {expected_T_high_input}, got {high_res_trend.shape[1]}")
            if target_T_high != expected_T_high_input: raise ValueError(f"Calculated target patch dim ({target_T_high}) doesn't match expected input patch dim ({expected_T_high_input}) for scale {original_high_res_scale_index}")
            high_res_trend = high_res_trend.permute(0, 2, 3, 1) + res
            current_low_res_trend = high_res_trend.permute(0, 3, 1, 2)
            out_trend_list_rev.append(current_low_res_trend)
        out_trend_list_rev.reverse()
        return out_trend_list_rev


class PDM_Deep_Block(nn.Module):
    """
    Past Decomposable Mixing Block for Deep Features (确保 T<=1 处理逻辑)
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
        for i, feature in enumerate(feature_list):
            B, T, N, D = feature.shape
            if T <= 1:
                season = torch.zeros_like(feature)
                trend = feature
                #print(f"Warning: Scale {i} feature length T={T} <= 1. Skipping decomposition, treating as trend.")
            else:
                feature_reshaped = feature.permute(0, 2, 1, 3).reshape(B * N, T, D)
                try:
                    season_r, trend_r = self.decomposition(feature_reshaped)
                    season = season_r.reshape(B, N, T, D).permute(0, 2, 1, 3)
                    trend = trend_r.reshape(B, N, T, D).permute(0, 2, 1, 3)
                except Exception as e:
                    print(f"Error during decomposition at scale {i} with input shape {feature_reshaped.shape}: {e}")
                    raise e
            season_list.append(season)
            trend_list.append(trend)
        try:
             mixed_season_list = self.mixing_multi_scale_season(season_list)
             mixed_trend_list = self.mixing_multi_scale_trend(trend_list)
        except ValueError as e:
             print(f"Error during mixing stage: {e}")
             print("Season list shapes before mixing:"); [print(f" Scale {idx}: {s.shape}") for idx, s in enumerate(season_list)]
             print("Trend list shapes before mixing:"); [print(f" Scale {idx}: {t.shape}") for idx, t in enumerate(trend_list)]
             raise e
        out_list = []
        num_scales = len(feature_list)
        if len(mixed_season_list) != num_scales or len(mixed_trend_list) != num_scales: raise RuntimeError(f"BUG: Mismatch in scales after mixing: Input ({num_scales}), Season ({len(mixed_season_list)}), Trend ({len(mixed_trend_list)}).")
        for i, (ori_feature, season, trend) in enumerate(zip(feature_list, mixed_season_list, mixed_trend_list)):
            if ori_feature.shape != season.shape or ori_feature.shape != trend.shape: raise RuntimeError(f"BUG: Shape mismatch before final projection at scale {i}. Original: {ori_feature.shape}, Season: {season.shape}, Trend: {trend.shape}.")
            fused_feature = self.out_projection[i](season + trend)
            out_list.append(ori_feature + fused_feature)
        return out_list