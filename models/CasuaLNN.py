import torch
import torch.nn as nn

from layers.Embed import PatchEmbedding as OriginalPatchEmbedding, PositionalEmbedding
from layers.CasuaLNN_layers import PDM_Deep_Block, get_num_patches
from layers.StandardNorm import Normalize
from layers.CausalTransformer_EncDec import CausalEncoder, CausalEncoderLayer
from layers.CausalAttention_Family import CausalFullAttention, CausalAttentionLayer
from layers.DynamicLiquid import DynamicLiquidEncoder, DynamicLiquidLayer


class EnhancedPatchEmbedding(nn.Module):
    """
    Enhanced PatchEmbedding using 1D-CNN.
    """

    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(EnhancedPatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.dropout = nn.Dropout(dropout)
        self.value_embedding = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=d_model // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(in_channels=d_model // 2, out_channels=d_model, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1)
        )
        self.position_embedding = PositionalEmbedding(d_model)

    def forward(self, x):
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        B, N, Np, P = x.shape
        x_embed = self.value_embedding(x.reshape(-1, 1, P))
        x_embed = x_embed.reshape(B * N, Np, -1)
        x_out = x_embed + self.position_embedding(x_embed)
        return self.dropout(x_out), n_vars


class CasuaLNN(nn.Module):
    """
    CasuaLNN: Causal Liquid Neural Network Model.
    """

    def __init__(self, configs):
        super(CasuaLNN, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_vars = configs.enc_in

        # 1. Multi-scale pyramid
        self.down_sampling_window = configs.down_sampling_window
        if self.configs.down_sampling_method == 'avg':
            self.down_pool = torch.nn.AvgPool1d(kernel_size=self.down_sampling_window, stride=self.down_sampling_window)
        elif self.configs.down_sampling_method == 'max':
            self.down_pool = torch.nn.MaxPool1d(kernel_size=self.down_sampling_window, stride=self.down_sampling_window)

        self.normalize_layers = torch.nn.ModuleList([
            Normalize(configs.enc_in, affine=True, non_norm=configs.use_norm == 0)
            for _ in range(configs.down_sampling_layers + 1)
        ])

        # 2. Learnable shared causal graph (logits) across scales
        self.causal_graph_logits = nn.Parameter(
            torch.randn(self.n_vars, self.n_vars)
        )
        nn.init.xavier_uniform_(self.causal_graph_logits)
        print(f">>> [CasuaLNN] Initialized shared causal graph (nn.Parameter) with shape: [{self.n_vars}, {self.n_vars}]")

        # 3. Intra-scale encoding blocks (IPT_Block)
        # The shared causal_graph_logits is injected into each block.
        self.ipt_blocks = nn.ModuleList([
            IPT_Block_Encoder(configs,
                              patch_len=configs.patch_len,
                              stride=configs.stride,
                              scale_idx=i,
                              use_enhanced_embedding=configs.use_enhanced_embedding,
                              use_dynamic_ode=configs.use_dynamic_ode,
                              causal_graph_logits=self.causal_graph_logits
                              )
            for i in range(configs.down_sampling_layers + 1)
        ])

        # 4. Cross-scale mixing blocks (PDM)
        self.pdm_blocks = nn.ModuleList([
            PDM_Deep_Block(configs) for _ in range(configs.e_layers_pdm)
        ])

        # 5. Prediction heads
        self.pred_heads = nn.ModuleList()
        for i in range(configs.down_sampling_layers + 1):
            scale_seq_len = self.seq_len // (self.down_sampling_window ** i)
            num_patches = get_num_patches(scale_seq_len, configs.patch_len, configs.stride)
            if num_patches <= 0:
                print(f"Warning: Model __init__ (scale {i}) resulted in 0 patches. Prediction head might receive 0 input.")
                head_nf = configs.d_model
            else:
                head_nf = configs.d_model * num_patches

            self.pred_heads.append(
                nn.Sequential(
                    nn.Flatten(start_dim=-2),
                    nn.Linear(head_nf, configs.pred_len)
                )
            )

    def __multi_scale_process_inputs(self, x_enc):
        x_enc_list = [x_enc]
        x_enc_temp = x_enc.permute(0, 2, 1)
        for _ in range(self.configs.down_sampling_layers):
            x_enc_temp = self.down_pool(x_enc_temp)
            x_enc_list.append(x_enc_temp.permute(0, 2, 1))
        return x_enc_list

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if x_enc.dim() == 4:
            x_enc = x_enc[..., 0]

        x_list = self.__multi_scale_process_inputs(x_enc)

        ipt_outputs = []
        for i, x_k in enumerate(x_list):
            x_k_norm = self.normalize_layers[i](x_k, 'norm')
            # The IPT_Block_Encoder now internally handles the complex logic of
            # CausalAttention -> Parameter Generation -> LNN.
            ipt_out = self.ipt_blocks[i](x_k_norm)
            ipt_outputs.append(ipt_out)

        pdm_inputs = [out.permute(0, 2, 1, 3) for out in ipt_outputs]
        mixed_outputs = pdm_inputs
        for pdm_block in self.pdm_blocks:
            mixed_outputs = pdm_block(mixed_outputs)

        y_preds = []
        for i, h_k in enumerate(mixed_outputs):
            h_k_permuted = h_k.permute(0, 2, 1, 3)
            y_pred = self.pred_heads[i](h_k_permuted)
            y_pred = y_pred.permute(0, 2, 1)
            y_pred = self.normalize_layers[i](y_pred, 'denorm')
            y_preds.append(y_pred)

        final_pred = torch.stack(y_preds, dim=-1).mean(dim=-1)

        # Note: The causal loss components (e.g., DAG constraint) are calculated
        # independently in the training loop by accessing self.causal_graph_logits.
        # This function only needs to return the final prediction.
        return final_pred

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None


class IPT_Block_Encoder(nn.Module):
    """
    IPT_Block_Encoder (Causal Attention + Dynamic ODE)

    This module implements the flow: [iT CausalController -> Parameter Generator -> LNN Executor].
    It uses a Causal Transformer as a controller to generate parameters for a Dynamic Liquid Encoder.
    """

    def __init__(self, configs, patch_len=16, stride=8, scale_idx=0,
                 use_enhanced_embedding=False,
                 use_dynamic_ode=False,
                 causal_graph_logits=None
                 ):

        super(IPT_Block_Encoder, self).__init__()
        self.d_model = configs.d_model
        self.use_dynamic_ode = use_dynamic_ode
        self.causal_graph_logits = causal_graph_logits

        if self.causal_graph_logits is None:
            print(f"Warning: IPT_Block (Scale {scale_idx}) did not receive causal_graph_logits. Causal Attention will be disabled.")

        # 1. Patching module
        if use_enhanced_embedding:
            self.patch_embedding = EnhancedPatchEmbedding(
                configs.d_model, patch_len, stride, stride, configs.dropout)
        else:
            self.patch_embedding = OriginalPatchEmbedding(
                configs.d_model, patch_len, stride, stride, configs.dropout)

        if self.use_dynamic_ode:
            print(f">>> [CasuaLNN] IPT_Block (Scale {scale_idx}) uses [Dynamic ODE] structure with [Causal Attention Controller].")
            self.lq_param_scaling = configs.lq_param_scaling
            self.tanh = nn.Tanh()

            # Module 1: iT Causal Controller
            self.iT_controller = CausalEncoder(
                [
                    CausalEncoderLayer(
                        CausalAttentionLayer(
                            CausalFullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation
                    ) for _ in range(configs.e_layers_controller)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model)
            )

            # Module 2: Parameter Generator
            self.param_gen_output_dim = configs.lq_units * 2
            self.param_generator = nn.Linear(configs.d_model, self.param_gen_output_dim)

            # Module 3: LNN Executor
            self.LNN_executor = DynamicLiquidEncoder(
                [
                    DynamicLiquidLayer(
                        configs=configs,
                        dropout=configs.dropout
                    ) for _ in range(configs.e_layers_pdm)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model)
            )

        else:
            # Fallback logic: Use only the Causal Inverted-Attention (iT) structure.
            print(f">>> [CasuaLNN] IPT_Block (Scale {scale_idx}) uses [Causal Inverted-Attention] (iT) structure only.")
            self.encoder = CausalEncoder(
                [
                    CausalEncoderLayer(
                        CausalAttentionLayer(
                            CausalFullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation
                    ) for _ in range(configs.e_layers_ipt)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model)
            )

    def forward(self, x):
        B, L_k, N = x.shape
        x = x.permute(0, 2, 1)

        # 1. Patching
        patch_out, n_vars = self.patch_embedding(x)
        N_p = patch_out.shape[1]

        if self.use_dynamic_ode:
            # --- Causal Controller Path ---
            # 2a. Reshape for iT Causal Controller
            controller_in = patch_out.reshape(B, N, N_p, self.d_model).permute(0, 2, 1, 3).reshape(B * N_p, N, self.d_model)

            # 3a. Run iT Causal Controller, injecting the causal mask
            correlation_state, _ = self.iT_controller(
                controller_in,
                causal_mask=self.causal_graph_logits
            )

            # 4a. Run Parameter Generator
            unscaled_params = self.param_generator(correlation_state)
            dynamic_params = self.lq_param_scaling * self.tanh(unscaled_params)

            # 5a. Align parameters for LNN Executor
            dynamic_params_seq = dynamic_params.reshape(B, N_p, N, -1).permute(0, 2, 1, 3).reshape(B * N, N_p, -1)

            # --- Executor Path ---
            # 6a. Run LNN Executor with dynamic parameters
            lnn_output, _ = self.LNN_executor(patch_out, dynamic_params=dynamic_params_seq)

            # 7a. Reshape final output
            final_out = lnn_output.reshape(B, N, N_p, self.d_model)

        else:
            # --- Fallback Causal iT Path ---
            # 2b. Reshape for Causal iT
            ipt_in = patch_out.reshape(B, N, N_p, self.d_model).permute(0, 2, 1, 3).reshape(B * N_p, N, self.d_model)

            # 3b. Run Causal iT, injecting the causal mask
            ipt_out, _ = self.encoder(
                ipt_in,
                causal_mask=self.causal_graph_logits
            )

            # 4b. Reshape back
            final_out = ipt_out.reshape(B, N_p, N, self.d_model).permute(0, 2, 1, 3)

        return final_out