import torch

"""
This script provides utility classes for generating attention masks used in
Transformer-based models, such as standard causal masks and probabilistic sparse masks.
"""

class TriangularCausalMask():
    """
    Creates a standard causal (triangular) mask to prevent attention to future positions.

    In a transformer decoder, this ensures that the prediction for a position can only
    depend on the known outputs at previous positions.
    """
    def __init__(self, B: int, L: int, device: str = "cpu"):
        """
        Initializes the causal mask.

        Args:
            B (int): Batch size.
            L (int): Sequence length.
            device (str, optional): The device to create the mask on. Defaults to "cpu".
        """
        # Define the shape of the mask: (Batch, Heads, Seq_Len, Seq_Len)
        # The number of heads is set to 1, as the mask is broadcastable.
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            # Create an upper triangular matrix of ones.
            # `torch.triu` with `diagonal=1` zeros out the main diagonal and everything below it.
            # The result is a boolean mask where `True` indicates a masked (forbidden) position.
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self) -> torch.Tensor:
        """
        Returns the generated causal mask tensor.
        """
        return self._mask


class ProbMask():
    """
    Creates a probabilistic sparse attention mask based on the "ProbSparse" mechanism
    from the Informer paper (https://arxiv.org/abs/2012.07436).

    Instead of attending to all keys, each query attends only to a subset of the most
    "important" keys, determined by their attention scores. This class generates the
    corresponding mask.
    """
    def __init__(self, B: int, H: int, L: int, index: torch.Tensor, scores: torch.Tensor, device: str = "cpu"):
        """
        Initializes the probabilistic sparse mask.

        Args:
            B (int): Batch size.
            H (int): Number of attention heads.
            L (int): Sequence length.
            index (torch.Tensor): The indices of the top-k most important queries.
                                  Shape: (B, H, num_top_queries).
            scores (torch.Tensor): The attention scores tensor. Used to determine the final mask shape.
                                   Shape: (B, H, L, S) where L is query length and S is key length.
            device (str, optional): The device to create the mask on. Defaults to "cpu".
        """
        # Create a base upper-triangular mask to ensure causality.
        # Shape: (L, S) where S is the key sequence length.
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        
        # Expand the base mask to match the batch and head dimensions.
        # Shape: (B, H, L, S)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        
        # Use advanced indexing to select rows from the expanded mask.
        # `index` contains the indices of the top-k queries for each head and batch item.
        # This effectively creates a mask where only the selected "important" queries
        # are allowed to attend to all positions (up to the current one, due to the triu).
        # Shape: (B, H, num_top_queries, S)
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        
        # Reshape the final mask to match the shape of the attention scores tensor.
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self) -> torch.Tensor:
        """
        Returns the generated probabilistic sparse mask tensor.
        """
        return self._mask