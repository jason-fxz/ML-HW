import torch
import numpy as np
from statsmodels.tsa.ar_model import AutoReg

class ARAligner:
    """
    Align the output prediction mean based on short-term AutoReg (AR) trend forecasting.
    """
    def __init__(self, ratio: float = 0.1, lags: int = 5, target_dims=None):
        """
        Args:
            ratio: Ratio of input sequence to use as AR training data.
            lags: Number of lags to use in AutoReg model.
            target_dims: Optional list of dimensions to apply alignment (default: all).
        """
        self.ratio = ratio
        self.lags = lags
        self.target_dims = target_dims  # list of dim indexes to align

    def align(self, y_pred: torch.Tensor, x_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: [B, T_out, D] tensor
            x_input: [B, T_in, D] tensor

        Returns:
            y_pred_aligned: [B, T_out, D]
        """
        B, T_out, D = y_pred.shape
        T_in = x_input.shape[1]
        tail_len = int(T_in * self.ratio)

        aligned = y_pred.clone()

        for b in range(B):
            for d in range(D):
                if self.target_dims is not None and d not in self.target_dims:
                    continue

                # extract tail of input sequence
                series = x_input[b, -tail_len:, d].detach().cpu().numpy()

                try:
                    model = AutoReg(series, lags=min(self.lags, len(series) - 1), old_names=False).fit()
                    forecast = model.predict(start=len(series), end=len(series) + T_out - 1)
                    offset = forecast.mean() - y_pred[b, :, d].mean().item()
                    aligned[b, :, d] += offset
                except Exception as e:
                    print(f"[ARAligner] Warning: AR failed on batch {b}, dim {d}. Error: {e}")

        return aligned.to(y_pred.device)