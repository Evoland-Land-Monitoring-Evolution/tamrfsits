# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
This module contains the masked LPIPS loss
"""
import torch
from einops import repeat
from piq import LPIPS  # type: ignore
from piq.utils import _reduce  # type: ignore


def mask_to_nan(data: torch.Tensor, mask: torch.Tensor):
    """
    Set masked pixels to nan
    """
    broadcast_mask = repeat(mask, "b w h -> b 3 w h")
    return torch.masked_fill(data, broadcast_mask, torch.nan)


class MaskedLPIPSLoss(torch.nn.Module):
    """
    This class implements masking on top of the lpips loss
    """

    def __init__(self, lpips: LPIPS):
        """
        Class initializer
        """
        super().__init__()

        # Store the lpips loss
        self.lpips = lpips

    def forward(
        self, pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor | None:
        """
        Implement masked forward
        """
        if pred.shape[0] == 0:
            return None

        assert pred.shape == ref.shape
        assert mask.dim() == 3
        for i, j in enumerate((0, 2, 3)):
            assert mask.shape[i] == pred.shape[j]

        pred_features = self.lpips.get_features(pred)
        ref_features = self.lpips.get_features(ref)

        # Do we need the extra processing
        if not torch.all(mask):
            # First, we set masked data to NaN
            # Here we assume that only the reference needs to be masked
            ref_masked = mask_to_nan(ref, ~mask)

            # We used masked data to obtain a validity mask of vgg features
            with torch.no_grad():
                validity_mask = [
                    ~torch.isnan(f) for f in self.lpips.get_features(ref_masked)
                ]
        else:
            validity_mask = [torch.ones_like(f) for f in pred_features]

        nb_valid_features = sum(v.sum() for v in validity_mask)
        total_nb_features = sum(v.numel() for v in validity_mask)

        if nb_valid_features == 0:
            return None

        # Compute distances
        distances = self.lpips.compute_distance(pred_features, ref_features)

        # Same thing as in original code, but weighted by validity_mask
        # https://github.com/photosynthesis-team/piq/blob/
        # 9948a52fc09ac5f7fb3618ce64b7086f5c3109da/piq/perceptual.py#L183
        loss = torch.cat(
            [
                (w.to(d) * d * m).sum(dim=(2, 3))
                / (
                    torch.maximum(
                        m.sum(dim=(2, 3)),
                        torch.tensor(1e-6, device=d.device),
                    )
                )
                for d, w, m in zip(distances, self.lpips.weights, validity_mask)
            ],
            dim=1,
        ).sum(dim=1)

        # This scaling ensure that less valid features does not automatically
        # yield lower loss values
        return (total_nb_features / nb_valid_features) * _reduce(
            loss, self.lpips.reduction
        )
