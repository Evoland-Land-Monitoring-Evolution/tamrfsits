# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

"""
Contains Frequency Domain Analysis related functions
"""

import torch


def compute_fft_profile(
    data: torch.Tensor, s: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute fft profile for given tensor, using circles of increasing frequencies
    """
    # pylint: disable=not-callable
    freqs = torch.fft.fftshift(torch.fft.fftfreq(s, d=float(data.shape[-1]) / s))
    half_freqs = freqs[freqs.shape[0] // 2 :]

    freqs_x, freqs_y = torch.meshgrid(freqs, freqs, indexing="ij")

    freq_dist = torch.sqrt(freqs_x**2 + freqs_y**2)

    def single_date_fft(current_data):
        """
        For better readability
        """
        return torch.abs(
            # pylint: disable=not-callable
            torch.fft.fftshift(
                # pylint: disable=not-callable
                torch.fft.fft2(
                    current_data.to(dtype=torch.float32),
                    norm="backward",
                    s=[s, s],
                )
            )
        )

    def build_profile(fft_data: torch.Tensor) -> torch.Tensor:
        """
        Build fft profile
        """
        return torch.stack(
            [
                fft_data[:, torch.logical_and(f1 < freq_dist, freq_dist < f2)].mean(
                    dim=1
                )
                for f1, f2 in zip(half_freqs[:-1:2], half_freqs[1::2])
            ],
            dim=1,
        )

    fft_prof = torch.stack(
        [build_profile(single_date_fft(data[:, i, ...])) for i in range(data.shape[1])],
        dim=2,
    )

    freq_values = torch.tensor(
        [(f1 + f2) / 2 for f1, f2 in zip(half_freqs[:-1:2], half_freqs[1::2])]
    )
    return freq_values, fft_prof


def compute_frr_referenceless(
    predicted_logprof: torch.Tensor,
    input_logprof: torch.Tensor,
    fmin: float = 0.0,
    fmax: float = 1.0,
) -> torch.Tensor:
    """
    Referenceless version of FRR
    """

    idx_min = int(fmin * predicted_logprof.shape[1])
    idx_max = int(fmax * (predicted_logprof.shape[1] - 1))
    if idx_min == 0:
        idx_min = 1
    return (
        predicted_logprof[:, idx_min:idx_max, :] - input_logprof[:, idx_min:idx_max, :]
    ).mean(dim=1)
