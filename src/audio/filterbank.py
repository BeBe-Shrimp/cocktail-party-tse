"""Auditory filterbanks: Gammatone and Mel implementations.

Gammatone filterbanks model the frequency selectivity of the human cochlea,
providing a biologically-inspired alternative to conventional STFT or Mel
filterbanks for audio front-end processing.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class GammatoneFilterbank(nn.Module):
    """Gammatone filterbank — a biologically-inspired model of cochlear filtering.

    The Gammatone filter is a widely-used model of the auditory periphery.
    Each filter approximates the impulse response of the basilar membrane
    at a specific characteristic frequency, with bandwidths that increase
    with center frequency (following the Equivalent Rectangular Bandwidth scale).

    This implementation generates Gammatone filters in the frequency domain
    and applies them via multiplication in the STFT domain for efficiency.

    Args:
        n_filters: Number of Gammatone filters (cochlear channels).
        sample_rate: Audio sample rate.
        f_min: Minimum center frequency (Hz).
        f_max: Maximum center frequency (Hz, defaults to Nyquist).
        n_fft: FFT size for frequency-domain filtering.
        trainable: If True, filter center frequencies and bandwidths are learnable.
    """

    def __init__(
        self,
        n_filters: int = 64,
        sample_rate: int = 16000,
        f_min: float = 80.0,
        f_max: Optional[float] = None,
        n_fft: int = 512,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        self.n_filters = n_filters
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max or sample_rate / 2
        self.n_fft = n_fft

        # Center frequencies on ERB scale
        cf = self._erb_scale_frequencies(n_filters, self.f_min, self.f_max)
        if trainable:
            self.cf = nn.Parameter(torch.tensor(cf, dtype=torch.float32))
        else:
            self.register_buffer("cf", torch.tensor(cf, dtype=torch.float32))

        # Build filterbank
        filters = self._build_gammatone_filters()
        if trainable:
            self.filters = nn.Parameter(filters)
        else:
            self.register_buffer("filters", filters)

    def _erb_scale_frequencies(self, n: int, f_min: float, f_max: float) -> list:
        """Compute n center frequencies equally spaced on the ERB scale.

        The Equivalent Rectangular Bandwidth (ERB) scale relates frequency
        to position along the basilar membrane: ERB(f) = 24.7 * (4.37*f/1000 + 1)
        """
        erb_min = 24.7 * (4.37 * f_min / 1000 + 1)
        erb_max = 24.7 * (4.37 * f_max / 1000 + 1)
        erb_points = torch.linspace(erb_min, erb_max, n + 2)[1:-1]  # exclude edges

        # Inverse: f = (ERB/24.7 - 1) * 1000 / 4.37
        frequencies = (erb_points / 24.7 - 1) * 1000 / 4.37
        return frequencies.tolist()

    def _gammatone_impulse_response(self, cf: float, bandwidth: float, t: torch.Tensor) -> torch.Tensor:
        """Gammatone impulse response: t^(n-1) * exp(-2*pi*b*t) * cos(2*pi*cf*t).

        Uses n=4 (4th-order Gammatone), which provides a good fit to human
        auditory nerve fiber responses.

        Args:
            cf: Center frequency (Hz).
            bandwidth: Equivalent rectangular bandwidth (Hz).
            t: Time indices (seconds).

        Returns:
            Impulse response of the Gammatone filter.
        """
        n = 4  # filter order
        b = 1.019 * bandwidth  # bandwidth parameter
        envelope = t ** (n - 1) * torch.exp(-2 * math.pi * b * t)
        carrier = torch.cos(2 * math.pi * cf * t)
        gammatone = envelope * carrier

        # Normalize
        gammatone = gammatone / (torch.norm(gammatone) + 1e-8)
        return gammatone

    def _build_gammatone_filters(self) -> torch.Tensor:
        """Build Gammatone filterbank in the frequency domain.

        Returns:
            Filterbank tensor of shape ``(n_filters, n_fft // 2 + 1)``.
        """
        # Time axis (centered around 0 for linear phase)
        t = torch.arange(self.n_fft, dtype=torch.float32) / self.sample_rate
        t = t - t.mean()

        filters = []
        for i in range(self.n_filters):
            cf = self.cf[i].item()
            # ERB bandwidth
            erb = 24.7 * (4.37 * cf / 1000 + 1)
            # Gammatone impulse response
            ir = self._gammatone_impulse_response(cf, erb, t)
            # Frequency response via FFT
            fr = torch.fft.rfft(ir, n=self.n_fft)
            # Magnitude response
            mag = fr.abs()
            filters.append(mag)

        filters = torch.stack(filters, dim=0)  # (n_filters, n_fft//2+1)

        # Normalize each filter to unit max
        max_vals = filters.max(dim=1, keepdim=True)[0]
        filters = filters / (max_vals + 1e-8)

        return filters

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply Gammatone filterbank to a magnitude spectrogram.

        Args:
            spectrogram: Magnitude spectrogram ``(B, F_stft, T)``.
                         ``F_stft`` must match ``n_fft // 2 + 1``.

        Returns:
            Filtered output ``(B, n_filters, T)`` — each channel corresponds
            to the response of one Gammatone filter.
        """
        # filters: (n_filters, n_fft//2+1)
        # spectrogram: (B, F_stft, T)
        if spectrogram.dim() == 3:
            # Batch multiply: (1, n_filters, F) @ (B, F, T) -> (B, n_filters, T)
            output = torch.matmul(
                self.filters.unsqueeze(0),  # (1, n_filters, F)
                spectrogram,  # (B, F, T)
            )
        else:
            output = torch.matmul(self.filters, spectrogram)
        return output


class MelFilterbank(nn.Module):
    """Mel-scale filterbank — perceptually-motivated frequency decomposition.

    The Mel scale approximates the human ear's non-linear frequency perception,
    with higher resolution at low frequencies and lower resolution at high
    frequencies. This is a standard front-end for many audio tasks.

    Args:
        n_filters: Number of Mel filters.
        sample_rate: Audio sample rate.
        f_min: Minimum frequency.
        f_max: Maximum frequency.
        n_fft: FFT size.
    """

    def __init__(
        self,
        n_filters: int = 80,
        sample_rate: int = 16000,
        f_min: float = 80.0,
        f_max: Optional[float] = None,
        n_fft: int = 512,
    ) -> None:
        super().__init__()
        self.n_filters = n_filters
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max or sample_rate / 2
        self.n_fft = n_fft

        filters = self._build_mel_filters()
        self.register_buffer("filters", filters)

    def _hz_to_mel(self, freq: torch.Tensor) -> torch.Tensor:
        """Convert Hz to Mel scale."""
        return 2595.0 * torch.log10(1.0 + freq / 700.0)

    def _mel_to_hz(self, mel: torch.Tensor) -> torch.Tensor:
        """Convert Mel scale to Hz."""
        return 700.0 * (10 ** (mel / 2595.0) - 1.0)

    def _build_mel_filters(self) -> torch.Tensor:
        """Build triangular Mel filterbank.

        Returns:
            Filterbank ``(n_filters, n_fft // 2 + 1)``.
        """
        n_freqs = self.n_fft // 2 + 1

        # Mel points
        mel_min = self._hz_to_mel(torch.tensor(float(self.f_min)))
        mel_max = self._hz_to_mel(torch.tensor(float(self.f_max)))
        mel_points = torch.linspace(mel_min, mel_max, self.n_filters + 2)
        hz_points = self._mel_to_hz(mel_points)

        # Convert to FFT bin indices
        bin_points = torch.floor((self.n_fft + 1) * hz_points / self.sample_rate).long()
        bin_points = torch.clamp(bin_points, 0, n_freqs - 1)

        # Build triangular filters
        filters = torch.zeros(self.n_filters, n_freqs)
        for i in range(self.n_filters):
            left = bin_points[i].item()
            center = bin_points[i + 1].item()
            right = bin_points[i + 2].item()

            if left < center:
                filters[i, left:center] = torch.linspace(0, 1, center - left)
            if center < right:
                filters[i, center:right] = torch.linspace(1, 0, right - center)

        return filters

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply Mel filterbank.

        Args:
            spectrogram: Magnitude spectrogram ``(B, F_stft, T)``.

        Returns:
            Mel spectrogram ``(B, n_filters, T)``.
        """
        if spectrogram.dim() == 3:
            output = torch.matmul(
                self.filters.unsqueeze(0), spectrogram
            )
        else:
            output = torch.matmul(self.filters, spectrogram)
        return output
