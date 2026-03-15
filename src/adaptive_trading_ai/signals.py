from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt, hilbert


@dataclass
class SignalProcessor:
    sample_rate: float = 1.0

    def bandpass(self, series: pd.Series, low: float, high: float, order: int = 3) -> pd.Series:
        nyquist = 0.5 * self.sample_rate
        b, a = butter(order, [low / nyquist, high / nyquist], btype="band")
        filtered = filtfilt(b, a, series.values)
        return pd.Series(filtered, index=series.index, name=f"band_{low}_{high}")


@dataclass
class FourierAnalyzer:
    window: int = 256
    top_k: int = 5
    sample_rate: float = 1.0

    def rolling_fft(self, series: pd.Series) -> pd.DataFrame:
        out = []
        values = series.values
        for i in range(self.window, len(values) + 1):
            w = values[i - self.window : i]
            spec = np.abs(rfft(w))
            freqs = rfftfreq(len(w), d=1 / self.sample_rate)
            top_idx = np.argsort(spec)[-self.top_k :][::-1]
            dominant = freqs[top_idx]
            amps = spec[top_idx] / len(w)
            out.append(
                {
                    "timestamp": series.index[i - 1],
                    "dominant_frequency": dominant[0] if len(dominant) else 0.0,
                    "dominant_strength": amps[0] if len(amps) else 0.0,
                    "spectral_entropy": self._spectral_entropy(spec),
                    "spectral_energy": float(np.sum(spec**2)),
                    "harmonic_frequencies": dominant,
                    "harmonic_amplitudes": amps,
                }
            )
        return pd.DataFrame(out).set_index("timestamp")

    def reconstruct_signal(self, series: pd.Series, frequencies: np.ndarray, amplitudes: np.ndarray) -> pd.Series:
        t = np.arange(len(series))
        recon = np.zeros_like(t, dtype=float)
        for f, a in zip(frequencies, amplitudes):
            omega = 2 * np.pi * f
            phase = np.angle(np.sum(series.values * np.exp(-1j * omega * t)))
            recon += a * np.sin(omega * t + phase)
        return pd.Series(recon, index=series.index, name="reconstructed_cycle")

    @staticmethod
    def _spectral_entropy(spec: np.ndarray) -> float:
        psd = spec**2
        psd /= np.sum(psd) + 1e-12
        return float(-np.sum(psd * np.log(psd + 1e-12)))


class WaveletAnalyzer:
    def transform(self, series: pd.Series, wavelet: str = "morl", scales: np.ndarray | None = None) -> Dict[str, np.ndarray]:
        import pywt

        if scales is None:
            scales = np.arange(1, 64)
        coeffs, freqs = pywt.cwt(series.values, scales=scales, wavelet=wavelet)
        power = np.abs(coeffs) ** 2
        return {"coefficients": coeffs, "frequencies": freqs, "power": power, "scales": scales}


class HilbertCycleDetector:
    def detect(self, series: pd.Series) -> pd.DataFrame:
        analytic = hilbert(series.values)
        amplitude = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))
        inst_freq = np.diff(phase, prepend=phase[0]) / (2 * np.pi)
        return pd.DataFrame(
            {
                "amplitude": amplitude,
                "phase": phase,
                "sin_phase": np.sin(phase),
                "cos_phase": np.cos(phase),
                "instantaneous_frequency": inst_freq,
            },
            index=series.index,
        )


@dataclass
class KalmanCycleTracker:
    process_var: float = 1e-5
    obs_var: float = 1e-2

    def track(self, observation: pd.Series) -> pd.DataFrame:
        x = np.array([observation.iloc[0], 0.0])
        p = np.eye(2)
        f = np.array([[1, 1], [0, 1]])
        h = np.array([[1, 0]])
        q = self.process_var * np.eye(2)
        r = np.array([[self.obs_var]])

        states: List[np.ndarray] = []
        for z in observation.values:
            x = f @ x
            p = f @ p @ f.T + q
            y = np.array([[z]]) - h @ x[:, None]
            s = h @ p @ h.T + r
            k = p @ h.T @ np.linalg.inv(s)
            x = x + (k @ y).ravel()
            p = (np.eye(2) - k @ h) @ p
            states.append(x.copy())

        arr = np.array(states)
        return pd.DataFrame(
            {
                "kalman_level": arr[:, 0],
                "kalman_trend": arr[:, 1],
            },
            index=observation.index,
        )
