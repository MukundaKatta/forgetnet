"""ForgettingCurveAnalyzer - models Ebbinghaus forgetting curves for AI."""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from forgetnet.models import ForgettingCurve, ForgettingCurvePoint


class ForgettingCurveAnalyzer:
    """Fits and analyzes Ebbinghaus-style forgetting curves.

    The Ebbinghaus forgetting curve is modeled as:
        R(t) = B * exp(-t / S)

    where:
        R(t) = retention at time t
        B    = initial memory strength (ideally 1.0)
        S    = stability (higher = slower forgetting)
        t    = time elapsed (or context distance)
    """

    @staticmethod
    def ebbinghaus(t: np.ndarray, b: float, s: float) -> np.ndarray:
        """Ebbinghaus forgetting curve: R(t) = B * exp(-t / S)."""
        return b * np.exp(-t / s)

    @staticmethod
    def power_law(t: np.ndarray, b: float, a: float) -> np.ndarray:
        """Power-law forgetting: R(t) = B * (1 + t)^(-a)."""
        return b * np.power(1.0 + t, -a)

    def fit_ebbinghaus(
        self,
        points: list[ForgettingCurvePoint],
    ) -> ForgettingCurve:
        """Fit an Ebbinghaus exponential forgetting curve to data."""
        t = np.array([p.time_step for p in points])
        r = np.array([p.retention_rate for p in points])
        if len(t) < 2:
            return ForgettingCurve(
                points=points, stability=1.0, initial_strength=1.0
            )
        # Normalize time to [0, 1] range for numerical stability
        t_max = t.max() if t.max() > 0 else 1.0
        t_norm = t / t_max
        try:
            popt, _ = curve_fit(
                self.ebbinghaus,
                t_norm,
                r,
                p0=[1.0, 0.5],
                bounds=([0.0, 1e-6], [2.0, 100.0]),
                maxfev=5000,
            )
            b, s_norm = popt
            s = s_norm * t_max  # scale back
            r_pred = self.ebbinghaus(t_norm, b, s_norm)
            ss_res = np.sum((r - r_pred) ** 2)
            ss_tot = np.sum((r - r.mean()) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        except RuntimeError:
            b, s, r_squared = 1.0, t_max / 2, 0.0
        return ForgettingCurve(
            points=points,
            stability=float(s),
            initial_strength=float(b),
            model_type="ebbinghaus",
            r_squared=float(r_squared),
        )

    def fit_power_law(
        self,
        points: list[ForgettingCurvePoint],
    ) -> ForgettingCurve:
        """Fit a power-law forgetting curve."""
        t = np.array([p.time_step for p in points])
        r = np.array([p.retention_rate for p in points])
        if len(t) < 2:
            return ForgettingCurve(points=points, model_type="power_law")
        try:
            popt, _ = curve_fit(
                self.power_law,
                t,
                r,
                p0=[1.0, 0.5],
                bounds=([0.0, 0.0], [2.0, 10.0]),
                maxfev=5000,
            )
            b, a = popt
            r_pred = self.power_law(t, b, a)
            ss_res = np.sum((r - r_pred) ** 2)
            ss_tot = np.sum((r - r.mean()) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        except RuntimeError:
            b, a, r_squared = 1.0, 0.5, 0.0
        return ForgettingCurve(
            points=points,
            stability=float(1.0 / a) if a > 0 else 1.0,
            initial_strength=float(b),
            model_type="power_law",
            r_squared=float(r_squared),
        )

    def predict_retention(
        self,
        curve: ForgettingCurve,
        time_step: float,
    ) -> float:
        """Predict retention at a given time step using the fitted curve."""
        b = curve.initial_strength
        s = curve.stability
        if curve.model_type == "ebbinghaus":
            return float(b * np.exp(-time_step / s))
        elif curve.model_type == "power_law":
            a = 1.0 / s if s > 0 else 0.5
            return float(b * (1.0 + time_step) ** (-a))
        return 0.5

    def half_life(self, curve: ForgettingCurve) -> float:
        """Calculate the half-life of memory (time to 50% retention)."""
        if curve.model_type == "ebbinghaus":
            # R(t) = B * exp(-t/S) = B/2 => t = S * ln(2)
            return curve.stability * np.log(2)
        elif curve.model_type == "power_law":
            # B * (1+t)^(-a) = B/2 => t = 2^(1/a) - 1
            a = 1.0 / curve.stability if curve.stability > 0 else 0.5
            return float(2.0 ** (1.0 / a) - 1.0) if a > 0 else float("inf")
        return 0.0
