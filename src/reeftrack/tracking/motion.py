"""Constant-velocity Kalman filter on the fish bounding box (the SAMURAI motion prior).

SAMURAI (Yang et al., 2024) showed that SAM2's memory tracker drifts onto look-alike
objects because it only uses appearance. A simple Kalman filter on box centre/size gives a
*prediction* of where the fish should be next; a mask far outside that prediction is a
strong hint of an identity switch (e.g. jumping to another triggerfish).

State x = [cx, cy, w, h, vcx, vcy, vw, vh]; measurement z = [cx, cy, w, h].
"""

from __future__ import annotations

import numpy as np


class BoxKalman:
    def __init__(self, box_cxcywh: np.ndarray, pos_std: float = 0.05, vel_std: float = 0.01):
        # Noise scales are relative to box size, so the filter behaves the same for a fish
        # that is 30 px or 300 px wide.
        self._pos_std = pos_std
        self._vel_std = vel_std
        self.x = np.zeros(8)
        self.x[:4] = box_cxcywh
        s = max(box_cxcywh[2], box_cxcywh[3], 1.0)
        self.P = np.diag(np.r_[np.full(4, (2 * pos_std * s) ** 2), np.full(4, (10 * vel_std * s) ** 2)])
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)

    def _scale(self) -> float:
        return max(self.x[2], self.x[3], 1.0)

    def predict(self) -> np.ndarray:
        s = self._scale()
        Q = np.diag(np.r_[np.full(4, (self._pos_std * s) ** 2), np.full(4, (self._vel_std * s) ** 2)])
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q
        return self.x[:4].copy()

    def _R(self) -> np.ndarray:
        return np.diag(np.full(4, (self._pos_std * self._scale()) ** 2))

    def update(self, z: np.ndarray) -> None:
        S = self.H @ self.P @ self.H.T + self._R()
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P

    def mahalanobis(self, z: np.ndarray) -> float:
        """Distance (in standard deviations) of a measured box centre from the prediction."""
        S = (self.H @ self.P @ self.H.T + self._R())[:2, :2]
        d = z[:2] - self.x[:2]
        return float(np.sqrt(d @ np.linalg.solve(S, d)))


def xyxy_to_cxcywh(b: tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = b
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0 + 1, y1 - y0 + 1], dtype=float)
