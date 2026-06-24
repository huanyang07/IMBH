"""Radial grid helpers for one-dimensional minidisk calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RadialGrid:
    """Finite-volume radial grid for an axisymmetric disk."""

    centers: np.ndarray
    edges: np.ndarray
    widths: np.ndarray
    area: np.ndarray


def make_log_grid(R_in: float, R_out: float, n: int) -> RadialGrid:
    """Return logarithmic radial cell centers, edges, widths, and areas."""

    if R_in <= 0.0:
        raise ValueError("R_in must be positive")
    if R_out <= R_in:
        raise ValueError("R_out must exceed R_in")
    if n < 1:
        raise ValueError("n must be at least 1")

    edges = np.geomspace(R_in, R_out, n + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    widths = np.diff(edges)
    area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    return RadialGrid(centers=centers, edges=edges, widths=widths, area=area)

