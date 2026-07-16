from __future__ import annotations

import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d.global_inner_projection import (
    global_supersonic_prefix_cell_count,
)


def test_supersonic_prefix_count_stops_at_first_subsonic_cell() -> None:
    assert global_supersonic_prefix_cell_count(
        np.array([-6.0, -2.0, -0.9, -1.5])
    ) == 2


@pytest.mark.parametrize(
    "values",
    (
        np.array([-0.5, -0.2]),
        np.array([-2.0, -1.5]),
        np.array([-2.0, np.nan, -0.2]),
        np.empty(0),
    ),
)
def test_supersonic_prefix_count_rejects_invalid_boundary_topologies(
    values: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        global_supersonic_prefix_cell_count(values)
