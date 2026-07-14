from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d import GlobalAdaptiveStepConfig


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_global_roche_adaptive_preflight import (  # noqa: E402
    _final_step_config,
    _target_time_tolerance,
)


def test_final_step_config_lands_below_controller_minimum() -> None:
    config = GlobalAdaptiveStepConfig(minimum_dt=1.0, maximum_dt=10.0)

    assert _final_step_config(config, 2.0) is config
    shortened = _final_step_config(config, 0.25)

    assert shortened.minimum_dt == 0.25
    assert shortened.maximum_dt == config.maximum_dt
    assert shortened.maximum_log_temperature_change == (
        config.maximum_log_temperature_change
    )
    with pytest.raises(ValueError, match="positive"):
        _final_step_config(config, 0.0)


def test_target_time_tolerance_is_roundoff_only() -> None:
    target = 1.5
    tolerance = _target_time_tolerance(target)

    assert tolerance >= abs(np.nextafter(target, np.inf) - target)
    assert tolerance < 1.0e-12 * target
