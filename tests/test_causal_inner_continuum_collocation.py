import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_continuum_truncation import (
    causal_sixth_order_inward_collocation_derivative,
)


def test_sixth_order_inward_derivative_for_outer_buffered_wave():
    errors = []
    for count in (65, 129, 257):
        nodes = np.linspace(0.0, 1.0, count)
        values = np.sin(np.pi * nodes) ** 8
        exact = (
            8.0
            * np.pi
            * np.sin(np.pi * nodes) ** 7
            * np.cos(np.pi * nodes)
        )
        derivative = causal_sixth_order_inward_collocation_derivative(nodes)
        errors.append(
            float(
                np.sqrt(
                    np.mean(
                        (
                            np.asarray(derivative @ values).ravel()
                            - exact
                        )
                        ** 2
                    )
                )
            )
        )
    assert np.log2(errors[0] / errors[1]) >= 5.5
    assert np.log2(errors[1] / errors[2]) >= 5.5
