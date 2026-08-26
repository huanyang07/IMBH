import run_causal_inner_projected_shear_hyperbolicity_blocker_certificate_wp10c9d6c7c3b5c4f25fizy as certificate


def test_manifest_authorizes_the_blocker_certificate():
    validated = certificate._validate_manifest(require_clean=False)
    assert validated["summary"]["blocker_certificate_authorized"]


def test_pass_classification_rejects_only_projected_shear():
    assert "one_amplitude_projected_shear_closure_rejected" in certificate.PASS_CLASSIFICATION


def test_selected_next_is_definitions_only_eleven_field_architecture():
    assert certificate.AUTHORIZED_NEXT.startswith("definitions_only_")
    assert "eleven_field" in certificate.AUTHORIZED_NEXT


def test_method_and_physical_failures_remain_distinct():
    assert certificate.METHOD_CLASSIFICATION != certificate.PHYSICAL_CLASSIFICATION
