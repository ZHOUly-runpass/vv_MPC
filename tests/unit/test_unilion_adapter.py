from r680_safety_planner.backbones.unilion import spconv_layout_permutation


def test_legacy_spconv_kernel_layout_is_detected() -> None:
    assert spconv_layout_permutation(
        (128, 128, 3, 3, 3), (128, 3, 3, 3, 128)
    ) == (0, 2, 3, 4, 1)


def test_non_spconv_shape_mismatch_is_not_silently_converted() -> None:
    assert spconv_layout_permutation((2, 3), (3, 2)) is None
    assert spconv_layout_permutation((16, 16, 3, 3, 3), (16, 5, 5, 5, 16)) is None
