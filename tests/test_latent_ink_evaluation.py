import pytest

from research.corpus.latent_ink_evaluation import capacity_gate


def fixture_rows():
    key = [{"anonymous_id": f"item-{i}.png", "text": "of", "sha256": str(i)} for i in range(24)]
    observed = [
        {"id": f"item-{i}", "transcription": "of", "complete_legible": True, "image_sha256": str(i)}
        for i in range(24)
    ]
    return key, observed


@pytest.mark.parametrize(
    "complete,decision",
    [
        (11, "stop"),
        (12, "one-capacity-extension-only"),
        (19, "one-capacity-extension-only"),
        (20, "expand"),
    ],
)
def test_fixed_capacity_gate(complete, decision):
    key, observed = fixture_rows()
    for q in observed[complete:]:
        q["complete_legible"] = False
    result = capacity_gate(key, observed, mean_dice=0.75, blank=0, severe_defects=False)
    assert result["decision"] == decision
    assert result["complete_and_exact"] == complete


def test_shape_score_cannot_override_content_and_missing_observations():
    key, observed = fixture_rows()
    for q in observed:
        q["transcription"] = "off"
    assert (
        capacity_gate(key, observed, mean_dice=0.99, blank=0, severe_defects=False)["decision"]
        == "stop"
    )
    with pytest.raises(ValueError, match="24"):
        capacity_gate(key, observed[:-1], mean_dice=0.99, blank=0, severe_defects=False)
    observed[0]["image_sha256"] = "other-image"
    with pytest.raises(ValueError, match="frozen"):
        capacity_gate(key, observed, mean_dice=0.99, blank=0, severe_defects=False)


def test_blank_severe_defects_and_low_dice_block_expansion():
    key, observed = fixture_rows()
    for options in [
        {"mean_dice": 0.3, "blank": 0, "severe_defects": False},
        {"mean_dice": 0.8, "blank": 1, "severe_defects": False},
        {"mean_dice": 0.8, "blank": 0, "severe_defects": True},
    ]:
        assert capacity_gate(key, observed, **options)["decision"] == "stop"
