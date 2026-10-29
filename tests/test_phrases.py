from openmixing.player.phrases import phrase_beats

STRUCT = {"phrase_bars": 8, "sections": [
    {"label": "intro", "start_bar": 0, "end_bar": 12}, {"label": "build", "start_bar": 12, "end_bar": 28},
    {"label": "drop", "start_bar": 28, "end_bar": 40}]}


def test_sections_split_by_phrase_and_short_tail_merged():
    out = phrase_beats(STRUCT, 0, 160)
    assert [(p["kind"], p["beat"] / 4, p["end_beat"] / 4) for p in out] == [
        ("intro", 0, 12), ("up", 12, 20), ("up", 20, 28), ("chorus", 28, 40)]


def test_clipped_to_clip_window():
    out = phrase_beats(STRUCT, 48, 40)
    assert out[0]["beat"] == 0 and out[-1]["end_beat"] == 40 and all(p["beat"] < p["end_beat"] for p in out)
