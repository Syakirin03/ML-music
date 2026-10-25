"""config.toml: defaults, settings that replace them, and mistakes caught early."""

import pytest

from recommender.config import DEFAULTS, ROOT, load_config


def test_no_file_gives_the_defaults(tmp_path):
    assert load_config(tmp_path / "config.toml") == DEFAULTS


def test_the_example_shows_every_default():
    # config.example.toml documents each setting with its default value: the two must stay in step.
    assert load_config(ROOT / "config.example.toml") == DEFAULTS


def test_a_setting_replaces_its_default_and_leaves_the_others(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[home]\ntimezone = "Europe/London"\n\n[taste]\nless_western = 1\n', encoding="utf-8")
    config = load_config(path)
    assert config["home"] == {"timezone": "Europe/London", "market": ""}
    assert config["taste"]["less_western"] == 1.0 and isinstance(config["taste"]["less_western"], float)
    assert config["pool"] == DEFAULTS["pool"]


def test_changing_a_loaded_setting_leaves_the_defaults_alone(tmp_path):
    load_config(tmp_path / "config.toml")["genres"]["extra"].append("white noise")
    assert DEFAULTS["genres"]["extra"] == []


@pytest.mark.parametrize("text", [
    '[hom]\ntimezone = "Europe/London"\n',  # a section that doesn't exist
    '[home]\ntime_zone = "Europe/London"\n',  # a setting that doesn't exist
    '[taste]\nmore_genres = "k-pop"\n',  # text where a list belongs
    '[taste]\nless_western = true\n',  # yes or no where a number belongs
])
def test_mistakes_are_caught(tmp_path, text):
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)
