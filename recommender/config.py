"""Your personal settings, from config.toml in the project folder.

Copy config.example.toml to config.toml and change what you like. config.toml is ignored by git, so what you
put there (your time zone, your favourite artists, genres you want more of, ...) stays on your computer. A
setting you leave out keeps its default: the value config.example.toml shows. The environment variable
MUSIC_PREFERENCE_CONFIG, if set, points to another file instead, such as a test's.

The scripts in scripts/ import this module too, so the pipeline and the recommender read the same settings.
"""

import copy
import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = Path(os.environ.get("MUSIC_PREFERENCE_CONFIG") or ROOT / "config.toml")

# Every setting and its default, by section. config.example.toml explains each one.
DEFAULTS = {
    "home": {"timezone": "", "market": ""},
    "genres": {"extra": [], "aliases": {}},
    "not_music": {"genres": [], "pattern": ""},
    "pool": {"favourite_artists": []},
    "taste": {"more_genres": [], "genre_boost": 0.25, "less_western": 0.0},
    "privacy": {"allow": []},
}


def load_config(path: Path = CONFIG_FILE) -> dict[str, dict]:
    """The settings in path over the defaults, or the defaults alone when there is no such file.

    Raises ValueError for a section or setting the defaults don't have (usually a typo) and for a value of
    the wrong kind, such as text where a list belongs.
    """
    found = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    config = copy.deepcopy(DEFAULTS)
    for section, values in found.items():
        if section not in DEFAULTS or not isinstance(values, dict):
            raise ValueError(f"{path.name}: there is no section [{section}]; config.example.toml lists them all")
        for name, value in values.items():
            if name not in DEFAULTS[section]:
                raise ValueError(f"{path.name}: [{section}] has no setting {name}; config.example.toml lists them all")
            default = DEFAULTS[section][name]
            if isinstance(default, float) and isinstance(value, int) and not isinstance(value, bool):
                value = float(value)
            if type(value) is not type(default):
                raise ValueError(f"{path.name}: [{section}] {name} should be {type(default).__name__}, not {value!r}")
            config[section][name] = value
    return config


CONFIG = load_config()
