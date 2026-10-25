"""Shared set-up: the tests run on made-up data and the default settings, never on yours.

This runs before any test imports the recommender, so the settings come from config.example.toml (the
defaults) instead of your config.toml, and the recommender's own files go to an empty scratch folder instead
of data/user/, where your ratings are.
"""

import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["MUSIC_PREFERENCE_CONFIG"] = str(ROOT / "config.example.toml")
os.environ["MUSIC_RECOMMENDER_USER_DIR"] = tempfile.mkdtemp(prefix="music-preference-tests-")
