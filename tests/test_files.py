"""Files the project writes: your Spotify login is readable by you alone, and replaced whole."""

import json
import os
import stat

import pytest

from recommender.spotify import write_private


def test_the_login_file_is_replaced_whole(tmp_path):
    path = tmp_path / "secrets" / "spotify_account.json"
    write_private(path, {"refresh_token": "old"})
    write_private(path, {"refresh_token": "new"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"refresh_token": "new"}
    assert not path.with_suffix(".tmp").exists()


@pytest.mark.skipif(os.name == "nt", reason="Windows ignores these permissions: the file gets its folder's")
def test_only_you_may_read_the_login_file(tmp_path):
    path = tmp_path / "spotify_account.json"
    write_private(path, {"refresh_token": "a token"})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
