"""Nothing personal may be published: run before every commit (the pre-commit hook in .githooks does).

Every check reads what git would commit, the files in its index: a notebook you ran on your computer still
passes, since the nbstripout filter strips its outputs as it is added. The checks against your own files
(your keys in secrets/, config.toml, the artists in your history) are skipped where those files don't exist,
as on GitHub's servers. docs/privacy.md explains each.
"""

import json
import re
import subprocess
import tomllib
from pathlib import Path

import duckdb
import pytest

ROOT = Path(__file__).resolve().parent.parent
# Files that hold your data, keys or settings, which only .gitkeep and README.md files may stand in for.
PRIVATE = re.compile(
    r"^(data/(raw|interim|processed|user)/(?!\.gitkeep$)|data/external/techlog_docs\.json$|secrets/(?!README\.md$)"
    r"|reports/(?!README\.md$)|config\.toml$|CLAUDE\.md$|\.env)|\.duckdb$|\.jsonl$"
)
# Third-party lists that can't hold anything of yours, and files that aren't text.
NOT_SCANNED = re.compile(r"^data/external/|\.(woff2?|png|jpe?g|gif|ico|pdf)$")
# Artists in your history with at least this many plays must not be named in what you publish.
ARTIST_PLAYS = 20


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True).stdout


@pytest.fixture(scope="module")
def tracked() -> dict[str, str]:
    """Every file git would commit, with its text as staged."""
    try:
        paths = git("ls-files", "--cached").splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("not a git repository")
    return {path: (git("show", f":{path}") if not NOT_SCANNED.search(path) else "") for path in paths}


def mentions(tracked: dict[str, str], words: set[str]) -> list[str]:
    """'path: word' for every tracked text file naming one of words as a whole word."""
    found = []
    for path, text in tracked.items():
        found += [f"{path}: {word}" for word in sorted(words) if re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text)]
    return found


def test_no_file_of_yours_is_committed(tracked):
    assert [path for path in tracked if PRIVATE.search(path)] == []


def test_notebooks_are_committed_without_outputs(tracked):
    for path in (p for p in tracked if p.endswith(".ipynb")):
        cells = json.loads(tracked[path])["cells"]
        assert not any(cell.get("outputs") or cell.get("execution_count") for cell in cells), path


def test_no_key_or_login_is_committed(tracked):
    secrets = [line.strip() for file in (ROOT / "secrets").glob("*.txt") for line in file.read_text(encoding="utf-8").splitlines()]
    account = ROOT / "secrets" / "spotify_account.json"
    if account.exists():  # the tokens; its scope names are in recommender/spotify.py
        secrets += [str(value) for key, value in json.loads(account.read_text(encoding="utf-8")).items() if key.endswith("token")]
    secrets = [secret for secret in secrets if len(secret) >= 16]
    if not secrets:
        pytest.skip("no secrets/ files on this computer")
    assert [path for path, text in tracked.items() if any(secret in text for secret in secrets)] == []


def test_none_of_your_settings_is_committed(tracked):
    path = ROOT / "config.toml"
    if not path.exists():
        pytest.skip("no config.toml on this computer")
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    words = {config.get("home", {}).get("timezone", ""), config.get("not_music", {}).get("pattern", "")}
    words |= set(config.get("pool", {}).get("favourite_artists", []))
    words |= set(config.get("not_music", {}).get("genres", [])) | set(config.get("genres", {}).get("extra", []))
    words |= set(config.get("genres", {}).get("aliases", {}))
    assert mentions(tracked, {word for word in words if len(word) >= 4}) == []


def test_no_artist_of_your_history_is_named(tracked):
    db = ROOT / "data" / "processed" / "spotify.duckdb"
    if not db.exists():
        pytest.skip("no database on this computer")
    try:
        with duckdb.connect(str(db), read_only=True) as con:
            artists = {name for (name,) in con.sql(f"SELECT artist_name FROM gold.artists WHERE plays >= {ARTIST_PLAYS}").fetchall()}
    except duckdb.Error as error:
        pytest.skip(f"database not readable now ({error.__class__.__name__}); is a build running?")
    # Names that are ordinary words in the code too ([privacy] allow in config.toml).
    config = ROOT / "config.toml"
    allowed = set(tomllib.loads(config.read_text(encoding="utf-8")).get("privacy", {}).get("allow", [])) if config.exists() else set()
    assert mentions(tracked, {name for name in artists - allowed if len(name) >= 4}) == []
