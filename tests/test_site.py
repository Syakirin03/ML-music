"""The demo website (site/) keeps its promise: it loads only its own files, and sends and stores nothing.

The page tells its visitors that what they click stays in their browser tab. Its Content-Security-Policy has the
browser enforce that: no connections, no forms, nothing loaded from another site. These checks make sure the
policy stays as it is, that no script has code that sends or stores, and that the demo's data, which
scripts/build_demo_site.py makes up in UTC, holds no other time zone (yours, for one).
"""

import re
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent / "site"
POLICY = re.compile(r'<meta http-equiv="Content-Security-Policy" content="([^"]+)">')
# Browser features that send something somewhere or keep something after the tab is closed.
SENDING_OR_STORING = re.compile(
    r"\b(fetch|XMLHttpRequest|sendBeacon|WebSocket|EventSource|RTCPeerConnection|localStorage|sessionStorage|"
    r"indexedDB|caches|serviceWorker|cookie)\b"
)


def page() -> str:
    return (SITE / "index.html").read_text(encoding="utf-8")


def test_the_policy_lets_the_page_load_its_own_files_only_and_send_nothing():
    found = POLICY.search(page())
    assert found, "index.html has no Content-Security-Policy"
    rules = dict(rule.strip().split(" ", 1) for rule in found.group(1).split(";"))
    assert rules["default-src"] == "'none'"
    assert rules["connect-src"] == "'none'" and rules["form-action"] == "'none'"
    assert all(value in ("'self'", "'self' data:", "'none'") for value in rules.values()), rules
    # The policy comes before anything the page loads, or the browser would apply it too late.
    assert page().index("Content-Security-Policy") < page().index("<link")


def test_nothing_is_loaded_from_elsewhere():
    loaded = re.findall(r'<(?:script|link|img|iframe|source|video|audio)\b[^>]*\b(?:src|href)="([^"]+)"', page())
    assert loaded and [source for source in loaded if "//" in source or source.startswith("data:")] == []
    css = (SITE / "assets" / "style.css").read_text(encoding="utf-8")
    assert [url for url in re.findall(r"url\(([^)]+)\)", css) if "//" in url] == []


def test_no_script_sends_or_stores_anything():
    for path in sorted((SITE / "assets").glob("*.js")):
        assert SENDING_OR_STORING.findall(path.read_text(encoding="utf-8")) == [], path.name


def test_the_demo_data_holds_no_time_zone_but_utc():
    text = (SITE / "data" / "demo.js").read_text(encoding="utf-8")
    assert text.startswith("// Made by scripts/build_demo_site.py")
    assert re.findall(r"\d{2}:\d{2}:\d{2}(?:\.\d+)?[+-](?!00:00)\d{2}:\d{2}", text) == []
