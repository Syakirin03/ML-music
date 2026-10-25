"""The website's look: the wallpaper, see-through panels over it, genre chips, the like-chance meter, the page
header and footer, and the scroll back to the top after a rating.

Streamlit draws the page; this adds CSS for what Streamlit's theme (.streamlit/config.toml) can't set. Song cards
and panels are containers with keys starting card- and panel-, which Streamlit gives the CSS class st-key-<key>.
Colours follow the light or dark mode Streamlit reports (st.context.theme), or your system's when it reports none.
"""

import base64
import html

import streamlit as st

from app_pages.common import WALLPAPER_FILE, state

# The wallpapers: soft blobs of colour over a base, each with a light and a dark version. A blob is
# (colour, x %, y %), drawn as a radial gradient.
WALLPAPERS = {
    "aurora": ("Aurora", [("29,155,240", 12, 8), ("120,86,255", 88, 18), ("0,186,124", 50, 100)]),
    "sunset": ("Sunset", [("255,122,89", 15, 85), ("249,24,128", 85, 15), ("120,86,255", 60, 60)]),
    "ocean": ("Ocean", [("0,194,255", 20, 20), ("29,155,240", 80, 40), ("0,186,124", 40, 100)]),
    "sakura": ("Sakura", [("255,143,177", 15, 15), ("255,179,199", 85, 70), ("195,166,255", 60, 10)]),
    "neon": ("Tokyo neon", [("249,24,128", 10, 90), ("0,229,255", 90, 10), ("120,86,255", 50, 50)]),
    "forest": ("Forest", [("0,186,124", 15, 20), ("120,200,80", 85, 80), ("0,120,90", 60, 30)]),
    "plain": ("Plain", []),
    "picture": ("Your own picture", []),
}
BASES = {"dark": "linear-gradient(160deg, #05070d 0%, #0b0f1a 55%, #070a12 100%)",
         "light": "linear-gradient(160deg, #f7f9fc 0%, #eef2f8 55%, #f5f3fb 100%)"}
BLOB_ALPHA = {"dark": 0.50, "light": 0.34}
# The colours of each mode: panels, text, the accent (X's blue) and the chips.
PALETTES = {
    "light": {"surface": "rgba(255,255,255,0.80)", "solid": "#ffffff", "text": "#0f1419", "muted": "#536471",
              "border": "rgba(15,20,25,0.10)", "accent": "#1d9bf0", "accent_soft": "rgba(29,155,240,0.14)",
              "chip": "rgba(15,20,25,0.06)", "heading_shadow": "0 1px 14px rgba(255,255,255,0.55)", "dim": "255,255,255"},
    "dark": {"surface": "rgba(22,24,28,0.74)", "solid": "#000000", "text": "#e7e9ea", "muted": "#8b98a5",
             "border": "rgba(255,255,255,0.10)", "accent": "#1d9bf0", "accent_soft": "rgba(29,155,240,0.22)",
             "chip": "rgba(255,255,255,0.08)", "heading_shadow": "0 1px 14px rgba(0,0,0,0.55)", "dim": "0,0,0"},
}
# Genre names written in capitals: Last.fm's tags are all lower case.
UPPER = {"edm", "r&b", "opm", "uk", "us", "idm", "ebm", "dnb", "ost", "dj"}


def genre_label(genre: str) -> str:
    """A genre as the website writes it: capital first letter, and acronyms in capitals (J-pop, R&B, UK garage)."""
    text = " ".join(word.upper() if word in UPPER else word for word in str(genre).split(" "))
    return text[:1].upper() + text[1:]


def theme_kind() -> str | None:
    """"light" or "dark" as Streamlit reports it, or None when it hasn't said yet (then CSS asks the system)."""
    kind = st.context.theme.type
    return kind if kind in PALETTES else None


@st.cache_data(show_spinner=False, max_entries=2)
def picture_url(modified: float) -> str | None:
    """Your wallpaper picture as a data: URL (read again when the file changes)."""
    if not WALLPAPER_FILE.exists():
        return None
    return "data:image/jpeg;base64," + base64.b64encode(WALLPAPER_FILE.read_bytes()).decode("ascii")


def wallpaper_css(name: str, kind: str) -> str | None:
    """The CSS background of a wallpaper in one mode, or None for the theme's own background."""
    if name == "picture":
        url = picture_url(WALLPAPER_FILE.stat().st_mtime) if WALLPAPER_FILE.exists() else None
        return f'url("{url}") center / cover no-repeat fixed' if url else None
    blobs = WALLPAPERS.get(name, WALLPAPERS["plain"])[1]
    if not blobs:
        return None
    alpha = BLOB_ALPHA[kind]
    layers = [f"radial-gradient(900px 620px at {x}% {y}%, rgba({rgb},{alpha}) 0%, rgba({rgb},0) 70%)" for rgb, x, y in blobs]
    return ", ".join(layers + [BASES[kind]]) + " fixed"


def variables(kind: str, wallpaper: str, dim: float) -> str:
    colours = PALETTES[kind]
    lines = [f"--mp-{name.replace('_', '-')}: {value};" for name, value in colours.items() if name != "dim"]
    background = wallpaper_css(wallpaper, kind)
    if background:
        lines.append(f"--mp-wallpaper: linear-gradient(rgba({colours['dim']},{dim}), rgba({colours['dim']},{dim})) fixed, {background};")
    else:
        lines.append(f"--mp-wallpaper: {colours['solid']};")
    return " ".join(lines)


def apply_look(settings: dict) -> None:
    """The site's CSS, with the wallpaper and dimming of your settings."""
    wallpaper, dim = settings["wallpaper"], float(settings["dim"])
    kind = theme_kind()
    if kind:
        roots = f":root {{ {variables(kind, wallpaper, dim)} }}"
    else:
        roots = (f":root {{ {variables('light', wallpaper, dim)} }}\n"
                 f"@media (prefers-color-scheme: dark) {{ :root {{ {variables('dark', wallpaper, dim)} }} }}")
    st.html(f"<style>{roots}\n{SITE_CSS}</style>")


SITE_CSS = """
[data-testid="stApp"] { background: var(--mp-wallpaper); }
[data-testid="stHeader"] {
  background: var(--mp-surface); backdrop-filter: blur(14px) saturate(140%);
  -webkit-backdrop-filter: blur(14px) saturate(140%); border-bottom: 1px solid var(--mp-border);
}
[data-testid="stSidebar"] {
  background: var(--mp-surface); backdrop-filter: blur(18px) saturate(140%);
  -webkit-backdrop-filter: blur(18px) saturate(140%);
}
[data-testid="stMainBlockContainer"] { max-width: 1180px; padding-top: 4.6rem; }
h1, h2, h3 { letter-spacing: -0.02em; }
h2 { font-weight: 800; }
[class*="st-key-card-"], [class*="st-key-panel-"] {
  background: var(--mp-surface); border: 1px solid var(--mp-border); border-radius: 20px;
  padding: 16px 18px; backdrop-filter: blur(14px) saturate(140%); -webkit-backdrop-filter: blur(14px) saturate(140%);
}
[class*="st-key-card-"]:hover { border-color: var(--mp-accent-soft); }
.st-key-panel-progress { position: sticky; top: 3.9rem; z-index: 100; padding: 10px 18px; }
.mp-hero {
  background: var(--mp-surface); border: 1px solid var(--mp-border); border-radius: 22px; padding: 20px 24px;
  margin-bottom: 6px; backdrop-filter: blur(14px) saturate(140%); -webkit-backdrop-filter: blur(14px) saturate(140%);
}
[data-testid="stAlert"] { background: var(--mp-surface); border-radius: 16px; backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px); }
[data-testid="stMainBlockContainer"] h1, [data-testid="stMainBlockContainer"] h2 { text-shadow: var(--mp-heading-shadow); }
.mp-kicker { color: var(--mp-accent); font-weight: 700; font-size: 0.85rem; letter-spacing: 0.01em; }
.mp-title { font-size: 2.3rem; font-weight: 800; letter-spacing: -0.025em; line-height: 1.12; margin: 2px 0 6px 0; color: var(--mp-text); }
.mp-sub { color: var(--mp-muted); font-size: 1.04rem; line-height: 1.5; max-width: 780px; }
.mp-stats { margin-top: 10px; }
.mp-stat { display: inline-block; padding: 5px 12px; border-radius: 999px; background: var(--mp-chip);
  margin: 6px 8px 0 0; font-size: 0.86rem; color: var(--mp-text); }
.mp-stat b { font-weight: 700; }
.mp-song-head { display: flex; gap: 12px; align-items: baseline; }
.mp-pos { color: var(--mp-muted); font-weight: 700; font-size: 0.95rem; min-width: 1.4em; }
.mp-song-title { font-size: 1.12rem; font-weight: 750; line-height: 1.25; color: var(--mp-text); }
.mp-song-artist { color: var(--mp-muted); font-size: 0.95rem; margin-top: 1px; }
.mp-chips { margin: 9px 0 2px 0; }
.mp-chip { display: inline-block; padding: 3px 10px; border-radius: 999px; background: var(--mp-chip);
  color: var(--mp-text); font-size: 0.8rem; font-weight: 500; margin: 0 6px 6px 0; }
.mp-chip.main { background: var(--mp-accent-soft); color: var(--mp-accent); font-weight: 700; }
.mp-chip.picked { box-shadow: inset 0 0 0 1.5px var(--mp-accent); }
.mp-match { margin: 8px 0 6px 0; max-width: 360px; }
.mp-match-top { display: flex; align-items: baseline; gap: 8px; }
.mp-match-value { font-size: 1.45rem; font-weight: 800; color: var(--mp-text); letter-spacing: -0.02em; }
.mp-match-label { color: var(--mp-muted); font-size: 0.9rem; }
.mp-meter { height: 7px; border-radius: 999px; background: var(--mp-accent-soft); overflow: hidden; margin-top: 4px; }
.mp-meter > div { height: 100%; border-radius: 999px; background: var(--mp-accent); }
.mp-second { color: var(--mp-muted); font-size: 0.86rem; margin-top: 5px; }
.mp-why { color: var(--mp-muted); font-size: 0.88rem; line-height: 1.45; margin-top: 6px; }
.mp-last { display: flex; gap: 10px; align-items: center; }
.mp-footer { color: var(--mp-muted); font-size: 0.84rem; text-align: center; padding: 26px 0 10px 0; }
.mp-footer b { color: var(--mp-text); }
.mp-step { display: flex; gap: 12px; margin: 10px 0; }
.mp-step-n { flex: 0 0 28px; height: 28px; border-radius: 999px; background: var(--mp-accent-soft); color: var(--mp-accent);
  font-weight: 800; display: flex; align-items: center; justify-content: center; font-size: 0.9rem; }
.mp-step-text { color: var(--mp-text); line-height: 1.5; }
.mp-step-text span { color: var(--mp-muted); }
.mp-swatch { height: 64px; border-radius: 14px; border: 1px solid var(--mp-border); }
"""


def hero(kicker: str, title: str, subtitle: str = "", stats: list[tuple[str, str]] | None = None) -> None:
    """The header of a page: a small line above, the title, a sentence, and figures as (value, label) pills.

    subtitle is HTML: escape whatever in it comes from data (html.escape), as song and artist names do.
    """
    pills = "".join(f'<span class="mp-stat"><b>{html.escape(value)}</b> {html.escape(label)}</span>' for value, label in stats or [])
    st.html(
        f'<div class="mp-hero"><div class="mp-kicker">{html.escape(kicker)}</div>'
        f'<div class="mp-title">{html.escape(title)}</div>'
        + (f'<div class="mp-sub">{subtitle}</div>' if subtitle else "")
        + (f'<div class="mp-stats">{pills}</div>' if pills else "")
        + "</div>"
    )


def chips(genres: list[str], picked=()) -> str:
    """Genre chips: the main genre highlighted, the genres you asked for more of outlined."""
    return "".join(
        f'<span class="mp-chip{" main" if number == 0 else ""}{" picked" if genre in picked else ""}">'
        f"{html.escape(genre_label(genre))}</span>"
        for number, genre in enumerate(genres)
    )


def meter(value: float, label: str, shown: str | None = None) -> str:
    """A big figure with a bar under it: value from 0 to 1, shown as a percentage unless shown says otherwise."""
    width = max(0.0, min(1.0, float(value))) * 100
    return (
        f'<div class="mp-match"><div class="mp-match-top"><span class="mp-match-value">{html.escape(shown or f"{value:.0%}")}</span>'
        f'<span class="mp-match-label">{html.escape(label)}</span></div>'
        f'<div class="mp-meter"><div style="width:{width:.1f}%"></div></div></div>'
    )


def steps(items: list[tuple[str, str]]) -> None:
    """A numbered list of (bold part, rest) lines, as on the How it works and Your data pages."""
    st.html("".join(
        f'<div class="mp-step"><div class="mp-step-n">{number}</div>'
        f'<div class="mp-step-text"><b>{html.escape(head)}</b> <span>{html.escape(rest)}</span></div></div>'
        for number, (head, rest) in enumerate(items, 1)
    ))


def footer() -> None:
    st.html(
        '<div class="mp-footer"><b>Music Preference</b> · a recommender that learns from your own Spotify history · '
        "runs only on this computer · data from your Spotify export and Last.fm · built with DuckDB, scikit-learn "
        "and Streamlit</div>"
    )


def scroll_to_top() -> None:
    """Scroll the page back to its top, once: after a rating the list is new, and its best song is up there."""
    state["scroll_count"] = state.get("scroll_count", 0) + 1
    st.html(
        f"<script>/* {state['scroll_count']} */"
        "for (const selector of ['[data-testid=\"stMain\"]', '[data-testid=\"stAppViewContainer\"]', '[data-testid=\"stApp\"]']) {"
        "  document.querySelectorAll(selector).forEach((element) => element.scrollTo({top: 0, behavior: 'smooth'}));"
        "}"
        "window.scrollTo({top: 0, behavior: 'smooth'});"
        "</script>",
        unsafe_allow_javascript=True,
    )
