"""The Your data page: where the data comes from, how it is cleaned and stored (the medallion layers of
spotify.duckdb), and a first look at it (exploratory data analysis).

Every figure is read from the database when the page opens (data.py data_overview()), so it stays true after a
rebuild. The charts follow the project's data-visualisation palette, the one the exploration notebook uses:
categorical colours in a fixed order, magnitudes in one blue ramp, "Other" in gray; each has its numbers in a
table under "Show the numbers".
"""

import altair as alt
import pandas as pd
import streamlit as st

from app_pages import common, look
from recommender.data import OPINIONS, SPOTIFY_DB

# The palette's categorical slots, light and dark steps, "Other", the blue ramp for magnitude, and the surface
# whose colour separates touching marks.
SERIES = {"light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
          "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]}
OTHER = {"light": "#c3c2b7", "dark": "#898781"}
RAMP = {"light": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"],
        "dark": ["#0d366b", "#104281", "#1c5cab", "#2a78d6", "#3987e5", "#6da7ec", "#9ec5f4"]}
SURFACE = {"light": "#ffffff", "dark": "#16181c"}
MUTED = "#898781"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
# Genres drawn in their own colour in the genre chart; the rest are "Other".
TOP_GENRE_COUNT = 5
# A year is drawn only with at least this many hours of music (a history's first and last years can have few).
MIN_YEAR_HOURS = 20
# A genre is in the ratings chart with at least this many likes and dislikes of songs whose main genre it is.
MIN_GENRE_RATINGS = 8
LAYERS = {
    "bronze": "The exports exactly as downloaded: every field, every duplicate, as text",
    "silver": "Typed, renamed and deduplicated: plays, tracks, devices and log events, linked by keys",
    "techlog": "One typed table per kind of technical-log event (likes, playlist adds, what Spotify showed you)",
    "gold": "Tables made for machine learning: one row per play with features known at that moment, sessions, genres",
}


def draw(chart: alt.Chart) -> None:
    """A chart on its panel: transparent, so the panel and wallpaper show through, and as wide as the panel."""
    st.altair_chart(chart.configure(background="transparent").configure_view(strokeWidth=0), width="stretch")


def numbers(frame: pd.DataFrame) -> None:
    """The chart's numbers as a table: every chart has one, so no value is only in a tooltip."""
    with st.expander("Show the numbers"):
        st.dataframe(frame, hide_index=True, width="stretch")


def months_chart(months: pd.DataFrame, kind: str) -> alt.Chart:
    colour = SERIES[kind][0]
    base = alt.Chart(months).encode(
        x=alt.X("month:T", title=None, axis=alt.Axis(format="%Y", tickCount="year", grid=False)),
        y=alt.Y("hours:Q", title="Hours of music"),
    )
    hover = alt.selection_point(fields=["month"], nearest=True, on="pointerover", empty=False, clear="pointerout")
    tooltip = [alt.Tooltip("month:T", title="Month", format="%b %Y"), alt.Tooltip("hours:Q", title="Hours", format=".0f"),
               alt.Tooltip("plays:Q", title="Plays", format=",")]
    return alt.layer(
        base.mark_area(color=colour, opacity=0.12),
        base.mark_line(color=colour, strokeWidth=2, strokeCap="round", strokeJoin="round"),
        base.mark_rule(color=MUTED, strokeWidth=1).encode(opacity=alt.condition(hover, alt.value(1), alt.value(0))),
        base.mark_point(size=80, filled=True, color=colour, stroke=SURFACE[kind], strokeWidth=2)
        .encode(opacity=alt.condition(hover, alt.value(1), alt.value(0)), tooltip=tooltip).add_params(hover),
    ).properties(height=280)


def hours_chart(hours: pd.DataFrame, kind: str) -> alt.Chart:
    return alt.Chart(hours).mark_rect(cornerRadius=3, stroke=SURFACE[kind], strokeWidth=2).encode(
        x=alt.X("hour:O", title="Hour of the day (your time zone)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("day:N", sort=DAYS, title=None),
        color=alt.Color("hours:Q", title="Hours", scale=alt.Scale(range=RAMP[kind])),
        tooltip=[alt.Tooltip("day:N", title="Day"), alt.Tooltip("hour:O", title="Hour"),
                 alt.Tooltip("hours:Q", title="Hours", format=".0f"), alt.Tooltip("plays:Q", title="Plays", format=",")],
    ).properties(height=250)


def share_chart(frame: pd.DataFrame, labels: list[str], colours: list[str], kind: str, legend: str) -> alt.Chart:
    """Bars of 100% per year, split by label in the fixed order of labels."""
    return alt.Chart(frame).mark_bar(size=26, stroke=SURFACE[kind], strokeWidth=2).encode(
        x=alt.X("year:O", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("hours:Q", stack="normalize", title="Share of the year", axis=alt.Axis(format="%")),
        color=alt.Color("label:N", title=legend, scale=alt.Scale(domain=labels, range=colours), sort=labels,
                        legend=alt.Legend(orient="top")),
        order=alt.Order("rank:Q"),
        tooltip=[alt.Tooltip("year:O", title="Year"), alt.Tooltip("label:N", title=legend),
                 alt.Tooltip("share:Q", title="Share", format=".0%"), alt.Tooltip("hours:Q", title="Hours", format=".0f")],
    ).properties(height=300)


def ratings_chart(table: pd.DataFrame, overall: float, kind: str) -> alt.Chart:
    bars = alt.Chart(table).mark_bar(size=18, cornerRadiusEnd=4, color=SERIES[kind][0]).encode(
        x=alt.X("liked:Q", title="Share liked", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("Genre:N", sort="-x", title=None),
        tooltip=[alt.Tooltip("Genre:N"), alt.Tooltip("liked:Q", title="Liked", format=".0%"),
                 alt.Tooltip("Likes:Q"), alt.Tooltip("Dislikes:Q")],
    )
    rule = alt.Chart(pd.DataFrame({"liked": [overall]})).mark_rule(color=MUTED, strokeWidth=1).encode(x="liked:Q")
    return (bars + rule).properties(height=max(160, 26 * len(table)))


overview = common.database_overview()
library = common.history()
steps = overview.steps
kind = look.theme_kind() or "light"
years = (steps.last_play - steps.first_play).days / 365.25
rated = common.current()

look.hero(
    "Your data", "What the recommender learns from",
    f"Your Spotify listening since {steps.first_play:%B %Y}, cleaned up and stored in one database file, then "
    "explored. Every number on this page is read from that database now.",
    [(f"{steps.plays:,}", "plays"), (f"{steps.hours:,.0f}", "hours"), (f"{len(library.songs):,}", "songs"),
     (f"{steps.artists:,}", "artists"), (f"{years:.1f}", "years")],
)

st.header("Where the data comes from")
sources = st.columns(3, gap="medium")
for number, (column, (name, text)) in enumerate(zip(sources, [
    ("Spotify streaming history", f"Every play since {steps.first_play:%B %Y}: when it stopped, what it was, how long you "
                                  "listened and how it ended. Spotify sends it as JSON files when you ask for your data."),
    ("Spotify technical log", f"{steps.log_records:,} records of {steps.log_event_types:,} kinds of app events, from "
                              f"{steps.log_first:%B %Y} to {steps.log_last:%B %Y} only: your likes, playlist adds and "
                              "what Spotify showed you." if steps.log_records else
                              "Not in your export. Spotify's technical log, a separate request, adds your likes, "
                              "playlist adds and what Spotify showed you."),
    ("Last.fm", "Genres for each song, from the tags Last.fm listeners give, and the pool of songs new to you: "
                "charts of your top artists, their similar artists, and your genres' top artists."),
])):
    with column.container(key=f"panel-source-{number}"):
        st.markdown(f"**{name}**")
        st.caption(text)
st.caption("The export also holds your IP addresses, email and device ids. They stay in the database and are never shown here.")

st.header("How it is cleaned")
look.steps([
    ("Loaded as exported.", f"{steps.raw_streams:,} streaming rows and {steps.log_records:,} technical-log records go "
                            "into the bronze layer unchanged, so nothing is ever lost."),
    ("Exact duplicates removed.", f"{steps.raw_streams - steps.streams:,} rows appear twice in the export, leaving "
                                  f"{steps.streams:,}."),
    ("Songs and podcasts apart.", f"{steps.song_streams:,} song plays and {steps.podcast_streams:,} podcast episodes; "
                                  "the recommender uses the songs."),
    ("One play counted once.", f"A song that stops at the same moment twice is one play: {steps.song_streams - steps.plays:,} "
                               f"repeats removed, {steps.plays:,} plays left."),
    ("Start times worked out.", "Spotify records when a song stopped; it started that long before. Times are in "
                                "your time zone (config.toml), because the export's country comes from IP addresses."),
    ("Sessions found.", f"A pause of 30 minutes or more starts a new session: {steps.sessions:,} sessions."),
    ("Each play labelled.", "Finished, skipped in its first 30 seconds, or partly heard. That gives each play a "
                            "listen score: 1, 0, or the share heard."),
    ("What isn't music set aside.", f"Spotify publishes some audio as music, such as white noise or audiobooks: "
                                    f"{steps.not_music_plays:,} plays ({steps.not_music_plays / steps.plays:.0%}) match "
                                    "the not-music rules of config.toml, and are kept apart from your music taste."),
    ("Genres added.", f"From Last.fm tags: {steps.plays_with_genre / steps.plays:.0%} of plays have a genre."),
    ("Versions merged.", f"Remixes, language versions and re-releases of one song are one song: {steps.track_ids:,} "
                         f"Spotify track ids make {len(library.songs):,} songs."),
])

st.header("How it is stored: the medallion layers")
st.markdown(
    "Data flows through layers that each add quality, a common design called the **medallion architecture**: "
    "**bronze** keeps the raw exports, **silver** cleans and links them, and **gold** holds tables ready for "
    "machine learning. All of it is one [DuckDB](https://duckdb.org) file: a database that runs inside Python, "
    "with no server, queried with SQL."
)
st.mermaid_chart(
    """flowchart LR
    E[Spotify export<br/>JSON] --> B[(Bronze<br/>as exported)]
    B --> S[(Silver<br/>clean, linked)]
    B --> L[(Techlog<br/>by event)]
    S --> G[(Gold<br/>ready for ML)]
    L --> G
    F[Last.fm] --> G
    F --> R
    G --> R[Recommender]
    R <-->|lists, ratings| W[This website]""",
    alt="Data flow from the Spotify exports through bronze, silver, techlog and gold layers to the recommender and website",
)
layers = overview.layers.assign(
    holds=lambda frame: frame.layer.map(LAYERS), row_count=lambda frame: frame.row_count.astype(int),
)[["layer", "holds", "tables", "row_count"]].rename(columns={"layer": "Layer", "holds": "What it holds", "tables": "Tables", "row_count": "Rows"})
st.table(layers.assign(Rows=layers.Rows.map("{:,}".format)).set_index("Layer"))
st.caption(
    f"{SPOTIFY_DB.name}: {overview.file_bytes / 1e6:,.0f} MB. Your ratings, taste profile and models are kept apart, "
    "in data/user/recommender.duckdb, because nothing can rebuild your ratings."
)

st.header("What you listen to")
st.caption("Music only: tracks that config.toml marks as not music are left out of these charts.")

months = overview.months.assign(month=lambda frame: pd.to_datetime(frame.month)).set_index("month")
every_month = pd.date_range(months.index.min(), months.index.max(), freq="MS")
months = months.reindex(every_month).fillna({"plays": 0, "hours": 0.0}).rename_axis("month").reset_index()
months["plays"] = months.plays.astype(int)
peak = months.loc[months.hours.idxmax()]
last_year = months[months.month > months.month.max() - pd.DateOffset(months=12)]
with st.container(key="panel-months"):
    st.subheader("Hours of music per month")
    st.caption(f"Your busiest month was {peak.month:%B %Y}, with {peak.hours:,.0f} hours. Over the last 12 months "
               f"you listened {last_year.hours.mean():,.0f} hours a month on average.")
    draw(months_chart(months, kind))
    numbers(months.assign(month=months.month.dt.strftime("%Y-%m"), hours=months.hours.round(1))[["month", "hours", "plays"]])

hours = overview.hours.assign(day=lambda frame: frame.weekday.map(lambda number: DAYS[number - 1]))
busiest = hours.loc[hours.hours.idxmax()]
by_hour = hours.groupby("hour").hours.sum()
with st.container(key="panel-hours"):
    st.subheader("When you listen")
    st.caption(f"Most of all on {busiest.day} at {busiest.hour}:00. Across the week, {by_hour.idxmax()}:00 is your "
               f"busiest hour and {by_hour.idxmin()}:00 your quietest.")
    draw(hours_chart(hours, kind))
    grid = hours.pivot(index="day", columns="hour", values="hours").reindex(DAYS).round(1)
    numbers(grid.rename(columns=lambda hour: f"{hour}:00").reset_index().rename(columns={"day": "Day"}))

genre_years = overview.genre_years
top = genre_years.dropna(subset=["genre"]).groupby("genre").hours.sum().nlargest(TOP_GENRE_COUNT).index.tolist()
labels = [look.genre_label(genre) for genre in top] + ["Other"]
genres = genre_years.assign(label=genre_years.genre.map(lambda genre: look.genre_label(genre) if genre in top else "Other"))
genres = genres.groupby(["year", "label"], as_index=False).hours.sum()
totals = genres.groupby("year").hours.transform("sum")
genres = genres[totals >= MIN_YEAR_HOURS].assign(share=lambda frame: frame.hours / frame.groupby("year").hours.transform("sum"),
                                                rank=lambda frame: frame.label.map(labels.index))
with st.container(key="panel-genres"):
    st.subheader("Your genres over the years")
    named = genres[(genres.year == genres.year.max()) & (genres.label != "Other")].sort_values("share", ascending=False).iloc[0]
    st.caption(f"Each bar is one year of listening (years with under {MIN_YEAR_HOURS} hours are left out). Your "
               f"{TOP_GENRE_COUNT} most played main genres overall have their own colour, the rest are Other. In "
               f"{named.year}, {named.label} took {named.share:.0%} of your music.")
    draw(share_chart(genres, labels, SERIES[kind][:TOP_GENRE_COUNT] + [OTHER[kind]], kind, "Main genre"))
    numbers(genres.pivot(index="year", columns="label", values="share").reindex(columns=labels).round(3).reset_index())

artists = overview.artist_years.assign(place=lambda frame: frame.groupby("year").cumcount() + 1)
artists = artists[artists.year.isin(genres.year.unique())]
table = artists.assign(text=lambda frame: frame.artist + " (" + (frame.share * 100).round().astype(int).astype(str) + "%)")
table = table.pivot(index="year", columns="place", values="text").rename(columns={1: "Most played", 2: "Second", 3: "Third"})
with st.container(key="panel-artists"):
    st.subheader("Your taste moves in phases")
    leader = artists[artists.place == 1].sort_values("share").iloc[-1]
    st.caption(f"Your three most played artists each year, with their share of the year's music. The largest share any "
               f"artist ever took was {leader.artist} in {leader.year}: {leader.share:.0%}.")
    st.dataframe(table.reset_index().rename(columns={"year": "Year"}), hide_index=True, width="stretch")

ends = overview.end_years.rename(columns={"plays": "hours"})
end_labels = {"finished": "Finished", "skipped": "Skipped", "interrupted": "Stopped another way"}
ends = ends.assign(label=ends.end_type.map(end_labels))
ends = ends[ends.year.isin(genres.year.unique())]
ends = ends.assign(share=lambda frame: frame.hours / frame.groupby("year").hours.transform("sum"),
                   rank=lambda frame: frame.label.map(list(end_labels.values()).index))
with st.container(key="panel-ends"):
    st.subheader("How your plays end")
    now = ends[ends.year == ends.year.max()].set_index("end_type").share
    st.caption(f"Share of each year's plays you finished, skipped, or that stopped another way (another app, a "
               f"closed laptop). In {ends.year.max()} you finished {now.get('finished', 0):.0%} and skipped "
               f"{now.get('skipped', 0):.0%}. Skips matter: a skip in the first 30 seconds counts against a song.")
    chart = share_chart(ends, list(end_labels.values()), [SERIES[kind][0], SERIES[kind][1], OTHER[kind]], kind, "How it ended")
    draw(chart.encode(tooltip=[alt.Tooltip("year:O", title="Year"), alt.Tooltip("label:N", title="How it ended"),
                               alt.Tooltip("share:Q", title="Share", format=".0%"), alt.Tooltip("hours:Q", title="Plays", format=",")]))
    numbers(ends.pivot(index="year", columns="label", values="share").round(3).reset_index())

opinions = rated[rated.rating.isin(OPINIONS)].assign(genre=lambda frame: frame.genres.map(lambda g: g[0] if len(g) else None))
by_genre = opinions.dropna(subset=["genre"]).groupby("genre").rating.agg(
    Likes=lambda r: int((r == 1).sum()), Dislikes=lambda r: int((r == -1).sum())
)
by_genre = by_genre[by_genre.Likes + by_genre.Dislikes >= MIN_GENRE_RATINGS]
by_genre = by_genre.assign(liked=by_genre.Likes / (by_genre.Likes + by_genre.Dislikes)).reset_index()
by_genre["Genre"] = by_genre.genre.map(look.genre_label)
with st.container(key="panel-rated"):
    st.subheader("What your ratings say so far")
    if len(by_genre) >= 2:
        overall = (opinions.rating == 1).mean()
        best, worst = by_genre.loc[by_genre.liked.idxmax()], by_genre.loc[by_genre.liked.idxmin()]
        st.caption(
            f"Share of your likes among your likes and dislikes, by the song's main genre, for genres with at least "
            f"{MIN_GENRE_RATINGS} ratings. The gray line is your overall share, {overall:.0%}. You liked {best.liked:.0%} of "
            f"{best.Genre} songs and {worst.liked:.0%} of {worst.Genre} ones: this is what the models learned from "
            "your ratings pick up (How it works)."
        )
        draw(ratings_chart(by_genre, overall, kind))
        numbers(by_genre[["Genre", "Likes", "Dislikes", "liked"]].rename(columns={"liked": "Share liked"}).round(3))
    else:
        st.caption("Rate a few more songs: this chart needs two genres with enough likes and dislikes.")
look.footer()
