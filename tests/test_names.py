"""The rules that decide when two titles are one song, and when two names are one artist or genre."""

import pandas as pd
import pytest

from app_pages.look import genre_label
from fetch_genres import plain_title
from recommender.data import artist_key, letters, plain_titles, song_key
from recommender.scoring import genre_pick
from recommender.spotify import same_song, track_uris

TITLES = [
    ("Song - Remastered 2011", "song"),
    ("Song (feat. Another Artist)", "song"),
    ('Song [From "A Film"]', "song"),
    ("Song - Japanese Ver.", "song"),
    ("曲（OP）", "曲"),
    ("(Intro)", "(intro)"),  # a title that is all brackets stays as it is
    ("Two  Spaces", "two spaces"),
]


@pytest.mark.parametrize(("title", "plain"), TITLES)
def test_plain_titles_drop_what_spotify_adds(title, plain):
    assert plain_titles(pd.Series([title])).iloc[0] == plain


@pytest.mark.parametrize(("title", "plain"), TITLES)
def test_the_genre_lookup_uses_the_same_rule(title, plain):
    # recommender/data.py copies plain_title() of scripts/fetch_genres.py, so the two must agree. An all-bracket
    # title gives "" there, and the lookup then keeps the full title, as plain_titles() does.
    assert (plain_title(title) or title).casefold() == plain


def test_artists_are_told_apart_by_their_exact_name():
    assert artist_key(pd.Series([" Band ", "BAND", None])).tolist() == ["Band", "BAND", ""]


def test_versions_of_a_song_share_a_key_and_namesakes_do_not():
    keys = song_key(pd.Series(["Band", "Band", "BAND"]), pd.Series(["Song", "Song - Live", "Song"]))
    assert keys[0] == keys[1] != keys[2]


def test_letters_keep_only_letters_and_digits():
    assert letters("(The) Band-Name 2") == "thebandname2"


@pytest.mark.parametrize(("genre", "label"), [
    ("j-pop", "J-pop"), ("r&b", "R&B"), ("uk garage", "UK garage"), ("edm", "EDM"), ("city pop", "City pop"),
])
def test_genre_labels_start_with_a_capital(genre, label):
    assert genre_label(genre) == label


def test_genre_pick_is_one_over_the_place_of_the_first_wanted_genre():
    lists = pd.Series([["k-pop", "pop"], ["pop", "k-pop"], ["rock", "pop", "k-pop"], ["rock"], None])
    assert genre_pick(lists, ["k-pop", "jazz"]) == pytest.approx([1, 1 / 2, 1 / 3, 0, 0])


def test_only_spotify_track_ids_become_uris():
    track = "0123456789abcdefghijkl"
    assert track_uris([track, "too-short", None, track + "x"]) == [f"spotify:track:{track}"]


def test_a_spotify_result_must_be_the_same_song_by_the_same_artist():
    found = {"name": "Song - Remastered", "artists": [{"name": "Band"}]}
    assert same_song(found, "Band", "Song")
    assert not same_song(found, "Someone Else", "Song")  # a cover
    assert not same_song(found, "Band", "Other Song")
    assert same_song({"name": "Song", "artists": [{"name": "Ca$h Band"}]}, "Cash Band", "Song")
