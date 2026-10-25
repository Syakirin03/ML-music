/* The recommender's ranking, run in the browser on the demo's data (data/demo.js).
 *
 * scripts/build_demo_site.py runs the real recommender on a made-up listener and keeps, for every song, the
 * parts of its score that the dials don't change. This file does the rest exactly as recommender/scoring.py
 * and recommend.py do: the score with the dials, the model learned from ratings reordering the best songs,
 * at most MAX_PER_ARTIST songs per artist, and variety (maximal marginal relevance). The build checks that it
 * gives the recommender's lists (scripts/check_demo_engine.js). It only computes: it sends and stores nothing.
 */
(function (root) {
  "use strict";

  // A song's genre row (recommender/features.py): 1 for its main genre, 1/2 for its second, and so on,
  // scaled to length 1. Only the genres in vocabulary count, when one is given.
  function genreRow(genres, vocabulary) {
    const row = new Map();
    genres.forEach((genre, place) => {
      if (!vocabulary || vocabulary.has(genre)) row.set(genre, 1 / (place + 1));
    });
    let squares = 0;
    row.forEach((value) => { squares += value * value; });
    const scale = 1 / (Math.sqrt(squares) || 1);
    row.forEach((value, genre) => row.set(genre, value * scale));
    return row;
  }

  function alike(a, b) {
    let sum = 0;
    a.forEach((value, genre) => {
      if (b.has(genre)) sum += value * b.get(genre);
    });
    return sum;
  }

  // pandas' rank(pct=True): ties share their average rank, divided by the count.
  function percentRanks(values) {
    const order = values.map((value, index) => index).sort((a, b) => values[a] - values[b]);
    const ranks = new Array(values.length);
    for (let start = 0; start < order.length;) {
      let end = start;
      while (end + 1 < order.length && values[order[end + 1]] === values[order[start]]) end += 1;
      const rank = (start + end) / 2 + 1;
      for (let i = start; i <= end; i += 1) ranks[order[i]] = rank / values.length;
      start = end + 1;
    }
    return ranks;
  }

  // scoring.py genre_pick: 1 / the place of the first wanted genre among the song's genres, 0 for none.
  function genrePick(genres, wanted) {
    for (let place = 0; place < genres.length; place += 1) {
      if (wanted.has(genres[place])) return 1 / (place + 1);
    }
    return 0;
  }

  // The model learned from ratings (feedback.py like_chances): its numbers standardised, then its genre row,
  // everything scaled to at most 1 (MaxAbsScaler), into a logistic regression.
  function likedChance(model, inputs, genres) {
    const values = model.numbers.map((name, i) => (inputs[name] - model.mean[i]) / model.scale[i]);
    const vocabulary = model.vocabulary || (model.vocabulary = new Map(model.genres.map((genre, i) => [genre, i])));
    const row = genreRow(genres, vocabulary);
    const columns = values.map((value, i) => [i, value]);
    row.forEach((value, genre) => columns.push([values.length + vocabulary.get(genre), value]));
    columns.sort((a, b) => a[0] - b[0]);
    let sum = 0;
    columns.forEach(([column, value]) => {
      if (value !== 0) sum += value * (1 / model.maxabs[column]) * model.coef[column];
    });
    return 1 / (1 + Math.exp(-(sum + model.intercept)));
  }

  // recommend.py varied_top: positions to list, from songs ranked best first.
  function variedTop(order, rows, artists, variety, top, constants) {
    const cap = constants.MAX_PER_ARTIST;
    if (variety === 0) {
      const seen = new Map();
      const listed = [];
      for (let i = 0; i < order.length && listed.length < top; i += 1) {
        const count = seen.get(artists[i]) || 0;
        seen.set(artists[i], count + 1);
        if (count < cap) listed.push(i);
      }
      return listed;
    }
    const shortlist = Math.min(order.length, Math.max(constants.SHORTLIST, 3 * top));
    let score = order.slice(0, shortlist);
    const low = Math.min(...score);
    const span = shortlist ? Math.max(...score) - low : 0;
    score = span > 0 ? score.map((value) => (value - low) / span) : score.map(() => 0);
    const alikeness = [];
    for (let i = 0; i < shortlist; i += 1) {
      alikeness.push(new Float64Array(shortlist));
      for (let j = 0; j < shortlist; j += 1) alikeness[i][j] = alike(rows[i], rows[j]);
    }
    const hasGenres = rows.slice(0, shortlist).map((row) => row.size > 0);
    const withGenres = [];
    for (let i = 0; i < shortlist; i += 1) {
      for (let j = 0; j < shortlist; j += 1) if (hasGenres[i] && hasGenres[j]) withGenres.push(alikeness[i][j]);
    }
    const neutral = hasGenres.filter(Boolean).length > 1 ? median(withGenres) : 0;
    for (let i = 0; i < shortlist; i += 1) {
      for (let j = 0; j < shortlist; j += 1) if (!hasGenres[i] || !hasGenres[j]) alikeness[i][j] = neutral;
    }
    const listed = [];
    const perArtist = new Map();
    const likeness = new Float64Array(shortlist);
    const available = new Array(shortlist).fill(true);
    while (listed.length < top && available.some(Boolean)) {
      let best = -1;
      let bestValue = -Infinity;
      for (let i = 0; i < shortlist; i += 1) {
        if (!available[i]) continue;
        const value = (1 - variety) * score[i] - variety * likeness[i];
        if (value > bestValue) { best = i; bestValue = value; }
      }
      available[best] = false;
      if ((perArtist.get(artists[best]) || 0) >= cap) continue;
      perArtist.set(artists[best], (perArtist.get(artists[best]) || 0) + 1);
      listed.push(best);
      for (let j = 0; j < shortlist; j += 1) likeness[j] = Math.max(likeness[j], alikeness[best][j]);
    }
    return listed;
  }

  function median(values) {
    const sorted = Float64Array.from(values).sort();
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  }

  // recommend.py list_positions: the list, and the positions on it only variety put there.
  function listPositions(order, rows, artists, variety, top, constants) {
    const listed = variedTop(order, rows, artists, variety, top, constants);
    if (variety === 0) return { listed, forVariety: new Set() };
    const plain = new Set(variedTop(order, rows, artists, 0, top, constants));
    return { listed, forVariety: new Set(listed.filter((position) => !plain.has(position))) };
  }

  // The pool's songs new to the listener, in the order the recommender takes them. The data keeps what all the
  // songs of an artist share once, with the artist; each song has its title, key, listeners and chart place.
  function poolSongs(demo) {
    if (!demo.poolSongs) {
      const artists = demo.discover.artists;
      demo.poolSongs = demo.discover.songs.map(([title, number, key, listeners, rank, shown]) => {
        const artist = artists[number];
        return {
          title, song_key: key, listeners, artist_rank: rank, artist: artist.name, artist_key: artist.key,
          genres: artist.genres, genre_match: artist.genre_match, artist_match: artist.artist_match,
          similar_sum: artist.similar_sum, asian: artist.asian, played_artist: artist.played, why_head: artist.why_head,
          why_tail: [artist.genre_phrase, `#${rank} of their songs on Last.fm, ${shown} listeners`],
        };
      });
    }
    return demo.poolSongs;
  }

  // scoring.py taste_parts: genre_match is the cosine between a song's genre row and the profile's genre weights,
  // over the genres of the history and of the songs being ranked. The data has it for the whole pool; with fewer
  // songs, a genre only left-out songs had drops out of the profile, whose length changes, so every song's
  // genre_match is rescaled by it. A song without genres gets the median of the others.
  function genreMatches(demo, songs) {
    const known = demo.genresOfHistory || (demo.genresOfHistory = new Set(demo.history.flatMap((song) => song.genres)));
    const length = (genres) => {
      let squares = 0;
      demo.discover.profile_genres.forEach(([genre, weight]) => {
        if (known.has(genre) || genres.has(genre)) squares += weight * weight;
      });
      return Math.sqrt(squares);
    };
    if (demo.profileLength === undefined) {
      demo.profileLength = length(new Set(poolSongs(demo).flatMap((song) => song.genres)));
    }
    const now = length(new Set(songs.flatMap((song) => song.genres)));
    const scale = now > 0 ? demo.profileLength / now : 0;
    const matches = songs.map((song) => (scale === 1 ? song.genre_match : song.genre_match * scale));
    const withGenres = matches.filter((value, i) => songs[i].genres.length);
    if (withGenres.length) {
      const middle = median(withGenres);
      songs.forEach((song, i) => { if (!song.genres.length) matches[i] = middle; });
    }
    return matches;
  }

  function defaults(demo, mode) {
    const c = demo.constants;
    if (mode === "discover") {
      return {
        top: 10, familiarity: c.FAMILIARITY, variety: c.VARIETY.discover, mainstream: c.POPULARITY_WEIGHT,
        genres: [], genre_boost: c.GENRE_BOOST, western: c.WESTERN, new_artists: false,
      };
    }
    if (mode === "rediscover") return { top: 10, variety: c.VARIETY.rediscover };
    return { top: 10, song: "", same_artist: false };
  }

  // Songs new to the listener: recommend.py discover() and rank_discover(), with the dials of settings. exclude
  // holds song keys left out (songs shown already), rest the artist keys having a rest.
  function discover(demo, given, exclude, rest) {
    const s = Object.assign(defaults(demo, "discover"), given);
    const c = demo.constants;
    let songs = poolSongs(demo).filter((song) => !exclude.has(song.song_key) && !rest.has(song.artist_key));
    if (s.new_artists) songs = songs.filter((song) => !song.played_artist);
    const notes = [];
    if (demo.discover.struck.length) {
      notes.push(`Left out ${demo.discover.struck.join(", ")}: the listener disliked ${c.STRIKES} or more of their songs new to them and liked none`);
    }
    if (!songs.length) return { rows: [], notes: notes.concat(["No songs in the pool qualify"]) };
    const genreMatch = genreMatches(demo, songs);
    const top = Math.max(...songs.map((song) => song.similar_sum));
    const popularity = percentRanks(songs.map((song) => song.listeners));
    const wanted = new Set(s.genres);
    let ranked = songs.map((song, i) => {
      const similar = top > 0 ? song.similar_sum / top : song.similar_sum;
      const pick = genrePick(song.genres, wanted);
      const western = song.asian ? 0 : 1;
      const familiar = s.familiarity * Math.max(song.artist_match, 0) + Math.min(song.artist_match, 0);
      const taste = (1 - s.familiarity) * (genreMatch[i] + similar) / 2 + familiar;
      const score = taste + s.mainstream * popularity[i] + s.genre_boost * pick - s.western * western;
      return { song, genreMatch: genreMatch[i], similar, popularity: popularity[i], pick, western, score, liked: NaN };
    });
    ranked.sort((a, b) => b.score - a.score || b.song.listeners - a.song.listeners);
    let order = ranked.map((item) => item.score);
    const model = demo.discover.model;
    if (model) {
      const best = ranked.slice(0, c.SHORTLIST);
      best.forEach((item) => {
        item.liked = likedChance(model, {
          genre_match: item.genreMatch, artist_match: item.song.artist_match, similar_match: item.similar,
          asian: item.song.asian ? 1 : 0, log_listeners: Math.log10(item.song.listeners + 1),
          log_rank: Math.log10(item.song.artist_rank),
        }, item.song.genres);
      });
      best.sort((a, b) => b.liked - a.liked || b.score - a.score);
      ranked = best.concat(ranked.slice(c.SHORTLIST));
      order = ranked.map((item) => (Number.isNaN(item.liked) ? 0 : item.liked));
    }
    const rows = ranked.map((item) => genreRow(item.song.genres));
    const { listed, forVariety } = listPositions(order, rows, ranked.map((item) => item.song.artist_key), s.variety, s.top, c);
    return {
      notes,
      rows: listed.map((position) => {
        const item = ranked[position];
        const why = item.song.why_head.slice();
        if (s.genre_boost > 0 && item.pick > 0) {
          why.push(`${item.song.genres.find((genre) => wanted.has(genre))}, a genre you asked for more of`);
        }
        why.push(...item.song.why_tail);
        if (forVariety.has(position)) why.push(c.FOR_VARIETY);
        return {
          title: item.song.title, artist: item.song.artist, artist_key: item.song.artist_key, song_key: item.song.song_key,
          genres: item.song.genres, score: item.score, liked_chance: item.liked, why,
          parts: { genre_match: item.genreMatch, artist_match: item.song.artist_match, similar_match: item.similar, popularity: item.popularity, genre_pick: item.pick, western: item.western },
        };
      }),
    };
  }

  // Songs to rediscover: ranked by the model learned from the history (in the data already), then the cap per
  // artist and variety, as in recommend.py rediscover().
  function rediscover(demo, given, exclude, rest) {
    const s = Object.assign(defaults(demo, "rediscover"), given);
    const songs = demo.rediscover.filter((song) => !exclude.has(song.song_key) && !rest.has(song.artist_key));
    if (!songs.length) return { rows: [], notes: ["No more songs to rediscover in this demo"] };
    const { listed, forVariety } = listPositions(
      songs.map((song) => song.chance), songs.map((song) => genreRow(song.genres)), songs.map((song) => song.artist_key),
      s.variety, s.top, demo.constants,
    );
    return {
      notes: [],
      rows: listed.map((position) => {
        const song = songs[position];
        const why = song.why.slice();
        if (forVariety.has(position)) why.push(demo.constants.FOR_VARIETY);
        return Object.assign({}, song, { why });
      }),
    };
  }

  // recommend.py find_song: the most played song whose title, "artist - title" or "title - artist" holds query.
  function findSong(demo, query) {
    const wanted = query.toLowerCase().trim();
    const ways = [
      (song) => song.title.toLowerCase(),
      (song) => `${song.artist.toLowerCase()} - ${song.title.toLowerCase()}`,
      (song) => `${song.title.toLowerCase()} - ${song.artist.toLowerCase()}`,
    ];
    for (const way of ways) {
      const found = demo.history.findIndex((song) => way(song).includes(wanted));
      if (found >= 0) return found;
    }
    return -1;
  }

  function times(count) {
    return `${count} time${count === 1 ? "" : "s"}`;
  }

  // Songs like one named: recommend.py similar_songs(), from the lists the build kept for every song.
  function like(demo, given, exclude) {
    const s = Object.assign(defaults(demo, "like"), given);
    const seed = findSong(demo, s.song);
    if (seed < 0) return { error: `No song in the listener's history matches '${s.song}'` };
    const named = demo.history[seed];
    const found = (demo.like[seed] || { other: [], same: [] })[s.same_artist ? "same" : "other"]
      .filter(([number]) => !exclude.has(demo.history[number].song_key))
      .slice(0, s.top);
    return {
      header: `Songs like ${named.title} by ${named.artist} (${named.genres.join(", ") || "no genres on Last.fm"})`,
      notes: found.length ? [] : ["No more songs like it in this demo"],
      rows: found.map(([number, similarity, session, genre, together, company]) => {
        const song = demo.history[number];
        const why = [];
        if (together > 0) why.push(`you played them in the same session ${times(together)}`);
        else if (company) why.push("you play it with the same songs");
        const shared = song.genres.filter((g) => named.genres.includes(g));
        if (shared.length) why.push(`shares ${shared.join(", ")}`);
        why.push(`you played it ${times(song.plays)}`);
        return Object.assign({}, song, { similarity, session_similarity: session, genre_similarity: genre, why });
      }),
    };
  }

  const engine = { defaults, discover, rediscover, like, findSong, genreRow, percentRanks };
  root.DemoEngine = engine;
  if (typeof module !== "undefined" && module.exports) module.exports = engine;
})(typeof window !== "undefined" ? window : globalThis);
