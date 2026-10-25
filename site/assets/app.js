/* The demo website: the five pages of app.py in plain HTML, on the made-up listener of data/demo.js.
 *
 * Lists come from engine.js, the recommender's ranking, and are made as on the website: a list at a time, Like,
 * Neutral and Dislike marked on each card, and the next list once every song has a mark or on Next songs, leaving
 * out the songs shown and giving other artists a turn. What differs is what happens to your choices: here they
 * stay in this tab (state.marks) and change no list, profile or model, and they are gone when the tab closes.
 * Nothing is stored or sent: this file has no network or storage code, and the page's Content-Security-Policy
 * (index.html) would block both anyway.
 */
(function () {
  "use strict";

  const demo = window.DEMO;
  const engine = window.DemoEngine;
  const charts = window.DemoCharts;
  const C = demo.constants;
  const REPO = "https://github.com/Syakirin03/ML-music";
  const PAGES = ["recommendations", "ratings", "data", "how-it-works", "settings"];
  const MODES = { discover: "Songs new to you", rediscover: "Songs to rediscover", like: "Songs like one you name" };
  const RATE = [["like", "Like", "i-like"], ["neutral", "Neutral", "i-neutral"], ["dislike", "Dislike", "i-dislike"]];
  const WORDS = { like: "liked", neutral: "neutral", dislike: "disliked" };
  const MODE_WORDS = { discover: "new to you", rediscover: "to rediscover", like: "like one you named" };
  const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
  // Genre names written in capitals, as on the website (app_pages/look.py).
  const UPPER = new Set(["edm", "r&b", "opm", "uk", "us", "idm", "ebm", "dnb", "ost", "dj"]);
  // The website's wallpapers (app_pages/look.py): blobs of colour (rgb, x %, y %) over a base, light and dark.
  const WALLPAPERS = {
    aurora: ["Aurora", [["29,155,240", 12, 8], ["120,86,255", 88, 18], ["0,186,124", 50, 100]]],
    sunset: ["Sunset", [["255,122,89", 15, 85], ["249,24,128", 85, 15], ["120,86,255", 60, 60]]],
    ocean: ["Ocean", [["0,194,255", 20, 20], ["29,155,240", 80, 40], ["0,186,124", 40, 100]]],
    sakura: ["Sakura", [["255,143,177", 15, 15], ["255,179,199", 85, 70], ["195,166,255", 60, 10]]],
    neon: ["Tokyo neon", [["249,24,128", 10, 90], ["0,229,255", 90, 10], ["120,86,255", 50, 50]]],
    forest: ["Forest", [["0,186,124", 15, 20], ["120,200,80", 85, 80], ["0,120,90", 60, 30]]],
    plain: ["Plain", []],
  };
  const BASES = {
    dark: "linear-gradient(160deg, #05070d 0%, #0b0f1a 55%, #070a12 100%)",
    light: "linear-gradient(160deg, #f7f9fc 0%, #eef2f8 55%, #f5f3fb 100%)",
  };
  const BLOB_ALPHA = { dark: 0.5, light: 0.34 };
  const DIM = { dark: "0,0,0", light: "255,255,255" };
  // What each input of the models is about, as on the website's How it works page.
  const NAMES = {
    genre_match: "Genres close to your taste", artist_match: "Artist you play", similar_match: "Similar to your artists",
    asian: "Asian artist", log_listeners: "Last.fm listeners", log_rank: "Lower in the artist's charts",
    song_match: "Played it a lot lately", log_listens: "Listens in all", log_days_dormant: "Days unplayed",
    log_days_known: "Days since first played", finish_rate: "Share finished", skip_rate: "Share skipped",
    artist_recent: "Artist played this month", genre_recent: "Genre played this month", has_genres: "Has genres",
  };

  const state = {
    mode: "discover",
    top: 10,
    settings: {
      discover: engine.defaults(demo, "discover"),
      rediscover: engine.defaults(demo, "rediscover"),
      like: engine.defaults(demo, "like"),
    },
    page: null, // the list on screen: { mode, rows, notes, header, error }
    shown: new Set(), // songs shown since the settings last changed
    artistSongs: new Map(), // artist -> songs shown since then
    choices: new Map(), // song_key -> { action, row, mode }: marked on the list on screen
    marks: [], // choices kept when a list was done: this tab's "ratings", which change nothing
    last: null, // what the last list's choices were
    look: { theme: "system", wallpaper: "aurora", dim: 0.35 },
    drawn: [], // the charts of the page on screen, to draw again when its width or colours change
    built: new Set(), // pages built already
  };

  // Small helpers.
  function h(tag, props, ...children) {
    const element = document.createElement(tag);
    Object.entries(props || {}).forEach(([name, value]) => {
      if (value === null || value === undefined || value === false) return;
      if (name === "class") element.className = value;
      else if (name === "text") element.textContent = value;
      else if (name.startsWith("on")) element.addEventListener(name.slice(2), value);
      else element.setAttribute(name, value === true ? "" : value);
    });
    children.flat().forEach((child) => {
      if (child !== null && child !== undefined && child !== false) element.append(child instanceof Node ? child : String(child));
    });
    return element;
  }

  function icon(name) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "icon");
    svg.setAttribute("aria-hidden", "true");
    const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    use.setAttribute("href", `#${name}`);
    svg.append(use);
    return svg;
  }

  const $ = (selector) => document.querySelector(selector);
  const number = (value) => Math.round(value).toLocaleString("en-US");
  const percent = (value) => `${Math.round(value * 100)}%`;
  const songs = (count) => `${count} song${count === 1 ? "" : "s"}`;
  const capital = (text) => text.charAt(0).toUpperCase() + text.slice(1);

  function genreLabel(genre) {
    const text = String(genre).split(" ").map((word) => (UPPER.has(word) ? word.toUpperCase() : word)).join(" ");
    return capital(text);
  }

  function monthYear(iso) {
    return `${MONTHS[Number(iso.slice(5, 7)) - 1]} ${iso.slice(0, 4)}`;
  }

  function hero(kicker, title, sub, stats) {
    return h("div", { class: "hero" },
      h("div", { class: "kicker", text: kicker }),
      h("h1", { class: "title", text: title }),
      sub ? h("p", { class: "sub" }, sub) : null,
      stats && stats.length ? h("div", { class: "stats" }, stats.map(([value, words]) => h("span", { class: "stat" }, h("b", { text: value }), ` ${words}`))) : null);
  }

  function info(...content) {
    return h("div", { class: "panel note" }, icon("i-info"), h("div", {}, ...content));
  }

  function steps(items) {
    return h("div", { class: "steps" }, items.map(([head, rest], i) => h("div", { class: "step" },
      h("div", { class: "step-n", text: String(i + 1) }),
      h("div", { class: "step-text" }, h("b", { text: head }), " ", h("span", { text: rest })))));
  }

  function table(columns, rows) {
    return h("div", { class: "table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, columns.map(([name, kind]) => h("th", { class: kind === "number" ? "number" : null, scope: "col", text: name })))),
      h("tbody", {}, rows.map((row) => h("tr", {}, row.map((cell, i) => h("td", { class: columns[i][1] === "number" ? "number" : null, text: cell })))))));
  }

  function numbers(columns, rows) {
    return h("details", { class: "numbers" }, h("summary", { text: "Show the numbers" }), table(columns, rows));
  }

  // A chart in its panel: drawn now, and again when the page's width or colours change.
  function chart(draw) {
    const box = h("div", { class: "chart", id: `chart-${state.drawn.length + Math.random().toString(36).slice(2, 7)}` });
    state.drawn.push(() => { if (box.isConnected) draw(box); });
    requestAnimationFrame(() => draw(box));
    return box;
  }

  function redraw() {
    charts.untip();
    state.drawn.forEach((draw) => draw());
  }

  // The look: light or dark, and the wallpaper with its dimming (app_pages/look.py wallpaper_css).
  function kind() {
    if (state.look.theme !== "system") return state.look.theme;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function wallpaper(name, mode, dim) {
    const blobs = WALLPAPERS[name][1];
    if (!blobs.length) return mode === "dark" ? "#000000" : "#ffffff";
    const layers = blobs.map(([rgb, x, y]) => `radial-gradient(900px 620px at ${x}% ${y}%, rgba(${rgb},${BLOB_ALPHA[mode]}) 0%, rgba(${rgb},0) 70%)`);
    const veil = `linear-gradient(rgba(${DIM[mode]},${dim}), rgba(${DIM[mode]},${dim}))`;
    return [veil, ...layers, BASES[mode]].join(", ");
  }

  function applyLook() {
    const root = document.documentElement;
    if (state.look.theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", state.look.theme);
    root.style.setProperty("--mp-wallpaper", wallpaper(state.look.wallpaper, kind(), state.look.dim));
    redraw();
  }

  // Lists, as on the website (app_pages/recommendations.py build()).
  function resting() {
    const rest = new Set();
    state.artistSongs.forEach((keys, artist) => { if (keys.size >= C.MAX_PER_ARTIST) rest.add(artist); });
    return rest;
  }

  function build(next) {
    const settings = Object.assign({}, state.settings[state.mode], { top: state.top });
    const exclude = next ? new Set(state.shown) : new Set();
    let result;
    if (state.mode === "discover") result = engine.discover(demo, settings, exclude, next ? resting() : new Set());
    else if (state.mode === "rediscover") result = engine.rediscover(demo, settings, exclude, next ? resting() : new Set());
    else if (settings.song) result = engine.like(demo, settings, exclude);
    else result = null;
    state.choices = new Map();
    if (!result) {
      state.page = null;
      return;
    }
    const rows = result.rows || [];
    rows.forEach((row) => {
      state.shown.add(row.song_key);
      if (!state.artistSongs.has(row.artist_key)) state.artistSongs.set(row.artist_key, new Set());
      state.artistSongs.get(row.artist_key).add(row.song_key);
    });
    state.page = { mode: state.mode, rows, notes: result.notes || [], header: result.header, error: result.error };
  }

  function keepChoices() {
    const choices = [...state.choices.values()];
    state.choices = new Map();
    state.last = null;
    if (!choices.length) return;
    const counts = {};
    choices.forEach((choice) => {
      counts[choice.action] = (counts[choice.action] || 0) + 1;
      state.marks.push({ row: choice.row, mode: choice.mode, action: choice.action, at: new Date() });
    });
    state.last = { counts, total: choices.length };
  }

  function settingsChanged() {
    if (state.choices.size) keepChoices();
    else state.last = null;
    state.shown = new Set();
    state.artistSongs = new Map();
    build(false);
    renderRecommendations();
    renderSidebar();
  }

  function nextList() {
    keepChoices();
    build(true);
    renderRecommendations();
    renderSidebar();
    const smooth = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    window.scrollTo({ top: 0, behavior: smooth ? "smooth" : "auto" });
    const first = $("#rec-cards .card");
    if (first) first.querySelector("button").focus({ preventScroll: true });
  }

  function choose(row, action, card) {
    const chosen = state.choices.get(row.song_key);
    if (chosen && chosen.action === action) state.choices.delete(row.song_key);
    else state.choices.set(row.song_key, { action, row, mode: state.page.mode });
    if (state.choices.size >= state.page.rows.length) {
      nextList();
      return;
    }
    markCard(card, row);
    renderProgress();
    renderNext();
    renderSidebar();
  }

  // The marks that count on Your ratings: the newest of each song.
  function currentMarks() {
    const newest = new Map();
    state.marks.forEach((mark) => newest.set(mark.row.song_key, mark));
    return [...newest.values()];
  }

  // The Recommendations page.
  function intro(mode) {
    if (mode === "discover") {
      return ["Songs new to you",
        `Songs you have never played, picked from a pool of ${number(demo.discover.pool_songs)} songs by ${demo.discover.pool_artists} ` +
        "artists that Last.fm's charts and similar-artist lists give for your taste. Choose Like, Neutral or Dislike for each: " +
        "once you've chosen them all, or press Next songs, the next songs come. In the app your choices would be saved and " +
        "the next songs picked with them; here they stay in this tab."];
    }
    if (mode === "rediscover") {
      return ["Songs to rediscover",
        "Songs you loved that have gone quiet: at least three full listens, none lately. A model trained on your history " +
        "guesses which you would go back to, and once you've rated enough of them, one trained on your ratings which you want back now."];
    }
    return ["Songs like one you name",
      "Songs from your history that you play in the same sessions as the one you name, or that share its genres. Type a title in the sidebar."];
  }

  function ordering(page) {
    if (!page || !page.rows.length) return [];
    if (page.mode === "discover" && page.rows.some((row) => !Number.isNaN(row.liked_chance))) {
      const average = page.rows.reduce((sum, row) => sum + row.liked_chance, 0) / page.rows.length;
      return [["Ranked by", "a model learned from your ratings"], [percent(average), "average chance you'll like these"]];
    }
    if (page.mode === "rediscover") return [["Ranked by", "a model learned from your history"]];
    if (page.mode === "like") return [["Ranked by", "how alike you play them, and their genres"]];
    return [["Ranked by", "the hand-set score"]];
  }

  function meter(value, words, shown) {
    const fill = h("span", {});
    fill.style.width = `${Math.max(0, Math.min(1, value)) * 100}%`;
    return h("div", { class: "meter" },
      h("div", { class: "meter-top" }, h("span", { class: "meter-value", text: shown || percent(value) }), h("span", { class: "meter-label", text: words })),
      h("div", { class: "bar" }, fill));
  }

  function valueBlock(mode, row) {
    if (mode === "like") return meter(Math.min(row.similarity, 1), "similarity to the song you named", row.similarity.toFixed(2));
    if (mode === "rediscover") return meter(row.chance, `chance you play it again within ${C.HORIZON_DAYS} days`);
    if (!Number.isNaN(row.liked_chance)) return meter(row.liked_chance, "chance you'll like it");
    return h("div", { class: "second", text: `Score ${row.score.toFixed(2)}: the model learned from your ratings isn't in use yet` });
  }

  function chips(genres, wanted) {
    return h("div", { class: "chips" }, genres.map((genre, i) => h("span", {
      class: `chip${i === 0 ? " main" : ""}${wanted.has(genre) ? " wanted" : ""}`, text: genreLabel(genre),
    })));
  }

  function card(row, position, mode, wanted) {
    const actions = h("div", { class: "actions", role: "group", "aria-label": `Rate ${row.title}` });
    const element = h("article", { class: "card", "aria-label": `${position}. ${row.title} by ${row.artist}` },
      h("div", { class: "player" },
        h("span", { class: "player-button" }, icon("i-play")),
        h("div", {}, h("b", { text: "Spotify's player" }), "plays the song here in the app")),
      h("div", { class: "info" },
        h("div", { class: "song-head" }, h("span", { class: "pos", text: String(position) }),
          h("div", {}, h("div", { class: "song-title", text: row.title }), h("div", { class: "song-artist", text: row.artist }))),
        row.genres.length ? chips(row.genres, wanted) : null,
        valueBlock(mode, row),
        h("p", { class: "why", text: capital(row.why.join("; ")) })),
      actions);
    RATE.forEach(([action, words]) => {
      actions.append(h("button", { class: "button", type: "button", "data-action": action, onclick: () => choose(row, action, element) }));
    });
    markCard(element, row);
    return element;
  }

  function markCard(element, row) {
    const chosen = state.choices.get(row.song_key);
    element.querySelectorAll(".actions button").forEach((button) => {
      const [action, words, name] = RATE.find(([value]) => value === button.dataset.action);
      const on = Boolean(chosen && chosen.action === action);
      button.setAttribute("aria-pressed", String(on));
      button.title = on ? "Your choice: click again to take it back" : "";
      button.replaceChildren(icon(on ? "i-chosen" : name), words);
    });
  }

  function renderProgress() {
    const box = $("#rec-progress");
    const page = state.page;
    if (!page || !page.rows.length) {
      box.replaceChildren();
      return;
    }
    const total = page.rows.length;
    const marked = state.choices.size;
    const fill = h("span", {});
    fill.style.width = `${(marked / total) * 100}%`;
    box.replaceChildren(h("div", { class: "panel sticky-progress", role: "status" },
      `Chosen ${marked} of ${total}. Once you've chosen all ${total}, or press Next songs, the next songs come. Click a choice again to take it back.`,
      h("div", { class: "bar" }, fill)));
  }

  function renderNext() {
    const box = $("#rec-next");
    const page = state.page;
    if (!page || !page.rows.length) {
      box.replaceChildren();
      return;
    }
    const marked = state.choices.size;
    const rest = page.mode === "like" ? "" :
      ` and gives other artists a turn: those with ${C.MAX_PER_ARTIST} songs shown rest (in the app, an artist you liked keeps their turn)`;
    box.replaceChildren(h("div", { class: "next" },
      h("button", { class: "button primary", type: "button", onclick: nextList },
        marked ? `Keep ${songs(marked)} here and show the next ones` : "Next songs", icon("i-next")),
      h("p", { text: `${marked ? "The songs you didn't choose are skipped. " : "None chosen: these songs are skipped. "}` +
        `The next list shows songs not shown yet since you last changed a setting${rest}.` })));
  }

  function renderRecommendations() {
    const page = state.page;
    const [title, sentence] = intro(state.mode);
    $("#rec-hero").replaceChildren(hero("Recommendations", title, sentence, ordering(page)));
    const notes = [];
    if (page && page.header) notes.push(h("p", { class: "small muted", text: page.header }));
    if (page) page.notes.forEach((note) => notes.push(info(note)));
    $("#rec-notes").replaceChildren(...notes);
    const last = state.last;
    $("#rec-sent").replaceChildren(last ? h("div", { class: "panel sent", role: "status" },
      h("p", {}, h("b", { text: `Marked ${songs(last.total)}` }),
        `: ${Object.entries(last.counts).map(([action, count]) => `${count} ${WORDS[action]}`).join(", ")}. ` +
        "In the app they would now be saved, count in your profile and teach the models; here they stay in this tab, " +
        "and the songs below are picked as before."),
      h("a", { class: "button quiet", href: "#ratings", text: "See them" })) : "");
    renderProgress();
    const wanted = new Set(state.mode === "discover" ? state.settings.discover.genres : []);
    let cards;
    if (!page) {
      cards = [info("Type a song you know in the sidebar, or pick one of those under it, and songs like it appear here.")];
    } else if (page.error) {
      cards = [info(page.error)];
    } else {
      cards = page.rows.map((row, i) => card(row, i + 1, page.mode, wanted));
    }
    $("#rec-cards").replaceChildren(...cards);
    renderNext();
  }

  // The sidebar: the list's settings, and how far the ratings are.
  function renderSidebar() {
    $("#sidebar-summary").textContent = `${MODES[state.mode]} · ${songs(state.top)}`;
    document.querySelectorAll("[data-mode]").forEach((block) => { block.hidden = block.dataset.mode !== state.mode; });
    const picked = $("#genre-picked");
    picked.replaceChildren(...state.settings.discover.genres.map((genre) => h("button", {
      type: "button", "aria-label": `Remove ${genreLabel(genre)}`,
      onclick: () => {
        state.settings.discover.genres = state.settings.discover.genres.filter((other) => other !== genre);
        settingsChanged();
      },
    }, genreLabel(genre), icon("i-close"))));
    const d = demo.discover;
    const rated = d.likes + d.dislikes + d.neutral;
    $("#ratings-summary").textContent = `${number(rated)} songs rated: ${d.likes} liked, ${d.neutral} neutral, ${d.dislikes} disliked.`;
    $("#ratings-progress").replaceChildren(...[["Likes", d.likes], ["Dislikes", d.dislikes]].map(([words, count]) => {
      const fill = h("span", {});
      fill.style.width = `${Math.min(count / C.MIN_EACH, 1) * 100}%`;
      return h("div", { class: "progress-line" }, `${words} of songs new to you: ${count} of the ${C.MIN_EACH} a model needs`, h("div", { class: "bar" }, fill));
    }));
    const kept = currentMarks().length;
    const pending = state.choices.size;
    $("#clicks-summary").textContent = kept || pending
      ? `Your clicks here: ${songs(kept)} kept in this tab${pending ? `, ${pending} more chosen on this list` : ""}. None of them is sent or changes the lists.`
      : "Your clicks here stay in this tab: none of them is sent or changes the lists.";
  }

  function wireSidebar() {
    document.querySelectorAll('input[name="mode"]').forEach((radio) => {
      radio.addEventListener("change", () => {
        if (!radio.checked) return;
        state.mode = radio.value;
        settingsChanged();
      });
    });
    const top = $("#top");
    top.addEventListener("input", () => { $("#top-out").textContent = top.value; });
    top.addEventListener("change", () => { state.top = Number(top.value); settingsChanged(); });
    document.querySelectorAll("[data-setting]").forEach((input) => {
      const mode = input.closest("[data-mode]").dataset.mode;
      const name = input.dataset.setting;
      const output = document.querySelector(`output[for="${input.id}"]`);
      if (input.type === "checkbox") input.checked = state.settings[mode][name];
      else input.value = state.settings[mode][name];
      const show = () => { if (output) output.textContent = Number(input.value).toFixed(2).replace(/0$/, ""); };
      show();
      input.addEventListener("input", show);
      input.addEventListener("change", () => {
        state.settings[mode][name] = input.type === "checkbox" ? input.checked : Number(input.value);
        settingsChanged();
      });
    });
    const add = $("#genre-add");
    demo.discover.genres.forEach((genre) => add.append(h("option", { value: genre, text: genreLabel(genre) })));
    add.addEventListener("change", () => {
      if (add.value && !state.settings.discover.genres.includes(add.value)) {
        state.settings.discover.genres = state.settings.discover.genres.concat([add.value]);
        settingsChanged();
      }
      add.value = "";
    });
    const song = $("#song");
    const options = $("#song-options");
    demo.history.forEach((item) => options.append(h("option", { value: `${item.title} - ${item.artist}` })));
    const named = new Set(demo.history.map((item) => `${item.title} - ${item.artist}`));
    const apply = () => {
      if (song.value.trim() === state.settings.like.song) return;
      state.settings.like.song = song.value.trim();
      settingsChanged();
    };
    song.addEventListener("change", apply);
    song.addEventListener("input", () => { if (named.has(song.value)) apply(); });
    const tried = new Set();
    const examples = demo.history.filter((item) => !tried.has(item.artist) && tried.add(item.artist)).slice(0, 3);
    $("#song-try").replaceChildren("Try: ", ...examples.map((item) => h("button", {
      type: "button", text: item.title,
      onclick: () => { song.value = `${item.title} - ${item.artist}`; state.settings.like.song = song.value; settingsChanged(); },
    })));
    $("#dormant-note").textContent = `Songs count as gone quiet after ${C.DORMANT_DAYS} days without a play (the app lets you change that).`;
    const toggle = $("#sidebar-toggle");
    const sidebar = $("#sidebar");
    if (window.matchMedia("(max-width: 960px)").matches) {
      sidebar.classList.add("closed");
      toggle.setAttribute("aria-expanded", "false");
    }
    toggle.addEventListener("click", () => {
      const closed = sidebar.classList.toggle("closed");
      toggle.setAttribute("aria-expanded", String(!closed));
    });
  }

  // Your ratings: the choices kept in this tab.
  function renderRatings() {
    const section = $("#page-ratings");
    const marks = currentMarks();
    const count = (action) => marks.filter((mark) => mark.action === action).length;
    const filters = state.ratingsFilter || (state.ratingsFilter = { kind: "all", from: "all" });
    const kinds = { all: "All", like: "Liked", neutral: "Neutral", dislike: "Disliked" };
    const lists = { all: "All lists", discover: "New to you", rediscover: "To rediscover", like: "Like one you named" };
    const segmented = (options, key) => h("div", { class: "segmented", role: "group" }, Object.entries(options).map(([value, words]) => h("button", {
      type: "button", "aria-pressed": String(filters[key] === value), text: words,
      onclick: () => { filters[key] = value; renderRatings(); },
    })));
    const chosen = marks.slice().reverse()
      .filter((mark) => filters.kind === "all" || mark.action === filters.kind)
      .filter((mark) => filters.from === "all" || mark.mode === filters.from);
    const pending = state.choices.size;
    const d = demo.discover;
    section.replaceChildren(
      hero("Your ratings", "What you told the recommender",
        "Every song you liked, rated neutral or disliked in this tab, newest first. In the app, a like or a dislike counts in " +
        "your taste profile and teaches the two models learned from your ratings, and Undo takes a rating back. Here your " +
        "choices change nothing, are never sent, and are gone when you close or reload the tab.",
        [[String(count("like")), "liked"], [String(count("neutral")), "neutral"], [String(count("dislike")), "disliked"], [String(marks.length), "songs in all"]]),
      pending ? info(`${songs(pending)} chosen on the Recommendations page ${pending === 1 ? "is" : "are"} not kept yet: choices are kept once you've chosen the whole list or press Next songs.`) : "",
      marks.length ? h("div", { class: "filters" }, segmented(kinds, "kind"), segmented(lists, "from")) : "",
      marks.length
        ? h("div", { class: "panel" }, chosen.length ? chosen.map((mark) => h("div", { class: "rated" },
          h("div", { class: "what" },
            h("div", {}, h("b", { text: capital(WORDS[mark.action]) }), " · ", h("b", { text: mark.row.title }), ` · ${mark.row.artist}`),
            h("span", { text: `${capital(MODE_WORDS[mark.mode])}${mark.row.genres.length ? `, ${mark.row.genres.slice(0, 3).map(genreLabel).join(", ")}` : ""}, ${mark.at.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}` })),
          h("button", {
            class: "button", type: "button", "aria-label": `Undo your rating of ${mark.row.title}`,
            onclick: () => { state.marks = state.marks.filter((other) => other.row.song_key !== mark.row.song_key); renderRatings(); renderSidebar(); },
          }, icon("i-undo"), "Undo"))) : h("p", { class: "muted", text: "No ratings of that kind." }))
        : info("Nothing chosen yet: like, rate neutral or dislike songs on the Recommendations page."),
      h("div", { class: "panel" }, h("p", {},
        h("b", { text: "The made-up listener's own ratings. " }),
        `Before you came, the listener rated ${d.likes + d.dislikes + d.neutral} songs new to them: ${d.likes} likes, ${d.dislikes} dislikes and ${d.neutral} neutral, ` +
        "made up by a rule of the demo's own that the recommender doesn't know. A model learned the rule from them, and it ranks the songs new to you: ",
        h("a", { href: "#how-it-works", text: "How it works" }), " shows what it learned.")),
    );
  }

  // Your data: the made-up listener's database, as the website's Your data page shows it.
  function renderData() {
    const data = demo.data;
    const s = data.steps;
    const years = (Date.parse(s.last_play) - Date.parse(s.first_play)) / (365.25 * 86400000);
    const layers = {
      bronze: "The exports exactly as downloaded: every field, every duplicate, as text",
      silver: "Typed, renamed and deduplicated: plays, tracks, devices and log events, linked by keys",
      techlog: "One typed table per kind of technical-log event (likes, playlist adds, what Spotify showed you)",
      gold: "Tables made for machine learning: one row per play with features known at that moment, sessions, genres",
    };
    const section = $("#page-data");
    section.replaceChildren(
      hero("Your data", "What the recommender learns from",
        `Your Spotify listening since ${monthYear(s.first_play)}, cleaned up and stored in one database file, then explored. ` +
        "Here it is the made-up listener's: plays the demo made up and ran through the project's real pipeline.",
        [[number(s.plays), "plays"], [number(s.hours), "hours"], [number(data.songs), "songs"], [number(s.artists), "artists"], [years.toFixed(1), "years"]]),
      h("h2", { text: "Where the data comes from" }),
      h("div", { class: "columns" },
        h("div", { class: "panel" }, h("b", { text: "Spotify streaming history" }), h("p", { class: "small muted", text:
          `Every play since ${monthYear(s.first_play)}: when it stopped, what it was, how long you listened and how it ended. Spotify sends it as JSON files when you ask for your data.` })),
        h("div", { class: "panel" }, h("b", { text: "Spotify technical log" }), h("p", { class: "small muted", text: s.log_records
          ? `${number(s.log_records)} records of ${s.log_event_types} kinds of app events: your likes, playlist adds and what Spotify showed you.`
          : "Not in this export. Spotify's technical log, a separate request, adds your likes, playlist adds and what Spotify showed you." })),
        h("div", { class: "panel" }, h("b", { text: "Last.fm" }), h("p", { class: "small muted", text:
          "Genres for each song, from the tags Last.fm listeners give, and the pool of songs new to you: charts of your top artists, their similar artists, and your genres' top artists. In this demo, made up too." }))),
      h("p", { class: "small muted", text: "A real export also holds your IP addresses, email and device ids. They stay in the database and are never shown on the website." }),
      h("h2", { text: "How it is cleaned" }),
      steps([
        ["Loaded as exported.", `${number(s.raw_streams)} streaming rows and ${number(s.log_records)} technical-log records go into the bronze layer unchanged, so nothing is ever lost.`],
        ["Exact duplicates removed.", `${number(s.raw_streams - s.streams)} rows appear twice in the export, leaving ${number(s.streams)}.`],
        ["Songs and podcasts apart.", `${number(s.song_streams)} song plays and ${number(s.podcast_streams)} podcast episodes; the recommender uses the songs.`],
        ["One play counted once.", `A song that stops at the same moment twice is one play: ${number(s.song_streams - s.plays)} repeats removed, ${number(s.plays)} plays left.`],
        ["Start times worked out.", "Spotify records when a song stopped; it started that long before. Times are in your time zone (config.toml), because the export's country comes from IP addresses."],
        ["Sessions found.", `A pause of 30 minutes or more starts a new session: ${number(s.sessions)} sessions.`],
        ["Each play labelled.", "Finished, skipped in its first 30 seconds, or partly heard. That gives each play a listen score: 1, 0, or the share heard."],
        ["What isn't music set aside.", `Spotify publishes some audio as music, such as white noise or audiobooks: ${number(s.not_music_plays)} plays (${percent(s.not_music_plays / s.plays)}) match the not-music rules of config.toml, and are kept apart from your music taste.`],
        ["Genres added.", `From Last.fm tags: ${percent(s.plays_with_genre / s.plays)} of plays have a genre.`],
        ["Versions merged.", `Remixes, language versions and re-releases of one song are one song: ${number(s.track_ids)} Spotify track ids make ${number(data.songs)} songs.`],
      ]),
      h("h2", { text: "How it is stored: the medallion layers" }),
      h("p", {}, "Data flows through layers that each add quality, a common design called the ", h("b", { text: "medallion architecture" }),
        ": bronze keeps the raw exports, silver cleans and links them, and gold holds tables ready for machine learning. All of it is one DuckDB file: a database that runs inside Python, with no server, queried with SQL."),
      h("div", { class: "flow", role: "img", "aria-label": "Spotify export, then bronze, silver and gold layers, then the recommender and the website; Last.fm feeds gold and the pool" },
        h("div", { class: "box" }, h("b", { text: "Spotify export" }), "JSON files"), h("span", { class: "arrow", text: "→" }),
        h("div", { class: "box layer" }, h("b", { text: "Bronze" }), "as exported"), h("span", { class: "arrow", text: "→" }),
        h("div", { class: "box layer" }, h("b", { text: "Silver" }), "clean, linked"), h("span", { class: "arrow", text: "→" }),
        h("div", { class: "box layer" }, h("b", { text: "Gold" }), "ready for ML"), h("span", { class: "arrow", text: "→" }),
        h("div", { class: "box" }, h("b", { text: "Recommender" }), "profile, models"), h("span", { class: "arrow", text: "⇄" }),
        h("div", { class: "box" }, h("b", { text: "Website" }), "lists, ratings"),
        h("span", { class: "muted small", text: "Last.fm adds genres to gold and the pool of new songs to the recommender." })),
      h("div", { class: "panel" }, table([["Layer"], ["What it holds"], ["Tables", "number"], ["Rows", "number"]],
        data.layers.map(([layer, tables, rows]) => [capital(layer), layers[layer] || "", String(tables), number(rows)])),
      h("p", { class: "small muted", text: `spotify.duckdb: ${data.file_mb} MB. Your ratings, taste profile and models are kept apart, in data/user/recommender.duckdb, because nothing can rebuild your ratings.` })),
      h("h2", { text: "What you listen to" }),
      h("p", { class: "small muted", text: "Music only: tracks that config.toml marks as not music are left out of these charts." }),
      monthsPanel(data), hoursPanel(data), genresPanel(data), artistsPanel(data), endsPanel(data), ratingsPanel(data),
    );
  }

  function monthsPanel(data) {
    const months = data.months.map(([month, hours, plays]) => ({ month, y: hours, plays }));
    const peak = months.reduce((best, item) => (item.y > best.y ? item : best), months[0]);
    const lastYear = months.slice(-12);
    const average = lastYear.reduce((sum, item) => sum + item.y, 0) / lastYear.length;
    return h("div", { class: "panel" },
      h("h3", { text: "Hours of music per month" }),
      h("p", { class: "small muted", text: `Your busiest month was ${monthYear(peak.month)}, with ${number(peak.y)} hours. Over the last 12 months you listened ${number(average)} hours a month on average.` }),
      chart((box) => charts.line(box, months, {
        description: "Hours of music per month, as a line",
        tick: (value) => charts.compact(value),
        xLabel: (item) => (item.month.endsWith("-01") ? item.month.slice(0, 4) : null),
        value: (item) => `${item.y.toFixed(1)} h`,
        words: (item) => `${monthYear(item.month)} · ${number(item.plays)} plays`,
      })),
      numbers([["Month"], ["Hours", "number"], ["Plays", "number"]], months.map((item) => [item.month, item.y.toFixed(1), number(item.plays)])));
  }

  function hoursPanel(data) {
    const cells = data.hours.map(([weekday, hour, hours, plays]) => ({ row: weekday - 1, column: hour, value: hours, plays }));
    const busiest = cells.reduce((best, item) => (item.value > best.value ? item : best), cells[0]);
    const byHour = new Array(24).fill(0);
    cells.forEach((item) => { byHour[item.column] += item.value; });
    const most = byHour.indexOf(Math.max(...byHour));
    const least = byHour.indexOf(Math.min(...byHour));
    const grid = DAYS.map((day, row) => [day, ...Array.from({ length: 24 }, (_, hour) => {
      const found = cells.find((item) => item.row === row && item.column === hour);
      return found ? found.value.toFixed(1) : "0.0";
    })]);
    return h("div", { class: "panel" },
      h("h3", { text: "When you listen" }),
      h("p", { class: "small muted", text: `Most of all on ${DAYS[busiest.row]} at ${busiest.column}:00. Across the week, ${most}:00 is your busiest hour and ${least}:00 your quietest. The made-up listener lives in UTC.` }),
      chart((box) => charts.heatmap(box, cells, {
        description: "Hours of music by day of the week and hour of the day",
        rows: DAYS, columns: Array.from({ length: 24 }, (_, hour) => String(hour)), every: 3, columnsTitle: "Hour of the day (your time zone)",
        value: (item) => `${item.value.toFixed(1)} h`,
        words: (item) => `${DAYS[item.row]} ${item.column}:00 · ${number(item.plays)} plays`,
        scaleLabel: (max) => `${number(max)} hours`,
      })),
      numbers([["Day"], ...Array.from({ length: 24 }, (_, hour) => [`${hour}:00`, "number"])], grid));
  }

  function genresPanel(data) {
    const labels = data.genre_labels;
    const years = [...new Set(data.genre_years.map(([year]) => year))].sort();
    const fills = ["--series-1", "--series-2", "--series-3", "--series-4", "--series-5"];
    const share = (year, label) => (data.genre_years.find(([y, l]) => y === year && l === label) || [0, "", 0, 0])[3];
    const hours = (year, label) => (data.genre_years.find(([y, l]) => y === year && l === label) || [0, "", 0, 0])[2];
    const name = (label) => (label === "Other" ? "Other" : genreLabel(label));
    const latest = years[years.length - 1];
    const lead = labels.filter((label) => label !== "Other").map((label) => [label, share(latest, label)]).sort((a, b) => b[1] - a[1])[0];
    const series = () => labels.map((label, i) => ({
      key: label, label: name(label),
      fill: getComputedStyle(document.documentElement).getPropertyValue(label === "Other" ? "--series-other" : fills[i]).trim(),
    }));
    return h("div", { class: "panel" },
      h("h3", { text: "Your genres over the years" }),
      h("p", { class: "small muted", text: `Each bar is one year of listening (years with under ${C.MIN_YEAR_HOURS} hours are left out). Your ${C.TOP_GENRE_COUNT} most played main genres overall have their own colour, the rest are Other. In ${latest}, ${name(lead[0])} took ${percent(lead[1])} of your music.` }),
      chart((box) => charts.stacked(box, {
        description: "Share of each year's music by main genre", categories: years, series: series(),
        share, words: (year, item) => `${item.label} · ${year} · ${hours(year, item.key).toFixed(0)} h`,
      })),
      numbers([["Year"], ...labels.map((label) => [name(label), "number"])], years.map((year) => [String(year), ...labels.map((label) => percent(share(year, label)))])));
  }

  function artistsPanel(data) {
    const years = [...new Set(data.artist_years.map(([year]) => year))].sort();
    const cell = (year, place) => {
      const found = data.artist_years.find(([y, p]) => y === year && p === place);
      return found ? `${found[2]} (${percent(found[3])})` : "";
    };
    const leader = data.artist_years.filter(([, place]) => place === 1).sort((a, b) => b[3] - a[3])[0];
    return h("div", { class: "panel" },
      h("h3", { text: "Your taste moves in phases" }),
      h("p", { class: "small muted", text: `Your three most played artists each year, with their share of the year's music. The largest share any artist ever took was ${leader[2]} in ${leader[0]}: ${percent(leader[3])}.` }),
      table([["Year"], ["Most played"], ["Second"], ["Third"]], years.map((year) => [String(year), cell(year, 1), cell(year, 2), cell(year, 3)])));
  }

  function endsPanel(data) {
    const kinds = [["finished", "Finished", "--series-1"], ["skipped", "Skipped", "--series-2"], ["interrupted", "Stopped another way", "--series-other"]];
    const years = [...new Set(data.end_years.map(([year]) => year))].sort();
    const find = (year, type) => data.end_years.find(([y, t]) => y === year && t === type) || [year, type, 0, 0];
    const latest = years[years.length - 1];
    return h("div", { class: "panel" },
      h("h3", { text: "How your plays end" }),
      h("p", { class: "small muted", text: `Share of each year's plays you finished, skipped, or that stopped another way (another app, a closed laptop). In ${latest} you finished ${percent(find(latest, "finished")[3])} and skipped ${percent(find(latest, "skipped")[3])}. Skips matter: a skip in the first 30 seconds counts against a song.` }),
      chart((box) => charts.stacked(box, {
        description: "Share of each year's plays by how they ended", categories: years,
        series: kinds.map(([key, label, fill]) => ({ key, label, fill: getComputedStyle(document.documentElement).getPropertyValue(fill).trim() })),
        share: (year, key) => find(year, key)[3], words: (year, item) => `${item.label} · ${year} · ${number(find(year, item.key)[2])} plays`,
      })),
      numbers([["Year"], ...kinds.map(([, label]) => [label, "number"])], years.map((year) => [String(year), ...kinds.map(([key]) => percent(find(year, key)[3]))])));
  }

  function ratingsPanel(data) {
    const rows = data.ratings_by_genre.map(([genre, likes, dislikes]) => ({ genre, likes, dislikes, value: likes / (likes + dislikes) }))
      .sort((a, b) => b.value - a.value);
    const panel = h("div", { class: "panel" }, h("h3", { text: "What your ratings say so far" }));
    if (rows.length < 2) {
      panel.append(h("p", { class: "small muted", text: "Rate a few more songs: this chart needs two genres with enough likes and dislikes." }));
      return panel;
    }
    const best = rows[0];
    const worst = rows[rows.length - 1];
    panel.append(
      h("p", { class: "small muted", text: `Share of your likes among your likes and dislikes, by the song's main genre, for genres with at least ${C.MIN_GENRE_RATINGS} ratings. The gray line is your overall share, ${percent(data.ratings_overall)}. You liked ${percent(best.value)} of ${genreLabel(best.genre)} songs and ${percent(worst.value)} of ${genreLabel(worst.genre)} ones: this is what the models learned from your ratings pick up (How it works).` }),
      chart((box) => charts.bars(box, rows.map((row) => ({ label: genreLabel(row.genre), value: row.value, words: `${genreLabel(row.genre)} · ${row.likes} likes, ${row.dislikes} dislikes` })), {
        description: "Share liked by main genre", max: 1, format: percent, rule: { value: data.ratings_overall, label: `overall ${percent(data.ratings_overall)}` },
      })),
      numbers([["Genre"], ["Likes", "number"], ["Dislikes", "number"], ["Share liked", "number"]], rows.map((row) => [genreLabel(row.genre), String(row.likes), String(row.dislikes), percent(row.value)])));
    return panel;
  }

  // How it works: the website's page, with this listener's models.
  function weightName(feature) {
    return feature.startsWith("genre: ") ? `Genre: ${genreLabel(feature.slice(7))}` : NAMES[feature] || feature;
  }

  function weightsPanel(weights, outcome, caption) {
    const rows = Object.entries(weights).map(([feature, value]) => ({ label: weightName(feature), value })).sort((a, b) => b.value - a.value);
    return [
      h("p", { class: "small muted", text: caption }),
      chart((box) => charts.diverging(box, rows, {
        description: `What the model learned: how each input changes the chance of ${outcome}`,
        up: `Makes ${outcome} likelier`, down: `Makes ${outcome} less likely`, axis: `Change in the log-odds of ${outcome}`,
      })),
      numbers([["Input"], ["Weight", "number"]], rows.map((row) => [row.label, `${row.value >= 0 ? "+" : "−"}${Math.abs(row.value).toFixed(2)}`])),
    ];
  }

  function renderHow() {
    const models = demo.models;
    const d = demo.discover;
    const inUse = (models.discover.in_use ? 1 : 0) + (/^in use/.test(models.rediscover.message) ? 1 : 0);
    const trained = models.trained;
    const profileRows = (rows) => rows.map(([name, weight]) => ({ label: name, value: weight }));
    $("#page-how-it-works").replaceChildren(
      hero("How it works", "A recommender that learns from you",
        "This website recommends music from your own Spotify history: songs new to you, songs you loved and forgot, and songs " +
        "like one you name. The app runs only on your computer, and every like, neutral and dislike you give it teaches it " +
        "something. This demo shows the same code at work on a made-up listener.",
        [[String(d.likes + d.dislikes + d.neutral), "ratings so far"], [`${inUse} of 2`, "models learned from your ratings in use"]]),
      h("h2", { text: "What it is for" }),
      h("ul", {},
        h("li", {}, h("b", { text: "Find new music you will like" }), ", without a streaming service deciding for you: songs you have never played, chosen for your taste, with dials for how mainstream, how familiar and which genres."),
        h("li", {}, h("b", { text: "Rediscover songs you loved" }), " and stopped playing."),
        h("li", {}, h("b", { text: "Explore your own listening" }), ": what you play, when, and how your taste moved (Your data)."),
        h("li", {}, h("b", { text: "Learn how a recommender works" }), ", end to end: data cleaning, features, models, honest tests.")),
      h("h2", { text: "How a list is made" }),
      steps([
        ["Your taste profile.", `Every play counts as a listen (1 finished, 0 skipped early, else the share heard), counting half as much every ${C.HALF_LIFE_DAYS} days. Your genres and artists get weights from those listens, and each like or dislike adds or takes away ${C.RATING_LISTENS} listens.`],
        ["Candidates.", "Songs new to you come from a pool built with Last.fm: the most played songs of your top artists, of the artists similar to them, and of your genres' top artists. Songs to rediscover are songs with at least 3 listens and none lately."],
        ["A score.", "Each candidate gets a score from parts between -1 and 1: how close its genres are to your taste, whether you play its artist, how strongly your artists point to it on Last.fm, plus your dials (Mainstream, More of these genres, Less Western)."],
        ["A learned model re-orders the best.", `The ${C.SHORTLIST} best by score are ordered by a model learned from your own ratings: your chance of liking each song. That chance is the big number on every card.`],
        ["Variety.", `The list is built one song at a time, trading a little score for genres it doesn't have yet, with at most ${C.MAX_PER_ARTIST} songs per artist.`],
        ["You rate, it learns.", "Your choices are sent together once you have chosen the whole list or press Next songs: likes and dislikes go into your profile and the next training of the models, and the next list is built with them. Neutral only keeps the song off your lists. (Not in this demo: here your choices stay in the tab.)"],
      ]),
      h("div", { class: "models" },
        h("div", { class: "panel" }, h("h3", { text: "Your profile: genres" }),
          h("p", { class: "small muted", text: "The weights of your ten top genres: the average genre row of your songs, each counted with its song's weight. Worked out by formula, no training." }),
          chart((box) => charts.bars(box, profileRows(models.profile_genres).map((row) => Object.assign(row, { label: genreLabel(row.label) })), { description: "Profile weights of the top genres", format: (value) => value.toFixed(3) })),
          numbers([["Genre"], ["Weight", "number"]], models.profile_genres.map(([genre, weight]) => [genreLabel(genre), weight.toFixed(4)]))),
        h("div", { class: "panel" }, h("h3", { text: "Your profile: artists" }),
          h("p", { class: "small muted", text: "From 1 for your top artist down to -1 for one you skip far more than you play. Ratings count too: an artist you never played but liked songs of has a weight." }),
          chart((box) => charts.bars(box, profileRows(models.profile_artists), { description: "Profile weights of the top artists", max: 1, format: (value) => value.toFixed(2) })),
          numbers([["Artist"], ["Weight", "number"]], models.profile_artists.map(([artist, weight]) => [artist, weight.toFixed(4)])))),
      h("h2", { text: "How the machine learns" }),
      h("p", {}, "Some parts are ", h("b", { text: "learned from data" }), " (trained models), some are ", h("b", { text: "worked out by formula" }),
        " from your plays, and a few are ", h("b", { text: "set by hand" }), " or chosen on held-out data. Telling them apart is the honest way to describe a system."),
      h("div", { class: "panel" }, table([["Part"], ["How it is set"], ["From what"]], [
        ["Taste profile (genre and artist weights)", "Worked out by formula", "Weighted averages of your listens; no training"],
        ["Score weights (familiarity 0.5, song share 0.5)", "Chosen on held-out data", "Picked on the valid window by rules fixed before looking"],
        ["Your chance of going back to a song", "Learned: logistic regression", "Trained on your history, month by month"],
        ["Your chance of liking a song new to you", "Learned: logistic regression", "Trained on your likes and dislikes"],
        ["Your chance of liking a song to rediscover", "Learned: logistic regression", "Trained on your likes and dislikes"],
        ["Songs you play together", "Learned without labels", "Co-occurrence in sessions, PPMI and SVD (song vectors)"],
        ["Mainstream, More of these genres, Less Western", "Set by you", "Your preferences; the tests leave them out"],
      ])),
      h("h3", { text: "1. Learning from your history: will you go back to a song?" }),
      h("p", { text: `A supervised problem: on the first day of every month from two years after your first play (a cutoff), take the songs you liked but hadn't played for ${C.DORMANT_DAYS} days; the label is 1 if you played one again within ${C.HORIZON_DAYS} days. For this listener that gave ${trained.cutoffs} cutoffs, ${number(trained.examples)} examples and ${number(trained.positives)} returns. A logistic regression learns a weight for each of its ${C.FEATURES} features (how much you played the song, how long ago, how often you finished or skipped it, ...). It may only learn from the past: a model judged at a moment learns from cutoffs whose ${C.HORIZON_DAYS} days were over by then.` }),
      h("div", { class: "panel" }, ...weightsPanel(models.ranker, "going back",
        "What it learned from the made-up history: per standard deviation more of each feature, the change in the log-odds of going back. " +
        "The weights follow from how the demo made up its listening, so they show how the model works, not how people listen: on the " +
        "author's real history, days unplayed weighed −0.41, plays lately +0.35 and plays in all +0.32.")),
      h("p", { text: "How well it did on the author's real data: checked month by month over 45 months, each month by a model trained only on the months before, it beat the hand-set formula in 32 of them, by 0.06 AUC on average, and on the test window its AUC was 0.82, against 0.73 for the formula. (AUC is the chance that a song gone back to is ranked above one that wasn't: 0.5 is guessing, 1 is perfect.)" }),
      h("h3", { text: "2. Learning from your ratings: will you like it?" }),
      h("p", { text: `Every like and dislike is a labelled example. Two more logistic regressions learn from them: one for songs new to you, one for songs to rediscover. Their inputs are the score's parts as they were when you saw the song, facts about it (Asian or not, Last.fm listeners, chart position), and its genres. A model is only used once it has ${C.MIN_EACH} likes and ${C.MIN_EACH} dislikes, and only if, cross-validated (each rating predicted by a model that didn't see it), it beats the score that listed the songs by at least ${C.MODEL_MARGIN} AUC: rules fixed before there were any ratings.` }),
      h("div", { class: "models" },
        h("div", { class: "panel" }, h("p", {}, h("b", { text: "Songs new to you" }), `: ${models.discover.message}.`),
          ...(models.discover.in_use ? weightsPanel(models.discover.weights, "a like", "Its 12 largest weights. A number: per standard deviation more. A genre: for the rated song with the most of it. The made-up listener's rule favoured some genres, and the model found them.") : [])),
        h("div", { class: "panel" }, h("p", {}, h("b", { text: "Songs to rediscover" }), `: ${models.rediscover.message}.`))),
      h("p", { text: "They were also checked forward in time: trained on earlier ratings, predicting the later ones, as they are really used. On the author's real ratings, songs new to you reached AUC 0.78 against 0.51 for the score (188 later ratings predicted), and songs to rediscover 0.67 against 0.57 (98)." }),
      h("h3", { text: "3. Learning without labels: songs you play together" }),
      h("p", { text: `For Songs like one you name, songs get vectors from your sessions: two songs are alike when you play them within ${C.WINDOW} songs of each other more often than chance would (PPMI), and a truncated SVD squeezes that into ${C.DIMENSIONS} numbers per song. Nobody labels anything: the structure comes from your listening alone.` }),
      h("h3", { text: "4. Testing it honestly" }),
      h("p", { text: "Your plays are split in time order: the oldest 80% to learn from, the next 10% (valid) to choose settings on, and the newest 10% (test) touched only to report. The test replays the past: it ranks the pool as of the test's first day and checks where the songs you went on to play ended up. On the author's real data, songs new to you reached AUC 0.90, with 4.3 of the top 10 later played and 53% of them in the top 100. Which new song will stick after its first listens reached only 0.46: nothing in a history predicts it, which is why the website asks for ratings." },),
      h("p", {}, "Every number, and the rule for every choice, is in ", h("a", { href: `${REPO}/blob/main/docs/results.md`, text: "docs/results.md" }), "."),
      h("h2", { text: "How it can get better" }),
      steps([
        ["More ratings, spread wide.", "The models learn only from songs you rate, and those are the ones the score already liked. Rating songs of new artists and other genres teaches them the most."],
        ["A little exploration.", "Showing a few songs outside the top on purpose (a bandit) would teach the models what the score misses, at the cost of some worse songs."],
        ["A fresh export.", "Spotify's next export brings your newest plays, and the songs you save to Liked Songs from the website, which count as likes in your profile."],
        ["Richer song features.", "Audio features (tempo, energy, mood) or lyrics could tell songs of one genre apart. They would mean sending your songs to another service, so they are left out for now."],
        ["Other people's taste.", "Big services learn from millions of listeners (collaborative filtering). With one listener, Last.fm's similar artists stand in for that."],
        ["Better genres.", "Last.fm tags names, not songs, so artists sharing a name can get the wrong genre."],
      ]),
      h("h2", { id: "about", tabindex: "-1", text: "How this demo works" }),
      h("div", { class: "panel" },
        h("ul", {},
          h("li", {}, h("b", { text: "Everything here is made up. " }),
            `A script, scripts/build_demo_site.py, makes up a listener: ${demo.history.length} songs by 14 artists played ${number(demo.data.steps.plays)} times from ${monthYear(demo.listener.first_play)} to ${monthYear(demo.listener.last_play)}, a pool of ${number(d.pool_songs)} songs by ${d.pool_artists} artists, and ${d.likes + d.dislikes + d.neutral} ratings by a rule the recommender isn't told. Any likeness to a real artist or song is a coincidence.`),
          h("li", {}, h("b", { text: "The real code ran on it. " }),
            "The script runs the project's pipeline and recommender on that listener, just as you would on your own export: the database builds, the taste profile, the model learned from the history, the lists, the model learned from the ratings. The page keeps what came out."),
          h("li", {}, h("b", { text: "Your browser ranks the songs. " }),
            "When you move a dial, assets/engine.js scores the pool again with the recommender's formulas. The script checks that it gives the same songs, in the same order, with the same reasons, as the Python code."),
          h("li", {}, h("b", { text: "Nothing you do leaves this tab. " }),
            "Your choices live in the page's memory and are gone when you close or reload it. Nothing is saved (no cookies, no storage), sent, or used to train anything: the models were trained before you came, and your clicks don't change them, the profile or the lists."),
          h("li", {}, h("b", { text: "Your browser enforces it. " }),
            "The page's Content Security Policy forbids it to connect to any server, send a form, or load anything from another site: no analytics, no fonts or scripts from elsewhere. Your browser's developer tools (Network) show only this site's own files. GitHub hosts the page, so, like any web host, it sees that your browser loaded it."),
          h("li", {}, h("b", { text: "Build your own. " }),
            "The real website runs on your own computer, on your own Spotify history, and learns from your ratings. The tutorial on ", h("a", { href: REPO, text: "GitHub" }), " walks you through it.")),
        h("p", { class: "small muted", text: "Typeface: Inter by Rasmus Andersson (SIL Open Font License). Icons: Material Icons by Google (Apache License 2.0)." })),
    );
  }

  // Settings: the look.
  function renderSettings() {
    const section = $("#page-settings");
    const tiles = Object.entries(WALLPAPERS).map(([name, [words]]) => {
      const swatch = h("span", { class: "swatch" });
      swatch.style.background = wallpaper(name, kind(), 0);
      return h("label", { class: "wallpaper" },
        h("input", { type: "radio", name: "wallpaper", value: name, checked: state.look.wallpaper === name, onchange: () => { state.look.wallpaper = name; applyLook(); } }),
        swatch, words);
    });
    const dim = h("input", { type: "range", id: "dim", min: "0", max: "0.8", step: "0.05", value: String(state.look.dim) });
    const dimOut = h("output", { for: "dim", text: state.look.dim.toFixed(2) });
    dim.addEventListener("input", () => { state.look.dim = Number(dim.value); dimOut.textContent = state.look.dim.toFixed(2); applyLook(); });
    const themes = [["system", "Like my system"], ["light", "Light"], ["dark", "Dark"]].map(([value, words]) => h("label", { class: "choice" },
      h("input", { type: "radio", name: "theme", value, checked: state.look.theme === value, onchange: () => { state.look.theme = value; applyLook(); renderSettings(); } }), words));
    section.replaceChildren(
      hero("Settings", "Make it yours", "The wallpaper and light or dark mode. In this demo they last until you close the tab; the app keeps them on your computer."),
      h("h2", { text: "Wallpaper" }),
      h("div", { class: "panel" },
        h("div", { class: "wallpapers", role: "radiogroup", "aria-label": "Wallpaper" }, tiles),
        h("div", { class: "field" }, h("div", { class: "field-top" }, h("label", { class: "label", for: "dim", text: "Dim the wallpaper" }), dimOut), dim),
        h("p", { class: "small muted", text: "In the app you can also use a picture of your own." })),
      h("h2", { text: "Light or dark" }),
      h("div", { class: "panel" }, h("div", { class: "themes", role: "radiogroup", "aria-label": "Light or dark" }, themes)),
      h("h2", { text: "Spotify" }),
      h("div", { class: "panel" }, h("p", { text: "In the app, Connect Spotify logs the website in to your account, so that it can save songs you like to your Liked Songs or a playlist, and play whole songs on the cards. It asks only for your library and playlists, and the login stays on your computer. The demo has no account to connect: its songs are made up." })),
      h("h2", { text: "Your data" }),
      h("div", { class: "panel" }, h("p", { text: "In the app, Reload history and pool reads the database again after you rebuild it from a newer export. The demo's data was built once, from the made-up listener." })),
    );
  }

  // Pages: one at a time, by the address's #.
  const renderers = { recommendations: () => {}, ratings: renderRatings, data: renderData, "how-it-works": renderHow, settings: renderSettings };
  let current = null;

  function route() {
    const hash = decodeURIComponent(window.location.hash.slice(1));
    let page = hash === "about" ? "how-it-works" : PAGES.includes(hash) ? hash : null;
    if (!page) page = current || "recommendations";
    const changed = page !== current;
    if (changed) {
      current = page;
      state.drawn = [];
      charts.untip();
      document.querySelectorAll(".page").forEach((section) => { section.hidden = section.dataset.page !== page; });
      document.querySelectorAll(".pages a").forEach((link) => {
        if (link.getAttribute("href") === `#${page}`) link.setAttribute("aria-current", "page");
        else link.removeAttribute("aria-current");
      });
      document.title = page === "recommendations" ? "Music Preference · Demo" : `${$(`.pages a[href="#${page}"]`).textContent} · Music Preference demo`;
      if (page === "recommendations") renderRecommendations();
      else renderers[page]();
    }
    if (hash === "about") {
      requestAnimationFrame(() => { const about = $("#about"); about.scrollIntoView(); about.focus({ preventScroll: true }); });
    } else if (hash === "main") {
      $("#main").focus();
    } else if (changed) {
      window.scrollTo(0, 0);
    }
  }

  let resizing = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizing);
    resizing = setTimeout(redraw, 150);
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", applyLook);
  window.addEventListener("hashchange", route);
  wireSidebar();
  build(false);
  renderSidebar();
  applyLook();
  route();
})();
