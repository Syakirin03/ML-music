// Checks that the demo page ranks songs as the recommender does: scripts/build_demo_site.py runs it with node,
// passing the page's data (site/data/demo.js) and the lists the recommender made for some settings.
//
//     node scripts/check_demo_engine.js site/data/demo.js references.json
//
// For each list it compares the songs, their order and their reasons, and the chance of a like (to 1e-9).
"use strict";

const fs = require("fs");
const path = require("path");

const [dataFile, referencesFile] = process.argv.slice(2);
globalThis.window = globalThis;
require(path.resolve(dataFile));
const engine = require(path.resolve(__dirname, "..", "site", "assets", "engine.js"));
const demo = globalThis.DEMO;
const cases = JSON.parse(fs.readFileSync(referencesFile, "utf8"));

const problems = [];
let lists = 0;

function compare(name, expected, rows) {
  lists += 1;
  const keys = rows.map((row) => row.song_key);
  if (JSON.stringify(keys) !== JSON.stringify(expected.keys)) {
    problems.push(`${name}: songs differ\n  recommender: ${JSON.stringify(expected.keys)}\n  page:        ${JSON.stringify(keys)}`);
    return;
  }
  rows.forEach((row, i) => {
    const why = row.why.join("; ");
    if (why !== expected.why[i].join("; ")) problems.push(`${name}, song ${i + 1}: reasons differ\n  recommender: ${expected.why[i].join("; ")}\n  page:        ${why}`);
    const chance = expected.liked_chance ? expected.liked_chance[i] : null;
    if (chance !== null && chance !== undefined && !(Math.abs(row.liked_chance - chance) < 1e-9)) {
      problems.push(`${name}, song ${i + 1}: chance of a like ${row.liked_chance} against ${chance}`);
    }
  });
}

cases.discover.forEach((item, i) => {
  const result = engine.discover(demo, item.settings, new Set(item.exclude), new Set(item.rest));
  compare(`songs new to you, case ${i + 1} ${JSON.stringify(item.settings)}`, item, result.rows);
});
cases.rediscover.forEach((item, i) => {
  const result = engine.rediscover(demo, item.settings, new Set(item.exclude), new Set(item.rest));
  compare(`songs to rediscover, case ${i + 1} ${JSON.stringify(item.settings)}`, item, result.rows);
});
cases.like.forEach((item, i) => {
  const result = engine.like(demo, { song: item.query, same_artist: item.same }, new Set());
  compare(`songs like "${item.query}", case ${i + 1}`, item, result.rows || []);
});

if (problems.length) {
  console.error(problems.join("\n"));
  process.exit(1);
}
console.log(`Checked the page's ranking against the recommender's: ${lists} lists, the same songs in the same order with the same reasons`);
