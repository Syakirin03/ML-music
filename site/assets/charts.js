/* The demo's charts, drawn as SVG with no library, so the page loads nothing from elsewhere.
 *
 * They follow the project's data-visualisation rules, as the website's Altair charts do: colours from the
 * validated palette (CSS variables in style.css, a step each for light and dark), 2px lines, bars at most 24px
 * thick with 4px rounded ends, 2px gaps between touching marks, hairline grids, a tooltip on every mark (also
 * from the keyboard), and the numbers of every chart in a table beside it (app.js). Text from the data goes in
 * with textContent, never as HTML.
 */
(function (root) {
  "use strict";

  const SVG = "http://www.w3.org/2000/svg";

  function colour(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function node(tag, attributes, parent) {
    const element = document.createElementNS(SVG, tag);
    Object.entries(attributes || {}).forEach(([name, value]) => element.setAttribute(name, value));
    if (parent) parent.append(element);
    return element;
  }

  function label(parent, x, y, content, attributes) {
    const element = node("text", Object.assign({ x, y, fill: colour("--chart-muted"), "font-size": 12 }, attributes), parent);
    element.textContent = content;
    return element;
  }

  function canvas(container, height, description) {
    container.replaceChildren();
    const width = Math.max(Math.floor(container.clientWidth), 280);
    const svg = node("svg", { viewBox: `0 0 ${width} ${height}`, width, height, role: "img", "aria-label": description }, container);
    return { svg, width };
  }

  // The tooltip: the value first, then what it is.
  function tip(value, words, x, y) {
    const box = document.getElementById("tooltip");
    const strong = document.createElement("b");
    strong.textContent = value;
    const rest = document.createElement("span");
    rest.textContent = words;
    box.replaceChildren(strong, rest);
    box.hidden = false;
    const width = box.offsetWidth;
    const height = box.offsetHeight;
    let left = x + 14;
    let top = y + 14;
    if (left + width > window.innerWidth - 8) left = x - width - 14;
    if (top + height > window.innerHeight - 8) top = y - height - 14;
    box.style.left = `${Math.max(8, left)}px`;
    box.style.top = `${Math.max(8, top)}px`;
  }

  function untip() {
    document.getElementById("tooltip").hidden = true;
  }

  // A mark with its tooltip on hover and on keyboard focus; it lifts with an outline while shown.
  function hoverable(mark, value, words) {
    mark.setAttribute("tabindex", "0");
    mark.setAttribute("aria-label", `${value}, ${words}`);
    const lift = (on) => {
      mark.setAttribute("stroke", on ? colour("--mp-text") : mark.dataset.stroke || "none");
      mark.setAttribute("stroke-width", on ? 1.5 : mark.dataset.strokeWidth || 0);
    };
    mark.addEventListener("pointermove", (event) => { lift(true); tip(value, words, event.clientX, event.clientY); });
    mark.addEventListener("pointerleave", () => { lift(false); untip(); });
    mark.addEventListener("focus", () => {
      const box = mark.getBoundingClientRect();
      lift(true);
      tip(value, words, box.right, box.top);
    });
    mark.addEventListener("blur", () => { lift(false); untip(); });
  }

  // Round axis steps: 1, 2, 2.5 or 5 times a power of ten.
  function axis(max, count) {
    const raw = (max || 1) / count;
    const power = 10 ** Math.floor(Math.log10(raw));
    const unit = raw / power;
    const step = (unit <= 1 ? 1 : unit <= 2 ? 2 : unit <= 2.5 ? 2.5 : unit <= 5 ? 5 : 10) * power;
    const top = Math.ceil((max || 1) / step) * step;
    const values = [];
    for (let value = 0; value <= top + step / 2; value += step) values.push(value);
    return { top, values };
  }

  function compact(value) {
    if (value >= 1000) return `${Number((value / 1000).toFixed(value >= 10000 ? 0 : 1))}K`;
    return String(Number(value.toFixed(1)));
  }

  // A bar with its data end rounded (4px) and its baseline end square.
  function bar(parent, x, y, width, height, end, fill) {
    const r = Math.min(4, width / 2, height / 2);
    let d;
    if (end === "top") d = `M${x},${y + height}V${y + r}Q${x},${y} ${x + r},${y}H${x + width - r}Q${x + width},${y} ${x + width},${y + r}V${y + height}Z`;
    else if (end === "right") d = `M${x},${y}H${x + width - r}Q${x + width},${y} ${x + width},${y + r}V${y + height - r}Q${x + width},${y + height} ${x + width - r},${y + height}H${x}Z`;
    else if (end === "left") d = `M${x + width},${y}H${x + r}Q${x},${y} ${x},${y + r}V${y + height - r}Q${x},${y + height} ${x + r},${y + height}H${x + width}Z`;
    else d = `M${x},${y}H${x + width}V${y + height}H${x}Z`;
    return node("path", { d, fill }, parent);
  }

  function legend(container, items, kind) {
    const box = document.createElement("div");
    box.className = "legend";
    items.forEach(([name, fill]) => {
      const item = document.createElement("span");
      const key = document.createElement("i");
      key.style.background = fill;
      if (kind === "line") { key.style.height = "2px"; key.style.borderRadius = "1px"; }
      item.append(key, document.createTextNode(name));
      box.append(item);
    });
    container.append(box);
  }

  // A line with a light area under it, one series; a crosshair finds the nearest point (pointer or arrow keys).
  function line(container, points, options) {
    const height = options.height || 260;
    const { svg, width } = canvas(container, height, options.description);
    const margin = { top: 12, right: 14, bottom: 26, left: 46 };
    const { top, values } = axis(Math.max(...points.map((point) => point.y)), 4);
    const step = (width - margin.left - margin.right) / Math.max(points.length - 1, 1);
    const x = (i) => margin.left + i * step;
    const y = (value) => margin.top + (1 - value / top) * (height - margin.top - margin.bottom);
    values.forEach((value) => {
      node("line", { x1: margin.left, x2: width - margin.right, y1: y(value), y2: y(value), stroke: colour(value === 0 ? "--chart-axis" : "--chart-grid"), "stroke-width": 1, "shape-rendering": "crispEdges" }, svg);
      label(svg, margin.left - 8, y(value) + 4, options.tick(value), { "text-anchor": "end" });
    });
    points.forEach((point, i) => {
      const text = options.xLabel(point, i);
      if (text) label(svg, x(i), height - 6, text, { "text-anchor": "middle" });
    });
    const series = colour("--series-1");
    const path = points.map((point, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(point.y).toFixed(1)}`).join("");
    node("path", { d: `${path}L${x(points.length - 1)},${y(0)}L${x(0)},${y(0)}Z`, fill: series, "fill-opacity": 0.1 }, svg);
    node("path", { d: path, fill: "none", stroke: series, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
    const rule = node("line", { y1: margin.top, y2: height - margin.bottom, stroke: colour("--chart-muted"), "stroke-width": 1, visibility: "hidden" }, svg);
    const dot = node("circle", { r: 4.5, fill: series, stroke: colour("--chart-surface"), "stroke-width": 2, visibility: "hidden" }, svg);
    const overlay = node("rect", { x: margin.left - step / 2, y: 0, width: width - margin.left - margin.right + step, height: height - margin.bottom, fill: "transparent" }, svg);
    let current = points.length - 1;
    const show = (i, at) => {
      current = i;
      rule.setAttribute("x1", x(i));
      rule.setAttribute("x2", x(i));
      dot.setAttribute("cx", x(i));
      dot.setAttribute("cy", y(points[i].y));
      rule.setAttribute("visibility", "visible");
      dot.setAttribute("visibility", "visible");
      const box = svg.getBoundingClientRect();
      const scale = box.width / width;
      const where = at || { x: box.left + x(i) * scale, y: box.top + y(points[i].y) * scale };
      tip(options.value(points[i]), options.words(points[i]), where.x, where.y);
    };
    const hide = () => {
      rule.setAttribute("visibility", "hidden");
      dot.setAttribute("visibility", "hidden");
      untip();
    };
    overlay.addEventListener("pointermove", (event) => {
      const box = svg.getBoundingClientRect();
      const along = (event.clientX - box.left) * (width / box.width);
      show(Math.min(points.length - 1, Math.max(0, Math.round((along - margin.left) / step))), { x: event.clientX, y: event.clientY });
    });
    overlay.addEventListener("pointerleave", hide);
    svg.setAttribute("tabindex", "0");
    svg.addEventListener("focus", () => show(current));
    svg.addEventListener("blur", hide);
    svg.addEventListener("keydown", (event) => {
      const moves = { ArrowLeft: -1, ArrowRight: 1, Home: -Infinity, End: Infinity };
      if (!(event.key in moves)) return;
      event.preventDefault();
      show(Math.min(points.length - 1, Math.max(0, current + moves[event.key])));
    });
  }

  function mix(from, to, share) {
    const parse = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
    const a = parse(from);
    const b = parse(to);
    return `rgb(${a.map((value, i) => Math.round(value + (b[i] - value) * share)).join(",")})`;
  }

  // Magnitude on the blue ramp, light (little) to dark (much) in light mode and the other way round in dark.
  function ramp(share) {
    const steps = [1, 2, 3, 4, 5, 6, 7].map((n) => colour(`--ramp-${n}`));
    const at = Math.max(0, Math.min(1, share)) * (steps.length - 1);
    const low = Math.floor(at);
    return low >= steps.length - 1 ? mix(steps[low], steps[low], 0) : mix(steps[low], steps[low + 1], at - low);
  }

  // A grid of cells (rows by columns) coloured by value, with a scale under it.
  function heatmap(container, cells, options) {
    const columns = options.columns.length;
    const rows = options.rows.length;
    const width = Math.max(Math.floor(container.clientWidth), 280);
    const margin = { top: 4, right: 6, bottom: 64, left: 40 };
    const cell = (width - margin.left - margin.right) / columns;
    const tall = Math.max(18, Math.min(32, cell));
    const height = margin.top + rows * tall + margin.bottom;
    const { svg } = canvas(container, height, options.description);
    const max = Math.max(...cells.map((item) => item.value));
    cells.forEach((item) => {
      const mark = node("rect", {
        x: margin.left + item.column * cell + 1, y: margin.top + item.row * tall + 1, width: Math.max(cell - 2, 1), height: tall - 2,
        rx: 3, fill: ramp(max ? item.value / max : 0),
      }, svg);
      hoverable(mark, options.value(item), options.words(item));
    });
    options.rows.forEach((name, i) => label(svg, margin.left - 8, margin.top + i * tall + tall / 2 + 4, name, { "text-anchor": "end" }));
    options.columns.forEach((name, i) => {
      if (i % options.every === 0) label(svg, margin.left + i * cell + cell / 2, margin.top + rows * tall + 16, name, { "text-anchor": "middle" });
    });
    label(svg, margin.left, margin.top + rows * tall + 32, options.columnsTitle, {});
    const defs = node("defs", {}, svg);
    const gradient = node("linearGradient", { id: `${container.id}-scale`, x1: 0, x2: 1, y1: 0, y2: 0 }, defs);
    for (let n = 0; n <= 10; n += 1) node("stop", { offset: n / 10, "stop-color": ramp(n / 10) }, gradient);
    const scaleWidth = Math.min(220, width - margin.left - 60);
    const scaleY = height - 18;
    node("rect", { x: margin.left, y: scaleY - 10, width: scaleWidth, height: 10, rx: 3, fill: `url(#${container.id}-scale)` }, svg);
    label(svg, margin.left - 6, scaleY, "0", { "text-anchor": "end" });
    label(svg, margin.left + scaleWidth + 6, scaleY, options.scaleLabel(max), {});
  }

  // Bars that each make up 100%, one per category, split by series in a fixed order (first at the bottom).
  function stacked(container, options) {
    const items = options.series.map((item) => [item.label, item.fill]);
    container.replaceChildren();
    legend(container, items, "bar");
    const holder = document.createElement("div");
    container.append(holder);
    const height = options.height || 280;
    const { svg, width } = canvas(holder, height, options.description);
    const margin = { top: 10, right: 10, bottom: 26, left: 46 };
    const band = (width - margin.left - margin.right) / options.categories.length;
    const thick = Math.min(24, band * 0.6);
    const y = (share) => margin.top + (1 - share) * (height - margin.top - margin.bottom);
    [0, 0.25, 0.5, 0.75, 1].forEach((value) => {
      node("line", { x1: margin.left, x2: width - margin.right, y1: y(value), y2: y(value), stroke: colour(value === 0 ? "--chart-axis" : "--chart-grid"), "stroke-width": 1, "shape-rendering": "crispEdges" }, svg);
      label(svg, margin.left - 8, y(value) + 4, `${Math.round(value * 100)}%`, { "text-anchor": "end" });
    });
    options.categories.forEach((category, i) => {
      const x = margin.left + i * band + (band - thick) / 2;
      const parts = options.series.map((item) => ({ item, share: options.share(category, item.key) })).filter((part) => part.share > 0);
      let below = 0;
      parts.forEach((part, n) => {
        const top = y(below + part.share);
        const bottom = y(below);
        const gap = n < parts.length - 1 ? 1 : 0;
        const tall = Math.max(bottom - top - (n > 0 ? 1 : 0) - gap, 0.5);
        const mark = bar(svg, x, top + gap, thick, tall, n === parts.length - 1 ? "top" : "none", part.item.fill);
        hoverable(mark, `${Math.round(part.share * 100)}%`, options.words(category, part.item));
        below += part.share;
      });
      label(svg, x + thick / 2, height - 6, String(category), { "text-anchor": "middle" });
    });
  }

  // Horizontal bars, one series: each name on its own line above its bar (it never has to be cut short), the
  // value at the tip, and an optional reference line.
  function bars(container, rows, options) {
    const row = 44;
    const height = rows.length * row + (options.rule ? 24 : 6);
    const { svg, width } = canvas(container, height, options.description);
    const left = 2;
    const right = 56;
    const top = options.rule ? 20 : 2;
    const max = options.max || Math.max(...rows.map((item) => item.value));
    const x = (value) => left + (value / max) * (width - left - right);
    const fill = colour("--series-1");
    rows.forEach((item, i) => {
      const y = top + i * row;
      label(svg, left, y + 13, item.label, { fill: colour("--mp-text"), "font-size": 13 });
      node("rect", { x: left, y: y + 20, width: width - left - right, height: 14, rx: 4, fill: colour("--chart-grid"), "fill-opacity": 0.45 }, svg);
      const mark = bar(svg, left, y + 20, Math.max(x(item.value) - left, 1), 14, "right", fill);
      hoverable(mark, options.format(item.value), item.words || item.label);
      label(svg, x(item.value) + 6, y + 31, options.format(item.value), { fill: colour("--mp-text"), "font-size": 12 });
    });
    if (options.rule) {
      const at = x(options.rule.value);
      node("line", { x1: at, x2: at, y1: 14, y2: height, stroke: colour("--chart-muted"), "stroke-width": 1 }, svg);
      label(svg, at, 10, options.rule.label, { "text-anchor": at > width / 2 ? "end" : "start" });
    }
  }

  // Bars to the right (up) or left (down) of zero, in the two colours of the diverging pair, for model weights.
  function diverging(container, rows, options) {
    container.replaceChildren();
    legend(container, [[options.up, colour("--diverge-up")], [options.down, colour("--diverge-down")]], "bar");
    const holder = document.createElement("div");
    container.append(holder);
    const row = 44;
    const height = rows.length * row + 22;
    const { svg, width } = canvas(holder, height, options.description);
    const reach = Math.max(...rows.map((item) => Math.abs(item.value))) || 1;
    const middle = width / 2;
    const half = width / 2 - 52;
    const x = (value) => middle + (value / reach) * half;
    rows.forEach((item, i) => {
      const y = 4 + i * row;
      label(svg, 2, y + 13, item.label, { fill: colour("--mp-text"), "font-size": 13 });
      const end = x(item.value);
      const up = item.value >= 0;
      const mark = bar(svg, Math.min(middle, end), y + 20, Math.max(Math.abs(end - middle), 1), 14, up ? "right" : "left", colour(up ? "--diverge-up" : "--diverge-down"));
      const shown = `${up ? "+" : "−"}${Math.abs(item.value).toFixed(2)}`;
      hoverable(mark, shown, `${item.label}: ${up ? options.up : options.down}`.toLowerCase().replace(/^./, (c) => c.toUpperCase()));
      label(svg, up ? end + 6 : end - 6, y + 31, shown, { fill: colour("--mp-text"), "text-anchor": up ? "start" : "end" });
    });
    node("line", { x1: middle, x2: middle, y1: 22, y2: height, stroke: colour("--chart-axis"), "stroke-width": 1 }, svg);
    label(svg, middle, height - 2, options.axis, { "text-anchor": "middle" });
  }

  root.DemoCharts = { line, heatmap, stacked, bars, diverging, compact, untip };
})(window);
