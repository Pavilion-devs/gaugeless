// Hand-drawn shapes for the illustrated scenes: seeded rough.js, so every render draws the same strokes.
// Each call appends a <g> to an <svg> and returns it; SK.on(tl, g, at, dur) draws its strokes on.
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const INK = "#1e1e1e";
  let seed = 7;
  const base = o => Object.assign({ roughness: 1.1, bowing: 0.8, stroke: INK, strokeWidth: 2.6, seed: ++seed,
                                    disableMultiStroke: false, preserveVertices: true }, o || {});
  function add(svg, node, cls) {
    const s = typeof svg === "string" ? document.querySelector(svg) : svg;
    if (cls) node.setAttribute("class", cls);
    s.appendChild(node);
    node.querySelectorAll("path").forEach(p => {
      if (p.getAttribute("stroke") === "none") return;
      const L = p.getTotalLength(); p.__len = L + 2;
      p.style.strokeDasharray = `${L + 2} ${L + 2}`; p.style.strokeDashoffset = 0;
      p.setAttribute("stroke-linecap", "round");
    });
    return node;
  }
  const rc = svg => rough.svg(typeof svg === "string" ? document.querySelector(svg) : svg);
  window.SK = {
    INK,
    rect: (svg, x, y, w, h, o, cls) => add(svg, rc(svg).rectangle(x, y, w, h, base(o)), cls),
    line: (svg, x1, y1, x2, y2, o, cls) => add(svg, rc(svg).line(x1, y1, x2, y2, base(o)), cls),
    ellipse: (svg, cx, cy, w, h, o, cls) => add(svg, rc(svg).ellipse(cx, cy, w, h, base(o)), cls),
    curve: (svg, pts, o, cls) => add(svg, rc(svg).curve(pts, base(o)), cls),
    poly: (svg, pts, o, cls) => add(svg, rc(svg).linearPath(pts, base(o)), cls),
    path: (svg, d, o, cls) => add(svg, rc(svg).path(d, base(o)), cls),
    // straight arrow with a two-stroke head
    arrow(svg, x1, y1, x2, y2, o, cls) {
      const g = document.createElementNS(NS, "g");
      const a = Math.atan2(y2 - y1, x2 - x1), h = (o && o.head) || 18;
      const r = rc(svg);
      g.appendChild(r.line(x1, y1, x2, y2, base(o)));
      g.appendChild(r.linearPath([[x2 - h * Math.cos(a - 0.45), y2 - h * Math.sin(a - 0.45)], [x2, y2],
                                  [x2 - h * Math.cos(a + 0.45), y2 - h * Math.sin(a + 0.45)]], base(o)));
      return add(svg, g, cls);
    },
    // draw the strokes of g (or a selector / array of g) on, starting at `at`
    on(tl, g, at, dur, ease) {
      const gs = typeof g === "string" ? [...document.querySelectorAll(g)] : (Array.isArray(g) ? g : [g]);
      const paths = gs.flatMap(x => [...x.querySelectorAll("path")]).filter(p => p.__len);
      if (!paths.length) return;
      tl.fromTo(paths, { strokeDashoffset: (i, t) => t.__len }, { strokeDashoffset: 0, duration: dur || 0.6,
                ease: ease || "power2.inOut", stagger: Math.min(0.05, (dur || 0.6) / paths.length) }, at);
    },
    // handwriting-style reveal of an HTML text element, left to right
    write(tl, el, at, dur) {
      tl.fromTo(el, { clipPath: "inset(-30% 100% -30% -2%)", opacity: 1 },
                    { clipPath: "inset(-30% -2% -30% -2%)", duration: dur || 0.6, ease: "power1.inOut" }, at);
    },
  };
})();
