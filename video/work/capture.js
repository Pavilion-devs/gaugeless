// Scripted 1080p capture of the live Gaugeless dashboard.
// node capture.js <scene> <out.webm>   scene: shake | refuse | human
const puppeteer = require("puppeteer-core");
const URL = process.env.URL || "http://100.63.108.38/";
const [scene, out] = process.argv.slice(2);
const sleep = ms => new Promise(r => setTimeout(r, ms));

(async () => {
  const b = await puppeteer.launch({
    executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    headless: "new", args: ["--hide-scrollbars", "--window-size=1920,1080"],
    defaultViewport: { width: 1920, height: 1080, deviceScaleFactor: 1 },
  });
  const p = await b.newPage();
  await p.goto(URL, { waitUntil: "networkidle0" });
  await p.evaluate(() => { document.body.style.zoom = String(4 / 3); });
  // a visible cursor, since headless Chrome draws none
  await p.evaluate(() => {
    const c = document.createElement("div");
    c.innerHTML = `<svg width="22" height="22" viewBox="0 0 22 22"><path d="M3 2 L3 18 L7.5 13.8 L10.6 20.5 L13.4 19.2 L10.4 12.6 L16.5 12.6 Z" fill="#fff" stroke="#111" stroke-width="1.4" stroke-linejoin="round"/></svg>`;
    Object.assign(c.style, { position: "fixed", left: "0", top: "0", zIndex: 99999, pointerEvents: "none", transform: "translate(960px,800px)", transformOrigin: "0 0", transition: "none" });
    c.id = "fakecursor"; document.documentElement.appendChild(c);
  });
  let mx = 960, my = 800;
  const mv = async (x, y) => { await p.mouse.move(x, y); await p.evaluate((x, y) => { document.getElementById("fakecursor").style.transform = `translate(${x - 4}px,${y - 3}px) scale(1.33)`; }, x, y); };
  await mv(mx, my);
  async function click(sel, text) {
    const h = text
      ? await p.evaluateHandle((s, t) => [...document.querySelectorAll(s)].find(e => e.textContent.trim().startsWith(t)), sel, text)
      : await p.$(sel);
    const bb = await h.boundingBox();
    const x = bb.x + bb.width / 2, y = bb.y + bb.height / 2;
    const steps = 28;
    for (let i = 1; i <= steps; i++) {
      const t = i / steps, e = t < .5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
      await mv(mx + (x - mx) * e, my + (y - my) * e); await sleep(8);
    }
    mx = x; my = y; await sleep(180); await p.mouse.down(); await sleep(60); await p.mouse.up(); await sleep(450);
  }
  async function typeIn(sel, txt) { await click(sel); for (const ch of txt) { await p.keyboard.type(ch); await sleep(90); } }
  const site = { shake: "La Morge", refuse: "Arrow · unseeded", human: "La Morge" }[scene];

  const fs = require("fs"), path = require("path");
  const dir = out.replace(/\.\w+$/, ""); fs.rmSync(dir, { recursive: true, force: true }); fs.mkdirSync(dir, { recursive: true });
  const cdp = await p.createCDPSession(); const frames = [];
  cdp.on("Page.screencastFrame", async f => {
    const n = path.join(dir, `f${String(frames.length).padStart(5, "0")}.jpg`);
    fs.writeFileSync(n, Buffer.from(f.data, "base64")); frames.push([n, f.metadata.timestamp]);
    await cdp.send("Page.screencastFrameAck", { sessionId: f.sessionId }).catch(() => {});
  });
  await cdp.send("Page.startScreencast", { format: "jpeg", quality: 92, maxWidth: 1920, maxHeight: 1080, everyNthFrame: 1 });
  const rec = { stop: async () => {
    await cdp.send("Page.stopScreencast");
    const end = Date.now() / 1000; let txt = "";
    frames.forEach(([n, t], i) => { const d = (i + 1 < frames.length ? frames[i + 1][1] : end) - t; txt += `file '${n}'\nduration ${Math.max(d, 0.001).toFixed(4)}\n`; });
    txt += `file '${frames[frames.length - 1][0]}'\n`;
    fs.writeFileSync(dir + ".txt", txt);
    require("child_process").execFileSync("ffmpeg", ["-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", dir + ".txt", "-vf", "fps=30,scale=1920:1080:flags=lanczos,format=yuv420p", "-c:v", "libx264", "-crf", "16", "-preset", "medium", out]);
  } };
  await sleep(1200);
  await click("#sites button.site", site); await sleep(1500);
  if (scene === "shake") {
    await click("#deg button", "Shake"); await sleep(400);
    const lv = await p.$$("#lvl button"); const last = lv[lv.length - 1];
    const bb = await last.boundingBox(); await click(`#lvl button:last-child`);
  }
  if (scene === "human") {
    await click("#planner button", "Claude"); await sleep(300);
    await typeIn("#flood", "1.0");
  }
  await sleep(500);
  const t0 = Date.now();
  await click("#go");
  // wait for the run to finish: the decision banner leaves the live/idle state
  await p.waitForFunction(() => /\b(ok|bad|warn)\b/.test(document.getElementById("decision").className), { timeout: 240000 });
  console.log(scene, "run took", ((Date.now() - t0) / 1000).toFixed(1), "s");
  await sleep(2500);
  if (scene === "human") {
    await p.waitForSelector("#approve", { visible: true });
    await click("#approve"); await sleep(3500);
  }
  for (let i = 1; i <= 20; i++) { await mv(mx + 10 * i, my + 6 * i); await sleep(15); } await sleep(1500);
  await rec.stop();
  console.log(JSON.stringify(await p.evaluate(() => ({ t: document.getElementById("dt")?.textContent, d: document.getElementById("dd")?.innerText }))));
  await b.close();
})().catch(e => { console.error(e); process.exit(1); });
