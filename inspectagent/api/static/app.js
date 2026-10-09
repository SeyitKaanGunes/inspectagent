/* InspectAgent single-page UI (vanilla JS, no build step).
   Talks to the FastAPI backend in inspectagent/api/app.py. All untrusted text is inserted with
   textContent; the only innerHTML use is for constant SVG icons and the escaped JSON viewer. */
(() => {
  "use strict";

  // ------------------------------------------------------------------------------------------ utils
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  function el(tag, props, ...children) {
    const node = document.createElement(tag);
    if (props) {
      for (const [k, v] of Object.entries(props)) {
        if (v === undefined || v === null || v === false) continue;
        if (k === "class") node.className = v;
        else if (k === "text") node.textContent = v;
        else if (k === "html") node.innerHTML = v; // constant markup only
        else if (k === "dataset") Object.assign(node.dataset, v);
        else if (k === "style") node.style.cssText = v;
        else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
        else if (v === true) node.setAttribute(k, "");
        else node.setAttribute(k, String(v));
      }
    }
    for (const c of children.flat(Infinity)) {
      if (c === null || c === undefined || c === false) continue;
      node.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return node;
  }

  const store = {
    get(k, d) { try { const v = localStorage.getItem("inspectagent." + k); return v === null ? d : v; } catch { return d; } },
    set(k, v) { try { localStorage.setItem("inspectagent." + k, v); } catch { /* storage unavailable */ } },
  };

  const reducedMotion = () => window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const isScalar = (v) => v === null || ["string", "number", "boolean"].includes(typeof v);

  function fmtNum(v, digits) {
    if (typeof v !== "number" || !isFinite(v)) return v === null || v === undefined ? "—" : String(v);
    if (Number.isInteger(v) && digits === undefined) return v.toLocaleString("en-US");
    const a = Math.abs(v);
    const d = digits !== undefined ? digits : a >= 1000 ? 0 : a >= 100 ? 1 : a >= 10 ? 2 : 3;
    return v.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: d });
  }
  const fmtMs = (v) => (typeof v === "number" ? (v >= 1000 ? fmtNum(v / 1000, 2) + " s" : fmtNum(v, v >= 100 ? 0 : 1) + " ms") : "—");
  const fmtPct = (v) => (typeof v === "number" ? fmtNum(v * 100, 1) + "%" : "—");
  const fmtSigned = (v, d = 2) => (typeof v === "number" ? (v > 0 ? "+" : v < 0 ? "−" : "±") + fmtNum(Math.abs(v), d) : "—");

  function fmtValue(v) {
    if (typeof v === "number") return fmtNum(v);
    if (typeof v === "boolean") return v ? "yes" : "no";
    if (Array.isArray(v)) return v.every(isScalar) ? v.map((x) => fmtValue(x)).join(", ") || "—" : `[${v.length}]`;
    if (v && typeof v === "object") return JSON.stringify(v);
    return v === null || v === undefined || v === "" ? "—" : String(v);
  }

  function relTime(iso) {
    const t = Date.parse(iso);
    if (!isFinite(t)) return "";
    const s = (t - Date.now()) / 1000;
    const rtf = new Intl.RelativeTimeFormat("en", { numeric: "auto" });
    const abs = Math.abs(s);
    if (abs < 60) return rtf.format(Math.round(s), "second");
    if (abs < 3600) return rtf.format(Math.round(s / 60), "minute");
    if (abs < 86400) return rtf.format(Math.round(s / 3600), "hour");
    return rtf.format(Math.round(s / 86400), "day");
  }

  class ApiError extends Error {
    constructor(status, message) { super(message); this.status = status; }
  }

  function detailText(body, status) {
    if (!body) return `HTTP ${status}`;
    const d = body.detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d)) return d.map((e) => `${(e.loc || []).slice(1).join(".") || "body"}: ${e.msg}`).join("; ");
    return JSON.stringify(d || body);
  }

  async function api(path, opts = {}) {
    let res;
    try {
      res = await fetch(path, opts);
    } catch (err) {
      throw new ApiError(0, "Network error — is the server reachable?");
    }
    let body = null;
    try { body = await res.json(); } catch { body = null; }
    if (!res.ok) throw new ApiError(res.status, detailText(body, res.status));
    return body;
  }

  // ----------------------------------------------------------------------------------------- toasts
  function toast(message, kind = "info", ms = 4200) {
    const t = el("div", { class: `toast toast--${kind}`, role: kind === "error" ? "alert" : "status" }, el("span", { text: message }));
    $("#toasts").append(t);
    setTimeout(() => { t.classList.add("is-leaving"); setTimeout(() => t.remove(), 300); }, ms);
  }

  // ----------------------------------------------------------------------------------------- icons
  const ICONS = {
    camera: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><path d="M4 8.5h3l1.6-2.5h6.8L17 8.5h3v10H4z"/><circle cx="12" cy="13.3" r="3.4"/></svg>',
    gauge: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M4.5 16a7.5 7.5 0 1 1 15 0"/><path d="M12 16l3.6-4.6"/><path d="M7 16h-.01M17 16h-.01M12 8.5v.01"/></svg>',
    scan: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M4 8V5.5A1.5 1.5 0 0 1 5.5 4H8M16 4h2.5A1.5 1.5 0 0 1 20 5.5V8M20 16v2.5a1.5 1.5 0 0 1-1.5 1.5H16M8 20H5.5A1.5 1.5 0 0 1 4 18.5V16"/><path d="M4 12h16"/><circle cx="14.5" cy="9" r="1.6"/></svg>',
    zoom: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><circle cx="10.5" cy="10.5" r="6"/><path d="M15 15l5 5M10.5 8v5M8 10.5h5"/></svg>',
    ruler: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><path d="M3.5 15.5l12-12 5 5-12 12z"/><path d="M7 12l2 2M10 9l1.5 1.5M13 6l2 2"/></svg>',
    person: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><circle cx="12" cy="8" r="3.5"/><path d="M5 20c.8-3.6 3.6-5.5 7-5.5s6.2 1.9 7 5.5"/></svg>',
    flag: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M5 21V4M5 4.5h11l-2 3.5 2 3.5H5"/></svg>',
    dot: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="12" cy="12" r="4"/></svg>',
    arrow: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12h15M14 7l5 5-5 5"/></svg>',
  };

  const TOOL_META = {
    capture_image: { stage: "perception", icon: "camera" },
    assess_quality: { stage: "perception", icon: "gauge" },
    detect_anomalies: { stage: "perception", icon: "scan" },
    inspect_region: { stage: "perception", icon: "zoom" },
    measure_defect: { stage: "perception", icon: "ruler" },
    request_human_review: { stage: "action", icon: "person" },
    finish: { stage: "action", icon: "flag" },
  };
  const toolMeta = (tool) => TOOL_META[tool] || { stage: "decision", icon: "dot" };

  const PRESET_HINTS = {
    clean: "Ideal capture: no photometric degradation, no pose jitter.",
    nominal: "Normal line variation: slight exposure, focus and pose drift.",
    dark: "Under-exposed scene (−1.6 to −2.4 EV) — expect an exposure correction.",
    bright: "Over-exposed scene (+1.2 to +1.8 EV) — expect a darker re-shoot.",
    defocus: "Out-of-focus optics (σ 2.5–4.5 px) — expect the agent to trigger autofocus.",
    glare: "Specular glare under ring light — expect a switch to diffuse light.",
    harsh: "One or two strong degradations combined at random.",
  };

  const ACTION_TEXT = {
    route_pass: "Released to the pass line",
    route_quarantine: "Quarantined — scrapping requires a person",
    route_scrap: "Scrapped",
    hold_for_review: "Held for human review",
  };

  // ------------------------------------------------------------------------------------------ state
  const state = {
    system: null,
    samples: [],
    catFilter: "all",
    selected: null, // {kind:"sample", sample} | {kind:"upload", file, url}
    run: null,
    runToken: 0,
    reviewStatus: "pending",
    policy: { auto_pass_max_prob: 0.15, auto_fail_min_prob: 0.9 },
  };

  // ------------------------------------------------------------------------------------------ theme
  const THEMES = ["auto", "light", "dark"];
  function applyTheme(t) {
    const root = document.documentElement;
    if (t === "auto") delete root.dataset.theme; else root.dataset.theme = t;
    const btn = $("#theme-toggle");
    btn.setAttribute("aria-label", `Color theme: ${t === "auto" ? "automatic" : t}`);
    btn.title = `Theme: ${t}`;
  }
  function initTheme() {
    let t = store.get("theme", "auto");
    if (!THEMES.includes(t)) t = "auto";
    applyTheme(t);
    $("#theme-toggle").addEventListener("click", () => {
      t = THEMES[(THEMES.indexOf(t) + 1) % THEMES.length];
      store.set("theme", t);
      applyTheme(t);
    });
  }

  // ------------------------------------------------------------------------------------------- tabs
  const TAB_IDS = ["inspect", "review", "results", "system"];
  function selectTab(id, focus = false) {
    if (!TAB_IDS.includes(id)) id = "inspect";
    for (const t of TAB_IDS) {
      const tab = $(`#tab-${t}`);
      const on = t === id;
      tab.setAttribute("aria-selected", String(on));
      tab.tabIndex = on ? 0 : -1;
      $(`#panel-${t}`).hidden = !on;
      if (on && focus) tab.focus();
    }
    if (location.hash !== `#${id}`) history.replaceState(null, "", `#${id}`);
    if (id === "review") loadReviews();
    if (id === "results") loadMetrics();
    if (id === "system" && state.system) renderSystem(state.system);
  }
  function initTabs() {
    const tabs = $$('[role="tab"]');
    tabs.forEach((tab, i) => {
      tab.addEventListener("click", () => selectTab(tab.id.replace("tab-", "")));
      tab.addEventListener("keydown", (e) => {
        let j = null;
        if (e.key === "ArrowRight") j = (i + 1) % tabs.length;
        else if (e.key === "ArrowLeft") j = (i - 1 + tabs.length) % tabs.length;
        else if (e.key === "Home") j = 0;
        else if (e.key === "End") j = tabs.length - 1;
        if (j !== null) { e.preventDefault(); selectTab(tabs[j].id.replace("tab-", ""), true); }
      });
    });
    window.addEventListener("hashchange", () => selectTab(location.hash.slice(1)));
    selectTab(location.hash.slice(1) || "inspect");
  }

  // --------------------------------------------------------------------------------------- lightbox
  function openLightbox(src, caption) {
    const dlg = $("#lightbox");
    $("#lightbox-img").src = src;
    $("#lightbox-img").alt = caption || "Evidence image";
    $("#lightbox-caption").textContent = caption || "";
    if (typeof dlg.showModal === "function") dlg.showModal(); else window.open(src, "_blank", "noopener");
  }
  function initLightbox() {
    const dlg = $("#lightbox");
    dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });
  }

  function evidenceButton(src, caption, alt) {
    return el("button", { type: "button", class: "evidence frame", "aria-label": `Enlarge: ${caption}`, onclick: () => openLightbox(src, caption) },
      el("img", { src, alt: alt || caption, loading: "lazy", decoding: "async" }),
      el("span", { class: "evidence-cap", text: caption }));
  }

  // ----------------------------------------------------------------------------------------- system
  async function loadSystem() {
    try {
      const sys = await api("/api/system");
      state.system = sys;
      if (sys.settings && sys.settings.policy) state.policy = sys.settings.policy;
      renderHeader(sys);
      renderControls(sys);
      if (!$("#panel-system").hidden) renderSystem(sys);
      $("#footer-version").textContent = "v" + sys.version;
    } catch (err) {
      toast(`System info unavailable: ${err.message}`, "error");
      renderControls(null);
    }
  }

  function renderHeader(sys) {
    const cv = sys.cv || {};
    const set = (id, v) => { $(`#${id} .sys-chip-v`).textContent = v; };
    set("chip-cv", `${cv.opencv_version || "?"}${cv.build_backend ? " · " + cv.build_backend : ""}`);
    const arch = (sys.deployment && sys.deployment.arch) || cv.arch || "?";
    set("chip-arch", arch + (sys.deployment && sys.deployment.graviton ? " · Graviton" : ""));
    $("#chip-arch").classList.toggle("is-graviton", !!(sys.deployment && sys.deployment.graviton));
    set("chip-runtime", sys.deployment && sys.deployment.lambda ? "AWS Lambda" : cv.runtime || "local");
  }

  function renderControls(sys) {
    const presetSel = $("#preset");
    const presets = (sys && sys.presets) || Object.keys(PRESET_HINTS);
    const current = store.get("preset", "nominal");
    presetSel.replaceChildren(...presets.map((p) => el("option", { value: p, text: p, selected: p === current })));
    if (!presets.includes(current)) presetSel.value = presets.includes("nominal") ? "nominal" : presets[0];
    updatePresetHint();

    const planners = (sys && sys.planners) || [{ name: "rule", available: true }];
    const bed = planners.find((p) => p.name === "bedrock");
    const bedOpt = $('#planner option[value="bedrock"]');
    bedOpt.disabled = !(bed && bed.available);
    if (bed && bed.available && bed.model_id) bedOpt.title = bed.model_id;
    const savedPlanner = store.get("planner", "rule");
    $("#planner").value = savedPlanner === "bedrock" && !bedOpt.disabled ? "bedrock" : "rule";
    updatePlannerHint();

    const cats = (sys && sys.models && sys.models.available_categories) || [];
    const upSel = $("#upload-category");
    upSel.replaceChildren(...(cats.length ? cats.map((c) => el("option", { value: c, text: c })) : [el("option", { value: "", text: "no model bundles installed" })]));
    upSel.disabled = !cats.length;
  }

  function updatePresetHint() {
    const p = $("#preset").value;
    let hint = PRESET_HINTS[p] || "";
    const s = state.selected && state.selected.kind === "sample" ? state.selected.sample : null;
    if (s && s.suggested_preset === p) hint = `Suggested for this part. ${hint}`;
    $("#preset-hint").textContent = hint;
  }
  function updatePlannerHint() {
    const v = $("#planner").value;
    const bedOpt = $('#planner option[value="bedrock"]');
    const bed = state.system && (state.system.planners || []).find((p) => p.name === "bedrock");
    $("#planner-hint").textContent = v === "bedrock"
      ? `Amazon Bedrock Converse tool-use planner${bed && bed.model_id ? ` (${bed.model_id})` : ""}; falls back to the rule planner on errors.`
      : bedOpt.disabled
        ? "Deterministic active-perception policy. Bedrock is not configured on this deployment."
        : "Deterministic active-perception policy — reproducible for a given seed.";
  }

  // ---------------------------------------------------------------------------------------- samples
  async function loadSamples() {
    try {
      const res = await api("/api/samples");
      state.samples = res.samples || [];
      $("#sample-count").textContent = `${state.samples.length} parts`;
      renderCatFilter();
      renderGallery();
    } catch (err) {
      $("#gallery").replaceChildren(el("p", { class: "empty-note", text: `Could not load samples: ${err.message}` }));
    }
  }

  function renderCatFilter() {
    const cats = ["all", ...new Set(state.samples.map((s) => s.category))];
    $("#cat-filter").replaceChildren(...cats.map((c) => el("button", {
      type: "button", class: "cat-chip", "aria-pressed": String(state.catFilter === c), text: c,
      onclick: () => { state.catFilter = c; renderCatFilter(); renderGallery(); },
    })));
  }

  function labelBadge(sample) {
    if (sample.defective === true) return el("span", { class: "badge badge--defective", text: "defective" });
    if (sample.defective === false) return el("span", { class: "badge badge--normal", text: "normal" });
    return el("span", { class: "badge badge--unknown", text: "unlabelled" });
  }

  function renderGallery() {
    const items = state.samples.filter((s) => state.catFilter === "all" || s.category === state.catFilter);
    if (!items.length) {
      $("#gallery").replaceChildren(el("p", { class: "empty-note", text: state.samples.length ? "No parts in this category." : "No curated samples found (samples/samples.json)." }));
      return;
    }
    const selId = state.selected && state.selected.kind === "sample" ? state.selected.sample.id : null;
    $("#gallery").replaceChildren(...items.map((s) => {
      const desc = `${s.id}, ${s.category}, ${s.defective === true ? "defective" : s.defective === false ? "normal" : "unlabelled"}${s.model_available ? "" : ", model not installed"}`;
      return el("button", {
        type: "button", class: "sample", "aria-pressed": String(s.id === selId), "aria-label": desc, title: s.description || s.id,
        onclick: () => selectSample(s),
      },
      el("span", { class: "sample-thumb" },
        el("img", { class: "sample-img", src: s.thumb_url, alt: "", loading: "lazy", decoding: "async" }),
        labelBadge(s),
        s.model_available ? null : el("span", { class: "sample-nomodel", text: "no model" })),
      el("span", { class: "sample-meta" }, el("span", { class: "sample-cat", text: s.category }), el("span", { class: "sample-id", text: s.id.split("_").pop() })));
    }));
  }

  function selectSample(s) {
    if (state.selected && state.selected.kind === "upload" && state.selected.url) URL.revokeObjectURL(state.selected.url);
    state.selected = { kind: "sample", sample: s };
    $("#upload").classList.remove("is-active");
    $("#upload-file").value = "";
    if (s.suggested_preset && $(`#preset option[value="${CSS.escape(s.suggested_preset)}"]`)) $("#preset").value = s.suggested_preset;
    updatePresetHint();
    renderGallery();
    renderRunPart();
  }

  function selectUpload(file) {
    if (!file) return;
    if (!["image/jpeg", "image/png"].includes(file.type)) { toast("Only JPEG or PNG images are accepted.", "error"); $("#upload-file").value = ""; return; }
    const limit = (state.system && state.system.limits && state.system.limits.max_upload_bytes) || 8 * 1024 * 1024;
    if (file.size > limit) { toast(`Image is ${fmtNum(file.size / 1048576, 1)} MB; the limit is ${fmtNum(limit / 1048576, 0)} MB.`, "error"); $("#upload-file").value = ""; return; }
    if (state.selected && state.selected.kind === "upload" && state.selected.url) URL.revokeObjectURL(state.selected.url);
    state.selected = { kind: "upload", file, url: URL.createObjectURL(file) };
    $("#upload").classList.add("is-active");
    renderGallery();
    renderRunPart();
    updatePresetHint();
  }

  function renderRunPart() {
    const sel = state.selected;
    const thumb = $("#run-part-thumb");
    if (!sel) {
      thumb.replaceChildren();
      $("#run-title").textContent = "No part selected";
      $("#run-sub").textContent = "Pick a sample part or upload an image.";
      return;
    }
    if (sel.kind === "sample") {
      const s = sel.sample;
      thumb.replaceChildren(el("img", { src: s.thumb_url, alt: "" }));
      $("#run-title").textContent = s.id;
      const sub = $("#run-sub");
      sub.replaceChildren(el("span", { class: "mono", text: s.category }), " · ", labelBadge(s),
        s.defect_type ? ` · ${s.defect_type}` : "", s.model_available ? "" : " · model bundle not installed");
    } else {
      thumb.replaceChildren(el("img", { src: sel.url, alt: "" }));
      $("#run-title").textContent = sel.file.name;
      $("#run-sub").textContent = `Uploaded image · ${fmtNum(sel.file.size / 1024, 0)} KB · category ${$("#upload-category").value || "—"}`;
    }
  }

  // ---------------------------------------------------------------------------------------- inspect
  let runTimer = null;
  function setRunStatus(text, mode) {
    const st = $("#run-status");
    st.className = "run-status" + (mode ? ` is-${mode}` : "");
    st.textContent = text;
  }

  async function runInspect() {
    const sel = state.selected;
    if (!sel) { toast("Select a sample part or upload an image first.", "warn"); return; }
    const preset = $("#preset").value;
    const planner = $("#planner").value;
    let seed = parseInt($("#seed").value, 10);
    if (!Number.isFinite(seed) || seed < 0) seed = 0;
    store.set("preset", preset);
    store.set("planner", planner);

    const btn = $("#inspect-btn");
    btn.disabled = true;
    btn.classList.add("is-busy");
    $(".btn-label", btn).textContent = "Running";
    $("#run").setAttribute("aria-busy", "true");
    const token = ++state.runToken;
    resetRunView(true);
    if (window.matchMedia("(max-width: 980px)").matches) $("#run").scrollIntoView({ block: "start", behavior: reducedMotion() ? "auto" : "smooth" });
    const t0 = performance.now();
    setRunStatus("agent running · 0.0 s", "running");
    clearInterval(runTimer);
    runTimer = setInterval(() => setRunStatus(`agent running · ${fmtNum((performance.now() - t0) / 1000, 1)} s`, "running"), 100);

    try {
      let res;
      if (sel.kind === "upload") {
        const fd = new FormData();
        fd.append("file", sel.file, sel.file.name);
        fd.append("category", $("#upload-category").value);
        fd.append("preset", preset);
        fd.append("planner", planner);
        fd.append("seed", String(seed));
        res = await api("/api/inspect", { method: "POST", body: fd });
      } else {
        res = await api("/api/inspect", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sample_id: sel.sample.id, preset, planner, seed }),
        });
      }
      clearInterval(runTimer);
      if (token !== state.runToken) return;
      const secs = (performance.now() - t0) / 1000;
      setRunStatus(`completed in ${fmtNum(secs, 2)} s · ${res.trace.steps.length} steps`, null);
      await renderRun(res, { animate: true, token });
      refreshReviewBadge();
    } catch (err) {
      clearInterval(runTimer);
      if (token !== state.runToken) return;
      resetRunView(false);
      setRunStatus(`error ${err.status || ""} · ${err.message}`, "error");
      toast(err.message, "error", 6500);
    } finally {
      if (token === state.runToken) {
        btn.disabled = false;
        btn.classList.remove("is-busy");
        $(".btn-label", btn).textContent = "Inspect";
        $("#run").setAttribute("aria-busy", "false");
      }
    }
  }

  function resetRunView(loading) {
    $("#timeline").replaceChildren();
    $("#timeline-empty").hidden = loading;
    $("#raw").hidden = true;
    $("#totals").hidden = true;
    $("#run-notes").hidden = true;
    resetPipeline();
    renderIdleVerdict(loading ? "Inspecting…" : "Awaiting inspection");
    if (loading) {
      for (let i = 0; i < 3; i++) {
        $("#timeline").append(el("li", { class: "step step--skeleton is-in", "aria-hidden": "true" },
          el("div", { class: "step-rail" }, el("span", { class: "step-icon" })), el("div", { class: "step-card" })));
      }
    }
  }

  function renderIdleVerdict(text) {
    const v = $("#verdict");
    v.className = "verdict verdict--idle";
    v.replaceChildren(
      el("p", { class: "verdict-kicker", text: "Verdict" }),
      el("p", { class: "verdict-label", text }),
      el("p", { class: "verdict-note", text: "The human-control policy decides what the agent may route on its own." }));
  }

  // --- pipeline strip
  function resetPipeline() {
    for (const st of $$(".pipe-stage")) {
      st.classList.remove("is-active");
      delete st.dataset.tone;
    }
    const defaults = { perception: "captures · quality · detection", decision: "planner + human-control policy", action: "routing on the line" };
    for (const [k, txt] of Object.entries(defaults)) {
      const st = $(`.pipe-stage[data-stage="${k}"]`);
      $('[data-field="count"]', st).textContent = "—";
      $('[data-field="latest"]', st).textContent = txt;
    }
  }

  function activateStage(stage) {
    for (const st of $$(".pipe-stage")) {
      const on = st.dataset.stage === stage;
      if (on) { st.classList.remove("is-active"); void st.offsetWidth; }
      st.classList.toggle("is-active", on);
    }
  }

  function perceptionSummary(step) {
    const tool = step.decision.tool;
    const o = step.observation || {};
    const a = step.decision.args || {};
    if (o.error) return `${tool}: error`;
    switch (tool) {
      case "capture_image": return `${o.capture_id || "capture"} · EV ${fmtSigned(a.exposure_ev ?? 0)} · ${a.light || "ring"}${a.autofocus ? " · AF" : ""}`;
      case "assess_quality": return `${o.capture_id || "quality"} · ${o.ok ? "quality ok" : "issues: " + (o.issues || []).join(", ")}`;
      case "detect_anomalies": return `${o.detector || a.detector || "detector"} · p=${fmtNum(o.prob, 3)}${o.reliable === false ? " · unreliable" : ""}`;
      case "inspect_region": return `zoom ${fmtNum(a.zoom ?? 3, 1)}× on ${a.region_id || "region"} · ${o.confirmed ? "confirmed" : "not confirmed"}`;
      case "measure_defect": return `${o.type_guess || "defect"} · ${fmtNum(o.area_mm2, 2)} mm²`;
      default: return tool;
    }
  }

  function updatePipeline(steps, upto, verdict) {
    const done = steps.slice(0, upto + 1);
    const cur = done[done.length - 1];
    const tools = done.map((s) => s.decision.tool);
    const count = (t) => tools.filter((x) => x === t).length;
    const per = $('.pipe-stage[data-stage="perception"]');
    const dec = $('.pipe-stage[data-stage="decision"]');
    const act = $('.pipe-stage[data-stage="action"]');
    const nCap = count("capture_image") + count("inspect_region");
    $('[data-field="count"]', per).textContent = `${nCap} cap · ${count("assess_quality")} qc · ${count("detect_anomalies")} det`;
    const lastPer = [...done].reverse().find((s) => toolMeta(s.decision.tool).stage === "perception");
    if (lastPer) $('[data-field="latest"]', per).textContent = perceptionSummary(lastPer);
    const overrides = done.filter((s) => s.overridden_by_policy).length;
    $('[data-field="count"]', dec).textContent = `${done.length} step${done.length === 1 ? "" : "s"}${overrides ? ` · ${overrides} override` : ""}`;
    $('[data-field="latest"]', dec).textContent = cur ? (cur.decision.rationale || cur.decision.tool) : "";
    const stage = cur ? toolMeta(cur.decision.tool).stage : "decision";
    if (verdict) {
      const tone = verdict.label === "PASS" ? "pass" : verdict.label === "FAIL" ? "fail" : "warn";
      act.dataset.tone = tone;
      $('[data-field="count"]', act).textContent = verdict.label === "NEEDS_REVIEW" ? "NEEDS REVIEW" : verdict.label;
      $('[data-field="latest"]', act).textContent = `${verdict.action} — ${ACTION_TEXT[verdict.action] || ""}`;
      activateStage("action");
    } else {
      activateStage(stage === "action" ? "action" : stage === "perception" ? "perception" : "decision");
      if (stage === "action") $('[data-field="latest"]', act).textContent = `${cur.decision.tool} proposed`;
    }
  }

  // --- rationale highlighting (DOM-built; no innerHTML)
  const TOKEN_RE = /(\b(?:cap|q|det|reg|art|rev|zoom|meas|deep|m)_\d[\w]*)|([-+−]?\b\d+(?:\.\d+)?(?:e[-+]?\d+)?(?:\s?(?:%|ms|px|°|×))?)|(->|<-|>=|<=|[<>≤≥→←])/g;
  function highlight(value) {
    const frag = document.createDocumentFragment();
    if (value === null || value === undefined || value === "") return frag;
    const text = String(value);
    let last = 0;
    for (const m of text.matchAll(TOKEN_RE)) {
      if (m.index > last) frag.append(text.slice(last, m.index));
      const cls = m[1] ? "rid" : m[2] ? "n" : "op";
      frag.append(el("span", { class: cls, text: m[0] }));
      last = m.index + m[0].length;
    }
    if (last < text.length) frag.append(text.slice(last));
    return frag;
  }

  // --- args + observation formatting
  const HIDDEN_ARGS = new Set(["rationale", "reason"]);
  function argChips(tool, args) {
    const chips = [];
    for (const [k, v] of Object.entries(args || {})) {
      if (HIDDEN_ARGS.has(k)) continue;
      let val;
      if (k === "exposure_ev") val = fmtSigned(v);
      else if (typeof v === "boolean") val = v ? "on" : "off";
      else if (Array.isArray(v)) val = v.length ? v.join(", ") : "—";
      else if (v && typeof v === "object") val = JSON.stringify(v);
      else val = typeof v === "number" ? fmtNum(v) : String(v);
      const key = { exposure_ev: "ev", autofocus: "af", proposed_label: "proposed", region_ids: "regions" }[k] || k;
      chips.push(el("span", { class: "arg" }, el("b", { text: key }), val));
    }
    return chips;
  }

  const OBS_KEYS = {
    capture_image: ["capture_id", "size", "mean_luma", "sim_time_ms"],
    assess_quality: ["ok", "issues", "sharpness", "sharpness_ratio", "mean_luma", "noise_sigma", "glare_frac", "clipped_frac", "suggested_exposure_delta"],
    detect_anomalies: ["detector", "prob", "fused_prob", "score", "reliable", "regions", "registration"],
    inspect_region: ["status", "confirmed", "prob", "fused_prob", "region_peak", "zoom_capture_id", "capture_id", "quality"],
    measure_defect: ["type_guess", "area_mm2", "length_mm", "width_mm", "elongation", "contrast"],
    request_human_review: ["review_id", "proposed_label", "confidence"],
    finish: ["label", "confidence", "prob", "region_ids"],
  };
  const OBS_LABEL = {
    capture_id: "capture", mean_luma: "mean luma", sim_time_ms: "sim time", sharpness_ratio: "sharpness ratio",
    noise_sigma: "noise σ", glare_frac: "glare", clipped_frac: "clipped", suggested_exposure_delta: "suggested ΔEV",
    region_peak: "region peak", zoom_capture_id: "zoom capture", type_guess: "type", area_mm2: "area", length_mm: "length",
    width_mm: "width", review_id: "review", proposed_label: "proposed", region_ids: "regions", prob: "P(anomalous)",
    fused_prob: "fused P",
  };

  function obsValue(key, v) {
    if (v === undefined) return null;
    if (key === "size" && Array.isArray(v) && v.length === 2) return { text: `${v[0]}×${v[1]} px` };
    if (key === "sim_time_ms") return { text: fmtMs(v) };
    if (key === "area_mm2") return { text: `${fmtNum(v, 2)} mm²` };
    if (key === "length_mm" || key === "width_mm") return { text: `${fmtNum(v, 2)} mm` };
    if (["glare_frac", "clipped_frac", "crushed_frac"].includes(key) && typeof v === "number") return { text: fmtNum(v, 4) };
    if (key === "suggested_exposure_delta") return { text: fmtSigned(v) + " EV" };
    if (key === "issues") return Array.isArray(v) && v.length ? { text: v.join(", "), cls: "is-warn" } : { text: "none", cls: "is-good" };
    if (key === "ok" || key === "reliable") return { text: v ? "yes" : "no", cls: v ? "is-good" : "is-bad" };
    if (key === "confirmed") return { text: v ? "yes" : "no", cls: v ? "is-bad" : "is-good" };
    if (key === "regions" && Array.isArray(v)) {
      if (!v.length) return { text: "0" };
      const peak = Math.max(...v.map((r) => (typeof r === "object" && r ? r.peak || 0 : 0)));
      return { text: `${v.length}${peak ? ` · peak ${fmtNum(peak, 2)}` : ""}`, cls: "is-warn" };
    }
    if (key === "registration" && v && typeof v === "object") return { text: `${v.method || "?"} · ${v.ok ? "ok" : "failed"}${typeof v.ncc === "number" ? ` · ncc ${fmtNum(v.ncc, 2)}` : ""}`, cls: v.ok ? "" : "is-bad" };
    if (key === "quality" && v && typeof v === "object") return { text: v.ok ? "ok" : (v.issues || []).join(", ") || "issues", cls: v.ok ? "is-good" : "is-warn" };
    if (key === "status") return { text: String(v), cls: v === "confirmed" ? "is-bad" : v === "refuted" ? "is-good" : "is-warn" };
    if ((key === "prob" || key === "fused_prob") && typeof v === "number") return { text: fmtNum(v, 3), cls: v >= state.policy.auto_fail_min_prob ? "is-bad" : v <= state.policy.auto_pass_max_prob ? "is-good" : "is-warn" };
    if (key === "label") return { text: String(v), cls: v === "PASS" ? "is-good" : v === "FAIL" ? "is-bad" : "is-warn" };
    if (key === "confidence" && typeof v === "number") return { text: fmtPct(v) };
    return { text: fmtValue(v) };
  }

  function observationList(tool, obs) {
    const keys = OBS_KEYS[tool] || Object.keys(obs).filter((k) => isScalar(obs[k])).slice(0, 8);
    const dl = el("dl", { class: "obs" });
    let n = 0;
    for (const k of keys) {
      if (!(k in obs)) continue;
      const val = obsValue(k, obs[k]);
      if (!val) continue;
      dl.append(el("dt", { text: OBS_LABEL[k] || k.replace(/_/g, " ") }), el("dd", { class: val.cls || null, text: val.text }));
      n++;
    }
    return n ? dl : null;
  }

  function artifactSrc(res, aid) {
    return (res.artifacts && res.artifacts[aid]) || (res.artifact_urls && res.artifact_urls[aid]) || null;
  }

  function stepCaption(step, aid) {
    const tool = step.decision.tool;
    const o = step.observation || {};
    const kind = { capture_image: "capture", assess_quality: "quality overlay", detect_anomalies: "anomaly heatmap", inspect_region: "zoom view", measure_defect: "measurement" }[tool] || "artifact";
    const cid = o.capture_id || o.zoom_capture_id;
    return `${aid} · ${kind}${cid ? " · " + cid : ""}`;
  }

  function renderStep(step, res) {
    const d = step.decision || {};
    const meta = toolMeta(d.tool);
    const obs = step.observation || {};
    const card = el("div", { class: "step-card" + (step.overridden_by_policy ? " is-override" : "") });
    card.append(el("div", { class: "step-top" },
      el("span", { class: "step-tool", text: d.tool }),
      el("span", { class: "step-args" }, argChips(d.tool, d.args)),
      step.overridden_by_policy ? el("span", { class: "badge badge--warn step-flag", text: "policy override" }) : null,
      !step.overridden_by_policy && d.planner && d.planner.includes("fallback") ? el("span", { class: "badge badge--accent step-flag", text: "fallback" }) : null));
    if (d.rationale) card.append(el("p", { class: "step-rationale" }, highlight(d.rationale)));
    const reason = d.args && d.args.reason;
    if (reason && reason !== d.rationale) card.append(el("p", { class: "step-rationale" }, el("span", { class: "op", text: "reason: " }), highlight(reason)));
    if (obs.error) card.append(el("div", { class: "step-error", text: `error: ${obs.error}` }));

    const arts = (step.artifacts || []).filter((aid) => artifactSrc(res, aid));
    const obsList = observationList(d.tool, obs);
    if (arts.length || obsList) {
      const body = el("div", { class: "step-body" + (arts.length ? "" : " step-body--noimg") });
      if (arts.length) {
        const figs = el("div", { style: arts.length > 1 ? "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px" : "" },
          arts.map((aid) => evidenceButton(artifactSrc(res, aid), stepCaption(step, aid), `${d.tool} evidence ${aid}`)));
        body.append(figs);
      }
      if (obsList) body.append(obsList);
      card.append(body);
    }
    card.append(el("div", { class: "step-foot" },
      el("span", null, "compute ", el("b", { text: fmtMs(step.latency_ms) })),
      el("span", null, "sim ", el("b", { text: fmtMs(step.sim_time_ms) })),
      d.evidence && d.evidence.length ? el("span", null, "evidence ", el("b", { text: d.evidence.join(", ") })) : null,
      el("span", null, "planner ", el("b", { text: d.planner || "—" }))));

    return el("li", { class: "step", dataset: { stage: meta.stage, tool: d.tool } },
      el("div", { class: "step-rail" },
        el("span", { class: "step-icon", html: ICONS[meta.icon] || ICONS.dot }),
        el("span", { class: "step-no", text: String(step.step ?? "").padStart(2, "0") })),
      card);
  }

  // A step "changed the plan because of visual evidence" when it cites evidence ids and is a
  // corrective capture, a zoom, a deep escalation or a human-review request.
  function evidenceDrivenCount(steps) {
    return steps.filter((s, i) => {
      const d = s.decision || {};
      if (!i || !(d.evidence && d.evidence.length)) return false;
      return d.tool === "capture_image" || d.tool === "inspect_region" || d.tool === "request_human_review" ||
        (d.tool === "detect_anomalies" && (d.args || {}).detector === "deep");
    }).length;
  }

  async function renderRun(res, { animate = true, token = state.runToken } = {}) {
    state.run = res;
    const trace = res.trace;
    const steps = trace.steps || [];
    $("#timeline").replaceChildren();
    $("#timeline-empty").hidden = true;
    resetPipeline();
    renderIdleVerdict("Reasoning…");
    const doAnimate = animate && !reducedMotion();
    for (let i = 0; i < steps.length; i++) {
      if (token !== state.runToken) return;
      const li = renderStep(steps[i], res);
      $("#timeline").append(li);
      updatePipeline(steps, i, null);
      if (doAnimate) {
        await sleep(30);
        li.classList.add("is-in");
        if (i > 0) li.scrollIntoView({ block: "nearest", behavior: "smooth" });
        await sleep(560);
      } else {
        li.classList.add("is-in");
      }
    }
    if (token !== state.runToken) return;
    updatePipeline(steps, steps.length - 1, trace.verdict);
    renderVerdict(trace, res);
    renderTotals(trace);
    renderNotes(res);
    renderRaw(trace);
  }

  // --- verdict
  function decisionScale(prob) {
    const lo = state.policy.auto_pass_max_prob ?? 0.15;
    const hi = state.policy.auto_fail_min_prob ?? 0.9;
    const p = typeof prob === "number" ? Math.min(1, Math.max(0, prob)) : null;
    return el("div", { class: "scale" },
      el("div", { class: "scale-head" }, el("span", { text: "P(anomalous) vs. policy" }), el("b", { text: p === null ? "—" : fmtNum(p, 3) })),
      el("div", { class: "scale-track", role: "img", "aria-label": `Fused probability ${p === null ? "unknown" : fmtNum(p, 3)}; auto-pass at or below ${lo}, auto-fail at or above ${hi}` },
        el("span", { class: "scale-zone scale-zone--pass", style: `width:${lo * 100}%` }),
        el("span", { class: "scale-zone scale-zone--fail", style: `width:${(1 - hi) * 100}%` }),
        el("span", { class: "scale-ticks" }),
        p === null ? null : el("span", { class: "scale-marker", style: `left:${p * 100}%` })),
      el("div", { class: "scale-legend" }, el("span", { text: `auto-pass ≤ ${lo}` }), el("span", { text: "human review" }), el("span", { text: `auto-fail ≥ ${hi}` })));
  }

  function findDefectBase(trace, res, regions) {
    // Pick the evidence image (heatmap preferred) whose capture holds most of the defect regions.
    const steps = trace.steps || [];
    const sizeOf = {};
    for (const s of steps) {
      const o = s.observation || {};
      if (o.capture_id && Array.isArray(o.size)) sizeOf[o.capture_id] = o.size;
      if (o.zoom_capture_id && Array.isArray(o.zoom_size)) sizeOf[o.zoom_capture_id] = o.zoom_size;
    }
    const byCap = {};
    for (const r of regions) (byCap[r.capture_id] = byCap[r.capture_id] || []).push(r);
    const caps = Object.keys(byCap).sort((a, b) => byCap[b].length - byCap[a].length);
    for (const cap of caps) {
      const size = sizeOf[cap];
      if (!size) continue;
      const pick = (tools) => {
        for (const s of [...steps].reverse()) {
          const o = s.observation || {};
          if (tools.includes(s.decision.tool) && o.capture_id === cap) {
            const aid = [...(s.artifacts || [])].reverse().find((a) => artifactSrc(res, a));
            if (aid) return aid;
          }
        }
        return null;
      };
      const aid = pick(["detect_anomalies"]) || pick(["capture_image"]);
      if (aid) return { aid, size, regions: byCap[cap], cap };
    }
    return null;
  }

  function renderVerdict(trace, res) {
    const v = trace.verdict;
    const box = $("#verdict");
    if (!v) { renderIdleVerdict("No verdict"); return; }
    const tone = v.label === "PASS" ? "pass" : v.label === "FAIL" ? "fail" : "review";
    box.className = `verdict verdict--${tone}`;
    void box.offsetWidth;
    box.classList.add("is-revealed");
    const children = [
      el("p", { class: "verdict-kicker", text: `Verdict · ${trace.planner || ""}` }),
      el("p", { class: "verdict-label", text: v.label === "NEEDS_REVIEW" ? "NEEDS REVIEW" : v.label }),
      el("div", { class: "verdict-action" }, el("span", { html: ICONS.arrow, style: "display:inline-grid" }),
        el("span", null, el("code", { text: v.action }), " ", ACTION_TEXT[v.action] || "")),
      decisionScale(v.prob_anomalous),
    ];
    const kv = el("dl", { class: "verdict-kv" },
      el("dt", { text: "Confidence" }), el("dd", { text: fmtPct(v.confidence) }),
      el("dt", { text: "Trace" }), el("dd", { text: trace.trace_id }));
    if (res.review) {
      kv.append(el("dt", { text: "Review" }), el("dd", null, el("button", { type: "button", class: "link-btn mono", text: res.review.review_id, onclick: () => selectTab("review") })));
    }
    children.push(kv);

    const regions = v.defect_regions || [];
    if (regions.length) {
      const base = findDefectBase(trace, res, regions);
      if (base) {
        const [w, h] = base.size;
        const src = artifactSrc(res, base.aid);
        const stage = el("button", { type: "button", class: "defect-stage", "aria-label": "Enlarge defect evidence", onclick: () => openLightbox(src, `${base.aid} · defect regions on ${base.cap}`) },
          el("img", { src, alt: `Defect regions on ${base.cap}` }),
          base.regions.map((r) => {
            const [x, y, bw, bh] = r.bbox_capture;
            return el("span", { class: "defect-box", style: `left:${(x / w) * 100}%;top:${(y / h) * 100}%;width:${Math.max(0.8, (bw / w) * 100)}%;height:${Math.max(0.8, (bh / h) * 100)}%` },
              el("span", { text: r.region_id }));
          }));
        children.push(el("figure", { class: "defect-fig" }, el("div", { class: "frame" }, stage),
          el("figcaption", { class: "defect-cap", text: regions.map((r) => `${r.region_id} ${r.source} peak ${fmtNum(r.peak, 2)}`).join(" · ") })));
      } else {
        children.push(el("p", { class: "defect-cap", text: regions.map((r) => `${r.region_id} @ ${r.bbox_scene.map((n) => Math.round(n)).join(",")}`).join(" · ") }));
      }
    }
    if (v.rationale) children.push(el("p", { class: "verdict-rationale" }, highlight(v.rationale)));
    const gt = trace.ground_truth && trace.ground_truth.label;
    if (gt === 0 || gt === 1) {
      let agree;
      if (v.label === "NEEDS_REVIEW") agree = el("span", { class: "badge badge--warn", text: "escalated" });
      else if ((v.label === "FAIL") === (gt === 1)) agree = el("span", { class: "badge badge--normal", text: "agrees" });
      else agree = el("span", { class: "badge badge--defective", text: gt === 1 ? "escape" : "false reject" });
      children.push(el("p", { class: "verdict-gt" }, "Ground truth:", el("strong", { text: gt === 1 ? "defective" : "normal" }), agree));
    }
    box.replaceChildren(...children);
  }

  function renderTotals(trace) {
    const t = trace.totals || {};
    const metric = (v, k, unit) => el("div", null, el("div", { class: "metric-v" }, v, unit ? el("small", { text: unit }) : null), el("div", { class: "metric-k", text: k }));
    const ms = (v) => (typeof v === "number" ? fmtNum(v, v >= 100 ? 0 : 1) : "—");
    const grid = el("div", { class: "totals-grid" },
      metric(fmtNum(t.n_steps ?? (trace.steps || []).length), "steps"),
      metric(fmtNum(t.n_captures ?? 0), "captures"),
      metric(fmtNum(t.n_deep ?? 0), "deep calls"),
      metric(ms(t.compute_ms), "compute", "ms"),
      metric(ms(t.sim_time_ms), "simulated", "ms"),
      metric(ms(t.cycle_time_ms), "cycle time", "ms"));
    const evidenceN = Array.isArray(t.evidence_driven_steps) ? t.evidence_driven_steps.length : evidenceDrivenCount(trace.steps || []);
    const sub = el("div", { class: "totals-sub" },
      el("div", null, el("strong", { text: String(evidenceN) }), ` evidence-driven action${evidenceN === 1 ? "" : "s"} (corrective capture, zoom, deep escalation or review)`));
    const c = trace.conditions;
    if (c && typeof c === "object") {
      const pick = [["preset", c.preset], ["ambient", c.ambient_ev !== undefined ? fmtSigned(c.ambient_ev) + " EV" : null], ["defocus σ", c.defocus_sigma], ["glare", c.glare_strength], ["noise", c.noise_sigma], ["rot", c.rot_deg !== undefined ? fmtNum(c.rot_deg, 2) + "°" : null]];
      sub.append(el("div", { style: "margin-top:8px" }, "Hidden simulator conditions (never shown to the agent):"),
        el("div", { class: "kvs" }, pick.filter(([, v]) => v !== null && v !== undefined).map(([k, v]) => el("span", null, `${k} ${typeof v === "number" ? fmtNum(v, 2) : v}`))));
    }
    $("#totals").replaceChildren(el("p", { class: "totals-title", text: "Run totals" }), grid, sub);
    $("#totals").hidden = false;
  }

  function renderNotes(res) {
    const notes = [...(res.notes || []), ...((res.trace && res.trace.errors) || [])];
    if (res.persisted === false) notes.unshift("Results were not persisted (storage unavailable).");
    const box = $("#run-notes");
    if (!notes.length) { box.hidden = true; return; }
    box.replaceChildren(el("strong", { text: "Notes" }), el("ul", null, notes.map((n) => el("li", { text: n }))));
    box.hidden = false;
  }

  // --- raw json viewer (escape first, then colour tokens of the escaped text)
  function renderRaw(trace) {
    const json = JSON.stringify(trace, null, 2);
    const esc = json.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    const html = esc.replace(/("(?:\\u[a-fA-F0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(?:true|false|null)\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g, (m) => {
      let cls = "j-n";
      if (m.startsWith('"')) cls = /:$/.test(m) ? "j-k" : "j-s";
      else if (/true|false|null/.test(m)) cls = "j-b";
      return `<span class="${cls}">${m}</span>`;
    });
    $("#raw-json").innerHTML = html;
    $("#raw-size").textContent = `${fmtNum(json.length / 1024, 1)} KB · ${trace.trace_id}`;
    $("#raw").hidden = false;
  }

  function initRawActions() {
    $("#raw-copy").addEventListener("click", async () => {
      if (!state.run) return;
      try { await navigator.clipboard.writeText(JSON.stringify(state.run.trace, null, 2)); toast("Trace JSON copied.", "success"); }
      catch { toast("Clipboard unavailable in this context.", "warn"); }
    });
    $("#raw-download").addEventListener("click", () => {
      if (!state.run) return;
      const blob = new Blob([JSON.stringify(state.run.trace, null, 2)], { type: "application/json" });
      const a = el("a", { href: URL.createObjectURL(blob), download: `${state.run.trace.trace_id}.json` });
      document.body.append(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    });
  }

  // ---------------------------------------------------------------------------------------- reviews
  async function refreshReviewBadge() {
    try {
      const res = await api("/api/reviews?status=pending&limit=500");
      const b = $("#review-badge");
      b.textContent = String(res.count);
      b.hidden = !res.count;
      b.setAttribute("aria-label", `${res.count} pending`);
    } catch { /* badge is best-effort */ }
  }

  async function loadReviews() {
    const list = $("#review-list");
    try {
      const res = await api(`/api/reviews?status=${encodeURIComponent(state.reviewStatus)}&limit=200`);
      if (!res.reviews.length) {
        list.replaceChildren(el("div", { class: "empty-state" },
          el("strong", { text: state.reviewStatus === "pending" ? "Nothing waiting for a human." : "No review items yet." }),
          "Run an inspection under harsh conditions — uncertain or failing parts land here."));
      } else {
        list.replaceChildren(...res.reviews.map(renderReview));
      }
      refreshReviewBadge();
    } catch (err) {
      list.replaceChildren(el("div", { class: "empty-state" }, el("strong", { text: "Could not load the review queue." }), err.message));
    }
  }

  function renderReview(r) {
    const kindBadge = r.kind === "scrap_approval"
      ? el("span", { class: "badge badge--defective", text: "scrap approval" })
      : el("span", { class: "badge badge--warn", text: "needs review" });
    const statusBadge = r.status === "pending" ? null : el("span", { class: "badge badge--unknown", text: r.status });
    const card = el("article", { class: "review", "aria-labelledby": `rv-${r.review_id}` });
    card.append(el("div", { class: "review-top" }, kindBadge, statusBadge, el("span", { class: "review-id", text: r.review_id }),
      el("span", { class: "review-time", title: r.created_at, text: relTime(r.created_at) })));
    card.append(el("h3", { class: "review-title", id: `rv-${r.review_id}`, text: `${r.part_id || "part"} · ${r.category || ""}` }));
    if (r.reason) card.append(el("p", { class: "review-reason" }, highlight(r.reason)));

    const strip = el("div", { class: "evidence-strip" },
      (r.evidence || []).map((e) => evidenceButton(e.url, `${e.artifact_id} · step ${e.step} · ${e.tool}`, `Evidence ${e.artifact_id}`)));
    if (!(r.evidence || []).length) strip.append(el("p", { class: "field-hint", text: "No image evidence stored." }));

    const side = el("div", { class: "review-side" });
    side.append(el("dl", { class: "review-kv" },
      el("dt", { text: "Agent verdict" }), el("dd", { text: r.verdict_label || "—" }),
      el("dt", { text: "Proposed label" }), el("dd", { text: r.proposed_label || "—" }),
      el("dt", { text: "Confidence" }), el("dd", { text: fmtPct(r.confidence) }),
      el("dt", { text: "P(anomalous)" }), el("dd", { text: fmtNum(r.prob_anomalous, 3) }),
      el("dt", { text: "Current routing" }), el("dd", { text: r.routing_action || "—" }),
      el("dt", { text: "Trace" }), el("dd", { text: r.trace_id })));

    if (r.status === "pending") {
      const noteId = `note-${r.review_id}`;
      const note = el("textarea", { id: noteId, maxlength: "2000", placeholder: "Optional note for the audit log" });
      side.append(el("label", { class: "field-label", for: noteId, text: "Note" }), note);
      const mk = (decision, label, cls) => el("button", { type: "button", class: `btn ${cls}`, text: label, onclick: (e) => decide(r, decision, note.value, card, e.currentTarget) });
      side.append(el("div", { class: "review-actions" },
        mk("approve_fail", "Approve FAIL · scrap", "btn-danger"),
        mk("override_pass", "Override · PASS", "btn-pass"),
        mk("rescan", "Rescan clean", "btn-ghost")));
    } else {
      const d = r.decision || {};
      side.append(el("div", { class: "review-outcome" },
        el("div", null, el("strong", { text: d.decision || r.status }), ` by ${d.reviewer || "—"} `, el("span", { class: "field-hint", text: relTime(d.ts) })),
        el("div", null, "Final routing ", el("code", { text: r.final_action || "—" }), r.final_label ? ` · ${r.final_label}` : ""),
        r.rescan_trace_id ? el("div", null, "Rescan trace ", el("code", { text: r.rescan_trace_id })) : null,
        d.note ? el("div", { class: "field-hint", text: `“${d.note}”` }) : null));
    }
    card.append(el("div", { class: "review-body" }, strip, side));

    const audit = r.audit || [];
    if (audit.length) {
      card.append(el("details", { class: "audit" },
        el("summary", null, `Audit log · ${audit.length} record${audit.length === 1 ? "" : "s"} · `,
          r.audit_valid === false ? el("span", { class: "chain-bad", text: "hash chain broken" }) : el("span", { class: "chain-ok", text: "hash chain intact" })),
        el("ol", null, audit.map((a) => el("li", null,
          el("strong", { text: a.event }), ` · ${a.actor || ""} · ${a.decision || a.action || ""} · `,
          el("span", { class: "field-hint", text: a.ts }), " ", el("span", { class: "hash", text: `#${String(a.hash || "").slice(0, 12)}` }))))));
    }
    return card;
  }

  async function decide(review, decision, note, card, btn) {
    const reviewer = ($("#reviewer").value || "").trim() || "operator";
    store.set("reviewer", reviewer);
    const buttons = $$(".review-actions .btn", card);
    buttons.forEach((b) => { b.disabled = true; });
    btn.classList.add("is-busy");
    const original = btn.textContent;
    btn.textContent = decision === "rescan" ? "Rescanning…" : "Saving…";
    try {
      const res = await api(`/api/reviews/${encodeURIComponent(review.review_id)}/decision`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ decision, reviewer, note }),
      });
      if (res.rescan) {
        const v = res.rescan.trace.verdict || {};
        toast(`Rescan → ${v.label} · ${res.action}`, v.label === "PASS" ? "success" : "warn", 6000);
        await loadReviews();
        const open = el("button", {
          type: "button", class: "btn btn-ghost btn-sm", text: "Open rescan in Inspect",
          onclick: () => {
            selectTab("inspect");
            setRunStatus(`rescan of ${review.trace_id} · preset clean`, null);
            renderRun(res.rescan, { animate: true, token: ++state.runToken });
          },
        });
        $("#review-list").prepend(el("div", { class: "notes" }, `Rescan of ${review.review_id} finished: ${v.label} (${res.action}). `, open));
      } else {
        toast(`${review.review_id}: ${res.action}`, res.action === "route_pass" ? "success" : "info");
        await loadReviews();
      }
    } catch (err) {
      toast(err.message, "error", 6500);
      buttons.forEach((b) => { b.disabled = false; });
      btn.classList.remove("is-busy");
      btn.textContent = original;
      if (err.status === 409) loadReviews();
    }
  }

  function initReviews() {
    $("#reviewer").value = store.get("reviewer", "");
    $$(".segmented .seg").forEach((seg) => seg.addEventListener("click", () => {
      state.reviewStatus = seg.dataset.status;
      $$(".segmented .seg").forEach((s) => s.setAttribute("aria-pressed", String(s === seg)));
      loadReviews();
    }));
    $("#review-refresh").addEventListener("click", loadReviews);
  }

  // ---------------------------------------------------------------------------------------- results
  function dataTable(title, columns, rows) {
    const label = (c) => ({ key: "group", key2: "item" }[c] || c.replace(/_/g, " "));
    const numericCol = columns.map((_, j) => rows.some((r) => typeof r[j] === "number") && rows.every((r) => r[j] === null || r[j] === undefined || typeof r[j] === "number"));
    return el("section", { class: "table-card" },
      el("h3", { text: title }),
      el("div", { class: "table-scroll" }, el("table", { class: "data" },
        el("thead", null, el("tr", null, columns.map((c, j) => el("th", { scope: "col", class: numericCol[j] ? "num" : null, text: label(c) })))),
        el("tbody", null, rows.map((r) => el("tr", null, r.map((v, j) => el("td", { class: numericCol[j] ? "num" : null, text: typeof v === "number" ? fmtNum(v, Number.isInteger(v) ? 0 : 4) : fmtValue(v) }))))))));
  }

  function verdictBar(byLabel) {
    const pass = byLabel.PASS || 0, fail = byLabel.FAIL || 0, rev = byLabel.NEEDS_REVIEW || 0;
    const total = pass + fail + rev;
    if (!total) return null;
    return el("div", { class: "tile-bar", role: "img", "aria-label": `${pass} pass, ${fail} fail, ${rev} review` },
      el("span", { style: `width:${(pass / total) * 100}%;background:var(--pass)` }),
      el("span", { style: `width:${(rev / total) * 100}%;background:var(--warn-fill)` }),
      el("span", { style: `width:${(fail / total) * 100}%;background:var(--fail)` }));
  }

  async function loadMetrics() {
    try {
      const m = await api("/api/metrics");
      const live = m.live || {};
      const by = live.by_label || {};
      const tile = (v, k, extra) => el("div", { class: "tile" }, el("div", { class: "metric-v", text: v }), el("div", { class: "metric-k", text: k }), extra || null);
      $("#live-tiles").replaceChildren(
        tile(fmtNum(live.inspections || 0), "inspections since start", verdictBar(by)),
        tile(`${by.PASS || 0} / ${by.FAIL || 0} / ${by.NEEDS_REVIEW || 0}`, "pass / fail / review"),
        tile(live.means && live.means.cycle_time_ms != null ? fmtMs(live.means.cycle_time_ms) : "—", "mean cycle time (compute + sim)"),
        tile(live.means && live.means.n_steps != null ? fmtNum(live.means.n_steps, 1) : "—", "mean agent steps"),
        tile(fmtNum((m.store && m.store.pending_reviews) || 0), "pending reviews"),
        tile(fmtNum(Object.values(live.decisions || {}).reduce((a, b) => a + b, 0)), "human decisions"));

      const ev = m.eval || {};
      const tables = $("#eval-tables");
      if (ev.available && ev.tables && ev.tables.length) {
        tables.replaceChildren(
          el("p", { class: "field-hint", style: "margin-bottom:10px", text: `Source ${ev.source}${ev.generated_at ? " · generated " + ev.generated_at : ""}` }),
          ...ev.tables.map((t) => dataTable(t.title, t.columns, t.rows)));
      } else {
        tables.replaceChildren(el("div", { class: "empty-state" },
          el("strong", { text: "No evaluation results on this deployment yet." }),
          "Run ", el("code", { text: "python scripts/evaluate.py" }), " to produce results/eval_summary.json and figures."));
      }
      const figs = m.figures || [];
      $("#figures").replaceChildren(...figs.map((f) => el("figure", { class: "figure-card" },
        el("button", { type: "button", "aria-label": `Enlarge figure ${f.title}`, onclick: () => openLightbox(f.url, f.title) }, el("img", { src: f.url, alt: f.title, loading: "lazy" })),
        el("figcaption", { text: f.title }))));
    } catch (err) {
      $("#eval-tables").replaceChildren(el("div", { class: "empty-state" }, el("strong", { text: "Metrics unavailable." }), err.message));
    }
  }

  // ----------------------------------------------------------------------------------------- system
  function kvList(obj, skip = []) {
    const dl = el("dl", { class: "kv" });
    for (const [k, v] of Object.entries(obj || {})) {
      if (skip.includes(k) || v === undefined) continue;
      if (v && typeof v === "object" && !Array.isArray(v)) continue;
      dl.append(el("dt", { text: k.replace(/_/g, " ") }), el("dd", { text: fmtValue(v) }));
    }
    return dl;
  }

  function renderSystem(sys) {
    const cv = sys.cv || {};
    const dep = sys.deployment || {};
    const set = sys.settings || {};
    const card = (title, ...body) => el("section", { class: "sys-card" }, el("h3", null, title), ...body);
    const group = (title, obj) => el("div", { class: "kv-group" }, el("div", { class: "kv-group-title", text: title }), kvList(obj));
    const archBadge = dep.graviton ? el("span", { class: "badge badge--accent", text: "Graviton" }) : null;
    $("#sys-grid").replaceChildren(
      card(["OpenCV", el("span", { class: "badge badge--unknown", text: cv.build_backend || "stock" })], el("div", { class: "sys-hero", text: cv.opencv_version || "?" }), kvList(cv, ["opencv_version"])),
      card(["Deployment", archBadge], el("div", { class: "sys-hero", text: dep.arch || "?" }), kvList(dep, ["arch"])),
      card("Runtime", kvList(sys.runtime || {}), group("limits", sys.limits || {})),
      card("Agent & storage",
        kvList({ planner_default: set.planner_default, bedrock_model_id: set.bedrock_model_id, bedrock_configured: set.bedrock_configured, cv_threads: set.cv_threads, log_level: set.log_level }),
        group("storage", set.storage || {}), group("policy", set.policy || {})),
      card("Models & samples",
        kvList({ available_categories: (sys.models || {}).available_categories, known_categories: (sys.models || {}).known_categories, presets: sys.presets, samples: sys.samples, started_at: sys.started_at })),
      card("Perception settings", group("quality", set.quality || {}), group("camera", set.camera || {}), group("detector", set.detector || {})));
  }

  function renderGeneric(obj, title) {
    const frag = document.createDocumentFragment();
    if (!obj || typeof obj !== "object") { frag.append(el("pre", { class: "raw-pre", text: JSON.stringify(obj, null, 2) })); return frag; }
    const scalars = {}, records = [], others = [];
    for (const [k, v] of Object.entries(obj)) {
      if (isScalar(v) || (Array.isArray(v) && v.every(isScalar))) scalars[k] = v;
      else if (!Array.isArray(v) && Object.values(v).every((x) => isScalar(x) || (Array.isArray(x) && x.every(isScalar)))) records.push([k, v]);
      else others.push([k, v]);
    }
    if (Object.keys(scalars).length) frag.append(el("div", { class: "table-card", style: "padding:10px 14px" }, title ? el("div", { class: "kv-group-title", text: title }) : null, kvList(scalars)));
    if (records.length) {
      const cols = [];
      for (const [, v] of records) for (const k of Object.keys(v)) if (!cols.includes(k)) cols.push(k);
      // Uniform records (e.g. per-op timings) form one table; heterogeneous ones get their own key/value card.
      const uniform = records.length > 1 && records.every(([, v]) => Object.keys(v).length >= 0.6 * cols.length);
      if (uniform) {
        frag.append(dataTable(title || "results", ["key", ...cols], records.map(([k, v]) => [k, ...cols.map((c) => v[c] ?? null)])));
      } else {
        for (const [k, v] of records) {
          frag.append(el("div", { class: "table-card", style: "padding:10px 14px" },
            el("div", { class: "kv-group-title", text: title ? `${title}.${k}` : k }), kvList(v)));
        }
      }
    }
    for (const [k, v] of others) {
      if (Array.isArray(v) && v.length && v.every((x) => x && typeof x === "object" && !Array.isArray(x))) {
        const cols = [];
        for (const r of v) for (const c of Object.keys(r)) if (!cols.includes(c)) cols.push(c);
        frag.append(dataTable(k, cols, v.map((r) => cols.map((c) => (isScalar(r[c]) ? r[c] : fmtValue(r[c]))))));
      } else if (v && typeof v === "object" && !Array.isArray(v)) frag.append(renderGeneric(v, title ? `${title}.${k}` : k));
      else frag.append(el("pre", { class: "raw-pre", text: `${k}: ${JSON.stringify(v, null, 2)}` }));
    }
    return frag;
  }

  async function runBench() {
    const btn = $("#bench-btn");
    let n = parseInt($("#bench-n").value, 10);
    if (!Number.isFinite(n) || n < 1) n = 10;
    n = Math.min(100, n);
    btn.disabled = true;
    btn.textContent = "Running…";
    $("#bench-out").replaceChildren(el("p", { class: "field-hint", text: `Running ${n} iterations on ${(state.system && state.system.deployment && state.system.deployment.arch) || "this host"}…` }));
    try {
      const res = await api(`/api/bench?n=${n}`);
      $("#bench-out").replaceChildren(
        el("p", { class: "field-hint", style: "margin-bottom:10px", text: `n=${res.n} · arch ${res.arch} · OpenCV ${(res.cv || {}).opencv_version || "?"} (${(res.cv || {}).build_backend || "stock"}) · wall ${fmtMs(res.wall_ms)}` }),
        renderGeneric(res.result, "benchmark"));
    } catch (err) {
      $("#bench-out").replaceChildren(el("div", { class: "empty-state" }, el("strong", { text: err.status === 501 ? "Benchmark not available on this deployment." : "Benchmark failed." }), err.message));
    } finally {
      btn.disabled = false;
      btn.textContent = "Run benchmark";
    }
  }

  // ------------------------------------------------------------------------------------------- init
  function init() {
    initTheme();
    initTabs();
    initLightbox();
    initRawActions();
    initReviews();
    $("#inspect-btn").addEventListener("click", runInspect);
    $("#preset").addEventListener("change", updatePresetHint);
    $("#planner").addEventListener("change", updatePlannerHint);
    $("#seed-random").addEventListener("click", () => { $("#seed").value = String(Math.floor(Math.random() * 100000)); });
    $("#upload-file").addEventListener("change", (e) => selectUpload(e.target.files && e.target.files[0]));
    $("#upload-category").addEventListener("change", () => { if (state.selected && state.selected.kind === "upload") renderRunPart(); });
    $("#results-refresh").addEventListener("click", loadMetrics);
    $("#system-refresh").addEventListener("click", loadSystem);
    $("#bench-btn").addEventListener("click", runBench);
    document.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter" && !$("#panel-inspect").hidden && !$("#inspect-btn").disabled) {
        e.preventDefault();
        runInspect();
      }
    });
    loadSystem();
    loadSamples();
    refreshReviewBadge();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
