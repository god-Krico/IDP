/* Showcase site behaviour: theme toggle, bar chart render, copy-to-clipboard. */

(function () {
  "use strict";

  // ── Theme toggle ─────────────────────────────────────────────────────────
  var STORAGE_KEY = "idp-theme";
  var root = document.documentElement;

  var stored = null;
  try { stored = localStorage.getItem(STORAGE_KEY); } catch (e) { /* private mode */ }
  if (stored === "light" || stored === "dark") root.setAttribute("data-theme", stored);

  var toggle = document.getElementById("themeToggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      var current = root.getAttribute("data-theme") || (prefersDark ? "dark" : "light");
      var next = current === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem(STORAGE_KEY, next); } catch (e) { /* ignore */ }
    });
  }

  // ── Detection-quality bar chart ──────────────────────────────────────────
  // Single series, so no legend: the figure title names the measure. Every bar
  // is directly labelled, which also covers the sub-3:1 contrast relief rule.
  var STAGES = [
    { label: "Vacuum lifting",       value: 0.995, map50: 0.995, instances: 207 },
    { label: "Curing",               value: 0.991, map50: 0.995, instances: 302 },
    { label: "Mould cleaning",       value: 0.948, map50: 0.982, instances: 298 },
    { label: "Surface finishing",    value: 0.947, map50: 0.984, instances: 298 },
    { label: "Concrete pouring",     value: 0.937, map50: 0.980, instances: 322 },
    { label: "Rebar cage placement", value: 0.860, map50: 0.944, instances: 114 }
  ];

  // Axis starts at 0.8 rather than 0 so the spread between classes is legible;
  // every bar carries its own value label, so no reader is misled by the zoom.
  var AXIS_MIN = 0.8;

  function renderBars() {
    var list = document.getElementById("barList");
    if (!list) return;

    STAGES.forEach(function (stage) {
      var pct = ((stage.value - AXIS_MIN) / (1 - AXIS_MIN)) * 100;

      var row = document.createElement("li");
      row.className = "bar-row";
      row.tabIndex = 0;

      var label = document.createElement("span");
      label.className = "bar-label";
      label.textContent = stage.label;

      var track = document.createElement("span");
      track.className = "bar-track";
      var fill = document.createElement("span");
      fill.className = "bar-fill";
      fill.dataset.width = pct.toFixed(1) + "%";
      track.appendChild(fill);

      var value = document.createElement("span");
      value.className = "bar-value";
      value.textContent = stage.value.toFixed(3);

      var tip = document.createElement("span");
      tip.className = "bar-tip";
      tip.setAttribute("role", "tooltip");
      tip.innerHTML =
        "<strong>" + stage.label + "</strong><br>" +
        "mAP50-95 &nbsp;" + stage.value.toFixed(3) + "<br>" +
        "mAP50 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" + stage.map50.toFixed(3) + "<br>" +
        stage.instances + " instances";

      row.append(label, track, value, tip);
      list.appendChild(row);
    });

    // Grow the bars once they are on screen.
    var fills = list.querySelectorAll(".bar-fill");
    function grow() {
      fills.forEach(function (f) { f.style.width = f.dataset.width; });
    }

    if ("IntersectionObserver" in window) {
      var observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) { grow(); observer.disconnect(); }
        });
      }, { threshold: 0.25 });
      observer.observe(list);
    } else {
      grow();
    }
  }

  // ── Copy the sample Drive link ───────────────────────────────────────────
  function wireCopy() {
    var btn = document.getElementById("copyBtn");
    if (!btn) return;

    btn.addEventListener("click", function () {
      var target = document.querySelector(btn.dataset.copy);
      if (!target) return;
      var text = target.textContent.trim();
      var original = btn.textContent;

      function done(ok) {
        btn.textContent = ok ? "Copied ✓" : "Press Ctrl+C";
        setTimeout(function () { btn.textContent = original; }, 1800);
      }

      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(
          function () { done(true); },
          function () { selectFallback(target); done(false); }
        );
      } else {
        selectFallback(target);
        done(false);
      }
    });
  }

  // Without clipboard access, select the text so Ctrl+C works.
  function selectFallback(node) {
    var range = document.createRange();
    range.selectNodeContents(node);
    var sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () { renderBars(); wireCopy(); });
  } else {
    renderBars();
    wireCopy();
  }
})();
