const state = {
  token: localStorage.getItem("contextrot_token"),
  username: localStorage.getItem("contextrot_username"),
  sessionId: Number(localStorage.getItem("contextrot_session_id")) || null,
  sessions: [],
  knowledgeBases: [],
  messages: [],
  attachedImage: null,
};

const $ = (id) => document.getElementById(id);
const apiUrl = () => {
  const host = window.location.hostname || "127.0.0.1";
  return `http://${host}:8000`;
};

function showError(element, message) {
  element.textContent = message;
  element.classList.toggle("hidden", !message);
}

function notify(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.add("show");
  window.setTimeout(() => toast.classList.remove("show"), 3200);
}

async function request(path, options = {}, retries = 1) {
  const headers = new Headers(options.headers || {});
  if (state.token) headers.set("Authorization", `Bearer ${state.token}`);
  let response;
  try {
    response = await fetch(`${apiUrl()}${path}`, { ...options, headers });
  } catch (err) {
    if (retries > 0) {
      await new Promise((res) => setTimeout(res, 1200));
      return request(path, options, retries - 1);
    }
    throw new Error("Could not connect to the server. Please check if the backend is running.");
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 401) {
      localStorage.removeItem("contextrot_token");
      state.token = null;
      setAuthenticated(false);
      throw new Error("Session expired. Please sign in again.");
    }
    const detail = typeof body.detail === "string" ? body.detail : "The request could not be completed.";
    const error = new Error(detail);
    error.status = response.status;
    throw error;
  }
  return body;
}

function setAuthenticated(authenticated) {
  $("auth-view").classList.toggle("hidden", authenticated);
  $("workspace-view").classList.toggle("hidden", !authenticated);
  $("logout-button").classList.toggle("hidden", !authenticated);
  $("user-badge").classList.toggle("hidden", !authenticated);
  $("user-badge").textContent = authenticated ? `Hi, ${state.username}` : "";
  if (authenticated) $("dashboard-name").textContent = state.username;
}

function formatDate(value) {
  return new Date(value).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function renderKnowledgeBases() {
  $("knowledge-base-list").innerHTML = state.knowledgeBases.length
    ? state.knowledgeBases.map((item) => `<article class="library-card"><div class="library-icon">◫</div><h2>${escapeHtml(item.name)}</h2><p>${escapeHtml(item.subject || "General reference material")}</p><label class="upload-library">Upload PDF<input type="file" accept=".pdf,application/pdf" data-library="${item.id}" hidden></label><div class="library-documents" id="library-docs-${item.id}">Loading documents…</div></article>`).join("")
    : '<div class="card empty-activity">No reference libraries yet. Create one for your course material.</div>';
  $("session-library").innerHTML = '<option value="">No library selected</option>' + state.knowledgeBases.map((item) => `<option value="${item.id}">${escapeHtml(item.name)}</option>`).join("");
  $("knowledge-base-list").querySelectorAll("[data-library]").forEach((input) => input.addEventListener("change", uploadReferenceDocument));
  state.knowledgeBases.forEach(async (item) => {
    try {
      const documents = await request(`/knowledge-bases/${item.id}/documents`);
      const target = $(`library-docs-${item.id}`);
      target.textContent = documents.length ? `${documents.length} reference document${documents.length === 1 ? "" : "s"}` : "No documents uploaded yet";
    } catch (error) { notify(error.message); }
  });
}

async function loadKnowledgeBases() {
  state.knowledgeBases = await request("/knowledge-bases");
  renderKnowledgeBases();
}

function renderSessions() {
  const list = $("session-list");
  list.innerHTML = "";
  if (!state.sessions.length) {
    list.innerHTML = '<p class="welcome-message">No sessions yet.<br>Start one to begin.</p>';
    return;
  }

  state.sessions.forEach((session) => {
    const button = document.createElement("button");
    button.className = `session-item ${session.id === state.sessionId ? "active" : ""}`;
    button.innerHTML = `<strong>${escapeHtml(session.title)}</strong><small>${escapeHtml(session.subject || "General")} · ${formatDate(session.started_at)}</small>`;
    button.addEventListener("click", () => selectSession(session.id));
    list.appendChild(button);
  });
  $("activity-list").innerHTML = state.sessions.length
    ? state.sessions.slice(0, 3).map((session) => `<button class="activity-row" data-session="${session.id}"><span class="activity-icon">◈</span><span><strong>${escapeHtml(session.title)}</strong><small>${escapeHtml(session.subject || "General")} · ${formatDate(session.started_at)}</small></span><b>›</b></button>`).join("")
    : '<div class="empty-activity">Your recent learning activity will appear here.</div>';
  $("activity-list").querySelectorAll("[data-session]").forEach((item) => item.addEventListener("click", () => selectSession(Number(item.dataset.session))));
  $("all-sessions").innerHTML = state.sessions.length
    ? state.sessions.map((session) => `<button class="all-session-row" data-session="${session.id}"><span class="activity-icon">◈</span><span><strong>${escapeHtml(session.title)}</strong><small>${escapeHtml(session.subject || "General")} · Started ${formatDate(session.started_at)}</small></span><b>Open session&nbsp; →</b></button>`).join("")
    : '<div class="card empty-activity">No sessions yet. Start your first AI study session.</div>';
  $("all-sessions").querySelectorAll("[data-session]").forEach((item) => item.addEventListener("click", () => selectSession(Number(item.dataset.session))));
}

function renderDocuments(documents) {
  const strip = $("document-strip");
  if (!documents || !documents.length) {
    strip.innerHTML = `
      <label class="document-pill document-pill-upload" title="Upload a PDF to ground your answers">
        <b>+</b><span>Upload a PDF to ground your answers</span>
        <input type="file" class="pill-file-input" accept=".pdf,application/pdf" hidden>
      </label>
    `;
  } else {
    strip.innerHTML = documents.map((doc) => `
      <div class="document-pill">
        <b>PDF</b><span>${escapeHtml(doc.filename)}</span><small>${doc.chunks} chunks</small>
      </div>
    `).join("") + `
      <label class="document-pill document-pill-upload" title="Add another PDF to this session">
        <b>+</b><span>Add PDF</span>
        <input type="file" class="pill-file-input" accept=".pdf,application/pdf" hidden>
      </label>
    `;
  }
  strip.querySelectorAll(".pill-file-input").forEach((input) => input.addEventListener("change", uploadDocument));
}

function renderMessages() {
  const container = $("chat-messages");
  if (!state.messages.length) {
    container.innerHTML = '<div class="welcome-message"><strong>Your notes, your questions.</strong>Upload a PDF, then ask for an explanation, summary, or quiz.</div>';
    return;
  }
  container.innerHTML = state.messages.map((message) => {
    const sources = message.sources?.length
      ? `<div class="sources">${message.sources.map((source) => `<span class="source">${escapeHtml(source.source)}${source.page ? ` · p. ${source.page}` : ""}</span>`).join("")}</div>`
      : "";
    let diagramsHtml = "";
    if (message.diagrams?.length) {
      diagramsHtml = message.diagrams
        .filter((d) => !message.content?.includes(`/images/${d.id}`))
        .map((d) => {
          const src = `${apiUrl()}${d.image_url}`;
          return `<div class="message-diagram"><img src="${src}" alt="${escapeHtml(d.caption)}" onclick="window.open(this.src, '_blank')" title="Click to view full size" /><div class="diagram-caption">📊 ${escapeHtml(d.caption)} (Page ${d.page})</div></div>`;
        })
        .join("");
    }
    let userAttachedHtml = "";
    if (message.role === "user" && message.image_data) {
      userAttachedHtml = `<div class="user-attached-image"><img src="${message.image_data}" alt="Attached image" onclick="window.open(this.src, '_blank')" title="Click to view full size" /></div>`;
    }
    const content = message.role === "assistant"
      ? renderAssistantContent(message.content)
      : escapeHtml(message.content);
    return `<article class="message ${message.role}"><div class="message-bubble">${content}${userAttachedHtml}${diagramsHtml}</div>${sources}<div class="message-meta">${message.role === "user" ? "You" : "Context Rot"}</div></article>`;
  }).join("");

  if (window.renderMathInElement) {
    try {
      renderMathInElement(container, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\[", right: "\\]", display: true },
          { left: "\\(", right: "\\)", display: false }
        ],
        throwOnError: false
      });
    } catch (_) {}
  }
  container.scrollTop = container.scrollHeight;
}

function parseMarkdownTable(tableLines, mathTokens) {
  if (tableLines.length < 2) return "";
  const headerCells = tableLines[0].replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
  let align = [];
  let startIndex = 1;

  if (tableLines.length > 1 && /^\|?\s*:?-+:?\s*\|/.test(tableLines[1])) {
    align = tableLines[1].replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => {
      const trimmed = c.trim();
      if (trimmed.startsWith(":") && trimmed.endsWith(":")) return "center";
      if (trimmed.endsWith(":")) return "right";
      return "left";
    });
    startIndex = 2;
  }

  const html = ['<div class="table-responsive"><table class="data-table"><thead><tr>'];
  headerCells.forEach((c, idx) => {
    const a = align[idx] || "left";
    html.push(`<th style="text-align:${a}">${formatInline(c, mathTokens)}</th>`);
  });
  html.push("</tr></thead><tbody>");

  for (let i = startIndex; i < tableLines.length; i++) {
    const rawLine = tableLines[i].trim();
    if (!rawLine || !rawLine.includes("|")) continue;
    const cells = rawLine.replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
    html.push("<tr>");
    headerCells.forEach((_, idx) => {
      const cellVal = cells[idx] !== undefined ? cells[idx] : "";
      const a = align[idx] || "left";
      html.push(`<td style="text-align:${a}">${formatInline(cellVal, mathTokens)}</td>`);
    });
    html.push("</tr>");
  }
  html.push("</tbody></table></div>");
  return html.join("");
}

function extractCoord(pt) {
  if (!pt) return null;
  if (Array.isArray(pt) && pt.length >= 2) {
    const x = Number(pt[0]), y = Number(pt[1]);
    return (!isNaN(x) && !isNaN(y)) ? { x, y } : null;
  }
  if (typeof pt === "object") {
    const x = Number(pt.x !== undefined ? pt.x : (pt.X !== undefined ? pt.X : (pt[0] !== undefined ? pt[0] : NaN)));
    const y = Number(pt.y !== undefined ? pt.y : (pt.Y !== undefined ? pt.Y : (pt[1] !== undefined ? pt[1] : NaN)));
    return (!isNaN(x) && !isNaN(y)) ? { x, y } : null;
  }
  return null;
}

function getLinePointsFromEquation(eqStr) {
  if (!eqStr) return [];
  const m = eqStr.match(/([+\-]?[0-9]*\.?[0-9]+)?\s*x\s*([+\-]\s*[0-9]*\.?[0-9]+)?\s*y\s*=\s*([+\-]?[0-9]*\.?[0-9]+)/i);
  if (m) {
    let aStr = (m[1] || "1").replace(/\s+/g, "");
    if (aStr === "+" || aStr === "") aStr = "1";
    if (aStr === "-") aStr = "-1";
    let bStr = (m[2] || "+1").replace(/\s+/g, "");
    if (bStr === "+" || bStr === "") bStr = "1";
    if (bStr === "-") bStr = "-1";
    const a = parseFloat(aStr), b = parseFloat(bStr), c = parseFloat(m[3]);
    if (!isNaN(a) && !isNaN(b) && !isNaN(c) && (a !== 0 || b !== 0)) {
      const pts = [];
      if (b !== 0) pts.push({ x: 0, y: c / b });
      if (a !== 0) pts.push({ x: c / a, y: 0 });
      if (pts.length >= 2 && Math.abs(pts[0].x - pts[1].x) > 1e-4) return pts;
      if (b !== 0) pts.push({ x: 2, y: (c - 2 * a) / b });
      return pts;
    }
  }
  return [];
}

function clipLineToBox(x1, y1, x2, y2, minX, maxX, minY, maxY) {
  const dx = x2 - x1, dy = y2 - y1;
  if (Math.abs(dx) < 1e-9 && Math.abs(dy) < 1e-9) return null;
  if (Math.abs(dx) < 1e-9) {
    if (x1 < minX || x1 > maxX) return null;
    return { x1, y1: minY, x2: x1, y2: maxY };
  }
  if (Math.abs(dy) < 1e-9) {
    if (y1 < minY || y1 > maxY) return null;
    return { x1: minX, y1, x2: maxX, y2: y1 };
  }
  const m = dy / dx;
  const c = y1 - m * x1;
  const candidates = [];
  const yAtMinX = m * minX + c;
  if (yAtMinX >= minY - 1e-5 && yAtMinX <= maxY + 1e-5) candidates.push({ x: minX, y: yAtMinX });
  const yAtMaxX = m * maxX + c;
  if (yAtMaxX >= minY - 1e-5 && yAtMaxX <= maxY + 1e-5) candidates.push({ x: maxX, y: yAtMaxX });
  const xAtMinY = (minY - c) / m;
  if (xAtMinY >= minX - 1e-5 && xAtMinY <= maxX + 1e-5) candidates.push({ x: xAtMinY, y: minY });
  const xAtMaxY = (maxY - c) / m;
  if (xAtMaxY >= minX - 1e-5 && xAtMaxY <= maxX + 1e-5) candidates.push({ x: xAtMaxY, y: maxY });

  const unique = [];
  for (const p of candidates) {
    if (!unique.some((u) => Math.abs(u.x - p.x) < 1e-4 && Math.abs(u.y - p.y) < 1e-4)) {
      unique.push(p);
    }
  }
  if (unique.length >= 2) {
    return { x1: unique[0].x, y1: unique[0].y, x2: unique[1].x, y2: unique[1].y };
  }
  return null;
}

function generateGeometrySvg(spec) {
  const title = spec.title || "Geometric Diagram";
  const width = Number(spec.width) || 580;
  const height = Number(spec.height) || 380;
  const elements = Array.isArray(spec.elements) ? spec.elements : [];

  const svgParts = [];
  const randId = Math.random().toString(36).substring(2, 8);
  const arrowId = `geom_arr_${randId}`;

  svgParts.push(`
    <defs>
      <marker id="${arrowId}" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#334155" />
      </marker>
    </defs>
    <rect width="${width}" height="${height}" fill="#ffffff" rx="12" />
  `);

  elements.forEach((el) => {
    const type = (el.type || "").toLowerCase();
    const color = el.color || "#2563eb";
    const strokeWidth = el.strokeWidth || 2;
    const isDashed = el.style === "dashed" || el.dashed;
    const dashArray = isDashed ? (el.dash || "5,4") : "none";

    if (type === "circle") {
      const cx = Number(el.cx !== undefined ? el.cx : el.x);
      const cy = Number(el.cy !== undefined ? el.cy : el.y);
      const r = Number(el.r || el.radius || 50);
      const fill = el.fill || "rgba(37, 99, 235, 0.04)";
      svgParts.push(`
        <circle cx="${cx}" cy="${cy}" r="${r}" fill="${fill}" stroke="${color}" stroke-width="${strokeWidth}" stroke-dasharray="${dashArray}" />
        <circle cx="${cx}" cy="${cy}" r="3" fill="${color}" />
      `);
      if (el.label) {
        svgParts.push(`<text x="${cx + 8}" y="${cy + 14}" font-size="12" font-weight="700" fill="#1e293b" font-family="Space Grotesk, sans-serif">${escapeHtml(el.label)}</text>`);
      }
    } else if (type === "triangle" || type === "polygon") {
      let pts = [];
      const rawPoints = el.points || el.vertices || [];
      if (Array.isArray(rawPoints)) {
        pts = rawPoints.map((p) => {
          const pt = extractCoord(p);
          return pt ? `${pt.x},${pt.y}` : "";
        }).filter(Boolean);
      }
      if (pts.length >= 3) {
        const fill = el.fill || "rgba(37, 99, 235, 0.06)";
        svgParts.push(`
          <polygon points="${pts.join(" ")}" fill="${fill}" stroke="${color}" stroke-width="${strokeWidth}" stroke-linejoin="round" stroke-dasharray="${dashArray}" />
        `);
      }
      if (Array.isArray(el.labels)) {
        el.labels.forEach((lbl, i) => {
          const pt = extractCoord(rawPoints[i]);
          if (pt) {
            svgParts.push(`
              <circle cx="${pt.x}" cy="${pt.y}" r="3.5" fill="#1e293b" />
              <text x="${pt.x + 8}" y="${pt.y - 6}" font-size="13" font-weight="700" fill="#1e293b" font-family="Space Grotesk, sans-serif">${escapeHtml(String(lbl))}</text>
            `);
          }
        });
      }
    } else if (type === "line" || type === "segment" || type === "ray") {
      const x1 = Number(el.x1 !== undefined ? el.x1 : (el.from ? el.from[0] : 0));
      const y1 = Number(el.y1 !== undefined ? el.y1 : (el.from ? el.from[1] : 0));
      const x2 = Number(el.x2 !== undefined ? el.x2 : (el.to ? el.to[0] : 0));
      const y2 = Number(el.y2 !== undefined ? el.y2 : (el.to ? el.to[1] : 0));
      const marker = type === "ray" ? `marker-end="url(#${arrowId})"` : "";
      svgParts.push(`
        <line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${strokeWidth}" stroke-dasharray="${dashArray}" ${marker} />
      `);
      if (el.label) {
        const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
        const lblW = el.label.length * 6.5 + 8;
        svgParts.push(`
          <rect x="${mx - lblW / 2}" y="${my - 16}" width="${lblW}" height="14" rx="3" fill="rgba(255,255,255,0.92)" stroke="#e2e8f0" stroke-width="0.6" />
          <text x="${mx}" y="${my - 6}" font-size="10" font-weight="600" fill="#334155" text-anchor="middle" font-family="DM Sans, sans-serif">${escapeHtml(el.label)}</text>
        `);
      }
    } else if (type === "right_angle") {
      const v = extractCoord(el.vertex || el.v);
      const p1 = extractCoord(el.p1);
      const p2 = extractCoord(el.p2);
      if (v && p1 && p2) {
        const size = Number(el.size || 14);
        const d1x = p1.x - v.x, d1y = p1.y - v.y;
        const len1 = Math.hypot(d1x, d1y) || 1;
        const u1x = (d1x / len1) * size, u1y = (d1y / len1) * size;
        const d2x = p2.x - v.x, d2y = p2.y - v.y;
        const len2 = Math.hypot(d2x, d2y) || 1;
        const u2x = (d2x / len2) * size, u2y = (d2y / len2) * size;

        const c1x = v.x + u1x, c1y = v.y + u1y;
        const c2x = v.x + u1x + u2x, c2y = v.y + u1y + u2y;
        const c3x = v.x + u2x, c3y = v.y + u2y;
        svgParts.push(`
          <path d="M ${c1x} ${c1y} L ${c2x} ${c2y} L ${c3x} ${c3y}" fill="none" stroke="#475569" stroke-width="1.5" />
        `);
      }
    } else if (type === "angle" || type === "arc") {
      const v = extractCoord(el.vertex || el.center);
      if (v) {
        const r = Number(el.r || el.radius || 24);
        const start = Number(el.startAngle || 0) * (Math.PI / 180);
        const end = Number(el.endAngle || 60) * (Math.PI / 180);
        const sx = v.x + r * Math.cos(start), sy = v.y - r * Math.sin(start);
        const ex = v.x + r * Math.cos(end), ey = v.y - r * Math.sin(end);
        const largeArc = Math.abs(end - start) > Math.PI ? 1 : 0;
        svgParts.push(`
          <path d="M ${sx} ${sy} A ${r} ${r} 0 ${largeArc} 0 ${ex} ${ey}" fill="none" stroke="${color}" stroke-width="1.5" />
        `);
        if (el.label) {
          const mid = (start + end) / 2;
          const lx = v.x + (r + 14) * Math.cos(mid);
          const ly = v.y - (r + 14) * Math.sin(mid);
          svgParts.push(`<text x="${lx}" y="${ly}" font-size="11" font-weight="700" fill="#1e293b" text-anchor="middle">${escapeHtml(el.label)}</text>`);
        }
      }
    } else if (type === "point") {
      const pt = extractCoord(el);
      if (pt) {
        const lbl = el.label || "";
        svgParts.push(`
          <circle cx="${pt.x}" cy="${pt.y}" r="4.5" fill="#ffffff" stroke="${color}" stroke-width="2.5" />
          <circle cx="${pt.x}" cy="${pt.y}" r="2" fill="${color}" />
        `);
        if (lbl) {
          const lx = pt.x + 8, ly = pt.y - 8;
          svgParts.push(`
            <rect x="${lx - 3}" y="${ly - 10}" width="${lbl.length * 7 + 8}" height="14" rx="3" fill="rgba(255,255,255,0.92)" stroke="#e2e8f0" stroke-width="0.8" />
            <text x="${lx + (lbl.length * 7 + 8) / 2 - 3}" y="${ly}" font-size="11" font-weight="700" fill="#1e293b" text-anchor="middle" font-family="Space Grotesk, sans-serif">${escapeHtml(lbl)}</text>
          `);
        }
      }
    } else if (type === "text" || type === "label") {
      const x = Number(el.x || 0), y = Number(el.y || 0);
      const textVal = el.text || el.label || "";
      const fontSize = el.fontSize || 12;
      svgParts.push(`
        <text x="${x}" y="${y}" font-size="${fontSize}" font-weight="600" fill="#334155" font-family="DM Sans, sans-serif">${escapeHtml(textVal)}</text>
      `);
    }
  });

  return `
    <div class="interactive-graph-container">
      <div class="graph-header">
        <div style="display:flex;align-items:center;gap:8px;">
          <span class="graph-tag" style="background:#e0e7ff;color:#3730a3;">📐 Geometry Diagram</span>
          <span class="graph-title-text">${escapeHtml(title)}</span>
        </div>
      </div>
      <div class="graph-svg-wrapper">
        <svg viewBox="0 0 ${width} ${height}" class="cartesian-graph-svg" xmlns="http://www.w3.org/2000/svg">
          ${svgParts.join("")}
        </svg>
      </div>
      <div class="graph-footer">
        <span style="color:#64748b;font-size:11px;">Geometric Construction</span>
        <div class="graph-controls">
          <button type="button" class="graph-btn" onclick="const s=this.closest('.interactive-graph-container').querySelector('svg').outerHTML;const w=window.open('');w.document.write('<html><head><title>Geometry Diagram</title><style>body{margin:0;display:grid;place-items:center;min-height:100vh;background:#f8fafc;}svg{max-width:90vw;max-height:90vh;box-shadow:0 8px 30px rgba(0,0,0,0.1);border-radius:12px;background:#fff;}</style></head><body>' + s + '</body></html>');w.document.close();">🔍 Open Fullscreen</button>
        </div>
      </div>
    </div>
  `;
}

function generateCartesianSvg(spec) {
  const title = spec.title || "Cartesian Coordinate Graph";
  const xLabel = spec.xLabel || "x";
  const yLabel = spec.yLabel || "y";
  const lines = Array.isArray(spec.lines) ? spec.lines : [];
  const points = Array.isArray(spec.points) ? spec.points : [];
  const nature = (spec.nature || "").toLowerCase();

  let allX = [];
  let allY = [];

  for (const l of lines) {
    if (Array.isArray(l.points)) {
      for (const pt of l.points) {
        const c = extractCoord(pt);
        if (c) { allX.push(c.x); allY.push(c.y); }
      }
    }
  }
  for (const pt of points) {
    const c = extractCoord(pt);
    if (c) { allX.push(c.x); allY.push(c.y); }
  }

  if (allX.length === 0) allX = [0, 5];
  if (allY.length === 0) allY = [0, 5];

  let minX = spec.xMin !== undefined ? Number(spec.xMin) : Math.min(...allX);
  let maxX = spec.xMax !== undefined ? Number(spec.xMax) : Math.max(...allX);
  let minY = spec.yMin !== undefined ? Number(spec.yMin) : Math.min(...allY);
  let maxY = spec.yMax !== undefined ? Number(spec.yMax) : Math.max(...allY);

  minX = Math.min(minX, 0) - 1;
  maxX = Math.max(maxX, 2) + 1.5;
  minY = Math.min(minY, 0) - 1;
  maxY = Math.max(maxY, 2) + 1.5;

  const spanX = maxX - minX;
  const spanY = maxY - minY;
  const stepX = spanX <= 12 ? 1 : (spanX <= 25 ? 2 : (spanX <= 60 ? 5 : 10));
  const stepY = spanY <= 12 ? 1 : (spanY <= 25 ? 2 : (spanY <= 60 ? 5 : 10));

  minX = Math.floor(minX / stepX) * stepX;
  maxX = Math.ceil(maxX / stepX) * stepX;
  minY = Math.floor(minY / stepY) * stepY;
  maxY = Math.ceil(maxY / stepY) * stepY;

  const width = 720, height = 440;
  const margin = { top: 48, right: 75, bottom: 55, left: 65 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;

  const toSvgX = (x) => margin.left + ((x - minX) / (maxX - minX)) * plotW;
  const toSvgY = (y) => margin.top + ((maxY - y) / (maxY - minY)) * plotH;

  const randId = Math.random().toString(36).substring(2, 8);
  const arrowId = `arr_${randId}`;

  const svgContent = [];
  svgContent.push(`
    <defs>
      <marker id="${arrowId}" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#334155" />
      </marker>
    </defs>
    <rect width="${width}" height="${height}" fill="#ffffff" rx="12" />
  `);

  const gridSvg = [];
  for (let x = minX; x <= maxX; x += stepX) {
    const sx = toSvgX(x);
    gridSvg.push(`<line x1="${sx}" y1="${margin.top}" x2="${sx}" y2="${margin.top + plotH}" stroke="#e9ecef" stroke-dasharray="2,2" stroke-width="1" />`);
    if (x !== 0 || minX === 0) {
      gridSvg.push(`<text x="${sx}" y="${margin.top + plotH + 16}" font-size="10" fill="#64748b" text-anchor="middle" font-family="DM Sans, sans-serif">${x}</text>`);
    }
  }
  for (let y = minY; y <= maxY; y += stepY) {
    const sy = toSvgY(y);
    gridSvg.push(`<line x1="${margin.left}" y1="${sy}" x2="${margin.left + plotW}" y2="${sy}" stroke="#e9ecef" stroke-dasharray="2,2" stroke-width="1" />`);
    if (y !== 0 || minY === 0) {
      gridSvg.push(`<text x="${margin.left - 8}" y="${sy + 3}" font-size="10" fill="#64748b" text-anchor="end" font-family="DM Sans, sans-serif">${y}</text>`);
    }
  }
  svgContent.push(`<g class="grid">${gridSvg.join("")}</g>`);

  const originX = toSvgX(0);
  const originY = toSvgY(0);
  const clampedOriginX = Math.max(margin.left, Math.min(margin.left + plotW, originX));
  const clampedOriginY = Math.max(margin.top, Math.min(margin.top + plotH, originY));

  // X & Y Axes: Arrow and Text anchored to prevent any right-edge cutoff
  svgContent.push(`
    <line x1="${margin.left}" y1="${clampedOriginY}" x2="${margin.left + plotW + 15}" y2="${clampedOriginY}" stroke="#334155" stroke-width="1.8" marker-end="url(#${arrowId})" />
    <text x="${margin.left + plotW + 12}" y="${clampedOriginY - 10}" font-size="11" font-weight="700" fill="#1e293b" text-anchor="end" font-family="Space Grotesk, sans-serif">${escapeHtml(xLabel)}</text>
    <line x1="${clampedOriginX}" y1="${margin.top + plotH}" x2="${clampedOriginX}" y2="${margin.top - 16}" stroke="#334155" stroke-width="1.8" marker-end="url(#${arrowId})" />
    <text x="${clampedOriginX + 10}" y="${margin.top - 18}" font-size="11" font-weight="700" fill="#1e293b" text-anchor="start" font-family="Space Grotesk, sans-serif">${escapeHtml(yLabel)}</text>
  `);

  if (minX <= 0 && maxX >= 0 && minY <= 0 && maxY >= 0) {
    svgContent.push(`<text x="${originX - 7}" y="${originY + 14}" font-size="11" font-weight="600" fill="#64748b">O</text>`);
  }

  const defaultColors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c"];
  const linesSvg = [];

  lines.forEach((line, idx) => {
    const color = line.color || defaultColors[idx % defaultColors.length];
    const isCoincident = nature.includes("coincid") || line.coincident || (line.nature && line.nature.toLowerCase().includes("coincid")) || (lines.length > 1 && idx === 1 && lines[0].equation === line.equation);
    const strokeDash = line.dash ? (Array.isArray(line.dash) ? line.dash.join(",") : line.dash) : (isCoincident ? "7,5" : "none");
    const strokeWidth = isCoincident ? (idx === 0 ? 4.5 : 2.5) : (line.strokeWidth || 2.5);

    let linePts = [];
    if (Array.isArray(line.points)) {
      linePts = line.points.map(extractCoord).filter(Boolean);
    }
    if (linePts.length < 2) {
      linePts = getLinePointsFromEquation(line.equation || line.name || line.label);
    }
    if (linePts.length < 2 && points.length >= 2) {
      const idxOffset = (idx * 2) % points.length;
      const c1 = extractCoord(points[idxOffset]);
      const c2 = extractCoord(points[(idxOffset + 1) % points.length]);
      if (c1 && c2 && (c1.x !== c2.x || c1.y !== c2.y)) {
        linePts = [c1, c2];
      }
    }

    if (linePts.length >= 2) {
      const x1 = linePts[0].x, y1 = linePts[0].y;
      const x2 = linePts[1].x, y2 = linePts[1].y;
      const seg = clipLineToBox(x1, y1, x2, y2, minX, maxX, minY, maxY);
      if (seg) {
        const sx1 = toSvgX(seg.x1), sy1 = toSvgY(seg.y1);
        const sx2 = toSvgX(seg.x2), sy2 = toSvgY(seg.y2);

        if (isCoincident && idx === 1) {
          // Distinct red dashed line drawn directly on top of the solid blue coincident line
          linesSvg.push(`
            <line x1="${sx1}" y1="${sy1}" x2="${sx2}" y2="${sy2}" stroke="#dc2626" stroke-width="2.8" stroke-dasharray="8,6" />
          `);
        } else if (isCoincident && idx === 0) {
          // Thick solid royal blue line
          linesSvg.push(`
            <line x1="${sx1}" y1="${sy1}" x2="${sx2}" y2="${sy2}" stroke="#2563eb" stroke-width="4.5" />
          `);
        } else {
          linesSvg.push(`
            <line x1="${sx1}" y1="${sy1}" x2="${sx2}" y2="${sy2}" stroke="${color}" stroke-width="${strokeWidth}" stroke-dasharray="${strokeDash}" />
          `);
        }
      }
    }
  });

  svgContent.push(`<g class="graph-lines">${linesSvg.join("")}</g>`);

  const allPlottedPoints = [...points];
  lines.forEach((l) => {
    if (Array.isArray(l.points)) {
      l.points.forEach((p) => {
        const c = extractCoord(p);
        if (c) {
          allPlottedPoints.push({ x: c.x, y: c.y, label: p.label || `(${c.x}, ${c.y})`, color: l.color });
        }
      });
    }
  });

  const seenPts = new Set();
  const uniquePts = [];
  for (const p of allPlottedPoints) {
    const c = extractCoord(p);
    if (!c) continue;
    const key = `${c.x},${c.y}`;
    if (!seenPts.has(key)) {
      seenPts.add(key);
      uniquePts.push({ x: c.x, y: c.y, label: p.label || `(${c.x}, ${c.y})`, color: p.color });
    }
  }

  const ptsSvg = [];
  uniquePts.forEach((pt) => {
    const sx = toSvgX(pt.x), sy = toSvgY(pt.y);
    const ptColor = pt.color || "#0f172a";
    const coordStr = `(${pt.x}, ${pt.y})`;
    let lbl = pt.label ? String(pt.label).trim() : coordStr;
    if (!lbl.includes(`(${pt.x}`) && !lbl.includes(`${pt.x},`)) {
      lbl = `${lbl} ${coordStr}`;
    }
    ptsSvg.push(`
      <circle cx="${sx}" cy="${sy}" r="5.5" fill="#ffffff" stroke="${ptColor}" stroke-width="2.5" />
      <circle cx="${sx}" cy="${sy}" r="2" fill="${ptColor}" />
    `);
    const lblWidth = lbl.length * 6.8 + 10;
    const lblY = (sy > margin.top + 30) ? sy - 12 : sy + 18;
    const lblX = Math.max(margin.left + 5, Math.min(margin.left + plotW - lblWidth, sx - lblWidth / 2));
    ptsSvg.push(`
      <rect x="${lblX}" y="${lblY - 10}" width="${lblWidth}" height="14" rx="4" fill="rgba(255,255,255,0.92)" stroke="#e2e8f0" stroke-width="0.8" />
      <text x="${lblX + lblWidth / 2}" y="${lblY}" font-size="10" font-weight="700" fill="#1e293b" text-anchor="middle" font-family="DM Sans, sans-serif">${escapeHtml(lbl)}</text>
    `);
  });
  svgContent.push(`<g class="points">${ptsSvg.join("")}</g>`);

  const legendHtml = [];
  lines.forEach((l, i) => {
    const col = l.color || defaultColors[i % defaultColors.length];
    const isCoinc = nature.includes("coincid") || l.coincident || (l.nature && l.nature.toLowerCase().includes("coincid"));
    const eq = l.equation || l.name || l.label || `Line ${i + 1}`;
    legendHtml.push(`
      <span style="display:inline-flex;align-items:center;gap:6px;font-size:11px;font-weight:600;color:#334155;">
        <span style="display:inline-block;width:16px;height:3.5px;background:${isCoinc && i === 1 ? "#dc2626" : col};border-radius:2px;${isCoinc && i === 1 ? "border-top: 2px dashed #ffffff;" : ""}"></span>
        ${escapeHtml(eq)}${isCoinc ? (i === 1 ? " (Coincident Overlay)" : " (Base)") : ""}
      </span>
    `);
  });

  let statusBadge = "";
  if (nature.includes("coincid") || lines.some((l) => l.coincident || (l.nature && l.nature.toLowerCase().includes("coincid")))) {
    statusBadge = '<span class="graph-status-badge coincident">⚪ Coincident Lines (Infinitely Many Solutions)</span>';
  } else if (nature.includes("parallel") || lines.some((l) => l.parallel || (l.nature && l.nature.toLowerCase().includes("parallel")))) {
    statusBadge = '<span class="graph-status-badge parallel">∥ Parallel Lines (No Solution)</span>';
  } else if (nature.includes("intersect") || lines.some((l) => l.intersecting || (l.nature && l.nature.toLowerCase().includes("intersect")))) {
    statusBadge = '<span class="graph-status-badge intersecting">✕ Intersecting Lines (Unique Solution)</span>';
  }

  return `
    <div class="interactive-graph-container">
      <div class="graph-header">
        <div style="display:flex;align-items:center;gap:8px;">
          <span class="graph-tag">📈 Graph Plot</span>
          <span class="graph-title-text">${escapeHtml(title)}</span>
        </div>
        <div style="display:flex;gap:12px;flex-wrap:wrap;">
          ${legendHtml.join("")}
        </div>
      </div>
      <div class="graph-svg-wrapper">
        <svg viewBox="0 0 ${width} ${height}" class="cartesian-graph-svg" xmlns="http://www.w3.org/2000/svg">
          ${svgContent.join("")}
        </svg>
      </div>
      <div class="graph-footer">
        ${statusBadge || '<span style="color:#64748b;font-size:11px;">Cartesian Coordinate System</span>'}
        <div class="graph-controls">
          <button type="button" class="graph-btn" onclick="const s=this.closest('.interactive-graph-container').querySelector('svg').outerHTML;const w=window.open('');w.document.write('<html><head><title>Graph Plot</title><style>body{margin:0;display:grid;place-items:center;min-height:100vh;background:#f8fafc;}svg{max-width:90vw;max-height:90vh;box-shadow:0 8px 30px rgba(0,0,0,0.1);border-radius:12px;background:#fff;}</style></head><body>' + s + '</body></html>');w.document.close();">🔍 Open Fullscreen</button>
        </div>
      </div>
    </div>
  `;
}

function synthesizeGraphFromText(text) {
  const lowered = text.toLowerCase();
  if (!/(?:graphical|graphically|plot|coordinate\s+table|geometrical)/i.test(lowered)) {
    return null;
  }

  const eqRegex = /([0-9]*\.?[0-9]*\s*[a-zA-Z]\s*[+\-]\s*[0-9]*\.?[0-9]*\s*[a-zA-Z]\s*=\s*[0-9]+(?:\.[0-9]+)?)/g;
  const eqMatches = [...text.matchAll(eqRegex)].map((m) => m[1].trim());

  const ptRegex = /\(\s*(-?[0-9]+(?:\.[0-9]+)?)\s*,\s*(-?[0-9]+(?:\.[0-9]+)?)\s*\)/g;
  const ptMatches = [];
  let match;
  while ((match = ptRegex.exec(text)) !== null) {
    ptMatches.push([parseFloat(match[1]), parseFloat(match[2])]);
  }

  if (ptMatches.length < 2 && eqMatches.length === 0) return null;

  let nature = "intersecting";
  if (lowered.includes("coincid") || lowered.includes("infinitely many")) {
    nature = "coincident";
  } else if (lowered.includes("parallel") || lowered.includes("no solution") || lowered.includes("inconsistent")) {
    nature = "parallel";
  }

  const lines = [];
  const colors = ["#2563eb", "#dc2626", "#059669"];

  if (eqMatches.length >= 2 && ptMatches.length >= 4) {
    lines.push({
      equation: eqMatches[0],
      points: [ptMatches[0], ptMatches[1]],
      color: colors[0],
      label: eqMatches[0],
    });
    lines.push({
      equation: eqMatches[1],
      points: [ptMatches[2], ptMatches[3]],
      color: colors[1],
      label: eqMatches[1],
      coincident: nature === "coincident",
    });
  } else if (eqMatches.length >= 2 && ptMatches.length >= 2 && nature === "coincident") {
    lines.push({
      equation: eqMatches[0],
      points: [ptMatches[0], ptMatches[1]],
      color: colors[0],
      label: eqMatches[0],
    });
    lines.push({
      equation: eqMatches[1],
      points: [ptMatches[0], ptMatches[1]],
      color: colors[1],
      label: eqMatches[1],
      coincident: true,
    });
  } else if (eqMatches.length >= 1 && ptMatches.length >= 2) {
    lines.push({
      equation: eqMatches[0] || "Line 1",
      points: [ptMatches[0], ptMatches[1]],
      color: colors[0],
      label: eqMatches[0] || "Line 1",
    });
    if (eqMatches.length >= 2) {
      lines.push({
        equation: eqMatches[1],
        points: ptMatches.length >= 4 ? [ptMatches[2], ptMatches[3]] : [ptMatches[0], ptMatches[1]],
        color: colors[1],
        label: eqMatches[1],
        coincident: nature === "coincident",
      });
    }
  }

  return {
    title: eqMatches.length ? `Graphical Representation of ${eqMatches.slice(0, 2).join(" & ")}` : "Graphical Representation",
    xLabel: "x",
    yLabel: "y",
    lines,
    points: ptMatches.map((p) => ({ x: p[0], y: p[1], label: `(${p[0]}, ${p[1]})` })),
    nature,
  };
}

function renderAssistantContent(value) {
  const mathTokens = [];
  function registerMath(tex, isDisplay) {
    const id = `@@MATH_${mathTokens.length}@@`;
    let html = "";
    if (window.katex && window.katex.renderToString) {
      try {
        html = window.katex.renderToString(tex.trim(), {
          displayMode: isDisplay,
          throwOnError: false,
        });
      } catch (_) {
        html = isDisplay ? `$$${escapeHtml(tex)}$$` : `$${escapeHtml(tex)}$`;
      }
    } else {
      html = isDisplay ? `$$${escapeHtml(tex)}$$` : `$${escapeHtml(tex)}$`;
    }
    mathTokens.push({ id, html, isDisplay });
    return isDisplay ? `\n\n${id}\n\n` : id;
  }

  let text = String(value);

  // Parse fenced geometry / graph / diagram / plot / json blocks
  const graphTokens = [];
  text = text.replace(/```(?:graph|plot|chart|geometry|diagram|json)?\s*([\s\S]*?)\s*```/gi, (match, blockContent) => {
    const trimmed = blockContent.trim();
    if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) {
      return match;
    }
    try {
      const cleaned = trimmed.replace(/,\s*([\]}])/g, "$1");
      const spec = JSON.parse(cleaned);
      if (spec.type === "geometry" || Array.isArray(spec.elements)) {
        const id = `@@GRAPH_${graphTokens.length}@@`;
        graphTokens.push({ id, html: generateGeometrySvg(spec) });
        return `\n\n${id}\n\n`;
      }
      if (spec.title || spec.lines || spec.points || spec.nature || spec.xLabel) {
        const id = `@@GRAPH_${graphTokens.length}@@`;
        graphTokens.push({ id, html: generateCartesianSvg(spec) });
        return `\n\n${id}\n\n`;
      }
      return match;
    } catch (_) {
      return match;
    }
  });

  // Strip code fences enclosing <svg>...</svg> so raw SVG is processed cleanly
  text = text.replace(/```(?:svg|xml|html)?\s*(<svg\b[\s\S]*?<\/svg>)\s*```/gi, "$1");

  // Parse inline raw SVG blocks
  const svgTokens = [];
  text = text.replace(/<svg\b[^>]*>[\s\S]*?<\/svg>/gi, (svgCode) => {
    const id = `@@SVG_${svgTokens.length}@@`;
    svgTokens.push({
      id,
      html: `<div class="interactive-graph-container"><div class="graph-svg-wrapper">${svgCode}</div></div>`,
    });
    return `\n\n${id}\n\n`;
  });

  // Fallback: If no explicit graph or SVG was returned, synthesize from coordinate tables/equations
  if (graphTokens.length === 0 && svgTokens.length === 0) {
    const synth = synthesizeGraphFromText(text);
    if (synth) {
      const id = `@@GRAPH_${graphTokens.length}@@`;
      text += `\n\n${id}\n\n`;
      graphTokens.push({ id, html: generateCartesianSvg(synth) });
    }
  }

  // Protect display math ($$...$$ and \[...\])
  text = text.replace(/\$\$([\s\S]*?)\$\$/g, (_, tex) => registerMath(tex, true));
  text = text.replace(/\\\[([\s\S]*?)\\\]/g, (_, tex) => registerMath(tex, true));

  // Protect inline math ($...$ and \(...\)) - skip standalone currency
  text = text.replace(/\$([^\$\n\r]+?)\$/g, (match, tex) => {
    if (/^\s*\d+(?:\.\d+)?\s*$/.test(tex)) return match;
    return registerMath(tex, false);
  });
  text = text.replace(/\\\(([\s\S]*?)\\\)/g, (_, tex) => registerMath(tex, false));

  const lines = text.split(/\r?\n/);
  const blocks = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();
    if (!trimmed) {
      blocks.push('<div class="answer-spacer"></div>');
      continue;
    }

    const tokenMatch = trimmed.match(/^@@MATH_(\d+)@@$/);
    if (tokenMatch) {
      const token = mathTokens[Number(tokenMatch[1])];
      if (token && token.isDisplay) {
        blocks.push(`<div class="math-display-block">${token.html}</div>`);
        continue;
      }
    }

    const graphMatch = trimmed.match(/^@@GRAPH_(\d+)@@$/);
    if (graphMatch) {
      const token = graphTokens[Number(graphMatch[1])];
      if (token) {
        blocks.push(token.html);
        continue;
      }
    }

    const svgMatch = trimmed.match(/^@@SVG_(\d+)@@$/);
    if (svgMatch) {
      const token = svgTokens[Number(svgMatch[1])];
      if (token) {
        blocks.push(token.html);
        continue;
      }
    }

    // Markdown table detection
    if (trimmed.includes("|") && trimmed.replace(/[^|]/g, "").length >= 2) {
      const tableLines = [];
      while (i < lines.length && lines[i].trim().includes("|") && lines[i].trim().replace(/[^|]/g, "").length >= 2) {
        tableLines.push(lines[i].trim());
        i++;
      }
      i--;
      const tableHtml = parseMarkdownTable(tableLines, mathTokens);
      if (tableHtml) {
        blocks.push(tableHtml);
        continue;
      }
    }

    const imgMatch = trimmed.match(/^!\[(.*?)\]\((.*?)\)$/);
    if (imgMatch) {
      let src = imgMatch[2];
      if (src.startsWith("/")) {
        src = `${apiUrl()}${src}`;
      }
      const alt = escapeHtml(imgMatch[1]);
      blocks.push(`<div class="message-diagram"><img src="${src}" alt="${alt}" onclick="window.open(this.src, '_blank')" title="Click to view full size" /><div class="diagram-caption">📊 ${alt}</div></div>`);
      continue;
    }

    if (/^#{1,3}\s+/.test(trimmed)) {
      blocks.push(`<h3>${formatInline(trimmed.replace(/^#{1,3}\s+/, ""), mathTokens)}</h3>`);
      continue;
    }

    if (/^(?:Q(?:uestion)?\s*\d*[:.)]|Answer\s*:)/i.test(trimmed)) {
      const labelMatch = trimmed.match(/^(Q(?:uestion)?\s*\d*|Answer)\s*([:.)])\s*(.*)$/i);
      if (labelMatch) {
        blocks.push(`<p class="answer-label ${labelMatch[1].toLowerCase().startsWith("answer") ? "answer-label-answer" : ""}"><strong>${escapeHtml(labelMatch[1])}${escapeHtml(labelMatch[2])}</strong> ${formatInline(labelMatch[3], mathTokens)}</p>`);
        continue;
      }
    }

    if (/^(?:[-•]|\d+[.)])\s+/.test(trimmed)) {
      const item = trimmed.replace(/^(?:[-•]|\d+[.)])\s+/, "");
      blocks.push(`<p class="answer-list-item">${formatInline(item, mathTokens)}</p>`);
      continue;
    }

    blocks.push(`<p>${formatInline(trimmed, mathTokens)}</p>`);
  }

  // Automatic graph synthesis fallback if graphical representation requested but no graph block rendered
  if (graphTokens.length === 0 && svgTokens.length === 0) {
    const autoSpec = synthesizeGraphFromText(value);
    if (autoSpec) {
      blocks.push(generateCartesianSvg(autoSpec));
    }
  }

  let result = blocks.join("");
  for (const token of mathTokens) {
    if (result.includes(token.id)) {
      result = result.replaceAll(token.id, token.html);
    }
  }
  return result;
}

function formatInline(value, mathTokens = []) {
  let output = escapeHtml(value);
  output = output.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  output = output.replace(/__(.+?)__/g, "<strong>$1</strong>");
  output = output.replace(/\*([^*\n]+)\*/g, "<em>$1</em>");
  output = output.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  if (mathTokens && mathTokens.length) {
    for (const token of mathTokens) {
      if (output.includes(token.id)) {
        output = output.replaceAll(token.id, token.html);
      }
    }
  }
  return output;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" })[character]);
}

async function loadSessions() {
  state.sessions = await request("/sessions");
  await loadKnowledgeBases();
  renderSessions();
  showDashboard();
}

function showDashboard() {
  setView("dashboard");
  $("dashboard-view").classList.remove("hidden");
  $("empty-state").classList.remove("hidden");
  $("empty-state").classList.add("hidden");
}

function setView(view) {
  document.querySelectorAll(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === view));
  $("dashboard-view").classList.toggle("hidden", view !== "dashboard");
  $("session-view").classList.toggle("hidden", view !== "ai");
  $("sessions-view").classList.toggle("hidden", view !== "sessions");
  $("content-view").classList.toggle("hidden", view !== "content");
  $("progress-view").classList.toggle("hidden", view !== "progress");
  if (view !== "ai") $("empty-state").classList.add("hidden");
  if (view === "ai" && !state.sessionId) $("empty-state").classList.remove("hidden");
}

async function fetchSessionDocuments(session) {
  if (!session) {
    session = state.sessions.find((item) => item.id === state.sessionId);
  }
  if (!session) return [];
  const docs = [];
  if (session.knowledge_base_id) {
    try {
      const kbDocs = await request(`/knowledge-bases/${session.knowledge_base_id}/documents`);
      docs.push(...kbDocs);
    } catch (_) {}
  }
  try {
    const sessionDocs = await request(`/integrations/sessions/${session.id}/documents`);
    docs.push(...sessionDocs);
  } catch (_) {}
  const unique = [];
  const seen = new Set();
  for (const d of docs) {
    const key = `${d.id}-${d.filename}`;
    if (!seen.has(key)) {
      seen.add(key);
      unique.push(d);
    }
  }
  return unique;
}

async function selectSession(id) {
  const session = state.sessions.find((item) => item.id === id);
  if (!session) return;
  state.sessionId = id;
  localStorage.setItem("contextrot_session_id", String(id));
  $("empty-state").classList.add("hidden");
  setView("ai");
  $("session-view").classList.remove("hidden");
  $("session-title").textContent = session.title;
  $("session-subject").textContent = session.subject || "Study session";
  renderSessions();
  try {
    const [history, documents] = await Promise.all([
      request(`/sessions/${id}/history`),
      fetchSessionDocuments(session),
    ]);
    state.messages = history.map((message) => ({ role: message.role, content: message.content, image_data: message.image_data }));
    renderMessages();
    renderDocuments(documents);
  } catch (error) {
    notify(error.message);
  }
}

function openNewSession() {
  $("session-dialog").showModal();
}

async function submitAuth(event) {
  event.preventDefault();
  const mode = document.querySelector(".auth-tab.active").dataset.mode;
  const payload = {
    username: $("username").value.trim(),
    password: $("password").value,
  };
  if (mode === "signup") payload.email = $("email").value.trim();
  showError($("auth-error"), "");
  const button = $("auth-submit-label");
  button.textContent = mode === "signup" ? "Creating…" : "Signing in…";
  try {
    if (mode === "signup") await request("/auth/signup", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    const login = await request("/auth/login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ username: payload.username, password: payload.password }) });
    state.token = login.access_token;
    state.username = payload.username;
    localStorage.setItem("contextrot_token", state.token);
    localStorage.setItem("contextrot_username", state.username);
    setAuthenticated(true);
    await loadSessions();
  } catch (error) {
    showError($("auth-error"), error.message);
  } finally {
    button.textContent = mode === "signup" ? "Create account" : "Sign in";
  }
}

async function createSession(event) {
  event.preventDefault();
  showError($("session-error"), "");
  try {
    const session = await request("/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: $("session-name").value.trim(), subject: $("session-subject-input").value.trim() || null, knowledge_base_id: Number($("session-library").value) || null }),
    });
    $("session-dialog").close();
    $("session-form").reset();
    state.sessions.unshift(session);
    await selectSession(session.id);
  } catch (error) {
    showError($("session-error"), error.message);
  }
}

async function createKnowledgeBase(event) {
    event.preventDefault();
    showError($("library-error"), "");
    try {
      await request("/knowledge-bases", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: $("library-name").value.trim(), subject: $("library-subject").value.trim() || null }),
      });
      $("library-dialog").close();
      $("library-form").reset();
      await loadKnowledgeBases();
      setView("content");
    } catch (error) { showError($("library-error"), error.message); }
  }

async function uploadReferenceDocument(event) {
    const input = event.target;
    const file = input.files[0];
    input.value = "";
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    try {
      await request(`/knowledge-bases/${input.dataset.library}/documents`, { method: "POST", body: form });
      await loadKnowledgeBases();
      notify(`${file.name} is now available to student sessions.`);
    } catch (error) { notify(error.message); }
}

async function uploadDocument(event) {
  const file = event.target.files[0];
  event.target.value = "";
  if (!file || !state.sessionId) return;
  if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
    notify("Please choose a PDF file.");
    return;
  }
  const form = new FormData();
  form.append("file", file);
  notify(`Indexing ${file.name}...`);
  try {
    await request(`/integrations/sessions/${state.sessionId}/documents`, { method: "POST", body: form });
    const documents = await fetchSessionDocuments();
    renderDocuments(documents);
    notify(`${file.name} is ready for questions.`);
  } catch (error) {
    notify(error.message);
  }
}

function showImagePreview(dataUrl, name) {
  $("image-preview-img").src = dataUrl;
  $("image-preview-name").textContent = name || "Attached screenshot";
  $("image-preview-bar").classList.remove("hidden");
  $("chat-input").placeholder = "Ask about this question/diagram (or press Enter to solve)...";
}

function clearImageAttachment() {
  state.attachedImage = null;
  $("image-preview-img").src = "";
  $("image-preview-bar").classList.add("hidden");
  $("chat-input").placeholder = "Ask anything, or attach a question/diagram screenshot…";
}

function handleImageFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    notify("Please choose an image file (PNG, JPG, WebP).");
    return;
  }
  const reader = new FileReader();
  reader.onload = (e) => {
    const rawDataUrl = e.target.result;
    const img = new Image();
    img.onload = () => {
      let width = img.width;
      let height = img.height;
      const maxDim = 1600;
      if (width > maxDim || height > maxDim) {
        if (width > height) {
          height = Math.round((height * maxDim) / width);
          width = maxDim;
        } else {
          width = Math.round((width * maxDim) / height);
          height = maxDim;
        }
      }
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0, width, height);
      const optimized = canvas.toDataURL("image/jpeg", 0.88);
      state.attachedImage = optimized;
      showImagePreview(optimized, file.name || "Screenshot");
    };
    img.src = rawDataUrl;
  };
  reader.readAsDataURL(file);
}

async function sendMessage(event) {
  event.preventDefault();
  const input = $("chat-input");
  const content = input.value.trim();
  const imageData = state.attachedImage;
  if (!content && !imageData) return;
  if (!state.sessionId) return;
  input.value = "";
  input.style.height = "auto";
  const userMsgText = content || "Please solve the question shown in the attached image.";
  state.messages.push({ role: "user", content: userMsgText, image_data: imageData });
  clearImageAttachment();
  renderMessages();
  try {
    const response = await request(`/sessions/${state.sessionId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content: userMsgText, image_data: imageData }),
    });
    state.messages.push({
      role: "assistant",
      content: response.assistant_message.content,
      sources: response.sources || [],
      diagrams: response.diagrams || [],
    });
    renderMessages();
  } catch (error) {
    state.messages.pop();
    renderMessages();
    notify(error.message);
  }
}

$("auth-form").addEventListener("submit", submitAuth);
document.querySelectorAll(".auth-tab").forEach((tab) => tab.addEventListener("click", () => {
  document.querySelectorAll(".auth-tab").forEach((item) => item.classList.remove("active"));
  tab.classList.add("active");
  const signup = tab.dataset.mode === "signup";
  $("email-field").classList.toggle("hidden", !signup);
  $("email").required = signup;
  $("auth-submit-label").textContent = signup ? "Create account" : "Sign in";
}));
$("new-session-button").addEventListener("click", openNewSession);
$("empty-new-session").addEventListener("click", openNewSession);
$("dashboard-ai-button").addEventListener("click", () => state.sessions.length ? selectSession(state.sessionId || state.sessions[0].id) : openNewSession());
$("list-new-session").addEventListener("click", openNewSession);
$("new-library-button").addEventListener("click", () => $("library-dialog").showModal());
$("close-library-dialog").addEventListener("click", () => $("library-dialog").close());
$("library-form").addEventListener("submit", createKnowledgeBase);
$("session-dialog").addEventListener("close", () => showDashboard());
document.addEventListener("click", (event) => {
  const nav = event.target.closest("[data-view]");
  if (nav) {
    if (nav.dataset.view === "ai" && state.sessions.length) {
      selectSession(state.sessionId || state.sessions[0].id);
    } else {
      setView(nav.dataset.view);
    }
  }
});
$("close-dialog").addEventListener("click", () => $("session-dialog").close());
$("session-library").addEventListener("change", (e) => {
  const libId = Number(e.target.value);
  const lib = state.knowledgeBases.find((k) => k.id === libId);
  if (lib) {
    $("session-subject-input").value = lib.subject || lib.name || "";
  }
});
$("session-form").addEventListener("submit", createSession);
$("chat-form").addEventListener("submit", sendMessage);
$("attach-image-button")?.addEventListener("click", () => $("image-attachment-input")?.click());
$("image-attachment-input")?.addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  e.target.value = "";
  if (file) handleImageFile(file);
});
$("remove-image-button")?.addEventListener("click", clearImageAttachment);
$("chat-input").addEventListener("paste", (event) => {
  const items = (event.clipboardData || event.originalEvent?.clipboardData)?.items;
  if (items) {
    for (const item of items) {
      if (item.type && item.type.indexOf("image") !== -1) {
        event.preventDefault();
        const file = item.getAsFile();
        if (file) handleImageFile(file);
        break;
      }
    }
  }
});
const composerEl = $("chat-form");
if (composerEl) {
  ["dragenter", "dragover"].forEach((name) => {
    composerEl.addEventListener(name, (e) => {
      e.preventDefault();
      e.stopPropagation();
      composerEl.classList.add("drag-active");
    });
  });
  ["dragleave", "drop"].forEach((name) => {
    composerEl.addEventListener(name, (e) => {
      e.preventDefault();
      e.stopPropagation();
      composerEl.classList.remove("drag-active");
    });
  });
  composerEl.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    if (dt && dt.files && dt.files.length && dt.files[0].type.startsWith("image/")) {
      handleImageFile(dt.files[0]);
    }
  });
}
$("chat-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("chat-form").requestSubmit();
  }
});
$("chat-input").addEventListener("input", (event) => {
  event.target.style.height = "auto";
  event.target.style.height = `${Math.min(event.target.scrollHeight, 130)}px`;
});
$("session-file-input")?.addEventListener("change", uploadDocument);
$("logout-button").addEventListener("click", () => {
  localStorage.removeItem("contextrot_token");
  localStorage.removeItem("contextrot_username");
  localStorage.removeItem("contextrot_session_id");
  window.location.reload();
});

function initializeAuth() {
  if (state.token) {
    setAuthenticated(true);
    loadSessions().catch((error) => {
      if (error.status === 401) {
        localStorage.removeItem("contextrot_token");
        localStorage.removeItem("contextrot_username");
        setAuthenticated(false);
      }
      notify(error.message);
    });
  } else setAuthenticated(false);
}

initializeAuth();
