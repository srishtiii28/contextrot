const API_URL = localStorage.getItem("contextCompassApiUrl") || "http://127.0.0.1:8000";
const state = { token: localStorage.getItem("contextCompassToken"), user: null, sessions: [], activeSession: null, messages: [], health: null, analyses: [], busy: false };

const app = document.querySelector("#app");
const escapeHtml = (text = "") => String(text).replace(/[&<>'"]/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]);
const date = value => value ? new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }).format(new Date(value)) : "Just now";
const scoreClass = score => score == null ? "neutral" : score >= 75 ? "good" : score >= 50 ? "warn" : "bad";

async function request(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  const response = await fetch(`${API_URL}${path}`, { ...options, headers });
  if (response.status === 204) return null;
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(body.detail || "Something went wrong. Please try again.");
    error.status = response.status;
    throw error;
  }
  return body;
}

function notice(message, type = "error") {
  const existing = document.querySelector(".toast");
  if (existing) existing.remove();
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  document.body.append(toast);
  setTimeout(() => toast.remove(), 4000);
}

function renderAuth(mode = "login") {
  app.innerHTML = `<main class="auth-page"><section class="auth-card">
    <div class="brand"><span class="brand-mark">◈</span><span>Context Compass</span></div>
    <p class="eyebrow">AI study companion</p>
    <h1>${mode === "login" ? "Welcome back" : "Start studying with clarity"}</h1>
    <p class="muted">${mode === "login" ? "Sign in to continue your focused learning sessions." : "Create an account to track conversation health as you learn."}</p>
    <form id="auth-form" class="stack" novalidate>
      ${mode === "signup" ? `<label>Username<input required minlength="3" name="username" autocomplete="username" placeholder="e.g. alex_studies" /></label><label>Email<input required type="email" name="email" autocomplete="email" placeholder="you@example.com" /></label>` : `<label>Username<input required name="username" autocomplete="username" placeholder="Your username" /></label>`}
      <label>Password<input required minlength="8" type="password" name="password" autocomplete="${mode === "login" ? "current-password" : "new-password"}" placeholder="At least 8 characters" /></label>
      <button class="primary" type="submit">${mode === "login" ? "Sign in" : "Create account"}</button>
    </form>
    <p class="switch">${mode === "login" ? "New here?" : "Already have an account?"} <button class="link-button" id="toggle-auth">${mode === "login" ? "Create an account" : "Sign in"}</button></p>
    <details class="api-settings"><summary>API connection</summary><label>Backend URL<input id="api-url" value="${escapeHtml(API_URL)}" /></label><button id="save-api" class="text-button">Save connection</button></details>
  </section></main>`;
  document.querySelector("#toggle-auth").onclick = () => renderAuth(mode === "login" ? "signup" : "login");
  document.querySelector("#save-api").onclick = () => { const url = document.querySelector("#api-url").value.replace(/\/$/, ""); localStorage.setItem("contextCompassApiUrl", url); location.reload(); };
  document.querySelector("#auth-form").onsubmit = async event => {
    event.preventDefault(); const form = new FormData(event.currentTarget); const payload = Object.fromEntries(form);
    const button = event.currentTarget.querySelector("button[type=submit]"); button.disabled = true;
    try {
      if (mode === "signup") await request("/auth/signup", { method: "POST", body: JSON.stringify(payload) });
      const login = await request("/auth/login", { method: "POST", body: JSON.stringify({ username: payload.username, password: payload.password }) });
      state.token = login.access_token; localStorage.setItem("contextCompassToken", state.token); await loadApp();
    } catch (error) { notice(error.message); button.disabled = false; }
  };
}

function layout() {
  app.innerHTML = `<aside class="sidebar"><div class="brand"><span class="brand-mark">◈</span><span>Context Compass</span></div><button class="new-session" id="new-session">＋ New study session</button><nav><p class="nav-title">YOUR SESSIONS</p><div id="session-list"></div></nav><div class="sidebar-bottom"><span class="user-name">${escapeHtml(state.user?.username || "")}</span><button id="logout" class="text-button">Sign out</button></div></aside><main class="main-content"><div id="workspace"></div></main>`;
  document.querySelector("#new-session").onclick = showNewSession;
  document.querySelector("#logout").onclick = () => { localStorage.removeItem("contextCompassToken"); state.token = null; renderAuth(); };
  renderSessions(); renderWorkspace();
}

function renderSessions() {
  const target = document.querySelector("#session-list"); if (!target) return;
  target.innerHTML = state.sessions.length ? state.sessions.map(session => `<button class="session-item ${state.activeSession?.id === session.id ? "active" : ""}" data-id="${session.id}"><span>${escapeHtml(session.title)}</span><small>${escapeHtml(session.subject || "General")}</small></button>`).join("") : `<p class="empty-side">No sessions yet</p>`;
  target.querySelectorAll(".session-item").forEach(button => button.onclick = () => selectSession(Number(button.dataset.id)));
}

function renderWorkspace() {
  const target = document.querySelector("#workspace"); if (!target) return;
  if (!state.activeSession) { target.innerHTML = `<section class="empty-workspace"><div class="empty-icon">✦</div><h1>Your focused study space</h1><p>Create a study session to chat with your tutor and keep an eye on your conversation health.</p><button class="primary" id="empty-new">Create first session</button></section>`; document.querySelector("#empty-new").onclick = showNewSession; return; }
  const health = state.health || { current_score: null, trend: "unavailable", alerts: [] };
  target.innerHTML = `<header class="workspace-header"><div><p class="eyebrow">${escapeHtml(state.activeSession.subject || "Study session")}</p><h1>${escapeHtml(state.activeSession.title)}</h1></div><div class="health-pill ${scoreClass(health.current_score)}"><span>●</span>${health.current_score == null ? "Awaiting analysis" : `${Math.round(health.current_score)}% healthy`}</div></header>
  <section class="dashboard"><article class="health-card"><div><p class="card-label">Conversation health</p><strong class="score ${scoreClass(health.current_score)}">${health.current_score == null ? "—" : `${Math.round(health.current_score)}%`}</strong><p class="muted">${health.current_score == null ? "Send a message to begin tracking." : `Trend: ${health.trend}`}</p></div><div class="score-ring ${scoreClass(health.current_score)}"><span>${health.trend === "degrading" ? "↘" : health.trend === "improving" ? "↗" : "→"}</span></div></article><article class="alert-card"><p class="card-label">Focus alerts</p>${health.alerts?.length ? health.alerts.map(alert => `<p class="alert">⚠ ${escapeHtml(alert)}</p>`).join("") : `<p class="clear">✓ No focus issues detected</p>`}</article></section>
  <section class="chat-panel"><div class="chat-header"><div><h2>Study chat</h2><p class="muted">Ask questions, test your understanding, or request an explanation.</p></div><button id="refresh" class="text-button">Refresh insights</button></div><div id="messages" class="messages">${messagesMarkup()}</div><form id="chat-form" class="composer"><textarea id="message-input" rows="1" maxlength="12000" placeholder="Ask your tutor anything…"></textarea><button class="primary send" type="submit" aria-label="Send message">↑</button></form></section>
  <section class="insights"><h2>Latest context signals</h2><div class="signal-grid">${signalsMarkup()}</div></section>`;
  document.querySelector("#chat-form").onsubmit = sendMessage;
  document.querySelector("#refresh").onclick = () => loadSessionData(state.activeSession.id);
  const input = document.querySelector("#message-input"); input.oninput = () => { input.style.height = "auto"; input.style.height = `${Math.min(input.scrollHeight, 140)}px`; };
  document.querySelector("#messages").scrollTop = document.querySelector("#messages").scrollHeight;
}

function messagesMarkup() { return state.messages.length ? state.messages.map(message => `<article class="message ${message.role === "user" ? "user" : "assistant"}"><div class="avatar">${message.role === "user" ? "You" : "AI"}</div><div><div class="message-bubble">${escapeHtml(message.content).replace(/\n/g, "<br>")}</div><time>${date(message.created_at)}</time></div></article>`).join("") : `<div class="chat-empty"><span>✦</span><p>Start the conversation. Your tutor is ready.</p></div>`; }
function signalsMarkup() {
  const analysis = state.analyses[0]; if (!analysis) return `<p class="muted">Signals appear after your tutor responds.</p>`;
  const signals = [["Topic focus", analysis.topic_score, analysis.topic_drift], ["Consistency", 1 - analysis.contradiction_score, analysis.contradiction_detected], ["Memory", analysis.memory_score, analysis.memory_violation], ["Scope", analysis.scope_score, false]];
  return signals.map(([name, value, issue]) => `<article class="signal"><div><span>${name}</span><strong>${Math.round(value * 100)}%</strong></div><div class="bar"><i class="${issue ? "bad" : "good"}" style="width:${Math.max(0, Math.min(100, value * 100))}%"></i></div><small class="${issue ? "bad-text" : "good-text"}">${issue ? "Needs attention" : "On track"}</small></article>`).join("");
}

function showNewSession() {
  const dialog = document.createElement("div"); dialog.className = "modal-backdrop"; dialog.innerHTML = `<section class="modal"><button class="close" aria-label="Close">×</button><p class="eyebrow">NEW SESSION</p><h2>What are you studying?</h2><form id="session-form" class="stack"><label>Session title<input name="title" required maxlength="200" placeholder="e.g. Calculus revision" /></label><label>Subject <span class="optional">(optional)</span><input name="subject" maxlength="120" placeholder="e.g. Mathematics" /></label><button class="primary" type="submit">Create session</button></form></section>`;
  document.body.append(dialog); dialog.querySelector(".close").onclick = () => dialog.remove(); dialog.onclick = event => { if (event.target === dialog) dialog.remove(); };
  dialog.querySelector("#session-form").onsubmit = async event => { event.preventDefault(); const payload = Object.fromEntries(new FormData(event.currentTarget)); try { const created = await request("/sessions", { method: "POST", body: JSON.stringify(payload) }); state.sessions.unshift(created); dialog.remove(); await selectSession(created.id); } catch (error) { notice(error.message); } };
}

async function selectSession(id) { state.activeSession = state.sessions.find(session => session.id === id); state.messages = []; state.health = null; state.analyses = []; renderSessions(); renderWorkspace(); await loadSessionData(id); }
async function loadSessionData(id) { try { const [messages, health, analyses] = await Promise.all([request(`/sessions/${id}/history`), request(`/sessions/${id}/health`), request(`/sessions/${id}/analysis`)]); if (state.activeSession?.id !== id) return; state.messages = messages; state.health = health; state.analyses = analyses.slice().reverse(); renderWorkspace(); } catch (error) { notice(error.message); } }
async function sendMessage(event) { event.preventDefault(); const input = document.querySelector("#message-input"); const content = input.value.trim(); if (!content || state.busy) return; state.busy = true; input.value = ""; const button = event.currentTarget.querySelector("button"); button.disabled = true; try { const reply = await request(`/sessions/${state.activeSession.id}/messages`, { method: "POST", body: JSON.stringify({ content }) }); state.messages.push(reply.user_message, reply.assistant_message); renderWorkspace(); await loadSessionData(state.activeSession.id); } catch (error) { input.value = content; notice(error.message); } finally { state.busy = false; button.disabled = false; } }
async function loadApp() { try { const [user, sessions] = await Promise.all([request("/users/me"), request("/sessions")]); state.user = user; state.sessions = sessions; layout(); if (sessions.length) await selectSession(sessions[0].id); } catch (error) { const wasAuthenticated = Boolean(state.token); if (error.status === 401) { localStorage.removeItem("contextCompassToken"); state.token = null; } renderAuth(); if (wasAuthenticated && error.status !== 401) notice(`Could not connect: ${error.message}`); } }

state.token ? loadApp() : renderAuth();
