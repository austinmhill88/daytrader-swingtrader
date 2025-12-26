async function api(path, opts={}) {
  const r = await fetch(path, { ...opts, headers: { 'Content-Type': 'application/json' }});
  if (!r.ok) throw await r.json().catch(()=>({error:r.statusText}));
  return r.json();
}
function setText(id, text){ const el = document.getElementById(id); if (el) el.textContent = text; }
function byId(id){ return document.getElementById(id); }
function scrollToBottom(el){ try{ el.scrollTop = el.scrollHeight; }catch(_){} }

// NaN/Inf-safe formatters
function fmtUsd(v){
  const n = Number(v);
  if (!isFinite(n)) return '—';
  return n < 0 ? `-$${Math.abs(n).toLocaleString()}` : `$${n.toLocaleString()}`;
}
function fmtPct(v){
  const n = Number(v);
  if (!isFinite(n)) return '—';
  return `${n.toFixed(2)}%`;
}

let CHAT_SESSION_ID = null;

async function openChat(){
  const out = byId('chatOutput');
  const body = { system: "You are a friendly, professional trading assistant. Be honest (never fabricate), concise, and keep continuity. Use logs only as context, never echo raw numbers. If the user says 'continue', pick up where the last answer left off. If web results are provided, cite sources with brief titles and URLs." };
  try {
    const resp = await api('/ui/chat/open', { method:'POST', body: JSON.stringify(body) });
    CHAT_SESSION_ID = resp.session_id;
    if (out) { out.textContent = 'Session started. How can I help?'; scrollToBottom(out); }
  } catch (e) {
    CHAT_SESSION_ID = null;
    if (out) { out.textContent = 'Stateful chat unavailable. Fallback to quick chat.\n'; scrollToBottom(out); }
  }
}

async function resetChat(){ CHAT_SESSION_ID = null; await openChat(); }

async function refreshStatus(){
  try { const s = await api('/ops/status'); setText('statusBadge', `Status: ${s.running ? 'RUNNING' : 'STOPPED'}${s.last_error ? ' • ' + s.last_error : ''}`); }
  catch { setText('statusBadge','Status: ERROR'); }
}

async function refreshSummary(){
  try {
    const d = await api('/dashboard/summary');
    setText('equityValue', fmtUsd(d.equity));
    setText('cashValue', `Cash: ${fmtUsd(d.cash)}${d.buying_power ? ` • BP: ${fmtUsd(d.buying_power)}` : ''}`);
    const realized = d.realized_pnl!=null? fmtUsd(d.realized_pnl) : '—';
    const unreal = d.unrealized_pnl!=null? fmtUsd(d.unrealized_pnl) : '—';
    setText('pnlValue', `${realized} / ${unreal}`); setText('pnlSplit', 'Realized / Unrealized');
    setText('drawdownValue', `DD: ${fmtPct(d.drawdown_pct)}`); setText('exposureValue', `Exposure: ${fmtPct(d.exposure_pct)}`);
    setText('regimeValue', `Regime: ${d.regime || '—'}`); setText('opsCounts', `Pos / Orders: ${d.positions_count ?? '—'} / ${d.open_orders_count ?? '—'}`);
    const caps = d.risk_caps || {};
    setText('capsText', `Caps • Max DD: ${caps.max_dd} • Per Trade: ${caps.per_trade} • Kill Switch: ${caps.kill_switch ? 'On' : 'Off'} • Exposure: ${caps.exposure_limits}`);
  } catch { setText('equityValue','—'); setText('cashValue','—'); setText('pnlValue','—'); setText('drawdownValue','—'); setText('exposureValue','—'); setText('regimeValue','—'); }
}

function renderTickers(data){
  const grid = byId('tickersGrid'); if (!grid) return;
  const items = (data.tickers||[]);
  grid.innerHTML = items.map(t=>{
    const up = t.change>=0;
    return `<div class="ticker ${up?'up':'down'}">
      <div class="sym">${t.symbol}</div>
      <div class="price">$${t.last.toFixed(2)}</div>
      <div class="chg">${up?'+':''}${t.change.toFixed(2)} (${up?'+':''}${t.change_pct.toFixed(2)}%)</div>
    </div>`;
  }).join('');
}

async function refreshTickers(){
  const w = byId('watchlist')?.value || '';
  try { renderTickers(await api('/dashboard/tickers?symbols='+encodeURIComponent(w))); }
  catch { renderTickers({tickers:[]}); }
}

async function refreshPositions(){
  try {
    const data = await api('/dashboard/positions');
    const rows = (data.positions||[]).map(p=>(
      `<tr><td>${p.symbol}</td><td>${p.qty}</td><td>$${Number(p.avg_price).toFixed(2)}</td><td>$${Number(p.mark).toFixed(2)}</td><td>${fmtUsd(p.pnl_usd)}</td><td>${fmtPct(p.pnl_pct)}</td></tr>`
    )).join('');
    const tbody = document.querySelector('#positionsTable tbody');
    if (tbody) tbody.innerHTML = rows || `<tr><td colspan="6">No positions</td></tr>`;
  } catch {
    const tbody = document.querySelector('#positionsTable tbody');
    if (tbody) tbody.innerHTML = `<tr><td colspan="6">Error loading positions</td></tr>`;
  }
}

// AI Assistant with optional web search
async function sendChat(){
  const input = byId('chatInput'); const out = byId('chatOutput');
  let text = (input?.value || '').trim();
  if (!text) return;
  const len = Number(byId('aiLen')?.value || 256);
  let useLogs = !!byId('aiUseLogs')?.checked;
  let useWeb = !!byId('aiUseWeb')?.checked;

  // Slash command: "/web query" forces web search for this turn
  if (text.toLowerCase().startsWith('/web ')) {
    useWeb = true;
    text = text.slice(5).trim();
  }

  if (input) input.value = '';
  if (out) { out.textContent += `\n> ${text}\n`; scrollToBottom(out); }

  try {
    if (CHAT_SESSION_ID) {
      const resp = await api('/ui/chat/send', {
        method:'POST',
        body: JSON.stringify({
          session_id: CHAT_SESSION_ID,
          content: text,
          include_logs: useLogs,
          include_web: useWeb,
          tokens: len
        })
      });
      if (out) { out.textContent += (resp.reply || JSON.stringify(resp)) + '\n'; scrollToBottom(out); }
      return;
    }
  } catch {
    // fall through to quick chat
  }

  // Fallback: one-shot chat (no web)
  try {
    const body = JSON.stringify({ messages: [{ role:'user', content: text }], options: { num_predict: len } });
    const resp = await api('/api/chat', { method:'POST', body });
    const msg = resp.message?.content || JSON.stringify(resp);
    if (out) { out.textContent += msg + '\n'; scrollToBottom(out); }
  } catch (e) {
    if (out) { out.textContent += 'Error: ' + JSON.stringify(e) + '\n'; scrollToBottom(out); }
  }
}

function bind(id, event, handler){ const el = byId(id); if (el) el.addEventListener(event, handler); }

bind('startBtn','click', async()=>{ await api('/ops/start',{method:'POST',body:'{}'}); refreshStatus(); refreshSummary(); refreshPositions(); });
bind('stopBtn','click', async()=>{ await api('/ops/stop',{method:'POST',body:'{}'}); refreshStatus(); });
bind('refreshTickers','click', refreshTickers);
bind('refreshPositions','click', refreshPositions);
bind('sendChat','click', sendChat);
bind('resetChat','click', resetChat);

const chatInput = byId('chatInput');
if (chatInput) chatInput.addEventListener('keydown', (e)=>{ if(e.key==='Enter'){ sendChat(); } });

(async function init(){
  await openChat();
  refreshStatus(); refreshSummary(); refreshTickers(); refreshPositions();
  setInterval(()=>{ refreshSummary(); refreshTickers(); }, 60_000);
})();