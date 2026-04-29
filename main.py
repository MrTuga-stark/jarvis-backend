"""
J.A.R.V.I.S. Backend v2
- Memória persistente (SQLite)
- Google OAuth (Gmail + Calendar)
- Busca na internet (DuckDuckGo, sem chave)
- Tarefas, notas, briefing
- Análise de imagem (via Claude)

Deploy: Railway.app (grátis)
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, HTMLResponse
from pydantic import BaseModel
from typing import Optional, List
import anthropic
import sqlite3
import httpx
import json
import os
import re
import datetime
from contextlib import asynccontextmanager

# ── ENV ──
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DB_PATH           = os.getenv("DB_PATH", "jarvis.db")
USER_NAME         = os.getenv("USER_NAME", "Usuário")
USER_CITY         = os.getenv("USER_CITY", "São Paulo")
PORT              = int(os.getenv("PORT", 8000))
GOOGLE_CLIENT_ID  = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_SECRET     = os.getenv("GOOGLE_CLIENT_SECRET", "")
BASE_URL          = os.getenv("BASE_URL", f"http://localhost:{PORT}")

# ── DB ──
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT, content TEXT, session_id TEXT DEFAULT 'default',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        key TEXT UNIQUE, value TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT, done INTEGER DEFAULT 0,
        priority TEXT DEFAULT 'normal', due_date TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT, tags TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS tokens (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        service TEXT UNIQUE, access_token TEXT,
        refresh_token TEXT, expires_at TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit()
    conn.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    print(f"✅ J.A.R.V.I.S. v2 ativo na porta {PORT}")
    yield

app = FastAPI(title="J.A.R.V.I.S. v2", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── MODELS ──
class ChatReq(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    image_base64: Optional[str] = None

class TaskReq(BaseModel):
    title: str
    priority: Optional[str] = "normal"
    due_date: Optional[str] = None

class NoteReq(BaseModel):
    content: str
    tags: Optional[str] = ""

class MemoryReq(BaseModel):
    key: str
    value: str

class SearchReq(BaseModel):
    query: str

# ── HELPERS ──
def db():
    return sqlite3.connect(DB_PATH)

def get_history(session_id, limit=20):
    conn = db()
    c = conn.cursor()
    c.execute("SELECT role, content FROM messages WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
              (session_id, limit))
    rows = c.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

def save_msg(role, content, session_id):
    conn = db()
    conn.execute("INSERT INTO messages (role, content, session_id) VALUES (?,?,?)", (role, content, session_id))
    conn.commit()
    conn.close()

def get_memory_str():
    conn = db()
    c = conn.cursor()
    c.execute("SELECT key, value FROM memory ORDER BY updated_at DESC LIMIT 30")
    rows = c.fetchall()
    conn.close()
    return "\n".join([f"- {r[0]}: {r[1]}" for r in rows]) if rows else ""

def save_memory_auto(text):
    """Detecta e salva memórias da resposta do Claude"""
    matches = re.findall(r'\[MEMÓRIA:\s*([^=]+)\s*=\s*([^\]]+)\]', text)
    if matches:
        conn = db()
        for k, v in matches:
            conn.execute(
                "INSERT OR REPLACE INTO memory (key, value, updated_at) VALUES (?,?,CURRENT_TIMESTAMP)",
                (k.strip(), v.strip()))
        conn.commit()
        conn.close()
    return re.sub(r'\[MEMÓRIA:[^\]]+\]', '', text).strip()

def get_tasks_str():
    conn = db()
    c = conn.cursor()
    c.execute("SELECT title, priority, due_date FROM tasks WHERE done=0 ORDER BY created_at DESC LIMIT 10")
    rows = c.fetchall()
    conn.close()
    if not rows: return ""
    return "\n".join([f"- [{r[1]}] {r[0]}" + (f" (até {r[2]})" if r[2] else "") for r in rows])

async def fetch_weather():
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"https://wttr.in/{USER_CITY}?format=3")
            return r.text.strip()
    except:
        return ""

async def web_search(query: str) -> str:
    """Busca via DuckDuckGo Instant Answer API (sem chave)"""
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.get("https://api.duckduckgo.com/", params={
                "q": query, "format": "json", "no_redirect": 1, "no_html": 1, "skip_disambig": 1
            })
            data = r.json()
            result = data.get("AbstractText") or data.get("Answer") or ""
            if not result and data.get("RelatedTopics"):
                result = data["RelatedTopics"][0].get("Text", "")
            return result or f"Busca realizada por '{query}'. Sem resultado direto."
    except Exception as e:
        return f"Erro na busca: {str(e)}"

def get_google_token():
    conn = db()
    c = conn.cursor()
    c.execute("SELECT access_token FROM tokens WHERE service='google'")
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

# ── ROUTES ──

@app.get("/")
def root():
    return {"status": "online", "system": "J.A.R.V.I.S.", "version": "2.0",
            "google_connected": bool(get_google_token())}

@app.get("/health")
def health():
    return {"ok": True, "time": datetime.datetime.now().isoformat()}

# ── CHAT ──
@app.post("/chat")
async def chat(req: ChatReq):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(500, "ANTHROPIC_API_KEY não configurada")

    now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    weather = await fetch_weather()
    memory  = get_memory_str()
    tasks   = get_tasks_str()

    system = f"""Você é J.A.R.V.I.S. (Just A Rather Very Intelligent System), assistente pessoal avançado de {USER_NAME}.

PERSONALIDADE: Sofisticado, preciso, levemente irônico, sempre útil. Respostas concisas e diretas.

CONTEXTO:
- Agora: {now}
- Clima: {weather}
{"- Memórias:\n" + memory if memory else ""}
{"- Tarefas pendentes:\n" + tasks if tasks else ""}

GOOGLE: {"✅ Conectado" if get_google_token() else "❌ Não conectado (peça ao usuário para acessar /auth/google)"}

INSTRUÇÕES ESPECIAIS:
- Para salvar memória: inclua [MEMÓRIA: chave = valor] na resposta
- Seja proativo: mencione tarefas pendentes relevantes
- Responda em português do Brasil
- Se análise de imagem: seja detalhado"""

    history = get_history(req.session_id)

    # Build user message (com imagem opcional)
    if req.image_base64:
        user_content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": req.image_base64}},
            {"type": "text", "text": req.message}
        ]
    else:
        user_content = req.message

    history.append({"role": "user", "content": user_content})
    save_msg("user", req.message, req.session_id)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1000,
        system=system,
        messages=history
    )

    reply = resp.content[0].text
    reply = save_memory_auto(reply)
    save_msg("assistant", reply, req.session_id)

    return {"reply": reply, "tokens": resp.usage.input_tokens + resp.usage.output_tokens}

# ── SEARCH ──
@app.post("/search")
async def search(req: SearchReq):
    result = await web_search(req.query)
    return {"query": req.query, "result": result}

# ── TASKS ──
@app.get("/tasks")
def list_tasks():
    conn = db()
    c = conn.cursor()
    c.execute("SELECT id,title,done,priority,due_date,created_at FROM tasks ORDER BY done,created_at DESC")
    rows = c.fetchall()
    conn.close()
    return [{"id":r[0],"title":r[1],"done":r[2],"priority":r[3],"due_date":r[4],"created_at":r[5]} for r in rows]

@app.post("/tasks")
def add_task(t: TaskReq):
    conn = db()
    c = conn.cursor()
    c.execute("INSERT INTO tasks (title,priority,due_date) VALUES (?,?,?)", (t.title,t.priority,t.due_date))
    tid = c.lastrowid
    conn.commit()
    conn.close()
    return {"id": tid, "title": t.title}

@app.patch("/tasks/{tid}/done")
def done_task(tid: int):
    conn = db()
    conn.execute("UPDATE tasks SET done=1 WHERE id=?", (tid,))
    conn.commit()
    conn.close()
    return {"ok": True}

@app.delete("/tasks/{tid}")
def del_task(tid: int):
    conn = db()
    conn.execute("DELETE FROM tasks WHERE id=?", (tid,))
    conn.commit()
    conn.close()
    return {"ok": True}

# ── MEMORY ──
@app.get("/memory")
def list_memory():
    conn = db()
    c = conn.cursor()
    c.execute("SELECT key,value,updated_at FROM memory ORDER BY updated_at DESC")
    rows = c.fetchall()
    conn.close()
    return [{"key":r[0],"value":r[1],"updated_at":r[2]} for r in rows]

@app.post("/memory")
def set_memory(m: MemoryReq):
    conn = db()
    conn.execute("INSERT OR REPLACE INTO memory (key,value,updated_at) VALUES (?,?,CURRENT_TIMESTAMP)", (m.key,m.value))
    conn.commit()
    conn.close()
    return {"ok": True}

# ── NOTES ──
@app.get("/notes")
def list_notes():
    conn = db()
    c = conn.cursor()
    c.execute("SELECT id,content,tags,created_at FROM notes ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return [{"id":r[0],"content":r[1],"tags":r[2],"created_at":r[3]} for r in rows]

@app.post("/notes")
def add_note(n: NoteReq):
    conn = db()
    c = conn.cursor()
    c.execute("INSERT INTO notes (content,tags) VALUES (?,?)", (n.content,n.tags))
    nid = c.lastrowid
    conn.commit()
    conn.close()
    return {"id": nid}

# ── BRIEFING ──
@app.get("/briefing")
async def briefing():
    weather = await fetch_weather()
    memory  = get_memory_str()
    tasks   = get_tasks_str()
    now     = datetime.datetime.now()
    greet   = "Bom dia" if now.hour < 12 else "Boa tarde" if now.hour < 18 else "Boa noite"

    prompt = f"""{greet}! Gere um briefing conciso para {USER_NAME}.
Clima: {weather}
{"Tarefas: " + tasks if tasks else "Sem tarefas pendentes."}
{"Memórias relevantes: " + memory[:500] if memory else ""}
Máximo 120 palavras. Inclua: saudação, clima, tarefas prioritárias, uma dica do dia."""

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    resp = client.messages.create(model="claude-opus-4-5", max_tokens=300,
                                   messages=[{"role":"user","content":prompt}])
    return {"briefing": resp.content[0].text, "time": now.isoformat()}

# ── GOOGLE OAUTH ──
@app.get("/auth/google")
def google_auth():
    if not GOOGLE_CLIENT_ID:
        return HTMLResponse("""
        <h2 style="font-family:monospace;color:#00d4ff">Configure GOOGLE_CLIENT_ID e GOOGLE_CLIENT_SECRET</h2>
        <p style="font-family:monospace">1. Acesse <a href="https://console.cloud.google.com">console.cloud.google.com</a><br>
        2. Crie credenciais OAuth 2.0<br>
        3. Adicione as variáveis no Railway<br>
        4. Reinicie o servidor</p>""")
    
    scopes = "https://www.googleapis.com/auth/gmail.readonly https://www.googleapis.com/auth/calendar.readonly"
    url = (f"https://accounts.google.com/o/oauth2/auth"
           f"?client_id={GOOGLE_CLIENT_ID}"
           f"&redirect_uri={BASE_URL}/auth/google/callback"
           f"&scope={scopes}"
           f"&response_type=code"
           f"&access_type=offline")
    return RedirectResponse(url)

@app.get("/auth/google/callback")
async def google_callback(code: str):
    async with httpx.AsyncClient() as c:
        r = await c.post("https://oauth2.googleapis.com/token", data={
            "code": code, "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_SECRET,
            "redirect_uri": f"{BASE_URL}/auth/google/callback",
            "grant_type": "authorization_code"
        })
        data = r.json()
    
    if "access_token" in data:
        conn = db()
        conn.execute("""INSERT OR REPLACE INTO tokens (service, access_token, refresh_token, expires_at, updated_at)
                       VALUES ('google', ?, ?, ?, CURRENT_TIMESTAMP)""",
                    (data["access_token"], data.get("refresh_token",""),
                     str(datetime.datetime.now() + datetime.timedelta(seconds=data.get("expires_in",3600)))))
        conn.commit()
        conn.close()
        return HTMLResponse("<h2 style='font-family:monospace;color:#00d4ff'>✅ Google conectado! Feche esta aba.</h2>")
    
    return HTMLResponse(f"<h2 style='color:red'>Erro: {data.get('error','unknown')}</h2>")

@app.get("/gmail/unread")
async def gmail_unread(limit: int = 5):
    token = get_google_token()
    if not token: raise HTTPException(401, "Google não conectado. Acesse /auth/google")
    
    async with httpx.AsyncClient() as c:
        r = await c.get(
            "https://gmail.googleapis.com/gmail/v1/users/me/messages",
            params={"q": "is:unread", "maxResults": limit},
            headers={"Authorization": f"Bearer {token}"}
        )
        data = r.json()
    
    if "messages" not in data: return {"emails": [], "total": 0}
    
    emails = []
    async with httpx.AsyncClient() as c:
        for msg in data["messages"][:limit]:
            r2 = await c.get(
                f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{msg['id']}",
                params={"format": "metadata", "metadataHeaders": ["Subject","From","Date"]},
                headers={"Authorization": f"Bearer {token}"}
            )
            m = r2.json()
            headers = {h["name"]: h["value"] for h in m.get("payload",{}).get("headers",[])}
            emails.append({"subject": headers.get("Subject",""), "from": headers.get("From",""), "date": headers.get("Date","")})
    
    return {"emails": emails, "total": data.get("resultSizeEstimate", 0)}

@app.get("/calendar/today")
async def calendar_today():
    token = get_google_token()
    if not token: raise HTTPException(401, "Google não conectado. Acesse /auth/google")
    
    now = datetime.datetime.utcnow()
    end = now.replace(hour=23, minute=59, second=59)
    
    async with httpx.AsyncClient() as c:
        r = await c.get(
            "https://www.googleapis.com/calendar/v3/calendars/primary/events",
            params={
                "timeMin": now.isoformat() + "Z",
                "timeMax": end.isoformat() + "Z",
                "orderBy": "startTime", "singleEvents": True
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        data = r.json()
    
    events = []
    for e in data.get("items", []):
        start = e.get("start", {}).get("dateTime") or e.get("start", {}).get("date","")
        events.append({"title": e.get("summary",""), "start": start, "location": e.get("location","")})
    
    return {"events": events, "date": now.strftime("%d/%m/%Y")}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
