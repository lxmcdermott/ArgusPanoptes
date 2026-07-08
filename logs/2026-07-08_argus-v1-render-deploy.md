# Argus Panoptes — Session Change Log

**Date:** 2026-07-08  
**Scope:** Production deployment of the NiceGUI operator dashboard to **Render**
(`https://arguspanoptes.onrender.com`). Includes Cursor MCP integration for
Render, diagnosis and fix of a failed first deploy (`status 127`), Render
service reconfiguration, code changes for PaaS port binding, and verification
of auto-deploy on `main`.  
**Goal:** host the dashboard publicly with push-to-deploy from GitHub.  
**Environment:** Windows 11 / PowerShell (local dev); Render web service (Docker,
`python:3.11-slim`, Ohio region, Standard plan).  
**Regression gate:** no application logic changes beyond dashboard host/port
resolution helpers; `docker-compose.yml` unchanged (still overrides CMD per
service).

**Prior context:** `deployment/Dockerfile` and `docker-compose.yml` were wired
during the NiceGUI dashboard session (`logs/2026-07-07_argus-v1-nicegui-dashboard.md`)
for local `api` + `dashboard` stacks. The Dockerfile default `CMD` pointed at
the FastAPI service (`uvicorn app.main:app`). The user created a Render web
service manually and hit a startup crash on the first deploy.

---

## 1. Problem statement (first deploy failure)

### Symptom

Render deploy built the Docker image successfully but the instance exited
immediately at runtime:

```
==> Deploying...
==> Setting WEB_CONCURRENCY=1 by default, based on available CPUs in the instance
/bin/sh: 1: python -m app.nicegui_dashboard: not found
==> Exited with status 127
==> Instance srv-d97bctd7vvec73fp2jig-p2lqn restarted
```

Exit code **127** = command not found.

### Root causes (two independent issues)

| Issue | Detail |
| --- | --- |
| **Broken Render `dockerCommand`** | The service had a custom Docker Command override: `/bin/sh -c "python -m app.nicegui_dashboard"`. Render/shell quoting caused the **entire string** `python -m app.nicegui_dashboard` to be treated as a single executable name (hence the colon-suffixed `not found` message from `/bin/sh`). |
| **Port / host mismatch** | NiceGUI defaulted to `127.0.0.1:8080`. Render injects `PORT=10000` and requires binding to `0.0.0.0` on that port. Even with a correct start command, the health check would have failed without honoring `PORT`. |

### Secondary observation

The Dockerfile `CMD` at the time of the first deploy still targeted the API:

```dockerfile
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

The Render service was configured to run the **dashboard**, not the API — so the
Dockerfile default and the Render intent were misaligned. Local development
remains correct because `docker-compose.yml` overrides `command` per service.

---

## 2. Cursor ↔ Render MCP setup

Before fixing the deploy, the Render MCP server was connected in Cursor so the
agent could list services, read logs, and update environment variables.

### Configuration file

Created `~/.cursor/mcp.json` (Windows: `C:\Users\logan\.cursor\mcp.json`):

```json
{
  "mcpServers": {
    "render": {
      "url": "https://mcp.render.com/mcp",
      "headers": {
        "Authorization": "Bearer <RENDER_API_KEY>"
      }
    }
  }
}
```

### MCP capabilities used in this session

| Tool | Purpose |
| --- | --- |
| `list_workspaces` | Auto-selected sole workspace (`My Workspace`) |
| `list_services` | Found `ArgusPanoptes` web service |
| `get_service` | Inspected `dockerCommand`, Dockerfile path, ports, auto-deploy |
| `list_logs` | Diagnosed build + runtime failure |
| `list_deploys` | Tracked deploy queue and final `live` status |
| `update_environment_variables` | Set `ARGUS_DASHBOARD_HOST=0.0.0.0` |

### MCP limitations encountered

Per [Render MCP docs](https://render.com/docs/mcp-server), the hosted MCP server
**cannot** modify service start commands or trigger deploys directly (except
indirectly when env-var updates auto-trigger a deploy). Clearing the broken
`dockerCommand` required the **Render REST API**:

```
PATCH https://api.render.com/v1/services/srv-d97bctd7vvec73fp2jig
Body: { "serviceDetails": { "envSpecificDetails": { "dockerCommand": "" } } }
```

> **Security note:** the API key was shared in chat during setup. Rotate it in
> Render Dashboard → Account Settings → API Keys if the conversation may have
> been exposed. Prefer `${env:RENDER_API_KEY}` in `mcp.json` over hard-coding.

---

## 3. Render service inventory (as configured)

| Field | Value |
| --- | --- |
| **Service name** | ArgusPanoptes |
| **Service ID** | `srv-d97bctd7vvec73fp2jig` |
| **URL** | https://arguspanoptes.onrender.com |
| **Dashboard** | https://dashboard.render.com/web/srv-d97bctd7vvec73fp2jig |
| **Repo** | https://github.com/lxmcdermott/ArgusPanoptes |
| **Branch** | `main` |
| **Runtime** | Docker |
| **Dockerfile** | `deployment/Dockerfile` |
| **Docker context** | `.` (repo root) |
| **Plan** | Standard |
| **Region** | Ohio |
| **Open port** | 10000 (Render-assigned; `PORT` env var) |
| **Health check** | `/` |
| **Persistent disk** | 5 GB at `/var/log/argus` |
| **Auto Deploy** | `yes`, trigger: `commit` on `main` |

### Deploy history (this session)

| Deploy ID | Commit | Trigger | Result |
| --- | --- | --- | --- |
| `dep-d97bctt7vvec73fp2l5g` | `84f1caf` (docs) | manual | `update_failed` — broken `dockerCommand` |
| `dep-d97bkv8k1i2s73d9t3e0` | `84f1caf` (docs) | API (env-var change) | `update_failed` — same start-command issue |
| `dep-d97bl5af1k9s73baave0` | `8d335ff` (fix) | `new_commit` | **`live`** ✅ |

---

## 4. Code and configuration changes

### 4.1 `deployment/Dockerfile` — default CMD → dashboard

**Before:**

```dockerfile
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**After:**

```dockerfile
# Default entrypoint is the NiceGUI operator dashboard (Render / single-service
# deploy). docker-compose overrides this for the `api` service; see
# deployment/docker-compose.yml.
CMD ["python", "-m", "app.nicegui_dashboard"]
```

**Rationale:**

- Render runs a **single** web service hosting the operator UI.
- Exec-form `CMD` avoids shell-quoting pitfalls that caused the original failure.
- `docker-compose.yml` is unchanged — the `api` service still overrides with
  `uvicorn app.main:app --host 0.0.0.0 --port 8000`.

### 4.2 `app/nicegui_dashboard.py` — PaaS port / host resolution

Added two helpers and wired them into `main()`:

```python
def _dashboard_host() -> str:
    if "ARGUS_DASHBOARD_HOST" in os.environ:
        return os.environ["ARGUS_DASHBOARD_HOST"]
    # PaaS hosts (e.g. Render) set PORT; bind externally when present.
    if os.environ.get("PORT"):
        return "0.0.0.0"
    return "127.0.0.1"


def _dashboard_port() -> int:
    for key in ("PORT", "ARGUS_DASHBOARD_PORT"):
        if key in os.environ:
            return int(os.environ[key])
    return 8080
```

**Resolution order:**

| Setting | Priority |
| --- | --- |
| **Host** | `ARGUS_DASHBOARD_HOST` → `0.0.0.0` if `PORT` set → `127.0.0.1` |
| **Port** | `PORT` (Render) → `ARGUS_DASHBOARD_PORT` (compose) → `8080` |

Local `docker compose` behavior is preserved via `ARGUS_DASHBOARD_PORT=8080` in
`deployment/docker-compose.yml`. Render injects `PORT=10000` automatically.

### 4.3 Render service changes (via API + MCP)

| Change | Method | Value |
| --- | --- | --- |
| Clear broken start command | Render REST API `PATCH` | `dockerCommand: ""` (use Dockerfile `CMD`) |
| Bind host explicitly | MCP `update_environment_variables` | `ARGUS_DASHBOARD_HOST=0.0.0.0` |

No `render.yaml` Blueprint was added — the service was created manually in the
Render Dashboard and is managed there plus via git auto-deploy.

---

## 5. Git commit and successful deploy

### Commit

```
8d335ff — Fix Render deploy: use Dockerfile CMD and bind to PORT.
```

Pushed to `origin/main`. Auto-deploy picked up the commit (`trigger: new_commit`).

### Successful runtime logs

```
NiceGUI ready to go on http://localhost:10000, and http://10.28.96.181:10000
==> Your service is live 🎉
==> Available at your primary URL https://arguspanoptes.onrender.com
```

Deploy `dep-d97bl5af1k9s73baave0` reached status **`live`** at
`2026-07-08T21:09:40Z`.

### Build notes

- Full Docker rebuild on the fix commit (~3 min image export + push) because
  dependency layers were not all cached on the fresh build path.
- Image installs `.[ml,dl,app,dashboard-nicegui]` including **PyTorch** and CUDA
  wheels — expect **multi-minute** deploy times on every rebuild until layer
  caching is effective.

---

## 6. Auto-deploy behavior (ongoing)

Pushes to **`main`** on `https://github.com/lxmcdermott/ArgusPanoptes` will
automatically:

1. Pull the new commit
2. Rebuild the Docker image from `deployment/Dockerfile`
3. Redeploy to https://arguspanoptes.onrender.com

**What git push does *not* change:** Render Dashboard settings (env vars, instance
size, custom commands, disk mounts, region). Those persist until edited in Render
or via API/MCP.

**What git push *does* change:** application code, Dockerfile, dependencies
(`pyproject.toml` / `requirements.txt`), and anything copied into the image via
`COPY . .`.

---

## 7. Architecture on Render (single-service model)

```
GitHub (main) ──webhook──▶ Render build (Dockerfile)
                              │
                              ▼
                    python:3.11-slim image
                    pip install -e ".[ml,dl,app,dashboard-nicegui]"
                              │
                              ▼
              CMD: python -m app.nicegui_dashboard
              bind: 0.0.0.0:$PORT (10000)
                              │
                              ▼
              https://arguspanoptes.onrender.com
              (NiceGUI + in-process StreamingPerceptor, standalone mode)
```

**Not deployed on Render (yet):** the FastAPI `api` service from
`docker-compose.yml`. The dashboard runs in **standalone (direct)** mode with an
in-process `StreamingPerceptor` — same as local `python -m app.nicegui_dashboard`
without `ARGUS_DASHBOARD_USE_API=1`. To add API-mode on Render, a second web
service or a combined entrypoint would be needed.

---

## 8. Operational runbook

### Redeploy manually

- **Automatic:** push to `main`
- **Dashboard:** Render → ArgusPanoptes → Manual Deploy
- **API:** `POST https://api.render.com/v1/services/srv-d97bctd7vvec73fp2jig/deploys`

### View logs

```text
Render Dashboard → ArgusPanoptes → Logs
```

Or via Cursor MCP: `list_logs` with `resource: ["srv-d97bctd7vvec73fp2jig"]`.

### Local parity commands

```bash
# Dashboard only (matches Render)
docker compose -f deployment/docker-compose.yml up dashboard

# Full stack (API + dashboard in API mode)
docker compose -f deployment/docker-compose.yml up api dashboard
```

### If deploy fails with `status 127` again

1. Check Render **Settings → Docker Command** — should be **empty** (use
   Dockerfile `CMD`).
2. Confirm `CMD` uses exec form: `["python", "-m", "app.nicegui_dashboard"]`.
3. Confirm logs show NiceGUI binding to `$PORT`, not `8080` on `127.0.0.1`.

### Recommended follow-ups (not done this session)

| Item | Why |
| --- | --- |
| Add `render.yaml` Blueprint | Infrastructure-as-code; reproducible service config |
| Slim Docker image (CPU-only torch / multi-stage) | Faster deploys; smaller image |
| Deploy FastAPI as second Render service | Enable dashboard API mode against a real backend |
| Rotate Render API key | Key was shared during MCP setup in chat |
| Add deploy status badge to `README.md` | Visibility for operators |

---

## 9. Files touched

| File | Change |
| --- | --- |
| `deployment/Dockerfile` | Default `CMD` → NiceGUI dashboard (exec form) |
| `app/nicegui_dashboard.py` | `_dashboard_host()`, `_dashboard_port()` for PaaS |
| `~/.cursor/mcp.json` | Render MCP server config (user home, not in repo) |
| `logs/2026-07-08_argus-v1-render-deploy.md` | This log |

**Not modified:** `deployment/docker-compose.yml`, `app/main.py`, `pyproject.toml`,
model artifacts, sensors, dsp, models.

---

## 10. Summary

| Before | After |
| --- | --- |
| First deploy crashed (`status 127`) | Service **live** at https://arguspanoptes.onrender.com |
| Custom `dockerCommand` with broken shell quoting | `dockerCommand` cleared; Dockerfile exec `CMD` |
| Dashboard bound `127.0.0.1:8080` | Honors Render `PORT` + `0.0.0.0` binding |
| No Cursor ↔ Render integration | Render MCP configured; workspace + logs accessible |
| Manual-only deploy expectation | Auto-deploy on every `main` push enabled |

The Argus Panoptes operator dashboard is now publicly hosted on Render with
continuous deployment from GitHub `main`.
