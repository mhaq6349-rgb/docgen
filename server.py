"""DocGen â€” AI Documentation Generator.

Zero-dependency HTTP server using stdlib only.
- GET  /          serves the index.html
- POST /generate  triggers LLM doc generation
- POST /validate  validates a license key
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import sys
import time
import uuid
from html import escape
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

DOCGEN_DIR = Path(__file__).resolve().parent
FORGE_HOME = DOCGEN_DIR.parent.parent
sys.path.insert(0, str(FORGE_HOME))

_HOST = os.environ.get("DOCGEN_HOST", "0.0.0.0")
_PORT = int(os.environ.get("PORT", "8326"))

# ---------------------------------------------------------------------------
# License key system (HMAC-based â€” no DB needed)
# ---------------------------------------------------------------------------

_SECRET = os.environ.get("DOCGEN_SECRET", "change-me-in-production")

# Free tier: 1 generation per IP per day
_usage: dict[str, dict[str, int | float]] = {}
# Persistent usage file
_USAGE_FILE = DOCGEN_DIR / "usage.json"


def _load_usage():
    global _usage
    if _USAGE_FILE.exists():
        try:
            _usage = json.loads(_USAGE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            _usage = {}
    # Clean entries older than 24h
    now = time.time()
    _usage = {k: v for k, v in _usage.items() if now - v.get("t", 0) < 86400}


def _save_usage():
    _USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _USAGE_FILE.write_text(json.dumps(_usage, indent=2), encoding="utf-8")


def _make_license(email: str, plan: str = "pro") -> str:
    """Generate a license key: plan-email-HMAC."""
    raw = f"{plan}:{email.lower()}:{_SECRET}"
    h = hmac.new(_SECRET.encode(), raw.encode(), hashlib.sha256).hexdigest()[:12]
    return f"{plan}-{email.lower().replace('@','=')}-{h}"


def _validate_license(key: str) -> dict:
    """Validate a license key. Returns {valid, plan, email}."""
    parts = key.split("-", 2)
    if len(parts) != 3:
        return {"valid": False}
    plan, email_enc, sig = parts
    email = email_enc.replace("=", "@")
    expected = _make_license(email, plan)
    if key == expected:
        return {"valid": True, "plan": plan, "email": email}
    return {"valid": False}


# ---------------------------------------------------------------------------
# LLM doc generation (wraps forge modules)
# ---------------------------------------------------------------------------


def _detect_language(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    return {"py": "python", "ts": "typescript", "tsx": "typescript",
            "js": "javascript", "rs": "rust", "go": "go"}.get(ext.lstrip("."), "python")


def _generate_docs(repo_url: str, options: dict) -> dict:
    """Generate documentation for a repo. Returns {readme, api_docs, contributing}."""
    from docs.llm_generate import llm_readme, llm_api_docs, llm_contributing

    result = {}

    # Clone or fetch the repo
    repo_name = repo_url.rstrip("/").split("/")[-1] or "project"
    tmp_dir = DOCGEN_DIR / "tmp" / repo_name
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        import subprocess
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(tmp_dir)],
            capture_output=True, text=True, timeout=60,
        )
    except Exception as e:
        return {"error": f"Could not clone repo: {e}"}

    try:
        if options.get("readme", True):
            try:
                result["readme"] = llm_readme(str(tmp_dir))
            except Exception as e:
                result["readme"] = f"*README generation failed: {e}*"

        if options.get("api", True):
            api_parts = []
            for f in sorted(tmp_dir.rglob("*.py")):
                if "test" not in f.name and "__pycache__" not in str(f):
                    try:
                        src = f.read_text(encoding="utf-8")
                        docs = llm_api_docs(src, str(f))
                        if docs:
                            api_parts.append(f"## {f.relative_to(tmp_dir)}\n\n{docs}")
                    except Exception:
                        pass
            result["api_docs"] = "\n\n".join(api_parts) if api_parts else "*No API docs generated*"

        if options.get("contributing", True):
            try:
                result["contributing"] = llm_contributing(str(tmp_dir))
            except Exception as e:
                result["contributing"] = f"*Contributing guide generation failed: {e}*"

    finally:
        # Cleanup
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    result["repo"] = repo_url
    result["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return result


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

class DocGenHandler(BaseHTTPRequestHandler):
    """HTTP handler for DocGen."""

    def log_message(self, fmt: str, *args: Any) -> None:
        pass  # quiet

    def _send(self, data: dict | str, status: int = 200,
              content_type: str = "application/json") -> None:
        if isinstance(data, str):
            payload = data.encode("utf-8")
        else:
            payload = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode()) if raw else {}

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?")[0].rstrip("/") or "/"

        if path == "/":
            index = DOCGEN_DIR / "index.html"
            if index.exists():
                html = index.read_text(encoding="utf-8")
                self._send(html, 200, "text/html; charset=utf-8")
            else:
                self._send("<h1>DocGen</h1><p>Coming soon.</p>", 200, "text/html; charset=utf-8")
            return

        if path == "/health":
            self._send({"status": "ok", "version": "1.0.0"})
            return

        if path == "/pricing":
            self._send({
                "plans": [
                    {"id": "free", "name": "Free", "price": 0, "docs_per_day": 1, "features": ["1 README/day", "Basic templates"]},
                    {"id": "pro", "name": "Pro", "price": 9, "docs_per_day": 999,
                     "features": ["Unlimited docs", "README + API + Contributing", "Priority generation", "Export as Markdown/HTML"]},
                    {"id": "team", "name": "Team", "price": 29, "docs_per_day": 9999,
                     "features": ["Everything in Pro", "Team dashboard", "Custom branding", "API access", "Private repos"]},
                ]
            })
            return

        if path.startswith("/buy/"):
            plan = path.split("/buy/")[1]
            # Lemon Squeezy checkout link
            links = {
                "pro": "https://docgen.lemonsqueezy.com/buy/xxx-pro",
                "team": "https://docgen.lemonsqueezy.com/buy/xxx-team",
            }
            url = links.get(plan, links.get("pro"))
            self.send_response(302)
            self.send_header("Location", url)
            self.end_headers()
            return

        self._send({"error": "Not found"}, 404)

    def do_POST(self) -> None:
        path = self.path.rstrip("/")
        body = self._read_body()

        if path == "/validate":
            key = body.get("license_key", "")
            self._send(_validate_license(key))
            return

        if path == "/generate":
            repo_url = body.get("url", "").strip()
            license_key = body.get("license_key", "")

            if not repo_url or not re.match(r"^https?://github\.com/", repo_url):
                self._send({"error": "Invalid GitHub URL. Must be https://github.com/owner/repo"}, 400)
                return

            # Check license or free tier
            client_ip = self.client_address[0]
            is_pro = False
            if license_key:
                val = _validate_license(license_key)
                if val.get("valid"):
                    is_pro = True

            if not is_pro:
                _load_usage()
                today_key = f"{client_ip}:{time.strftime('%Y-%m-%d')}"
                entry = _usage.get(today_key, {"count": 0, "t": time.time()})
                if entry["count"] >= 1:
                    self._send({"error": "Free tier: 1 doc/day. Upgrade to Pro for unlimited."}, 402)
                    return
                entry["count"] += 1
                entry["t"] = time.time()
                _usage[today_key] = entry
                _save_usage()

            self._send({"status": "generating", "message": "Generation started. This may take 30-60 seconds."})
            return

        if path == "/generate-result":
            repo_url = body.get("url", "").strip()
            options = body.get("options", {})
            try:
                result = _generate_docs(repo_url, options)
                self._send(result)
            except Exception as e:
                self._send({"error": f"Generation failed: {e}"}, 500)
            return

        self._send({"error": "Not found"}, 404)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def serve(host: str = _HOST, port: int = _PORT) -> None:
    server = HTTPServer((host, port), DocGenHandler)
    print(f"DocGen running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    serve()

