"""DocGen — AI Documentation Generator.

Zero-dependency HTTP server using stdlib only.
- GET  /          serves the index.html
- POST /generate  triggers LLM doc generation
- POST /validate  validates a license key
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import re
import shutil
import sys
import time
import uuid
import zipfile
from html import escape
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

DOCGEN_DIR = Path(__file__).resolve().parent
FORGE_HOME = DOCGEN_DIR.parent.parent
sys.path.insert(0, str(FORGE_HOME))

_HOST = os.environ.get("DOCGEN_HOST", "0.0.0.0")
_PORT = int(os.environ.get("PORT", "8326"))

# ---------------------------------------------------------------------------
# License key system (HMAC-based — no DB needed)
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
# LLM doc generation (self-contained, direct OpenRouter API)
# ---------------------------------------------------------------------------

_OPENROUTER_KEY: str | None = None

# Fallback chain: primary first, then valid-but-often-rate-limited models.
_MODELS = [
    "tencent/hy3:free",
    "qwen/qwen3-coder:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
]

_SOURCE_EXT = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".rb",
    ".php", ".cs", ".cpp", ".cc", ".c", ".h", ".hpp", ".swift", ".kt",
    ".scala", ".m", ".mm", ".lua", ".dart", ".ex", ".exs", ".vue", ".svelte",
}
_SKIP_DIRS = {
    "node_modules", "dist", "build", ".git", "vendor", "__pycache__",
    ".venv", "venv", "env", "target", ".next", "out", "coverage",
    ".idea", ".vscode", "migrations", "third_party", "deps", ".tox",
    "site-packages", "bower_components", ".cache",
}
_ENTRY_HINTS = ("main", "index", "app", "lib", "__init__", "server",
                "cli", "core", "api", "mod", "init")
_LANG_MAP = {
    "py": "python", "ts": "typescript", "tsx": "tsx", "js": "javascript",
    "jsx": "jsx", "rs": "rust", "go": "go", "java": "java", "rb": "ruby",
    "php": "php", "cs": "csharp", "cpp": "cpp", "cc": "cpp", "c": "c",
    "h": "c", "hpp": "cpp", "swift": "swift", "kt": "kotlin",
    "scala": "scala", "lua": "lua", "dart": "dart", "ex": "elixir",
    "exs": "elixir", "vue": "vue", "svelte": "svelte", "m": "objc",
}


def _get_key() -> str:
    global _OPENROUTER_KEY
    if _OPENROUTER_KEY:
        return _OPENROUTER_KEY
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    if not key:
        try:
            auth = Path.home() / ".config" / "opencode" / "auth.json"
            if auth.exists():
                data = json.loads(auth.read_text())
                key = data.get("OPENROUTER_API_KEY", "") or data.get("openai_api_key", "")
        except (OSError, json.JSONDecodeError):
            pass
    _OPENROUTER_KEY = key
    return key


def _strip_fences(text: str) -> str:
    """Remove a single wrapping ```markdown ... ``` fence if the model added one."""
    t = text.strip()
    if t.startswith("```"):
        first_nl = t.find("\n")
        if first_nl != -1:
            t = t[first_nl + 1:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def _llm_ask(prompt: str, system: str = "", temperature: float = 0.3,
             max_tokens: int = 4096) -> str:
    key = _get_key()
    if not key:
        raise RuntimeError(
            "No API key configured. Set the OPENROUTER_API_KEY environment variable."
        )

    last_err: Exception | None = None
    for model in _MODELS:
        for attempt in range(3):
            body = json.dumps({
                "model": model,
                "messages": [
                    {"role": "system", "content": system or "You are a technical documentation expert. Output only clean Markdown, no preamble."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }).encode()
            req = Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {key}",
                    "HTTP-Referer": "https://docgen.pro",
                    "X-Title": "DocGen",
                },
            )
            try:
                resp = urlopen(req, timeout=90)
                data = json.loads(resp.read())
                content = data["choices"][0]["message"]["content"]
                if content and content.strip():
                    return content
                last_err = RuntimeError("empty response")
                break
            except HTTPError as e:
                last_err = e
                if e.code == 429:
                    time.sleep(2 * (attempt + 1))
                    continue
                break
            except Exception as e:
                last_err = e
                time.sleep(1)
                continue
    raise RuntimeError(f"All models failed. Last error: {last_err}")


def _detect_language(filepath: str) -> str:
    ext = Path(filepath).suffix.lower().lstrip(".")
    return _LANG_MAP.get(ext, "text")


def _read_first(tmp_dir: Path, pattern: str, n: int = 120) -> list[tuple[str, str]]:
    results = []
    for f in sorted(tmp_dir.glob(pattern)):
        if not f.is_file():
            continue
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
            content = "\n".join(lines[:n])
            if content.strip():
                results.append((f.name, content))
        except OSError:
            pass
    return results


def _collect_sources(tmp_dir: Path, limit: int = 6,
                     max_lines: int = 220) -> list[tuple[str, str]]:
    """Pick the most important source files across all languages."""
    candidates: list[Path] = []
    for f in tmp_dir.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() not in _SOURCE_EXT:
            continue
        if any(part in _SKIP_DIRS for part in f.parts):
            continue
        low = f.name.lower()
        if "test" in low or "spec" in low or low.endswith(".min.js"):
            continue
        candidates.append(f)

    def score(f: Path) -> int:
        s = 0
        stem = f.stem.lower()
        if any(stem == h or stem.startswith(h) for h in _ENTRY_HINTS):
            s -= 1000
        depth = len(f.relative_to(tmp_dir).parts)
        s += depth * 10
        try:
            size = f.stat().st_size
            if 200 < size < 40000:
                s -= min(size, 20000) // 2000
        except OSError:
            pass
        return s

    candidates.sort(key=score)
    picked: list[tuple[str, str]] = []
    for f in candidates[:limit]:
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
            src = "\n".join(lines[:max_lines])
            if src.strip():
                picked.append((str(f.relative_to(tmp_dir)).replace("\\", "/"), src))
        except OSError:
            pass
    return picked


def _generate_readme(tmp_dir: Path) -> str:
    context_parts = []
    project_name = tmp_dir.resolve().name

    for name, content in _read_first(tmp_dir, "README*"):
        context_parts.append(f"--- Existing README ({name}) ---\n{content}\n")
    for pattern in ("pyproject.toml", "package.json", "Cargo.toml",
                    "go.mod", "composer.json", "Gemfile", "pom.xml"):
        found = _read_first(tmp_dir, pattern, 80)
        if found:
            for name, content in found:
                context_parts.append(f"--- {name} ---\n{content}\n")
            break

    for rel, src in _collect_sources(tmp_dir, limit=4, max_lines=80):
        context_parts.append(f"--- {rel} (excerpt) ---\n{src}\n")

    context = "\n".join(context_parts) or f"(Repository '{project_name}' — limited context available.)"
    prompt = (
        f"Generate a professional, comprehensive README.md for the project '{project_name}'. "
        f"Include these sections: title, one-line description, badges placeholder, Features, "
        f"Installation, Usage (with code examples), Configuration, Project Structure, "
        f"Contributing, and License. Use proper Markdown. Do not invent a license if unknown.\n\n"
        f"Project context:\n{context}"
    )
    return _strip_fences(_llm_ask(prompt, temperature=0.4, max_tokens=4096))


def _generate_api_docs(tmp_dir: Path) -> str:
    sources = _collect_sources(tmp_dir, limit=6, max_lines=220)
    if not sources:
        return ("No source files found to document. The repository may contain only "
                "config, data, or documentation files.")

    blocks = []
    for rel, src in sources:
        lang = _detect_language(rel)
        blocks.append(f"### File: `{rel}`\n```{lang}\n{src}\n```")
    joined = "\n\n".join(blocks)

    prompt = (
        "Generate a comprehensive API reference in Markdown for the following source "
        "files. Start with a short '# API Reference' heading and a one-paragraph overview. "
        "Then for EACH file, add a '## `<path>`' section describing the file's purpose, "
        "followed by its public functions, classes, and methods — each with signature, "
        "parameters, return values, and a brief usage example where helpful. "
        "Be accurate to the code shown; do not invent APIs.\n\n"
        f"{joined}"
    )
    return _strip_fences(_llm_ask(prompt, temperature=0.3, max_tokens=6000))


def _generate_contributing(tmp_dir: Path) -> str:
    project_name = tmp_dir.resolve().name
    context_parts = []
    for name, content in _read_first(tmp_dir, "CONTRIBUTING*"):
        context_parts.append(f"--- Existing {name} ---\n{content}\n")
    for pattern in ("pyproject.toml", "package.json", "Cargo.toml", "go.mod", "Makefile"):
        found = _read_first(tmp_dir, pattern, 60)
        for name, content in found:
            context_parts.append(f"--- {name} ---\n{content}\n")

    tooling = []
    for probe, label in [("pytest.ini", "pytest"), ("tox.ini", "tox"),
                         (".pre-commit-config.yaml", "pre-commit"),
                         ("ruff.toml", "ruff"), (".eslintrc", "eslint"),
                         ("jest.config.js", "jest"), (".github", "GitHub Actions")]:
        if list(tmp_dir.glob(probe)) or (tmp_dir / probe).exists():
            tooling.append(label)

    context = "\n".join(context_parts)
    prompt = (
        f"Generate a CONTRIBUTING.md guide for the project '{project_name}'. "
        f"Include: Getting Started / dev environment setup, how to run tests, "
        f"code style guidelines, branching and commit conventions, the pull-request "
        f"process, and how to report issues. Use Markdown."
        + (f" Detected tooling: {', '.join(tooling)}." if tooling else "")
        + (f"\n\nProject context:\n{context}" if context else "")
    )
    return _strip_fences(_llm_ask(prompt, temperature=0.4, max_tokens=3072))


def _download_repo(repo_url: str, dst: Path) -> str:
    """Download a GitHub repo as zip using stdlib. Returns project name."""
    parts = repo_url.rstrip("/").split("/")
    owner, repo = parts[-2], parts[-1].replace(".git", "")

    data = None
    last_err: Exception | None = None
    for branch in ("main", "master", "develop"):
        zip_url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{branch}"
        try:
            resp = urlopen(Request(zip_url, headers={"User-Agent": "DocGen"}), timeout=45)
            data = resp.read()
            break
        except Exception as e:
            last_err = e
            continue
    if data is None:
        raise RuntimeError(f"repo not found or private ({last_err})")

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        root = zf.namelist()[0].rstrip("/")
        zf.extractall(dst.parent)
        extracted = dst.parent / root
        if extracted.exists():
            if dst.exists():
                shutil.rmtree(dst, ignore_errors=True)
            extracted.rename(dst)
    return repo


def _generate_docs(repo_url: str, options: dict) -> dict:
    """Generate documentation for a repo. Returns {readme, api_docs, contributing}."""
    result: dict[str, str] = {}

    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "") or "project"
    tmp_dir = DOCGEN_DIR / "tmp" / f"{repo_name}-{uuid.uuid4().hex[:8]}"

    try:
        _download_repo(repo_url, tmp_dir)
    except Exception as e:
        return {"error": f"Could not download repo: {e}"}

    try:
        want_readme = options.get("readme", True)
        want_api = options.get("api", True)
        want_contributing = options.get("contributing", True)

        if want_readme:
            try:
                result["readme"] = _generate_readme(tmp_dir)
            except Exception as e:
                result["readme"] = f"*README generation failed: {e}*"

        if want_api:
            try:
                result["api_docs"] = _generate_api_docs(tmp_dir)
            except Exception as e:
                result["api_docs"] = f"*API docs generation failed: {e}*"

        if want_contributing:
            try:
                result["contributing"] = _generate_contributing(tmp_dir)
            except Exception as e:
                result["contributing"] = f"*Contributing guide generation failed: {e}*"

    finally:
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
            client_ip = (self.headers.get("X-Forwarded-For", "")
                         .split(",")[0].strip() or self.client_address[0])
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
    server = ThreadingHTTPServer((host, port), DocGenHandler)
    server.daemon_threads = True
    print(f"DocGen running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    serve()
