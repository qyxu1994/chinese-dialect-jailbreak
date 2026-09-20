"""Configuration for API targets, dialect registry, and local-model paths.

API keys are read from environment variables. On import this module also
loads a project-root `.env` file (if present) into `os.environ` for any keys
not already set in the real shell environment — real env vars always win.

Set whichever endpoints you intend to call:
    OPENAI_API_KEY         # gpt-4o (target / judge / attack / translate)
    DEEPSEEK_API_KEY       # deepseek-reasoner, deepseek-chat
    ANTHROPIC_API_KEY      # claude-sonnet-4-6 via Anthropic OpenAI-compat
    OPENROUTER_API_KEY     # qwen-max via OpenRouter (qwen/qwen-max)
"""

import os
from openai import OpenAI


def _load_env_file(path: str) -> None:
    """Minimal `.env` loader. Lines like `KEY=value`; `#` comments allowed.

    Real shell env vars take precedence: a key already in os.environ is left
    alone. We avoid pulling in python-dotenv to keep deps minimal.
    """
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            # Strip surrounding quotes if present.
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


# Load .env from project root (one level above this file's directory).
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_load_env_file(os.path.join(_PROJECT_ROOT, ".env"))


# ── derived API keys ──────────────────────────────────────────────────────────
# Defined after _load_env_file so callers that import these constants directly
# (translate.py uses API_SECRET_KEY + BASE_URL) get the correct value.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Legacy alias retained for translate.py — its `openai_client` is actually a
# DeepSeek client (BASE_URL points at DeepSeek). Renaming would touch every
# call site; keeping the alias is the smallest correct change.
API_SECRET_KEY = DEEPSEEK_API_KEY
BASE_URL = "https://api.deepseek.com"

# ── local-model knobs (unchanged) ─────────────────────────────────────────────
LOCAL_MODEL_PATH = ""
DEVICE = "cuda"

# ── Ollama ────────────────────────────────────────────────────────────────────
BASE_URL_ollama = ""
OLLAMA_MODEL = "llama3:8b"


# ── target / judge / attack model registry ───────────────────────────────────
# Each entry describes one model endpoint. `roles` lists the roles a given
# entry is valid for (a model can serve as target AND judge if it supports
# both), so the same registry covers --target_model and --judge_model.
#
# All entries use `client: "openai-compat"` — every supported endpoint is
# reachable through the OpenAI Python SDK with the right base_url override.
# Anthropic via its native OpenAI-compat layer; Qwen via OpenRouter.
#
# `api_key_env` names the env var to read at client-construction time; values
# come from the shell environment or `.env` (loaded above). No keys live in
# this file.
TARGETS = {
    "gpt-4o": {
        "client": "openai-compat",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model_id": "gpt-4o",
        "roles": ("target", "judge", "attack", "translate"),
    },
    "deepseek-reasoner": {
        "client": "openai-compat",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "model_id": "deepseek-reasoner",
        "roles": ("target",),
    },
    "deepseek-chat": {
        "client": "openai-compat",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "model_id": "deepseek-chat",
        "roles": ("target", "attack", "translate"),
    },
    # Qwen via OpenRouter — vendor-neutral router with an OpenAI-compatible
    # surface. Model IDs are namespaced like `qwen/qwen-max`. Swap the
    # model_id string here if a newer Qwen variant is preferred.
    "qwen-max": {
        "client": "openai-compat",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model_id": "qwen/qwen3.6-max-preview",
        "roles": ("target", "judge"),
    },
    # Anthropic's native OpenAI-compat endpoint (https://api.anthropic.com/v1/)
    # accepts the OpenAI Python SDK with the right base_url + API key.
    "claude-sonnet-4-6": {
        "client": "openai-compat",
        "base_url": "https://api.anthropic.com/v1/",
        "api_key_env": "ANTHROPIC_API_KEY",
        "model_id": "claude-sonnet-4-6",
        "roles": ("target", "judge"),
    },
}


def _resolve_api_key(spec: dict) -> str:
    """Read API key for a registry entry from os.environ."""
    return os.environ.get(spec["api_key_env"], "")


def get_target_client(model_name: str, role: str = "target"):
    """Return an OpenAI-compatible client for the named model.

    The same registry is used for any role (target, judge, attack, translate);
    `role` is validated against the entry's `roles` tuple to catch typos like
    using `deepseek-reasoner` (target-only) as a judge.
    """
    if model_name not in TARGETS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Registered models: {sorted(TARGETS.keys())}"
        )
    spec = TARGETS[model_name]
    if role not in spec["roles"]:
        raise ValueError(
            f"Model '{model_name}' is not valid for role '{role}'. "
            f"Allowed roles for this model: {spec['roles']}"
        )
    if spec["client"] == "openai-compat":
        # Finite per-request timeout + bounded retries so a hung endpoint raises
        # a catchable error instead of blocking the run forever. gen.py counts
        # such errors and skips the prompt (resumable on the next pass) rather
        # than wedging. Override via env if a slow reasoning model needs longer.
        timeout = float(os.environ.get("LLM_REQUEST_TIMEOUT", "60"))
        max_retries = int(os.environ.get("LLM_MAX_RETRIES", "2"))
        return OpenAI(
            api_key=_resolve_api_key(spec),
            base_url=spec["base_url"],
            timeout=timeout,
            max_retries=max_retries,
        )
    raise ValueError(f"Unknown client kind '{spec['client']}' for {model_name}")


def get_model_id(model_name: str) -> str:
    """Return the literal `model` string to pass to the client."""
    if model_name not in TARGETS:
        raise ValueError(f"Unknown model '{model_name}'")
    return TARGETS[model_name]["model_id"]


# ── dialect registry ──────────────────────────────────────────────────────────
# To add a new dialect: add one entry here, then create the corresponding
# file under dialects/. The FOA optimizer and translation pipeline pick up
# the new dialect automatically without changes to gen.py.
#
# Fields:
#   module    — importable path to the dialects/ file; None if not yet
#               implemented. Chat_template lives in gen.py for classical_chinese
#               (historical reason); all others live in their dialect module.
#   template  — name of the Chat_template variable inside that module.
#   display_name — human-readable label used in log headers and warnings.
#   implemented  — False causes a clear error rather than a silent wrong run.
SUPPORTED_DIALECTS = {
    "classical_chinese": {
        "module": None,
        "template": "Chat_template",
        "display_name": "Classical Chinese (文言文)",
        "implemented": True,
    },
    "shanghainese": {
        "module": "dialects.shanghainese",
        "template": "Chat_template_shanghainese",
        "display_name": "Shanghainese / Wu dialect (沪语)",
        "implemented": True,
    },
    "cantonese": {
        "module": "dialects.cantonese",
        "template": "Chat_template_cantonese",
        "display_name": "Cantonese (粤语)",
        "implemented": True,
    },
    "hokkien": {
        "module": None,
        "template": None,
        "display_name": "Hokkien (闽南语)",
        "implemented": False,
    },
    # Culture-neutral pseudo-dialect used by foa_generic_* ablation conditions.
    # Template resolution is handled by template_kind in ablation_dimensions.yaml;
    # the "template" field here is unused for generic conditions.
    "generic": {
        "module": "dialects.generic",
        "template": None,
        "display_name": "Generic (culture-neutral control)",
        "implemented": True,
    },
}
