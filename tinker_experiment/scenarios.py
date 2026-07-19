"""Programmatic fact-sheet construction for synthetic transcripts.

Each fact sheet: a scenario (stack) + 6-8 facts, at least half of which are
default-traps (the value contradicts the popular default). Facts are chosen
here, not by the LLM, so verification is mechanical substring matching.

Contamination control: nothing from the original eval — no Express, SQLite,
bcryptjs, node:test, or task-management scenarios (BANNED_TERMS enforced at
generation time too).
"""

from __future__ import annotations

import random

BANNED_TERMS = ["express", "sqlite", "bcryptjs", "node:test", "task management", "task-management"]

# (framework, language, database, domain) — domains deliberately far from
# task management APIs.
SCENARIOS = [
    ("FastAPI", "Python", "Postgres", "recipe ingredient inventory service"),
    ("Flask", "Python", "MySQL", "conference talk submission portal"),
    ("Litestar", "Python", "Postgres", "birdwatching sighting logger"),
    ("Django", "Python", "Postgres", "gym class booking system"),
    ("Fastify", "TypeScript", "Postgres", "podcast episode metadata service"),
    ("Hono", "TypeScript", "MariaDB", "plant nursery stock tracker"),
    ("NestJS", "TypeScript", "MongoDB", "vinyl record collection catalog"),
    ("Koa", "JavaScript", "MongoDB", "ski lift wait-time tracker"),
    ("Go + chi", "Go", "Postgres", "beehive telemetry collector"),
    ("Go + echo", "Go", "MySQL", "food truck location broadcaster"),
    ("Rails", "Ruby", "Postgres", "community tool-lending library"),
    ("Sinatra", "Ruby", "MariaDB", "escape room booking backend"),
    ("Phoenix", "Elixir", "Postgres", "houseplant watering scheduler"),
    ("Axum", "Rust", "Postgres", "climbing route beta archive"),
    ("Spring Boot", "Java", "MySQL", "aquarium water quality monitor"),
]

# category -> language-compat -> list of (value, popular_default) trap pairs.
# None in compat = any language.
TRAP_PAIRS = {
    "password_hashing": {
        "Python": [("argon2-cffi", "bcrypt"), ("scrypt via hashlib", "bcrypt")],
        "TypeScript": [("argon2", "bcrypt"), ("scrypt from node:crypto", "bcrypt")],
        "JavaScript": [("argon2", "bcrypt")],
        "Go": [("argon2id via x/crypto", "bcrypt")],
        "Ruby": [("argon2 gem", "bcrypt")],
        "Elixir": [("argon2_elixir", "bcrypt")],
        "Rust": [("argon2 crate", "bcrypt")],
        "Java": [("argon2-jvm", "BCrypt")],
    },
    "test_framework": {
        "Python": [("ward", "pytest"), ("nose2", "pytest")],
        "TypeScript": [("vitest", "jest"), ("node:assert with tsx", "jest")],
        "JavaScript": [("vitest", "jest"), ("tape", "jest")],
        "Go": [("gotestsum", "go test alone")],
        "Ruby": [("minitest", "RSpec")],
        "Elixir": [("ex_unit with mneme", "plain ExUnit")],
        "Rust": [("nextest", "cargo test")],
        "Java": [("TestNG", "JUnit")],
    },
    "package_manager": {
        "Python": [("uv", "pip"), ("pdm", "pip"), ("hatch", "pip")],
        "TypeScript": [("pnpm", "npm"), ("bun", "npm")],
        "JavaScript": [("pnpm", "npm"), ("yarn berry", "npm")],
        "Ruby": [("bundler with local vendoring", "system gems")],
        "Java": [("Gradle", "Maven")],
    },
    "id_scheme": {
        None: [("ULID", "UUID"), ("nanoid", "UUID"), ("KSUID", "UUID"), ("Snowflake IDs", "UUID")],
    },
    "linter": {
        "Python": [("ruff", "flake8"), ("pyflakes only", "flake8")],
        "TypeScript": [("biome", "eslint"), ("oxlint", "eslint")],
        "JavaScript": [("biome", "eslint")],
        "Go": [("staticcheck", "golangci-lint")],
        "Ruby": [("standardrb", "rubocop")],
        "Rust": [("clippy with pedantic", "default clippy")],
    },
    "cache": {
        None: [("valkey", "redis"), ("dragonfly", "redis"), ("keydb", "redis")],
    },
    "http_client": {
        "Python": [("httpx", "requests"), ("niquests", "requests")],
        "TypeScript": [("ky", "axios"), ("got", "axios"), ("undici fetch", "axios")],
        "JavaScript": [("ky", "axios"), ("got", "axios")],
        "Go": [("resty", "net/http alone")],
        "Ruby": [("faraday", "net/http")],
        "Java": [("OkHttp", "Apache HttpClient")],
    },
    "logging": {
        "Python": [("loguru", "logging module"), ("structlog", "logging module")],
        "TypeScript": [("pino", "winston")],
        "JavaScript": [("pino", "winston")],
        "Go": [("zerolog", "logrus"), ("slog", "logrus")],
        "Ruby": [("semantic_logger", "Logger")],
        "Rust": [("tracing", "log + env_logger")],
        "Java": [("Log4j2", "Logback")],
    },
}

# Non-trap fact categories: concrete specifics for the transcript.
TABLE_NAMES = ["inventory_items", "submissions", "sightings", "bookings", "episodes",
               "stock_lots", "records", "lift_status", "hives", "locations",
               "loans", "rooms", "plants", "routes", "readings"]
ENV_VARS = ["APP_SIGNING_SECRET", "SVC_DB_DSN", "CACHE_ENDPOINT", "API_RATE_CAP",
            "WEBHOOK_TARGET", "METRICS_TOKEN"]
MAIN_FILES = ["app_main", "server_core", "svc_entry", "boot", "gateway"]

QUESTION_TEMPLATES = {
    "password_hashing": "Which library is used for password hashing?",
    "test_framework": "Which test framework does the project use?",
    "package_manager": "Which package manager is used to install dependencies?",
    "id_scheme": "What scheme is used for generating record IDs?",
    "linter": "Which linter is configured for the project?",
    "cache": "What technology is used as the caching layer?",
    "http_client": "Which HTTP client library is used for outbound requests?",
    "logging": "Which logging library does the project use?",
    "port": "What port does the server listen on?",
    "main_file": "What is the name of the main entry-point file?",
    "table": "What is the name of the primary database table?",
    "env_var": "What environment variable holds the signing/connection secret?",
}


def _compat_pairs(category: str, language: str) -> list[tuple[str, str]]:
    table = TRAP_PAIRS[category]
    if None in table:
        return table[None]
    return table.get(language, [])


def build_fact_sheet(index: int, rng: random.Random) -> dict:
    framework, language, database, domain = SCENARIOS[index % len(SCENARIOS)]

    ext = {"Python": ".py", "TypeScript": ".ts", "JavaScript": ".js", "Go": ".go",
           "Ruby": ".rb", "Elixir": ".ex", "Rust": ".rs", "Java": ".java"}[language]

    facts = []

    # Trap facts: sample 4 applicable categories.
    trap_categories = [c for c in TRAP_PAIRS if _compat_pairs(c, language)]
    rng.shuffle(trap_categories)
    for category in trap_categories[:4]:
        value, default = rng.choice(_compat_pairs(category, language))
        facts.append({
            "category": category,
            "question": QUESTION_TEMPLATES[category],
            "answer": value,
            "is_trap": True,
            "popular_default": default,
        })

    # Non-trap specifics: 3 concrete values.
    port = rng.choice([3172, 4817, 5203, 6390, 7145, 8531, 9276])
    main_file = rng.choice(MAIN_FILES) + ext
    table = rng.choice(TABLE_NAMES)
    env_var = rng.choice(ENV_VARS)
    non_traps = [
        {"category": "port", "question": QUESTION_TEMPLATES["port"], "answer": str(port)},
        {"category": "main_file", "question": QUESTION_TEMPLATES["main_file"], "answer": main_file},
        {"category": "table", "question": QUESTION_TEMPLATES["table"], "answer": table},
        {"category": "env_var", "question": QUESTION_TEMPLATES["env_var"], "answer": env_var},
    ]
    rng.shuffle(non_traps)
    for f in non_traps[:3]:
        f["is_trap"] = False
        f["popular_default"] = None
        facts.append(f)

    return {
        "id": f"syn{index:03d}",
        "framework": framework,
        "language": language,
        "database": database,
        "domain": domain,
        "facts": facts,
    }
