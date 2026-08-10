# LLD — Configuration and drift detection

**Owns:** `Dependencies/env.example`, `Dependencies/.env` (gitignored),
`Dependencies/check_env_config.py`, and the `_env_*` / `_scaled_*` helpers in
the master file (~L396–510)
**Related ADR:** [0008 — a single `.env` as the only config source](../adr/0008-single-env-as-config-source.md)

---

## 1. Responsibility

One file decides everything about a run. No per-run flags, no profiles, no
environment-specific overrides. `python algo.py run` behaves entirely according
to `Dependencies/.env`.

---

## 2. The three places a setting lives

This is the source of every configuration bug in this repo, so it is worth
drawing:

```
   Dependencies/env.example        committed template  ── the ONLY discovery surface
   Dependencies/.env               gitignored, real    ── what actually runs
   in-code default in _env_*(…)    the fallback        ── silently governs when .env omits the key
```

A key present in the code and the template but **missing from `.env` is not an
error** — the runner just uses the in-code default. That is exactly what makes
it dangerous: an unseen default ends up governing a live-money run, and nothing
complains.

`python algo.py check-env` exists for that one failure mode.

---

## 3. Reading configuration

| Helper | Use for |
|---|---|
| `_env_str` / `_env_bool` / `_env_int` / `_env_float` | Ordinary knobs |
| `_scaled_int` / `_scaled_float` | **Size-bearing** knobs (`_LOTS`, `_MAX_LOTS`, `_RISK_BUDGET`, `_MAX_LOSS`) |
| `_strategy_size_multiplier(prefix)` | Resolve and validate `<PREFIX>_SIZE_MULTIPLIER` |

**Never use ad-hoc `os.getenv`.** Two reasons: the `check-env` audit and the CI
drift gate both find keys by walking the AST for `_env_*` calls, so a raw
`os.getenv` is invisible to both; and size knobs read without `_scaled_*` silently
ignore the multiplier. A drift-guard test fails if a new strategy reads a size
knob with the raw helpers.

Per-strategy knobs are namespaced `<PREFIX>_*`; the name→prefix map is
`STRATEGY_ENV_PREFIX`.

---

## 4. `algo.py check-env`

Read-only audit implemented in `Dependencies/check_env_config.py`. It reports:

- settings the code reads that are **missing from `.env`** (an unseen in-code
  default is in force)
- **mistyped or stale keys** — a typo means the setting you intended is not being
  applied at all
- knobs **missing from the template**

It exits non-zero on findings so it can gate a pre-flight script, and it prints
key **names only — never a value out of `.env`** — so its output is safe to paste
into an issue or share with a reviewer.

---

## 5. The CI drift gate

`Tests/Dependencies/test_repository_policy.py` imports the *same* helpers
(`audit`, `env_keys_read_by`, `source_files`) that the operator command uses, so
the gate and the tool can never disagree about what "documented" means.

`test_every_env_setting_the_code_reads_is_documented_in_env_example` fails the
build when a new `_env_*` key lands without an `env.example` entry.

**One direction only (code → template).** The reverse would flag the ~200
per-strategy `<PREFIX>_*` knobs that `_signal_gen_ops` builds from f-strings,
which are real settings the AST cannot see. The test also asserts that the AST
walk finds >300 keys — a sanity check, so that a renamed helper cannot make the
gate silently pass while checking nothing.

---

## 6. Secrets

- `.env` is gitignored. `env.example` holds **blank placeholders only**.
- Also gitignored: `shoonyakey.txt`, `Dependencies/gsheet_oauth_*.json`,
  `Dependencies/all_instrument*.csv`.
- Every `.env` value whose key looks sensitive and is ≥8 characters is fed to the
  root-logger redaction filter at startup — see
  [`risk-and-safety.md`](risk-and-safety.md) §8.

**Never commit a secret.** If one is committed, rotating it at the broker is the
fix; removing the commit is not.

---

## 7. Config-driven safety

Configuration is not just tuning here; several safety behaviours are decided at
read time:

| Setting | Fail-closed behaviour |
|---|---|
| `LIVE_BROKER` | Unknown value → live disabled entirely |
| `MARKET_DATA_SOURCE` | Anything but `WEBSOCKET` → REST |
| `<PREFIX>_SIZE_MULTIPLIER` | Malformed (`0`, `2.5`, `30`, `"two"`) → 1 for paper, **blocked from live** |
| `<PREFIX>_VIRTUAL_TRADING` | false → the thread never starts |
| `LIVE_TRADING_ENABLED` | Absent → false → paper |
| `GSHEET_ID`, `TELEGRAM_*` | Absent → safe no-op |

`_live_config_errors(...)` collects these at startup so a misconfiguration
surfaces before the first bar, not at the first order.

---

## 8. Adding a setting — checklist

1. Read it through the right helper (`_env_*`, or `_scaled_*` if it is
   size-bearing).
2. Add it to `Dependencies/env.example` with a blank/placeholder value and a
   comment explaining what it does. **CI fails without this.**
3. If it can be malformed, decide the fail-closed direction and add it to
   `_live_config_errors` if it affects live trading.
4. Run `python algo.py check-env` and confirm it is clean.
5. Document it in the relevant LLD if it changes a component's contract.
