# ADR-0013: Dismiss the five CodeQL alerts rather than change the flagged code

**Status:** Accepted
**Date:** 2026-08-31
**Deciders:** repository owner

## Context

GitHub code scanning reports **five open CodeQL alerts, all rated "high"**, against
`main`. They were reviewed one by one on 2026-08-31. **None is a vulnerability**,
and three of the five describe code whose removal would make this repository
measurably *less* safe.

| # | Rule | Location |
|---|---|---|
| 1 | `py/clear-text-logging-sensitive-data` | `Tests/Dependencies/test_secret_redaction.py:153` |
| 2 | `py/clear-text-logging-sensitive-data` | `Tests/Dependencies/test_secret_redaction.py:157` |
| 3 | `py/clear-text-logging-sensitive-data` | `Tests/Dependencies/test_secret_redaction.py:189` |
| 4 | `py/weak-sensitive-data-hashing` | `Dependencies/Shoonya API/NorenApi.py:229` |
| 5 | `py/weak-sensitive-data-hashing` | `Dependencies/Flattrade API/flattrade_execution.py:483` |

### Alerts 1–3: the redaction tests log a canary ON PURPOSE

Each flagged line logs a **canary** string — `CANARY-SUPER-SECRET`,
`CANARY-INSTALL-SECRET` — through a logger whose handler has `RedactingFilter`
installed, writing into an in-memory `io.StringIO`. The very next lines assert
`secret not in output` and `REDACTED in output`.

The logging call is not incidental to those tests; it **is** the test. It is the
only way to prove the filter scrubs a secret before it reaches a handler. Deleting
or defanging the call to satisfy the scanner would delete the coverage and leave
the redaction filter unverified.

That filter is not decorative. As `CLAUDE.md` records, `dhanhq`'s marketfeed puts
the live access token in its websocket URL, so a connect error would otherwise
write a working credential verbatim into a log file operators routinely share.
These three tests are what keep that from regressing silently.

The alerts are correct that a secret reaches a logging call. They cannot see that
the secret is fake, the sink is a memory buffer, and the assertion immediately
afterwards proves nothing escaped.

### Alert 4: Shoonya's wire protocol, in vendored code

`hashlib.sha256(password)` at `NorenApi.py:229` is what Shoonya's `/authorize`
endpoint requires on the wire. Two independent reasons not to change it:

- The digest format is **not ours to choose**. Substituting bcrypt/scrypt/argon2
  produces a value the broker rejects, breaking login outright.
- The file is a **vendored** third-party client (`CLAUDE.md`, Broker layer).
  Editing it silently forks the vendor's code and makes the next upgrade a manual
  merge.

The rule targets password **storage**, where an expensive KDF is the right answer
because an attacker who steals the store gets offline guesses. There is no password
store here: this is a transport digest computed in memory and sent immediately.

### Alert 5: not password hashing at all

`flattrade_execution.py:483` computes
`sha256(api_key + request_code + api_secret)`. That is Flattrade's documented
proof-of-possession for its token exchange, and the code already carries the
explanation:

> the API secret is never sent raw. Instead `sha256(api_key + request_code +
> api_secret)` proves we hold the secret while binding this exchange to THIS
> request code — a replayed digest is useless once the code expires.

CodeQL taints `raw_secret` as a "password" because of its name. The construction is
a keyed one-time proof, not a stored credential, and a slow KDF would be the wrong
primitive as well as a rejected one.

### Why dismissal, and not configuration

CodeQL here runs through GitHub's **default setup** (`state: configured`, no
workflow file, no config). Default setup does **not** read
`.github/codeql/codeql-config.yml`, so path filters and query exclusions are
unavailable. Adding such a file would be dead configuration that reads, to a future
maintainer, as protection that does not exist.

Dismissal with a recorded reason is therefore the supported remediation. It is also
the honest one: the finding is real as a pattern match and wrong as a conclusion,
which is exactly what "false positive" is for.

## Decision

1. **Change none of the five sites.**
2. **Dismiss all five alerts** with reasons recorded in GitHub: alerts 1–3 as *used
   in tests*, alerts 4–5 as *false positive*.
3. **Add a repository-policy guard** asserting the canary-logging tests still exist
   and still assert the secret is absent, so the "fix" this ADR rejects cannot be
   applied later by a well-meaning edit or an automated suggestion.
4. **Treat a recurrence as a dismissal, not a defect.** If these alerts reappear —
   after a re-scan, a CodeQL version bump, or a file move — re-dismiss them and
   point at this ADR. Do not patch the code to silence the scanner.

## Consequences

- The Security tab reads clean, and each dismissal carries a reason a reviewer can
  audit rather than an unexplained silence.
- The redaction coverage is now protected from both directions: the tests prove the
  filter works, and the policy guard proves the tests still exist.
- Alerts 4 and 5 will reappear if either broker file is substantially rewritten,
  since the flagged construction is inherent to both APIs. That is expected.
- The repository is **public**. Checked alongside this triage, and clean:
  `Dependencies/.env` was never committed, no credential/key/token files are
  tracked, and the only secret-shaped literal in tracked code is
  `"not-a-real-token"` in a broker test fixture.

## Action items

- [x] Review all five alerts against the source.
- [x] Record the triage here and link it from `docs/README.md`.
- [x] Add `test_secret_redaction_canary_coverage_survives_codeql_pressure` to
      `Tests/Dependencies/test_repository_policy.py`.
- [ ] Operator dismisses the five alerts in the Security tab with the reasons above.
