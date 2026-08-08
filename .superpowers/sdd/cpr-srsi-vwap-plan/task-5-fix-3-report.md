# Task 5 fix round 3 report

## Outcome

The optional CPR AI child-process boundary now has only three line-scoped
Bandit acknowledgements: the runner's intentional `subprocess` import, its
fixed `shell=False` invocation, and the test-only `CompletedProcess` import.
The surrounding beginner-friendly documentation explains that callers and
model input cannot choose the fixed local interpreter or child script, shell
expansion is disabled, and the test import does not launch a command.

The existing runtime safeguards remain unchanged: fixed local argv, sanitized
allowlist environment, temporary working directory, 90-second timeout,
structured stdin/stdout, and host-side validation. No Bandit configuration,
exclusion, or repository-wide skip list was broadened.

## RED evidence

The exact CI Bandit command was run before the annotation change:

```text
python -m bandit -r . -q -x "./Backtest Outputs,./My Backtest Files (For Reference),./Dependencies/Shoonya API/NorenApi.py" --skip B101,B105,B110
```

It exited 1 with these three findings:

```text
B404 Signal Generators/CPR AI Agent/cpr_ai_codex_runner.py:12
B603 Signal Generators/CPR AI Agent/cpr_ai_codex_runner.py:92
B404 Signal Generators/CPR AI Agent/tests/test_cpr_ai_runtime.py:11
```

## GREEN evidence

```text
python -m bandit -r . -q -x "./Backtest Outputs,./My Backtest Files (For Reference),./Dependencies/Shoonya API/NorenApi.py" --skip B101,B105,B110
exit code 0; no findings

python -m pytest "Signal Generators/CPR AI Agent/tests" -q
54 passed, 1 warning in 2.78s

python -m ruff check "Signal Generators/CPR AI Agent/cpr_ai_codex_runner.py" "Signal Generators/CPR AI Agent/tests/test_cpr_ai_runtime.py"
All checks passed!

python -m py_compile "Signal Generators/CPR AI Agent/cpr_ai_codex_runner.py" "Signal Generators/CPR AI Agent/tests/test_cpr_ai_runtime.py"
exit code 0

python -m pre_commit run --all-files
ruff check, merge-conflict, YAML, large-file, and debug-statement hooks passed

git diff --check
exit code 0
```

The direct `pre-commit` executable was not on `PATH`; the equivalent installed
module command above completed successfully. The CPR AI suite's sole warning
is the existing `python-dateutil` `utcfromtimestamp()` deprecation warning.
Bandit still prints existing parser warnings from unrelated legacy `nosec`
comments (and an unrelated `algo.py` acknowledgement), but the exact CI
command exits 0 and reports no findings. No real Codex, broker, authenticated
smoke, or order boundary was called.
