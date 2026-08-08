# Task 5 fix round 2 report

## Outcome

Stale CPR AI responses can no longer overwrite accepted-regime memory when an
open position closes or is replaced during inference. The worker validates the
frozen position and CPR trade-state identities, plus the position's active
state, before persisting `outcome.accepted_regime` or processing the response.
The mandatory proposal/outcome audit is preserved and labels this case
`STALE_POSITION_RESPONSE` before the worker returns without execution.

Flat-start decisions and still-current open-position decisions retain their
normal regime-memory behavior. The fix-round-1 mechanical recheck, entry
exposure gates, market-quality gates, and the direct risk-reducing EXIT path
were not reordered.

## RED evidence

The regression extended
`test_stale_scale_in_cannot_apply_to_a_replacement_position` with literal
`SIDEWAYS` prior memory and a stale `TRENDING` outcome. It preserves the
no-order/no-scale assertions and also checks that the next frozen context still
contains the literal prior regime.

Command:

```text
python -m unittest test_nifty_multi_strategy_master.TestCPRAIWorkerFoundation.test_stale_scale_in_cannot_apply_to_a_replacement_position
```

Failure before the production edit:

```text
FAIL: test_stale_scale_in_cannot_apply_to_a_replacement_position
AssertionError: 'TRENDING' != 'SIDEWAYS'
Ran 1 test in 0.039s
FAILED (failures=1)
```

This proved the stale response currently changed `_prior_accepted_regime`,
which would feed `TRENDING` into the next `_latest_frozen_context()` instead of
the required prior literal `SIDEWAYS`.

An audit-contract review then added a second RED expectation before revising
the production flow. Against the first minimal fix, the same focused command
failed with:

```text
AssertionError: Expected 'write' to have been called once. Called 0 times.
Ran 1 test in 0.017s
FAILED (failures=1)
```

This proved that an early stale-response return would silently skip the
mandatory proposal/outcome audit. The final implementation records one audit
with `submitted: false` and `status: STALE_POSITION_RESPONSE`, does not mutate
regime memory, and returns before all execution branches.

## GREEN evidence

Focused regression:

```text
python -m unittest test_nifty_multi_strategy_master.TestCPRAIWorkerFoundation.test_stale_scale_in_cannot_apply_to_a_replacement_position
Ran 1 test in 0.027s
OK
```

Required broader verification:

```text
python -m pytest "Signal Generators/CPR AI Agent/tests" -q
54 passed, 1 warning in 5.05s

python -m unittest -b test_nifty_multi_strategy_master
Ran 482 tests in 5.094s
OK (skipped=52)

python -m ruff check "Nifty Multi Strategy Front Test - Master File.py" "test_nifty_multi_strategy_master.py"
All checks passed!

python -m py_compile "Nifty Multi Strategy Front Test - Master File.py" "test_nifty_multi_strategy_master.py"
exit code 0

git diff --check
exit code 0
```

The CPR AI pytest warning is an existing `python-dateutil` deprecation warning.
No real model, broker, authenticated smoke, or order boundary was called.
