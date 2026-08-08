# Task 5 fix round 1 report

## Commits and scope

- Starting clean HEAD: `ae33226d9583d571a4f45ab10d3ba796ecb60384`
- Code and regression tests: `a550a830ce975621f888b6857d32cf7c9fe98fe3`
  (`Fix CPR AI post-inference safety gates`)
- This report is committed separately after the verified code commit.

No real Codex/model, MCP-authenticated smoke, broker, web, authentication, or
order call occurred. All agent and broker boundaries in the tests were local
fakes or mocks.

## RED evidence captured before production edits

Every command below exited 1 for the intended missing behavior.

1. `python -m pytest "Signal Generators/CPR AI Agent/tests/test_cpr_ai_runtime.py::test_generated_mcp_command_reaches_the_real_server_parser" -q`
   - Failed with `SystemExit: 2` and `unrecognized arguments: --snapshot` when
     the exact generated arguments reached the real server parser.
2. `python -m unittest test_nifty_multi_strategy_master.TestCPRAIWorkerFoundation.test_scale_in_rechecks_fresh_spot_stop_after_inference`
   - Failed because `exit_position("CPR_AI_HARD_STOP")` was called 0 times after
     spot changed from 101 to 94 during fake inference.
3. `python -m unittest test_nifty_multi_strategy_master.TestCPRAIWorkerFoundation.test_scale_in_rechecks_fresh_max_loss_after_inference`
   - Failed because `handle_max_loss_and_stop(-450.0, -450.0)` was called 0
     times after the locked option mark changed from 10 to 1 during inference.
4. `python -m unittest test_nifty_multi_strategy_master.TestCPRAIWorkerFoundation.test_scale_in_rejects_a_wide_locked_contract_spread_without_consuming_add`
   - Failed because the role-A BUY broker boundary was called despite a 20%
     spread against a 2% cap.
5. `python -m unittest test_nifty_multi_strategy_master.TestCPRAIWorkerFoundation.test_scale_in_rejects_failed_liquidity_score_without_consuming_add`
   - Failed because the role-A BUY broker boundary was called despite a 0.4
     liquidity score against a 30.0 floor.
6. `python -m unittest test_nifty_multi_strategy_master.TestCPRAIWorkerFoundation.test_stale_scale_in_cannot_apply_to_a_replacement_position`
   - With the identity guard deliberately absent, failed because the obsolete
     outcome submitted a role-A BUY against `NIFTY-REPLACEMENT`.

The first spot-test GREEN attempt exposed an incomplete synthetic fixture: the
new max-loss recheck correctly requested an option mark, but the fake broker
returned a `MagicMock`. The test fixture was completed with a local option LTP;
production behavior was not weakened. A focused compatibility run also exposed
an old assertion that rejected any broker-boundary call, including a required
SELL flatten. It now asserts that no `opens_exposure=True` call occurs.

## Changed behavior

- The generated MCP command now supplies the snapshot as the positional path
  accepted by `cpr_ai_mcp_server.py`; the real parser and real four-tool server
  construction are exercised while only `FastMCP.run()` is replaced.
- The worker retains the exact position and CPR sidecar identities used for the
  frozen context. A response is ignored if reconciliation closed/replaced that
  position during inference.
- An accepted premise `EXIT` is still honored before exposure-only gates.
- Before HOLD or scale-in can finish, the worker reuses `_run_prebar_safety()`
  with fresh shared-market facts. This rechecks lifecycle shutdown, max-loss,
  15:15 square-off, stale-feed liquidation, and hard-stop/final-target spot
  boundaries. A closed scale-in exposure gate is audited, then the same safety
  pass performs any associated flattening before returning.
- `_execute_scale_in()` now applies the existing spread and liquidity gates to
  the locked direction, symbol, strike, option right, and expiry before setting
  `scale_in_used`, creating a paper add, or submitting a live role-A BUY.

## GREEN verification

All commands exited 0.

- Focused MCP parser boundary: 1 passed.
- `python -m pytest "Signal Generators/CPR AI Agent/tests/test_cpr_ai_runtime.py" -q`: 20 passed.
- `python -m unittest test_nifty_multi_strategy_master.TestCPRAIWorkerFoundation`: 30 passed.
- `python -m pytest "Signal Generators/CPR AI Agent/tests" -q`: 54 passed, 1 existing third-party `dateutil` deprecation warning.
- `python -m pytest "Signal Generators/CPR Strategy" -q`: 17 passed, the same existing warning.
- `python -m pytest "Signal Generators/Regime Adaptive Strategy" "Signal Generators/test_trading_bot_ports.py" -q`: 146 passed, the same existing warning.
- `python -m unittest test_nifty_multi_strategy_master`: 482 passed, 52 skipped.
- `python -m unittest test_market_data_health`: 26 passed.
- `python -m ruff check -- "Nifty Multi Strategy Front Test - Master File.py" test_nifty_multi_strategy_master.py "Signal Generators/CPR AI Agent/cpr_ai_codex_subprocess.py" "Signal Generators/CPR AI Agent/tests/test_cpr_ai_runtime.py"`: passed.
- `python -m py_compile "Nifty Multi Strategy Front Test - Master File.py" test_nifty_multi_strategy_master.py "Signal Generators/CPR AI Agent/cpr_ai_codex_subprocess.py" "Signal Generators/CPR AI Agent/tests/test_cpr_ai_runtime.py"`: passed.
- `git diff --check`: passed; Git printed only the repository's Windows LF-to-CRLF working-copy notices.

## Concern

No functional concern remains in this fix scope. The only warning is the
pre-existing third-party `dateutil` deprecation warning reported by pytest.
