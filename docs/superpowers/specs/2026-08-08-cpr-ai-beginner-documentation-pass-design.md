# CPR AI Beginner Documentation Pass Design

## Goal

Make the new CPR SRSI/VWAP Codex agent understandable to a beginner who knows
basic Python but is new to this repository, agent isolation, and live-trading
safety. This is an editorial pass only: trading rules, validation, timing,
execution, configuration, and test behavior must remain unchanged.

## Scope

The pass covers:

- every Python module in `Signal Generators/CPR AI Agent/`;
- the CPR AI worker, configuration, construction, execution, reconciliation,
  and P&L sections added to the master runner;
- reusable CPR AI test helpers and non-obvious test setup where an explanation
  helps a beginner understand the safety contract.

The pass does not add comments to obvious assignments, individual assertions,
environment-template lines, generated reports, or unrelated strategies. It
does not refactor code merely to create places for comments.

## Documentation Style

Module docstrings will explain where the module sits in the overall flow and
which responsibilities it deliberately does not own. Class and function
docstrings will explain purpose, important inputs and outputs, authoritative
versus advisory data, and fail-closed behavior where relevant.

Inline comments will be placed immediately before non-obvious blocks. They
will explain why the block exists, especially when the safest behavior is not
the most obvious behavior. Comments will use plain English, introduce acronyms
on first use, and avoid simply translating the next line of Python into prose.

## Safety Boundaries to Explain

The pass will give particular attention to:

- excluding forming one-minute candles and deciding once per immutable
  completed five-minute bucket;
- literal Wilder RSI and Stochastic RSI construction;
- the boundary between deterministic host evidence and nondeterministic model
  judgment;
- the four frozen, no-argument MCP tools and strict structured decision schema;
- the process-isolated, authentication-only Codex home and sanitized child
  environment;
- host-side geometry, stop-width, reward/risk, timing, and stale-result checks;
- mechanical stop, max-loss, final-target, trailing, and square-off behavior;
- one-time same-contract R1 scale-in rules and ambiguous live-fill handling;
- execution-ledger reconciliation and actual PAPER/LIVE provenance;
- credential-safe decision logging that preserves legitimate strategy evidence.

## Behavioral Invariant

Only comments and docstrings may change in the implementation pass. No
executable statement, expression, constant, annotation, import, public
interface, schema, prompt text, environment default, or test expectation may
change.

As an additional guard, the final verification will compare Python abstract
syntax trees against commit `9d40d01` after removing docstring nodes. Any
difference means executable behavior changed and the pass must be corrected
before publication.

## Verification and Publication

After the editorial changes:

1. compare docstring-stripped Python syntax trees with `9d40d01`;
2. run the CPR AI suite, master suite, market-health suite, and full repository
   pytest suite;
3. run branch coverage and its safety thresholds;
4. run compileall, Ruff, mypy, Bandit, pre-commit, and the fake order-free CPR
   AI smoke command;
5. obtain a fresh read-only review of the documentation-only diff;
6. commit with the required Codex co-author trailer, push the existing feature
   branch, and open a draft pull request into `main` whose body records the
   implementation, safety boundaries, verification, and Codex co-authorship.
