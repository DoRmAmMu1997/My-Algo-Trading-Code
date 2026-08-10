# ADR-0011: A committed `docs/` set, with the Superpowers workspace untracked

**Status:** Accepted
**Date:** 2026-08-10
**Deciders:** repository owner

## Context

`docs/` existed but held only `docs/superpowers/` — two files (a plan and a
design spec) written by an agent during one development session in July 2026:

```
docs/superpowers/plans/2026-07-27-mat-111-sl-hunting-cooldown.md
docs/superpowers/specs/2026-07-27-mat-111-sl-hunting-cooldown-design.md
```

Both were tracked in git. They describe *how one session's work was planned*,
not how the product works, and they were already stale relative to the shipped
cooldown implementation.

Meanwhile the actual architecture was documented across `README.md` (operator
setup and a running changelog), `CLAUDE.md` / `AGENTS.md` (condensed rules for
coding agents), and six per-folder `Readme.md` files. There was no document a
new reader could open to understand the system as a system, and nothing recorded
*why* the significant decisions were made — only what they were.

`.superpowers/` at the repository root was already gitignored for exactly this
reason; the `docs/superpowers/` subtree had simply been missed.

## Decision

Two parts.

**1. Untrack `docs/superpowers/`.** Add it to `.gitignore` and `git rm --cached`
the two files. They stay on disk; they leave source control. This makes the rule
consistent with the existing `.superpowers/` entry.

**2. Fill `docs/` with a committed architecture set:**

```
docs/
  README.md      index and reading order
  hld/           one high-level design for the whole repository
  lld/           one low-level design per component (12)
  adr/           decision records (11)
```

The division of labour between this and the existing documentation:

| Document | Answers | Audience |
|---|---|---|
| root `README.md` | How do I set it up and run it? | operator |
| `CLAUDE.md` / `AGENTS.md` | What rules must I follow when changing code here? | coding agents |
| per-folder `Readme.md` | What is in this folder? | anyone in that folder |
| `docs/hld/` | How does the system fit together? | new reader |
| `docs/lld/` | How does this component work, and what breaks it? | anyone changing it |
| `docs/adr/` | Why is it like this? | anyone about to change it back |

## Options considered

### Option A: Committed HLD + LLDs + ADRs, Superpowers untracked (chosen)

**Pros:** a new reader has one entry point; component internals are documented
where they can be reviewed in the same pull request as the code; the *reasons*
behind the safety model are recorded, which matters most for decisions that look
like over-engineering until you know the incident behind them.
**Cons:** documentation drifts unless maintained; ~24 files to keep current;
some overlap with the per-folder Readmes.

### Option B: Keep everything in `README.md`

**Pros:** one file; already the habit.
**Cons:** it is already 27 KB and mixes setup, a changelog and architecture. It
serves the operator well and would serve none of them well if the architecture
were added.

### Option C: Commit the Superpowers plans as the design record

**Pros:** they already exist.
**Cons:** they are session artefacts — planned rather than shipped, dated, and
already stale. Presenting them as the design record documents an intention, not
the system.

## Trade-off analysis

The cost of Option A is drift, and it is a real cost — stale architecture
documentation is worse than none, because it is believed. Two conventions are
adopted against it:

1. **The LLD changes in the same commit as the component.** If a pull request
   changes behaviour, its LLD is part of that pull request.
2. **Never restate a number the code owns.** Prefer naming the file that holds
   the default to copying the value.

ADRs are treated as append-only history: when a decision changes, a new ADR
supersedes the old one and the old one is marked `Superseded` rather than
edited. That keeps the record of *why the previous answer was chosen*, which is
the part that is expensive to reconstruct.

## Consequences

**Easier:** onboarding a reader (human or agent); reviewing a change against a
stated design; arguing against a decision from its actual reasoning rather than
from a guess about it.

**Harder:** ~24 more files to keep current. The mitigation is convention, not
tooling.

**To revisit when:** the docs are observed to be stale. The existing repository
already has precedent for enforcing documentation freshness in CI —
`test_repository_policy.py` fails the build on stale worker-roster claims in
`README.md`, `CLAUDE.md`, `AGENTS.md` and the master file. Extending that gate
to cover `docs/hld/` is a reasonable follow-up; it is deliberately **not** done
in this change, to keep the restructure surgical.

## Action items

- [x] `docs/superpowers/` added to `.gitignore`, files untracked with
      `git rm --cached` (kept on disk).
- [x] `docs/README.md` index with reading order.
- [x] `docs/hld/system-overview.md`.
- [x] 12 LLDs under `docs/lld/`.
- [x] 11 ADRs under `docs/adr/`.
- [ ] **Follow-up:** consider adding `docs/hld/system-overview.md` to the
      architecture-staleness gate in `Tests/Dependencies/test_repository_policy.py`.
