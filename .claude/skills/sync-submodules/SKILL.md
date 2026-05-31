---
name: sync-submodules
description: Use when updating the vendored submodules (BeatSync-Engine, auto-gaming-montage-maker, crispy) to latest upstream, when the *_source.txt packs look stale, or after cloning when submodules are empty.
---

# Sync Submodules

## Overview

PhonkPipeline vendors three upstream projects as git submodules. They are
read-only references; the deliverable is the distilled root modules plus the
`*_source.txt` packs generated from the submodules by `pack_repos.py`.

This skill drives `sync.sh`, which advances each submodule to its upstream's
latest commit, regenerates **only the affected** packs, runs the smoke test,
and **stages** the result for your review. It never commits and never pushes.

## When to Use

- "Update / sync the submodules", "pull upstream into the vendored repos"
- The `*_source.txt` packs look stale vs. the vendored source
- Fresh clone where the submodule directories are empty
- Before harvesting a capability, to distill from current upstream

## How to Run

```bash
bash .claude/skills/sync-submodules/sync.sh --check   # report what would change, no writes
bash .claude/skills/sync-submodules/sync.sh           # sync + repack + stage + smoke test
```

Then review and commit yourself if happy:

```bash
git diff --cached
git commit -m "Sync submodules to upstream + repack"
```

## What It Does

1. **Preflight** — confirm repo root + required files; init any empty submodule.
2. **Per-submodule** — `git fetch`, compare pinned vs. upstream tip. Already
   current → skip. Behind → fast-forward checkout to the tip.
3. **Repack** — run `pack_repos.py` only if a *packed* submodule advanced.
4. **Stage** — `git add` the advanced gitlinks + regenerated packs.
5. **Verify** — run `smoke_test.py`.
6. **Report** — old→new SHAs, what repacked, smoke result, commit command.

## Submodule → Pack Mapping

| Submodule | Pack regenerated |
|-----------|------------------|
| `BeatSync-Engine` | `beatsync_source.txt` |
| `auto-gaming-montage-maker` | `montage_fx_source.txt` |
| `crispy` | *(none — reference-only, not packed)* |

Syncing only `crispy` updates its gitlink but triggers no repack.

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Synced (or all current) **and** smoke test passed |
| 3 | Synced but smoke test FAILED — changes left **staged**, not committed |
| 4 | Preflight failure (wrong dir, missing files) |

## On Smoke-Test Failure (exit 3)

Changes are left staged so you can inspect, never auto-committed.

```bash
git diff --cached                                   # inspect what synced
# roll back to the previously pinned commits + packs:
git submodule update --checkout
git checkout -- beatsync_source.txt montage_fx_source.txt
```

## Common Mistakes

- **"Smoke test always fails on my machine."** `smoke_test.py` Stage 1 imports
  the heavy deps (`librosa`, `moviepy`, `torch`, `transformers`). Without them
  installed it reports FAIL and `sync.sh` exits 3 even though the sync itself
  worked. Run on your provisioned dev machine, or treat a Stage-1-only failure
  as expected and commit after eyeballing `git diff --cached`.
- **Dirty submodule working trees.** Generated junk (e.g. `__pycache__`) inside
  a submodule shows it as `...-dirty`. Preflight warns but proceeds; `sync.sh`
  only stages submodules that actually advanced, so the noise won't be swept
  into your commit. Clean with `git submodule foreach git clean -fdx` if desired.
- **Committing without review.** The skill deliberately stops at staged. Look at
  `git diff --cached` before committing — an upstream bump can change a lot.

## Reachability Guarantee

`sync.sh` only ever fast-forwards to a commit fetched from upstream, so the new
pins always exist on the remote — a fresh `clone --recurse-submodules` will
always populate them.
