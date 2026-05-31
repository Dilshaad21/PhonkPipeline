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

The script follows a strict, per-submodule algorithm to avoid ordering and
error-handling ambiguity. High-level steps are: initialize any empty
submodules, fetch upstream for each submodule, fast-forward those that can be
fast-forwarded, mark packed submodules that advanced for repack, run
`pack_repos.py` if needed (and restore on failure), stage changed gitlinks and
packs, then run `smoke_test.py`.

Per-submodule algorithm (pseudocode):

- For each submodule in [BeatSync-Engine, auto-gaming-montage-maker, crispy]:
  1. If submodule directory is empty, run `git submodule update --init --recursive`.
    - If this command fails, print "failed to initialize submodule <name>: <error>"
     and exit 4.
  2. Run `git -C <submodule> fetch <remote>`.
    - If `git fetch` fails (network/auth), print "fetch failed for <submodule>: <error>"
     and exit 4 without staging changes.
  3. Determine the upstream tip (e.g. `<remote>/<branch>` or remote HEAD).
    - If the pinned SHA equals the upstream tip, continue to next submodule.
  4. If the submodule has local uncommitted changes or local commits preventing
    a fast-forward, print "Submodule <name> has local commits; please clean, stash, or rebase."
    and exit 4. (To restore previous state: `git submodule update --checkout`.)
  5. Attempt to fast-forward: `git -C <submodule> merge --ff-only <remote>/<branch>`.
    - If not fast-forwardable, abort with the message above and exit 4.
  6. If the submodule advanced and the submodule is one of
    `BeatSync-Engine` or `auto-gaming-montage-maker`, mark it for repack.

- After the per-submodule loop:
  1. If any packed submodule was marked, run `python pack_repos.py`.
    - If `pack_repos.py` exits non-zero or errors, restore previous pins with
     `git submodule update --checkout`, do not stage any changes, print
     "pack_repos.py failed: <error>", and exit 5.
  2. `git add` the advanced gitlinks and any regenerated packs (`beatsync_source.txt`,
    `montage_fx_source.txt`) to the index (staged changes).
  3. Run `python smoke_test.py` to verify the workspace.
    - If `smoke_test.py` fails and the failure is due only to import errors for the
     heavy dependencies (`librosa`, `moviepy`, `torch`, `transformers`),
     print "Stage-1 missing deps: <list>" and exit 3 by default.
    - The script accepts `--allow-stage1-fail`; when supplied and a Stage-1-only
     failure is detected, exit 0 and leave changes staged for commit.
    - For all other smoke-test failures, exit 3 and leave changes staged for review.
  4. Report old→new SHAs, which packs were regenerated, smoke test result,
    and the suggested commit command.

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
| 4 | Preflight or fetch/fast-forward failure (wrong dir, missing files, network/auth, or local commits blocking fast-forward) |
| 5 | `pack_repos.py` failed — previous pins restored, no changes staged |

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
  the heavy deps (`librosa`, `moviepy`, `torch`, `transformers`). When the
  smoke test fails solely due to import errors for these packages the script
  prints `Stage-1 missing deps: <list>` and exits 3 by default. If you want to
  allow this specific condition and leave changes staged for commit, run
  `sync.sh --allow-stage1-fail` which exits 0 on Stage-1-only failures.
- **`pack_repos.py` failed during repack.** If `pack_repos.py` errors the
  script restores previous submodule pins with `git submodule update --checkout`,
  does not stage changes, prints `pack_repos.py failed: <error>`, and exits 5.
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
