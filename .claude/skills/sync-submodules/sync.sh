#!/usr/bin/env bash
#
# sync.sh — smart submodule sync for PhonkPipeline.
#
# Updates each vendored submodule to its upstream's latest commit (skipping any
# already current), regenerates only the affected *_source.txt packs, runs the
# smoke test, and stages the result for review. It NEVER commits or pushes.
#
# Portable to macOS's default bash 3.2 (no associative arrays).
#
# Exit codes:
#   0  success (something synced or all current) AND smoke test passed
#   3  sync done but smoke test FAILED (changes left staged for inspection)
#   4  preflight failure (not in repo root, missing files, bad submodule)
#
# Usage:
#   bash .claude/skills/sync-submodules/sync.sh [--check]
#     --check : report what would change (fetch + compare) and exit; no updates.

set -uo pipefail

# Submodules to sync, in order.
SUBMODULES="BeatSync-Engine auto-gaming-montage-maker crispy"

# submodule -> pack file. An empty result means "reference-only, no repack"
# (crispy is intentionally not packed; see CLAUDE.md).
pack_for() {
  case "$1" in
    BeatSync-Engine)            echo "beatsync_source.txt" ;;
    auto-gaming-montage-maker)  echo "montage_fx_source.txt" ;;
    *)                          echo "" ;;
  esac
}

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

say()  { printf '%s\n' "$*"; }
hdr()  { printf '\n=== %s ===\n' "$*"; }
fail() { printf 'ABORT: %s\n' "$*" >&2; exit 4; }

# --- preflight -------------------------------------------------------------
hdr "Preflight"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || fail "not inside a git repository"
cd "$REPO_ROOT" || fail "cannot cd to repo root"
[ -f pack_repos.py ] && [ -f smoke_test.py ] && [ -f .gitmodules ] \
  || fail "missing pack_repos.py / smoke_test.py / .gitmodules — run from PhonkPipeline"
say "repo root: $REPO_ROOT"

# Warn (don't block) on unrelated unstaged changes so the sync diff stays readable.
if ! git diff --quiet -- . ':(exclude).claude' 2>/dev/null; then
  say "note: working tree has unstaged changes outside .claude/ — sync diff will include only submodule/pack edits."
fi

# Ensure submodules are initialized (materialize empty ones from pinned commits).
for d in $SUBMODULES; do
  if [ -z "$(ls -A "$d" 2>/dev/null)" ]; then
    say "initializing empty submodule: $d"
    git submodule update --init -- "$d" || fail "could not init submodule $d"
  fi
done

# --- per-submodule check / update -----------------------------------------
hdr "Submodule sync"
CHANGED=""        # space-separated list of submodules that moved
REPACK_NEEDED=0
SUMMARY=""        # newline-separated human summary lines

for d in $SUBMODULES; do
  git -C "$d" fetch -q origin || { say "$d: FETCH FAILED (offline?) — skipping"; continue; }

  # Resolve upstream default branch tip (origin/HEAD), fall back to remote show.
  upstream_ref="$(git -C "$d" symbolic-ref -q --short refs/remotes/origin/HEAD 2>/dev/null)"
  [ -z "$upstream_ref" ] && upstream_ref="$(git -C "$d" remote show origin 2>/dev/null \
      | awk '/HEAD branch/{print "origin/"$NF}')"
  [ -z "$upstream_ref" ] && { say "$d: cannot determine upstream branch — skipping"; continue; }

  old="$(git -C "$d" rev-parse HEAD)"
  new="$(git -C "$d" rev-parse "$upstream_ref")"

  if [ "$old" = "$new" ]; then
    say "$d: already current ($(git -C "$d" rev-parse --short HEAD))"
    continue
  fi

  subj="$(git -C "$d" log -1 --format=%s "$new")"
  say "$d: $(echo "$old" | cut -c1-9) -> $(echo "$new" | cut -c1-9)  ($subj)"

  if [ $CHECK_ONLY -eq 0 ]; then
    git -C "$d" checkout -q "$new" || { say "$d: checkout failed — skipping"; continue; }
  fi

  CHANGED="$CHANGED $d"
  SUMMARY="${SUMMARY}  synced  $d  $(echo "$old" | cut -c1-9) -> $(echo "$new" | cut -c1-9)  ($subj)
"
  [ -n "$(pack_for "$d")" ] && REPACK_NEEDED=1
done

CHANGED="$(echo "$CHANGED" | xargs)"   # trim

if [ $CHECK_ONLY -eq 1 ]; then
  hdr "Check summary"
  if [ -z "$CHANGED" ]; then
    say "All submodules current. Nothing to sync."
  else
    say "Would sync: $CHANGED"
    [ $REPACK_NEEDED -eq 1 ] && say "Would repack: pack_repos.py (beatsync_source.txt, montage_fx_source.txt)"
  fi
  exit 0
fi

if [ -z "$CHANGED" ]; then
  hdr "Result"; say "All submodules already current — no changes, nothing staged."; exit 0
fi

# --- repack (only if a mapped submodule changed) --------------------------
hdr "Repack"
if [ $REPACK_NEEDED -eq 1 ]; then
  python3 pack_repos.py || fail "pack_repos.py failed"
else
  say "no packed submodule changed (only crispy) — skipping repack"
fi

# --- stage (gitlinks + regenerated packs) ---------------------------------
hdr "Stage"
for d in $CHANGED; do git add -- "$d"; done
[ $REPACK_NEEDED -eq 1 ] && git add -- beatsync_source.txt montage_fx_source.txt
git --no-pager diff --cached --stat

# --- verify ---------------------------------------------------------------
hdr "Verify (smoke test)"
if python3 smoke_test.py; then SMOKE_OK=1; else SMOKE_OK=0; fi

# --- report ---------------------------------------------------------------
hdr "Summary"
printf '%s' "$SUMMARY"
[ $REPACK_NEEDED -eq 1 ] && say "  repacked beatsync_source.txt, montage_fx_source.txt"

if [ $SMOKE_OK -eq 1 ]; then
  say ""
  say "Smoke test PASSED. Changes are staged. To commit:"
  say "    git commit -m \"Sync submodules to upstream + repack\""
  exit 0
else
  say ""
  say "Smoke test FAILED. Changes left STAGED for inspection — NOT committed."
  say "Inspect:   git diff --cached"
  say "Roll back: git submodule update --checkout && git checkout -- beatsync_source.txt montage_fx_source.txt"
  exit 3
fi
