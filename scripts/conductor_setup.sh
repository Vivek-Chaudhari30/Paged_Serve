#!/usr/bin/env bash
#
# Per-worktree setup for parallel agent sessions (Conductor, git worktrees, or
# anything else that gives each session its own checkout).
#
# The problem it solves: `_private/` is gitignored, so DESIGN.md and ROADMAP.md
# do not exist in a fresh worktree. AGENTS.md §1 tells every agent to stop and
# refuse to work if it cannot see them. Without this, every parallel session you
# start halts on its first message.
#
# Point your tool's per-workspace setup command at this script, or run it by
# hand inside a new worktree before sending the first prompt.
#
# Usage:
#   scripts/conductor_setup.sh
#   PRIVATE_SRC=/path/to/canonical/_private scripts/conductor_setup.sh

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
WORKTREE="$PWD"

# Where the real _private/ lives. Defaults to the main working tree, resolved
# from git rather than hardcoded so this keeps working if the repo moves.
if [ -n "${PRIVATE_SRC:-}" ]; then
    SRC="${PRIVATE_SRC}"
else
    # `git rev-parse --path-format=absolute --git-common-dir` points at the main
    # checkout's .git even from inside a linked worktree.
    COMMON_GIT=$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null || echo "")
    if [ -n "${COMMON_GIT}" ]; then
        SRC="$(dirname "${COMMON_GIT}")/_private"
    else
        SRC=""
    fi
fi

if [ -z "${SRC}" ] || [ ! -d "${SRC}" ]; then
    echo "ERROR: cannot find the canonical _private/ directory." >&2
    echo "       Tried: ${SRC:-<unresolved>}" >&2
    echo "       Set PRIVATE_SRC to its absolute path and run this again." >&2
    exit 1
fi

if [ "${SRC}" = "${WORKTREE}/_private" ]; then
    echo "this IS the main worktree; _private/ is already here"
    exit 0
fi

ln -sfn "${SRC}" "${WORKTREE}/_private"

# Verify rather than assume. A broken symlink and a working one look identical
# in `ls`, and the failure only shows up as an agent refusing to start.
for f in DESIGN.md ROADMAP.md; do
    [ -r "${WORKTREE}/_private/${f}" ] || {
        echo "ERROR: _private/${f} is not readable through the link." >&2
        exit 1
    }
done

echo "linked _private/ -> ${SRC}"
echo "verified: DESIGN.md, ROADMAP.md"
