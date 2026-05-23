#!/usr/bin/env bash
# Apply Intel's vllm_for_multi_arc.patch to a v0.19.0 checkout with our
# curated exclusions (see curated-excludes.txt for the list).
#
# Usage:  apply-curated.sh /path/to/vllm-checkout
#
# After this, the working tree will have ~50 files with conflict markers
# that need manual resolution. See analysis/conflict-resolution-plan.md.

set -euo pipefail

PATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH="$PATCH_DIR/raw/vllm_for_multi_arc.patch"
EXCLUDES="$PATCH_DIR/curated-excludes.txt"

TARGET="${1:?usage: $0 /path/to/vllm-checkout}"

cd "$TARGET"

# Build --exclude args from the exclude list
EXCLUDE_ARGS=""
while IFS= read -r line; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue ;; esac
  EXCLUDE_ARGS="$EXCLUDE_ARGS --exclude=$line"
done < "$EXCLUDES"

echo "Applying $PATCH"
echo "Excluding $(grep -cE '^[^#].' "$EXCLUDES") files (see $EXCLUDES)"
echo ""

# git apply --3way leaves conflict markers in the tree on conflict,
# but is atomic — if ANY hunk errors, nothing is written. The exclude
# list removes the 10 known hard-error files so the apply proceeds.
eval git apply --3way "$EXCLUDE_ARGS" "$PATCH" || true

echo ""
echo "=== summary ==="
echo "modified files:      $(git diff HEAD --name-only | wc -l)"
echo "new untracked files: $(git ls-files --others --exclude-standard | wc -l)"
echo "files with <<<<<<<:  $(grep -rlE '^<<<<<<< ' --include='*.py' --include='*.txt' --include='*.md' --include='*.sh' --include='*.yaml' . 2>/dev/null | grep -v '\.git/' | wc -l)"
