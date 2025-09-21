#!/usr/bin/env bash
set -euo pipefail

# Generate TIMELINE.md by pulling dates from git history for key receipts.
# Usage:
#   bash generate_timeline.sh
# (Run this from the repo root where .git is present.)

outfile="TIMELINE.md"
echo "# Receipt Timeline" > "$outfile"
echo "" >> "$outfile"
echo "_Auto-generated: $(date -u +"%Y-%m-%d %H:%M:%SZ")_" >> "$outfile"
echo "" >> "$outfile"

# Repo context
echo "## Repo Context" >> "$outfile"
echo "" >> "$outfile"
echo "- HEAD: $(git rev-parse --short HEAD) — $(git log -1 --date=iso --format='format:%ad %s')" >> "$outfile"
echo "- Default branch: $(git symbolic-ref --short HEAD 2>/dev/null || echo 'N/A')" >> "$outfile"
echo "" >> "$outfile"

paths=(
  "browser project/browser.py"
  "q/main.py"
  "trent.pdf"
  "aeon.json"
  "the ghost.json"
  "system optimizer.py"
)

for p in "${paths[@]}"; do
  echo "## ${p}" >> "$outfile"
  if git ls-files --error-unmatch "$p" >/dev/null 2>&1; then
    first_line=$(git log --follow --diff-filter=A --date=short --format="%ad  %h  %s" -- "$p" | head -n 1 || true)
    last_line=$(git log --follow --date=short --format="%ad  %h  %s" -n 1 -- "$p" || true)
    num_commits=$(git log --follow --oneline -- "$p" | wc -l | awk '{print $1}')
    echo "- First added: ${first_line:-unknown}" >> "$outfile"
    echo "- Latest change: ${last_line:-unknown}" >> "$outfile"
    echo "- Commits touching this path: ${num_commits:-0}" >> "$outfile"
  else
    echo "- File not found in repo at this path." >> "$outfile"
  fi
  echo "" >> "$outfile"
done

echo "### Notes" >> "$outfile"
echo "- Dates are from \`git log\` with \`--follow\` to track renames." >> "$outfile"
echo "- Re-run \`bash generate_timeline.sh\` after new commits to refresh." >> "$outfile"
echo "" >> "$outfile"

echo "Wrote ${outfile}"
