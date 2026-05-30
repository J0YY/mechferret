#!/usr/bin/env bash
# Install MechFerret as a global `mechferret` (and `mf`) command -- no `python -m` needed.
set -e
cd "$(dirname "$0")"
if command -v pipx >/dev/null 2>&1; then
  echo "Installing with pipx (isolated global command)..."
  pipx install --force .
else
  echo "pipx not found; installing with pip --user..."
  python3 -m pip install --user .
fi
echo
echo "Done. Now just run:"
echo "  mechferret                 # headline discovery (like 'claude')"
echo "  mechferret /skills"
echo "  mechferret discover --skill find-induction-heads"
echo "  mechferret /modal run --skill ioi-circuit"
echo "  mechferret /cluster run --skill ioi-circuit"
