#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$ROOT/external}"
mkdir -p "$DEST"
clone_pinned() {
  local name="$1" url="$2" revision="$3" dir="$DEST/$1"
  if [[ -e "$dir" ]]; then
    [[ -d "$dir/.git" ]] || { echo "Refusing non-repository directory: $dir" >&2; return 1; }
    [[ "$(git -C "$dir" rev-parse --show-toplevel)" == "$(cd "$dir" && pwd -P)" ]] || { echo "Refusing parent repository: $dir" >&2; return 1; }
    [[ "$(git -C "$dir" remote get-url origin)" == "$url" ]] || { echo "Unexpected remote: $dir" >&2; return 1; }
    [[ -z "$(git -C "$dir" status --porcelain)" ]] || { echo "Uncommitted benchmark changes: $dir" >&2; return 1; }
  else
    git clone --no-checkout "$url" "$dir"
  fi
  git -C "$dir" fetch origin "$revision"
  git -C "$dir" checkout --detach "$revision"
  [[ "$(git -C "$dir" rev-parse HEAD)" == "$revision" ]]
  echo "$name pinned at $revision"
}
clone_pinned BenchForm https://github.com/Zhiyuan-Weng/BenchForm.git 6425ecaebd9dd273a13ea28e32767c452e04c6a6
clone_pinned KAIROS https://github.com/declare-lab/KAIROS.git a22b81cb1c7b448122b6520b4da11c924a242824
