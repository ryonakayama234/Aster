#!/usr/bin/env bash
# Invoke from the user's Aster checkout, using a reviewed commit's Git blob.
# Check only: no predictors run, no diagnostic Run created, no existing data removed.
set -eu

aster_revision=${1:-}
if [[ ! "$aster_revision" =~ ^[0-9a-f]{40}$ ]]; then
  echo 'レビュー済みの40桁Git版を指定してください。'
  exit 2
fi

cd "$(git rev-parse --show-toplevel)"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo '未コミットの変更があるため停止しました。手元の作業は変更していません。'
  exit 2
fi

aster_previous=$(git symbolic-ref -q --short HEAD || git rev-parse HEAD)
trap 'aster_rc=$?; trap - EXIT; git checkout --quiet --no-overwrite-ignore "$aster_previous"; exit "$aster_rc"' EXIT
git switch --quiet --no-overwrite-ignore --detach "$aster_revision"
.venv/bin/python -I scripts/run_lang_v03_context.py \
  --root . --expected-commit "$aster_revision" --check-only
