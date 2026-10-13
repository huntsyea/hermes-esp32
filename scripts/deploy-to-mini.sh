#!/usr/bin/env bash
#
# Deploy the committed HEAD of this adapter repo to the Hermes runtime on the
# Mac mini, run the test suite against the runtime's own Python, and restart
# the gateway so the xiaozhi_companion platform picks up the new code.
#
# One-time setup (already done, recorded here for rebuilds):
#   ssh mac-mini 'git -C ~/.hermes/plugins/hermes-esp32 \
#       config receive.denyCurrentBranch updateInstead'
#   git remote add mini ssh://mac-mini/~/.hermes/plugins/hermes-esp32
#
# Usage: scripts/deploy-to-mini.sh [--no-restart] [--allow-dirty]

set -euo pipefail

MINI_HOST="${MINI_HOST:-mac-mini}"
# Remote paths stay unexpanded here so the mini's shell resolves ~.
HERMES_HOME="${HERMES_HOME:-~/.hermes}"
PLUGIN_PATH="${HERMES_HOME}/plugins/hermes-esp32"
HERMES_PYTHON="${HERMES_HOME}/hermes-agent/venv/bin/python"
GATEWAY_LABEL="ai.hermes.gateway"
GATEWAY_LOG="${HERMES_HOME}/logs/gateway.log"

RESTART=1
ALLOW_DIRTY=0
for argument in "$@"; do
    case "${argument}" in
        --no-restart) RESTART=0 ;;
        --allow-dirty) ALLOW_DIRTY=1 ;;
        *) echo "unknown argument: ${argument}" >&2; exit 2 ;;
    esac
done

cd "$(git rev-parse --show-toplevel)"

if [[ "${ALLOW_DIRTY}" -eq 0 ]] && ! git diff --quiet HEAD; then
    echo "Working tree has uncommitted changes; deploys ship committed HEAD only." >&2
    echo "Commit first, or pass --allow-dirty to ship HEAD anyway." >&2
    exit 1
fi

echo "==> Running local tests"
python3 -m unittest discover -s tests -v 2>&1 | tail -3

BRANCH="$(ssh "${MINI_HOST}" "git -C ${PLUGIN_PATH} branch --show-current")"
echo "==> Pushing HEAD to ${MINI_HOST}:${PLUGIN_PATH} (branch ${BRANCH})"
git push mini "HEAD:${BRANCH}"

echo "==> Running tests on the mini with the Hermes runtime Python"
ssh "${MINI_HOST}" "cd ${PLUGIN_PATH} && ${HERMES_PYTHON} -m unittest discover -s tests 2>&1 | tail -3"

if [[ "${RESTART}" -eq 1 ]]; then
    echo "==> Restarting Hermes gateway (${GATEWAY_LABEL})"
    ssh "${MINI_HOST}" "launchctl kickstart -k gui/\$(id -u)/${GATEWAY_LABEL}"
    echo "==> Waiting for xiaozhi_companion to reconnect"
    ssh "${MINI_HOST}" "
        for _ in \$(seq 1 30); do
            if tail -40 ${GATEWAY_LOG} | grep -q 'xiaozhi_companion connected'; then
                tail -40 ${GATEWAY_LOG} | grep 'xiaozhi_companion' | tail -2
                exit 0
            fi
            sleep 2
        done
        echo 'Timed out waiting for xiaozhi_companion to reconnect; check ${GATEWAY_LOG}' >&2
        exit 1
    "
else
    echo "==> Skipped gateway restart (--no-restart); new code loads on next restart"
fi

echo "==> Deploy complete: $(git rev-parse --short HEAD) on ${MINI_HOST}"
