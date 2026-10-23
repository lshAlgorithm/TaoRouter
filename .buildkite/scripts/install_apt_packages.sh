#!/usr/bin/env bash
# Install one named apt set from .buildkite/ci_apt_packages.conf.
#
# Usage:
#   bash .buildkite/scripts/install_apt_packages.sh rust
#   bash .buildkite/scripts/install_apt_packages.sh python
#   bash .buildkite/scripts/install_apt_packages.sh nvidia_pd

set -euo pipefail

if [ "$#" -ne 1 ] || [ -z "${1}" ]; then
  echo "usage: install_apt_packages.sh <name>" >&2
  exit 1
fi

name="$1"
root="$(cd "$(dirname "$0")/.." && pwd)"
config="${root}/ci_apt_packages.conf"
packages=""

while read -r key rest; do
  if [ -z "${key}" ] || [ "${key#\#}" != "${key}" ]; then
    continue
  fi
  if [ "${key}" = "${name}" ]; then
    packages="${rest}"
    break
  fi
done < "${config}"

if [ -z "${packages}" ]; then
  echo "unknown apt package set: ${name}" >&2
  echo "known sets:" >&2
  while read -r key rest; do
    if [ -n "${key}" ] && [ "${key#\#}" = "${key}" ]; then
      echo "  ${key}" >&2
    fi
  done < "${config}"
  exit 1
fi

echo "Installing apt set ${name}: ${packages}"
export DEBIAN_FRONTEND=noninteractive
apt-get update
# shellcheck disable=SC2086
apt-get install -y ${packages}
