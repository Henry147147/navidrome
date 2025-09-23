#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CACHE_DIR="${ROOT_DIR}/.cache/taglib"
VERSION=${CROSS_TAGLIB_VERSION:-2.1.1-1}
GOOS=${GOOS:-$(go env GOOS)}
GOARCH=${GOARCH:-$(go env GOARCH)}

platform="${GOOS}-${GOARCH}"
case "${platform}" in
  linux-amd64|linux-arm64|linux-armv6|linux-armv7|linux-386|\
  darwin-amd64|darwin-arm64|windows-amd64|windows-386)
    ;;
  *)
    echo "Unsupported platform: ${platform}" >&2
    exit 1
    ;;
esac

DEST_DIR="${CACHE_DIR}/v${VERSION}/${platform}"
MARKER="${DEST_DIR}/.complete"

if [ ! -f "${MARKER}" ]; then
  URL="https://github.com/navidrome/cross-taglib/releases/download/v${VERSION}/taglib-${platform}.tar.gz"
  TMP_DIR=$(mktemp -d)
  ARCHIVE="${TMP_DIR}/taglib.tar.gz"
  mkdir -p "${DEST_DIR}"
  echo "Fetching TagLib ${VERSION} for ${platform}..." >&2
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "${URL}" -o "${ARCHIVE}"
  elif command -v wget >/dev/null 2>&1; then
    wget -q "${URL}" -O "${ARCHIVE}"
  else
    echo "Neither curl nor wget found." >&2
    exit 1
  fi
  rm -rf "${DEST_DIR}"/*
  tar -xzf "${ARCHIVE}" -C "${DEST_DIR}" --strip-components=1
  touch "${MARKER}"
  rm -rf "${TMP_DIR}"
fi

printf '%s\n' "${DEST_DIR}"
