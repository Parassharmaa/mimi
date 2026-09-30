#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
APP="${1:-$ROOT/.build/Mimi.app/Contents/MacOS/Mimi}"
[[ -x "$APP" ]] || { print -u2 "Build Mimi before running the live field checks."; exit 1; }
FIXTURE_DIR="$(mktemp -d -t mimi-field-fixture)"
swiftc -parse-as-library "$ROOT/scripts/voice_typing_fixture.swift" -o "$FIXTURE_DIR/fixture"
FIXTURE_PID=""
trap '[[ -z "$FIXTURE_PID" ]] || kill "$FIXTURE_PID" 2>/dev/null || true' EXIT

check_case() {
  local case_name="$1" expected="$2"
  shift 2
  local fixture_args=() app_args=()
  while [[ "$1" != "--" ]]; do fixture_args+=("$1"); shift; done
  shift
  app_args=("$@")
  "$FIXTURE_DIR/fixture" "${fixture_args[@]}" > "$FIXTURE_DIR/$case_name.jsonl" &
  FIXTURE_PID="$!"
  sleep 0.5
  "$APP" --e2e-stream-insert 'there|there now' --e2e-delay 1 "${app_args[@]}"
  sleep 0.3
  local actual="$(tail -n 1 "$FIXTURE_DIR/$case_name.jsonl")"
  kill "$FIXTURE_PID"
  wait "$FIXTURE_PID" 2>/dev/null || true
  FIXTURE_PID=""
  [[ "$actual" == "$expected" ]] || { print -u2 "FAIL $case_name: $actual"; exit 1; }
  print "PASS $case_name: full field contents and second-field isolation"
}

check_case selection '["hello world","untouched second field"]' --location 6 --length 5 --
check_case insertion '["hello world","untouched second field"]' --location 6 --length 0 --
check_case unicode '["a😀 東京です b","untouched second field"]' --text 'a😀 東京です b' --location 4 --length 2 --
check_case preparation-cancel '["hello world","untouched second field"]' --location 6 --length 5 -- --e2e-rollback-only
check_case focus-change '["hello there","untouched second field"]' --switch-after 2 -- --e2e-step-delay 2 --e2e-expect-focus-change
check_case user-edit '["user changed this field","untouched second field"]' --edit-after 2 -- --e2e-step-delay 2 --e2e-expect-destination-change
check_case caret-change '["hello there","untouched second field"]' --move-caret-after 2 -- --e2e-step-delay 2 --e2e-expect-destination-change
check_case secure-field '["hello world","untouched second field"]' --secure -- --e2e-expect-secure-field
print "Evidence is synthetic and retained at $FIXTURE_DIR"
