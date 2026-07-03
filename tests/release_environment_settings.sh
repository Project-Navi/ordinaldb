#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-Project-Navi/ordinaldb}"
EXPECTED_REVIEWER_TYPE="${EXPECTED_REVIEWER_TYPE:-Team}"
EXPECTED_REVIEWER="${EXPECTED_REVIEWER:-stewards}"
EXPECT_ENV_REVIEWER="${EXPECT_ENV_REVIEWER:-0}"

fail() {
  echo "::error::release environment settings audit failed: $*"
  exit 1
}

command -v gh >/dev/null 2>&1 || fail "gh CLI not found"

check_environment() {
  local env="$1"
  local policy="$2"
  local env_path="repos/${REPO}/environments/${env}"
  local policies_path="${env_path}/deployment-branch-policies?per_page=100"

  local env_data
  env_data="$(gh api "$env_path" --jq '[
    (.name // ""),
    (([.protection_rules[]? | select(.type == "required_reviewers") | .reviewers[]? | "\(.type):\(.reviewer.login // .reviewer.slug // .reviewer.name // "unknown")"] | join(", ")) as $reviewers | if $reviewers == "" then "__none__" else $reviewers end),
    (.deployment_branch_policy.custom_branch_policies | tostring),
    (.deployment_branch_policy.protected_branches | tostring)
  ] | @tsv')" || fail "cannot read ${env_path}"

  local env_name reviewer_summary custom_branch_policies protected_branches
  IFS=$'\t' read -r env_name reviewer_summary custom_branch_policies protected_branches <<< "$env_data"
  [ "$reviewer_summary" = "__none__" ] && reviewer_summary=""

  [ "$env_name" = "$env" ] || fail "${env}: environment not found"
  if [ "$EXPECT_ENV_REVIEWER" = "1" ]; then
    [ "$reviewer_summary" = "${EXPECTED_REVIEWER_TYPE}:${EXPECTED_REVIEWER}" ] \
      || fail "${env}: expected required reviewer ${EXPECTED_REVIEWER_TYPE}:${EXPECTED_REVIEWER}; found ${reviewer_summary:-none}"
  fi
  [ "$custom_branch_policies" = "true" ] \
    || fail "${env}: expected custom branch/tag policies"
  [ "$protected_branches" = "false" ] \
    || fail "${env}: expected protected_branches=false"

  local policies_data
  policies_data="$(gh api "$policies_path" --jq '[
    (.total_count | tostring),
    (.branch_policies[0].type // ""),
    (.branch_policies[0].name // "")
  ] | @tsv')" || fail "cannot read ${policies_path}"

  local policy_total policy_type policy_name
  IFS=$'\t' read -r policy_total policy_type policy_name <<< "$policies_data"
  [ "$policy_total" = "1" ] || fail "${env}: expected exactly one deployment policy"
  [ "$policy_type" = "tag" ] || fail "${env}: expected tag deployment policy"
  [ "$policy_name" = "$policy" ] \
    || fail "${env}: expected deployment policy ${policy}; found ${policy_name:-none}"
}

check_environment crates-io "ordinaldb-v*"
check_environment pypi "ordinaldb-py-v*"

echo "OK: release environment settings match the pre-tag policy."
