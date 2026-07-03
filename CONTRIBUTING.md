# Contributing

Welcome — and thank you for taking the time.

## How contributions flow

1. **Start with an issue.** Describe the bug, gap, or idea with enough
   detail that discussion can begin immediately: what you observed, what
   you expected, and why it matters.
2. **Design happens in the issue.** Propose an approach there if you have
   one; agreement on direction comes before any code.
3. **Code lands by invitation.** If you'd like to implement something
   yourself, say so in the issue and ask for contributor access. I look at
   the discussion so far and, if it's a fit, add you as a collaborator so
   you can open the PR.

An issue that never becomes your PR is still a real contribution — someone
else may pick it up, and the framing work is often the hard part.

Merges to `main` are done by the maintainer only.

## Why invitation-only PRs

Reviewing an unexpected PR means reverse-engineering its intent: what
problem it solves, whether the approach fits the project's direction,
whether the details are right. When that reconstruction costs more than
writing the change would have, the project loses time instead of gaining
it. Anchoring every change to a prior issue keeps review focused on the
code itself, because the intent is already settled.

Sharp problem framings, good questions, and benchmark observations are the
contributions this project is shortest on. Polished diffs without shared
context are not.

## Conventions for invited contributors

- Keep each PR to a single logical change; split refactors from behavior
  changes.
- Write commit titles in the imperative and use the body to explain the
  reasoning — the diff already shows the mechanics. Preserve multi-line
  body formatting (a shell HEREDOC works well).
- Link the motivating issue with `Closes #N` and describe how you verified
  the change.
- Contributions are accepted under the
  [Developer Certificate of Origin](DCO); sign off your commits with
  `git commit -s`.
- Commits co-authored with AI coding assistants may keep their
  `Co-Authored-By:` trailers.

## Adding framework integrations

New integrations belong outside the core surface. Structure them alongside
the existing adapters and validate behavior against the canonical in-tree
reference store for the framework in question.

## Building and testing

Commands for building, testing, and benchmarking live in the README's
[Build And Inspect](README.md#build-and-inspect) section.
