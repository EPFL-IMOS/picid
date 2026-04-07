# TabPHM PR Topology Reference (April 2026)

This document captures the current pull request topology and branch relationships, including the post-stack/08 extensions on top of the datasource refactor chain.

## Current Strategic Structure

- `interface_clean` is the clean integration base for the datasource refactor stack.
- Remote stack PRs `#13` to `#20` are chained and now consistent on `origin` from `interface_clean` through `stack/09`.
- The previously stale first edge was repaired by replaying the stack and force-updating remote stack branches with `--force-with-lease`.
- `stack/09-forecasting-umar_clean` is now stacked on top of `stack/08-numpy-doc-cleaning` (PR `#22`).
- Original `interface` PR `#8` remains open for traceability, but the active path is `#21` (`interface_clean`).

## Key PRs and Their Roles

- `#21` [Interface (clean)](https://github.com/EPFL-IMOS/TabPHM/pull/21)
  - `main` <- `interface_clean`
  - Purpose: clean interface integration branch (new baseline for stack).

- `#8` [Interface](https://github.com/EPFL-IMOS/TabPHM/pull/8)
  - `main` <- `interface`
  - Purpose: legacy/original interface PR with messy history; kept open as historical reference.

- `#13` [Stack 01/08](https://github.com/EPFL-IMOS/TabPHM/pull/13)
  - `interface_clean` <- `stack/01-datasource-core-contracts`
  - Purpose: first PR in the datasource refactor stack, now rebased in PR topology onto `interface_clean`.

- `#14` to `#20`
  - Continue the stacked chain from `stack/01` up to `stack/08`.

- `#10` [Forecasting: support + UMAR](https://github.com/EPFL-IMOS/TabPHM/pull/10)
  - `main` <- `david/experiments`
  - Purpose: legacy/original forecasting branch; now superseded for stack integration by `#22`.

- `#22` [Stack 09/09: Forecasting support + UMAR (ported)](https://github.com/EPFL-IMOS/TabPHM/pull/22)
  - `stack/08-numpy-doc-cleaning` <- `stack/09-forecasting-umar_clean`
  - Purpose: top-of-stack port of forecasting + UMAR onto the refactored datasource APIs.

- `#12` [New Results Dashboard and Experiment Fixes for TabPHM Paper](https://github.com/EPFL-IMOS/TabPHM/pull/12)
  - `main` <- `report_results_improvement`
  - Purpose: separate workstream, independent from interface+datasource stacking.

## Active Stacked Chain

Current chain (base <- head):

1. `#21`: `main` <- `interface_clean`
2. `#13`: `interface_clean` <- `stack/01-datasource-core-contracts`
3. `#14`: `stack/01-datasource-core-contracts` <- `stack/02-concrete-loader-migration`
4. `#15`: `stack/02-concrete-loader-migration` <- `stack/03-preprocessing-split-strategies`
5. `#16`: `stack/03-preprocessing-split-strategies` <- `stack/04-data-objects-split`
6. `#17`: `stack/04-data-objects-split` <- `stack/05-pronostia-raw-datasource`
7. `#18`: `stack/05-pronostia-raw-datasource` <- `stack/06-protocol-splits-and-cv`
8. `#19`: `stack/06-protocol-splits-and-cv` <- `stack/07-unit-metadata-propagation`
9. `#20`: `stack/07-unit-metadata-propagation` <- `stack/08-numpy-doc-cleaning`
10. `#22`: `stack/08-numpy-doc-cleaning` <- `stack/09-forecasting-umar_clean`

### Verified Ancestry Status (2026-04-06 preflight)

- `origin/interface_clean <- origin/stack/01-datasource-core-contracts`: **OK**.
- `origin/stack/01 <- ... <- origin/stack/09`: **OK**.
- Stacked PR merge states `#13` to `#20` and `#22`: **CLEAN**.

### Stack-09 Branch Normalization Status

- `stack/09-forecasting-umar_clean` is a normal remote branch on `origin` (tracking `origin/stack/09-forecasting-umar_clean`).
- PR `#22` is open and not draft, targeting `stack/08-numpy-doc-cleaning`.
- Verified ancestry: merge-base(`origin/stack/08-numpy-doc-cleaning`, `origin/stack/09-forecasting-umar_clean`) equals the current `origin/stack/08-numpy-doc-cleaning` tip.

## Planned Next Stack Addition

- Active next branch: `stack/10-threewloader-datasource` on `origin`.
- Active PR: `#23` (`stack/09-forecasting-umar_clean` <- `stack/10-threewloader-datasource`).
- Legacy local branch `stack/09-threewloader-datasource` was removed to avoid stack index ambiguity.

## Where `interface_clean` Came From

`interface_clean` did **not** come from rewriting `interface` in place.

It was created by a clean-port workflow:

1. Start from current `origin/main` in an isolated worktree.
2. Create new branch `interface_clean`.
3. Curate and replay the intended interface functionality from `origin/interface` onto `interface_clean`.
4. Keep `origin/interface` untouched (no history rewrite there).
5. Open a brand-new PR (`#21`) from `interface_clean` to `main`.
6. Retarget stack base (`#13`) to `interface_clean`.

### Important Safety Guarantees Used

- No force-push was used for creating/publishing `interface_clean`.
- No history rewrite was performed on `origin/interface`.
- Original PR `#8` was left intact.

## Practical Merge Guidance

Recommended merge order for this topology:

1. Merge `#21` (`interface_clean` into `main`).
2. Merge stack in order: `#13`, `#14`, `#15`, `#16`, `#17`, `#18`, `#19`, `#20`, `#22`, waiting for checks after each merge.
3. Add the `threewloader` branch as the next stacked PR after `#22` (recommended as stack/10 naming).
4. Handle unrelated PRs (`#12`, `#8`, and legacy `#10`) independently according to their own conflict/risk profile.

## Operational Notes for Future Branch Work

- Treat `interface_clean` as the canonical interface base for the datasource stack.
- Avoid mixing stack-chain retargeting with unrelated PR streams.
- If future interface fixes are needed before stack merge, put them on `interface_clean` (or a child branch/PR from it), not on legacy `interface`.
- Keep stack index naming unique (`stack/09`, `stack/10`, ...) to avoid branch/PR ambiguity.
- Keep this file updated whenever a stack base changes, a PR is superseded, or merge order shifts.
