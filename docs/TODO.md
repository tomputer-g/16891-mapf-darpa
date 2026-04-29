# Report Discrepancy TODO

This is the short action list companion to
`docs/FINAL_REPORT_CODE_DISCREPANCIES.md`.

Priority is ordered by behavioral mismatch, not by small numeric mismatches.

## High Priority

1. Rewrite the overall system framing so the report says what the code does now:
   auction-based task allocation plus CBS coordination for active ground-agent
   paths, with drones planned independently.

2. Rewrite the building-investigation description:
   the code does not have a separate `InvestTask`, and ground sensing already
   reveals occupied vs empty buildings before the 1-step investigation task
   completes.

3. Rewrite SSIA / SSIA-Collateral / SSICA path-failure handling around the
   current repair-before-reauction heuristic:
   local repair first, compare against stored runner-up snapshot, then
   reauction only on repair failure, clear runner-up loss, or CBS failure.

4. Rewrite the global-reauction reset language:
   the preserved work is only active on-target dwell work, not all
   goal-reached work; in SSICA the queued tail is cleared.

5. Rewrite the Simulation Loop / Allocator Configurations sections:
   the greedy baseline and the SSIA-family methods do not use one identical
   harness, and the SSIA-family methods can auction/update multiple times
   within one outer step.

6. Add one explicit SSICA sentence that only the active queue head carries the
   repair snapshot metadata; queued tail tasks do not.

## Lower Priority

- Fix `R_triage` values in the report:
  - SSIA = `3`
  - SSIA-Collateral = `8`
  - SSICA = `8`

- Fix greedy-baseline wording so it says tasks are processed in queue order,
  not globally ordered by priority tier.

- Update the visualization caption:
  paths and task stars are per-agent colored, not fixed cyan/gold.

- Soften the drone-collision claim:
  CBS ignores drones, but allocator-level same-type next-move checks can still
  catch drone-drone conflicts.

## Verified / No Immediate Action Needed

- The abstract and results-table completion-step values currently match the
  checked root result files:
  - `naive_results.txt`
  - `ssia_results.txt`
  - `ssia_collateral_results.txt`
  - `ssica_results.txt`
