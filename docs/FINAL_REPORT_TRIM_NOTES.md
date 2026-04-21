# `reports/final_report.tex` Space-Reduction Notes

## Companion Repo Notes

The report draft now has companion engineering notes in the repo so the paper-writing context is easier to recover later:

- [before_after.md](before_after.md): benchmark comparison for the `ranais/refactor` branch
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md): top-level runtime map
- [README.md](README.md): detailed per-module reference set
- [FINAL_REPORT_CODE_DISCREPANCIES.md](FINAL_REPORT_CODE_DISCREPANCIES.md): report/code mismatches grouped by report section

If you resume report edits later, check those files first before re-reading the whole codebase.

All line references below refer to `reports/final_report.tex`.

Goal: recover roughly `0.5-1` IEEE conference page without changing the overall structure.

The report is currently repetitive in a few predictable places:

- the same system summary appears in the Abstract, Introduction, and Conclusion
- allocator behavior is explained once in Method and then again in Results
- scenario details are repeated in the figure intro, figure caption, and bullet list
- several subsections restate ideas that are already clear from the equations and table

My estimate: the cuts below can recover roughly `450-800 words`, which is usually enough for about `0.5-1 page` in this format.

## Highest-Yield Cuts

## 1. Abstract overlaps heavily with the Introduction and Conclusion

- Location: `final_report.tex:42`
- Issue:
  - The Abstract names all four allocators, explains their distinguishing features, reports the exact ranking, and restates heterogeneity.
  - The Introduction and Conclusion repeat almost all of that.
- Safe compression:
  - Keep the setup, the main technical idea, and one concise performance sentence.
  - Remove the final sentence about heterogeneous ground/aerial agents, since that is already explicit in the body.
  - Compress the long allocator-enumeration sentence.
- Likely savings: `40-70 words`

## 2. Introduction repeats the system summary that already appears in the Abstract

- Location: `final_report.tex:48-59`
- Issue:
  - Lines 50-55 restate the full system decomposition.
  - Line 57 repeats the agent heterogeneity summary that is later covered in Agent Model.
  - Line 59 repeats the performance claim.
- Safe compression:
  - Keep the problem statement and the 3-item list.
  - Cut the separate heterogeneity paragraph to one sentence.
  - Remove the final sentence at line 59 or fold it into the preceding paragraph.
- Likely savings: `50-90 words`

## 3. Related Work can be tightened without losing substance

- Location: `final_report.tex:66-78`
- Issue:
  - The auction subsection spends two full paragraphs on standard SSIA background before immediately restating your own contribution.
  - The CBS paragraph is generic textbook background and could be shorter.
- Safe compression:
  - Keep one sentence for Dias et al., one for repeated auctions, one for your distinction.
  - Compress the CBS paragraph to one sentence plus citations.
- Likely savings: `40-80 words`

## 4. Task semantics are repeated later in the bid-formulation section

- Location: `final_report.tex:115-123` and `final_report.tex:147-154`
- Issue:
  - Task Types already define exploration, investigation, triage, rewards, and dwell times.
  - Reward-Shaped Bid Formulation explains the same reward and dwell information again.
- Safe compression:
  - In Task Types, keep task meaning and completion condition.
  - In the SSIA bid section, reference the already-defined dwell/reward values instead of re-listing all of them in full.
- Likely savings: `40-70 words`

## 5. Greedy baseline paragraph is longer than necessary

- Location: `final_report.tex:127-132`
- Issue:
  - The paragraph explains the score, then adds several sentences comparing Greedy to SSIA.
  - Those comparison points reappear in Results.
- Safe compression:
  - Keep the equation and one sentence: greedy assigns each queued task to the nearest eligible idle agent under the lexicographic score.
  - Remove the final comparison sentence beginning “Like SSIA...”
- Likely savings: `30-50 words`

## 6. SSIA subsubsections can be merged more aggressively

- Location: `final_report.tex:136-187`
- Issue:
  - Availability, task ordering, sequential assignment, and agent-type constraints are each given their own heading and explanatory paragraph.
  - Some of that is already obvious from the algorithm block.
- Safe compression:
  - Keep `Reward-Shaped Bid Formulation` and `Global Reauction Trigger` as distinct subsections.
  - Merge `Availability Constraint`, `Task Ordering`, `Sequential Assignment`, and `Agent-Type Constraints` into a shorter combined paragraph.
  - Let Algorithm 1 carry more of the procedural explanation.
- Likely savings: `60-110 words`

## 7. SSIA-Collateral intro is wordier than it needs to be

- Location: `final_report.tex:214-235`
- Issue:
  - Line 214 explains “collateral exploration” in prose before the math immediately formalizes it.
  - Line 235 repeats the intuitive effect of the bonus, which can be said more briefly.
- Safe compression:
  - Replace line 214 with one sentence: SSIA-Collateral augments reward with an information-gain bonus based on newly revealed cells along the path.
  - Shorten the post-equation explanation to one sentence.
- Likely savings: `30-60 words`

## 8. SSICA method section is one of the biggest compression opportunities

- Location: `final_report.tex:237-250`
- Issue:
  - The section explains queue semantics, then re-explains them in the next paragraph, then adds a third conceptual paragraph, then adds a fourth speculative paragraph.
  - Much of this is later repeated again in `Why SSICA Underperforms`.
- Safe compression:
  - Keep:
    - one short paragraph defining queue-based bidding
    - Eq. `\ref{eq:bid_ssica}`
    - one short sentence saying queues provide look-ahead but can become stale in partially observed maps
  - Move the longer underperformance discussion to Results only.
- Likely savings: `100-170 words`

## 9. CBS section includes generic background that can be reduced

- Location: `final_report.tex:252-281`
- Issue:
  - The section is correct, but it explains standard CBS at textbook length.
  - The algorithm box already covers the high-level process.
- Safe compression:
  - Shorten the high-level CT-node description.
  - Compress Drone Handling to one sentence.
  - Keep only the implementation-specific details: ground agents use CBS, drones are planned independently, 5,000-node limit.
- Likely savings: `50-90 words`

## 10. Simulation Loop is slightly repetitive

- Location: `final_report.tex:310-321`
- Issue:
  - The list spells out observation and update logic twice across the two microsteps.
  - The final sentence restates the triage/no-movement behavior already described in item 6.
- Safe compression:
  - Merge items 2-5 into fewer steps.
  - Remove the separate “If no agent moves...” sentence and fold it into the triage-progress item.
- Likely savings: `25-45 words`

## 11. Test-scenario description is repeated in three places

- Location:
  - `final_report.tex:332`
  - `final_report.tex:337`
  - `final_report.tex:341-355`
- Issue:
  - The prose before the figure, the figure caption, and the DARPA-map bullets all repeat map properties.
  - “Starting zone” is repeated many times.
  - The figure caption is especially long.
- Safe compression:
  - Keep the figure caption, but shorten it by removing repeated color/marker explanations already stated in the preceding sentence.
  - In the bullet list, reduce each map to its one distinguishing feature.
  - Remove the final sentence about `darpa3`-`darpa7` being modeled after DARPA staging-area deployment, since it is already clear from the bullets and figure intro.
- Likely savings: `120-200 words`

## 12. Allocator Configurations repeats the Method section

- Location: `final_report.tex:360-368`
- Issue:
  - Each item restates behavior already explained in detail in Proposed Method.
- Safe compression:
  - Keep one short phrase per allocator and the metric sentence.
  - Example: “Greedy baseline,” “SSIA reward-shaped auction,” “SSIA with collateral bonus,” “SSICA queue-based concurrent variant.”
- Likely savings: `40-70 words`

## 13. Qualitative Analysis and later “Effect of ...” subsections overlap heavily

- Location:
  - `final_report.tex:385-390`
  - `final_report.tex:428-453`
- Issue:
  - The same four claims appear twice:
    - role specialization
    - reward shaping helps triage
    - collateral bonus improves exploration
    - CBS/reauction improves coordination
- Safe compression:
  - Choose one:
    - keep the qualitative bullet list and cut most of the later effect subsections, or
    - cut the qualitative bullet list and keep the later effect subsections.
  - If your goal is purely page recovery, the easiest move is to keep the table + key observations and shrink the qualitative bullet list to 1-2 sentences.
- Likely savings: `120-220 words`

## 14. Quantitative “Key observations” and the following subsections partially duplicate each other

- Location:
  - `final_report.tex:418-425`
  - `final_report.tex:437-453`
- Issue:
  - The observations list already states:
    - SSIA-Collateral is best
    - SSIA beats Greedy
    - collateral helps most on interior-start maps
    - SSICA underperforms
    - maze maps narrow the gap
  - The next subsections restate most of that.
- Safe compression:
  - Keep the numbered observations and shorten the later subsections.
  - Or keep the later interpretation subsections and compress the numbered list to 3 points.
- Likely savings: `60-120 words`

## 15. The Discussion “Strengths” section repeats the Introduction

- Location: `final_report.tex:458-464`
- Issue:
  - Modularity, adaptability, and heterogeneous coordination are all already established earlier.
- Safe compression:
  - Convert the 3 strengths into one short paragraph or cut the subsection entirely if space is tight.
- Likely savings: `40-70 words`

## 16. Limitations can be tightened without losing content

- Location: `final_report.tex:466-474`
- Issue:
  - Each limitation is a full mini-paragraph.
- Safe compression:
  - Keep the 4 headings but reduce each explanation to one sentence.
- Likely savings: `40-80 words`

## 17. Conclusion repeats the Abstract almost point-for-point

- Location: `final_report.tex:479`
- Issue:
  - It repeats the system framing, the four-way comparison, the main ranking, and the SSICA explanation.
- Safe compression:
  - Keep one concise summary sentence and one sentence on the main takeaway.
- Likely savings: `50-90 words`

## 18. Future Work is longer than needed for the current target

- Location: `final_report.tex:481-495`
- Issue:
  - Six future-work items is more than you need if the goal is just recovering 0.5-1 page.
- Safe compression:
  - Keep the three strongest:
    - improved CBS variants
    - combinatorial auctions
    - communication constraints or adaptive collateral weight
  - Remove or shorten the others.
- Likely savings: `70-130 words`

## Fastest Path To Recover 0.5-1 Page

If you want the least painful edits, I would cut in this order:

1. Shorten `Qualitative Analysis` and the later `Effect of ...` subsections so the same claims appear only once.
2. Compress `Test Scenarios` and the figure caption.
3. Trim the `SSICA` method section.
4. Shorten the `Conclusion and Future Work`.
5. Tighten the `Abstract` and `Introduction`.

That should usually be enough without changing the paper’s structure.

## If You Want A More Aggressive Trim

If you need a full page back, the easiest structural-but-safe reductions are:

- collapse `Effect of Reward Shaping`, `Effect of Collateral Exploration Bonus`, and `Effect of CBS Integration` into one short interpretation subsection
- reduce the DARPA-map bullets to one clause each
- cut the `Strengths` subsection entirely
- keep only 3 future-work items
