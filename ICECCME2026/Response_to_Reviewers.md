# Response to Reviewers (Point-by-Point)

Paper ID: 634  
Title: Characterizing Speculative Decoding Dynamics for Large Language Models on Consumer-Class GPUs

This document maps each reviewer comment in ICECCME_review_1.md to the revision action and manuscript location.

## Condensed Rebuttal for CMT (IEEE-Style)

We thank all reviewers for the constructive feedback. Below, each point follows the IEEE-style structure: Reviewer Comment, Author Response, and Author Action.

### Reviewer 1

#### R1-1
Reviewer Comment: ACSD appears more like a robustness mechanism than a true performance improvement.

Author Response: We agree. We now explicitly frame the adaptive-$k$ policy as a tail-latency and runtime-stability controller rather than a peak-throughput optimizer.

Author Action: Revised framing in Introduction, Implementation/Results adaptive-$k$ notes, and Discussion; removed language implying universal throughput superiority.

#### R1-2
Reviewer Comment: Scope is limited (single GPU and single model family), so generalization is unclear.

Author Response: We addressed this by expanding hardware coverage and preserving cross-family evidence.

Author Action: Added RTX5090-laptop results, retained Qwen2.5-family RTX4090 findings, and included explicit cross-family/cross-hardware trend-portability discussion.

#### R1-3
Reviewer Comment: CUDA-graph discussion was unconvincing because prior capture-rate reporting implied no effective benefit.

Author Response: We agree and further narrowed this discussion.

Author Action: Reduced CUDA-graph text to a runtime implementation note, explicitly stating that capture status does not imply standalone speedup benefit; retained only end-to-end trend claims and runtime-path dependence.

### Reviewer 2

#### R2-1
Reviewer Comment: Statistical depth should be improved (variance across seeds, stronger testing detail).

Author Response: Agreed. We expanded reporting beyond sample bootstrap summaries.

Author Action: Added seed-level mean+-std, TTFT/TPOT/P99 analysis, and corrected paired testing references (Holm-Bonferroni) in the results/statistics narrative.

#### R2-2
Reviewer Comment: ACSD section needs stronger validation and less position-paper style discussion.

Author Response: Agreed. We reduced speculative narrative and tightened empirical interpretation.

Author Action: Further condensed the adaptive-$k$ run-set discussion to an evidence-first summary (ablation ranges, corrected paired-test outcomes, and bounded deployment guidance), and removed formal/controller exposition that was not required for the paper's main claims.

#### R2-3
Reviewer Comment: Presentation should be less repetitive and less overstated.

Author Response: Agreed. We tightened language and calibrated claims.

Author Action: Streamlined later sections and changed wording to trend-level guidance with explicit limits.

#### R2-4
Reviewer Comment: Summarization quality evaluation is weak if relying mainly on ROUGE-L.

Author Response: We agree this is a limitation and expanded the reported perspective where available.

Author Action: Referenced multi-metric quality artifacts (ROUGE-L, BLEU, BERTScore) and added explicit limitations text on remaining evaluation gaps.

### Reviewer 3

#### R3-1
Reviewer Comment: Novelty/theoretical contribution is unclear.

Author Response: We clarified contribution scope and claim level.

Author Action: Reframed the paper as a reproducible systems characterization and deployment-guidance study, not a fundamentally new speculative decoding algorithm.

#### R3-2
Reviewer Comment: Include clearer comparison context with related methods.

Author Response: Agreed.

Author Action: Updated Related Work positioning and clarified adaptive-$k$ policy framing relative to training-heavy or architecture-modifying alternatives.

#### R3-3
Reviewer Comment: Abstract formatting suggestion (all-bold text).

Author Response: We adopted this formatting adjustment to match the conference template used for camera-ready preparation.

Author Action: Updated the abstract to use all-bold formatting, including bold equations, while preserving the revised technical content and scope calibration.

### Additional Clarification: Model-Family Applicability

Reviewer Comment: Findings may not transfer across model families.

Author Response: We agree that this is important for confidence in applicability.

Author Action: The paper now keeps both Qwen2.5-family RTX4090 fixed-policy findings and Qwen3-family results (RTX4090 and RTX5090-laptop). We report that low-k preference and deterministic advantage are consistent at the trend level, while absolute speedups remain hardware/runtime dependent.

### Overall Revision Outcome

Reviewer Comment: Core concern is claim strength versus evidence breadth.

Author Response: We aligned claims to evidence.

Author Action: Narrowed contribution claims, expanded hardware/model evidence, increased statistical transparency, and positioned the adaptive-$k$ run set as a stability-oriented mechanism with bounded claims.

## Detailed Point-by-Point Mapping

## Reviewer #1

| ID | Reviewer Comment | Response / Action Taken | Revision Location |
|---|---|---|---|
| R1-1 | Experimental design is clear and controlled (strength). | Thank you. We retained and emphasized the controlled protocol with frozen manifests and matched baselines. | Sec. 02 (Experimental Protocol), Sec. 01 (Contributions) |
| R1-2 | Larger k reduces speedup despite larger accepted blocks (strength). | Confirmed and re-reported with revised Qwen3 numbers. We now explicitly present the overhead inflection interpretation. | Sec. 06 (RQ2), Table headline for RTX4090 |
| R1-3 | Deterministic decoding is usually faster than stochastic (strength). | Confirmed with revised fixed-k matrix and explicit per-k deltas. | Sec. 06 (RQ3), Table headline for RTX4090 |
| R1-4 | ACSD does not beat best fixed setting; reads as robustness mechanism. | We keep the reviewer term ACSD here, but in the manuscript we reframe it as an adaptive-$k$ run set (a policy layer, not a new algorithm). Claims were narrowed accordingly. | Sec. 01 (Introduction framing), Sec. 04/06 (adaptive-$k$ notes), Sec. 05 (Discussion/Conclusion) |
| R1-5 | Scope limited to one GPU and one model family. | Addressed by adding cross-hardware comparison (RTX4090 and RTX5090-laptop) with Qwen3-family results and portability discussion. | Sec. 02 (Hardware/model scope), Sec. 06 (RQ4), Sec. 05 (Threats to validity) |
| R1-6 | CUDA graph discussion was unconvincing due to zero capture in prior run family. | Revised text now treats CUDA graphs as an implementation detail, avoids standalone graph-benefit claims, and keeps only end-to-end trend-level claims with explicit runtime-path dependence. | Sec. 04 (Implementation), Sec. 05 (Threats) |

## Reviewer #2

| ID | Reviewer Comment | Response / Action Taken | Revision Location |
|---|---|---|---|
| R2-1 | Controlled methodology and reproducibility are strong (strength). | Thank you. We preserved frozen-manifest paired design and clarified statistical reporting details. | Sec. 02, Sec. 03, Sec. 06 |
| R2-2 | Non-monotonic acceptance vs speedup relationship is useful (strength). | Retained and updated with revised Qwen3 fixed-k evidence and concise systems interpretation. | Sec. 06 (RQ2) |
| R2-3 | Transparent ACSD framing as tail-latency guardrail is appropriate (strength). | Kept and strengthened this framing; in manuscript text we rename ACSD to adaptive-$k$ run set to avoid implying a new standalone algorithm. | Sec. 01, Sec. 04/06, Sec. 05 |
| R2-4 | Novelty is limited; ACSD is an engineering policy, not a new algorithm. | Addressed by explicitly positioning contribution as empirical systems characterization and policy guidance, and by renaming ACSD to adaptive-$k$ run set in manuscript text to avoid algorithmic over-claiming. | Abstract, Sec. 01 (positioning/contributions), Sec. 05 (Novelty and scope) |
| R2-5 | Scope is narrow (single GPU, single model family). | Addressed by adding a second hardware profile (RTX5090-laptop) and trend-transfer analysis; claims are now bounded to trend portability versus absolute gains. | Sec. 02, Sec. 06 (RQ4), Sec. 05 (Threats/Conclusion) |
| R2-6 | Missing variance across independent runs/seeds. | Addressed with seed-level mean+-std reporting for fixed-k runs. | Sec. 06 (Variance subsection) |
| R2-7 | TTFT and TPOT defined but not deeply analyzed. | Addressed by adding TTFT/TPOT values in result tables and explicit discussion of token-timing behavior across k and regimes. | Sec. 06 (tables and RQ3 + variance subsection) |
| R2-8 | Summarization quality relies too much on ROUGE-L. | Partially addressed: quality discussion now references expanded multi-metric ablation artifacts (ROUGE-L, BLEU, BERTScore), while acknowledging remaining limits. | Sec. 06 (statistical detail paragraph), Sec. 05 (Evaluation breadth) |
| R2-9 | Statistical testing details were light (paired tests, correction, effect). | Addressed by specifying corrected paired tests (Holm-Bonferroni) and clarifying test usage in revised reporting. | Sec. 02 (Endpoints), Sec. 06 (statistical detail), Sec. 03 |
| R2-10 | ACSD section reads like position paper; little ablation on thresholds/windows. | Addressed with a condensed, ablation-backed summary and narrower claims; in manuscript text ACSD is expressed as adaptive-$k$ run-set behavior (threshold-sensitive and non-uniformly superior), with deployment guidance kept brief. | Sec. 06 (adaptive-$k$ comparison note), Sec. 05 (discussion summary) |
| R2-11 | Repetitiveness in Sec. VII-VIII and overstated breadth. | Addressed by integrating concise notes across implementation/results/discussion and reducing universal language in claims; terminology is shifted from ACSD to adaptive-$k$ run set in manuscript text. | Sec. 04, Sec. 06, Sec. 05 rewrites |
| R2-12 | Grammar issue: "does ... delivers". | Corrected. | Sec. 01 (RQ wording) |
| R2-13 | Figure 2 too dense for grayscale print. | Partially addressed in text prioritization; figure-level redesign is planned for final camera-ready visual pass. | Pending figure refinement in figures assets |

## Reviewer #3

| ID | Reviewer Comment | Response / Action Taken | Revision Location |
|---|---|---|---|
| R3-1 | Practical contribution is clear, but novelty/theoretical contribution is unclear. | Addressed by clearly narrowing novelty claims and strengthening formal systems interpretation (latency-overhead/acceptance relationship and portability limits). | Abstract, Sec. 01, Sec. 05 |
| R3-2 | Mention and compare with other methods. | Addressed in related work positioning and clearer distinction from Medusa/EAGLE/cascade approaches; we clarify where the ACSD (renamed as adaptive-$k$ run set in manuscript text) framing sits relative to training-heavy and architecture-modifying methods. | Sec. 01b (Related Work) |
| R3-3 | Abstract formatting request: every word bolded. | Adopted for camera-ready according to the provided conference template, including bold mathematical expressions in the abstract. | sec/00_abstract.tex, conference_101719.tex class/style compliance |
| R3-4 | Add brief related-work introduction to help readability. | Addressed by maintaining and refining dedicated related-work section and manuscript flow from introduction to related work. | Sec. 01, Sec. 01b |

## Global claim updates made in this revision

| Item | Action |
|---|---|
| Contribution scope | Reframed from universal acceleration claim to reproducible deployment characterization and policy guidance. |
| Hardware generalization | Added cross-hardware evidence (RTX4090 and RTX5090-laptop) and trend-vs-absolute portability distinction. |
| Adaptive-$k$ claim strength | Reframed as robustness/stability control with bounded, non-universal benefits. |
| Statistical transparency | Added seed-level variance and corrected paired-test reporting references. |
| Deployment guidance | Recommendation now explicitly conditional on hardware/runtime profile. |
