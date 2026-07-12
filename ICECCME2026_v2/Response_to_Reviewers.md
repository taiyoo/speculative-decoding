# Response to Reviewers (Point-by-Point)

Paper ID: 634  
Title: Characterizing Speculative Decoding Dynamics for Large Language Models on Consumer-Class GPUs

This document maps each reviewer comment in ICECCME_review_1.md to the revision action and manuscript location.

**Note on this revision (data-integrity pass):** All statistical and quality-metric responses below (R2-6 through R2-10) have been re-verified and, where the original response referenced numbers that were not yet backed by real measurements, updated to reference only genuine, independently-verifiable data computed directly from the project's real per-sample artifacts (`artifacts_submission/qwen3_rtx4090/results_csv/`) and real CNN/DailyMail outputs/references (`manifests/cnndm_data.json`). Where a requested analysis (the adaptive-$k$ hyperparameter ablation, R2-10) was not actually performed on real data, we say so explicitly rather than imply otherwise.

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

Author Response: Agreed. We expanded reporting beyond sample bootstrap summaries using real per-sample measurements.

Author Action: Added sample-level mean$\pm$std dispersion (computed across the real $n{=}1000$ evaluated samples per configuration), full TTFT/TPOT breakdowns, and Holm-Bonferroni-corrected one-sided paired $t$-tests with Cohen's $d$ effect sizes, comparing each fixed-$k$ configuration against its regime-matched baseline. We additionally ran seed-stability extensions (seeds 123 and 999, in addition to the primary seed 42) for both Qwen2.5 and Qwen3 at $k{=}4$ on both hardware profiles, providing initial seed-level stability evidence alongside the sample-level dispersion; full multi-seed repetition across the entire $k\in\{4,8,16\}$ grid was not performed and remains a threat to validity and future work.

#### R2-2
Reviewer Comment: ACSD section needs stronger validation and less position-paper style discussion.

Author Response: Agreed. We reduced speculative narrative and tightened empirical interpretation.

Author Action: Further condensed the adaptive-$k$ run-set discussion to an evidence-first summary and bounded deployment guidance, and removed formal/controller exposition that was not required for the paper's main claims. On review, we also found that a systematic hyperparameter ablation (rescue-threshold/window/cooling) had not actually been run for this mechanism; we removed all text implying such a sweep was performed and instead scope it explicitly as a single fixed-configuration study with the sweep identified as future work.

#### R2-3
Reviewer Comment: Presentation should be less repetitive and less overstated.

Author Response: Agreed. We tightened language and calibrated claims.

Author Action: Streamlined later sections and changed wording to trend-level guidance with explicit limits.

#### R2-4
Reviewer Comment: Summarization quality evaluation is weak if relying mainly on ROUGE-L.

Author Response: We agree this is a limitation and computed additional real quality metrics on the actual model outputs.

Author Action: Computed real BLEU and BERTScore alongside ROUGE-L on the saved CNN/DailyMail outputs and reference summaries, and reported all three; this surfaced a genuine speed-quality trade-off (quality loss largest at low $k$) not previously identified. Explicit limitations text on remaining evaluation gaps (factuality, task-specific utility) is retained.

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
| R2-6 | Missing variance across independent runs/seeds. | Addressed with real sample-level mean$\pm$std reporting ($n{=}1000$ real samples per configuration) for all six fixed-$k$ runs, plus seed-stability extensions (seeds 123 and 999) for both Qwen2.5 and Qwen3 at $k{=}4$ on both hardware profiles, showing low-to-modest seed-to-seed drift. Full multi-seed repetition across the entire $k\in\{4,8,16\}$ grid was not performed and is named as a threat to validity for future work. | Sec. 05 (Sample-level dispersion subsection, seed-stability paragraphs), Sec. 06 (Threats to validity) |
| R2-7 | TTFT and TPOT defined but not deeply analyzed. | Addressed with real per-configuration TTFT/TPOT mean$\pm$std computed from the same per-sample data underlying the headline tables (rather than a single qualitative sentence). These now match the headline tables exactly by construction, resolving a prior internal inconsistency between the two. | Sec. 05 (Sample-level dispersion subsection) |
| R2-8 | Summarization quality relies too much on ROUGE-L. | Addressed. We computed real BLEU (NLTK, method1 smoothing) and BERTScore (DistilBERT-base-uncased) in addition to ROUGE-L, directly on the saved real CNN/DailyMail model outputs and reference summaries ($n{=}200$). All three metrics are defined in the metrics table and reported for baseline and each fixed-$k$ configuration. The three move together and reveal a genuine, previously-unreported speed-quality trade-off: quality loss is largest at $k{=}4$ and partially recovers at higher $k$, the reverse of the speed trend. | Sec. 03 (Evaluation metrics), Sec. 05 (Sample-level dispersion subsection), Sec. 06 (Variance and evaluation depth) |
| R2-9 | Statistical testing details were light (paired tests, correction, effect). | Addressed with a real one-sided paired $t$-test ($H_0{:}\,S\le1$) per fixed-$k$ configuration against its regime-matched baseline, paired by sample, Holm-Bonferroni-corrected across all six configurations, with Cohen's $d$ effect sizes now reported. Five of six configurations show large, highly significant effects ($d$ up to 1.24, $p_{\text{adj}} \ll 0.001$); stochastic $k{=}16$ is correctly not significant, consistent with its below-baseline mean speedup. | Sec. 03 (Evaluation metrics), Sec. 05 (Sample-level dispersion subsection) |
| R2-10 | ACSD section reads like position paper; little ablation on thresholds/windows. | Partially addressed. On review, we confirmed that a systematic ablation across rescue-threshold, window-size, and cooling-period hyperparameters was never actually run for the adaptive-$k$ mechanism in this study — only a single fixed configuration was evaluated. We removed all manuscript language implying such a sweep had been performed (including the word "ablation" in the Introduction's contribution list and a Discussion sentence describing "threshold-sensitive gains... competitive in some regions"), and added an explicit Threats-to-Validity item stating the single-configuration scope, with a systematic sweep named as future work. We chose to disclose this gap rather than report an analysis that was not genuinely performed. | Sec. 01 (Contributions), Sec. 06 (Discussion, Threats to validity) |
| R2-11 | Repetitiveness in Sec. VII-VIII and overstated breadth. | Addressed. We removed a redundant Results subsection ("Summary of practical operating point") that restated trend claims already made in Discussion's "Main empirical takeaways," so each finding is now stated exactly once; terminology is shifted from ACSD to adaptive-$k$ run set throughout, and claims are calibrated to the tested hardware/model scope. | Sec. 05 (Results), Sec. 06 (Discussion) |
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
| Statistical transparency | Added real sample-level mean$\pm$std dispersion, Holm-Bonferroni-corrected paired $t$-tests, and Cohen's $d$ effect sizes, all computed directly from real per-sample measurements; added seed-stability extensions (seeds 123, 999) for both model families and both hardware profiles at $k{=}4$; full multi-seed repetition across the entire $k$ grid remains future work. |
| Quality metric breadth | Added real BLEU and BERTScore (previously ROUGE-L only), computed on the actual saved CNN/DailyMail outputs; revealed a genuine speed-quality trade-off (largest quality loss at low $k$). |
| Adaptive-$k$ ablation scope | Corrected: removed language implying a hyperparameter sweep was performed for the adaptive-$k$ mechanism; explicitly scoped as a single fixed configuration, with a systematic sweep named as future work. |
| Deployment guidance | Recommendation now explicitly conditional on hardware/runtime profile. |
