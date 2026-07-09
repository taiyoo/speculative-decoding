# Comprehensive Review & Academic Quality Audit Report

**Paper**: "Characterizing Speculative Decoding Dynamics for Large Language Models on Consumer-Class GPUs" (ICECCME 2026, Paper ID 634)
**Sources audited**: `./Speculative Decoding Review.pdf` (3 reviewers), `./ICECCME2026/sec/*.tex`, `./ICECCME2026/Response_to_Reviewers.md` (authors' own prior rebuttal), `./ICECCME2026/main.bib`, and the underlying data artifacts referenced by the revised text (`Review_RTX4090/`, `Review_RTX5090_Laptop/`, `artifacts_submission/`, `src/`).

**Revision 3 update: the `.tex` files listed below WERE modified in this revision, at the user's explicit request, to remediate the data-integrity issue described below with genuine empirical data. See "✅ Remediation Complete" immediately below for what changed.**

---

## ✅ Remediation Complete (Revision 3)

All mock/simulated numbers identified in the Headline Finding (below) have been **purged and replaced with genuine empirical values** computed directly from the real per-sample artifacts in `ICECCME2026/artifacts_submission/qwen3_rtx4090/results_csv/` and `manifests/cnndm_data.json`. No simulated, hand-tuned, or fabricated numbers remain in `sec/*.tex`.

**Important premise correction.** The plan to "extract seed-level variance from `STABILITY_SEEDS = [42, 123, 999]` and real drift/stability CSVs" turned out not to be executable as originally assumed: `run_stability_analysis()` in `src/speculative.py` is real code but was **never actually run** (no `results/stability/` directory or `seed123`/`seed999` files exist anywhere in the repository — every real per-sample CSV for the Qwen3 RTX4090 fixed-$k$ configs contains exactly 1000 rows, all under `seed=42`), and the `results/drift_*.csv` files are from an unrelated, broken side-feature ("DriftDiffuse") with garbage outputs and placeholder `speedup=0.0`/`quality=0.0` fields. This was surfaced to the user directly, along with concrete real-data alternatives, before any file was touched (see chat transcript). The user chose, for each of three decision points: (1) replace seed-level variance with real **sample-level** dispersion (relabeled honestly, not conflated with seed-to-seed variance), (2) remove all ablation-sweep-implying language rather than fabricate a threshold sweep that was never run, and (3) install `nltk`/`bert_score`/`rouge_score` and compute genuine BLEU/BERTScore/ROUGE-L from the real saved CNN/DailyMail outputs.

**What is now in the paper, and where it came from:**

| Claim | Real source | New number(s) |
|---|---|---|
| Per-sample speedup dispersion (6 configs) | `spec_qwen3_0.6B_k{4,8,16}_{det,stoch}.csv` joined to `baseline_{deterministic,stochastic}.csv` by `sample_id`, $n{=}1000$ real samples/config, seed 42 | det: $2.7561\pm0.5950$ ($k{=}4$), $2.0010\pm0.6147$ ($k{=}8$), $1.3549\pm0.6533$ ($k{=}16$); stoch: $2.4945\pm0.4565$, $1.8303\pm0.5644$, $1.2173\pm0.6206$ |
| TTFT/TPOT dispersion | same real per-sample CSVs | TTFT det $132.57\pm73.85 \to 349.58\pm72.72$ ms ($k{=}4\to16$); TPOT det $41.23\pm9.59 \to 94.24\pm39.46$ ms (means now match Table V exactly — the former internal 132.57-vs-"50–52ms" contradiction is gone) |
| Paired significance + effect size | One-sided paired $t$-test (baseline vs. spec latency, paired by `sample_id`), Holm-Bonferroni-corrected across the 6 configs, Cohen's $d$ | 5/6 configs significant with large effect ($d{=}0.29$ to $1.24$, $p_{\text{adj}} \ll 0.001$); stochastic $k{=}16$ not significant ($d{=}-0.08$, $p_{\text{adj}}{=}0.996$), consistent with its $S{=}0.9719$ |
| CNN/DailyMail ROUGE-L | `rouge_score.RougeScorer(["rougeL"], use_stemmer=True)` — the exact function already used by `src/evaluate.py::cnndm_rouge_l`, applied to real `output_text` vs. real `reference` from `manifests/cnndm_data.json`, $n{=}200$ | baseline $19.58\pm5.61$; $k{=}4$: $11.50\pm4.85$; $k{=}8$: $14.04\pm4.93$; $k{=}16$: $14.92\pm5.14$ (baseline and $k{=}16$ figures independently reproduce the pre-existing `table_quality_comparison.csv` to 2 decimal places, confirming methodology consistency) |
| CNN/DailyMail BLEU | `nltk.translate.bleu_score.sentence_bleu` (method1 smoothing), same real texts | baseline $3.20\pm3.09$; $k{=}4$: $1.39\pm1.81$; $k{=}8$: $1.66\pm1.98$; $k{=}16$: $2.04\pm2.36$ |
| CNN/DailyMail BERTScore | `bert_score` library (`distilbert-base-uncased`), same real texts — genuinely distinct computation this time, not a ROUGE-L duplicate | baseline $0.793\pm0.031$; $k{=}4$: $0.710\pm0.039$; $k{=}8$: $0.729\pm0.038$; $k{=}16$: $0.741\pm0.037$ |
| ACSD/adaptive-$k$ hyperparameter ablation | N/A — confirmed no real sweep exists (only one fixed Qwen2.5 config was ever run) | All sweep-implying language removed; explicit new Threats-to-Validity item added stating the single-configuration scope and naming a systematic sweep as future work |

**A genuinely new, honest finding emerged from this real data**: CNN/DailyMail quality loss under speculative decoding is largest at $k{=}4$ (the fastest configuration) and partially recovers at higher $k$ — the reverse of the speed trend. This is a real speed–quality trade-off that was not previously reported (the mock data had shown no such pattern) and is now written up in `sec/06_discussion.tex`.

**Files changed**: `sec/00_abstract.tex`, `sec/01_introduction.tex`, `sec/01b_related_work.tex`, `sec/02_experiment.tex`, `sec/03_metrics.tex`, `sec/05_results.tex`, `sec/06_discussion.tex`. `main.bib` was not touched in this revision. Full diffs are in the working tree; every change is additive/corrective, no unrelated content was altered.

**Compilation validation**: `latexmk -pdf` on `conference_101719.tex` completes cleanly — **zero undefined citations, zero undefined references, zero LaTeX/BibTeX warnings** (`grep -c "Warning" conference_101719.log` → 0). The only non-zero `latexmk` messages are cosmetic `Underfull`/`Overfull \hbox` typesetting notices (sub-1pt table-width and paragraph-spacing badness scores), which are not warnings and are common in any two-column IEEE paper; they were spot-checked by rendering the affected pages to images and confirmed to display correctly with no visible overflow or misalignment. Output grew from 7 to 8 pages due to the added real content.

---

## ⚠ Headline Finding (historical record — now resolved, see above)

The revised text's answer to Reviewer #2's statistical-rigor requests — seed-level mean±std, TTFT/TPOT depth, Holm-Bonferroni paired testing, and expanded ROUGE-L/BLEU/BERTScore quality metrics (`sec/05_results.tex`, "Variance, TTFT/TPOT, and statistical testing detail"; `sec/06_discussion.tex`, "Variance and evaluation depth") — is **sourced entirely from a synthetic, hand-simulated dataset**, not from real repeated hardware runs on GSM8K/MMLU/CNN-DailyMail.

Concretely: `Review_RTX4090/new_experiments.ipynb` (and an identical copy in `Review_RTX5090_Laptop/`) states in its own header cell: *"The notebook is self-contained with a mock summarization dataset (50 samples), while importing PyTorch and Transformers to match the target stack."* Its `build_mock_dataset()` fabricates 50 fake "Article {i}: Analysts covering {topic}..." strings, and its `simulate_fixed_k_sample()`/`simulate_acsd_sample()` functions generate latency/acceptance numbers from **hand-picked parametric formulas plus Gaussian noise** (e.g. `ar_ttft = max(0.015, 0.055 + 0.028*diff + rng.normal(0, 0.003))`) — no model is ever loaded or run.

I confirmed this is not a benign coincidence: the exact numbers in the paper —
`sec/05_results.tex:118-121`: *"deterministic speedup means are $2.2876\pm0.0242$ ($k{=}4$), $2.2712\pm0.0295$ ($k{=}8$), and $2.0511\pm0.0273$ ($k{=}16$); stochastic means are $1.8910\pm0.0220$, $1.9055\pm0.0257$, and $1.6698\pm0.0133$"* — are bit-for-bit the same (to 4 decimal places) as `Review_RTX4090/Review/tables_new_experiments/fixedk_summary_mean_std.csv`, which is the direct output of the mock-data notebook above.

A second, compounding bug: in that same notebook, `bertscore_f1()` silently falls back to `rouge_l_f1()` whenever the `bert_score` package isn't available (`except Exception: vals = [rouge_l_f1(p, r) ...]`). The resulting CSVs (e.g. `fixedk_summary_mean_std.csv`) show `bertscore_mean` **identical** to `rouge_l_mean` in every single row — i.e., even within the synthetic experiment, "BERTScore" was never actually computed.

Finally, the TTFT number this subsection reports ("TTFT is stable (about 50–52 ms)") **contradicts** the paper's own real-hardware Table II two pages earlier, which reports TTFT = 132.57 ms for the *same* nominal configuration (Qwen3 0.6B draft, $k{=}4$, deterministic, RTX4090). A 2.6× discrepancy between two supposedly-comparable numbers in the same section, with no explanation, is itself strong internal evidence that the two numbers come from different (and undisclosed) sources.

**This is not flagged anywhere in the paper text or in `Response_to_Reviewers.md`.** The response letter states plainly that these items were "Addressed" (R2-6, R2-7, R2-8, R2-9, R2-10 in the authors' own table) with no caveat about simulation. Presenting simulated numbers as if they were measured seed-repeat/statistical-testing/quality-metric results, without disclosure, is a serious research-integrity risk if a reviewer or reader traces the numbers back to their source (as this audit did). This finding drives several "Missing"/"Partially Addressed" verdicts below and is the single most important item for the authors to resolve before submission — either by (a) re-running the seed/statistical/quality analysis on the real benchmark data and regenerating the reported numbers, or (b) if only the mock pipeline is currently feasible, explicitly and prominently disclosing in the paper that this specific subsection uses a simulated proxy pipeline and is illustrative of methodology rather than a measured result.

---

## Part 1: Reviewer Comments Reconciliation Status
* Total Comments Found: 18
* Perfectly Addressed: 11 *(was 5 in Revision 2 — R2-3, R2-4, R2-5, R2-6, R2-8, and R3-4 upgraded to Addressed in Revision 3 with real data; see their entries below)*
* Partially/Weakly Addressed: 7 *(R2-7 upgraded from Missing to Partially Addressed — the false claim is gone and honestly scoped as future work, but the reviewer's original ask for an actual ablation is still not delivered)*
* Unaddressed/Missing: 0

*(Counting distinct actionable concerns/requests from the review PDF, excluding pure praise/strengths. Several concerns are raised by more than one reviewer — each reviewer's phrasing is tracked as its own item since they may be weighted independently by an editor, but cross-references are noted.)*

### Detailed Reconciliation Breakdown

#### Reviewer Comment R1-1 (Reviewer #1)
`"The main weakness is that the proposed adaptive method does not outperform the best simple fixed setting... the new controller reads more like a robustness mechanism than a true performance improvement."`
- **Current Paper Status**: Addressed
- **Evidence in Source**: `sec/06_discussion.tex:21-28` ("Novelty and contribution scope": *"We do not claim a new speculative decoding algorithm that universally improves throughput... we provide a controlled, reproducible characterization..."*); `sec/06_discussion.tex:68-69` (Conclusion: *"The adaptive-$k$ run set remains useful as a runtime-stability policy layer, but is not presented as a universal throughput optimizer."*). The term "ACSD" has been fully removed from the manuscript (0 occurrences in `sec/*.tex`) and replaced by "adaptive-$k$ run set" throughout (14 occurrences), consistent with the authors' own `Response_to_Reviewers.md` (R1-4/R2-4).
- **Critique & Action Item**: This is a genuine, consistent reframing carried through Introduction, Implementation, Results, Discussion, and Conclusion — not just a one-off sentence. No further action needed.

#### Reviewer Comment R1-2 (Reviewer #1)
`"the evaluation is limited to one GPU, the RTX 4090, and one model family, Qwen2.5, so it is unclear whether the same trends would hold on other GPUs such as the RTX 3090, RTX 4080, or A100, or with other model families such as Llama or Mistral."`
- **Current Paper Status**: Partially Addressed
- **Evidence in Source**: `sec/00_abstract.tex:5-6` (*"a controlled study across two model families (Qwen2.5 and Qwen3) and two consumer setups (RTX4090 and RTX5090-laptop)"*); `sec/05_results.tex` Table (`tab:headline-5090`, RQ4) adds RTX5090-laptop; `sec/05_results.tex:138-158` ("Cross-family consistency") adds Qwen2.5.
- **Critique & Action Item**: A genuine second hardware profile and second model family were added — this is real, substantive work, not a cosmetic patch. However: (1) neither of the reviewer's specifically suggested GPUs (RTX 3090, RTX 4080, A100) nor cross-vendor families (Llama, Mistral) were added — only a same-vendor, same-family variant (Qwen3) and a laptop SKU of the same Ada/Blackwell lineage; (2) the added RTX5090-laptop evidence is a *negative* result (Table `tab:headline-5090`: all six configurations show $S<1$, i.e., speculative decoding is net-slower than autoregressive baseline on that hardware) — this actually reinforces the reviewer's underlying worry (results don't reliably generalize) rather than allaying it, even though it is honestly reported. Recommend: either soften the abstract's framing so "two consumer setups" doesn't read as fuller generalization evidence than it is, or add one truly independent GPU/vendor data point if feasible before camera-ready.

#### Reviewer Comment R1-3 (Reviewer #1)
`"I was also unconvinced by the CUDA graph discussion. The paper presents a CUDA-graph-capable setup, but later states that the capture rate was zero, which means the main results did not actually benefit from CUDA graph execution."`
- **Current Paper Status**: Partially Addressed
- **Evidence in Source**: `sec/04_execution.tex:35-41` ("CUDA-graph runtime note": *"In the Qwen3 run families, graph-capture metadata differs by hardware. The RTX4090 fixed-policy matrix reports stable captured execution, whereas RTX5090-laptop sensitivity runs show that graph-enabled toggles do not consistently improve end-to-end latency. We therefore treat CUDA-graph usage as an implementation detail rather than a standalone performance claim..."*)
- **Critique & Action Item**: The direct self-contradiction the reviewer flagged (claiming a CUDA-graph-capable setup while separately admitting 0% capture) has been removed — the text no longer claims a capture-driven speedup. However, no explicit capture-rate percentage or measurement methodology is given for *either* GPU now; the claim "RTX4090... reports stable captured execution" is asserted without a number, which is the same kind of unverifiable claim style the reviewer originally objected to, just pointed in the opposite direction. Recommend adding one concrete sentence with the actual measured capture rate (e.g., "captured in X% of decode steps under Y condition") to make this auditable rather than asserted.

#### Reviewer Comment R2-1 (Reviewer #2)
`"the novelty is somewhat limited... ACSD is presented more as an engineering policy/controller than a new decoding method, and its gains over the best fixed policy are not demonstrated."`
- **Current Paper Status**: Addressed
- **Evidence in Source**: Same as R1-1 above.
- **Critique & Action Item**: Same as R1-1 — no further action needed.

#### Reviewer Comment R2-2 (Reviewer #2)
`"the scope is narrow. All experiments are conducted on a single GPU type (RTX4090) and one model family (Qwen2.5)... Even adding one additional hardware class... would substantially strengthen the claims."`
- **Current Paper Status**: Partially Addressed
- **Evidence in Source**: Same as R1-2 above.
- **Critique & Action Item**: Same as R1-2 — the literal request ("even adding one additional hardware class") was fulfilled, but see caveat above about the negative-result framing.

#### Reviewer Comment R2-3 (Reviewer #2)
`"The paper lacks variance across repeated independent runs/seeds beyond bootstrap resampling over samples."`
- **Current Paper Status**: Addressed (with honest scope narrowing)
- **Evidence in Source**: `sec/05_results.tex` "Sample-level dispersion, TTFT/TPOT, and statistical testing detail" now reports genuine mean±std computed from the real per-sample CSVs ($n{=}1000$/config, `spec_qwen3_0.6B_k{4,8,16}_{det,stoch}.csv`), explicitly labeled as **sample-level** dispersion, not seed-level. `sec/06_discussion.tex` Threats to Validity adds a new item, "Sample-level versus seed-level dispersion," stating plainly that independent multi-seed repetition was not performed and is future work.
- **Critique & Action Item (Revision 3 resolution)**: The mock-data problem identified in Revision 2 is fully resolved — no simulated numbers remain. However, this is a genuine, real substitute for the reviewer's literal request, not the request itself: sample-to-sample dispersion (natural spread across different prompts within one run) and seed-to-seed dispersion (spread across independent re-executions with different random seeds) are different statistical quantities, and only the former is now reported. This is now transparently disclosed rather than silently substituted, which is the responsible way to handle a genuine data-availability gap. If reviewers still want true seed-repeat variance, real GPU time on matching RTX4090/RTX5090 hardware would be needed to run `STABILITY_SEEDS = [42, 123, 999]` for real.

#### Reviewer Comment R2-4 (Reviewer #2)
`"TTFT and TPOT are defined but not deeply analyzed despite being important deployment metrics."`
- **Current Paper Status**: Addressed
- **Evidence in Source**: The same real-data subsection now reports per-$k$, per-regime TTFT and TPOT mean±std (not a single vague sentence), and the previous internal contradiction (this subsection claiming "~50–52 ms" while Table V reported 132.57 ms for the same config) is gone — the real numbers now match Table V exactly by construction, since both come from the same source CSVs.
- **Critique & Action Item**: Resolved. A further enhancement (not required, optional future polish) would be explicitly tying the TPOT-vs-$k$ growth back to the $T_{\text{draft}} = k \cdot t_{\text{draft step}}$ relationship in Related Work Eq. 1, to strengthen the mechanistic story.

#### Reviewer Comment R2-5 (Reviewer #2)
`"The quality evaluation is somewhat weak for summarization; relying only on ROUGE-L is limiting."`
- **Current Paper Status**: Addressed
- **Evidence in Source**: `sec/03_metrics.tex` Table III now defines $Q_{\text{bleu}}$ (sentence-level BLEU, NLTK) and $Q_{\text{bert}}$ (BERTScore F1, DistilBERT-base-uncased) alongside $Q_{\text{sum}}$ (ROUGE-L), and `sec/05_results.tex`/`sec/06_discussion.tex` report genuine computed values for baseline and $k{=}4/8/16$ deterministic, all independently computed from the real saved `output_text` vs. real reference summaries in `manifests/cnndm_data.json` ($n{=}200$). The three metrics were verified to move together and are **not** duplicates of one another (unlike the Revision 2 mock data, where BERTScore silently equaled ROUGE-L in every row due to an unguarded fallback).
- **Critique & Action Item**: Resolved with real, distinct, independently-computed metrics. The paper now also reports a genuine, previously-unknown finding: quality loss is largest at $k{=}4$ and partially recovers at higher $k$, the reverse of the speed trend — a real speed-quality trade-off worth highlighting as a contribution in its own right.

#### Reviewer Comment R2-6 (Reviewer #2)
`"Statistical testing methodology is only lightly specified (e.g., exact paired tests, multiple-comparison corrections, effect sizes)."`
- **Current Paper Status**: Addressed
- **Evidence in Source**: `sec/05_results.tex` now reports a genuine one-sided paired $t$-test ($H_0: S\le1$) comparing each fixed-$k$ configuration against its regime-matched baseline, paired by `sample_id`, with real Holm-Bonferroni correction across all six configurations and real Cohen's $d$ effect sizes — computed directly with `scipy.stats.ttest_rel` on the real per-sample latencies. `sec/03_metrics.tex` Table III now defines both $p_S$ (Holm-Bonferroni corrected) and $d_S$ (Cohen's $d$) as reported metrics.
- **Critique & Action Item**: Fully resolved, and the numbers are genuinely informative: 5 of 6 configurations show large, highly significant effects; stochastic $k{=}16$ does not, consistent with its below-baseline mean speedup — a clean, real, internally-consistent statistical story.

#### Reviewer Comment R2-7 (Reviewer #2)
`"parts of the ACSD section read more like a position/design paper than a fully validated systems contribution... many policy thresholds and heuristics appear manually chosen, and there is little ablation analysis on rescue thresholds, switching policies, or cooling windows."`
- **Current Paper Status**: Partially Addressed (upgraded from Missing — integrity issue fixed, but the reviewer's underlying request is still not fulfilled)
- **Evidence in Source**: Investigation confirmed that real ACSD data exists for exactly **one** fixed configuration (Qwen2.5 RTX4090 only; `artifacts_submission/qwen25_rtx4090/results_csv/acsd_summary.csv`), not a sweep — the rescue-$\alpha$/window/cooling ablation grid only ever existed in the synthetic mock notebook. All sweep-implying language has been removed: `sec/01_introduction.tex` contribution item no longer says "and ablation"; `sec/06_discussion.tex` no longer claims "threshold-sensitive gains were competitive in some regions"; a new Threats-to-Validity item, "Adaptive-$k$ hyperparameter scope," now states plainly that a single fixed configuration was evaluated and a systematic sweep is future work.
- **Critique & Action Item**: The misrepresentation is fixed (no more implying a sweep that didn't happen), but this is honest retreat, not fulfillment — the reviewer explicitly asked for ablation analysis, and none is now presented (correctly, since none is real). If this comment needs to be more than "honestly scoped as future work" for the next review round, a real sweep would need to be run — ideally on Qwen3/RTX4090 to match the rest of the paper's evidence, which the current single real ACSD data point (Qwen2.5) does not.

#### Reviewer Comment R2-8 (Reviewer #2)
`"The paper is occasionally repetitive, especially in Sections VII–VIII."`
- **Current Paper Status**: Addressed
- **Evidence in Source**: The redundant `sec/05_results.tex` "Summary of practical operating point" subsection (which restated "deterministic $k{=}4$ best," "worsens as $k$ increases," and "trend portable, absolute gains not" — all already said in `sec/06_discussion.tex`'s "Main empirical takeaways") has been removed. Results now ends with "Cross-family consistency," which reports new data rather than repeating interpretation; the single restatement of the operating-point recommendation remains only in the Conclusion, which is the appropriate place for it.
- **Critique & Action Item**: Resolved.

#### Reviewer Comment R2-9 (Reviewer #2)
`"Some claims are overstated relative to the experimental breadth (e.g., broader deployment implications)."`
- **Current Paper Status**: Addressed
- **Evidence in Source**: `sec/00_abstract.tex:17-19` (*"this work provides reproducible, deployment-oriented characterization and policy guidance rather than a universally throughput-improving algorithm"*); `sec/05_results.tex:111-112` (*"conclusions are more portable than absolute speedup magnitudes"*); `sec/06_discussion.tex` "Threats to validity" section explicitly scopes hardware/runtime, model-family, and evaluation-breadth limits.
- **Critique & Action Item**: Claims are now consistently hedged across abstract, results, discussion, and a dedicated threats-to-validity section. No further action needed, beyond the R1-2/R2-2 caveat above about the RTX5090-laptop negative result being framed somewhat softly.

#### Reviewer Comment R2-10 (Reviewer #2)
`'Minor grammatical issues appear throughout (e.g., "How much end-to-end speedup does vanilla speculative decoding delivers...").'`
- **Current Paper Status**: Addressed
- **Evidence in Source**: `sec/01_introduction.tex:41-42` now reads *"How much end-to-end speedup does vanilla speculative decoding deliver against autoregressive decoding..."* — subject-verb agreement corrected.
- **Critique & Action Item**: Fixed. A full grammar/proofreading pass of the remaining sections is still recommended before camera-ready (not in scope of this citation/logic audit), but the specific example cited by the reviewer is resolved.

#### Reviewer Comment R2-11 (Reviewer #2)
`"Figure 2 is information-dense and somewhat difficult to interpret in grayscale print form."`
- **Current Paper Status**: Partially Addressed
- **Evidence in Source**: The current manuscript contains exactly **one** figure (`sec/04_execution.tex`, the TikZ draft/verify flow diagram, `fig:specdec-impl-flow`) — there is no "Figure 2" anywhere in `sec/*.tex` any more. A file `figures/specdec_configs_vs_baseline_metrics.pdf` (a Matplotlib-generated, presumably multi-panel configs-vs-baseline plot — consistent with "information-dense") still exists on disk but is not referenced by any `\includegraphics` call in the current sources. Meanwhile, `Response_to_Reviewers.md` (R2-13) claims: *"Partially addressed in text prioritization; figure-level redesign is planned for final camera-ready visual pass."*
- **Critique & Action Item**: The dense figure appears to have been **silently dropped**, not redesigned — its content is arguably now covered by Tables I–IV (Qwen2.5/Qwen3 fixed-policy tables), which is a defensible substitution (tables are often clearer than dense grayscale plots), but this is not what the response letter claims ("pending... refinement" implies it will still appear, redesigned) and is not explained anywhere in the paper. Recommend either (a) explicitly deciding tables replace the figure and removing the stale asset / updating the response letter to say so, or (b) if a redesigned Figure 2 is still intended for camera-ready, adding it now so this isn't an open loose end at submission time.

#### Reviewer Comment R3-1 (Reviewer #3)
`"the novelty and theoretical contribution of the work is not clear. The authors should consider revising the paper to also highlight the theoretical part of the work."`
- **Current Paper Status**: Partially Addressed
- **Evidence in Source**: `sec/06_discussion.tex:21-28` ("Novelty and contribution scope") narrows the contribution claim to "primarily empirical and systems-oriented," rather than adding new theoretical content.
- **Critique & Action Item**: This resolves the *clarity* half of the reviewer's comment (no longer ambiguous about what kind of contribution this is) but not the *"also highlight the theoretical part"* half — the reviewer's phrasing suggests they wanted more theory surfaced, not a disclaimer that there isn't much. The paper already has a small theoretical seed in `sec/01b_related_work.tex` Eq. 1 ($L = (T_{\text{draft}}+T_{\text{verify}})/\tau$); this could be extended into a short formal derivation of the observed "overhead inflection point" (the $k$ value beyond which $T_{\text{draft}}$ growth outpaces $B_{\text{eff}}$ gains) — which is exactly the empirical phenomenon Reviewers #1 and #2 both flagged as the paper's most interesting finding. Formalizing that relationship would satisfy R3-1 without overclaiming a "new algorithm."

#### Reviewer Comment R3-2 (Reviewer #3)
`"it would benefit the work if other methods can also be mentioned and compared along the proposed method used for the study."`
- **Current Paper Status**: Partially Addressed
- **Evidence in Source**: `sec/01b_related_work.tex` extensively surveys Medusa, the EAGLE family (1/2/3), DiffuSpec, SpecDiff-2, PARD, DFlash, CAS-Spec, and Faster Cascades, with a "Position of this work" paragraph explicitly contrasting the adaptive-$k$ framing against them.
- **Critique & Action Item**: The *mention* half is thoroughly done (arguably the strongest section of the paper, and independently verified for citation accuracy in `reference.md`). The *compare* half is conceptual only — there is no empirical head-to-head number (e.g., running Medusa or EAGLE-3 on the same RTX4090/Qwen setup, or even citing their reported consumer-GPU numbers side-by-side with this paper's Table II). Given the paper's own framing as an empirical characterization study, adding even a small qualitative comparison table (method / requires retraining? / target-independent? / reported consumer-GPU speedup range) would likely fully satisfy this reviewer without requiring new experiments.

#### Reviewer Comment R3-3 (Reviewer #3)
`"the format in the abstract should be revised, every word in the abstract should be bolded"`
- **Current Paper Status**: Addressed
- **Evidence in Source**: `sec/00_abstract.tex:2,20` wraps the entire abstract body in `{\bfseries\boldmath ... }`, which bolds all text *and* all inline math (confirmed rendering correctly in the compiled PDF from the prior citation-audit pass).
- **Critique & Action Item**: Fully and literally satisfied. No further action needed.

#### Reviewer Comment R3-4 (Reviewer #3)
`"there should be a brief introduction for related work so that it would be easier for the readers to follow."`
- **Current Paper Status**: Addressed
- **Evidence in Source**: `sec/01b_related_work.tex` now opens with a 3-sentence roadmap paragraph immediately after `\label{sec:related}`, naming the four categories of prior work reviewed (vanilla speculative decoding, drafting-cost reduction, diffusion-based drafters, cascaded/precision techniques) before the first `\paragraph`.
- **Critique & Action Item**: Resolved.

---

## Part 2: IEEE Academic Quality & Logical Flow Critique

* **Structure & Organization** *(Revision 3 update)*: The section-to-section skeleton (Intro → Related Work → Protocol → Metrics → Implementation → Results → Discussion/Threats/Conclusion) is sound and typical for an empirical systems paper, and the RQ1–RQ4 framing in the Introduction is a good organizing device that the Results section faithfully mirrors subsection-by-subsection. The two weakest joints identified in Revision 2 are now fixed: (1) Related Work now opens with a 3-sentence roadmap paragraph naming the four categories reviewed (R3-4, resolved); (2) the redundant "Summary of practical operating point" subsection has been removed from Results, so Results now stays empirical and Discussion holds the interpretive takeaway exactly once (R2-8, resolved).

* **Technical Depth & Rigor** *(Revision 3 update)*: The core experimental design (frozen manifests, paired regime-matched baselines, real Holm-Bonferroni-corrected paired tests with Cohen's $d$ effect sizes) is genuinely rigorous and now fully backed by real data — this is the paper's real strength, matching what Reviewer #2 called out as its best quality. The three concrete issues surfaced in Revision 2 are now resolved:
  1. **The mock-data provenance problem** — fully purged; see "✅ Remediation Complete" above for the complete real-data replacement.
  2. **The internal TTFT numerical inconsistency** (Table V's 132.57 ms vs. the old "~50–52 ms" claim) — gone; the real sample-level numbers now match Table V exactly by construction, since both are computed from the same source CSVs.
  3. **The duplicated "Secondary:" sentence** in `sec/03_metrics.tex` — removed; one clean sentence remains.

  Notation is consistent throughout ($S$, $\alpha$, $B_{\text{eff}}$, $k$, TTFT/TPOT, and now $p_S$/$d_S$/$\sigma_S$/$Q_{\text{bleu}}$/$Q_{\text{bert}}$ are used identically across Metrics/Results/Discussion), and the core mechanistic argument (Eq. 1's $L=(T_{\text{draft}}+T_{\text{verify}})/\tau$, and the observation that $B_{\text{eff}}$ rising while $S$ falls indicates an "overhead inflection") remains the paper's most academically satisfying piece of reasoning — formalizing it further would still directly help with R3-1 (theoretical contribution), which remains open.

* **Specific Polish/Revision Recommendations — status after Revision 3**:

  1. ~~`sec/03_metrics.tex` duplicate "Secondary:" sentence~~ — **Fixed.**
  2. ~~`sec/05_results.tex` mock-data variance/statistical-testing subsection~~ — **Fixed**, replaced with real sample-level dispersion, real paired tests, and real Cohen's $d$; see "✅ Remediation Complete" above.
  3. ~~`sec/01b_related_work.tex` missing intro~~ — **Fixed.**
  4. ~~`sec/06_discussion.tex` unsubstantiated BLEU/BERTScore claim~~ — **Fixed**, now backed by real, independently-computed numbers.
  5. ~~`sec/05_results.tex` + `sec/06_discussion.tex` redundant summary~~ — **Fixed**, redundant subsection removed from Results.
  6. **`sec/06_discussion.tex` Threats to Validity — ACSD ablation still not performed for real.** This remains open by design: no real rescue-threshold/window/cooling sweep exists (only one fixed Qwen2.5 configuration was ever run), so the honest fix was to remove the false claim and disclose the gap, not to fabricate a replacement. If a future revision wants to fully satisfy R2-7, a real sweep on Qwen3/RTX4090 (matching the rest of the paper's evidence) would need to be run and reported.

---

## Verdict Summary (Revision 3)

The paper's core empirical contribution (the RTX4090/RTX5090-laptop, Qwen2.5/Qwen3 fixed-policy characterization) is real, carefully measured, and well-reasoned — this part of the revision genuinely and substantively responds to the reviewers on scope (R1-2/R2-2), novelty framing (R1-1/R2-1), overstatement (R2-9), grammar (R2-10), and abstract formatting (R3-3). Related Work is thorough, now has a proper introduction (R3-4), and (per the prior citation audit in `reference.md`) is fully accurate.

The data-integrity issue that dominated Revision 2's findings — mock/simulated numbers standing in for seed-level variance, TTFT/TPOT depth, quality metrics, and statistical testing (R2-3 through R2-6) — has been **fully resolved with genuine empirical data**: real sample-level dispersion, a real paired-test-with-effect-size analysis (5 of 6 configurations significant, one genuinely not), and real ROUGE-L/BLEU/BERTScore computed from the actual saved CNN/DailyMail outputs, which surfaced a real, previously-unreported speed–quality trade-off. The one item that could not be honestly "fixed" rather than "honestly disclosed" is R2-7 (ACSD hyperparameter ablation): no real sweep data exists, and manufacturing one was correctly ruled out, so the paper now transparently scopes this as a single-configuration study with the sweep named as future work, rather than implying an analysis that didn't happen.

The manuscript compiles cleanly (`latexmk -pdf`, zero warnings, zero undefined references/citations) and is now, to the best of this audit's ability to verify, **free of fabricated or unverifiable claims**. Remaining open items (R1-3/CUDA-graph capture-rate transparency, R2-11/Figure 2, R3-1/theoretical depth, R3-2/empirical method comparison, R2-7/real ACSD ablation) are legitimate future-work items, not integrity issues.
