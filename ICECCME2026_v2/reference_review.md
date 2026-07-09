# Citation Accuracy Audit Report (ICECCME 2026)

## Overall Summary
* Total Citations Checked: 17 (currently active `\cite{}` keys in the `.tex` sources)
* Fully Compliant: 17
* At Risk / Requires Revision: 0
* Recommended for Deletion: 0
* Unverifiable (Reference PDF Missing): 0

**Update (Revision 3):** The `See2017` PDF has been supplied and verified — **Accurate**, and confirmed to be precisely matched to the `abisee/cnn_dailymail` dataset actually used in the experiment code. All 17 citations currently active in the paper are now fully verified against their source PDFs with no outstanding issues. The `Narayan2018` entry is retained below only as a historical record of the original hallucination finding (it is no longer cited in the paper).

**Prior updates:**
* **Revision 2:** The two previously-missing reference PDFs (`Stern2018blockwise`, `Sandler2025SpecDiff2`) were supplied and verified — both confirmed **Accurate**. Two discrepancies identified in Revision 1 (`Li2025Eagle3`, `Chen2026dflash`) were fixed in the LaTeX source; see each entry's **Resolution** line for what changed. The `Narayan2018` misattribution was resolved by swapping in `See2017` (pending PDF verification at that time).

---

## Detailed Audit List

### Leviathan2023
* **LaTeX Context**:
  1. `"...a smaller draft model that proposes a length-$k$ token block, followed by one target-side verification step; modified rejection sampling preserves the target distribution~\cite{Leviathan2023, Chen2023specsampling}."` (sec/01_introduction.tex:9)
  2. `"Leviathan \emph{et al.}~\cite{Leviathan2023} introduced the canonical draft--verify scheme: a smaller \emph{draft} model proposes $k$ tokens autoregressively and the larger \emph{target} verifies them in a single parallel forward pass, with modified rejection sampling guaranteeing that the output distribution is identical to that of the target."` (sec/01b_related_work.tex:5-9)
* **Corresponding PDF**: `./reference/Fast Inference from Transformers via Speculative Decoding.pdf`
* **Ground Truth Verification**: The paper's abstract and introduction confirm exactly this scheme: a smaller "approximation"/draft model proposes tokens, the larger target model verifies them in parallel, and a novel (modified) rejection/speculative sampling method guarantees the output distribution is unchanged ("without changing the model output distribution... with identical outputs").
* **Verdict**: Accurate
* **Recommendation**: Keep as is.

### Chen2023specsampling
* **LaTeX Context**:
  1. `"...modified rejection sampling preserves the target distribution~\cite{Leviathan2023, Chen2023specsampling}."` (sec/01_introduction.tex:9)
  2. `"Chen \emph{et al.}~\cite{Chen2023specsampling} concurrently proposed an equivalent scheme."` (sec/01b_related_work.tex:9-10)
* **Corresponding PDF**: `./reference/Accelerating Large Language Model Decoding with Speculative Sampling.pdf`
* **Ground Truth Verification**: Abstract states the algorithm generates multiple tokens per transformer call via a draft model scored by a target model, "combined with a novel modified rejection sampling scheme which preserves the distribution of the target model." This matches the "concurrently proposed an equivalent scheme" framing (both papers were released within days of each other in late 2022/early 2023).
* **Verdict**: Accurate
* **Recommendation**: Keep as is.

### Stern2018blockwise
* **LaTeX Context**: `"Stern \emph{et al.}~\cite{Stern2018blockwise} earlier explored blockwise parallel decoding within a single model."` (sec/01b_related_work.tex:21-22)
* **Corresponding PDF**: `./reference/Blockwise Parallel Decoding for Deep Autoregressive Models.pdf` *(supplied in Revision 2)*
* **Ground Truth Verification**: Abstract confirms: "we propose a novel blockwise parallel decoding scheme in which we make predictions for multiple time steps in parallel then back off to the longest prefix validated by a scoring model." Critically, the method is built as a **single combined scoring-and-prediction model**: the paper's Figure 3 shows the block predictors p2...pk are implemented as "a multi-output feedforward layer with residual connections after the original decoder output layer" of the *same* base Transformer p1, not as a separate model. This directly substantiates "within a single model," in contrast to the two-model (draft + target) speculative-decoding paradigm.
* **Verdict**: Accurate
* **Recommendation**: Keep as is.

### Cai2024Medusa
* **LaTeX Context**: `"Medusa~\cite{Cai2024Medusa} eliminates the external draft by attaching multiple prediction heads to the target model and verifying the resulting candidate tree in one forward pass."` (sec/01b_related_work.tex:25-27)
* **Corresponding PDF**: `./reference/MEDUSA Simple LLMInference Acceleration Framework with Multiple.pdf`
* **Ground Truth Verification**: Abstract confirms: "we present MEDUSA, an efficient method that augments LLM inference by adding extra decoding heads to predict multiple subsequent tokens in parallel. Using a tree-based attention mechanism, MEDUSA constructs multiple candidate continuations and verifies them simultaneously in each decoding step." This directly matches the "eliminates the external draft... attaching multiple prediction heads... verifying the resulting candidate tree in one forward pass" description.
* **Verdict**: Accurate
* **Recommendation**: Keep as is.

### Li2024Eagle
* **LaTeX Context**: `"The EAGLE family~\cite{Li2024Eagle,Li2024Eagle2,Li2025Eagle3} keeps an external draft but conditions it on target hidden features, reporting substantial speedups under their settings."` (sec/01b_related_work.tex:27-30)
* **Corresponding PDF**: `./reference/EAGLE Speculative Sampling Requires Rethinking Feature Uncertainty.pdf`
* **Ground Truth Verification**: The paper's Figure 1 reports speedups of 1.27x-3.07x across Vicuna/LLaMA2-Chat models, and EAGLE is built around conditioning draft generation on the target model's second-top-layer hidden features. Matches "keeps an external draft but conditions it on target hidden features, reporting substantial speedups."
* **Verdict**: Accurate
* **Recommendation**: Keep as is.

### Li2024Eagle2
* **LaTeX Context**: `"The EAGLE family~\cite{Li2024Eagle,Li2024Eagle2,Li2025Eagle3} keeps an external draft but conditions it on target hidden features, reporting substantial speedups under their settings."` (sec/01b_related_work.tex:27-30)
* **Corresponding PDF**: `./reference/EAGLE-2 Faster Inference of Language Models with Dynamic Draft Trees.pdf`
* **Ground Truth Verification**: Abstract confirms EAGLE-2 builds on EAGLE's draft model (conditioned on target hidden features) and reports speedup ratios of 3.05x-4.26x, "20%-40% faster than EAGLE-1." Consistent with "substantial speedups under their settings."
* **Verdict**: Accurate
* **Recommendation**: Keep as is.

### Li2025Eagle3
* **LaTeX Context**: `"The EAGLE family~\cite{Li2024Eagle,Li2024Eagle2,Li2025Eagle3} keeps an external draft but conditions it on target hidden features, reporting substantial speedups under their settings. Despite these refinements, $T_{\text{draft}}$ remains linear in $k$, forcing EAGLE-3 to a single transformer layer and capping reported gains at about $2$--$3\times$ on strong GPUs."` (sec/01b_related_work.tex:27-32)
* **Corresponding PDF**: `./reference/EAGLE-3 Scaling up Inference Acceleration of Large Language Models via Training-Time Test.pdf`
* **Ground Truth Verification**: The paper confirms the draft network uses "a single layer decoder" (line 272 of extracted text), supporting the "single transformer layer" claim. However, the paper's headline claim is a speedup "up to 6.5x" (idealized, batch-size-1 decoding-step metric, Table 1), not "2-3x." The "~2-3x" figure in the LaTeX text is only accurate for the paper's *production-framework* throughput numbers under realistic serving conditions (e.g., SGLang on a single H100: 373.25 vs. 158.34 tokens/s ≈ 2.36x at batch size 1; vLLM on A100 throughput gains fall in the 1.0x-1.75x range as batch size grows). The text conflates this narrower "strong GPU / production serving" number with the paper's general claim without clarifying it excludes the paper's own headline 6.5x figure.
* **Verdict**: Minor Discrepancy
* **Recommendation**: Revise text to clarify scope, e.g.: "...capping reported *production-throughput* gains (e.g., on H100/A100 serving frameworks) at about 2–3×, well below the paper's idealized best-case decoding speedup of up to 6.5×." This avoids implying 2-3x is the paper's overall claimed ceiling.
* **Resolution (Revision 2)**: Fixed. `sec/01b_related_work.tex` now reads: "...capping reported production-framework serving throughput (e.g., in SGLang/vLLM) at about $2$--$3\times$ on strong GPUs, well below the paper's idealized best-case decoding speedup of up to $6.5\times$." Scope is now explicit and no longer implies 2-3x is the paper's overall ceiling.

### Li2025DiffuSpec
* **LaTeX Context**: `"DiffuSpec~\cite{Li2025DiffuSpec} and SpecDiff-2~\cite{Sandler2025SpecDiff2} explore diffusion-based parallel drafting to improve speculative-decoding throughput."` (sec/01b_related_work.tex:36-37)
* **Corresponding PDF**: `./reference/DiffuSpec Unlocking Diffusion Language Models for Speculative Decoding.pdf`
* **Ground Truth Verification**: Abstract confirms DiffuSpec "uses a pretrained diffusion language model (DLM) to produce multi-token drafts in a single forward pass, while remaining compatible with standard AR verifiers," reporting "up to 3× wall-clock speedup." Matches "diffusion-based parallel drafting to improve speculative-decoding throughput."
* **Verdict**: Accurate
* **Recommendation**: Keep as is.

### Sandler2025SpecDiff2
* **LaTeX Context**: `"DiffuSpec~\cite{Li2025DiffuSpec} and SpecDiff-2~\cite{Sandler2025SpecDiff2} explore diffusion-based parallel drafting to improve speculative-decoding throughput."` (sec/01b_related_work.tex:36-37)
* **Corresponding PDF**: `./reference/SpecDiff-2 Scaling Diffusion Drafter Alignment For Faster Speculative Decoding.pdf` *(supplied in Revision 2)*
* **Ground Truth Verification**: Abstract confirms: "It leverages discrete diffusion as a non-autoregressive drafter to address bottleneck (1) [the autoregressive dependency during drafting which limits parallelism]... obtaining up to 5.5× average speed-up over standard decoding." This directly matches "diffusion-based parallel drafting to improve speculative-decoding throughput."
* **Verdict**: Accurate
* **Recommendation**: Keep as is.

### An2025PARD
* **LaTeX Context**: `"PARD~\cite{An2025PARD} focuses on a low-cost parallel draft model adaptation."` (sec/01b_related_work.tex:38-39)
* **Corresponding PDF**: `./reference/PARD Accelerating LLM Inference with Low-Cost PARallel Draft Model Adaptation.pdf`
* **Ground Truth Verification**: Abstract confirms: "we propose PARD (PARallel Draft), a novel speculative decoding method featuring target-independence and parallel token prediction... To further reduce the training adaptation cost of PARD, we propose a COnditional Drop-token (COD) mechanism... enabling autoregressive draft models to be adapted into parallel draft models at low-cost." Directly matches "low-cost parallel draft model adaptation."
* **Verdict**: Accurate
* **Recommendation**: Keep as is.

### Chen2026dflash
* **LaTeX Context**: `"DFlash~\cite{Chen2026dflash} uses a lightweight 5-layer block-diffusion drafter conditioned on target hidden features and reports up to $5\times$ speedup over autoregressive baselines on H200/B200 hardware."` (sec/01b_related_work.tex:39-41)
* **Corresponding PDF**: `./reference/DFlash Block Diffusion for Flash Speculative Decoding.pdf`
* **Ground Truth Verification**: Confirmed: draft model uses 5 layers ("we set the number of layers to 5... target hidden features are extracted from 5 layers"), conditioned on target hidden features, and all experiments run on NVIDIA H200 (Tables 1, 2, 5) or B200 GPUs (Tables 3, 4, SGLang serving benchmarks). The "up to 5x" figure matches Table 3's real-serving-scenario number ("achieving up to a 5.1× speedup on Qwen3-8B" on a single B200 GPU). However, the paper's headline/abstract claim is higher: "DFlash achieves over 6× lossless acceleration" and Figure 1/Table 1 report "up to a 6.1× speedup" (and even 6.5x+ in some Table 1 rows) under idealized H200 decoding-speedup benchmarks. So "up to 5×" understates the paper's overall best-reported number, though it is accurate for the specific serving/production benchmark context implied by "H200/B200 hardware."
* **Verdict**: Minor Discrepancy
* **Recommendation**: Either (a) update to "up to 6×" to match the paper's headline claim, or (b) keep "up to 5×" but clarify it refers specifically to real-world serving throughput (SGLang, B200) as distinct from the paper's idealized decoding-speedup ceiling of ~6.1-6.5× (H200, Table 1).
* **Resolution (Revision 2)**: Fixed via option (b). `sec/01b_related_work.tex` now reads: "...and reports up to $5\times$ speedup over autoregressive baselines in real-world serving scenarios on B200 hardware (over $6\times$ under idealized H200 decoding benchmarks)." Both figures are now attributed to their correct evaluation context.

### Ning2025CASSpec
* **LaTeX Context**: `"CAS-Spec builds an on-the-fly hierarchy of draft stages from a single target model using Dynamically Switchable Inference Acceleration (DSIA) strategies (for example, layer sparsity and activation quantization), coordinated by a Dynamic Tree Cascade (DyTC) algorithm~\cite{Ning2025CASSpec}."` (sec/01b_related_work.tex:44-48)
* **Corresponding PDF**: `./reference/CAS-Spec Cascade Adaptive Self-Speculative Decoding for On-the-Fly Lossless Inference Acceleration of LLMs.pdf`
* **Ground Truth Verification**: Abstract confirms: "CAS-Spec... constructs speculative draft models by leveraging dynamically switchable inference acceleration (DSIA) strategies, including layer sparsity and activation quantization... We introduce a Dynamic Tree Cascade (DyTC) algorithm that adaptively routes the multi-level draft models." Exact match to the LaTeX description.
* **Verdict**: Accurate
* **Recommendation**: Keep as is.

### Narasimhan2024FasterCascades
* **LaTeX Context**: `"Speculative Cascades applies a deferral rule, invoking larger models only for hard inputs while routing easier cases to smaller models~\cite{Narasimhan2024FasterCascades}."` (sec/01b_related_work.tex:49-50)
* **Corresponding PDF**: `./reference/Faster Cascades via Speculative Decoding.pdf`
* **Ground Truth Verification**: The paper coins the term "speculative cascades" (Section 4: "Speculative Cascades: Leveraging the Best of Both Worlds") and states cascades "employ a deferral rule that invokes the larger model only for 'hard' inputs," which their method implements via speculative execution. Matches the LaTeX description closely.
* **Verdict**: Accurate
* **Recommendation**: Keep as is.

### Dettmers2022Int8
* **LaTeX Context**: `"Low-precision techniques such as LLM.int8()~\cite{Dettmers2022Int8} reduce weight movement and can interact with speculative decoding in two opposing ways..."` (sec/01b_related_work.tex:61-64)
* **Corresponding PDF**: `./reference/LLM.int8 8-bit Matrix Multiplication for Transformers at Scale.pdf`
* **Ground Truth Verification**: Abstract confirms LLM.int8() is an Int8 quantization procedure that "cut[s] the memory needed for inference by half while retaining full precision performance," consistent with "reduce weight movement." The subsequent claim about interaction with speculative decoding (opposing effects on per-step cost vs. acceptance behavior) is the authors' own analytical argument, not attributed to the Dettmers et al. paper itself, and is presented appropriately as such (the paper does not discuss speculative decoding).
* **Verdict**: Accurate
* **Recommendation**: Keep as is.

### Cobbe2021
* **LaTeX Context**: `"\textbf{GSM8K}~\cite{Cobbe2021} test subset: 300 samples."` (sec/02_experiment.tex:8)
* **Corresponding PDF**: `./reference/Training Verifiers to Solve Math Word Problems.pdf`
* **Ground Truth Verification**: Abstract confirms: "we introduce GSM8K, a dataset of 8.5K high quality linguistically diverse grade school math word problems." Correct source for the GSM8K dataset.
* **Verdict**: Accurate
* **Recommendation**: Keep as is.

### Hendrycks2020
* **LaTeX Context**: `"\textbf{MMLU}~\cite{Hendrycks2020} subset: 500 samples (5 subjects $\times$ 100 each)."` (sec/02_experiment.tex:9)
* **Corresponding PDF**: `./reference/Measuring Massive Multitask Language Understanding.pdf`
* **Ground Truth Verification**: Abstract confirms: "We propose a new test to measure a text model's multitask accuracy. The test covers 57 tasks..." This is the MMLU benchmark paper. Correct source.
* **Verdict**: Accurate
* **Recommendation**: Keep as is.

### Narayan2018 — *(superseded, kept for audit trail)*
* **LaTeX Context**: `"\textbf{CNN/DailyMail}~\cite{Narayan2018} subset: 200 samples."` (sec/02_experiment.tex:10, **as it read prior to Revision 2**)
* **Corresponding PDF**: `./reference/Dont Give Me the Details, Just the Summary Topic-Aware Convolutional Neural Networks for Extreme Summarization.pdf`
* **Ground Truth Verification**: This paper is titled "Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization." It explicitly **introduces the XSum ("Extreme Summarization") dataset**, built from single-sentence BBC summaries, and its stated purpose is to create a task that "does not favor extractive strategies" — the opposite framing of CNN/DailyMail. The paper's own text repeatedly *contrasts* its dataset against CNN/DailyMail (e.g., "the performance of these models on CNN/DailyMail (See et al....)" is discussed as a separate, prior benchmark used by other systems, not the dataset this paper introduces). This is a clear **misattribution**: Narayan et al. (2018) is not the correct source for the CNN/DailyMail dataset.
* **Verdict**: Major Hallucination (Resolved)
* **Resolution (Revision 2)**: Checked `src/config.py` (`DATASETS["cnndm"]["hf_name"] = "abisee/cnn_dailymail"`, config `"3.0.0"`) and `src/data_loader.py` (`_load_cnndm` reads `item["article"]`/`item["highlights"]`) — the experiment **definitively used the real CNN/DailyMail dataset**, not XSum. Fixed: `sec/02_experiment.tex` now cites `\cite{See2017}` instead of `\cite{Narayan2018}` (text "CNN/DailyMail" unchanged), and a bib entry for See et al. 2017 was added to `main.bib`. This entry is retained here only as a historical record of the original finding — it is **no longer the active citation** in the paper. See the `See2017` entry below for the final verification of the citation now in use.

### See2017 — *(current citation, replaces Narayan2018)*
* **LaTeX Context**: `"\textbf{CNN/DailyMail}~\cite{See2017} subset: 200 samples."` (sec/02_experiment.tex:10, current text)
* **Corresponding PDF**: `./reference/Get To The Point Summarization with Pointer-Generator Networks.pdf` *(supplied in Revision 3)*
* **Ground Truth Verification**: The paper states: "We use the CNN/Daily Mail dataset (Hermann et al., 2015; Nallapati et al., 2016)... we operate directly on the original text (or non-anonymized version)" — i.e., See et al. (2017) is the paper responsible for popularizing the specific non-anonymized, original-text variant of the CNN/DailyMail summarization dataset. This is precisely the variant served by the HuggingFace dataset `abisee/cnn_dailymail` (config `3.0.0`) that `src/config.py` loads — the dataset repository is hosted under this paper's first author's namespace for exactly this reason. The citation is therefore both factually accurate (the paper genuinely uses and describes CNN/DailyMail, unlike the prior `Narayan2018` misattribution) and precisely matched to the actual data source used in the experiment.
* **Verdict**: Accurate
* **Recommendation**: Keep as is. (Optional: cite alongside Hermann et al. 2015 for full provenance of the underlying dataset, since See et al. build on that original QA dataset — not required, but common practice in summarization papers.)

---

## Notes on Bibliography Hygiene
* Three entries in `main.bib` — `Chang2022MaskGIT`, `Vaswani2017Attention`, and `Loshchilov2019AdamW` — were defined but never referenced via `\cite{}` anywhere in the `.tex` sources. **(Revision 2: disabled via a single `@Comment{...}` wrapper in `main.bib` rather than deleted, so they can be restored easily if a future revision cites them. Note: a bare `%` prefix does NOT work as a BibTeX comment — see the compilation note below.)**
* `Narayan2018` is now defined in `main.bib` but, like the three entries above, is no longer referenced via `\cite{}` anywhere in the `.tex` sources (superseded by `See2017`). It was left active rather than commented out since it may be worth citing intentionally alongside XSum-related discussion in the future; consider commenting it out too if it remains unused.
* `See2017` (added in Revision 2, verified in Revision 3) is now the active, fully-verified citation for the CNN/DailyMail dataset.

---

## Compilation Verification (Revision 3)
The full document (`conference_101719.tex`) was compiled with `latexmk -pdf` (pdfTeX via TeX Live 2025, IEEEtran class, BibTeX with `IEEEtran.bst`):
* **Result: Success.** `conference_101719.pdf` (7 pages) generated with **zero undefined citations and zero undefined references** on the final pass, and **zero BibTeX warnings/errors** (`.blg` reports `warning$ -- 0`; 17 entries processed, matching the 17 active `\cite{}` keys).
* **One real bug was caught and fixed during this step**: the initial `%`-prefixed "disabling" of the three unused bib entries (Revision 2) is not valid BibTeX syntax — BibTeX has no native line-comment character, so it still parsed `@inproceedings{...}` as real entries and threw "missing a field name" errors, which cascaded into 21 undefined-citation warnings across the whole document. Fixed by wrapping those three entries in a single `@Comment{ ... }` block instead, which BibTeX genuinely ignores. Recompiling afterward produced a clean build.
* The generated `.bbl` was spot-checked: all 17 `\bibitem`s render with correct author/title/venue text, and in-text markers resolve correctly (e.g., `CNN/DailyMail [17]` → the See et al. 2017 entry).
