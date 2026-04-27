# Qwen2.5-7B vs Qwen2.5-3B as the Speculative-Decoding Target

The choice changes both **what speedup is achievable** and **which draft pairing makes sense**. The relevant variables are the per-step latency $t_{\text{target}}$, the draft/target cost ratio $c = t_{\text{draft}}/t_{\text{target}}$, and the achievable acceptance rate $\alpha$. Recall the per-cycle latency

$$L \;=\; \frac{T_{\text{draft}} + T_{\text{verify}}}{\tau} \;=\; \frac{k\,t_{\text{draft}} + t_{\text{target}}}{\tau},\qquad \tau = \frac{1-\alpha^{k+1}}{1-\alpha}.$$

## 1. Verifier cost ($t_{\text{target}}$)

| Target | Params | int8 weight footprint | Per-token verify (RTX 5090, our setup) |
|---|---|---|---|
| Qwen2.5-7B | 7.6 B | ~7.6 GB | ~109 ms (baseline 9.19 tok/s) |
| Qwen2.5-3B | 3.1 B | ~3.1 GB | ~50–55 ms (≈2× faster) |

A 3B target roughly halves $t_{\text{target}}$. That **shrinks the absolute latency budget** that speculation has to attack — the same algorithmic speedup is worth less wall-clock time.

## 2. Draft/target ratio $c$

The available drafts are 0.5B and 1.5B.

- With **7B target**: $c_{0.5} \approx 0.07$, $c_{1.5} \approx 0.20$. Both drafts are "cheap" relative to verification, so a wide range of $k$ is profitable. We measured best $S=1.65\times$ at $k{=}8$.
- With **3B target**: $c_{0.5} \approx 0.15$, $c_{1.5} \approx 0.45$. The 1.5B draft is now nearly half the cost of the verifier — speculation overhead eats most of the gain. Only the 0.5B draft remains useful, and the achievable speedup ceiling drops sharply (typical reports: $\sim$1.2–1.4×).

The break-even condition $c < (\tau-1)/k$ tightens as the target shrinks; on a 3B target, $k{=}16$ with the 1.5B draft is almost certainly below break-even at our observed $\bar\alpha \in [0.30, 0.45]$.

## 3. Acceptance rate $\alpha$

Acceptance depends on the agreement between the draft's and target's distributions.

- 0.5B → 7B: large capacity gap, $\bar\alpha \approx 0.30$–$0.40$ in our runs.
- 0.5B → 3B: smaller gap, expect $\bar\alpha \approx 0.45$–$0.55$ (the 0.5B draft is "closer" in capacity to a 3B target).
- 1.5B → 3B: drafts are now ~half the target's capacity; $\alpha$ rises further but $c$ kills the gain.

Higher $\alpha$ partially compensates for the smaller absolute budget, but only partially — the geometric law saturates and the linear-in-$k$ draft cost remains.

## 4. Quality and task fit

A 3B target has materially weaker reasoning and long-form quality:

- GSM8K (8-shot CoT): 7B ≈ 80%, 3B ≈ 65%.
- MMLU: 7B ≈ 74%, 3B ≈ 65%.

Since the speculative-decoding output distribution **equals** the target's, the target choice **caps task quality**. Picking 3B trades $\sim$10 quality points for at most $\sim$2× faster baseline latency.

## 5. Implication for DriftDiffuse

DriftDiffuse's value proposition is breaking the linear-in-$k$ draft cost. That argument is **strongest when the verifier is large**, because $T_{\text{verify}}$ is large enough to amortise a multi-step parallel drafter. On a 3B target, the parallel drafter would need $n_{\text{denoise}}$ very small ($\leq 2$) to remain profitable, which sharpens the design constraint considerably.

## TL;DR

| Aspect | 7B target (current) | 3B target |
|---|---|---|
| Baseline tok/s | 9.2 | ~17–18 |
| Best draft | 0.5B | 0.5B only |
| Achievable $S$ | ~1.5–1.65× (measured) | ~1.2–1.4× (expected) |
| $\bar\alpha$ with 0.5B | 0.30–0.45 | 0.45–0.55 |
| Task quality ceiling | high | ~10 pts lower |
| DriftDiffuse headroom | clear | marginal |

**Recommendation:** keep Qwen2.5-7B as the target. It is the configuration in which (a) speculation has enough verifier cost to attack, (b) both drafts are usefully cheap, and (c) DriftDiffuse's parallel-drafting advantage has room to pay off.
