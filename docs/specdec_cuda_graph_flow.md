# Speculative Decoding Flow: CUDA Graph vs Non-Graph

This flowchart is aligned to the control flow around _cuda_graph_possible and verify execution in src/speculative.py.

For CLI export, use the raw Mermaid source file:
- docs/specdec_cuda_graph_flow.mmd

Example export command:

```bash
npx -y @mermaid-js/mermaid-cli -i docs/specdec_cuda_graph_flow.mmd -o figures/specdec_cuda_graph_flow.png -b transparent
```

```mermaid
flowchart TD
  A[Start speculative_decode_sample]
  B[Init devices, buffers, stable_step_shapes]
  C{advanced CUDA path supported?}
  D[Disable stream pipeline<br/>Optionally relax fixed-shape path]
  E[Build graph_state.enabled via cuda_graph_possible]
  F{graph_state.enabled?}
  G[Force stream_pipeline_enabled = False]
  H[Optional stream pipeline for draft/verify]
  I[Per-step draft generation]
  J[Prepare target_input and verify_window]
  K{can_try_graph?<br/>enabled and not failed and cache supports crop and fixed_shape_step}
  L{Graph already captured?}
  M[Attempt capture<br/>warmup then torch.cuda.CUDAGraph]
  N{Capture success?}
  O[Set graph_state.failed = True<br/>graph_capture_note = capture_failed_fallback_eager]
  P{GPU_REQUIRE_CUDA_GRAPHS?}
  Q[Raise runtime error]
  R[Eager verify fallback]
  S[Graph replay path<br/>copy input then replay]
  T{Replay success?}
  U[Set failed and note replay_failed_fallback_eager]
  V[Eager verify fallback]
  W[Stream verify path]
  X[Eager verify path]
  Y[Verify block result: emitted, n_acc]
  Z[Crop caches, append tokens, EOS checks]
  AA[Record gpu_graph_capture note in output]

  A --> B --> C
  C -- No --> D --> E
  C -- Yes --> E
  E --> F
  F -- Yes --> G --> I
  F -- No --> H --> I

  I --> J --> K
  K -- No --> H2{verify_stream active?}
  H2 -- Yes --> W --> Y
  H2 -- No --> X --> Y

  K -- Yes --> L
  L -- No --> M --> N
  N -- Yes --> S
  N -- No --> O --> P
  P -- Yes --> Q
  P -- No --> R --> Y

  L -- Yes --> S
  S --> T
  T -- Yes --> Y
  T -- No --> U --> P2{GPU_REQUIRE_CUDA_GRAPHS?}
  P2 -- Yes --> Q
  P2 -- No --> V --> Y

  Y --> Z --> AA
```
