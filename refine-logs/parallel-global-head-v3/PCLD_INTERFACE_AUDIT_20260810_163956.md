# PCLD-16R Real-Interface Audit

## Confirmed source geometry

- scripts/train_domino_backbone_lora.py materialize_prompt_inputs with full_b16 builds the target teacher sequence as context + anchor + gold[0:15].
- With context length L, target_outputs.last_hidden_state[:, L:L+16] is therefore:
  - row 0: hidden after anchor, predicting gold token 0;
  - row 15: hidden after gold token 14, predicting gold token 15.
- The same materializer reconstructs all 16 labels and explicitly checks the stored canonical prefix.
- parallel_hidden_rows retains all 16 DFlash rows rather than the legacy 15-row supervision slice.

## Confirmed shared score interface

- third_party/Domino/code/dflash.py returns self.norm(hidden_states) from the DFlash block.
- Released inference computes base_logits = target.lm_head(parallel_hiddens) directly from those rows.
- The Qwen3-4B target checkpoint has hidden size 2560 and tie_word_embeddings=true.
- The authoritative lexical rows for the PCLD score identity are nevertheless defined as target.lm_head.weight[C]; tied input embeddings are an implementation fact, not an assumption in the contract.

Thus, in real arithmetic, the proposed identity is attached to the actual released score path:

\[
B(C,H)+W_{\mathrm{lm}}[C](T-H)=W_{\mathrm{lm}}[C]T.
\]

## Still unproven

- BF16 full-vocabulary GEMM and FP32 gathered dot can disagree near ties even when the real-arithmetic identity is exact.
- The cached target teacher pass and ordinary verifier may take different numerical paths.
- A shared rank-256 output subspace may not preserve enough candidate-score geometry on disjoint prompts.
- None of the above proves that the online global head can predict the clean residual.

Therefore implementation must preserve the original base score tensor for exact zero fallback and report direct-GEMM versus gathered-dot stable-row parity. Rank-256 PCA is diagnostic only: it is not a candidate-score upper bound and cannot replace the learned disjoint efficacy experiment.
