# DFlash 全局候选选择 Head：源码审计、当前实现诊断与 GCLS-v1 设计

> 日期：2026-07-30  
> 状态：设计冻结前审计稿；尚未提交 GCLS-v1 正式训练  
> 首要验证范围：Qwen3-4B、DFlash-b16、greedy decoding（`T=0`）  
> 相关作业：数据采集 `10035436`；旧 SPH 六次训练 `10035437`；GCLS-v0
> capacity probe `10056893`

## 0. 一页结论

当前结果**不能否定“利用整个 DFlash candidate lattice 选择每个位置 token”这个
idea**，但也完全不能证明它有效。现有 GCLS-v0 的架构与训练方式确实有系统性
问题，最重要的结论如下。

1. 用户要验证的对象是：

   \[
   s_{i,k}=f_\theta(\text{整个 }L\times K\text{ 候选格子})_{i,k},
   \qquad
   \hat y_i=C_{i,\arg\max_k s_{i,k}},
   \]

   即每个位置的最终候选分数都直接利用整块信息。当前 GCLS-v0 的主监督对象却是
   “已知上一位置 gold candidate 后，下一条 pairwise edge 选哪个 token”。它主要
   学到了 teacher-forced transition，而不是干净的全局逐位置选择。

2. 当前 `local` 和 `global` 只在 candidate-node attention mask 上不同；二者最后
   都有跨位置一阶 pairwise transition。因此 `local` 不是“每个位置独立选择”的
   干净对照，`global-local` 也不能直接解释成全局候选信息的净收益。

3. 当前实现名为 `dpace` 的 loss 与官方 D-PACE 公式不一致：

   - 漏掉了当前位置自身的 \(q_i\)；
   - 用 `clamp(min=0.05)` 代替官方
     \(\tilde q_i=(1-\alpha)q_i+\alpha,\alpha=0.5\)；
   - 把位置权重重新归一化为均值 1；
   - 默认又乘了 `sqrt(gold_rank+1)`。

   这不是小超参数差异，而是 P0 级 objective implementation bug。

4. GCLS-v0 有 10,813,957 个参数，其中一个随机初始化的
   `151,936 × 64` trainable vocabulary table 就有 9,723,904 个参数，占
   89.92%。现有数据中只出现约 50,985 个 candidate token，未出现 token 的表项
   永远是随机的。这使 capacity probe 很容易记忆 token-specific bias，却严重损害
   泛化与可移植性。

5. `10056893` 只是在相同 512 blocks 上训练和评估的 memorization/capacity
   probe。global 模型能把 teacher-forced/pairwise 任务记到很高，但没有任何
   held-out 证据：

   | scope | direct unary EAL | pairwise greedy EAL | survival EAL | oracle EAL |
   |---|---:|---:|---:|---:|
   | local | 9.360 | 9.874 | 9.874 | 10.121 |
   | global | 8.361 | 10.047 | 10.049 | 10.121 |

   global 的**直接 unary 反而比 local 低 0.999**；收益来自 pairwise greedy，
   survival decoder 只额外贡献约 0.002。这证明模型有记忆 transition 的容量，
   不能证明“全局直接选择”成立。

6. 新的 GCLS-v1 应当先去掉 GRU、pairwise transition、Viterbi、survival
   product 和 trainable vocabulary table，只保留一个约 1.1M 参数、两层、
   240-node 的 bidirectional candidate transformer。输出必须是直接的
   per-position top-16 logits：

   \[
   s_{i,k}=\log p_{\mathrm{DFlash}}(C_{i,k})+\Delta_{i,k}.
   \]

   `Δ` 零初始化，故 epoch 0 的 argmax 必须与原始 DFlash 完全一致。

7. 第一阶段冻结 target 和已发布 DFlash 的全部参数，只训练外挂 head。这样才能
   做出清晰的 plug-in contribution。若后续要做 LoRA 或 joint training，只能作为
   第二阶段上界实验，不能混入首轮验证。

8. 当前 1,987 个 train prompts、15,886 个 correlated blocks 足够做单元测试、
   capacity gate 和 development gate，不足以支持正式结论。Domino 使用 1.42M
   prompts，DSpark 使用 1.3M prompts 并训练 10 epochs，DeLS-Spec 至少使用
   100K 文本。若 2K-prompt development gate 出现可信信号，应扩到至少 100K
   target-generated prompts，再做正式判断。

最终判断是：

> 目前效果不好的主要可确认原因是实现、目标和评测协议没有对齐，不能据此否定
> idea；但 top-16 oracle 只证明 availability，不证明 selectability。新的 direct
> global-vs-local、context-shuffle 和在线 rollout 实验才是对 idea 的有效检验。

---

## 1. 要解决的任务到底是什么

### 1.1 记号

- DFlash block size 为 \(B=16\)，其中一个 anchor 后有 \(L=B-1=15\) 个
  draft positions。
- 每个位置保留 DFlash top-\(K\) 候选，首轮取 \(K=16\)：

  \[
  C_i=\{c_{i,1},\ldots,c_{i,K}\}.
  \]

- \(h_i\in\mathbb{R}^{2560}\) 是 DFlash 在位置 \(i\) 的 parallel hidden。
- \(b_{i,k}\) 是 DFlash 对候选 \(c_{i,k}\) 的原始 full-vocabulary logit。
- \(y_i\) 是 target 在 gold prefix 上的 greedy token。
- 若 \(y_i\in C_i\)，其候选下标记为 \(r_i\)；否则该位置是 `OTHER`。

greedy 下一个 block 的 accepted draft length 是：

\[
A(\hat y,y)=\sum_{m=1}^{L}\prod_{i=1}^{m}\mathbf{1}[\hat y_i=y_i].
\]

这意味着第一个错误之后的所有预测在这一轮都没有收益。

### 1.2 “全局”应当严格是什么意思

这里的全局不是“最后用一个 CRF/Viterbi 在完整路径上求最优”，也不是“某个
辅助 confidence 读了全局 pooled feature”。本项目要验证的严格定义是：

\[
s_{i,k}
=f_\theta\!\left(
\{h_j,C_{j,1:K},b_{j,1:K}\}_{j=1}^{L},
\text{anchor/context feature}
\right)_{i,k}.
\]

对任意位置 \(i\)，它的每一个候选分数都可以直接看到：

- 当前与其他位置的 DFlash hidden；
- 所有位置的 candidate identities；
- 所有位置的 candidate logits、rank、margin、retained mass；
- 整个 block 中跨位置的模式一致性。

随后每个位置直接选择：

\[
\hat r_i=\arg\max_k s_{i,k}.
\]

输出可以一次并行产生。这里“逐位置选择”不等于“局部选择”：各位置最终
argmax 是独立执行的，但产生这些 logits 的网络具有完整 block receptive field。

### 1.3 什么不能输入

为了不作弊并保持部署一致，head 不能输入：

- target 在未来位置的 hidden/logits；
- \(y_i\) 或 gold rank；
- 由 target 产生的未来 token；
- 训练时才存在、推理时不存在的 teacher-forced draft prefix。

DFlash 的整张候选格子可以输入，因为它在真正 drafting 时可由一次 parallel
forward 得到，不包含未来 target token。

---

## 2. 当前 GCLS-v0 到底是什么

实现位于：

- `src/sph/candidate_lattice_selector.py`
- `scripts/train_candidate_selector.py`

### 2.1 输入张量

对 batch 中每个 block，当前 head 接收：

| 输入 | shape | 含义 |
|---|---|---|
| `hidden` | `[B,L,2560]` | DFlash parallel hidden |
| `candidate_ids` | `[B,L,K]` | DFlash top-K token ids |
| `candidate_embeddings` | `[B,L,K,2560]` | 冻结 target embedding rows |
| `candidate_logits` | `[B,L,K]` | DFlash top-K raw logits |
| `base_logsumexp` | `[B,L]` | DFlash full-vocab logsumexp |
| `anchor_ids` | `[B]` | 当前 block anchor token |

当前离线数据并没有存 target full distribution，也没有存每个 gold prefix 上的
target last hidden。它能监督 greedy hard-label selection，但不能复现 DSpark 的
full-vocabulary L1/TV distillation。

### 2.2 candidate node encoder

每个候选节点的初始 state 是以下六项之和：

1. 当前位置 DFlash hidden 的投影；
2. 冻结 target candidate embedding 的投影；
3. 随机初始化、可训练的 64-d vocabulary embedding 的投影；
4. position embedding；
5. candidate rank embedding；
6. 四个 scalar features 的 MLP：

   - top-K conditional log probability；
   - top-1 与当前 candidate 的 logit gap；
   - top-K retained full-vocabulary mass；
   - top-K conditional entropy。

随后运行两层 `d_model=128, n_head=8` 的 transformer。

### 2.3 local、causal、global 的精确定义

三个模型参数量完全相同，差别只在 node attention mask：

- `local`：一个 candidate node 只能 attend 同位置的 K 个 candidates；
- `causal`：可以 attend 当前和更早位置的 candidates；
- `global`：可以 attend 全部 \(L\times K\) candidates。

但是这三个 scope 后面**全部**接了相同的一阶 pairwise transition：

\[
e_i(u,v)=s_i^{\mathrm{unary}}(v)
+\lambda_{\mathrm{tr}}
\left\langle W_{\mathrm{prev}}z_{i-1,u},
W_{\mathrm{next}}z_{i,v}\right\rangle.
\]

所以当前 `local` 只表示“node mixer 是 local”，不表示整个 selector 是独立
per-position 的。它仍然利用上一位置 candidate identity 来选当前 token。

### 2.4 unary 分数

当前代码先在每个位置把 DFlash top-K logits 标准化：

\[
\bar b_{i,k}
=\frac{b_{i,k}-\operatorname{mean}_{k'}b_{i,k'}}
{\operatorname{std}_{k'}b_{i,k'}}.
\]

再形成：

\[
s^{\mathrm{unary}}_{i,k}
=\operatorname{softplus}(\alpha)\bar b_{i,k}
+w^\top z_{i,k}.
\]

这保留了同位置的 base 排序，却破坏了原始概率温度、absolute margin 和
full-vocabulary calibration。后面的 edge softmax 因而不能再解释成原始 DFlash
proposal probability。

### 2.5 pairwise 输出与 auxiliary heads

主要输出是 `[B,L,K,K]` 的 `edge_scores`。同时还有两个 pooled-position
binary heads：

- `in_lattice_logits`：预测 gold 是否在当前位置 top-K；
- `base_correct_logits`：预测 DFlash top-1 是否等于 gold。

`base_correct_logits` 被训练和报告 calibration，却没有进入主要 inference
decoder。`in_lattice` 表示 coverage，而不是“所选 token 正确的概率”，当前
survival decoder 却把它与 edge conditional probability 相乘，这两个事件的语义
并不相同。

### 2.6 参数组成

总参数量是 10,813,957：

| 部分 | 参数量 | 占比 |
|---|---:|---:|
| trainable token table `151936×64` | 9,723,904 | 89.92% |
| 其他 encoder/mixer/transition/heads | 1,090,053 | 10.08% |

capacity probe 里看到过的 unique candidate token 约 50,985，只占 vocabulary
约三分之一。这个 table 很适合记忆小数据中的 token bias，但不适合作为新 head
的主要 lexical representation。冻结 target embedding 已经提供了可泛化的 token
语义，不应再引入一个几乎占满参数预算的随机词表。

---

## 3. 当前训练与推理为什么没有对齐

### 3.1 当前主 loss 是 teacher-forced edge CE

对位置 \(i\)，代码先用上一位置的 gold candidate \(r_{i-1}\) 选择 edge row：

\[
\ell_i(\cdot)=e_i(r_{i-1},\cdot),
\]

然后才对当前 \(r_i\) 做 K-way CE。

这件事需要准确理解：

- 对 Domino/DSpark 这类 causal head，teacher forcing 本身是合理的。只有当前面
  draft tokens 都正确时，位置 \(i\) 才会被 target 检查并贡献 accepted length；
  在这个 accepted-prefix 条件下，前缀确实等于 gold。
- 但它监督的是**条件 transition**，不是用户提出的 direct global unary。
- 当前报告的 `candidate_accuracy` 也使用 gold predecessor row，因此不能当作部署
  中 direct per-position selector 的准确率。
- Viterbi/MAP 会使用大量非 gold predecessor rows，而这些 rows 没有被直接监督。

### 3.2 当前所谓 D-PACE 并不是 D-PACE

官方 D-PACE 对 gold token 的 draft probability \(q_i\) 做：

\[
\tilde q_i=(1-\alpha)q_i+\alpha,\quad \alpha=0.5,
\]

\[
P_m=\prod_{i\le m}\tilde q_i,
\qquad
w_j=\operatorname{stopgrad}\left(\sum_{m=j}^{L}P_m\right),
\]

\[
\mathcal L_{\mathrm{D\mbox{-}PACE}}
=\sum_j w_j(-\log q_j).
\]

官方实现最后按 block batch size 求和除法，不把 weights 再归一化为 token-wise
均值，也不叠加 rank weighting。

当前 `dpace_position_weights` 做的是：

\[
w_i^{\mathrm{current}}
=\underbrace{\prod_{j<i}\max(q_j,0.05)}_{\text{漏掉自身 }q_i}
\times
\underbrace{\left(1+\sum_{m>i}\prod_{j=i}^{m-1}\max(q_j,0.05)\right)}
_{\text{continuation}},
\]

随后又：

- 把所有 active weights 重标到均值 1；
- 默认乘 \((r_i+1)^{0.5}\)；
- 再次归一化。

因此当前 loss 的 gradient credit assignment 与论文、官方代码都不同。官方公式
和实现见 [D-PACE paper](https://arxiv.org/html/2605.18810) 与
[official repository](https://github.com/Lucas-TY/D-PACE)，本地对应
`third_party/D-PACE/specforge/core/dflash.py:234-264,365-376`。

### 3.3 当前总 loss

当前实际目标为：

\[
\mathcal L
=\mathcal L_{\mathrm{edge\ CE}}
+0.1\mathcal L_{\mathrm{coverage\ BCE}}
+0.25\mathcal L_{\mathrm{base\ correct\ BCE}}.
\]

其中 edge CE 默认还带错误版本的 D-PACE 与 rank-sqrt weighting。三个目标没有
一个直接监督 `unary_scores` 的全局逐位置 argmax。

### 3.4 当前四种 decoder

- `unary`：忽略 pairwise，只对 `unary_scores` 逐位置 argmax；
- `greedy`：用上一位置已选 candidate 的 edge row 逐步 argmax；
- `map`：对整张一阶 lattice 做 Viterbi；
- `survival`：将 edge softmax 与 coverage sigmoid 组合后做生存式路径选择；
- `keep_base`：根据在 validation 上校准的 margin，选择 base 或 learned path。

当前主要 checkpoint selection 使用 `survival`，但训练主目标只是 teacher-forced
row CE；coverage 不是 token-correctness；`base_correct` 又没有被 decoder 使用。
因此 training target、reported probability 和 deployment rule 没有统一的概率语义。

---

## 4. 已有实验应该如何解释

### 4.1 数据作业 `10035436`

数据采集正常完成：

| split | prompts | blocks |
|---|---:|---:|
| train | 1,987 | 15,886 |
| validation-select | 147 | 1,175 |
| validation-gate | 149 | 1,192 |
| total | 2,283 | 18,253 |

每个 prompt 最多 8 个 anchors；每个 block 保存 15 个 DFlash hidden、top-64
ids/logits、full-vocab logsumexp 和 target greedy gold。split 在 prompt 级隔离。

`validation-gate` 上：

- DFlash top-1 accepted-draft EAL：5.117；
- top-16 oracle accepted-draft EAL：9.916；
- oracle gap：+4.799；
- DFlash 首次错误位置的 target token 在 top-16 内约 92.27%。

这是非常强的 **availability upper bound**，但 oracle 在选择时直接看了 target。
它没有证明 draft-side features 能识别正确 candidate。

### 4.2 旧六次训练 `10035437`

这六次是旧的 rank-32 `no_mixer` SPH/absorbing-CRF，不是当前 GCLS-v0，更不是
本文提出的 GCLS-v1。它们连训练数据都只恢复约 2.4%–6.8% oracle gap，详细复盘
见 `docs/phase3_failure_analysis.md`。

### 4.3 GCLS-v0 capacity probe `10056893`

该作业故意选 512 blocks，并在相同 blocks 上训练和验证，只回答“能否记住”，
不能回答泛化。

| 指标 | local，selected epoch 46 | global，selected epoch 60 |
|---|---:|---:|
| teacher-forced candidate accuracy | 97.844% | 99.351% |
| teacher-forced non-top1 accuracy | 89.898% | 97.449% |
| base prompt-balanced EAL | 7.441 | 7.441 |
| direct unary EAL | 9.360 | 8.361 |
| pairwise greedy EAL | 9.874 | 10.047 |
| Viterbi MAP EAL | 9.852 | 9.517 |
| survival EAL | 9.874 | 10.049 |
| oracle EAL | 10.121 | 10.121 |
| first-miss repair rate | 99.61% | 100% |
| oracle gap recovered | 90.80% | 97.34% |
| harmed blocks | 0 | 0 |

local capacity gate 的 nominal failure 仅因为 hard accuracy 为 89.898%，低于 90%
阈值一个样本；不是训练崩溃。global 通过 capacity gate。

真正重要的是 decoder decomposition：

- global 相对 local 的 direct unary：`-0.999`；
- pairwise greedy：`+0.173`；
- MAP：`-0.335`；
- survival：`+0.175`；
- survival 相对 greedy 只多 `+0.002`。

所以目前唯一成立的结论是：

> 大容量 GCLS-v0 可以在 512 个已见 blocks 上记住 teacher-forced pairwise
> transition；当前实验没有证明直接全局 candidate scoring，且没有任何 held-out
> generalization 结果。

计划中的 full development jobs `10056894/10056895` 已取消，没有运行结果，不能
把 capacity 数字当作开发集结果。

---

## 5. DFlash 源码审计

论文与官方代码：

- [DFlash paper](https://arxiv.org/html/2602.06036)
- [DFlash repository](https://github.com/z-lab/DFlash)
- 本地 commit：`94e4abc5e0c31b67bc1a9d30f1cc34ece28a8756`

### 5.1 输入和前向

DFlash 先由 target 在当前 verified prefix 上产生：

- bonus/anchor token；
- 从若干 target layers 提取的 context hidden features。

draft block 输入是一个 clean anchor 加 \(B-1\) 个 mask positions。target
multi-layer features 经 projection 后持续注入每一层 draft transformer。block
内部是 bidirectional attention，所有 mask positions 在一次 forward 中产生 hidden：

\[
H_{1:L}=\mathrm{DFlash}(C_t,[x_{\mathrm{anchor}},M,\ldots,M]).
\]

随后用 target 的 LM head 一次并行投影：

\[
L_i^{\mathrm{base}}=\mathrm{LMHead}_{T}(H_i).
\]

官方 inference 在 `third_party/dflash/dflash/model.py:108-119` 明确使用
`is_causal=False`，并在一次 parallel forward 后应用 target LM head。

### 5.2 冻结与训练

DFlash 训练时：

- target model 冻结；
- target token embedding 与 LM head 共享给 draft，并冻结；
- 训练 draft transformer layers、target-feature fusion/projection 等 draft modules。

target 在 clean prompt+response 上提供 training context features；随机采样 response
中的 anchor，后面的 block positions 被 mask，训练目标是并行 next-block CE。不同
block 之间通过 attention mask 隔离。

原始 DFlash 使用位置指数衰减：

\[
w_k=\exp(-(k-1)/\gamma),
\]

强调早位置。它没有候选选择 head，也没有 intra-block generated-prefix input。

### 5.3 对本项目的启示

- DFlash hidden 已经包含 target prefix feature 与全 block bidirectional mask
  interaction，是新 head 最重要的 long-context input。
- 但 DFlash 每个位置最后仍单独用 LM head top-1/sample，缺少显式 candidate-level
  cross-position mode selection。
- 新 head 应复用 DFlash 一次 forward 的 hidden/logits，不重新训练 target，也不
  再做一个 full-vocabulary LM head。

---

## 6. Domino 源码与论文审计

论文与 reviewer code：

- [Domino paper](https://arxiv.org/html/2605.29707)
- [Domino repository](https://github.com/AI9Stars/Domino)
- 本地 commit：`930e5cd823f4bbbaa82ae150acad03928a3a859f`

训练代码没有在 reviewer repository 中完整发布；训练 objective 与数据细节以下以
论文为准，inference/模块结构以本地源码交叉验证。

### 6.1 Domino head 的输入

parallel DFlash-style backbone 先产生整个 block hidden \(H_i\)，冻结 target LM
head 得到 base logits：

\[
L_i^{\mathrm{base}}=\mathrm{LMHead}_{T}(H_i).
\]

Domino 的 GRU 读取 block 内已经实现的前缀 token embeddings：

\[
S_{i-1}=\mathrm{GRU}(E_{\le i-1}),\qquad d_S=1024.
\]

第 \(i\) 个位置的 correction head 输入是：

\[
[H_i;S_{i-1}],
\]

经过 rank 256 bottleneck 与 SiLU，输出 full-vocabulary residual：

\[
\Delta L_i=W_2\,\mathrm{SiLU}(W_1[H_i;S_{i-1}]),
\]

\[
L_i=L_i^{\mathrm{base}}+\Delta L_i.
\]

源码位置：

- `third_party/Domino/code/dflash.py:239-303`：GRU 与 low-rank head；
- `third_party/Domino/code/dflash.py:426-488`：parallel base logits 与 sequential
  correction loop。

### 6.2 冻结了什么

论文主训练：

- target model 始终冻结；
- target embedding 与 LM head 作为共享/投影模块冻结；
- parallel draft backbone 与 Domino head 一起训练。

reviewer code 另有 `freeze_backbone()` helper
（`third_party/Domino/code/dflash.py:558-571`），可在已有 backbone 上只训练
GRU/correction branch。但这只是可用的 plug-in fine-tuning helper，不能误写成
论文主结果的统一训练方式；论文的 base-anchored curriculum 明确需要优化 parallel
backbone。

### 6.3 训练 target 与 loss

Domino 对 GRU 使用 teacher forcing：训练时喂 ground-truth preceding token
embeddings。原因是只有 accepted prefix 对当前位置有用；错误 prefix 后本轮已经
停止接受。

只优化 final logits 会让 clean-prefix correction branch shortcut backbone，因此
Domino 联合监督：

\[
\mathcal L_t=(1-\lambda_t)\mathcal L_{\mathrm{final}}
+\lambda_t\mathcal L_{\mathrm{base}},
\]

其中 \(\lambda_t\) 从 1 线性退火到 0。两个 CE 都使用位置指数衰减。

### 6.4 数据和训练规模

- 1.42M Open-PerfectBlend prompts；
- responses 由相应 target model 重新生成；
- sequence length 3072，block size 16；
- 3 epochs，8×A100-80GB；
- global batch 16；
- AdamW，lr `6e-4`，weight decay 0，clip 1.0；
- cosine schedule，warmup ratio 0.04，bf16。

### 6.5 它与本 idea 的关系

Domino 解决的是“用已生成 prefix 补回 causal dependency”。它的第 \(i\) 个选择：

- 能看 \(H_i\)；
- 能看已生成的 \(x_{<i}\)；
- **不能在选择 \(x_i\) 时直接读取未来位置的 candidate sets/identities**。

所以 Domino 是强 causal baseline，不是本项目要验证的 direct global selector。
本项目如果只是把 GRU 换个名字，novelty 和 hypothesis 都不成立。

---

## 7. DSpark / DeepSpec 源码审计

论文与官方代码：

- [DSpark paper](https://arxiv.org/html/2607.05147)
- [DeepSpec repository](https://github.com/deepseek-ai/DeepSpec)
- 本地 commit：`005e03b81cec38b7da6399833d609ee89a2587f2`

### 7.1 parallel backbone

Qwen3-4B 官方配置：

- block size 7；
- 5 draft transformer layers；
- target feature layers `[1,9,17,25,33]`；
- 每条 sequence 随机 512 anchors。

与 DFlash 相似，anchor+mask block 在 draft backbone 内 bidirectionally attend，
一次产生全部 base hidden。

### 7.2 冻结边界

`Qwen3DSparkModel.initialize_embeddings_and_head(..., freeze=True)`：

- 从 target 拷贝 token embedding 与 LM-head weights；
- 将二者冻结；
- target model 不在普通 draft training forward 中参与反向传播；
- target hidden/last hidden 可预先缓存，target checkpoint 只在初始化时用于拷贝
  frozen embedding/head。

trainable 部分是：

- draft backbone 与 target-feature projection；
- Markov 或 RNN head；
- confidence head。

源码位置：

- `deepspec/modeling/dspark/qwen3/modeling.py:201-307`；
- `deepspec/trainer/base_trainer.py`；
- `deepspec/trainer/dspark_trainer.py`。

### 7.3 默认 Markov head

默认不是 RNN，而是 rank-256 first-order Markov：

\[
B(x_{i-1},\cdot)=W_1[x_{i-1}]W_2,
\]

其中 \(W_1\in\mathbb{R}^{V\times256}\)，
\(W_2\in\mathbb{R}^{256\times V}\)。final logits：

\[
L_i=L_i^{\mathrm{base}}+B(x_{i-1},\cdot).
\]

训练时 previous ids 是 `[anchor, gold_ids[:-1]]`；推理时 left-to-right sample，
再把 sample 作为下一步 previous id。

### 7.4 RNN variant

RNN variant 的输入为：

\[
z_i=[s_{i-1};W_1[x_{i-1}];h_i].
\]

它用一个 joint projection 产生 gate、candidate state 与 output state，并通过
\(W_2\) 输出 full-vocabulary bias。paper 报告 RNN 在更长 block 上只有小幅额外
收益，默认部署仍使用 Markov，因为实现更简单、延迟更好。

源码位于：

- `deepspec/modeling/dspark/markov_head.py:8-90`：VanillaMarkov；
- `...:93-123`：GatedMarkov；
- `...:125-225`：RNNHead。

### 7.5 confidence head

confidence 输入：

\[
[h_i;W_1[x_{i-1}]],
\]

目标不是“candidate 是否在 top-K”，而是 draft 与 target full distributions 的
analytical acceptance rate：

\[
c_i^*=1-\frac12\|p_i^d-p_i^t\|_1.
\]

每步 confidence 的 prefix cumulative product 用来决定验证到多长，而不是决定选
哪个 token。它还需要 post-hoc calibration；paper 明确观察到 raw estimator
over-confident。

### 7.6 DSpark loss

DSpark 同时使用：

1. ground-truth CE；
2. draft/target full-vocabulary probability L1；
3. confidence BCE。

Qwen3-4B 默认：

\[
\mathcal L=0.1\mathcal L_{\mathrm{CE}}
+0.9\mathcal L_{\mathrm{L1}}
+1.0\mathcal L_{\mathrm{conf}},
\]

位置权重 gamma 为 4。

源码：

- `deepspec/modeling/dspark/loss.py:60-87`：TV acceptance 与 L1；
- `...:90-163`：CE/L1/confidence terms；
- `...:227-252`：loss combination。

### 7.7 数据规模

- Open-PerfectBlend 1.3M prompts；
- 只使用原 prompts，由每个 target 重新生成 responses；
- 10 epochs；
- Qwen3-4B config：global batch 512，lr `6e-4`，wd 0，clip 1.0，
  warmup 0.04，bf16。

### 7.8 它与本 idea 的关系

DSpark 的核心也是 semi-autoregressive local dependency：

- backbone parallel；
- head 依赖上一 token 或生成 prefix；
- 默认 Markov 甚至不读其他位置候选。

它提供的重要训练启示是 full-distribution distillation 和语义正确的 acceptance
confidence，但不是 direct global selection 的实现模板。

---

## 8. DeLS-Spec：最需要正面对比的 plug-in baseline

论文与源码：

- [DeLS-Spec paper](https://arxiv.org/html/2607.07409)
- [DeLS-Spec repository](https://github.com/dt-3t/DeLS-Spec)
- 本地 commit：`ab9be1b4d4d470064cd98dd25f7cd1c124b86ad0`

DeLS-Spec 的研究定位与本项目非常接近：固定已有 DFlash，在后面挂一个轻量 local
head，不重新训练 DFlash。

### 8.1 local head

默认 local expert：

- 复用并冻结 target token embeddings；
- 一层 bias-free GRU，hidden 1024；
- rank projection 256；
- SiLU；
- low-rank full-vocabulary LM head。

它只读 block 内已生成 token prefix，不读 DFlash hidden，也不读 target hidden。

### 8.2 独立训练

local head 在 plain target-generated text 上做标准 next-token CE：

\[
\mathcal L_S=-\sum_i\log p_S(x_i\mid x_{<i}^{\mathrm{block}}).
\]

DFlash、target、local head 不需要 joint forward。Qwen3-4B 使用 DFlash authors 的
100K 数据，1 epoch、lr `6e-4`、block 16、512 anchors。

### 8.3 inference fusion

\[
\ell_{\mathrm{final}}
=\ell_{\mathrm{DFlash}}
+\alpha\ell_{\mathrm{short}}
-\beta\ell_{\mathrm{unigram}},
\]

默认 \(\alpha=\beta=0.3\)。第一 draft token 使用 DFlash，后续 token 顺序运行
local head。

源码：

- `third_party/DeLS-Spec/code/dels.py:140-310`：GRU local head；
- `third_party/DeLS-Spec/code/dflash.py:450-592`：DFlash+local fusion。

### 8.4 为什么必须作为 baseline

如果一个独立、只建模 short prefix 的 DeLS head 就能取得相同收益，那么无需全局
candidate lattice。GCLS 必须在相同 frozen DFlash 上显著优于：

- DFlash top-1；
- DeLS/Markov local causal correction；
- matched local candidate selector；
- matched causal candidate selector。

否则论文贡献最多是另一种 local correction，不是“全局信息选择候选”。

---

## 9. 方法对照表

| 方法 | 附加模块输入 | 位置 \(i\) 可见信息 | 选择过程 | 冻结边界 | 主要输出/目标 |
|---|---|---|---|---|---|
| DFlash | target context features、anchor+mask block | parallel block hidden；无已生成 block prefix | 全位置并行 | target、shared embed/head 冻结；draft backbone 训练 | full-vocab per-position CE |
| Domino | \(H_i\)+gold/generated prefix GRU state | long context + \(x_{<i}\) | sequential low-rank correction | target 冻结；论文主训练更新 backbone+head | base/final CE curriculum |
| DSpark | \(H_i\)+previous token/recursive state | long context + \(x_{<i}\) | default Markov sequential | target、shared embed/head 冻结；backbone+heads 训练 | CE+full-vocab L1+confidence |
| DeLS-Spec | generated prefix only | \(x_{<i}\) | local GRU sequential | target、DFlash、embed 冻结；只训 local head | independent next-token CE |
| GCLS-v0 | 整张 candidate lattice + previous candidate edge | attention scope +一阶 transition | greedy/Viterbi/survival | 离线 frozen features；训 selector | teacher-forced edge CE + two BCE |
| **GCLS-v1** | **整张 candidate lattice** | **所有位置 candidates/hiddens/logits** | **一次并行 direct per-position logits** | **target 与 DFlash 全冻；只训约 1.1M head** | **direct K-way exact D-PACE CE** |

---

## 10. 当前方案的全部问题

### 10.1 P0：不修就不能继续正式训练

#### P0-1：研究对象错位

用户的 hypothesis 是 direct global per-position selection；当前主输出和 loss 是
teacher-forced pairwise transition。即使它有效，也更接近一个 candidate-only
Domino/DSpark，而不是干净的 global selector。

#### P0-2：local control 不干净

local node attention 后仍有跨位置 pairwise edge。它不是 independent local
reranker，无法隔离“其他位置候选信息”的因果贡献。

#### P0-3：D-PACE 实现错误

漏自身 \(q_i\)、错误 smoothing、额外 normalization、rank-sqrt 权重，必须用
官方公式和数值单元测试替换。

#### P0-4：评价指标 teacher-forced

`candidate_accuracy` 给了上一位置 gold candidate。它可以作为 causal conditional
diagnostic，但不能作为 direct deployment accuracy，更不能作为全局 hypothesis 的
主要证据。

#### P0-5：direct global unary 没有被有效训练

capacity probe 中 global unary 比 local 低约 1 EAL；全部优势来自 pairwise greedy。
这说明现有实验没有命中要验证的输出。

#### P0-6：随机 trainable vocabulary table 主导模型

89.92% 参数是稀疏被访问的 token lookup table，未见 token 随机。它制造小数据
memorization，降低泛化，并掩盖候选语义是否来自 DFlash/target frozen features。

#### P0-7：base logits 被标准化

per-position z-score 删除真实 margin/temperature。之后产生的 softmax 不是 DFlash
proposal distribution，不能用于 calibrated survival，更不能直接用于 `T>0`
lossless verifier。

#### P0-8：coverage 与 correctness 混淆

`in_lattice` 的 event 是 \(y_i\in C_i\)，不是“当前选择正确”。把 coverage sigmoid
乘 edge probability 并称为 survival probability 没有一致的概率模型。

#### P0-9：`base_correct` 训练但部署不用

0.25 权重的 auxiliary 消耗 capacity 和 gradient，却没有进入 decoder。要么定义
明确的 abstain/KEEP_BASE decision 并单独校准，要么从 v1 主实验删除。

#### P0-10：没有 held-out GCLS 结果

`10056893` 是 same-512-block capacity probe；full development jobs 未运行。当前
任何“global 已有效”的结论都是越过证据等级。

#### P0-11：`T>0` proposal probability 未定义

若用新 head 选择/sample token，verifier 必须使用新 head 实际的 proposal
distribution \(q\)，不能继续拿原 DFlash probability 当 \(q\)。否则标准 rejection
sampling 不再 lossless。当前 hard-label 数据和 decoder 首先只支持验证 `T=0`。

### 10.2 P1：会影响泛化与最终论文结论

#### P1-1：prompt 数量不足

现有 train 只有 1,987 prompts；8 anchors/prompt 产生 15,886 blocks，但这些 blocks
高度相关。增加同 prompt anchors 不能替代 prompt diversity。

#### P1-2：没有 target distribution

当前只存 target greedy token。不能做 DSpark-style full-distribution L1/TV，也不能
学习语义正确的 acceptance confidence。

#### P1-3：采样分布与在线 decoding cycle 有差异

随机/even anchors 是合理开发数据增强，但最终要在真实 speculative rollout 中
验证：在线 anchor、上下文长度、domain mix 和模型自身 accepted-prefix 分布可能
不同。

#### P1-4：训练 schedule 未对齐成熟 recipe

v0 用 constant lr `1e-3`、wd `0.01`。DFlash、Domino、DSpark、D-PACE 普遍使用
`6e-4`、wd 0、warmup 0.04、cosine、clip 1.0。schedule 不是主失败原因，但正式
对照应统一。

#### P1-5：没有 latency / throughput 证据

accepted length 不是最终目标。head 的 candidate gather、attention、解码和 kernel
launch 必须计入端到端 latency；否则可能提高 EAL 但降低 speedup。

#### P1-6：oracle 被过度解释

top-16 oracle 使用 target knowledge。候选存在但 draft-side observables 可能无法
唯一识别 target choice，尤其开放式 chat 有多种合理模式。这是 idea 自身真实的
Bayes/identifiability ceiling。

#### P1-7：没有全局信息的负对照

即使 global 优于 local，也可能来自更多 capacity、position shortcut 或 prompt
memorization。需要 matched parameter counts、scope invariance test 和跨 block
context shuffle。

### 10.3 P2：第二阶段再处理

- top-K 取 8/16/32 的 latency-quality trade-off；
- axial attention 与 full attention 的速度差异；
- explicit `OTHER` / abstention；
- pairwise CRF v2；
- DFlash LoRA 或 joint tuning；
- variable block length；
- `T=1` full-distribution distillation；
- fused Triton/CUDA Graph inference。

这些都不应在 v1 一次加入，否则无法知道核心 global information 是否有用。

---

## 11. GCLS-v1：要真正训练的全局 head

### 11.1 冻结边界

首轮严格冻结：

- target model 全部参数；
- released DFlash 全部参数；
- target token embedding / LM head rows；
- 离线采集的 DFlash hidden/logits。

只训练 Global Candidate-Lattice Selector。这样如果性能提高，可以明确归因于
candidate selection，而不是把 DFlash 重新训练好了。

后续可做两个上界 ablation：

1. selector + DFlash LoRA；
2. selector + full DFlash joint training。

它们不能替代 frozen plug-in 主结果。

### 11.2 输入特征

对每个 candidate node \((i,k)\)：

1. `LN(h_i)`：DFlash parallel hidden；
2. `LN(E_T[c_{i,k}])`：冻结 target input embedding；
3. 原始 full-vocab log probability：

   \[
   \log p_D(c_{i,k})=b_{i,k}-\operatorname{LSE}_{V}(b_i);
   \]

4. top-K conditional log probability；
5. top-1 gap \(b_{i,1}-b_{i,k}\)；
6. top-K retained mass；
7. conditional entropy；
8. position embedding；
9. rank embedding；
10. anchor token 的冻结 embedding，使用与 candidate 相同的 projection，并广播到
    所有 nodes 或作为一个特殊 node。

禁止 trainable full-vocabulary table。

Qwen3-4B 的 input embedding 与 LM head tied。以后若模型 untied，可将 frozen input
embedding row 与 frozen LM-head row 分别低维投影后拼接，作为 portability ablation。

### 11.3 node encoder

\[
x_{i,k}=\mathrm{LN}\left(
W_h\mathrm{LN}(h_i)
+W_e\mathrm{LN}(E_T[c_{i,k}])
+W_s\phi_{i,k}
+p_i+r_k+a
\right).
\]

推荐首版：

- \(L=15,K=16,N=240\)；
- `d_model=128`；
- 2 transformer blocks；
- 8 attention heads；
- FFN multiplier 4；
- dropout 0；
- relative position bias \(i-j\)；
- same-position bias。

每层只有 \(240^2=57,600\) 个 attention pairs，远小于 target/DFlash 主体计算。
总 trainable parameters 约 1.1M。

### 11.4 matched scopes

同一套参数定义三种 attention masks：

- `local-direct`：node 只读同位置 K candidates；
- `causal-direct`：node 读当前位置和过去位置 candidates；
- `global-direct`：node 读所有 \(L\times K\) candidates。

三者：

- 参数量完全相同；
- 输入特征完全相同；
- loss 完全相同；
- 都没有 pairwise/GRU；
- 都直接输出 `[B,L,K]` logits。

这是第一次能够干净回答：

> 在每个位置选择 token 时，看其他位置候选是否真的比只看当前位置更好？

### 11.5 base-anchored direct output

最终输出：

\[
\Delta_{i,k}=w_o^\top z_{i,k},
\]

\[
s_{i,k}=\log p_D(c_{i,k})+\Delta_{i,k}.
\]

要求：

- output projection 零初始化；
- 不学习 base scale；
- 不标准化 base logits；
- epoch 0 对每一个 block、每一个位置的 argmax 必须等于 DFlash top-1；
- 可令 \(\Delta_i\leftarrow\Delta_i-\operatorname{mean}_k\Delta_{i,k}\)，去掉无意义的
  common shift。

这是 frozen-DFlash 场景下比 Domino curriculum 更合适的 base preservation：
DFlash backbone 根本不会 collapse，identity residual 初始化即可。不要机械照搬
Domino 的 base loss curriculum。

### 11.6 v1 不要加什么

第一版明确不加：

- GRU；
- Markov pairwise transition；
- Viterbi；
- survival product；
- trainable token table；
- `base_correct` auxiliary；
- coverage × edge probability；
- KEEP_BASE threshold；
- DFlash joint training。

如果 direct global 不先通过，继续堆结构只会再次失去可解释性。

---

## 12. GCLS-v1 的正确训练目标

### 12.1 active positions

若 target token 在某位置不在 top-K，candidate-only selector 无法为该位置得到
K-way gold label，本轮 oracle accepted prefix 也在这里终止。因此 active mask 是：

\[
m_i=\mathbf 1[y_1\in C_1,\ldots,y_i\in C_i].
\]

即只监督直到第一个 top-K miss 之前、且包含当前位置的可观测 candidate labels。

### 12.2 direct candidate probability

\[
q_i
=\operatorname{softmax}(s_i)_{r_i}.
\]

这里 q 来自 direct global logits，不再来自 teacher-forced edge row。

### 12.3 exact D-PACE

默认 \(\alpha=0.5\)：

\[
\tilde q_i=(1-\alpha)\operatorname{stopgrad}(q_i)+\alpha.
\]

对 active positions 做 inclusive prefix product：

\[
P_i=\prod_{j\le i}\tilde q_j.
\]

位置 \(i\) 的 detached weight：

\[
w_i=\operatorname{stopgrad}\left(\sum_{t=i}^{L}P_t m_t\right).
\]

最终：

\[
\mathcal L_{\mathrm{GCLS}}
=\frac1{\text{batch size}}\sum_{\mathrm{blocks}}\sum_i
m_iw_i(-\log q_i).
\]

不做 token-count mean normalization，不乘 rank-sqrt weight。

对应伪代码：

```python
gold_log_q = log_softmax(scores.float(), dim=-1).gather(
    -1, safe_gold_rank[..., None]
).squeeze(-1)
q = gold_log_q.exp()

with torch.no_grad():
    q_tilde = (1.0 - alpha) * q + alpha
    q_tilde = torch.where(active, q_tilde, torch.ones_like(q_tilde))
    prefix = torch.cumprod(q_tilde, dim=1)  # inclusive: contains q_i
    suffix = torch.flip(
        torch.cumsum(
            torch.flip(prefix * active.float(), dims=[1]),
            dim=1,
        ),
        dims=[1],
    )
    weights = suffix

loss = (-(gold_log_q) * weights * active.float()).sum() / batch_size
```

必须用官方 D-PACE 实现和手算小例子做逐元素一致性测试。

### 12.4 loss ablations

至少比较：

1. uniform direct K-way CE；
2. DFlash fixed exponential CE；
3. exact D-PACE direct CE；
4. exact D-PACE + small base KL trust region（只作为 ablation）。

不要同时改变 architecture、loss、sampler，避免无法归因。

### 12.5 hard examples

rank 3–16 repair 是关键，但不能再用未经论证的 `sqrt(rank)` 直接扭曲 primary
objective。推荐 block-level stratified sampler：

- natural blocks；
- reachable first-miss hard blocks；
- full-top1 easy/protection blocks。

记录 sampling probability，并用 inverse-propensity block weight 恢复目标数据分布。
若要有意提高 hard-example utility，作为单独的 `hard CE auxiliary` ablation 报告，
不要把它伪装成 D-PACE。

---

## 13. 数据是否够、还需要采什么

### 13.1 当前数据够做什么

当前 1,987 train prompts 足够：

- 验证 tensor pipeline；
- exact-D-PACE unit test；
- 512-block memorization/capacity gate；
- local/causal/global development comparison；
- 快速发现 leakage、decoder mismatch 和明显无信号。

它不够：

- 做最终 generalization claim；
- 证明跨 domain 稳定；
- 与 100K–1.4M data methods 做公平结论；
- 训练一个以随机 vocabulary table 为主体的 10.8M 模型。

去掉随机词表后，约 1.1M 的 v1 会更 data-efficient，但 2K prompts 仍只能算
development evidence。

### 13.2 正式规模

若当前 development gate 出现稳定 global signal：

- 至少 100K unique prompts；
- responses 全部由同一 target checkpoint 重新生成；
- 每 prompt 8–16 个随机 anchors；
- 约 0.8M–1.6M blocks；
- prompt-level train/select/calibration/gate split；
- chat/math/code/general instruction 保持覆盖。

现有 record 每 block 约 80–90KB。只保存 top-16 而不是 top-64，可把 0.8M blocks
控制在约 60–75GB；必要时采用 online candidate extraction 或按 epoch 重采 anchors。

### 13.3 为 `T=1` 需要新增字段

首轮 `T=0` 不需要 full target distribution。第二阶段若做 sampling，应在每个 gold
prefix 额外收集：

- target 对 DFlash candidate set 的 logits；
- target full-vocab logsumexp；
- target top-M ids/logits，最好取 target top-64 与 DFlash top-K 的 union；
- 可选 target last hidden，以便离线用 frozen LM head 重算。

然后可训练 target 在候选 support 上的 conditional distribution：

\[
p_T^K(c)=
\frac{p_T(c)}
{\sum_{c'\in C_i}p_T(c')}.
\]

DSpark 的 exact full-vocabulary L1 需要 full target distribution 或 target hidden+
frozen LM head；当前数据不满足。

---

## 14. `T=0` 与 `T>0` 的 lossless correctness

### 14.1 greedy `T=0`

任意 draft tokens 都可以由 target greedy verification 修正，最终输出仍与 target
greedy decoding 一致。GCLS-v1 只改变 proposal，不改变 target verifier，因此先在
`T=0` 验证 idea 最干净。

### 14.2 sampling `T>0`

head 必须暴露它真实使用的 proposal distribution：

\[
q_i(c)=\operatorname{softmax}(s_i)_c,\quad c\in C_i.
\]

standard speculative rejection 使用这个 \(q_i\) 与 target \(p_i\) 计算 acceptance/
residual，不能使用原始 DFlash \(q_D\)。

global head 读取未来位置的**确定性 DFlash candidate features**并不自动违反
non-anticipation：这些 features 在 sample token 之前已由当前 verified prefix 一次
确定。只要各位置从明确的 \(q_i\) sample，joint proposal 可定义为
\(\prod_iq_i\)。

但若以后加入依赖已 sample token 的 pairwise/GRU，必须向 verifier 提供对应的
conditional \(q_i(\cdot\mid x_{<i})\)。

DSpark confidence scheduler 还有额外约束：是否继续验证不能依赖尚未实现的未来
sample。GCLS-v1 暂不做 adaptive verification，避免把 token selection 与 scheduling
混在一起。

---

## 15. 分阶段实验门禁

### Gate 0：实现正确性，不用 GPU 大训练

必须全部通过：

1. exact D-PACE 与官方代码/naive formula 数值一致；
2. inclusive prefix weight 确实包含当前位置 \(q_i\)；
3. epoch-0 direct argmax 100% 等于 DFlash top-1；
4. `local-direct` 对其他位置 candidate replacement 严格 invariant；
5. `global-direct` 能响应其他位置 candidate replacement；
6. 所有 target/DFlash/frozen embedding tensors 无 gradient；
7. forward inputs 不包含 gold ids/ranks；
8. output shape 只有 `[B,L,K]` direct logits，没有 edge row；
9. padding、duplicate candidate ids、first-position OTHER 等边界测试通过。

### Gate A：512-block capacity

仍可复用相同 blocks，但主要指标必须全部来自 direct logits：

- direct candidate accuracy；
- direct non-top1/hard accuracy；
- direct first-miss repair；
- direct EAL/oracle-gap recovered；
- harmed blocks。

建议 global-direct capacity threshold：

- overall candidate accuracy ≥ 99%；
- hard candidate accuracy ≥ 97%；
- oracle gap recovered ≥ 95%；
- harmed blocks ≤ 1%。

如果 direct global 连 512 blocks 都记不住，优先排查 implementation/capacity，不
进入大数据训练。

### Gate B：现有 2K prompts 的 development experiment

严格流程：

1. train 只用于优化；
2. `validation_select` 选 epoch/hyperparameter；
3. 单独 calibration split（必要时）只校准 abstention；
4. sealed `validation_gate` 最后只读一次；
5. 3 seeds；
6. cluster bootstrap unit 是 prompt，不是 block。

同参数比较：

- DFlash top-1；
- rank/logit-only MLP；
- local-direct；
- causal-direct；
- global-direct；
- global-direct with context-shuffle evaluation；
- DeLS/Markov causal baseline。

development go signal：

- global-direct 相对 local-direct 平均至少 +0.15 accepted-draft EAL；
- paired prompt bootstrap 方向稳定；
- context shuffle 后收益显著消失；
- first-token 和各 domain 没有不可接受回退；
- 三个 seed 均不靠单一 domain 支撑。

这只是扩大数据的依据，不是最终论文结论。

### Gate C：100K-prompt confirmatory training

- 预先锁定 architecture/loss；
- 至少 3 seeds；
- prompt-disjoint sealed test；
- 统一数据、training budget 和 checkpoint selection；
- matched local/causal/global、DeLS/Markov、DFlash；
- 报告 mean、seed spread、prompt-cluster bootstrap CI；
- 同时报 draft accepted tokens 与含 target bonus token 的 \(\tau\)，避免口径混淆。

### Gate D：在线 `T=0` rollout

必须从真实 speculative cycles 收集：

- accepted length histogram；
- first-token acceptance；
- domain/context-length breakdown；
- end-to-end latency；
- head latency；
- target verification latency；
- tokens/s 与 speedup；
- actual online anchor distribution。

离线 gold-prefix EAL 只作为诊断，不能替代 rollout。

### Gate E：`T=1` lossless sampling

只有新增 target distributions 后才做：

- conditional target distillation；
- actual selector proposal \(q\) 接入 verifier；
- distributional correctness test；
- stochastic rollout；
- target-only reference 的 statistical equivalence；
- throughput under batch/concurrency。

---

## 16. 如何证明“收益真的来自全局信息”

单纯 global > local 还不够。需要三层证据。

### 16.1 matched architecture

local/causal/global 只能改变 attention mask，参数、数据、loss、训练步数完全一致。

### 16.2 query-wise context replacement

评估位置 \(i\) 时构造 hybrid block：

- 保留当前 block 的 anchor 与位置 \(i\) 全部 nodes；
- 将其他位置 nodes 替换为 prompt-disjoint 随机 block 对应位置；
- 只读取位置 \(i\) 的输出。

可分别做：

- `all-other shuffle`：替换所有 \(j\ne i\)；
- `future-only shuffle`：只替换 \(j>i\)；
- `past-only shuffle`：只替换 \(j<i\)。

需要每个 block 做最多 L 次 forward，但只用于评估。它保持当前位置 feature 不变，
能直接测量其他位置 context 的因果作用。

### 16.3 signal localization

对修复样本报告：

- base first-miss rank；
- global/local score margin；
- attention mass 到哪些位置/ranks；
- shuffle 后 margin 变化；
- math/code/chat；
- pattern consistency，例如固定短语、括号/缩进、数字格式、代码 identifier。

attention 本身不是因果解释，但与 controlled replacement 联合后可定位信号来源。

---

## 17. idea 问题还是实现问题

### 17.1 已经能确认的实现/协议问题

- objective entity 不对；
- local control 不干净；
- D-PACE 公式实现错误；
- primary metric teacher-forced；
- 89.92% 随机 vocabulary table；
- base-logit calibration 被破坏；
- coverage/correctness 混淆；
- auxiliary 与 inference 不一致；
- 没有 held-out GCLS evidence；
- `T>0` proposal probability 未定义。

这些问题足以解释当前结果为什么不能验证 idea。因此现在不能用“效果不好”判定
全局信息无效。

### 17.2 idea 自身仍可能失败的原因

即使实现完全正确，仍有四种真实失败模式：

1. **信息不可识别**：target 的最终选择不由 DFlash hidden/top-K lattice 唯一决定；
2. **多模态平均**：不同合理 continuation 在 lattice 上都一致可行，global context
   仍无法知道 target greedy tie-breaking；
3. **全局相关但无增量信息**：DFlash hidden 已经通过 bidirectional block attention
   编码了能用的全局信息，额外 candidate mixer 没有新信号；
4. **收益小于延迟**：EAL 上升，但 240-node mixer 使实际 speedup 下降。

### 17.3 明确证伪标准

在以下条件同时满足后，若 global-direct 仍不优于 local/causal，才应认为核心 idea
缺乏价值：

- exact loss 与 identity init 通过；
- 512-block direct capacity 通过；
- ≥100K prompt-diverse target-generated data；
- matched local/causal/global；
- 3 seeds；
- sealed held-out 与在线 rollout；
- context shuffle 不产生预期变化；
- latency 纳入最终指标。

若 global-direct 在 capacity 可记忆、但 100K held-out 与 shuffle control 均无
增益，说明主要瓶颈是 selectability/Bayes ceiling，应转向 Domino/DSpark/DeLS
式真实 prefix causality，而不是继续放大全局 mixer。

---

## 18. pairwise/structured v2 应该怎么做

只有 direct GCLS-v1 已证明全局 unary 有用后，才考虑：

\[
E(r_{1:L})
=\sum_i s_{i,r_i}^{\mathrm{global}}
+\sum_{i>1}T_i(r_{i-1},r_i).
\]

此时不能再只训练 gold predecessor row 并用未监督 rows 做 Viterbi。更一致的方案：

- globally normalized linear-chain CRF；
- 对首次 top-K miss 做 censored prefix likelihood；
- forward-backward 精确计算 partition/marginals；
- 或直接训练 conditional greedy policy，并明确只用于 accepted-prefix regime。

v2 必须分别报告：

- global unary-only；
- global unary + Markov greedy；
- global unary + CRF MAP；
- DeLS/DSpark causal head。

若 pairwise 增益存在，也要明确它是“全局 lattice signal”还是“generated-prefix causal
signal”，不能再次混写。

---

## 19. 下一步执行顺序

1. 新增 `GlobalDirectCandidateSelector`，不改旧 v0，便于回归对照。
2. 实现 exact official D-PACE，并增加 naive formula/official parity tests。
3. 删除 v1 trainable vocabulary table、pairwise transitions 和 auxiliary heads。
4. 实现 local/causal/global 三个 matched masks。
5. 实现 identity-init parity test 与 scope invariance test。
6. 先跑 CPU/small-GPU smoke。
7. 跑 512-block direct capacity gate。
8. capacity 通过后，才提交当前 2K prompts 的 3-seed development array。
9. development 达到 go signal 后，再采集 ≥100K prompts。
10. 最后做 online `T=0` 与 `T=1`。

在第 1–5 步完成前，不应继续提交旧 GCLS-v0 的 full training；那只会消耗 GPU，
并重复回答一个与核心 hypothesis 不一致的问题。

---

## 20. 源码与论文索引

### 本地源码快照

| 项目 | 本地路径 | commit |
|---|---|---|
| DFlash | `third_party/dflash` | `94e4abc5e0c31b67bc1a9d30f1cc34ece28a8756` |
| Domino | `third_party/Domino` | `930e5cd823f4bbbaa82ae150acad03928a3a859f` |
| DeepSpec/DSpark | `third_party/DeepSpec` | `005e03b81cec38b7da6399833d609ee89a2587f2` |
| D-PACE | `third_party/D-PACE` | `f36bad6e6b0f9f5b59e1e6cf405c705b46d2b43f` |
| DeLS-Spec | `third_party/DeLS-Spec` | `ab9be1b4d4d470064cd98dd25f7cd1c124b86ad0` |

### 主要论文/官方仓库

- [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/html/2602.06036)
- [Domino: Decoupling Causal Modeling from Autoregressive Drafting](https://arxiv.org/html/2605.29707)
- [DSpark: Confidence-Scheduled Speculative Decoding](https://arxiv.org/html/2607.05147)
- [DeepSpec official repository](https://github.com/deepseek-ai/DeepSpec)
- [D-PACE: Dynamic Position-Aware Cross-Entropy](https://arxiv.org/html/2605.18810)
- [D-PACE official repository](https://github.com/Lucas-TY/D-PACE)
- [DeLS-Spec: Decoupled Long-Short Contexts](https://arxiv.org/html/2607.07409)
- [DeLS-Spec official repository](https://github.com/dt-3t/DeLS-Spec)

