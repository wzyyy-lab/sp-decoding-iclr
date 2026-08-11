# FINAL PROPOSAL：PGCF-16 Parallel Global Candidate Fusion Head

> ARIS research-refine：`READY 9.3/10`  
> Authoritative constraints：`USER_CONSTRAINT_CONTRACT.md`  
> Exact executable specification：`round-1-refinement.md` + `round-2-refinement.md`

## Problem

Released Domino-DFlash backbone full16 base Top-1 EAL为`6.068513120`，released Domino为`7.239552964`。同job 1.15x门槛为`8.325485909`；纯base Top16 oracle为`10.909256560`，因此方法必须回收base→oracle gap的`46.6245%`。

## Method

PGCF-16一次读取：

- parallel hidden `H[B,16,2560]`；
- pure base Top16 IDs/logits `[B,16,16]`；
- anchor与candidate frozen embeddings。

每个position-candidate构成一个node：

```text
q_i   = W_h RMS(H_i) + W_e RMS(E_anchor)
e_ik  = W_e RMS(E[C_ik])
x0_ik = LN(q_i + e_ik + W_mul(q_i*e_ik) + W_phi(phi_ik) + p_i + r_k)
```

完整256 nodes运行两层`d256/h8/FFN2x`无mask self-attention。任一candidate在第一层即可访问所有16位置、所有256 candidates。连续hidden层之间没有argmax或selected-token input。

```text
scores[B,16,16] = base_top16_logits + zero_init_residual(X2)
proposal[B,16]  = gather(candidate_ids, argmax(scores, -1))
```

一次head invocation、一次张量argmax、唯一16-token链。禁止causal/GRU/token loop、额外target forward、迭代、sequence search、beam/tree/trie/forest/multipath verifier。

## Cost

- trainable parameters：精确`2,438,400`；
- headless DFlash占比：约`0.454%`；
- dense matrix work：约`0.3632G MAC`；
- checkpoint后预投影vocab table：`77,791,232 bytes` BF16；
- complete eager pipeline必须`<=1.20x`公平eager Domino development指导线；最终同栈SGLang TPS必须`>=1.15x` Domino。

## Training

1. R047 full16 512-block mechanics/capacity。
2. 在15,886/1,175 disjoint split验证global信号与A40 latency。
3. 通过后，把旧OPB 25K/~199.8K重采为严格full16；旧L15 cache不用于claim-bearing训练。
4. 先短期supported Domino-action warm-start，再平滑切换到target-only soft accepted-prefix objective与0.05 target-candidate KL。
5. 最终用OPB:Phase3=`3:1`做三seed adaptation；不解冻DFlash，除非train与heldout共同显示欠拟合且重新profile。

Safe prefix loss、teacher schedule、所有bias、remote intervention、seed/checkpoint和formal协议由`round-2-refinement.md`精确定义。

## Gates

- Gate0：full16/one-chain/no-loop合规、base identity、remote visibility、safe loss、exact params、base16 oracle。
- Gate1：512 same-set gold-in-K accuracy≥99%、hard≥97%、oracle gap recovery≥95%、harm≤1%、teacher action reconstruction≥99%。
- Gate2：global-local ΔEAL≥0.15且paired prompt-bootstrap CI lower>0；remote erasure≥50%；complete A40 eager≤1.20x Domino。
- Gate3：full16 teacher EAL≥7.080272109；target Stage-A EAL≥7.55。
- Gate4：development EAL≥8.325485909；三域相对Domino容差≤0.05；freeze recipe与primary checkpoint。
- Gate5：fresh formal fixed EAL≥1.15x Domino、每域不退化；dynamic≥1.15x；最终SGLang TPS≥1.15x。

199.8K训练只在Gate0/1/2都通过后启动。任何失败只允许在同一并行单链机制内定位，不授权causal、serial-target、iterative或tree路线。
