# ARIS Research Refinement Report

## Outcome

- final method：PGCF-16 Parallel Global Candidate Fusion Head
- final verdict：`READY`
- final score：`9.3/10`
- rounds：3
- review independence：same-family provisional

## Convergence

| round | score | verdict | decisive change |
|---|---:|---|---|
| 1 | 7.1 | REFINE | 修正off-spec oracle；要求full16 OPB；暴露4-mode瓶颈 |
| 2 | 8.8 | REFINE | 删除modes，改完整256-node attention；收缩loss与参数 |
| 3 | 9.3 | READY | safe loss、exact biases/cost、数字化gates、formal协议闭合 |

## Material influence

ARIS审查实质改变了方法：Round-0 的4-mode压缩被删除，主方法改为完整candidate-node全局通信；L15 Stage-A被禁止，必须重采full16；部署oracle从含causal Domino action的`10.999879`纠正为纯base Top16 `10.909257`；验证从模糊“显著/接近”改为可执行数字。

## Next handoff

`experiment-plan → experiment-bridge`：先实现Gate0、512 capacity和A40 eager profile；只有全部通过后才重采/训练199.8K full16数据。
