# GFPR Refinement Report

初始OPAL把on-policy数据、candidate KL和12–30M lookahead adapter捆在一起。三轮ARIS review后，方案收敛为GFPR：先复用Domino现有50.8M causal head，只改变训练状态分布、greedy frontier目标和position-0接口。

关键修订：

1. 295.6K大训练集仍是固定offset，不是实际policy anchors；
2. GLCS-v1冻结position 0，validation-select有139/1175 blocks在head可作用前即失败；
3. 删除首错后suffix与candidate-renormalized KL；
4. 明确draft-prefix target、GRU reset、r+1和full-accept bonus；
5. A–C统一为full-vocabulary adapted Domino policy；
6. keep loss按block归一化，并绑定paired bootstrap与harm gates；
7. 新架构、LoRA和SGLang全部改为效果门后的条件步骤。

ARIS已授权进入Gate A和Gate B；这不等于结果已达到8.325。
