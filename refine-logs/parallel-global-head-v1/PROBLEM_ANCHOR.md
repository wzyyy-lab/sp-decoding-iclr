# Problem Anchor：并行全局单序列 Head

- **Bottom-line problem：** 完全解决 DFlash 接受长度不高的问题；新方法的 held-out accepted length 必须显著超过 released Domino，并保持最终端到端吞吐优势。
- **Must-solve bottleneck：** DFlash 在 16 个并行位置的 Top-16 候选中包含大量正确 token，但现有逐位置或因果选择器不能利用完整 draft block 的全局一致性来同时选对这些 token。
- **Required mechanism：** 一个轻量、非因果的全局 head，一次读取全部 16 个 DFlash 位置，使每个位置都能看到整段 draft，然后并行输出唯一一条 16-token 序列。
- **Non-goals：** 不允许自回归/因果 token feedback、Domino GRU rollout、串行 target decode、target seed、迭代修复、beam、tree、trie、forest、多路径验证或额外 target inference。
- **Constraints：** Top-16 只能作为每个位置内部的候选维；线上只使用 DFlash 可得特征；初始新参数预算约 10.75M，可在 held-out 收益与 latency 证据支持下增加；先公平 eager 比较，最终同栈集成 SGLang。
- **Success condition：** fixed held-out EAL 和 dynamic EAL 均至少达到同次 released Domino 的 1.15x，同时最终 A40 SGLang end-to-end TPS 至少达到 Domino 的 1.15x。

