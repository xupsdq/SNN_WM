# V6.1 延后处理包 L1：层间 successor-state 机制全文联动

**状态：DEFERRED / 未写入 `v6_1.docx`**  
**作者决定：先完成其余逐项审查，再返回整体处理。**

## 统一机制

```text
Layer-l inherited STSP
→ conditions current-input processing at layer l
→ changes feedforward activity and downstream updating
→ forms a Layer-(l+1) successor state
→ the successor becomes the inherited condition for the next transition
```

禁止把中心机制压缩成“同层 inherited state 被当前输入改写成自己的 successor”。局部 STSP 数值更新仍可按协议描述；本处理包不建立 `rewrite` 禁词表。

## 待同步处理清单

- [ ] **L1.0 — Discussion P054：** 以 evidence-bounded inter-layer synthesis 替换正式版/副本段落；保留 tested-sequence 与 fixed-circuit 范围。
- [ ] **L1.1 — Abstract P005：** 将 `current-input processing ... rewrote that state into a successor` 改为 inherited STSP conditions processing，processing forms a downstream successor。
- [ ] **L1.2 — Introduction P009：** 将 `retained synaptic history ... is rewritten for the next` 改为 repeated inter-layer transitions。
- [ ] **L1.3 — Introduction P010：** 消除 `retained states ... being rewritten into successor states` 与段末正确 downstream-formation 表述之间的内部冲突，并避免重复。
- [ ] **L1.4 — Results roadmap P012：** 将问题 2 从 `input rewrites the retained state` 改为 history-conditioned processing forms a downstream successor；问题 4 对齐 structural/functional organization。
- [ ] **L1.5 — Results P033：** 将 `update rewrites the inherited state into a successor` 改为 current-layer processing/downstream activity forms a successor state。
- [ ] **L1.6 — Fig. 4g caption：** 改写为 Layer-l inherited STSP → current-input processing → Layer-(l+1) successor formation → next inherited condition。
- [ ] **L1.7 — Fig. 4g artwork：** 重画当前同层 `S_k → spiking → decay → S_{k+1}` 示意；显式标注 Layer l、Layer l+1 与 feedforward/downstream formation。只改文字而保留旧图不可接受。
- [ ] **L1.8 — Results P038：** 将 `recurrent input-associated rewriting` 校准为 `recurrent input-associated updating`。
- [ ] **L1.9 — Results P039：** 删除 `the synaptic state was rewritten again`，改为 successor reuse + next-input processing + subsequent downstream successor formation。
- [ ] **L1.10 — Discussion P055：** 删除 `each input rewrites the inherited STSP state`；把本段恢复为 maintenance → continuous processing 的领域意义段，避免重复 P054 的机制细节。

## 已扫描且不因本处理包修改

- Introduction P007 的一般性 `representations ... are transformed`：领域背景，不定义 successor 方向。
- Results P032、P037：Layer 1 transfer → Layer 2 successor 与 post-B Layer 2 successor → identical C → post-C Layer 3 successor 的方向正确。
- Methods P102 的 static-frozen `without rewriting STSP`：指局部 u/x 更新被禁止，是控制操作定义。
- Methods P100、P116、P117 与 Supplementary successor-transfer 描述：当前层级和方向正确。

## 返回本处理包时的执行要求

1. 先共同确认 P054 中心表述；
2. 同步确认 L1.1–L1.10，不做孤立替换；
3. 定位并修改 Fig. 4g 可复现源文件，不直接手工覆盖最终位图；
4. 重建主图、写入 versioned DOCX、重新渲染 PDF；
5. 全文搜索 `rewrite/rewritten/transform/successor`，按语义而非禁词复核；
6. 核对 Abstract、Introduction、Results、caption、Discussion 与 Supplementary 的层级方向一致。
