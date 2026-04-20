from __future__ import annotations

from typing import Any

import numpy as np

from src.experiments.common.diagnostic_mask_utils import connected_component_count, mask_bbox


def _region_phrase(mask: np.ndarray) -> str:
    mask_bool = np.asarray(mask, dtype=bool)
    if not mask_bool.any():
        return "未检测到稳定关键区域"
    rows, cols = np.where(mask_bool)
    height, width = mask_bool.shape
    mean_row = float(rows.mean())
    mean_col = float(cols.mean())
    vertical = "上部" if mean_row < height / 3.0 else "下部" if mean_row > (2.0 * height / 3.0) else "中部"
    horizontal = "左侧" if mean_col < width / 3.0 else "右侧" if mean_col > (2.0 * width / 3.0) else "中央"
    return f"{horizontal}{vertical}"


def _shape_tokens(mask: np.ndarray) -> list[str]:
    mask_bool = np.asarray(mask, dtype=bool)
    if not mask_bool.any():
        return []
    bbox = mask_bbox(mask_bool)
    if bbox is None:
        return []
    row0, row1, col0, col1 = bbox
    height = row1 - row0 + 1
    width = col1 - col0 + 1
    components = connected_component_count(mask_bool)
    tokens: list[str] = []
    if components > 1:
        tokens.append("junction")
    if height >= width * 1.6:
        tokens.append("stem")
    elif width >= height * 1.6:
        tokens.append("bar")
    if components == 1 and height > 2 and width > 2 and mask_bool[row0:row1 + 1, col0:col1 + 1].all():
        tokens.append("loop")
    if len(tokens) == 0:
        tokens.append("hook")
    return tokens[:2]


def summarize_mask_description(
    mask: np.ndarray,
    *,
    true_label: int,
    competitor_label: int,
    mean_importance: np.ndarray | None = None,
) -> dict[str, Any]:
    location = _region_phrase(mask)
    tokens = _shape_tokens(mask)
    token_text = "与".join(tokens) if tokens else "区域"
    importance_note = ""
    if mean_importance is not None and np.asarray(mask, dtype=bool).any():
        masked_values = np.asarray(mean_importance, dtype=np.float64)[np.asarray(mask, dtype=bool)]
        importance_note = f" 平均判别性重要性为 {float(np.nanmean(masked_values)):.4f}。"
    text = (
        f"该 probe 的稳定关键区域集中在{location}的{token_text}；"
        f"删除后，真实类别 {int(true_label)} 相对最强竞争类别 {int(competitor_label)} 的最终电压优势下降。"
        f"{importance_note}"
    )
    return {
        "summary_text": text,
        "location": location,
        "shape_tokens": tokens,
    }
