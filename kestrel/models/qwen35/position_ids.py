"""Host-side packed multimodal position metadata shared by Qwen runtimes."""

import numpy as np


def fill_multimodal_position_ids(
    out: np.ndarray,
    *,
    start: int,
    end: int,
    mm_token_type_ids: np.ndarray,
    image_grid_thw: np.ndarray,
    spatial_merge_size: int,
) -> tuple[int, tuple[tuple[int, int], ...]]:
    cursor = start
    current_pos = 0
    image_idx = 0
    image_token_spans: list[tuple[int, int]] = []
    while cursor < end:
        modality_type = int(mm_token_type_ids[cursor - start])
        group_end = cursor + 1
        while (
            group_end < end
            and int(mm_token_type_ids[group_end - start]) == modality_type
        ):
            group_end += 1

        if modality_type == 0:
            text_len = group_end - cursor
            positions = np.arange(
                current_pos, current_pos + text_len, dtype=np.int64
            )
            out[:, 0, cursor:group_end] = positions
            current_pos += text_len
        elif modality_type == 1:
            if image_idx >= image_grid_thw.shape[0]:
                raise RuntimeError("Qwen image grid metadata ended early")
            grid_t, grid_h_raw, grid_w_raw = (
                int(v) for v in image_grid_thw[image_idx]
            )
            image_idx += 1
            grid_h = grid_h_raw // spatial_merge_size
            grid_w = grid_w_raw // spatial_merge_size
            image_len = grid_t * grid_h * grid_w
            if image_len != group_end - cursor:
                raise RuntimeError(
                    "Qwen image token count does not match image grid: "
                    f"tokens={group_end - cursor}, grid={image_len}"
                )
            image_token_spans.append((cursor, group_end))
            offset = current_pos
            temporal = np.repeat(np.arange(grid_t, dtype=np.int64), grid_h * grid_w)
            height = np.tile(
                np.repeat(np.arange(grid_h, dtype=np.int64), grid_w),
                grid_t,
            )
            width = np.tile(np.arange(grid_w, dtype=np.int64), grid_h * grid_t)
            out[0, 0, cursor:group_end] = temporal + offset
            out[1, 0, cursor:group_end] = height + offset
            out[2, 0, cursor:group_end] = width + offset
            current_pos += max(grid_h_raw, grid_w_raw) // spatial_merge_size
        else:
            raise NotImplementedError("Qwen packed prefill does not support video")

        cursor = group_end

    if image_idx != image_grid_thw.shape[0]:
        raise RuntimeError("Qwen image grid metadata has unused rows")
    rope_delta = int(out[:, 0, start:end].max()) + 1 - (end - start)
    return rope_delta, tuple(image_token_spans)

