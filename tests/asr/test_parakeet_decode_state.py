"""Host policy bookkeeping for Parakeet TDT decode."""

from types import SimpleNamespace
from typing import cast

from kestrel.models.parakeet_tdt.config import ParakeetTdtConfig
from kestrel.models.parakeet_tdt.model import _TdtBatchDecodeState


def _config(*, max_symbols_per_step: int = 2) -> ParakeetTdtConfig:
    return cast(
        ParakeetTdtConfig,
        SimpleNamespace(
            blank_token_id=0,
            durations=(0, 1, 2),
            max_symbols_per_step=max_symbols_per_step,
        ),
    )


def test_commit_reports_only_new_host_policy_stops() -> None:
    state = _TdtBatchDecodeState.create(
        _config(), valid_lengths=[3, 1, 4], max_tokens=1
    )

    stopped = state.commit(
        decisions=[[1, 0], [0, 1], [0, 0]],
        active=state.active(),
    )

    assert stopped == [0]
    assert state.active() == [False, False, True]
    assert state.commit([[2, 0], [2, 0], [2, 0]], state.active()) == [2]
    assert state.commit([[2, 0], [2, 0], [2, 0]], state.active()) == []


def test_a_symbol_repeated_at_duration_zero_leaves_its_frame_at_the_cap() -> None:
    """Staying on a frame ``max_symbols_per_step`` times moves the row on.

    Before the per-frame cap, this row spent its whole step budget on frame
    zero and stopped there with frame one never decoded.
    """
    state = _TdtBatchDecodeState.create(
        _config(max_symbols_per_step=1), valid_lengths=[2], max_tokens=None
    )

    assert state.commit([[1, 0]], state.active()) == []
    assert state.frames == [1]
    assert state.commit([[1, 0]], state.active()) == []
    assert state.frames == [2]
    assert state.active() == [False]
    assert state.durations == [[0, 1, 1]]


def test_the_per_frame_count_resets_when_the_frame_advances() -> None:
    state = _TdtBatchDecodeState.create(
        _config(max_symbols_per_step=4), valid_lengths=[6], max_tokens=None
    )

    for _ in range(2):
        state.commit([[1, 0]], state.active())
    assert state.frames == [0]
    state.commit([[1, 1]], state.active())  # leaves frame zero on its own
    assert state.frames == [1]
    for _ in range(3):
        state.commit([[1, 0]], state.active())
    assert state.frames == [1]
    state.commit([[1, 0]], state.active())  # the fourth decision reaches the cap
    assert state.frames == [2]


def test_a_decision_that_reaches_the_cap_moves_one_frame_further() -> None:
    """NeMo counts the leaving decision too and adds a frame at the cap."""
    state = _TdtBatchDecodeState.create(
        _config(max_symbols_per_step=3), valid_lengths=[6], max_tokens=None
    )

    for _ in range(2):
        state.commit([[1, 0]], state.active())
    state.commit([[1, 1]], state.active())  # third decision, duration one

    assert state.frames == [2]
    assert state.durations == [[0, 0, 0, 2]]


def _nemo_greedy_tdt(
    decide, frames: int, max_symbols: int, blank: int, durations: tuple[int, ...]
) -> list[tuple[int, int]]:
    """NeMo's ``GreedyTDTInfer._greedy_decode`` frame walk: (token, frame) emitted."""
    emitted: list[tuple[int, int]] = []
    time_idx = 0
    while time_idx < frames:
        symbols_added, need_loop = 0, True
        while need_loop and symbols_added < max_symbols:
            token, duration_index = decide(time_idx, len(emitted))
            skip = durations[duration_index]
            if token != blank:
                emitted.append((token, time_idx))
            symbols_added += 1
            time_idx += skip
            need_loop = skip == 0
        if symbols_added == max_symbols:
            time_idx += 1
    return emitted


def _kestrel_host(decide, frames: int, config) -> list[tuple[int, int]]:
    state = _TdtBatchDecodeState.create(config, valid_lengths=[frames], max_tokens=None)
    emitted: list[tuple[int, int]] = []
    while any(state.active()):
        frame = state.frames[0]
        token, duration_index = decide(frame, len(emitted))
        state.commit([[token, duration_index]], state.active())
        if token != config.blank_token_id:
            emitted.append((token, frame))
    return emitted


def test_the_host_walk_matches_nemo_on_random_decisions() -> None:
    """Decisions depend only on the frame and the tokens emitted so far, as the
    model's do (a blank leaves the decoder state alone), so both walks ask the
    same questions and must emit the same tokens at the same frames."""
    import random

    durations = (0, 1, 2, 3, 4)
    for seed in range(300):
        rng = random.Random(seed)
        cap = rng.choice((1, 2, 3, 10))
        frames = rng.randint(1, 12)
        table: dict[tuple[int, int], tuple[int, int]] = {}

        def decide(frame: int, emitted: int) -> tuple[int, int]:
            key = (frame, emitted)
            if key not in table:
                token = 0 if rng.random() < 0.4 else rng.randint(1, 3)
                table[key] = (token, 0 if rng.random() < 0.6 else rng.randint(1, 4))
            return table[key]

        config = cast(
            ParakeetTdtConfig,
            SimpleNamespace(blank_token_id=0, durations=durations, max_symbols_per_step=cap),
        )
        assert _kestrel_host(decide, frames, config) == _nemo_greedy_tdt(
            decide, frames, cap, 0, durations
        ), seed


def test_the_generated_decoders_mirror_keeps_its_kernels_walk() -> None:
    """The generated kernel walks frames on the device with no per-frame cap,
    so the host state that mirrors it must not add one."""
    state = _TdtBatchDecodeState.create(
        _config(max_symbols_per_step=1),
        valid_lengths=[2],
        max_tokens=None,
        per_frame_cap=False,
    )

    assert state.commit([[1, 0]], state.active()) == []
    assert state.commit([[1, 0]], state.active()) == [0]
    assert state.frames == [0]
    assert state.active() == [False]
