"""Greedy TDT leaves a frame after ``max_symbols_per_step`` decisions there.

NeMo's greedy TDT decoder counts the decisions it takes at one encoder frame
and moves to the next frame once ``max_symbols`` is reached. A checkpoint
that keeps predicting a symbol at duration zero on one frame -- seen with
nvidia/parakeet-tdt_ctc-0.6b-ja on noisy YouTube audio -- would otherwise use
the whole ``max_symbols_per_step * frames`` budget on that frame and leave the
rest of the clip undecoded. These pin the cap on the single-row and the batch
host paths.
"""

from __future__ import annotations

import torch

from kestrel.models.parakeet_tdt.config import ParakeetEncoderConfig, ParakeetTdtConfig
from kestrel.models.parakeet_tdt.model import ParakeetTdt

FRAMES = 6
STUCK_TOKEN = 5


def _tiny_config() -> ParakeetTdtConfig:
    config = ParakeetTdtConfig(
        encoder=ParakeetEncoderConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_mel_bins=128,
            conv_kernel_size=9,
            subsampling_conv_channels=8,
            subsampling_conv_kernel_size=3,
            subsampling_conv_stride=2,
            subsampling_factor=8,
            max_position_embeddings=1024,
            hidden_act="silu",
        ),
        blank_token_id=3,
        decoder_hidden_size=16,
        durations=(0, 1, 2, 3, 4),
        max_symbols_per_step=10,
        num_decoder_layers=1,
        pad_token_id=3,
        vocab_size=8,
        hidden_act="relu",
    )
    config.validate()
    return config


def _stuck_on_frame_zero(model: ParakeetTdt) -> torch.Tensor:
    """Encoded rows whose first feature is the frame index, and a joint that
    predicts ``STUCK_TOKEN`` at duration zero on frame zero, blank elsewhere."""

    config = model.config
    width = config.vocab_size + len(config.durations)
    blank = config.blank_token_id

    class StuckJoint(torch.nn.Module):
        def forward(
            self, encoded: torch.Tensor, decoder_hidden: torch.Tensor
        ) -> torch.Tensor:
            del decoder_hidden
            frame = encoded[..., 0].reshape(-1)
            logits = torch.full((frame.numel(), 1, 1, width), -10.0)
            on_zero = frame == 0
            logits[on_zero, 0, 0, STUCK_TOKEN] = 10.0
            logits[on_zero, 0, 0, config.vocab_size + 0] = 10.0  # duration 0
            logits[~on_zero, 0, 0, blank] = 10.0
            logits[~on_zero, 0, 0, config.vocab_size + 1] = 10.0  # duration 1
            return logits

    model.joint = StuckJoint()
    hidden = model.config.encoder.hidden_size
    encoded = torch.zeros(1, FRAMES, hidden)
    encoded[0, :, 0] = torch.arange(FRAMES, dtype=torch.float32)
    return encoded


def test_a_single_row_reaches_the_end_of_the_encoding() -> None:
    model = ParakeetTdt(_tiny_config()).eval()
    model.reset_nonpersistent_buffers()
    encoded = _stuck_on_frame_zero(model)
    valid = torch.ones(1, FRAMES, dtype=torch.bool)
    cap = model.config.max_symbols_per_step

    with torch.inference_mode():
        out = model.generate_encoded(encoded, valid)

    sequence = out.sequences[0, : int(out.lengths[0])].tolist()
    durations = out.durations[0, : int(out.lengths[0])].tolist()
    assert sequence.count(STUCK_TOKEN) == cap
    assert sum(durations) == FRAMES
    assert sequence[-1] == model.config.blank_token_id


def test_a_batch_reaches_the_end_of_the_encoding() -> None:
    model = ParakeetTdt(_tiny_config()).eval()
    model.reset_nonpersistent_buffers()
    encoded = _stuck_on_frame_zero(model).expand(2, -1, -1).contiguous()
    valid = torch.ones(2, FRAMES, dtype=torch.bool)
    cap = model.config.max_symbols_per_step

    with torch.inference_mode():
        out = model._generate_batch(encoded, valid, max_tokens=None)

    for row in range(2):
        length = int(out.lengths[row])
        sequence = out.sequences[row, :length].tolist()
        assert sequence.count(STUCK_TOKEN) == cap
        assert sum(out.durations[row, :length].tolist()) == FRAMES
