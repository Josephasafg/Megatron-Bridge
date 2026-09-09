import math

import pytest
import torch
import torch.nn.functional as F

from megatron.bridge.data.datasets.preference import preference_collate_fn
from megatron.bridge.training.dpo import DPOLossConfig, dpo_loss, sequence_logprob_sums, split_pair_rows


def test_sequence_logprob_sums_hand_computed():
    """Masked sums per row, accumulated in float32 from bf16 model output; a fully masked stub row yields (0.0, 0)."""
    nll = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=torch.bfloat16)
    mask = torch.tensor([[0, 1, 1], [1, 0, 0], [0, 0, 0]], dtype=torch.long)
    sums, counts = sequence_logprob_sums(nll, mask)
    assert sums.dtype == torch.float32
    assert counts.dtype == torch.long
    assert sums.tolist() == [-5.0, -4.0, 0.0]
    assert counts.tolist() == [2, 1, 0]


def test_sums_through_real_collate_land_per_side():
    """Drive the actual training collate, fabricate per-token NLL from the labels,
    and check chosen/rejected sums come out on the right side of the split."""
    records = [
        {
            "pair_id": p,
            "chosen_input_ids": [11, 12, 13, 21 + p, 22 + p],  # ctx len 3, completion [21+p, 22+p]
            "chosen_context_len": 3,
            "rejected_input_ids": [11, 12, 13, 31 + p],  # ctx len 3, completion [31+p]
            "rejected_context_len": 3,
        }
        for p in range(2)
    ]
    batch = preference_collate_fn(records, pad_token_id=0, require_ref_logprobs=False)

    # NLL encodes the label id, so masked sums are predictable per side.
    nll = batch["labels"].float() * 0.01
    sums, counts = sequence_logprob_sums(nll, batch["loss_mask"])
    chosen_sums, rejected_sums = split_pair_rows(sums)
    chosen_counts, rejected_counts = split_pair_rows(counts)

    for p in range(2):
        assert chosen_sums[p].item() == pytest.approx(-0.01 * (21 + p + 22 + p))
        assert rejected_sums[p].item() == pytest.approx(-0.01 * (31 + p))
    assert chosen_counts.tolist() == [2, 2]
    assert rejected_counts.tolist() == [1, 1]


# --- dpo_loss -----------------------------------------------------------------


def make_loss_batch(
    per_row_nll: list[list[float]],
    loss_mask: list[list[float]],
    ref_sums: list[float],
    ref_counts: list[int] | None = None,
    multipliers: list[float] | None = None,
    pair_ids: list[int] | None = None,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Row-interleaved (chosen@even, rejected@odd) loss inputs from plain lists."""
    num_rows = len(per_row_nll)
    output_tensor = torch.tensor(per_row_nll, dtype=torch.float32, requires_grad=requires_grad)
    if pair_ids is None:
        pair_ids = [p for p in range(num_rows // 2) for _ in range(2)]
    if multipliers is None:
        multipliers = [1.0] * num_rows
    if ref_counts is None:
        ref_counts = [int(sum(row)) for row in loss_mask]
    batch = {
        "loss_mask": torch.tensor(loss_mask, dtype=torch.long),
        "pair_id": torch.tensor(pair_ids, dtype=torch.long),
        "loss_multiplier": torch.tensor(multipliers, dtype=torch.float32),
        "ref_logprob_sum": torch.tensor(ref_sums, dtype=torch.float32),
        "ref_num_tokens": torch.tensor(ref_counts, dtype=torch.long),
    }
    return output_tensor, batch


def nemo_rl_dpo_reference(output_tensor: torch.Tensor, batch: dict[str, torch.Tensor], config: DPOLossConfig) -> float:
    """NeMo-RL DPOLossFn semantics (RL/nemo_rl/algorithms/loss/loss_functions.py),
    restated at pair granularity: the global per-live-pair mean of the total loss."""
    policy_sums, token_counts = sequence_logprob_sums(output_tensor, batch["loss_mask"])
    policy, ref = policy_sums, batch["ref_logprob_sum"].float()
    if config.preference_average_log_probs:
        policy = policy / token_counts.clamp(min=1)
        ref = ref / batch["ref_num_tokens"].clamp(min=1)
    rewards = policy - ref
    delta = rewards[0::2] - rewards[1::2]
    sample_mask = (batch["loss_multiplier"][0::2] > 0).float()
    live = sample_mask.sum()

    preference = (-F.logsigmoid(config.reference_policy_kl_penalty * delta) * sample_mask).sum() / live
    total = config.preference_loss_weight * preference
    if config.sft_loss_weight > 0:
        sft = -policy_sums[0::2]
        if config.sft_average_log_probs:
            sft = sft / token_counts[0::2].clamp(min=1)
        total = total + config.sft_loss_weight * (sft * sample_mask).sum() / live
    return total.item()


def neg_logsigmoid(x: float) -> float:
    return -math.log(1 / (1 + math.exp(-x)))


@pytest.mark.parametrize("beta", [0.05, 0.5])
def test_dpo_loss_golden_hand_computed(beta):
    """Two pairs, hand-computed margins 4 and -2; beta must scale the margin inside the sigmoid."""
    output_tensor, batch = make_loss_batch(
        per_row_nll=[[5.0, 5.0], [10.0, 10.0], [15.0, 15.0], [12.5, 12.5]],
        loss_mask=[[1, 1]] * 4,
        ref_sums=[-12.0, -18.0, -29.0, -26.0],
    )
    loss, num_live_pairs, metrics = dpo_loss(DPOLossConfig(reference_policy_kl_penalty=beta), batch, output_tensor)

    # margins: (-10 - -20) - (-12 - -18) = 4;  (-30 - -25) - (-29 - -26) = -2
    expected = neg_logsigmoid(beta * 4) + neg_logsigmoid(beta * -2)
    assert loss.item() == pytest.approx(expected, rel=1e-6)
    assert num_live_pairs.item() == 2
    assert metrics["margin"].tolist() == pytest.approx([2.0, 2.0])
    assert metrics["accuracy"].tolist() == pytest.approx([1.0, 2.0])
    assert metrics["rewards chosen"].tolist() == pytest.approx([1.0, 2.0])  # 2 + (-1)
    assert metrics["rewards rejected"].tolist() == pytest.approx([-1.0, 2.0])  # -2 + 1
    assert metrics["dpo loss"].tolist() == pytest.approx([expected, 2.0], rel=1e-6)


def test_dpo_loss_stub_pair_excluded_from_loss_count_and_metrics():
    """A loss_multiplier=0 stub pair contributes nothing anywhere. Every metric is a detached
    [sum, live_pairs] pair, except live fraction whose denominator counts ALL pairs."""
    live_rows_nll = [[5.0, 5.0], [10.0, 10.0]]
    live_mask = [[1, 1], [1, 1]]
    output_live, batch_live = make_loss_batch(live_rows_nll, live_mask, ref_sums=[-12.0, -18.0])
    loss_live, count_live, metrics_live = dpo_loss(DPOLossConfig(), batch_live, output_live)

    output_tensor, batch = make_loss_batch(
        per_row_nll=live_rows_nll + [[99.0, 99.0], [99.0, 99.0]],
        loss_mask=live_mask + [[0, 0], [0, 0]],  # stub rows are fully masked
        ref_sums=[-12.0, -18.0, 0.0, 0.0],
        multipliers=[1.0, 1.0, 0.0, 0.0],
    )
    loss, num_live_pairs, metrics = dpo_loss(DPOLossConfig(), batch, output_tensor)

    assert loss.item() == pytest.approx(loss_live.item(), rel=1e-6)
    assert num_live_pairs.item() == count_live.item() == 1
    assert set(metrics) == {
        "dpo loss",
        "preference loss",
        "sft loss",
        "margin",
        "accuracy",
        "rewards chosen",
        "rewards rejected",
        "live fraction",
    }
    for key, value in metrics.items():
        assert value.shape == (2,), key
        if key == "live fraction":
            continue
        assert value[1].item() == 1, key
        assert value.tolist() == pytest.approx(metrics_live[key].tolist(), rel=1e-6), key
    assert metrics_live["live fraction"].tolist() == [1, 1]
    assert metrics["live fraction"].tolist() == [1, 2]  # 1 live of 2 pairs -> 0.5 live share


def test_dpo_loss_adjacency_tripwire():
    output_tensor, batch = make_loss_batch(
        per_row_nll=[[1.0, 1.0]] * 4,
        loss_mask=[[1, 1]] * 4,
        ref_sums=[0.0] * 4,
        pair_ids=[0, 1, 1, 0],  # rows are not (chosen, rejected) adjacent
    )
    with pytest.raises(ValueError, match="pair_id"):
        dpo_loss(DPOLossConfig(), batch, output_tensor)


@pytest.mark.parametrize(
    "config",
    [
        DPOLossConfig(),
        DPOLossConfig(preference_average_log_probs=True),
        DPOLossConfig(sft_loss_weight=0.3),
        DPOLossConfig(sft_loss_weight=0.3, sft_average_log_probs=True),
        DPOLossConfig(preference_loss_weight=2.0, sft_loss_weight=0.1, preference_average_log_probs=True),
    ],
    ids=["sum", "avg-pref", "sft", "sft-avg", "weights-mixed"],
)
def test_dpo_loss_matches_nemo_rl_reference(config):
    """(loss_sum / live_pairs) must equal NeMo-RL DPOLossFn's global per-pair mean,
    across sum/average modes and SFT mixing — including a stub pair in the batch."""
    torch.manual_seed(7)
    num_pairs, seq = 4, 6
    nll = torch.rand(2 * num_pairs, seq) * 3
    mask = (torch.rand(2 * num_pairs, seq) > 0.3).long()
    mask[:, 0] = 1  # no accidentally-empty live completion
    mask[4:6] = 0  # pair 2 is a stub: fully masked, multiplier 0
    ref_counts = mask.sum(-1).tolist()
    output_tensor, batch = make_loss_batch(
        per_row_nll=nll.tolist(),
        loss_mask=mask.tolist(),
        ref_sums=(torch.randn(2 * num_pairs) * 5 - 10).tolist(),
        ref_counts=ref_counts,
        multipliers=[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
    )
    loss, num_live_pairs, _ = dpo_loss(config, batch, output_tensor)
    ours = loss.item() / num_live_pairs.item()
    assert ours == pytest.approx(nemo_rl_dpo_reference(output_tensor, batch, config), rel=1e-5)


def test_dpo_loss_sft_term_golden():
    """sft_loss_weight adds w_s * chosen NLL sum on top of the preference term."""
    output_tensor, batch = make_loss_batch(
        per_row_nll=[[5.0, 5.0], [10.0, 10.0]],
        loss_mask=[[1, 1]] * 2,
        ref_sums=[-12.0, -18.0],
    )
    base, _, base_metrics = dpo_loss(DPOLossConfig(), batch, output_tensor)
    loss, _, metrics = dpo_loss(DPOLossConfig(sft_loss_weight=0.1), batch, output_tensor)
    assert loss.item() == pytest.approx(base.item() + 0.1 * 10.0, rel=1e-6)  # chosen NLL sum = 10
    assert metrics["sft loss"].tolist() == pytest.approx([10.0, 1.0])  # reported unweighted, NeMo-RL style
    assert base_metrics["sft loss"].tolist() == pytest.approx([0.0, 1.0])


def test_dpo_loss_gradient_flow():
    """Gradients reach live masked positions only; stub rows and metrics stay detached."""
    output_tensor, batch = make_loss_batch(
        per_row_nll=[[5.0, 5.0], [10.0, 10.0], [99.0, 99.0], [99.0, 99.0]],
        loss_mask=[[1, 0], [1, 1], [0, 0], [0, 0]],
        ref_sums=[-12.0, -18.0, 0.0, 0.0],
        multipliers=[1.0, 1.0, 0.0, 0.0],
        requires_grad=True,
    )
    loss, _, metrics = dpo_loss(DPOLossConfig(), batch, output_tensor)
    assert loss.requires_grad
    assert all(not value.requires_grad for value in metrics.values())
    loss.backward()
    grad = output_tensor.grad
    assert grad[0, 0].item() != 0.0
    assert grad[0, 1].item() == 0.0  # masked-out position
    assert torch.all(grad[2:] == 0)  # stub rows
