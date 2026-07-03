"""Training recipes: the architecture + optimizer-internal values that
upstream nanochat changed between leaderboard runs.

Upstream pins one recipe per commit; this port supports training with more
than one, so the values live here as a small registry. A recipe is named
after the upstream nanochat commit whose training recipe it pins, and
covers only the axes that have no CLI flag of their own:

- ``config``: :class:`~nanochat_jax.gpt.GPTConfig` field overrides
  (architecture + init). An empty dict means the GPTConfig defaults, which
  are the ``dc54a1a`` values.
- AdamW group tables, the value-embeds LR scale, NorMuon ``beta2``, and the
  Polar Express pre-norm constant (consumed by
  :func:`nanochat_jax.optim.setup_optimizer_param_groups`).
- ``wd_schedule``: the weight-decay envelope shape
  (:func:`nanochat_jax.train_core.get_weight_decay`).
- ``lr_envelope``: optional pinned LR/schedule flag values. LR/schedule
  flags always stay explicit on the command line; when a recipe pins an
  envelope, ``scripts/base_train.py`` asserts the flags against it at
  startup so a mixed recipe fails loudly instead of silently training the
  wrong thing.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Recipe:
    """One upstream run's non-CLI axis values. See the module docstring."""

    config: dict = field(default_factory=dict)
    lm_head_betas: tuple[float, float] = (0.8, 0.96)
    lm_head_wd: float = 0.01
    embedding_betas: tuple[float, float] = (0.8, 0.995)
    embedding_wd: float = 0.001
    ve_lr_scale: float = 0.5
    ve_betas: tuple[float, float] = (0.8, 0.995)
    ve_wd: float = 0.01
    resid_wd: float = 0.05
    muon_beta2: float = 0.9
    pe_prenorm: float = 1.01
    wd_schedule: str = "cosine"
    lr_envelope: dict | None = None


RECIPES: dict[str, Recipe] = {
    # The recipe of upstream nanochat at commit dc54a1a — the checkout this
    # port mirrors and is verified against. The published d24 baseline
    # (leaderboard row 1) trained with these values; the Recipe defaults
    # above and the GPTConfig defaults ARE them.
    "dc54a1a": Recipe(),
    # Upstream leaderboard Run 4 (commit 324e69c, the ClimbMix run): pre-autoresearch
    # architecture (no smear/backout, softcap 20, RoPE base 1e4, seq//2
    # short windows, 32-channel value-embed gates at 2*sigmoid with zero
    # init, no QK pre-scale, plain inits, flat lambdas), uniform AdamW
    # betas (0.8, 0.95) with weight decay only on the Muon groups, no
    # value-embeds LR halving, NorMuon beta2 0.95, Polar Express pre-norm
    # 1.02, and a linear weight-decay envelope.
    "324e69c": Recipe(
        config=dict(
            softcap_cap=20.0,
            rope_base=10000.0,
            # short_window is sequence-length dependent (seq // 2);
            # scripts/base_train.py fills it in from --seq-len.
            ve_gate_channels=32,
            ve_gate_mult=2.0,
            qk_prescale=1.0,
            use_smear=False,
            use_backout=False,
            wte_init_std=1.0,
            c_fc_init_scale=1.0,
            lambda_init="flat",
            ve_gate_init="zeros",
        ),
        lm_head_betas=(0.8, 0.95),
        lm_head_wd=0.0,
        embedding_betas=(0.8, 0.95),
        embedding_wd=0.0,
        ve_lr_scale=1.0,
        ve_betas=(0.8, 0.95),
        ve_wd=0.0,
        resid_wd=0.0,
        muon_beta2=0.95,
        pe_prenorm=1.02,
        wd_schedule="linear",
        lr_envelope=dict(
            embedding_lr=0.3,
            unembedding_lr=0.004,
            weight_decay=0.2,
            warmup_steps=0,
            warmdown_ratio=0.5,
            final_lr_frac=0.0,
            matrix_lr=0.02,
            scalar_lr=0.5,
        ),
    ),
}

DEFAULT_RECIPE = "dc54a1a"
