"""Flow Tower: shared LLM backbone with JEPA + rectified flow exits."""

from __future__ import annotations

import random
from typing import Any

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from tower.train.config import TrainConfig
from tower.train.losses import compute_resolution_noise_scale, sample_flow_batch
from tower.train.vision_batch import reconcile_vision_inputs
from tower.unify.tower_config import TowerConfig, TowerExitSpec, load_tower_config
from tower.unify.tower_exits import ElfFlowTowerExit, JepaTowerExit
from tower.unify.tower_masking import sample_image_token_mask
from tower.unify.train_model import SenseNovaTrainModel, _build_indexes_with_hw


class FlowJepaTowerTrainModel(SenseNovaTrainModel):
    """Full-tower trainer: L0 JEPA + upper stacked ELF rectified-flow exits.

    Exit layout and per-stage loss weights: ``note/tower.yml``.
    """

    def __init__(
        self,
        model,
        cfg: TrainConfig,
        tower_cfg: TowerConfig | None = None,
    ):
        super().__init__(model, cfg)
        self.tower_cfg = tower_cfg or load_tower_config()
        self.tower_exits = nn.ModuleDict()
        hidden = self.model.config.llm_config.hidden_size
        audio_dim = int(getattr(self.cfg, "audio_patch_dim", 80))
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self._build_tower_exits()
        self._tower_global_step: int = 0

    def _build_tower_exits(self) -> None:
        hidden = self.model.config.llm_config.hidden_size
        patch = self.model.patch_size
        merge = int(1 / self.model.downsample_ratio)
        patch_dim = 3 * (patch * merge) ** 2

        for spec in self.tower_cfg.exits:
            if spec.exit_type == "elf_fm":
                out_dim = patch_dim if spec.latent == "pixel_patch" else hidden
                self.tower_exits[spec.name] = ElfFlowTowerExit(
                    hidden,
                    out_dim,
                    elf_depth=spec.elf_depth,
                    t_embedder=self.model.fm_modules["timestep_embedder"],
                )
                continue
            if spec.exit_type == "jepa":
                self.tower_exits[spec.name] = JepaTowerExit(
                    hidden,
                    ema_momentum=spec.ema_momentum,
                )

    def _exit_map(self) -> dict[int, TowerExitSpec]:
        return {spec.after_layer: spec for spec in self.tower_cfg.exits}

    def set_curriculum_step(self, step: int) -> None:
        self._tower_global_step = max(int(step), 0)

    def _current_stage(self) -> str:
        return self.cfg.curriculum_stage_for_step(self._tower_global_step)

    def _active_exit_specs(self) -> list[TowerExitSpec]:
        stage = self._current_stage()
        return [e for e in self.tower_cfg.exits if self.tower_cfg.loss_weight(e.name, stage) > 0]

    def _max_hook_layer(self, active: list[TowerExitSpec]) -> int | None:
        if not active:
            return None
        return max(spec.after_layer for spec in active)

    def _fm_sample(
        self,
        clean: torch.Tensor,
        *,
        noise_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model_cfg = self.model.config
        z, t, _ = sample_flow_batch(
            clean.unsqueeze(0) if clean.ndim == 2 else clean,
            t_eps=float(getattr(model_cfg, "t_eps", 0.05)),
            p_mean=float(getattr(model_cfg, "P_mean", -0.8)),
            p_std=float(getattr(model_cfg, "P_std", 0.8)),
            time_schedule=self._fm_time_schedule(),
            noise_scale=noise_scale,
        )
        if clean.ndim == 2:
            z, t = z.squeeze(0), t.squeeze(0)
        # Keep timestep as 1-D tensor to avoid scalar indexing errors downstream.
        t = t.reshape(-1)
        return z, t

    def _run_backbone_layers(
        self,
        hidden_states: torch.Tensor,
        *,
        indexes,
        attn,
        indicators: torch.Tensor,
        stop_layer: int | None = None,
        hook_layers: set[int] | None = None,
    ) -> dict[int, torch.Tensor]:
        llm = self.model.language_model.model
        exist_non = (~indicators).any()
        exist_gen = indicators.any()
        hooks: dict[int, torch.Tensor] = {}
        if hook_layers is None:
            hook_layers = set(self._exit_map().keys())
        full_attention = attn["full_attention"] if isinstance(attn, dict) and "full_attention" in attn else attn

        h = hidden_states
        for layer_idx, decoder_layer in enumerate(llm.layers):
            if stop_layer is not None and layer_idx > stop_layer:
                break
            h = decoder_layer(
                h,
                image_gen_indicators=indicators.unsqueeze(0),
                exist_non_image_gen_tokens=exist_non,
                exist_image_gen_tokens=exist_gen,
                indexes=indexes,
                attention_mask=full_attention,
            )
            if layer_idx in hook_layers:
                hooks[layer_idx] = h

        return hooks

    def _text_supervision_mask(self, input_ids: torch.Tensor, labels: torch.Tensor | None, selected: torch.Tensor) -> torch.Tensor:
        if labels is not None:
            mask = labels[0] != -100
        else:
            mask = torch.ones(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
        return mask & (~selected)

    def _prepare_text_batch(self, batch: dict[str, Any]) -> dict[str, Any] | None:
        input_ids = batch["input_ids"]
        labels = batch.get("labels")
        hidden = self.model.language_model.get_input_embeddings()(input_ids)
        selected = input_ids[0] == self.model.img_context_token_id
        text_mask = self._text_supervision_mask(input_ids, labels, selected)
        if not text_mask.any():
            return None

        clean_text = hidden[0, text_mask].detach()
        z_text, t_text = self._fm_sample(clean_text)

        seq_len = input_ids.shape[1]
        indexes = _build_indexes_with_hw(
            self.model, input_ids, batch["indexes"], batch.get("image_grid_hw")
        )
        from sensenova_u1.models.neo_unify.modeling_qwen3 import create_block_causal_mask

        return {
            "input_ids": input_ids,
            "hidden": hidden,
            "indexes": indexes,
            "attn": {"full_attention": create_block_causal_mask(indexes[0])},
            "indicators": torch.zeros(seq_len, dtype=torch.bool, device=self.device),
            "selected": selected,
            "text_mask": text_mask,
            "clean_text": clean_text,
            "z_text": z_text,
            "t_text": t_text,
            "model_cfg": self.model.config,
        }

    def _prepare_tower_batch(
        self,
        batch: dict[str, Any],
        *,
        gen_mode: bool,
    ) -> dict[str, Any] | None:
        input_ids = batch["input_ids"]
        pixel_values = batch.get("pixel_values")
        has_image = pixel_values is not None and len(pixel_values) > 0 and pixel_values[0] is not None

        clean = None
        grid_hw_raw = batch.get("image_grid_hw", [None])[0]
        if has_image:
            clean = pixel_values[0].to(device=self.device, dtype=self.dtype)
            if grid_hw_raw is not None:
                if not isinstance(grid_hw_raw, torch.Tensor):
                    grid_hw_for_clean = torch.tensor(grid_hw_raw, device=self.device)
                else:
                    grid_hw_for_clean = grid_hw_raw.to(device=self.device)
                clean, grid_hw_for_clean = reconcile_vision_inputs(
                    clean,
                    grid_hw_for_clean,
                    spatial_merge=int(1 / self.model.downsample_ratio),
                )
                batch = dict(batch)
                batch["pixel_values"] = [clean]
                batch["image_grid_hw"] = [grid_hw_for_clean]
                pixel_values = batch["pixel_values"]
                grid_hw_raw = grid_hw_for_clean
            grid_h, grid_w = self._parse_grid_hw(grid_hw_raw, num_patches=clean.shape[0])
            noise_scale = self._fm_noise_scale(grid_h, grid_w)
        else:
            grid_h = grid_w = 0
            noise_scale = 1.0

        if grid_hw_raw is None:
            if has_image:
                merge = int(1 / self.model.downsample_ratio)
                grid_size = max(grid_h // merge, 1)
                grid_hw = torch.tensor([[grid_size * merge, grid_size * merge]], device=self.device)
            else:
                grid_hw = None
        elif not isinstance(grid_hw_raw, torch.Tensor):
            grid_hw = torch.tensor(grid_hw_raw, device=self.device)
        else:
            grid_hw = grid_hw_raw.to(device=self.device)

        hidden = self.model.language_model.get_input_embeddings()(input_ids)
        hidden = self._inject_vision(
            input_ids,
            hidden,
            batch.get("pixel_values"),
            batch.get("image_grid_hw"),
            gen=gen_mode,
        )
        audio_mask = self._audio_token_mask(input_ids, batch)
        clean_audio = self._extract_audio_clean(batch)
        if clean_audio is not None and audio_mask.any():
            hidden = self._inject_audio(hidden, clean_audio, audio_mask)

        selected = input_ids[0] == self.model.img_context_token_id
        if gen_mode and self._should_cfg_label_drop(batch):
            hidden = self._apply_cfg_label_drop(hidden, selected)

        seq_len = input_ids.shape[1]
        indicators = (
            torch.ones(seq_len, dtype=torch.bool, device=self.device)
            if gen_mode
            else torch.zeros(seq_len, dtype=torch.bool, device=self.device)
        )

        indexes = _build_indexes_with_hw(
            self.model, input_ids, batch["indexes"], batch.get("image_grid_hw")
        )
        from sensenova_u1.models.neo_unify.modeling_qwen3 import create_block_causal_mask

        attn = {"full_attention": create_block_causal_mask(indexes[0])}

        clean_embed_und = clean_embed_gen = None
        z_world = t_world = None
        z_embed = t_embed = None
        z_pixel = t_pixel = None
        if clean is not None and grid_hw is not None:
            clean_embed_und = (
                self.model.extract_feature(clean, gen_model=False, grid_hw=grid_hw)
                .reshape(-1, hidden.shape[-1])
            )
            clean_embed_gen = (
                self.model.extract_feature(clean, gen_model=True, grid_hw=grid_hw)
                .reshape(-1, hidden.shape[-1])
            )

            z_world, t_world = self._fm_sample(clean_embed_und)
            z_embed, t_embed = self._fm_sample(clean_embed_gen)
            z_pixel, t_pixel = self._fm_sample(clean, noise_scale=noise_scale)

        labels = batch.get("labels")
        text_mask = self._text_supervision_mask(input_ids, labels, selected)
        clean_text = hidden[0, text_mask].detach() if text_mask.any() else None
        z_text = t_text = None
        if clean_text is not None and clean_text.numel() > 0:
            z_text, t_text = self._fm_sample(clean_text)
        z_audio = t_audio = None
        if clean_audio is not None and clean_audio.numel() > 0 and audio_mask.any():
            z_audio, t_audio = self._fm_sample(clean_audio)

        return {
            "hidden": hidden,
            "indexes": indexes,
            "attn": attn,
            "indicators": indicators,
            "selected": selected,
            "audio_mask": audio_mask,
            "text_mask": text_mask,
            "clean_embed_und": clean_embed_und,
            "clean_embed_gen": clean_embed_gen,
            "clean_pixel": clean,
            "clean_audio": clean_audio,
            "clean_text": clean_text,
            "z_world": z_world,
            "t_world": t_world,
            "z_embed": z_embed,
            "t_embed": t_embed,
            "z_pixel": z_pixel,
            "t_pixel": t_pixel,
            "z_audio": z_audio,
            "t_audio": t_audio,
            "z_text": z_text,
            "t_text": t_text,
            "noise_scale": noise_scale,
            "model_cfg": self.model.config,
        }

    def _latent_bundle(self, spec: TowerExitSpec, ctx: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        latent = spec.latent
        if latent == "vision_embed_und":
            if ctx.get("clean_embed_und") is None or ctx.get("z_world") is None:
                return None
            return ctx["z_world"], ctx["t_world"], ctx["clean_embed_und"]
        if latent == "vision_embed":
            if ctx.get("clean_embed_gen") is None or ctx.get("z_embed") is None:
                return None
            return ctx["z_embed"], ctx["t_embed"], ctx["clean_embed_gen"]
        if latent == "pixel_patch":
            if ctx.get("clean_pixel") is None or ctx.get("z_pixel") is None:
                return None
            return ctx["z_pixel"], ctx["t_pixel"], ctx["clean_pixel"]
        if latent == "audio_embed":
            if ctx.get("clean_audio") is None or ctx.get("z_audio") is None:
                return None
            return ctx["z_audio"], ctx["t_audio"], ctx["clean_audio"]
        if latent == "token_hidden":
            if ctx.get("clean_text") is None or ctx.get("z_text") is None:
                return None
            return ctx["z_text"], ctx["t_text"], ctx["clean_text"]
        return None

    def _select_hidden_for_exit(
        self,
        spec: TowerExitSpec,
        hook_hidden: torch.Tensor,
        ctx: dict[str, Any],
    ) -> torch.Tensor:
        if spec.latent == "audio_embed":
            return hook_hidden[0, ctx["audio_mask"]]
        if spec.latent == "token_hidden":
            return hook_hidden[0, ctx["text_mask"]]
        n = min(int(ctx["selected"].sum().item()), hook_hidden.shape[1])
        return hook_hidden[0, ctx["selected"]][:n]

    def _compute_exit_loss(
        self,
        spec: TowerExitSpec,
        hook_hidden: torch.Tensor,
        ctx: dict[str, Any],
    ) -> torch.Tensor:
        if spec.exit_type == "jepa":
            if spec.latent == "vision_embed_und":
                clean = ctx["clean_embed_und"]
            elif spec.latent == "vision_embed":
                clean = ctx["clean_embed_gen"]
            else:
                return hook_hidden.sum() * 0.0

            h = self._select_hidden_for_exit(spec, hook_hidden, ctx)
            n = min(h.shape[0], clean.shape[0])
            if n <= 0:
                return hook_hidden.sum() * 0.0

            pred_mask = sample_image_token_mask(n, device=self.device)
            module: JepaTowerExit = self.tower_exits[spec.name]  # type: ignore[assignment]
            loss = module(h[:n], clean[:n], pred_mask)
            module.ema_update_target()
            return loss

        bundle = self._latent_bundle(spec, ctx)
        if bundle is None:
            return hook_hidden.sum() * 0.0

        z, t, clean = bundle
        h = self._select_hidden_for_exit(spec, hook_hidden, ctx)
        n = min(h.shape[0], clean.shape[0], z.shape[0])
        if n <= 0:
            return hook_hidden.sum() * 0.0

        h, clean, z = h[:n], clean[:n], z[:n]
        module: ElfFlowTowerExit = self.tower_exits[spec.name]
        model_cfg = ctx["model_cfg"]
        t_eps = float(getattr(model_cfg, "t_eps", 0.05))

        noise_scale_emb = None
        if (
            spec.latent == "pixel_patch"
            and getattr(model_cfg, "add_noise_scale_embedding", False)
            and "noise_scale_embedder" in self.model.fm_modules
        ):
            max_value = float(getattr(model_cfg, "noise_scale_max_value", 8.0))
            ns = ctx["noise_scale"] / max_value
            noise_scale_emb = self.model.fm_modules["noise_scale_embedder"](
                torch.full((n,), ns, device=self.device, dtype=self.dtype)
            )

        t_part = t[: min(n, t.shape[0])]
        if t_part.numel() == 1 and n > 1:
            t_part = t_part.expand(n)

        self_cond = None
        if spec.latent == "pixel_patch":
            self_cond_prob = float(
                getattr(
                    self.cfg,
                    "tower_self_cond_prob",
                    getattr(model_cfg, "self_cond_prob", 0.0),
                )
            )
            if self_cond_prob > 0 and random.random() < self_cond_prob:
                cfg_min = float(getattr(self.cfg, "tower_self_cond_cfg_min", 1.0))
                cfg_max = float(getattr(self.cfg, "tower_self_cond_cfg_max", cfg_min))
                cfg_max = max(cfg_max, cfg_min)
                cfg_scale = cfg_min if cfg_max == cfg_min else random.uniform(cfg_min, cfg_max)
                with torch.no_grad():
                    base_pred = module.predict_x(
                        h,
                        t_part,
                        noise_scale_emb=noise_scale_emb,
                    )
                    cond_pred = module.predict_x(
                        h,
                        t_part,
                        noise_scale_emb=noise_scale_emb,
                        self_cond=base_pred,
                    )
                    guided = base_pred + cfg_scale * (cond_pred - base_pred)
                    self_cond = guided.detach()

        return module(
            h,
            z,
            t_part,
            clean,
            t_eps=t_eps,
            noise_scale_emb=noise_scale_emb,
            self_cond=self_cond,
        )

    def _accumulate_exit_losses(
        self,
        active: list[TowerExitSpec],
        hooks: dict[int, torch.Tensor],
        ctx: dict[str, Any],
        *,
        latents: set[str],
    ) -> torch.Tensor:
        total = torch.tensor(0.0, device=self.device)
        stage = self._current_stage()
        for spec in active:
            if spec.latent not in latents:
                continue
            w = self.tower_cfg.loss_weight(spec.name, stage)
            if w <= 0 or spec.after_layer not in hooks:
                continue
            total = total + w * self._compute_exit_loss(spec, hooks[spec.after_layer], ctx)
        return total

    def _tower_forward(self, batch: dict[str, Any]) -> torch.Tensor:
        active = self._active_exit_specs()
        if not active:
            return torch.tensor(0.0, device=self.device)

        total = torch.tensor(0.0, device=self.device)
        stop_layer = self._max_hook_layer(active)
        hook_layers = {spec.after_layer for spec in active}

        if not self._batch_has_images(batch) and not self._batch_has_audio(batch):
            ctx = self._prepare_text_batch(batch)
            if ctx is not None:
                hooks = self._run_backbone_layers(
                    ctx["hidden"],
                    indexes=ctx["indexes"],
                    attn=ctx["attn"],
                    indicators=ctx["indicators"],
                    stop_layer=stop_layer,
                    hook_layers=hook_layers,
                )
                total = total + self._accumulate_exit_losses(
                    active, hooks, ctx, latents={"token_hidden"}
                )
            return total

        und_latents = {"vision_embed_und", "token_hidden", "audio_embed"}
        if any(s.latent in und_latents for s in active):
            ctx = self._prepare_tower_batch(batch, gen_mode=False)
            if ctx is not None:
                hooks = self._run_backbone_layers(
                    ctx["hidden"],
                    indexes=ctx["indexes"],
                    attn=ctx["attn"],
                    indicators=ctx["indicators"],
                    stop_layer=stop_layer,
                    hook_layers=hook_layers,
                )
                total = total + self._accumulate_exit_losses(active, hooks, ctx, latents=und_latents)

        gen_latents = {"vision_embed", "pixel_patch", "audio_embed"}
        if any(s.latent in gen_latents for s in active):
            ctx = self._prepare_tower_batch(batch, gen_mode=True)
            if ctx is not None:
                hooks = self._run_backbone_layers(
                    ctx["hidden"],
                    indexes=ctx["indexes"],
                    attn=ctx["attn"],
                    indicators=ctx["indicators"],
                    stop_layer=stop_layer,
                    hook_layers=hook_layers,
                )
                total = total + self._accumulate_exit_losses(active, hooks, ctx, latents=gen_latents)

        return total

    def _batch_has_images(self, batch: dict[str, Any]) -> bool:
        pixel_values = batch.get("pixel_values")
        return pixel_values is not None and len(pixel_values) > 0 and pixel_values[0] is not None

    def _batch_has_audio(self, batch: dict[str, Any]) -> bool:
        audio_values = batch.get("audio_values")
        if audio_values is None:
            return False
        if isinstance(audio_values, (list, tuple)):
            return len(audio_values) > 0 and audio_values[0] is not None
        return True

    def _audio_token_mask(self, input_ids: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        provided = batch.get("audio_token_mask")
        if isinstance(provided, torch.Tensor):
            mask = provided
            if mask.ndim == 2:
                mask = mask[0]
            return mask.to(device=input_ids.device, dtype=torch.bool)

        audio_id = getattr(self.model, "audio_context_token_id", None)
        if audio_id is None:
            audio_id = getattr(self.cfg, "audio_context_token_id", -1)
        if int(audio_id) >= 0:
            return input_ids[0] == int(audio_id)
        return torch.zeros(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)

    def _extract_audio_clean(self, batch: dict[str, Any]) -> torch.Tensor | None:
        audio_values = batch.get("audio_values")
        if audio_values is None:
            return None
        if isinstance(audio_values, (list, tuple)):
            if len(audio_values) == 0 or audio_values[0] is None:
                return None
            audio = audio_values[0]
        else:
            audio = audio_values
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio)
        audio = audio.to(device=self.device, dtype=self.dtype)
        if audio.ndim == 3 and audio.shape[0] == 1:
            audio = audio[0]
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        elif audio.ndim >= 3:
            audio = audio.reshape(audio.shape[0], -1)
        in_dim = self.audio_proj[0].in_features
        if audio.shape[-1] != in_dim:
            if audio.shape[-1] > in_dim:
                audio = audio[..., :in_dim]
            else:
                audio = torch.nn.functional.pad(audio, (0, in_dim - audio.shape[-1]))
        clean_audio = self.audio_proj(audio)
        return clean_audio.reshape(-1, clean_audio.shape[-1]).detach()

    def _inject_audio(
        self,
        hidden: torch.Tensor,
        clean_audio: torch.Tensor,
        audio_mask: torch.Tensor,
    ) -> torch.Tensor:
        if clean_audio.numel() == 0 or not audio_mask.any():
            return hidden
        hidden = hidden.clone()
        n = min(int(audio_mask.sum().item()), clean_audio.shape[0])
        if n <= 0:
            return hidden
        hidden[0, audio_mask][:n] = clean_audio[:n]
        return hidden

    def _has_supervised_tokens(self, batch: dict[str, Any]) -> bool:
        labels = batch.get("labels")
        if labels is None:
            return False
        return bool((labels != -100).any().item())

    def forward(self, **batch):
        if not getattr(self.cfg, "use_flow_tower", False):
            return super().forward(**batch)
        tower_loss = self._tower_forward(batch)

        decoder_prob = float(getattr(self.cfg, "tower_decoder_prob", 0.0))
        decoder_prob = min(max(decoder_prob, 0.0), 1.0)
        if decoder_prob <= 0 or not self._has_supervised_tokens(batch):
            return CausalLMOutputWithPast(loss=tower_loss)

        ce_loss = self._ce_forward(batch)
        if decoder_prob >= 1.0:
            return CausalLMOutputWithPast(loss=ce_loss)

        # ELF-style branch mixing: each step samples decoder (CE) vs denoiser (tower FM).
        if random.random() < decoder_prob:
            return CausalLMOutputWithPast(loss=ce_loss)
        return CausalLMOutputWithPast(loss=tower_loss)
