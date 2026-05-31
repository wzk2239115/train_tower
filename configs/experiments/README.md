# Experiment profiles

Coordinated **model size + curriculum + viz** bundles under `configs/experiments/`.

| Profile | Size | Steps | Datasets |
|---------|------|-------|----------|
| `500m_continuous` | 500m | 420k | Full PT/MT/SFT mix (see yaml) |
| `tiny_smoke` | tiny_smoke | 100 | `blip3o_short_pt` only |

```bash
tower experiment list
tower train --experiment 500m_continuous
tower viz experiment --profile 500m_continuous
```

See `_schema.yaml` for profile fields.
