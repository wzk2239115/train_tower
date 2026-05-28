# Vendored third-party source (for offline / source builds)

Training depends on two upstream codebases, vendored under `third_party/`:

| Directory | Upstream | Copied path | License |
|-----------|----------|-------------|---------|
| `third_party/NEO/` | [EvolvingLMMs-Lab/NEO](https://github.com/EvolvingLMMs-Lab/NEO) | `VLMTrainKit/` (`neo` package) | see upstream |
| `third_party/SenseNova-U1/` | [OpenSenseNova/SenseNova-U1](https://github.com/OpenSenseNova/SenseNova-U1) | `src/` (`sensenova_u1` package) | Apache-2.0 |

Pinned revisions: see [`VENDOR_REVISIONS`](VENDOR_REVISIONS).

## Refresh vendored source

From git (needs network):

```bash
./scripts/vendor_third_party.sh
```

From local clones:

```bash
./scripts/vendor_third_party.sh --from-local /path/to/NEO /path/to/SenseNova-U1
```

## Runtime path setup

`tower.paths.ensure_train_paths()` prepends:

- `third_party/NEO`
- `third_party/SenseNova-U1/src`

No manual `PYTHONPATH` or symlinks required when vendored source is present.

**Do not** vendor model weights (`SenseNova-U1/models/`). Only Python source is included (~1.5 MB).
