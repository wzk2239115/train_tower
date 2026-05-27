# Third-party dependencies

Clone or symlink these repos before training:

```bash
mkdir -p third_party
git clone https://github.com/EvolvingLMMs-Lab/NEO.git third_party/NEO
# Use NEO/VLMTrainKit on PYTHONPATH, or symlink:
# ln -sfn /path/to/NEO/VLMTrainKit third_party/NEO

git clone https://github.com/OpenSenseNova/SenseNova-U1.git third_party/SenseNova-U1
```

Or with submodules from repo root:

```bash
git submodule update --init --recursive
```
