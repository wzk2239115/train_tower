# Dataset download commands (hf CLI)

Use proxy as needed:

```bash
env -u ALL_PROXY -u all_proxy HTTP_PROXY=http://127.0.0.1:7897 HTTPS_PROXY=http://127.0.0.1:7897 \
  hf download kakaobrain/coyo-700m data/part-00000-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet \
  --local-dir data/raw/coyo-700m --repo-type dataset

env -u ALL_PROXY -u all_proxy HTTP_PROXY=http://127.0.0.1:7897 HTTPS_PROXY=http://127.0.0.1:7897 \
  hf download BLIP3o/BLIP3o-Pretrain-Long-Caption sa_000000.tar \
  --local-dir data/raw/BLIP3o-Pretrain-Long-Caption --repo-type dataset

env -u ALL_PROXY -u all_proxy HTTP_PROXY=http://127.0.0.1:7897 HTTPS_PROXY=http://127.0.0.1:7897 \
  hf download BLIP3o/BLIP3o-Pretrain-Short-Caption 00000.tar \
  --local-dir data/raw/BLIP3o-Pretrain-Short-Caption --repo-type dataset

env -u ALL_PROXY -u all_proxy HTTP_PROXY=http://127.0.0.1:7897 HTTPS_PROXY=http://127.0.0.1:7897 \
  hf download wanng/wukong100m data/train-00000-of-00032.parquet \
  --local-dir data/raw/wukong100m --repo-type dataset

env -u ALL_PROXY -u all_proxy HTTP_PROXY=http://127.0.0.1:7897 HTTPS_PROXY=http://127.0.0.1:7897 \
  hf download Lin-Chen/ShareGPT4V sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json \
  --local-dir data/raw/ShareGPT4V --repo-type dataset

env -u ALL_PROXY -u all_proxy HTTP_PROXY=http://127.0.0.1:7897 HTTPS_PROXY=http://127.0.0.1:7897 \
  hf download liuhaotian/LLaVA-Instruct-150K llava_v1_5_mix665k.json \
  --local-dir data/raw/LLaVA-Instruct-150K --repo-type dataset

env -u ALL_PROXY -u all_proxy HTTP_PROXY=http://127.0.0.1:7897 HTTPS_PROXY=http://127.0.0.1:7897 \
  hf download lmms-lab/RefCOCO data/test-00000-of-00002.parquet \
  --local-dir data/raw/RefCOCO --repo-type dataset

env -u ALL_PROXY -u all_proxy HTTP_PROXY=http://127.0.0.1:7897 HTTPS_PROXY=http://127.0.0.1:7897 \
  hf download lmms-lab/TextCaps data/test-00000-of-00002.parquet \
  --local-dir data/raw/TextCaps --repo-type dataset

env -u ALL_PROXY -u all_proxy HTTP_PROXY=http://127.0.0.1:7897 HTTPS_PROXY=http://127.0.0.1:7897 \
  hf download lmms-lab/DocVQA DocVQA/train-00000-of-00012.parquet \
  --local-dir data/raw/DocVQA --repo-type dataset

env -u ALL_PROXY -u all_proxy HTTP_PROXY=http://127.0.0.1:7897 HTTPS_PROXY=http://127.0.0.1:7897 \
  hf download HuggingFaceM4/ChartQA data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet \
  --local-dir data/raw/ChartQA --repo-type dataset
```
