"""Print Step3p7's RoPE source so we can patch the batched shape bug precisely."""
import inspect, sys, transformers as tf
import torch.nn as nn

path = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/jovyan/h800fast/wangzekai/Step-3.7-Flash"
print("loading (weights not needed but trust_remote_code needs a load)...")
m = tf.AutoModelForCausalLM.from_pretrained(
    path, dtype=torch.bfloat16, trust_remote_code=True, device_map="cpu",
    low_cpu_mem_usage=True)

import modeling_step3p7 as M  # the remote-code module
names = [n for n in dir(M) if "rotary" in n.lower() or "rope" in n.lower()]
print("\n=== rotary-related names in modeling_step3p7 ===")
print(names)

for fn_name in ["apply_rotary_pos_emb"]:
    fn = getattr(M, fn_name, None)
    if fn is not None:
        print(f"\n=== source: {fn_name} ===")
        print(inspect.getsource(fn))

# the rotary embedding class (produces cos/sin)
for cand in names:
    obj = getattr(M, cand, None)
    if inspect.isclass(obj):
        print(f"\n=== class: {cand} ===")
        try:
            print(inspect.getsource(obj))
        except Exception as e:
            print("  (no source:", e, ")")
print("\nDONE")
