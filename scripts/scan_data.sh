#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:?Usage: $0 DATA_ROOT [OUTPUT]}"
OUTPUT="${2:-data_inventory.json}"

echo "Scanning $DATA_ROOT ..." >&2

# 1. 目录级 du (只扫1层)
echo "  Computing dir sizes ..." >&2
declare -A DIR_SIZE
while IFS=$'\t' read -r size path; do
    name=$(basename "$path")
    DIR_SIZE["$name"]="$size"
done < <(du -sh "$DATA_ROOT"/*/ 2>/dev/null || true)

# 2. 找所有标注文件，统计行数+采样前3条
echo "  Scanning annotation files ..." >&2
tmp=$(mktemp)
trap "rm -f $tmp" EXIT

find "$DATA_ROOT" -maxdepth 4 \( -name "*.jsonl" -o -name "*.json" -o -name "*.csv" \) -type f 2>/dev/null | sort | while read -r f; do
    rel=$(realpath --relative-to="$DATA_ROOT" "$f")
    dir_name=$(echo "$rel" | cut -d/ -f1)
    size=$(stat -c%s "$f" 2>/dev/null || echo 0)
    lines=$(wc -l < "$f" 2>/dev/null || echo 0)
    # 前3行的keys
    keys=$(head -3 "$f" 2>/dev/null | python3 -c "
import sys,json
keys=set()
for line in sys.stdin:
    line=line.strip()
    if not line: continue
    try:
        keys.update(json.loads(line).keys())
    except: pass
print(','.join(sorted(keys)) if keys else '')
" 2>/dev/null || echo "")
    echo -e "$dir_name\t$rel\t$size\t$lines\t$keys"
done > "$tmp"

# 3. 组装 JSON
echo "  Building report ..." >&2
python3 - "$DATA_ROOT" "$tmp" "$OUTPUT" << 'PYEOF'
import sys, json, os
from datetime import datetime
from collections import defaultdict

data_root = sys.argv[1]
tmp_file = sys.argv[2]
output = sys.argv[3]

dirs = defaultdict(lambda: {"annotation_files": [], "total_lines": 0, "total_size": 0})

with open(tmp_file) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        dir_name, rel, size, lines, keys = parts
        d = dirs[dir_name]
        d["annotation_files"].append({
            "path": rel,
            "size_mb": round(int(size) / 1024 / 1024, 1),
            "lines": int(lines),
            "fields": keys.split(",") if keys else [],
        })
        d["total_lines"] += int(lines)
        d["total_size"] += int(size)

total_lines = 0
total_ann = 0
for d in dirs.values():
    total_lines += d["total_lines"]
    total_ann += len(d["annotation_files"])

result = {
    "scanned_at": datetime.now().isoformat(),
    "data_root": data_root,
    "total_dirs": len(dirs),
    "total_annotation_files": total_ann,
    "total_annotation_lines": total_lines,
    "directories": dict(sorted(dirs.items())),
}

with open(output, "w") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"Dirs: {len(dirs)}  |  Annotation files: {total_ann}  |  Total lines: {total_lines:,}")
for name in sorted(dirs):
    d = dirs[name]
    fields_summary = set()
    for af in d["annotation_files"]:
        fields_summary.update(af.get("fields", []))
    print(f"  {name:40s}  {d['total_lines']:>10,} lines  {len(d['annotation_files'])} files  fields: {','.join(sorted(fields_summary))[:80]}")

print(f"\nExported to {output}")
PYEOF
