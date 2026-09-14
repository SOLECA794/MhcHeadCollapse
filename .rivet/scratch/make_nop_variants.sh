#!/bin/bash
# NOP 标定五连提交生成器（METHODOLOGY.md §三）
# 从 v119 源码生成 5 个 NOP 变体（N=0/100/500/1000/5000），打包到 variants/ 下
# 每个变体只在 Process() 尾部插入标量空转循环
set -e
SRC=/workspace/notebook13/MhcHeadCollapse/.rivet/scratch/v117_stage/code
BASE=/workspace/notebook13/MhcHeadCollapse/.rivet/scratch/nop_variants
rm -rf "$BASE"; mkdir -p "$BASE"

# 插入点：主 kernel 类 Process() 中的最后一个闭合（group_/else 分支后）
# 用 python 做精确文本手术，比 sed 稳
for NOP in 0 100 500 1000 5000; do
  DEST="$BASE/nop_$NOP"
  mkdir -p "$DEST"
  cp -r "$SRC"/op_kernel "$SRC"/op_host "$SRC"/CMakeLists.txt "$DEST/"
  if [ "$NOP" != "0" ]; then
    python3 - "$DEST/op_kernel/mhc_head_collapse.cpp" "$NOP" << 'PYEOF'
import sys, re
path, nops = sys.argv[1], int(sys.argv[2])
src = open(path).read()
# 在主类 Process() 的结尾插入 NOP：定位 "            for (uint32_t row = block_idx; row < outer_; row += block_dim_) {\n                ProcessVectorRow(row);\n            }\n        }"
anchor = """            for (uint32_t row = block_idx; row < outer_; row += block_dim_) {
                ProcessVectorRow(row);
            }
        }
    }"""
nop_code = f"""            for (uint32_t row = block_idx; row < outer_; row += block_dim_) {{
                ProcessVectorRow(row);
            }}
        }}
        // NOP-CALIB: burn {nops} iterations of scalar spin (~4-5 cycles each)
        volatile uint32_t nop_acc = 0;
        for (uint32_t i = 0; i < {nops}; ++i) {{
            nop_acc += i;
        }}
        (void)nop_acc;
    }}"""
assert anchor in src, "anchor not found"
src = src.replace(anchor, nop_code, 1)
open(path, 'w').write(src)
print(f"NOP={nops} injected")
PYEOF
  else
    echo "NOP=0 (baseline, no injection)"
  fi
done
ls "$BASE"