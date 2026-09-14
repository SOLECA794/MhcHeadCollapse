# Generate WA timing probes from v100 (PATH2 kernel only). Each probe removes one
# stage while keeping the rest identical, so per-case times attribute cost by subtraction.
# Probe results are WRONG ANSWER by design; we only read the reported time.
import re, sys, io

BASE = r"D:/Desktop/OP-Learning/Ascend/xingchen-operators/C组赛题/【C组中等题】MhcHeadCollapse 算子（mHC四路归一）/versions"
SRC = BASE + "/v100_v095_nosv/code/op_kernel/mhc_head_collapse.cpp"

src = open(SRC, encoding="utf-8").read()
assert src.count("__aicore__ inline void ProcessVectorRow(uint32_t row) {") == 1

# ---------- P1 floor: empty per-row body (cadence/launch floor for PATH2) ----------
p1, n1 = re.subn(
    r"(    __aicore__ inline void ProcessVectorRow\(uint32_t row\) \{.*?\n    \}\n)(\n    __aicore__ static inline uint32_t AlignBytes)",
    r"    __aicore__ inline void ProcessVectorRow(uint32_t row) {\n"
    r"        // P1 floor probe: per-row body emptied. Times Case3-5 = cadence+launch floor.\n"
    r"    }\n\2",
    src, count=1, flags=re.S)
assert n1 == 1, n1

# ---------- P2 gateconst: drop the 8 gate dot-products + scalar sigmoid ----------
P2_OLD = """        const float hc_scale = cst_buf_.Get<float>().GetValue(8);

        // v086: gates from UB-cached weight rows; n ReduceSums issued back-to-back.
        // Gate results go to reduce_buf_ blocks 1..n (block 0 holds rms); cst_buf_ holds base/scale.
        AscendC::LocalTensor<float> w_cache = weight_cache_buf_.Get<float>();
        AscendC::LocalTensor<float> cst = cst_buf_.Get<float>();
        for (uint32_t i = 0; i < n_; ++i) {
            AscendC::Mul(work, x_float, w_cache[i * nH_], static_cast<int32_t>(nH_));
            AscendC::ReduceSum(reduced[32U * (i + 1U)], work, reduce_tmp, static_cast<int32_t>(nH_));
        }
        float gates[kSmallN];
        for (uint32_t i = 0; i < n_; ++i) {
            const float gate = reduced[32U * (i + 1U)].GetValue(0) * rms_inv * hc_scale + cst.GetValue(i);
            gates[i] = ScalarSigmoid(gate) + eps_hc_;
        }
"""
P2_NEW = """        // P2 gateconst probe: 8 gate dots + sigmoid -> constants. rms path and the
        // h-long fold/store stay identical; gates[0] keeps rms_inv live.
        float gates[kSmallN];
        gates[0] = 0.5f * rms_inv + eps_hc_;
        for (uint32_t i = 1; i < n_; ++i) {
            gates[i] = 0.5f + eps_hc_;
        }
"""
assert src.count(P2_OLD) == 1
p2 = src.replace(P2_OLD, P2_NEW, 1)

# ---------- P3 nofold: keep dots+sigmoid, replace h-length fold/store with a scalar GM write ----------
P3_OLD = """        AscendC::LocalTensor<DT_X> y_local = y_queue_.AllocTensor<DT_X>();
        AscendC::LocalTensor<float> output;
        if constexpr (sizeof(DT_X) == sizeof(float)) {
            output = y_local.template ReinterpretCast<float>();
        } else {
            output = output_buf_.Get<float>();
        }

        AscendC::Muls(output, x_float, gates[0], static_cast<int32_t>(h_));
        for (uint32_t i = 1; i < n_; ++i) {
            const uint32_t stream_offset = i * h_;
            AscendC::Muls(work, x_float[stream_offset], gates[i], static_cast<int32_t>(h_));
            AscendC::Add(output, output, work, static_cast<int32_t>(h_));
        }

        if constexpr (sizeof(DT_X) != sizeof(float)) {
            AscendC::Cast(y_local, output, AscendC::RoundMode::CAST_RINT, h_);
        }
        y_queue_.EnQue(y_local);
        y_local = y_queue_.DeQue<DT_X>();
        AscendC::DataCopy(y_gm_[y_offset], y_local, h_);
        y_queue_.FreeTensor(y_local);
        x_queue_.FreeTensor(x_local);
"""
P3_NEW = """        // P3 nofold probe: scalar sink keeps the whole gate chain live; the h-long
        // Muls/Adds/Cast/DataCopy fold+store path is removed.
        float acc = 0.0f;
        for (uint32_t i = 0; i < n_; ++i) {
            acc += gates[i];
        }
        y_gm_.SetValue(y_offset, ScalarFromFloat<DT_X>(acc));
        x_queue_.FreeTensor(x_local);
"""
assert src.count(P3_OLD) == 1
p3 = src.replace(P3_OLD, P3_NEW, 1)

for name, body in [("p1_floor_probe", p1), ("p2_gateconst_probe", p2), ("p3_nofold_probe", p3)]:
    out = BASE + "/" + name + "/code/op_kernel/mhc_head_collapse.cpp"
    body = body.replace("// v093: v090 PATH3 fix", "// " + name + ": generated from v100_v095_nosv (WA timing probe)") if False else body
    open(out, "w", encoding="utf-8").write(body)
    print("wrote", out)
print("OK")
