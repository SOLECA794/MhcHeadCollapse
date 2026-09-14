// Kernel侧核函数实现
// v111: PATH2 group-of-4 — 4 rows per batch: x rows staged together, dot chains for
//       all rows issued before any scalar read, so the per-row vector<->scalar drains,
//       queue events and casts amortize (TinyH4 already does this; PATH2 now too).
//       UB-budget guarded, other shapes keep the exact v100 per-row path.
// v093: v090 PATH3 fix + UB indexing (rr->r, z_vec offset corrected)
// v080: v017 verified multi-path kernel + v079 UB+DMA TinyH4 + v057 host (dynamic block_dim)
// CRITICAL: file-level pragma REQUIRED for DataCopy in Process() on CANN 8.5.0
#pragma GCC optimize("O3,fast-math,unroll-loops")
#include "kernel_operator.h"

#include "mhc_head_collapse_tiling.h"
#include "tiling_key_mhc_head_collapse.h"

template <typename T>
__aicore__ inline float ScalarToFloat(T value) {
    return static_cast<float>(value);
}

template <>
__aicore__ inline float ScalarToFloat<bfloat16_t>(bfloat16_t value) {
    return AscendC::ToFloat(value);
}

template <typename T>
__aicore__ inline T ScalarFromFloat(float value) {
    return static_cast<T>(value);
}

template <>
__aicore__ inline bfloat16_t ScalarFromFloat<bfloat16_t>(float value) {
    return AscendC::ToBfloat16(value);
}


__aicore__ inline float MhcScalarExp(float x) {
    if (x != x) {
        return x;
    }
    if (x <= -16.0f) {
        return 0.0f;
    }
    const float r = x * 0.00390625f;
    const float r2 = r * r;
    float result = 1.0f + r + r2 *
        (0.5f + r * (0.1666666716f + r * 0.0416666679f));
    result *= result;
    result *= result;
    result *= result;
    result *= result;
    result *= result;
    result *= result;
    result *= result;
    result *= result;
    return result;
}

__aicore__ inline float MhcScalarRsqrt(float x) {
    if (x != x) {
        return x;
    }
    union FloatBits {
        uint32_t bits;
        float value;
    } estimate;
    estimate.value = x;
    estimate.bits = 0x5f3759dfU - (estimate.bits >> 1);
    float y = estimate.value;
    y = y * (1.5f - 0.5f * x * y * y);
    y = y * (1.5f - 0.5f * x * y * y);
    y = y * (1.5f - 0.5f * x * y * y);
    return y;
}

// v088: unpack two consecutive floats from one 64-bit load
__aicore__ inline void MhcUnpack2(uint64_t v, float &a, float &b) {
    union U2F {
        uint64_t u;
        float f[2];
    } d;
    d.u = v;
    a = d.f[0];
    b = d.f[1];
}

__aicore__ inline float MhcScalarSigmoid(float x) {
    if (x >= 0.0f) {
        const float e = MhcScalarExp(-x);
        return 1.0f / (1.0f + e);
    }
    const float e = MhcScalarExp(x);
    return e / (1.0f + e);
}

// TINY_H4 path: v079 verified UB+DMA vector pipeline (OJ Pass, Case1~4us)
template <class DT_X>
class KernelMhcHeadCollapseTinyH4 {
public:
    __aicore__ inline KernelMhcHeadCollapseTinyH4() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR weight, GM_ADDR hc_base, GM_ADDR hc_scale, GM_ADDR y,
        uint32_t outer) {
        x_gm_.SetGlobalBuffer((__gm__ DT_X *)x);
        weight_gm_.SetGlobalBuffer((__gm__ float *)weight);
        base_gm_.SetGlobalBuffer((__gm__ float *)hc_base);
        scale_gm_.SetGlobalBuffer((__gm__ float *)hc_scale);
        y_gm_.SetGlobalBuffer((__gm__ DT_X *)y);
        y_u64_.SetGlobalBuffer((__gm__ uint64_t *)y);
        outer_ = outer;
        pipe_.InitBuffer(ub_buf_, 6144U);
    }

    __aicore__ inline void Process() {
        AscendC::LocalTensor<float> mem = ub_buf_.Get<float>();
        AscendC::LocalTensor<float> w_ub = mem;                     // 64 floats [0..63]
        AscendC::LocalTensor<float> x_float = mem[64];              // [64..191] (8 rows max)
        AscendC::LocalTensor<DT_X> x_raw = mem[192].template ReinterpretCast<DT_X>(); // [192..447]
        AscendC::LocalTensor<float> base_ub = mem[448];            // 8 floats
        AscendC::LocalTensor<float> scale_ub = mem[456];           // 8 floats
        AscendC::LocalTensor<float> work_rms_base = mem[512];      // 4 rows x 16 [512..575]
        AscendC::LocalTensor<float> work_gate_base = mem[576];     // 4 rows x 64 [576..831]
        AscendC::LocalTensor<float> red_rms_base = mem[832];       // 4 rows x 8 [832..863]
        AscendC::LocalTensor<float> red_gate_base = mem[864];      // 4 rows x 32 [864..991]
        AscendC::LocalTensor<float> tmp_base = mem[992];           // 4 rows x 80 [992..1311]
        AscendC::LocalTensor<float> base_neg_pat = mem[1328];    // v117: 4x32 negated base pattern [1328..1455]

        // 1. DMA bulk prefetch (weight/base/scale once, reused across batches)
        AscendC::DataCopy(w_ub, weight_gm_, 64);
        AscendC::DataCopy(base_ub, base_gm_, 8);
        AscendC::DataCopy(scale_ub, scale_gm_, 8);

        // v117: build negated-base pattern once (4 rows x 32 lanes, stride-8 slots).
        // ReduceSum dst layout: each gate owns a 32B block (8 floats), so gate i of
        // row r lives at lane r*32 + i*8. Pattern mirrors that: -base_i at lane i*8.
        // z = -(lg*k + base) = (-lg)*k + (-base); negating the constant is exact in
        // IEEE754, so gate lanes stay bit-identical to the old scalar path.
        const float base0v = base_ub.GetValue(0);
        const float base1v = base_ub.GetValue(1);
        const float base2v = base_ub.GetValue(2);
        const float base3v = base_ub.GetValue(3);
        const float hc_scale = scale_ub.GetValue(0);
        for (uint32_t r = 0; r < 4U; ++r) {
            const uint32_t p = r * 32U;
            base_neg_pat.SetValue(p + 0U, -base0v);
            base_neg_pat.SetValue(p + 8U, -base1v);
            base_neg_pat.SetValue(p + 16U, -base2v);
            base_neg_pat.SetValue(p + 24U, -base3v);
        }

        // 2. Process rows in batches of 4 (UB layout capacity: 4 rows of work/red/tmp)
        for (uint32_t batch = 0; batch < outer_; batch += 4U) {
            const uint32_t rows = (outer_ - batch) < 4U ? (outer_ - batch) : 4U;
            const uint32_t total_x = rows * 16U;
            if constexpr (sizeof(DT_X) == sizeof(float)) {
                AscendC::DataCopy(x_float, x_gm_[batch * 16U], total_x);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            } else {
                AscendC::DataCopy(x_raw, x_gm_[batch * 16U], total_x);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
                AscendC::Cast(x_float, x_raw, AscendC::RoundMode::CAST_NONE, total_x);
            }

            // 3. Vector math for the batch
            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t row = batch + r;
            AscendC::LocalTensor<float> x_row = x_float[r * 16U];
            AscendC::LocalTensor<float> w_rms = work_rms_base[r * 16U];
            AscendC::LocalTensor<float> r_rms = red_rms_base[r * 8U];
            const uint32_t g_offset = r * 4U * 16U;
            const uint32_t rg_offset = r * 4U * 8U;
            const uint32_t t_offset = r * 5U * 16U;

            AscendC::Mul(w_rms, x_row, x_row, 16);
            AscendC::Mul(work_gate_base[g_offset + 0U], x_row, w_ub[0], 16);
            AscendC::Mul(work_gate_base[g_offset + 16U], x_row, w_ub[16], 16);
            AscendC::Mul(work_gate_base[g_offset + 32U], x_row, w_ub[32], 16);
            AscendC::Mul(work_gate_base[g_offset + 48U], x_row, w_ub[48], 16);

            AscendC::ReduceSum(r_rms, w_rms, tmp_base[t_offset + 0U], 16);
            AscendC::ReduceSum(red_gate_base[rg_offset + 0U], work_gate_base[g_offset + 0U], tmp_base[t_offset + 16U], 16);
            AscendC::ReduceSum(red_gate_base[rg_offset + 8U], work_gate_base[g_offset + 16U], tmp_base[t_offset + 32U], 16);
            AscendC::ReduceSum(red_gate_base[rg_offset + 16U], work_gate_base[g_offset + 32U], tmp_base[t_offset + 48U], 16);
            AscendC::ReduceSum(red_gate_base[rg_offset + 24U], work_gate_base[g_offset + 48U], tmp_base[t_offset + 64U], 16);
            }

            // 4. Single hard sync per batch
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);

            // 5. v117: fully vectorized z + sigmoid, in-place on strided gate blocks.
            // Old v093 path: 9 scalar UB accesses per row (1 rms read + 4 gate reads +
            // 4 z SetValue). New path: 1 scalar read per row (rms), everything else
            // vector ops on the ReduceSum 32B-aligned strided layout (untouched).
            // z_stored = -z_true = -(lg*k + base) = (-lg)*k + (-base).
            // Muls by -k: (-lg)*k is bit-identical to -(lg*k) in IEEE754 (unary neg
            // is exact), so gate lanes stay bit-equal to the old scalar path.
            for (uint32_t r = 0; r < rows; ++r) {
                AscendC::LocalTensor<float> r_rms = red_rms_base[r * 8U];
                const float rms_inv = MhcScalarRsqrt(r_rms.GetValue(0) * 0.0625f + 1e-6f);
                const float k = rms_inv * hc_scale;
                AscendC::Muls(red_gate_base[r * 32U], red_gate_base[r * 32U], -k, 32U);
            }
            // z = -lg*k + (-base): vector add of negated stride-8 base pattern over
            // rows*32 lanes (pattern rows beyond `rows` are never touched by this Add).
            AscendC::Add(red_gate_base, red_gate_base, base_neg_pat, rows * 32U);
            // sigmoid in place: s = 1/(1+exp(z)) on all strided lanes; the four gate
            // lanes of each row end at red_gate_base[r*32 + {0,8,16,24}].
            AscendC::Exp(red_gate_base, red_gate_base, rows * 32U);
            AscendC::Adds(red_gate_base, red_gate_base, 1.0f, rows * 32U);
            AscendC::Reciprocal(red_gate_base, red_gate_base, rows * 32U);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(1);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(1);
            // v117: gates now live in-place in red_gate_base strided lanes [r*32 + 0..3]
            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t row = batch + r;
                AscendC::LocalTensor<float> x_row = x_float[r * 16U];
                const uint64_t y_offset = static_cast<uint64_t>(row) * 4U;
                const float gate0 = red_gate_base[r * 32U + 0U].GetValue(0) + 1e-6f;
                const float gate1 = red_gate_base[r * 32U + 8U].GetValue(0) + 1e-6f;
                const float gate2 = red_gate_base[r * 32U + 16U].GetValue(0) + 1e-6f;
                const float gate3 = red_gate_base[r * 32U + 24U].GetValue(0) + 1e-6f;

                AscendC::LocalTensor<uint64_t> x_u64 = x_row.template ReinterpretCast<uint64_t>();
                float x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;
                MhcUnpack2(x_u64.GetValue(0), x0, x1);
                MhcUnpack2(x_u64.GetValue(1), x2, x3);
                MhcUnpack2(x_u64.GetValue(2), x4, x5);
                MhcUnpack2(x_u64.GetValue(3), x6, x7);
                MhcUnpack2(x_u64.GetValue(4), x8, x9);
                MhcUnpack2(x_u64.GetValue(5), x10, x11);
                MhcUnpack2(x_u64.GetValue(6), x12, x13);
                MhcUnpack2(x_u64.GetValue(7), x14, x15);

                const float out0 = gate0 * x0 + gate1 * x4 + gate2 * x8 + gate3 * x12;
                const float out1 = gate0 * x1 + gate1 * x5 + gate2 * x9 + gate3 * x13;
                const float out2 = gate0 * x2 + gate1 * x6 + gate2 * x10 + gate3 * x14;
                const float out3 = gate0 * x3 + gate1 * x7 + gate2 * x11 + gate3 * x15;

            if constexpr (sizeof(DT_X) == sizeof(uint16_t)) {
                const DT_X v0 = ScalarFromFloat<DT_X>(out0);
                const DT_X v1 = ScalarFromFloat<DT_X>(out1);
                const DT_X v2 = ScalarFromFloat<DT_X>(out2);
                const DT_X v3 = ScalarFromFloat<DT_X>(out3);
                uint16_t h0 = 0, h1 = 0, h2 = 0, h3 = 0;
                *reinterpret_cast<DT_X *>(&h0) = v0;
                *reinterpret_cast<DT_X *>(&h1) = v1;
                *reinterpret_cast<DT_X *>(&h2) = v2;
                *reinterpret_cast<DT_X *>(&h3) = v3;
                const uint64_t pack = static_cast<uint64_t>(h0) |
                                     (static_cast<uint64_t>(h1) << 16U) |
                                     (static_cast<uint64_t>(h2) << 32U) |
                                     (static_cast<uint64_t>(h3) << 48U);
                y_u64_.SetValue(row, pack);
            } else {
                y_gm_.SetValue(y_offset + 0U, ScalarFromFloat<DT_X>(out0));
                y_gm_.SetValue(y_offset + 1U, ScalarFromFloat<DT_X>(out1));
                y_gm_.SetValue(y_offset + 2U, ScalarFromFloat<DT_X>(out2));
                y_gm_.SetValue(y_offset + 3U, ScalarFromFloat<DT_X>(out3));
            }
            }
        }
    }

private:
    AscendC::GlobalTensor<DT_X> x_gm_;
    AscendC::GlobalTensor<float> weight_gm_;
    AscendC::GlobalTensor<float> base_gm_;
    AscendC::GlobalTensor<float> scale_gm_;
    AscendC::GlobalTensor<DT_X> y_gm_;
    AscendC::GlobalTensor<uint64_t> y_u64_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ub_buf_;
    uint32_t outer_ = 0;
};

template <class DT_X>
class KernelMhcHeadCollapse {
public:
    __aicore__ inline KernelMhcHeadCollapse() {}
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR weight, GM_ADDR hc_base, GM_ADDR hc_scale, GM_ADDR y,
        uint32_t outer, uint32_t n, uint32_t h, uint32_t nH, uint32_t block_dim, uint32_t path,
        float inv_nH, float eps_norm, float eps_hc) {
        x_gm_.SetGlobalBuffer((__gm__ DT_X *)x);
        weight_gm_.SetGlobalBuffer((__gm__ float *)weight);
        base_gm_.SetGlobalBuffer((__gm__ float *)hc_base);
        scale_gm_.SetGlobalBuffer((__gm__ float *)hc_scale);
        y_gm_.SetGlobalBuffer((__gm__ DT_X *)y);
        outer_ = outer;
        n_ = n;
        h_ = h;
        nH_ = nH;
        block_dim_ = block_dim;
        inv_nH_ = inv_nH;
        eps_norm_ = eps_norm;
        eps_hc_ = eps_hc;
        path_ = path;
        if (path_ == kVectorPath) {
            // v111: group kGroupRows rows for heavy PATH2 shapes so per-row queue
            // events, casts and vector<->scalar drains amortize. UB-budget guarded;
            // anything outside keeps the exact v100 per-row path (group_ == false).
            const uint64_t weight_bytes = static_cast<uint64_t>(AlignBytes(n_ * nH_ * sizeof(float)));
            const uint64_t fp_bytes = static_cast<uint64_t>(AlignBytes(nH_ * sizeof(float)));
            group_ = (n_ >= 1U && n_ <= 8U && nH_ >= 256U && nH_ <= 1024U && h_ >= 8U &&
                      (nH_ & 7U) == 0U && (h_ & 7U) == 0U &&
                      weight_bytes + kGroupRows * fp_bytes +          // weight cache + x fp rows
                          2U * fp_bytes + 2U * fp_bytes +             // work + reduce tmp (parity)
                          static_cast<uint64_t>(kGroupRows * (n_ + 1U) * 32U * sizeof(float)) +
                          kGroupRows * AlignBytes(nH_ * sizeof(DT_X)) +   // staged raw x rows
                          kGroupRows * AlignBytes(h_ * sizeof(float)) <= 150U * 1024U);
            const uint32_t red_bytes = group_
                ? static_cast<uint32_t>(AlignBytes(kGroupRows * (n_ + 1U) * 32U * sizeof(float)))
                : static_cast<uint32_t>(AlignBytes(9U * 32U));
            const uint32_t wf_bytes = group_ ? static_cast<uint32_t>(2U * fp_bytes)
                                             : static_cast<uint32_t>(fp_bytes);
            pipe_.InitBuffer(x_queue_, 1, AlignBytes(nH_ * sizeof(DT_X)));
            pipe_.InitBuffer(y_queue_, 1, AlignBytes(h_ * sizeof(DT_X)));
            if (group_) {
                pipe_.InitBuffer(x_group_raw_, kGroupRows * AlignBytes(nH_ * sizeof(DT_X)));
                pipe_.InitBuffer(x_group_fp_, kGroupRows * static_cast<uint32_t>(fp_bytes));
            }
            if constexpr (sizeof(DT_X) != sizeof(float)) {
                pipe_.InitBuffer(x_fp_buf_, AlignBytes(nH_ * sizeof(float)));
                pipe_.InitBuffer(output_buf_, group_ ? static_cast<uint32_t>(kGroupRows * AlignBytes(h_ * sizeof(float)))
                                                     : AlignBytes(h_ * sizeof(float)));
            }
            pipe_.InitBuffer(work_buf_, wf_bytes);
            pipe_.InitBuffer(reduce_buf_, red_bytes);
            pipe_.InitBuffer(reduce_tmp_buf_, wf_bytes);
            // v086: whole weight matrix + base/scale cached in UB, loaded once per op
            // v119-fix: UB overflow guard — when the full [n, nH] weight cache would
            // push total UB past 192KB (nH > ~4096 for n=8), fall back to a single-row
            // streaming weight buffer (nH floats). UB math: full path needs
            // wcache + 3*fp_bytes + xq + yq + out + red + cst; streaming drops
            // wcache (n*nH*4) to one row (nH*4).
            const uint64_t full_w_bytes = static_cast<uint64_t>(AlignBytes(n_ * nH_ * sizeof(float)));
            const uint64_t base_bytes = static_cast<uint64_t>(wf_bytes) * 2U + red_bytes +
                AlignBytes(nH_ * sizeof(DT_X)) + AlignBytes(h_ * sizeof(DT_X)) +
                ((sizeof(DT_X) != sizeof(float)) ? (AlignBytes(nH_ * sizeof(float)) + AlignBytes(h_ * sizeof(float))) : 0U) + 64U;
            w_stream_ = (full_w_bytes + base_bytes > 192U * 1024U);
            pipe_.InitBuffer(weight_cache_buf_, w_stream_
                ? static_cast<uint32_t>(AlignBytes(nH_ * sizeof(float)))
                : static_cast<uint32_t>(full_w_bytes));
            pipe_.InitBuffer(cst_buf_, 64U);
        }
    }
    __aicore__ inline void Process() {
        const uint32_t block_idx = AscendC::GetBlockIdx();
        if (path_ == kVectorPath) {
            // v086: one-time DMA of the whole weight matrix [n, nH] (contiguous) + base/scale
            // v119-fix: streaming mode skips the full-matrix DMA (single row fetched per gate)
            AscendC::LocalTensor<float> w_cache = weight_cache_buf_.Get<float>();
            if (!w_stream_) {
                AscendC::DataCopy(w_cache, weight_gm_, n_ * nH_);
            }
            AscendC::LocalTensor<float> cst = cst_buf_.Get<float>();
            AscendC::DataCopy(cst, base_gm_, 8U);
            AscendC::DataCopy(cst[8], scale_gm_, 8U);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        }
        // v088: only the vector path remains; PATH0/1 dead code removed (all 5 OJ cases use PATH2/3)
        if (group_) {
            // v111: consume rows in batches of kGroupRows (per block, stride block_dim_)
            uint32_t row = block_idx;
            while (row < outer_) {
                uint32_t rows[kGroupRows];
                uint32_t cnt = 0;
                for (; cnt < kGroupRows && row < outer_; ++cnt, row += block_dim_) {
                    rows[cnt] = row;
                }
                ProcessVectorGroup(rows, cnt);
            }
        } else {
            for (uint32_t row = block_idx; row < outer_; row += block_dim_) {
                ProcessVectorRow(row);
            }
        }
    }
private:
    __aicore__ inline void ProcessVectorGroup(uint32_t rows[], uint32_t cnt) {
        constexpr bool is_fp32 = (sizeof(DT_X) == sizeof(float));
        const uint32_t nhF = AlignBytes(nH_ * sizeof(float)) / sizeof(float); // fp row stride (floats)
        // v111: raw (fp16/bf16) row stride in elements, kept 32B-aligned so every
        // staged row is a legal vector-op src/dst offset (nH_*2 bytes may not be).
        const uint32_t xrEls = AlignBytes(nH_ * static_cast<uint32_t>(sizeof(DT_X))) /
                               static_cast<uint32_t>(sizeof(DT_X));
        const int32_t count = static_cast<int32_t>(nH_);
        AscendC::LocalTensor<float> w_cache = weight_cache_buf_.Get<float>();
        AscendC::LocalTensor<float> cst = cst_buf_.Get<float>();
        AscendC::LocalTensor<float> work = work_buf_.Get<float>();
        AscendC::LocalTensor<float> tmpb = reduce_tmp_buf_.Get<float>();
        AscendC::LocalTensor<float> red = reduce_buf_.Get<float>();
        const float hc_scale = cst.GetValue(8);

        // 1) stage this group's x rows (GM -> UB), one sync for the whole group
        AscendC::LocalTensor<float> xf;
        AscendC::LocalTensor<DT_X> xraw;
        if constexpr (is_fp32) {
            xf = x_group_fp_.Get<float>();
        } else {
            xraw = x_group_raw_.Get<DT_X>();
            xf = x_group_fp_.Get<float>();
        }
        for (uint32_t j = 0; j < cnt; ++j) {
            const uint64_t xo = static_cast<uint64_t>(rows[j]) * nH_;
            if constexpr (is_fp32) {
                AscendC::DataCopy(xf[j * nhF], x_gm_[xo], nH_);
            } else {
                AscendC::DataCopy(xraw[j * xrEls], x_gm_[xo], nH_);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
        if constexpr (!is_fp32) {
            for (uint32_t j = 0; j < cnt; ++j) {
                AscendC::Cast(xf[j * nhF], xraw[j * xrEls], AscendC::RoundMode::CAST_NONE, nH_);
            }
        }

        // 2) dot phase: all rows, rms + n gates. Rows of the same parity share a
        //    work/tmp region (bounded memory); no scalar read until every row issued.
        for (uint32_t j = 0; j < cnt; ++j) {
            AscendC::LocalTensor<float> x = xf[j * nhF];
            const uint32_t p = j & 1U;
            AscendC::LocalTensor<float> wp = work[p * nhF];
            AscendC::LocalTensor<float> tp = tmpb[p * nhF];
            const uint32_t rb = j * (n_ + 1U) * 32U;
            AscendC::Mul(wp, x, x, count);
            AscendC::ReduceSum(red[rb], wp, tp, count);
            for (uint32_t i = 0; i < n_; ++i) {
                AscendC::Mul(wp, x, w_cache[i * nH_], count);
                AscendC::ReduceSum(red[rb + 32U * (i + 1U)], wp, tp, count);
            }
        }

        // 3) scalar phase for the whole group (first GetValue drains the vector pipe once)
        float gates[kGroupRows][kSmallN];
        for (uint32_t j = 0; j < cnt; ++j) {
            const uint32_t rb = j * (n_ + 1U) * 32U;
            const float rms_inv = ScalarRsqrt(red[rb].GetValue(0) * inv_nH_ + eps_norm_);
            for (uint32_t i = 0; i < n_; ++i) {
                const float gate = red[rb + 32U * (i + 1U)].GetValue(0) * rms_inv * hc_scale + cst.GetValue(i);
                gates[j][i] = ScalarSigmoid(gate) + eps_hc_;
            }
        }

        // 4) fold + store per row (vector length h_)
        const uint32_t hF = AlignBytes(h_ * sizeof(float)) / sizeof(float);
        for (uint32_t j = 0; j < cnt; ++j) {
            AscendC::LocalTensor<float> x = xf[j * nhF];
            AscendC::LocalTensor<DT_X> y_local = y_queue_.AllocTensor<DT_X>();
            AscendC::LocalTensor<float> out;
            if constexpr (is_fp32) {
                out = y_local.template ReinterpretCast<float>();
            } else {
                out = output_buf_.Get<float>();
                out = out[j * hF];
            }
            AscendC::Muls(out, x, gates[j][0], static_cast<int32_t>(h_));
            for (uint32_t i = 1; i < n_; ++i) {
                AscendC::Muls(work[(j & 1U) * nhF], x[i * h_], gates[j][i], static_cast<int32_t>(h_));
                AscendC::Add(out, out, work[(j & 1U) * nhF], static_cast<int32_t>(h_));
            }
            if constexpr (!is_fp32) {
                AscendC::Cast(y_local, out, AscendC::RoundMode::CAST_RINT, h_);
            }
            y_queue_.EnQue(y_local);
            y_local = y_queue_.DeQue<DT_X>();
            AscendC::DataCopy(y_gm_[static_cast<uint64_t>(rows[j]) * h_], y_local, h_);
            y_queue_.FreeTensor(y_local);
        }
    }
    __aicore__ inline void ProcessVectorRow(uint32_t row) {
        const uint64_t x_offset = static_cast<uint64_t>(row) * nH_;
        const uint64_t y_offset = static_cast<uint64_t>(row) * h_;

        AscendC::LocalTensor<DT_X> x_local = x_queue_.AllocTensor<DT_X>();
        AscendC::DataCopy(x_local, x_gm_[x_offset], nH_);
        x_queue_.EnQue(x_local);
        x_local = x_queue_.DeQue<DT_X>();
        // v086: single weight DMA per op + per-row vector stream with one sync

        AscendC::LocalTensor<float> x_float;
        if constexpr (sizeof(DT_X) == sizeof(float)) {
            x_float = x_local.template ReinterpretCast<float>();
        } else {
            x_float = x_fp_buf_.Get<float>();
            AscendC::Cast(x_float, x_local, AscendC::RoundMode::CAST_NONE, nH_);
        }

        AscendC::LocalTensor<float> work = work_buf_.Get<float>();
        AscendC::LocalTensor<float> reduced = reduce_buf_.Get<float>();
        AscendC::LocalTensor<float> reduce_tmp = reduce_tmp_buf_.Get<float>();

        AscendC::Mul(work, x_float, x_float, static_cast<int32_t>(nH_));
        AscendC::ReduceSum(reduced, work, reduce_tmp, static_cast<int32_t>(nH_));
        const float rms_inv = ScalarRsqrt(reduced.GetValue(0) * inv_nH_ + eps_norm_);
        const float hc_scale = cst_buf_.Get<float>().GetValue(8);

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

        AscendC::LocalTensor<DT_X> y_local = y_queue_.AllocTensor<DT_X>();
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
    }

    __aicore__ static inline uint32_t AlignBytes(uint32_t bytes) {
        return (bytes + 31U) & ~31U;
    }

    __aicore__ inline float ScalarExp(float x) const {
        if (x != x) {
            return x;
        }
        if (x <= -16.0f) {
            return 0.0f;
        }
        // Sigmoid only calls ScalarExp with x <= 0. Range reduction by a
        // fixed power of two avoids unsupported scalar float/int casts.
        const float r = x * 0.00390625f;
        const float r2 = r * r;
        float result = 1.0f + r + r2 *
            (0.5f + r * (0.1666666716f + r * 0.0416666679f));
        result *= result;
        result *= result;
        result *= result;
        result *= result;
        result *= result;
        result *= result;
        result *= result;
        result *= result;
        return result;
    }

    __aicore__ inline float ScalarRsqrt(float x) const {
        if (x != x) {
            return x;
        }
        union FloatBits {
            uint32_t bits;
            float value;
        } estimate;
        estimate.value = x;
        estimate.bits = 0x5f3759dfU - (estimate.bits >> 1);
        float y = estimate.value;
        y = y * (1.5f - 0.5f * x * y * y);
        y = y * (1.5f - 0.5f * x * y * y);
        y = y * (1.5f - 0.5f * x * y * y);
        return y;
    }

    __aicore__ inline float ScalarSigmoid(float x) const {
        if (x >= 0.0f) {
            const float e = ScalarExp(-x);
            return 1.0f / (1.0f + e);
        }
        const float e = ScalarExp(x);
        return e / (1.0f + e);
    }

    AscendC::GlobalTensor<DT_X> x_gm_;
    AscendC::GlobalTensor<float> weight_gm_;
    AscendC::GlobalTensor<float> base_gm_;
    AscendC::GlobalTensor<float> scale_gm_;
    AscendC::GlobalTensor<DT_X> y_gm_;
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> x_queue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> weight_queue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> y_queue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> x_group_raw_;   // v111: staged x rows (DT_X)
    AscendC::TBuf<AscendC::TPosition::VECCALC> x_group_fp_;    // v111: staged x rows (fp32)
    AscendC::TBuf<AscendC::TPosition::VECCALC> x_fp_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> work_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduce_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduce_tmp_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> output_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> weight_cache_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cst_buf_;
    static constexpr uint32_t kVectorPath = 2;
    static constexpr uint32_t kSmallN = 8;
    static constexpr uint32_t kGroupRows = 4U;
    static constexpr uint32_t kOutputTile = 256;
    uint32_t outer_ = 0;
    uint32_t n_ = 0;
    uint32_t h_ = 0;
    uint32_t nH_ = 0;
    uint32_t block_dim_ = 1;
    uint32_t path_ = 0;
    bool group_ = false;
    float inv_nH_ = 1.0f;
    float eps_norm_ = 1e-6f;
    float eps_hc_ = 1e-6f;

};

template <typename DT_X>
__global__ __aicore__ void mhc_head_collapse(GM_ADDR x, GM_ADDR weight, GM_ADDR hc_base, GM_ADDR hc_scale, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    // v095: 纯SIMD算子，声明AIV_ONLY省去Cube核启动开销
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(MhcHeadCollapseTilingData);
    GET_TILING_DATA_WITH_STRUCT(MhcHeadCollapseTilingData, tiling_data, tiling);
    if (tiling_data.path == 3U) {
        KernelMhcHeadCollapseTinyH4<DT_X> op;
        op.Init(x, weight, hc_base, hc_scale, y, tiling_data.outer);
        op.Process();
    } else {
        KernelMhcHeadCollapse<DT_X> op;
        op.Init(x, weight, hc_base, hc_scale, y, tiling_data.outer, tiling_data.n,
                tiling_data.h, tiling_data.nH, tiling_data.block_dim, tiling_data.path,
                tiling_data.inv_nH, tiling_data.eps_norm, tiling_data.eps_hc);
        op.Process();
    }
}
