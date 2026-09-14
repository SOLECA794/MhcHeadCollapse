// MhcHeadCollapse shape 扫描 harness —— 本地 vs OJ 对拍核心件
// 口径：
//   percall  = 每 call 前后各打一个 event + sync（端到端，最接近 OJ 口径候选）
//   streamed = N 次 call 夹在两个 event 间，除以 N（kernel 均摊，消除 launch 噪声）
//   warm     = 丢弃前 kWarm 次（NPU 800→1650MHz 热身，PITFALLS 已记录）
// 用法：./mhc_shape_bench <n> <h> <outer> <fp16|fp32> [iters] [mode]
//   mode: percall(默认) | streamed | both
// 输出（单行，便于脚本采集）：
//   RESULT n=8 h=128 outer=2 dtype=fp16 mode=percall med=4.123 min=4.056 p90=4.312 nsamp=30
#include <acl/acl.h>
#include <aclnn_mhc_head_collapse.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK_ACL(expr)                                                        \
    do {                                                                       \
        const aclError status = (expr);                                        \
        if (status != ACL_SUCCESS) {                                           \
            std::fprintf(stderr, "%s failed: %d\n", #expr, (int)status);       \
            return 1;                                                          \
        }                                                                      \
    } while (0)

static uint16_t FloatToHalfBits(float v) {
    __fp16 h = static_cast<__fp16>(v);
    uint16_t bits;
    __builtin_memcpy(&bits, &h, 2);
    return bits;
}

static float Median(std::vector<float> &v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n & 1) ? v[n / 2] : 0.5f * (v[n / 2 - 1] + v[n / 2]);
}

int main(int argc, char **argv) {
    if (argc < 5) {
        std::fprintf(stderr,
            "usage: %s <n> <h> <outer> <fp16|fp32> [iters=50] [mode=both]\n", argv[0]);
        return 1;
    }
    const int64_t kN = std::atoll(argv[1]);
    const int64_t kH = std::atoll(argv[2]);
    const int64_t kOuter = std::atoll(argv[3]);
    const bool use_fp16 = std::string(argv[4]) == "fp16";
    const int kIters = (argc > 5) ? std::atoi(argv[5]) : 50;
    std::string mode = (argc > 6) ? argv[6] : "both";
    if (mode != "percall" && mode != "streamed" && mode != "both") mode = "both";
    const int64_t kNH = kN * kH;
    constexpr int kWarm = 25;  // NPU 频率热身（ENVIRONMENT §六）

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    // 输入数据（与 mhc_correctness 相同分布，保证可比）
    std::vector<float> x_f(kOuter * kNH), weight_f(kN * kNH), base_f(kN);
    for (size_t i = 0; i < x_f.size(); ++i)
        x_f[i] = std::sin(static_cast<float>(i) * 0.013f) * 0.9f;
    for (size_t i = 0; i < weight_f.size(); ++i)
        weight_f[i] = std::cos(static_cast<float>(i) * 0.0071f) * 0.05f;
    for (size_t i = 0; i < base_f.size(); ++i)
        base_f[i] = std::cos(static_cast<float>(i) * 0.31f) * 0.07f;
    std::vector<float> scale_f(kN, 1.0f);

    // device buffers
    void *x_dev = nullptr, *w_dev = nullptr, *b_dev = nullptr, *s_dev = nullptr, *y_dev = nullptr;
    const size_t x_bytes = use_fp16 ? kOuter * kNH * 2 : kOuter * kNH * 4;
    const size_t w_bytes = kN * kNH * 4;
    const size_t y_bytes = use_fp16 ? kOuter * kH * 2 : kOuter * kH * 4;
    CHECK_ACL(aclrtMalloc(&x_dev, x_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&w_dev, w_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&b_dev, kN * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&s_dev, kN * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&y_dev, y_bytes, ACL_MEM_MALLOC_HUGE_FIRST));

    if (use_fp16) {
        std::vector<uint16_t> x_h(x_f.size()), w_h(weight_f.size());
        for (size_t i = 0; i < x_f.size(); ++i) x_h[i] = FloatToHalfBits(x_f[i]);
        CHECK_ACL(aclrtMemcpy(x_dev, x_bytes, x_h.data(), x_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
        // weight/base/scale 保持 fp32（赛题规格）
        CHECK_ACL(aclrtMemcpy(w_dev, w_bytes, weight_f.data(), w_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(b_dev, kN * 4, base_f.data(), kN * 4, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(s_dev, kN * 4, scale_f.data(), kN * 4, ACL_MEMCPY_HOST_TO_DEVICE));
    } else {
        CHECK_ACL(aclrtMemcpy(x_dev, x_bytes, x_f.data(), x_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(w_dev, w_bytes, weight_f.data(), w_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(b_dev, kN * 4, base_f.data(), kN * 4, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(s_dev, kN * 4, scale_f.data(), kN * 4, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    // tensors
    aclTensor *x_t = nullptr, *w_t = nullptr, *b_t = nullptr, *s_t = nullptr, *y_t = nullptr;
    {
        const int64_t x_dims[3] = {1, kOuter, kNH};
        const int64_t w_dims[2] = {kN, kNH};
        const int64_t b_dims[1] = {kN};
        const int64_t y_dims[3] = {1, kOuter, kH};
        const aclDataType xdt = use_fp16 ? ACL_FLOAT16 : ACL_FLOAT;
        x_t = aclCreateTensor(x_dims, 3, xdt, nullptr, 0);
        w_t = aclCreateTensor(w_dims, 2, ACL_FLOAT, nullptr, 0);
        b_t = aclCreateTensor(b_dims, 1, ACL_FLOAT, nullptr, 0);
        s_t = aclCreateTensor(b_dims, 1, ACL_FLOAT, nullptr, 0);
        y_t = aclCreateTensor(y_dims, 3, xdt, nullptr, 0);
    }
    uint64_t ws_size = 0;
    aclOpExecutor *exec = nullptr;
    int ret = aclnnMhcHeadCollapseGetWorkspaceSize(x_t, w_t, b_t, s_t, 1e-6f, 1e-6f, y_t,
                                                   &ws_size, &exec);
    if (ret != 0) {
        std::fprintf(stderr, "GetWorkspaceSize failed: %d\n", ret);
        return 1;
    }
    void *ws = nullptr;
    if (ws_size > 0) CHECK_ACL(aclrtMalloc(&ws, ws_size, ACL_MEM_MALLOC_HUGE_FIRST));

    auto run_once = [&]() -> int {
        return aclnnMhcHeadCollapse(ws, ws_size, exec, stream);
    };

    // 热身
    for (int i = 0; i < kWarm; ++i) run_once();
    CHECK_ACL(aclrtSynchronizeStream(stream));

    const char *dtype = use_fp16 ? "fp16" : "fp32";

    if (mode == "percall" || mode == "both") {
        aclrtEvent ev0, ev1;
        CHECK_ACL(aclrtCreateEvent(&ev0));
        CHECK_ACL(aclrtCreateEvent(&ev1));
        std::vector<float> samples;
        samples.reserve(kIters);
        for (int i = 0; i < kIters; ++i) {
            CHECK_ACL(aclrtRecordEvent(ev0, stream));
            run_once();
            CHECK_ACL(aclrtRecordEvent(ev1, stream));
            CHECK_ACL(aclrtSynchronizeEvent(ev1));
            float ms = 0.0f;
            CHECK_ACL(aclrtElapsedTime(&ms, ev0, ev1));
            samples.push_back(ms * 1000.0f);  // → μs
        }
        std::vector<float> sorted(samples);
        float med = Median(sorted);
        float mn = *std::min_element(samples.begin(), samples.end());
        std::sort(samples.begin(), samples.end());
        float p90 = samples[(size_t)(samples.size() * 0.9)];
        std::printf("RESULT n=%lld h=%lld outer=%lld dtype=%s mode=percall "
                    "med=%.3f min=%.3f p90=%.3f nsamp=%d\n",
                    (long long)kN, (long long)kH, (long long)kOuter, dtype,
                    med, mn, p90, kIters);
    }
    if (mode == "streamed" || mode == "both") {
        aclrtEvent ev0, ev1;
        CHECK_ACL(aclrtCreateEvent(&ev0));
        CHECK_ACL(aclrtCreateEvent(&ev1));
        std::vector<float> samples;
        const int kBatch = 10;
        for (int b = 0; b < kIters / kBatch; ++b) {
            CHECK_ACL(aclrtRecordEvent(ev0, stream));
            for (int i = 0; i < kBatch; ++i) run_once();
            CHECK_ACL(aclrtRecordEvent(ev1, stream));
            CHECK_ACL(aclrtSynchronizeEvent(ev1));
            float ms = 0.0f;
            CHECK_ACL(aclrtElapsedTime(&ms, ev0, ev1));
            samples.push_back(ms * 1000.0f / kBatch);  // μs/call
        }
        std::vector<float> sorted(samples);
        float med = Median(sorted);
        float mn = *std::min_element(samples.begin(), samples.end());
        std::sort(samples.begin(), samples.end());
        float p90 = samples[(size_t)(samples.size() * 0.9)];
        std::printf("RESULT n=%lld h=%lld outer=%lld dtype=%s mode=streamed "
                    "med=%.3f min=%.3f p90=%.3f nsamp=%d\n",
                    (long long)kN, (long long)kH, (long long)kOuter, dtype,
                    med, mn, p90, (int)samples.size());
    }
    return 0;
}
