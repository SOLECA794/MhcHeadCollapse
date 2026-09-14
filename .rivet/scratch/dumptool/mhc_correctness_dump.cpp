// v080 correctness runner: rank-3 [1,outer,nH] with FP16/FP32, CPU reference compare.
// usage: ./mhc_correctness <n> <h> <outer> <fp16|fp32>
#include <acl/acl.h>
#include <aclnn_mhc_head_collapse.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
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

static float HalfBitsToFloat(uint16_t bits) {
    __fp16 h;
    __builtin_memcpy(&h, &bits, 2);
    return static_cast<float>(h);
}

int main(int argc, char **argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s <n> <h> <outer> <fp16|fp32>\n", argv[0]);
        return 1;
    }
    const int64_t kN = std::atoll(argv[1]);
    const int64_t kH = std::atoll(argv[2]);
    const int64_t kOuter = std::atoll(argv[3]);
    const bool use_fp16 = std::string(argv[4]) == "fp16";
    const int64_t kNH = kN * kH;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    // host inputs (float domain)
    std::vector<float> x_f(kOuter * kNH);
    std::vector<float> weight_f(kN * kNH);
    std::vector<float> base_f(kN);
    for (size_t i = 0; i < x_f.size(); ++i)
        x_f[i] = std::sin(static_cast<float>(i) * 0.013f) * 0.9f;
    for (size_t i = 0; i < weight_f.size(); ++i)
        weight_f[i] = std::cos(static_cast<float>(i) * 0.0071f) * 0.05f;
    for (size_t i = 0; i < base_f.size(); ++i)
        base_f[i] = (static_cast<float>(i) - 1.0f) * 0.2f;
    const float scale_f = 0.7f;

    const size_t x_bytes = x_f.size() * (use_fp16 ? 2 : 4);
    const size_t y_elems = kOuter * kH;
    const size_t y_bytes = y_elems * (use_fp16 ? 2 : 4);
    std::vector<uint16_t> x_h(x_f.size());
    std::vector<unsigned char> y_raw(y_bytes);
    std::vector<uint16_t> w_h(weight_f.size()), b_h(base_f.size()), s_h(1);
    if (use_fp16) {
        for (size_t i = 0; i < x_f.size(); ++i) x_h[i] = FloatToHalfBits(x_f[i]);
        for (size_t i = 0; i < weight_f.size(); ++i) w_h[i] = FloatToHalfBits(weight_f[i]);
        for (size_t i = 0; i < base_f.size(); ++i) b_h[i] = FloatToHalfBits(base_f[i]);
        s_h[0] = FloatToHalfBits(scale_f);
    }

    void *x_dev = nullptr, *w_dev = nullptr, *b_dev = nullptr, *s_dev = nullptr, *y_dev = nullptr;
    CHECK_ACL(aclrtMalloc(&x_dev, x_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&w_dev, weight_f.size() * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&b_dev, base_f.size() * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&s_dev, 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&y_dev, y_bytes, ACL_MEM_MALLOC_HUGE_FIRST));

    // weight/base/scale are FP32 per op def
    CHECK_ACL(aclrtMemcpy(w_dev, weight_f.size() * 4, weight_f.data(), weight_f.size() * 4, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(b_dev, base_f.size() * 4, base_f.data(), base_f.size() * 4, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(s_dev, 4, &scale_f, 4, ACL_MEMCPY_HOST_TO_DEVICE));
    if (use_fp16) {
        CHECK_ACL(aclrtMemcpy(x_dev, x_bytes, x_h.data(), x_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    } else {
        CHECK_ACL(aclrtMemcpy(x_dev, x_bytes, x_f.data(), x_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    std::vector<int64_t> x_shape = {1, kOuter, kNH};
    std::vector<int64_t> x_strides = {kOuter * kNH, kNH, 1};
    std::vector<int64_t> w_shape = {kN, kNH};
    std::vector<int64_t> w_strides = {kNH, 1};
    std::vector<int64_t> b_shape = {kN};
    std::vector<int64_t> b_strides = {1};
    std::vector<int64_t> s_shape = {1};
    std::vector<int64_t> s_strides = {1};
    std::vector<int64_t> y_shape = {1, kOuter, kH};
    std::vector<int64_t> y_strides = {kOuter * kH, kH, 1};
    const aclDataType x_dt = use_fp16 ? ACL_FLOAT16 : ACL_FLOAT;

    aclTensor *x_t = aclCreateTensor(x_shape.data(), 3, x_dt, x_strides.data(), 0, ACL_FORMAT_ND, x_shape.data(), 3, x_dev);
    aclTensor *w_t = aclCreateTensor(w_shape.data(), 2, ACL_FLOAT, w_strides.data(), 0, ACL_FORMAT_ND, w_shape.data(), 2, w_dev);
    aclTensor *b_t = aclCreateTensor(b_shape.data(), 1, ACL_FLOAT, b_strides.data(), 0, ACL_FORMAT_ND, b_shape.data(), 1, b_dev);
    aclTensor *s_t = aclCreateTensor(s_shape.data(), 1, ACL_FLOAT, s_strides.data(), 0, ACL_FORMAT_ND, s_shape.data(), 1, s_dev);
    aclTensor *y_t = aclCreateTensor(y_shape.data(), 3, x_dt, y_strides.data(), 0, ACL_FORMAT_ND, y_shape.data(), 3, y_dev);

    uint64_t ws_size = 0;
    aclOpExecutor *exec = nullptr;
    CHECK_ACL(aclnnMhcHeadCollapseGetWorkspaceSize(x_t, w_t, b_t, s_t, 1e-6, 1e-6, y_t, &ws_size, &exec));
    void *ws = nullptr;
    if (ws_size) CHECK_ACL(aclrtMalloc(&ws, ws_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclnnMhcHeadCollapse(ws, ws_size, exec, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));
    // DUMP: read full y, print first 8 of each row
    {
        std::vector<unsigned char> yall(y_bytes);
        CHECK_ACL(aclrtMemcpy(yall.data(), y_bytes, y_dev, y_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
        for (int64_t r = 0; r < kOuter; ++r) {
            std::fprintf(stderr, "DUMP_R%d:", (int)r);
            for (int i = 0; i < 8; ++i) {
                if (use_fp16) {
                    uint16_t bits; std::memcpy(&bits, &yall[(r*kH+i)*2], 2);
                    __fp16 h; std::memcpy(&h, &bits, 2);
                    std::fprintf(stderr, " %.5f", (float)h);
                } else {
                    float f; std::memcpy(&f, &yall[(r*kH+i)*4], 4);
                    std::fprintf(stderr, " %.5f", f);
                }
            }
            std::fprintf(stderr, "NEWLINE_TOKEN");
        }
    }

    CHECK_ACL(aclrtMemcpy(y_raw.data(), y_bytes, y_dev, y_bytes, ACL_MEMCPY_DEVICE_TO_HOST));

    // CPU reference in float (input quantized to fp16 when use_fp16)
    double max_err = 0.0;
    for (int64_t row = 0; row < kOuter; ++row) {
        double sum_sq = 0.0;
        for (int64_t k = 0; k < kNH; ++k) {
            float xv = use_fp16 ? HalfBitsToFloat(x_h[row * kNH + k]) : x_f[row * kNH + k];
            sum_sq += static_cast<double>(xv) * xv;
        }
        const float rms_inv = static_cast<float>(1.0 / std::sqrt(sum_sq / kNH + 1e-6));
        float gates[16];
        for (int64_t i = 0; i < kN; ++i) {
            double logit = 0.0;
            for (int64_t k = 0; k < kNH; ++k) {
                float xv = use_fp16 ? HalfBitsToFloat(x_h[row * kNH + k]) : x_f[row * kNH + k];
                logit += static_cast<double>(weight_f[i * kNH + k]) * xv;
            }
            const float z = static_cast<float>(logit) * rms_inv * scale_f + base_f[i];
            gates[i] = 1.0f / (1.0f + std::exp(-z)) + 1e-6f;
        }
        for (int64_t d = 0; d < kH; ++d) {
            float acc = 0.0f;
            for (int64_t i = 0; i < kN; ++i) {
                const int64_t xoff = row * kNH + i * kH + d;
                const float xv = use_fp16 ? HalfBitsToFloat(x_h[xoff]) : x_f[xoff];
                acc += gates[i] * xv;
            }
            float got = 0.0f;
            if (use_fp16) {
                uint16_t bits;
                std::memcpy(&bits, &y_raw[(row * kH + d) * 2], 2);
                got = HalfBitsToFloat(bits);
            } else {
                std::memcpy(&got, &y_raw[(row * kH + d) * 4], 4);
            }
            const double err = std::fabs(static_cast<double>(acc) - got);
            if (err > max_err) max_err = err;
        }
    }

    const double tol = use_fp16 ? 2e-3 : 1e-5;
    std::printf("n=%lld h=%lld outer=%lld %s max_err=%.3e %s\n",
                (long long)kN, (long long)kH, (long long)kOuter, argv[4],
                max_err, max_err <= tol ? "PASS" : "FAIL");
    return max_err <= tol ? 0 : 2;
}
