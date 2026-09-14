// mhc_bench_percall.cpp — 还原 OJ 计时口径的 bench
//
// 背景（FACTS.md F2.1d）：
//   现有 mhc_bench.cpp 测的是 1000 次连续 launch 的均摊时间（stream 满载），
//   不含 launch 开销。实测它与 OJ 存在系统性差异（Case5: 4.48 vs 7.74, 差 3.26us）。
//
// 本 bench 改为 per-call 计时：每次 launch 后同步，测端到端单次耗时。
// 这是为了还原 OJ 的计时口径，从而在本地做精确调优。
//
// 用法: ./mhc_bench_percall <n> <h> <outer> <dtype>
#include <acl/acl.h>
#include <aclnn_mhc_head_collapse.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#define CHECK_ACL(x)                                                          \
  do {                                                                        \
    aclError _r = (x);                                                        \
    if (_r != ACL_SUCCESS) {                                                  \
      std::printf("ACL FAIL %s rc=%d\n", #x, (int)_r);                        \
      return 1;                                                               \
    }                                                                         \
  } while (0)

static double now_us() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

int main(int argc, char **argv) {
  if (argc < 5) {
    std::printf("usage: %s <n> <h> <outer> <dtype>\n", argv[0]);
    return 2;
  }
  const int64_t kN = atoll(argv[1]);
  const int64_t kH = atoll(argv[2]);
  const int64_t kOuter = atoll(argv[3]);
  const std::string dtype = argv[4];
  const bool useFp16 = (dtype == "fp16");
  const int64_t kNH = kN * kH;

  const size_t xBytes = (size_t)kOuter * kNH * (useFp16 ? 2 : 4);
  const size_t wBytes = (size_t)kN * kNH * 4;   // weight 恒 fp32
  const size_t bBytes = (size_t)kN * 4;         // base fp32
  const size_t sBytes = 4;                      // scale fp32
  const size_t yBytes = (size_t)kOuter * kH * 2;

  CHECK_ACL(aclInit(nullptr));
  CHECK_ACL(aclrtSetDevice(0));
  aclrtStream stream = nullptr;
  CHECK_ACL(aclrtCreateStream(&stream));

  void *xDev = nullptr, *wDev = nullptr, *bDev = nullptr, *sDev = nullptr,
       *yDev = nullptr;
  CHECK_ACL(aclrtMalloc(&xDev, xBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc(&wDev, wBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc(&bDev, bBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc(&sDev, sBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc(&yDev, yBytes, ACL_MEM_MALLOC_HUGE_FIRST));

  std::vector<char> host(xBytes + wBytes + bBytes + sBytes, 0);
  CHECK_ACL(aclrtMemcpy(xDev, xBytes, host.data(), xBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(wDev, wBytes, host.data(), wBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(bDev, bBytes, host.data(), bBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(sDev, sBytes, host.data(), sBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));

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
  const aclDataType xDt = useFp16 ? ACL_FLOAT16 : ACL_FLOAT;

  aclTensor *x_t = aclCreateTensor(x_shape.data(), 3, xDt, x_strides.data(), 0,
                                   ACL_FORMAT_ND, x_shape.data(), 3, xDev);
  aclTensor *w_t = aclCreateTensor(w_shape.data(), 2, ACL_FLOAT,
                                   w_strides.data(), 0, ACL_FORMAT_ND,
                                   w_shape.data(), 2, wDev);
  aclTensor *b_t = aclCreateTensor(b_shape.data(), 1, ACL_FLOAT,
                                   b_strides.data(), 0, ACL_FORMAT_ND,
                                   b_shape.data(), 1, bDev);
  aclTensor *s_t = aclCreateTensor(s_shape.data(), 1, ACL_FLOAT,
                                   s_strides.data(), 0, ACL_FORMAT_ND,
                                   s_shape.data(), 1, sDev);
  aclTensor *y_t = aclCreateTensor(y_shape.data(), 3, xDt, y_strides.data(), 0,
                                   ACL_FORMAT_ND, y_shape.data(), 3, yDev);

  size_t wsSize = 0;
  aclOpExecutor *exec = nullptr;
  CHECK_ACL(aclnnMhcHeadCollapseGetWorkspaceSize(x_t, w_t, b_t, s_t, 1e-6, 1e-6,
                                                 y_t, &wsSize, &exec));
  void *wsDev = nullptr;
  if (wsSize > 0) {
    CHECK_ACL(aclrtMalloc(&wsDev, wsSize, ACL_MEM_MALLOC_HUGE_FIRST));
  }

  // warmup
  for (int i = 0; i < 20; ++i) {
    CHECK_ACL(aclnnMhcHeadCollapse(wsDev, wsSize, exec, stream));
  }
  CHECK_ACL(aclrtSynchronizeStream(stream));

  // ---- 模式 A: per-call 同步（端到端，含 launch 开销）----
  const int kIters = 200;
  double t0 = now_us();
  for (int i = 0; i < kIters; ++i) {
    CHECK_ACL(aclnnMhcHeadCollapse(wsDev, wsSize, exec, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));
  }
  double t1 = now_us();
  const double perCall = (t1 - t0) / kIters;
  // ---- 模式 B: 1000 次连发（旧 bench 口径，流式均摊）----
  aclrtEvent ev0, ev1;
  CHECK_ACL(aclrtCreateEvent(&ev0));
  CHECK_ACL(aclrtCreateEvent(&ev1));
  const int kIters2 = 1000;
  CHECK_ACL(aclrtRecordEvent(ev0, stream));
  for (int i = 0; i < kIters2; ++i) {
    CHECK_ACL(aclnnMhcHeadCollapse(wsDev, wsSize, exec, stream));
  }
  CHECK_ACL(aclrtRecordEvent(ev1, stream));
  CHECK_ACL(aclrtSynchronizeEvent(ev1));
  float ms = 0.0f;
  CHECK_ACL(aclrtEventElapsedTime(&ms, ev0, ev1));
  const double streamed = ms * 1000.0f / kIters2;

  std::printf(
      "PERCALL n=%lld h=%lld outer=%lld %s  per_call_us=%.3f  streamed_us=%.3f  "
      "launch_overhead_us=%.3f\n",
      (long long)kN, (long long)kH, (long long)kOuter, dtype.c_str(), perCall,
      streamed, perCall - streamed);

  if (wsDev) aclrtFree(wsDev);
  aclrtFree(xDev); aclrtFree(wDev); aclrtFree(bDev); aclrtFree(sDev);
  aclrtFree(yDev);
  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();
  return 0;
}
