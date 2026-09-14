// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MhcHeadCollapseTilingData {
    uint32_t outer;
    uint32_t n;
    uint32_t h;
    uint32_t nH;
    uint32_t block_dim;
    uint32_t path;
    float inv_nH;
    float eps_norm;
    float eps_hc;
};
