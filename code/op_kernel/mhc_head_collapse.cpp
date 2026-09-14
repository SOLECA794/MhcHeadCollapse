// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mhc_head_collapse_tiling.h"
#include "tiling_key_mhc_head_collapse.h"

template <class DT_X>
class KernelMhcHeadCollapse {
public:
    __aicore__ inline KernelMhcHeadCollapse() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR hc_base, GM_ADDR hc_scale, GM_ADDR y, uint32_t length) {

    }
    __aicore__ inline void Process() {

    }
private:

};

template <typename DT_X>
 __global__ __aicore__ void mhc_head_collapse(GM_ADDR x, GM_ADDR weight, GM_ADDR hc_base, GM_ADDR hc_scale, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MhcHeadCollapseTilingData);
    GET_TILING_DATA_WITH_STRUCT(MhcHeadCollapseTilingData, tiling_data, tiling);
    KernelMhcHeadCollapse<DT_X> op;
    op.Init(x, weight, hc_base, hc_scale, y, tiling_data.length);
    op.Process();
}
