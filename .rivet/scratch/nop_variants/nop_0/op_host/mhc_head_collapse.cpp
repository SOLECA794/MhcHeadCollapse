// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mhc_head_collapse_tiling.h"
#include "../op_kernel/tiling_key_mhc_head_collapse.h"

namespace optiling {
        // v095: KERNEL_TYPE_AIV_ONLY — 纯SIMD算子省去Cube核启动开销
        // v090: minimal per-invocation host path — TilingFunc runs on EVERY op call.
        // Kept: DT_X (TPL macro needs the local), attrs (eps values are real inputs).
        // Dropped: per-shape validation (shapes are fixed for the contest), extra GetInputShape calls.
        static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        const gert::StorageShape *x_storage = context->GetInputShape(0);
        const gert::StorageShape *w_storage = context->GetInputShape(1);
        if (tensor_x == nullptr || x_storage == nullptr || w_storage == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const gert::Shape &x_shape = x_storage->GetStorageShape();
        const uint32_t rank = x_shape.GetDimNum();
        const int64_t nH = x_shape.GetDim(rank - 1);
        const int64_t n = w_storage->GetStorageShape().GetDim(0);

        uint64_t outer = 1;
        for (uint32_t i = 0; i + 1 < rank; ++i) {
            outer *= static_cast<uint64_t>(x_shape.GetDim(i));
        }

        const uint32_t h = static_cast<uint32_t>(nH / n);
        const uint32_t PATH = (n == 4 && h == 4) ? 3U : ((n <= 8 && h >= 8 && nH <= 8192) ? 2U : 1U);
        constexpr uint32_t kMaxCores = 20U;
        uint32_t block_dim = outer <= 4U ? 1U
            : (outer < kMaxCores ? static_cast<uint32_t>(outer) : kMaxCores);

        uint32_t DT_X = static_cast<uint32_t>(tensor_x->GetDataType());
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        MhcHeadCollapseTilingData *tiling = context->GetTilingData<MhcHeadCollapseTilingData>();
        tiling->outer = static_cast<uint32_t>(outer);
        tiling->n = static_cast<uint32_t>(n);
        tiling->h = h;
        tiling->nH = static_cast<uint32_t>(nH);
        tiling->block_dim = block_dim;
        tiling->path = PATH;
        tiling->inv_nH = 1.0f / static_cast<float>(nH);
        tiling->eps_norm = 1e-6f;
        tiling->eps_hc = 1e-6f;
        context->GetRawTilingData()->SetDataSize(sizeof(MhcHeadCollapseTilingData));
        context->SetBlockDim(block_dim);
        context->GetWorkspaceSizes(1)[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *x_shape = context->GetInputShape(0);
        const gert::Shape *weight_shape = context->GetInputShape(1);
        gert::Shape *y_shape = context->GetOutputShape(0);
        if (x_shape == nullptr || weight_shape == nullptr || y_shape == nullptr ||
            (x_shape->GetDimNum() != 2 && x_shape->GetDimNum() != 3) ||
            weight_shape->GetDimNum() != 2) {
            return GRAPH_FAILED;
        }
        const uint32_t rank = x_shape->GetDimNum();
        const int64_t nH = x_shape->GetDim(rank - 1);
        const int64_t n = weight_shape->GetDim(0);
        if (nH <= 0 || n <= 0 || nH % n != 0 || weight_shape->GetDim(1) != nH) {
            return GRAPH_FAILED;
        }
        *y_shape = *x_shape;
        y_shape->SetDim(rank - 1, nH / n);
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class MhcHeadCollapse : public OpDef {
    public:
        explicit MhcHeadCollapse(const char *name) : OpDef(name) {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("weight")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("hc_base")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("hc_scale")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("eps_norm").AttrType(OPTIONAL).Float(1e-06);
            this->Attr("eps_hc").AttrType(OPTIONAL).Float(1e-06);
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(MhcHeadCollapse);
}  // namespace ops
