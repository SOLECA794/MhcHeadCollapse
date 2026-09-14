#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(MhcHeadCollapse)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(weight, ge::TensorType::ALL())
    .INPUT(hc_base, ge::TensorType::ALL())
    .INPUT(hc_scale, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .ATTR(eps_norm, Float, 1e-06)
    .ATTR(eps_hc, Float, 1e-06)
    .OP_END_FACTORY_REG(MhcHeadCollapse);

}

#endif
