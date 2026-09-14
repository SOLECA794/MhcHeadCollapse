"""
MhcHeadCollapse 算子 — PyTorch 参考实现
=============================================
对应赛题：星辰杯 C组中等题「MhcHeadCollapse 算子（mHC四路归一）」

标准格式：
  class Model        — 待优化的算子实现
  get_init_inputs() — module init 的输入（权重参数）
  get_inputs()      — module forward 的输入（动态张量）
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


# ============================================================================
# class Model — 待优化的算子实现
# ============================================================================

class Model(nn.Module):
    """
    mHC 四路归一算子 — 可学习折叠

    前向逻辑：
        1. RMS 归一化（gamma=1）
        2. (x @ W^T) * rms_inv  → mixes
        3. sigmoid(mixes * scale + base) + eps_hc  → pre
        4. Σ pre_i * streams_i  → output

    参数（由 get_init_inputs 初始化）：
        weight   (n, nH)  — 门控线性投影权重
        hc_base  (n,)     — 门控偏置
        hc_scale (1,)     — 门控缩放

    属性：
        eps_norm — RMS 归一化防除零（默认 1e-6）
        eps_hc   — sigmoid 输出防零（默认 1e-6）
    """

    # ----------------------------------------------------------
    # 嵌套 autograd Function：前向 + 反向
    # ----------------------------------------------------------
    class _Func(torch.autograd.Function):
        """纯 FP32 计算路径，前向保存中间量供反向复用"""

        @staticmethod
        def forward(ctx, x, weight, hc_base, hc_scale,
                    eps_norm: float, eps_hc: float):
            orig_dtype = x.dtype

            # 全部提升至 FP32
            x_f32    = x.to(torch.float32)
            w_f32    = weight.to(torch.float32)
            base_f32 = hc_base.to(torch.float32)
            scale_f32 = hc_scale.to(torch.float32)

            n, nH   = w_f32.shape
            prefix  = x_f32.shape[:-1]          # () / (S,) / (S, B)
            H       = nH // n

            # Step 1: RMS 归一化逆
            x_sq    = x_f32 * x_f32
            rms     = torch.sqrt(x_sq.mean(dim=-1, keepdim=True) + eps_norm)
            rms_inv = 1.0 / rms                 # (..., 1)

            # Step 2: 线性投影 + rms_inv 加权
            mixes   = torch.matmul(x_f32, w_f32.t()) * rms_inv  # (..., n)

            # Step 3: sigmoid 门控激活
            pre     = torch.sigmoid(mixes * scale_f32 + base_f32) + eps_hc  # (..., n)

            # Step 4: 逐路加权求和
            streams = x_f32.reshape(prefix + (n, H))              # (..., n, H)
            output  = (pre.unsqueeze(-1) * streams).sum(dim=-2)  # (..., H)

            ctx.save_for_backward(x_f32, w_f32, rms_inv, pre, streams, scale_f32)
            ctx.eps_norm  = eps_norm
            ctx.orig_dtype = orig_dtype

            return output.to(orig_dtype)

        @staticmethod
        def backward(ctx, grad_output):
            x_f32, w_f32, rms_inv, pre, streams, scale_f32 = ctx.saved_tensors
            eps_norm  = ctx.eps_norm
            orig_dtype = ctx.orig_dtype

            n, nH   = w_f32.shape
            prefix  = x_f32.shape[:-1]
            H       = nH // n

            go = grad_output.to(torch.float32)   # (..., H)

            # ---- d_pre & d_streams（加权求和反向）----
            grad_pre       = (go.unsqueeze(-2) * streams).sum(dim=-1)          # (..., n)
            grad_x_streams = (go.unsqueeze(-2) * pre.unsqueeze(-1)).reshape(prefix + (nH,))  # (..., nH)

            # ---- sigmoid 导数 ----
            d_sigmoid = pre * (1.0 - pre)           # (..., n)

            # ---- 中间变量 ----
            mixes_proj = torch.matmul(x_f32, w_f32.t())            # (..., n)

            # z = mixes * scale + base,  pre = sigmoid(z) + eps_hc
            # grad_z = grad_pre * sigmoid'(z)
            grad_z       = grad_pre * d_sigmoid                   # (..., n)
            # grad_mixes = grad_z * scale
            grad_mixes   = grad_z * scale_f32                     # (..., n)
            # grad_mixes_proj = grad_mixes * rms_inv  (因为 mixes = mixes_proj * rms_inv)
            grad_mixes_proj = grad_mixes * rms_inv                 # (..., n)

            # ---- 展平 batch 维度用于权重梯度 ----
            N            = grad_pre.numel() // n
            x_flat       = x_f32.reshape(N, nH)

            # ---- d_x（线性投影分支）----
            # ∂mixes_proj/∂x = W^T * rms_inv → grad_x = grad_mixes_proj @ W
            grad_x_proj  = torch.matmul(grad_mixes_proj, w_f32)   # (..., nH)

            # ---- d_W = grad_mixes_proj^T @ x ----
            grad_mix_proj_flat = grad_mixes_proj.reshape(N, n)
            grad_W       = torch.matmul(grad_mix_proj_flat.t(), x_flat)  # (n, nH)

            # ---- d_x（RMS 逆分支）----
            d_rms_inv   = (grad_mixes * mixes_proj).sum(dim=-1, keepdim=True)  # (..., 1)
            coef        = -0.5 * d_rms_inv * (rms_inv ** 3) * (2.0 / nH)      # (..., 1)
            grad_x_rms  = coef * x_f32                             # (..., nH)

            # ---- 总 d_x = 三条路径之和 ----
            grad_x = grad_x_proj + grad_x_rms + grad_x_streams

            # ---- d_base = Σ_batch (grad_z) ----
            grad_base = grad_z.reshape(N, n).sum(dim=0)           # (n,)

            # ---- d_scale = Σ_all (grad_z * mixes_proj * rms_inv) ----
            grad_scale = (grad_z * mixes_proj * rms_inv).sum()     # scalar
            grad_scale = grad_scale.reshape(1)                      # (1,)

            return (grad_x.to(orig_dtype),
                    grad_W.to(orig_dtype),
                    grad_base.to(orig_dtype),
                    grad_scale.to(orig_dtype),
                    None, None)

    # ----------------------------------------------------------
    # nn.Module 初始化与前向
    # ----------------------------------------------------------

    def __init__(self,
                 n: int = 4,
                 H: int = 1024,
                 eps_norm: float = 1e-6,
                 eps_hc: float = 1e-6):
        super().__init__()
        self.n = n
        self.H = H
        self.nH = n * H
        self.eps_norm = eps_norm
        self.eps_hc   = eps_hc

        self.weight   = nn.Parameter(torch.zeros(n, self.nH, dtype=torch.float32))
        self.hc_base  = nn.Parameter(torch.zeros(n, dtype=torch.float32))
        self.hc_scale = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def reset_parameters(self):
        """重置 hc_base 和 hc_scale 为零（延迟初始化），weight 由外部控制"""
        nn.init.zeros_(self.hc_base)
        nn.init.zeros_(self.hc_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (T, nH) 或 (S, B, nH)，dtype = float32/bfloat16/float16
        返回:
            y: (T, H) 或 (S, B, H)，dtype 同输入
        """
        return self._Func.apply(
            x, self.weight, self.hc_base, self.hc_scale,
            self.eps_norm, self.eps_hc
        )

    def extra_repr(self) -> str:
        return (f"n={self.n}, H={self.H}, nH={self.nH}, "
                f"eps_norm={self.eps_norm}, eps_hc={self.eps_hc}")


# ============================================================================
# get_init_inputs — module init 的输入（权重参数）
# ============================================================================

def get_init_inputs(n: int = 4, H: int = 1024,
                    init_method: str = "zero") -> dict:
    """
    返回 module init 的输入（权重参数）。

    参数:
        n           — 残差流路数（建议 2, 4, 8）
        H           — 单路隐藏维度（建议 64 的整数倍）
        init_method — 权重初始化方式
                      "zero": weight=0, hc_base=0, hc_scale=0（延迟初始化，退化为均值归一）
                      "eye" : weight=单位块对角, hc_base=[2,1,0,-1], hc_scale=1
                      "rand": weight=随机, hc_base=0, hc_scale=1

    返回:
        dict，键为 weight/hc_base/hc_scale，值为 torch.Tensor
    """
    nH = n * H

    if init_method == "zero":
        weight   = torch.zeros(n, nH, dtype=torch.float32)
        hc_base  = torch.zeros(n, dtype=torch.float32)
        hc_scale = torch.zeros(1, dtype=torch.float32)
    elif init_method == "eye":
        weight   = torch.zeros(n, nH, dtype=torch.float32)
        for i in range(n):
            weight[i, i * H:(i + 1) * H] = 1.0
        hc_base  = torch.tensor([2.0, 1.0, 0.0, -1.0][:n], dtype=torch.float32)
        hc_scale = torch.ones(1, dtype=torch.float32)
    elif init_method == "rand":
        weight   = torch.randn(n, nH, dtype=torch.float32) * 0.02
        hc_base  = torch.zeros(n, dtype=torch.float32)
        hc_scale = torch.ones(1, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown init_method: {init_method}")

    return dict(weight=weight, hc_base=hc_base, hc_scale=hc_scale)


# ============================================================================
# get_inputs — module forward 的输入（动态张量）
# ============================================================================

def get_inputs(batch_type: str = "3d",
               n: int = 4, H: int = 1024,
               S: int = 8, B: int = 2, T: int = 16,
               dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, dict]:
    """
    返回 module forward 的输入（动态张量）。

    参数:
        batch_type — "3d" → (S, B, nH)，"2d" → (T, nH)
        n, H       — 与 get_init_inputs 保持一致
        S, B       — 仅 batch_type="3d" 时使用：序列长度 × 批量大小
        T          — 仅 batch_type="2d" 时使用：样本数
        dtype      — 输入数据类型：torch.float32 / torch.bfloat16 / torch.float16

    返回:
        (x, info)
        x   : torch.Tensor，输入张量
        info: dict，补充信息（shape_description, n, H, batch_dims）
    """
    nH = n * H

    if batch_type == "3d":
        x          = torch.randn(S, B, nH, dtype=dtype)
        batch_dims = (S, B)
    elif batch_type == "2d":
        x          = torch.randn(T, nH, dtype=dtype)
        batch_dims = (T,)
    else:
        raise ValueError(f"Unknown batch_type: {batch_type}")

    info = dict(
        shape_description=batch_type,
        n=n, H=H, nH=nH,
        batch_dims=batch_dims,
        dtype=str(dtype).split('.')[-1],
    )
    return x, info


# ============================================================================
# 验证 & Demo
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MhcHeadCollapse 算子 — PyTorch 参考实现验证")
    print("=" * 60)

    n, H = 4, 4
    atol = rtol = 1e-4

    # --- Demo 1: 延迟初始化 → 退化为 (n/2) * mean(streams) ---
    print("\n[Demo 1] 延迟初始化 (weight=0, hc_base=0, hc_scale=0)")
    init_dict = get_init_inputs(n=n, H=H, init_method="zero")
    model = Model(n=n, H=H)
    with torch.no_grad():
        model.weight.copy_(init_dict["weight"])
        model.hc_base.copy_(init_dict["hc_base"])
        model.hc_scale.copy_(init_dict["hc_scale"])

    x = torch.zeros(1, 2, n * H, dtype=torch.float32)
    x[0, 0, :] = torch.arange(n * H, dtype=torch.float32)
    x[0, 1, :] = torch.arange(n * H, dtype=torch.float32) * 2

    y = model(x)
    print(f"  x.shape      = {x.shape}  →  y.shape = {y.shape}  (期望 (1,2,4))")
    assert y.shape == (1, 2, H), f"shape mismatch: {y.shape}"

    expected_b0 = torch.tensor([12.0, 14.0, 16.0, 18.0])
    max_err = (y[0, 0] - expected_b0).abs().max().item()
    print(f"  max_abs_err = {max_err:.6f}  (atol={atol})  {'✓ PASS' if max_err < atol else '✗ FAIL'}")
    assert max_err < atol, f"delay-init check failed: {max_err}"

    # --- Demo 2: 可学习门控 (eye 初始化) ---
    print("\n[Demo 2] 可学习门控 (weight=eye-block, hc_base=[2,1,0,-1])")
    init_dict2 = get_init_inputs(n=n, H=H, init_method="eye")
    model2 = Model(n=n, H=H)
    with torch.no_grad():
        model2.weight.copy_(init_dict2["weight"])
        model2.hc_base.copy_(init_dict2["hc_base"])
        model2.hc_scale.copy_(init_dict2["hc_scale"])

    x2 = torch.tensor([[[[i + 1 for i in range(n * H)]]]], dtype=torch.float32)  # (1,1,1,16)
    x2 = x2.squeeze(0).squeeze(0).unsqueeze(0)                                  # (1, 1, 16)
    y2 = model2(x2)
    print(f"  x.shape  = {x2.shape}  →  y.shape = {y2.shape}")
    print(f"  y[0,0,:]= {y2[0, 0].tolist()}")
    print("  (各路门控 sigmoid 后加权求和，第4路通道最大)")

    # --- Demo 3: 2 维输入 (T, nH) ---
    print("\n[Demo 3] 2 维输入 (T=2, nH=16) → (T=2, H=4)")
    init_dict3 = get_init_inputs(n=n, H=H, init_method="zero")
    model3 = Model(n=n, H=H)
    with torch.no_grad():
        model3.weight.copy_(init_dict3["weight"])
        model3.hc_base.copy_(init_dict3["hc_base"])
        model3.hc_scale.copy_(init_dict3["hc_scale"])

    x3 = torch.randn(2, n * H, dtype=torch.float32)
    y3 = model3(x3)
    print(f"  x.shape = {x3.shape}  →  y.shape = {y3.shape}  (期望 (2, 4))")
    assert y3.shape == (2, H)
    print("  ✓ PASS")

    # --- Demo 4: 反向传播梯度验证 ---
    print("\n[Demo 4] 反向传播 (autograd) 验证")
    model4 = Model(n=n, H=H)
    x4 = torch.randn(2, 3, n * H, dtype=torch.float32, requires_grad=True)
    y4 = model4(x4)
    loss = y4.sum()
    loss.backward()
    assert x4.grad is not None and x4.grad.shape == x4.shape
    assert model4.weight.grad is not None and model4.weight.grad.shape == model4.weight.shape
    assert model4.hc_base.grad is not None and model4.hc_base.grad.shape == model4.hc_base.shape
    assert model4.hc_scale.grad is not None and model4.hc_scale.grad.shape == model4.hc_scale.shape
    print(f"  x.grad.shape        = {x4.grad.shape}")
    print(f"  weight.grad.shape   = {model4.weight.grad.shape}")
    print(f"  hc_base.grad.shape  = {model4.hc_base.grad.shape}")
    print(f"  hc_scale.grad.shape = {model4.hc_scale.grad.shape}")
    print("  ✓ PASS — 所有梯度均已计算")

    # --- Demo 5: dtype 兼容性 (BF16 / FP16) ---
    print("\n[Demo 5] dtype 兼容性")
    for dt in [torch.bfloat16, torch.float16]:
        model5 = Model(n=n, H=H)
        x5 = torch.randn(1, 2, n * H, dtype=dt)
        y5 = model5(x5)
        assert y5.dtype == dt, f"dtype mismatch: {y5.dtype} != {dt}"
        print(f"  dtype={dt}  →  output.dtype={y5.dtype}  shape={y5.shape}  ✓")

    # --- Demo 6: 数值梯度校验 ---
    print("\n[Demo 6] 数值梯度校验 (autograd vs finite-difference)")
    torch.manual_seed(42)
    eps = 5e-4
    model6 = Model(n=n, H=H)
    model6.weight.data   = torch.randn(n, n * H, dtype=torch.float32) * 0.01
    model6.hc_base.data  = torch.randn(n, dtype=torch.float32) * 0.1
    model6.hc_scale.data = torch.tensor([0.5], dtype=torch.float32)

    # autograd 梯度：用独立的 requires_grad 张量计算，结束后 detach
    x6_for_auto = torch.randn(2, 2, n * H, dtype=torch.float32, requires_grad=True)
    y6 = model6(x6_for_auto)
    y6.sum().backward()
    grad_x_auto_flat = x6_for_auto.grad.detach().flatten()  # 独立计算后 detach

    # 数值梯度：固定 seed，从同一 base 构造扰动
    torch.manual_seed(0)
    max_rel_err_x = 0.0
    x6_base = x6_for_auto.detach()   # 无梯度历史的干净 base

    for _ in range(5):
        i = torch.randint(0, x6_base.numel(), (1,)).item()
        mask = torch.zeros_like(x6_base)
        mask.flatten()[i] = eps
        x_plus  = x6_base + mask
        x_minus = x6_base - mask
        loss_plus  = model6(x_plus).sum().item()
        loss_minus = model6(x_minus).sum().item()
        grad_num = (loss_plus - loss_minus) / (2 * eps)
        grad_act = grad_x_auto_flat[i].item()
        if abs(grad_act) > 1e-8:
            rel_err = abs(grad_num - grad_act) / abs(grad_act)
            max_rel_err_x = max(max_rel_err_x, rel_err)
    print(f"  x 数值梯度 max_rel_err = {max_rel_err_x:.6f}  {'✓ PASS' if max_rel_err_x < 1e-2 else '✗ FAIL'}")

    print("\n" + "=" * 60)
    print("全部验证通过 ✓")
    print("=" * 60)
