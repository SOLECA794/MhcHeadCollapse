---
name: 我最想要的-是通过本地和oj数据的对拍-测出真实的case形状-这也是交接的下一步工作和换到这个
description: 我最想要的 是通过本地和oj数据的对拍 测出真实的case形状 这也是交接的下一步工作和换到这个有NPU的环境的原因，你先把notebook的环境配置相关的信息直接固化到.rivet/进行记录 防止下次再踩坑 — verified by 14 checks
triggers: ['ENVIRONMENT', '.rivet', 'null', '我最想要的']
---

# 我最想要的-是通过本地和oj数据的对拍-测出真实的case形状-这也是交接的下一步工作和换到这个

> 自动从会话 83642d4c 蒸馏的草稿。审核后用 `/skill approve 我最想要的-是通过本地和oj数据的对拍-测出真实的case形状-这也是交接的下一步工作和换到这个` 入库，或 `/skill reject 我最想要的-是通过本地和oj数据的对拍-测出真实的case形状-这也是交接的下一步工作和换到这个` 丢弃。

## Steps
1. 阅读 / 搜索：echo "=== 1. NPU 设备 ==="
ls /dev/davinci* 2>/dev/n、# 读完整输出
cat /tmp/rivet-raw/146d7fa93a68534eb8377bf、/tmp/rivet-raw/146d7fa93a68534eb8377bfa.raw
2. 修改：/workspace/notebook13/MhcHeadCollapse/.rivet/ENVIRONMENT.md
3. 操作：memory
4. 修改：/workspace/notebook13/MhcHeadCollapse/.rivet.md
5. 阅读 / 搜索：grep -n "LD_LIBRARY_PATH\|/opt/conda\|/usr/local/A、find . -type f | head -20 && echo "=== 根 CMakeList、/tmp/rivet-raw/4949598754b764af27f0435a.raw
6. 验证：echo test && /bin/echo PATH=$PATH

## Verified by
- cd /workspace/notebook13/MhcHeadCollapse/.rivet/scratch/v117_stage/code && \
export ASCEND_HOME_PATH=/opt/conda/Ascend/cann-9.0.0 && \
export ASCEND_TOOLKIT_HOME=/opt/conda/Ascend/cann-9.0.0 && \
expo (passed 0)
- cd /workspace/notebook13/MhcHeadCollapse/.rivet/scratch/v117_stage/code/build && \
export ASCEND_HOME_PATH=/opt/conda/Ascend/cann-9.0.0 && \
export ASCEND_TOOLKIT_HOME=/opt/conda/Ascend/cann-9.0.0 &&  (passed 0)
- head -40 /workspace/notebook13/MhcHeadCollapse/.rivet/scratch/v117_stage/tests/mhc_correctness.cpp; echo "==="; grep -n "include\|extern" /workspace/notebook13/MhcHeadCollapse/.rivet/scratch/v117_stag (passed 0)
- B=/workspace/notebook13/MhcHeadCollapse/.rivet/scratch/v117_stage/code/build
find $B -name "aclnn_mhc_head_collapse.h" 2>/dev/null | head -3
echo "=== 库产物 ==="
find $B -name "libcust_op*.so" 2>/dev/nu (passed 0)
- S=/workspace/notebook13/MhcHeadCollapse/.rivet/scratch/v117_stage
export ASCEND_HOME_PATH=/opt/conda/Ascend/cann-9.0.0
export LD_LIBRARY_PATH=/opt/conda/Ascend/cann-9.0.0/aarch64-linux/lib64:/usr/loca (passed 0)

<!-- skill-draft-key: ac7341674779 -->
<!-- source-session: 83642d4c -->
