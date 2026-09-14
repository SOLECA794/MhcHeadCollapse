---
name: 1-把那个打印脚本里的ms替换成us-平台显示的是微妙-这里搞错了
description: 1、把那个打印脚本里的ms替换成us，平台显示的是微妙 这里搞错了 — verified by 1 check
triggers: ['cannjudge_cli', 'probe-ws-auth', 'npu_console', 'browserless-route']
---

# 1-把那个打印脚本里的ms替换成us-平台显示的是微妙-这里搞错了

> 自动从会话 4f88bf50 蒸馏的草稿。审核后用 `/skill approve 1-把那个打印脚本里的ms替换成us-平台显示的是微妙-这里搞错了` 入库，或 `/skill reject 1-把那个打印脚本里的ms替换成us-平台显示的是微妙-这里搞错了` 丢弃。

## Steps
1. 阅读 / 搜索：S="d:/Desktop/OP-Learning/Ascend/cann-learing-hub/、N="d:/Desktop/OP-Learning/Ascend/ascend-npu-consol
2. 修改：d:\Desktop\OP-Learning\Ascend\cann-learing-hub\skills\cannjudge-submit-plaintext\cannjudge_cli.py
3. 阅读 / 搜索：P="d:/Desktop/OP-Learning/Ascend/cann-learing-hub/、d:\Desktop\OP-Learning\Ascend\ascend-npu-console\scripts\notebook.js、N="d:/Desktop/OP-Learning/Ascend/ascend-npu-consol
4. 修改：d:\Desktop\OP-Learning\.rivet\scratch\probe-ws-auth.py
5. 阅读 / 搜索：timeout 60 python "d:/Desktop/OP-Learning/.rivet/s、C:\Users\34511\AppData\Local\Temp\rivet-raw\04bcdbb1354d531800cfdb64.raw
6. 修改：d:\Desktop\OP-Learning\Ascend\ascend-npu-console\scripts\npu_console.py
7. 阅读 / 搜索：N="d:/Desktop/OP-Learning/Ascend/ascend-npu-consol、echo "=== 探查: 是否有 API 能列出用户的 Notebook 实例 ==="
for、echo "=== 关键验证: 用户能否手动提供 instanceId（最低成本路径） ==="
e
8. 修改：d:\Desktop\OP-Learning\Ascend\ascend-npu-console\docs\browserless-route.md
9. 阅读 / 搜索：S="d:/Desktop/OP-Learning/Ascend"
echo "════════ 交

## Verified by
- echo "=== 测试 REST 端点是否真免 cookie(关键假设) ==="
timeout 25 python -c "
import requests
# 用一个假 instanceId 试，看返回什么(区分'认证拒绝' vs '实例不存在')
for iid in ['test','nonexistent']:
    try:
        r = requests.get(f' (passed 0)

<!-- skill-draft-key: 730529343211 -->
<!-- source-session: 4f88bf50 -->
