---
name: 利用cannbot或ascendkg等工具-把当前环境配置成8-5-或配置双环境-使我们后续的工
description: 利用cannbot或ascendkg等工具，把当前环境配置成8.5（或配置双环境），使我们后续的工作在8.5环境上进行；然后official里面还有一个cann-ops-ladder-2026也可以参考 这个是算子天梯赛 东西也挺多的，尽量吧official里面的看完了 ；先做这两件事 — verified by 2 checks
triggers: ['null', 'ENVIRONMENT', '利用cannbot或ascendkg等工具', '或配置双环境']
---

# 利用cannbot或ascendkg等工具-把当前环境配置成8-5-或配置双环境-使我们后续的工

> 自动从会话 83642d4c 蒸馏的草稿。审核后用 `/skill approve 利用cannbot或ascendkg等工具-把当前环境配置成8-5-或配置双环境-使我们后续的工` 入库，或 `/skill reject 利用cannbot或ascendkg等工具-把当前环境配置成8-5-或配置双环境-使我们后续的工` 丢弃。

## Steps
1. 操作：todo
2. 阅读 / 搜索：echo "=== 本机已有 CANN 相关包/安装器 ==="
ls /opt/conda/Asc、# 本机没有 8.5。查官方下载渠道——Ascend 社区版 toolkit 8.5 下载页
ech
3. 验证：export ASCEND_KG_API_KEY="kg-test-1489df447809de55
4. 阅读 / 搜索：# 取最高分节点的正文（torch-npu CI Docker Images 的安装说明）
expo、grep -oE "https?://[^ \"')]+8\.5[^ \"')]*" /worksp、curl -s -m 15 -I "https://ascend-repo.obs.cn-east-
5. 操作：todo
6. 阅读 / 搜索：cd /workspace/notebook13/MhcHeadCollapse/.rivet/sc、/tmp/rivet-raw/2380b812e78ff44f2b66d34b.raw、/tmp/rivet-raw/91fde890b88cb40787fb5dc4.raw
7. 操作：job
8. 阅读 / 搜索：whoami; mkdir -p /workspace/cann85-pkg 2>/dev/null、ls -ld /workspace 2>/dev/null; mkdir -p /workspace、curl -s -o Ascend-cann-toolkit_8.5.0_linux-aarch64
9. 操作：job
10. 阅读 / 搜索：cd /tmp/cann85-pkg
./Ascend-cann-toolkit_8.5.0_lin、cd /tmp/cann85-pkg
# 安装到 /tmp/cann85（toolkit 完整模式）、cd /workspace/notebook13/MhcHeadCollapse/.rivet/sc
11. 操作：job
12. 阅读 / 搜索：ls /tmp/cann85/ 2>/dev/null; du -sh /tmp/cann85 2>、ls -la /tmp/cann85-pkg/; find /tmp -maxdepth 2 -na
13. 操作：job
14. 阅读 / 搜索：cd /tmp/cann85-pkg
# 非 root 安装器通常需要先看它真正支持的静默选项
./、cd /tmp/cann85-pkg
./Ascend-cann-toolkit_8.5.0_lin
15. 操作：job
16. 阅读 / 搜索：echo "=== 目录结构 ==="; ls /tmp/cann85/cann-8.5.0/ |、cd /workspace/notebook13/MhcHeadCollapse/.rivet/sc、S=/workspace/notebook13/MhcHeadCollapse/.rivet/scr
17. 修改：/workspace/notebook13/MhcHeadCollapse/.rivet/ENVIRONMENT.md
18. 操作：todo、memory
19. 阅读 / 搜索：S=/workspace/notebook13/MhcHeadCollapse/.rivet/scr、grep -rn "507035" /tmp/cann85/cann-8.5.0/include/
20. 操作：memory

## Verified by
- export ASCEND_KG_API_KEY="kg-test-1489df447809de55e30251bf59bf2d6c"
curl -s --compressed --max-time 30 -X POST https://ascend.wiki/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $AS (passed 0)
- # 取最高分节点的正文（torch-npu CI Docker Images 的安装说明）
export ASCEND_KG_API_KEY="kg-test-1489df447809de55e30251bf59bf2d6c"
curl -s --compressed --max-time 30 -X POST https://ascend.wiki/source \
  -H "Content- (passed 0)

<!-- skill-draft-key: c3f3bc4b3692 -->
<!-- source-session: 83642d4c -->
