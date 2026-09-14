# 脱浏览器控制 NPU 环境（npu_console.py）

> **背景**：`ascend-npu-console` 原工具包依赖 `mcp__node_repl__js` 宿主运行时
> （`notebook.js` 内用 `agent.browsers` 找标签页）。在没有该运行时的环境里，外层不可用。
>
> 本方案拆开分析后发现：**外层（找标签页）是浏览器专属，内层（REST + WebSocket）是纯网络**。
> 后者可以脱离浏览器。

---

## 一、已证实的事实

| 项 | 结论 | 证据 |
|---|---|---|
| REST 列终端是否免 cookie | **是** | 假 instanceId → HTTP **502**（非 401/403），说明鉴权在实例层 |
| WS 握手鉴权顺序 | **先查实例存在性，再查 cookie** | 假 instanceId → WS 握手 **502**，响应 `Server: elb`（未到应用层） |
| 网络可达性 | gitcode.com / aihub-run.gitcode.com / cannjudge.cn **TCP 443 均可达** | 探针实测 |
| instanceId 能否免浏览器获取 | **否** | 5 个候选 API 端点全 404；`SKILL.md:106` 确认只能从 iframe src 读 |

复现命令：
```bash
# WS 鉴权层判定（不碰真实实例）
python scripts/npu_console.py --inst nonexistent-probe --term probe probe
# -> 握手状态: HTTP/1.1 502 Bad Gateway
# -> => 实例/终端不存在：id 或 term 有误，或实例未启动
```

---

## 二、⚠ 关键未验证假设

> **"实例存在时，WS 也不需要 cookie" —— 本方案依赖此假设，但尚未证实。**

- 已证实的只是：*实例不存在*时服务器没要求 cookie（因为没走到那一步）。
- **尚未证实**：*实例存在*时是否要求 cookie。可能性有二：
  - (A) 鉴权完全在实例层（instanceId 本身就是凭据）→ **可直连** ✅
  - (B) 实例层之后还有 cookie 校验 → 需要从浏览器导出 cookie

**判别方法**（拿到真实 instanceId 后执行一条命令即可）：
```bash
python scripts/npu_console.py --inst <真实id> --term <终端名> probe
```
- 返回 **101** → 假设 (A) 成立，**可直连**
- 返回 **401/403** → 假设 (B) 成立，需要 cookie（见下方预案）

---

## 三、怎么用

### Step 1 — 拿 instanceId（唯一需要手动的一步，约 30 秒）

instanceId 每次 Notebook 重建都会变，且**只能从浏览器读**。二选一：

**方式 A：浏览器控制台（推荐）**
1. 打开 gitcode.com 的 Notebook，确认已进入实例（地址含 `/notebook/lab`）
2. F12 → Console，粘贴：
   ```js
   document.querySelector('iframe[src*="aihub-run"]').src
   ```
3. 从输出里取 `/online-action/<这里是id>/` 中间那段

**方式 B：元素检查**
在 iframe 上右键 → 检查 → 看 `src` 属性。

### Step 2 — 探测能否直连

```bash
cd D:/Desktop/OP-Learning/Ascend/ascend-npu-console
python scripts/npu_console.py --inst <id> --term <terminalName> list   # 列终端（免 cookie 已证实）
python scripts/npu_console.py --inst <id> --term <terminalName> probe  # 判定 A 还是 B
```

> `--term` 从 Step 2 的 `list` 输出里取（通常是 `1` 或 `terminal-1` 之类）。
> 若 Notebook 里没开终端，先在网页上开一个。

### Step 3 — 跑命令

```bash
python scripts/npu_console.py --inst <id> --term <name> run "npu-smi info"
python scripts/npu_console.py --inst <id> --term <name> run "cd ~/mhc && bash verify.sh v116"
python scripts/npu_console.py --inst <id> --term <name> run "ls -la" --timeout-ms 60000
```

---

## 四、若判定为 (B)：需要 cookie 的预案

不要手工导出 cookie（易泄密且时效短）。更稳的做法：

**方案 B1 — 只把 instanceId 交给浏览器一次**
用浏览器调试工具的页面上下文 `eval` 执行原版 `notebook.js` 的内层逻辑
（REST + WebSocket 那段），跳过 `agent.browsers`。页面上下文自带 cookie。

**方案 B2 — 从浏览器读 cookie 后传给本脚本**
读 cookie 需要放开浏览器白名单（`RIVET_BROWSER_ALLOWLIST`），
且**不要把 cookie 写进文件**——通过环境变量或一次性参数传入。
⚠ 本脚本当前**不支持** `--cookie` 参数（避免凭据落盘）。需要时再加。

---

## 五、为什么这条路值得走（对比纯 OJ 提交）

| 维度 | OJ 提交 | NPU 直连 |
|---|---|---|
| 反馈粒度 | 只有 5 个 case 的最终 time | **任意命令、任意 shape、任意中间量** |
| 迭代速度 | 受 429 限流（70-80s/次） | 任意快 |
| 性能信息 | 无 | `npu-smi` / msprof / nsys 级 |
| 编译错误定位 | 只能看最终 status | **完整的编译器输出** |
| 代价 | 无需环境 | 需一次手工拿 instanceId |

**关键**：`FACTS.md F2.1c` 指出最大缺口是 Case1（PATH3/TinyH4 的 3.6 浪费）。
定位这个需要**看编译器输出和逐段计时**，OJ 给不了——这正是 NPU 直连的价值。

---

## 六、诚实的边界

- 本方案**尚未在真实实例上验证过**（`probe` 只跑过假 instanceId）。
- 未能完成的部分：用户当前没有打开 Notebook，我无法获取真实 instanceId。
- 若假设 (A) 不成立（需要 cookie），本脚本的 `run` 会失败——但 `probe` 会**明确告知**，
  不会静默失败。
- Notebook 是**临时环境（~2 小时）**，重启后本地文件全删——要保留的先 commit。
