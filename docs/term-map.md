# Term map: GH-600 ↔ runtime agents ↔ Google ADK ↔ Claude

Two ladders, one idea. GH-600 (Copilot): **suggest → draft PR → PR with checks → auto-merge on low risk**. Runtime agents: **1 Suggest → 2 Act with approval → 3 Act and report → 4 Fully autonomous** (each level is a *rung*). GH-600 splits rung 2 finer and has no clean rung 3.

![GH-600 vs runtime vs ADK vs Claude ladder](diagrams/w01d05-gh600-vs-runtime-ladder.svg)
*Source: [diagrams/w01d05-gh600-vs-runtime-ladder.excalidraw](diagrams/w01d05-gh600-vs-runtime-ladder.excalidraw)*

| GH-600 term (W1 D5 step) | Rung | Runtime-agent term | Google ADK | Claude API / Agent SDK |
|---|---|---|---|---|
| **Suggest** (`m1-ladder`) | **1** | Agent proposes, human does everything | Read-only tools, or `output_schema` only | Agent SDK `plan` mode; Messages API `tool_choice: {"type": "none"}` |
| **Draft PR**: Copilot's work waits unmerged (`m3-assign`) | **2** | Draft output awaiting approval: `interrupt()` pauses before the action executes | `FunctionTool(tool, require_confirmation=True)`: execution pauses until confirmed | `canUseTool` callback (`permission_mode="default"`); Messages API: your app holds the `tool_use` until a human approves |
| **Required checks / PR with checks**: CI must pass (`m5-workflows`) | **2 + gate** | Validation step / eval gate before an action takes effect | `before_tool_callback` / `before_model_callback`: a truthy return skips the call | `PreToolUse` hook (runs first; its deny applies even in `bypassPermissions`); `disallowed_tools` deny rules |
| **Steering** via PR comment or session prompt box (`m4-pr-comment`, `m4-prompt-box`) | Intervention at **2** | Human input mid-run: `Command(resume=...)` with approve / reject / edit | `tool_context.request_confirmation(hint, payload)`; reply arrives as a `FunctionResponse` (`/run`, `/run_sse`) | New message mid-session (`ClaudeSDKClient.query()`) or `set_permission_mode()` |
| **Independent approval**: the assigner's own approval doesn't count (`m6-approve`) | **2** | Separation of duties: approver ≠ requester | Confirmation answered by a different user/role (app-level check) | `canUseTool` answered by a different reviewer (app-level check) |
| **`copilot/` branch**: can't push to `main` (`m5-branch`) | Containment, keeps it below **4** | Least-privilege tool surface | Give the agent only scoped tools | `disallowed_tools` (bare name removes the tool; scoped rules like `Bash(rm *)` deny in every mode); path-scoped `Edit(...)` rules |
| **Session log** (`m2-traceability`, `m2-trace-commits`) | The "report" of **3** | Trace: one span per LLM call and tool call (AAI 10.3) | Session events + `after_tool_callback` logging + Web UI trace | `tool_use` / `tool_result` message stream + `PostToolUse` hook logging |
| **Auto-merge on low risk** (`m1-ladder`) | **4** | Fully autonomous, audited low-risk actions only (e.g. auto-close documented false positives, QA sampled) | No confirmation, or `require_confirmation=<threshold fn>` so only low-risk calls skip the human | `bypassPermissions` (hooks and deny rules still apply), or `allowed_tools` for specific tools |
| **Bypass actor** (`m6-ruleset`) | Jump **2 → 4** | Removing the gate | Deleting `require_confirmation` | Switching to `bypassPermissions` |

GH-600 steps `m0`, `m7`, `m8`, `m9` (bets, controls file, Learn module, commit) aren't rungs.

*Same idea in all four columns: autonomy is earned per action, the human gate sits **before** anything irreversible, and real controls live in the layer that can't be skipped (rulesets, `before_tool_callback`, `PreToolUse` hooks / deny rules).*

Sources: [ADK tool confirmation](https://adk.dev/tools-custom/confirmation/) · [ADK callbacks](https://adk.dev/callbacks/types-of-callbacks/) · [Claude tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview) · [Claude Agent SDK permissions](https://code.claude.com/docs/en/agent-sdk/permissions)
