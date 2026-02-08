# Agent Step Design (Latency POC)

## Context
- Current workflow uses Temporal activities for analysis, fix planning, fix application, and validation.
- New requirement: insert one agent step so the AI agent handles analyze_latency, propose_fix, apply_fix, and validate_improvement.
- Use LangChain as the agent framework; target LLM is MINMAX via real API (no LLM mock).

## Goals and success criteria
- Agent drives analysis, proposes fixes, applies fixes, and validates improvement within a single Temporal activity.
- MINMAX is the target LLM; use the real API from day one.
- Tool responses look realistic and align with existing mock telemetry/config structures.
- Agent loop is bounded (max 10 rounds) and cannot run indefinitely.
- Framework is extensible so more tools and real LLM calls can be added later.

## Non-goals
- Replacing Temporal orchestration entirely.
- Building real integrations to production telemetry/logging systems.
- Implementing complex multi-agent collaboration in v1.

## Proposed workflow shape
- collect_telemetry (pre)
- agent_step (analyze + propose + apply + validate, and collect post telemetry internally)
- generate_report

Rationale:
- The agent needs post-fix telemetry to validate improvement; doing that internally keeps the agent as a single workflow step.
- Reporting remains a separate activity so JSON output stays consistent.

## Agent responsibilities
- Analyze latency symptoms from pre-telemetry and logs.
- Propose a fix plan with rationale and expected impact.
- Apply the fix by invoking a simulation tool.
- Collect post-telemetry and validate improvement.
- Return structured artifacts for reporting.

## Agent interface
Input
- scenario: str
- pre_telemetry: Dict[str, Any]

Output (AgentResult)
- diagnosis: Dict[str, Any]
- fix_plan: Dict[str, Any]
- fix_result: Dict[str, Any]
- post_telemetry: Dict[str, Any]
- validation: Dict[str, Any]
- agent_trace: List[Dict[str, Any]]

## LangChain setup
- Use a single agent with tool-calling (ReAct or OpenAI-tools style) depending on MINMAX support.
- Configure MINMAX API credentials via environment variable:
  - `ANTHROPIC_AUTH_TOKEN` (per current environment constraint)
- Runtime configuration (optional env vars):
  - `ANTHROPIC_BASE_URL` (Minimax endpoint)
  - `MINIMAX_BASE_URL` (fallback if ANTHROPIC_BASE_URL unset; default `https://api.minimax.chat`)
  - `MINIMAX_CHAT_COMPLETIONS_URL` (override full endpoint)
  - `MINIMAX_MODEL` (model name, default `minimax-m2.1`)
- Evaluate whether MINMAX supports LangChain tools/skills directly.
  - If not, use a custom tool router with structured JSON outputs.

## Real-world reference data
- vLLM metrics format (Prometheus text): `src/mock_data/vllm_metrics_prometheus.txt`
- vLLM latency/hang example (issue #33369): `src/mock_data/vllm_issue_33369_logs.txt`
  - Scenario summary: vLLM 0.15.0 container hangs during engine startup after upgrade; logs repeat "Waiting for 1 local, 0 remote core engine proc(s) to start."
- Linux memory pressure (OOM killer): `src/mock_data/linux_oom_killer.log`
- Linux CPU stall (soft lockup): `src/mock_data/linux_cpu_soft_lockup.log`
- NVIDIA GPU driver error (Xid): `src/mock_data/nvidia_xid_errors.log`
- NCCL network/communication error: `src/mock_data/nccl_net_socket_truncated.log`
- NCCL IB/socket errors (timeouts/resets): `src/mock_data/nccl_net_ib_errors.log`
- CUDA ECC failure (runtime): `src/mock_data/cuda_ecc_uncorrectable_error.log`
- vLLM CUDA OOM example: `src/mock_data/vllm_cuda_oom_issue_26863.log`

## Tools (planned, mocked but realistic)
All tool responses should be deterministic per scenario and compatible with existing mock schemas.

1) fetch_vllm_logs
   - Input: {scenario, window, limit, level, topics}
   - Output: {source, window, lines: [str], parsed: [{level, message}], errors: [str]}
   - Notes: topics can include "nccl"/"network" to surface communication logs.

2) fetch_system_logs
   - Input: {scenario, window, sources: ["dmesg"|"journal"], limit, topics}
   - Output: {source, window, lines: [str], errors: [str]}
   - Notes: topics can include "gpu" or "ecc" to surface driver/ECC lines.

3) fetch_prometheus_metrics
   - Input: {scenario, endpoint}
   - Output: {endpoint, raw_text, samples: [{name, labels, value}]}

4) fetch_config
   - Input: {scenario} or {config_id}
   - Output: {config, runtime: {batch_size, max_num_batched_tokens, max_num_seqs}, version}

5) fetch_runtime_state
   - Input: {scenario}
   - Output: {queue_depth, running, waiting, scheduler_backlog_ms, worker_health: {healthy, stuck}}

6) fetch_gpu_stats
   - Input: {scenario}
   - Output: {gpus: [{id, util, mem_util, mem_used_gb, mem_total_gb, power_w, temp_c}]}

7) analyze_metrics
   - Input: {pre_telemetry, prometheus_samples}
   - Output: {latency_breakdown, derived_metrics, hotspot}

8) apply_fix_simulation
   - Input: {current_config, fix_plan}
   - Output: {updated_config, diff, expected_impact}

9) collect_post_telemetry
   - Input: {scenario, updated_config}
   - Output: {post_telemetry}

10) validate_improvement
   - Input: {pre_telemetry, post_telemetry}
   - Output: {status, deltas, summary}

Notes
- The agent makes the decisions; tools provide facts and deterministic simulations.
- Tool responses should be consistent across runs for the same scenario.

## Prompting and structure
- System: role is latency incident fixer; prefer concise, structured outputs.
- Require final output to follow the AgentResult schema.
- Include a tool-use plan, but avoid verbose chain-of-thought in logs.

## Loop control and safety
- max_iterations = 10 for the LangChain agent.
- If the agent reaches the limit, return best-effort outputs and mark validation as "unknown".
- Add a hard timeout at the Temporal activity level.

## Error handling
- Tool failures: return a structured error in agent_trace and proceed with best-effort.
- Missing data: default to conservative fix plan and note uncertainty.

## Useful CLI tools (reference for agent)
Logs and system events
- `journalctl -u vllm -S "2026-02-01 00:00"`
- `journalctl -k` / `dmesg -T`
- `docker logs <container>` / `kubectl logs <pod> -n <ns>`

Memory and CPU
- `free -h` / `vmstat 1` / `cat /proc/meminfo`
- `ps -eo pid,cmd,rss --sort=-rss | head`
- `top -H -p <pid>` / `pidstat -u -p <pid> 1` / `mpstat -P ALL 1`

GPU and driver health
- `nvidia-smi -q -d ECC,MEMORY,TEMPERATURE,POWER`
- `nvidia-smi dmon -s u`
- `dcgmi diag -r 3`
- `nvidia-bug-report.sh`

Network and NCCL
- `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=NET`
- `ip addr` / `ip route` / `ethtool -S <iface>`
- `ss -tnp` / `ping <host>` / `mtr <host>` / `iperf3 -c <host>`

Metrics endpoint
- `curl -s http://<host>:<port>/metrics`

## Reporting integration
- generate_report consumes the same artifacts as today, sourced from agent outputs.
- Include agent_trace summary in the report for debugging.

## Open questions
- Confirm MINMAX tool-calling compatibility with LangChain and required model name/endpoint.
- Decide whether to keep collect_telemetry as a separate activity for observability, or fully encapsulate it in the agent step.

## Next steps (implementation)
- Add a new Temporal activity: src/activities/agent.py
- Update src/workflows/latency_workflow.py to replace four activities with agent_step
- Add LangChain dependencies and MINMAX API integration
- Implement the tool layer backed by `src/mock_data/`
