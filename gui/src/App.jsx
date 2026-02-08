import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = "/api";

const statusColors = {
  COMPLETED: "status success",
  FAILED: "status danger",
  RUNNING: "status warning",
  UNKNOWN: "status neutral",
};

const formatTimeShort = (iso) => {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleString(undefined, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
};

const shorten = (value, max = 120) => {
  if (!value) return "";
  if (value.length <= max) return value;
  return `${value.slice(0, max)}…`;
};

const summarizeTrace = (trace) => {
  const toolCalls = trace.filter((item) => item.log === "tool_call").length;
  const toolResults = trace.filter((item) => item.log === "tool_result").length;
  const llmCalls = trace.filter((item) => item.log === "llm_usage").length;
  return { toolCalls, toolResults, llmCalls };
};

const formatThinking = (text) => {
  if (!text) return null;
  // Simple splitter for code blocks
  const parts = text.split(/(```[\s\S]*?```)/g);
  return parts.map((part, index) => {
    if (part.startsWith("```") && part.endsWith("```")) {
      // Try to extract language
      const firstLineEnd = part.indexOf('\n');
      let content = part.slice(3, -3);
      if (firstLineEnd > -1 && firstLineEnd < 10) {
         content = part.slice(firstLineEnd + 1, -3);
      }
      return <div key={index} className="thinking-code-block">{content}</div>;
    }
    // Split for inline code
    return (
        <span key={index} className="thinking-text">
            {part.split(/(`[^`]+`)/g).map((subPart, i) => {
                if (subPart.startsWith("`") && subPart.endsWith("`")) {
                    return <code key={i} className="thinking-code-inline">{subPart.slice(1, -1)}</code>;
                }
                // Handle bold
                return subPart.split(/(\*\*[^*]+\*\*)/g).map((subSubPart, j) => {
                     if (subSubPart.startsWith("**") && subSubPart.endsWith("**")) {
                         return <strong key={j} className="thinking-bold">{subSubPart.slice(2, -2)}</strong>;
                     }
                     return subSubPart;
                });
            })}
        </span>
    );
  });
};

export default function App() {
  const [runs, setRuns] = useState([]);
  const [runsError, setRunsError] = useState("");
  const [selectedId, setSelectedId] = useState(() => {
    if (typeof window === "undefined") return null;
    try {
      return window.localStorage.getItem("selectedWorkflowId");
    } catch (e) {
      return null;
    }
  });
  const [selectedRun, setSelectedRun] = useState(null);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const thinkingListRef = useRef(null);
  const previousThinkingCountRef = useRef(0);
  const previousSelectedIdRef = useRef(null);

  const fetchRuns = async () => {
    try {
      const response = await fetch(`${API_BASE}/runs`);
      if (!response.ok) {
        throw new Error(`Failed to load runs (${response.status})`);
      }
      const data = await response.json();
      const list = Array.isArray(data.runs) ? data.runs : [];
      setRuns(list);
      setRunsError("");
    } catch (error) {
      console.error(error);
      setRuns([]);
      setRunsError("Unable to load runs from API.");
    } finally {
      setLoading(false);
    }
  };

  const fetchRunDetail = async (workflowId, silent = false) => {
    if (!workflowId) return;
    if (!silent) setDetailLoading(true);
    try {
      const response = await fetch(`${API_BASE}/runs/${workflowId}`);
      if (response.ok) {
        const data = await response.json();
        setSelectedRun(data);
      }
    } catch (e) {
      console.error(e);
    } finally {
      if (!silent) setDetailLoading(false);
    }
  };

  useEffect(() => {
    fetchRuns();
    const interval = setInterval(fetchRuns, 5000);
    return () => clearInterval(interval);
  }, []);

  // Poll active run details
  useEffect(() => {
    if (!selectedId || !selectedRun) return;
    
    // Only poll if the run is active or we suspect it is
    const isActive = selectedRun.status === "RUNNING" || selectedRun.status === "UNKNOWN";
    if (!isActive) return;

    const interval = setInterval(() => {
      fetchRunDetail(selectedId, true);
    }, 1000);

    return () => clearInterval(interval);
  }, [selectedId, selectedRun?.status]);

  useEffect(() => {
    if (!selectedId && runs.length) {
      const first = runs[0].workflow_id;
      setSelectedId(first);
      if (typeof window !== "undefined") {
        try {
          window.localStorage.setItem("selectedWorkflowId", first);
        } catch (e) {}
      }
    }
  }, [runs, selectedId]);

  const handleSelect = (workflowId) => {
    setSelectedId(workflowId);
    if (typeof window !== "undefined") {
      try {
        window.localStorage.setItem("selectedWorkflowId", workflowId);
      } catch (e) {}
    }
  };

  useEffect(() => {
    if (selectedId) {
      fetchRunDetail(selectedId);
    }
  }, [selectedId]);

  const trace = selectedRun?.agent_trace || [];
  const llmStats = trace.filter((item) => item.log === "llm_usage");
  const thinkingBlocks = llmStats
    .map((item, index) => ({
      index: index + 1,
      thinking: item.output?.thinking,
      outputTokens: item.output?.output_tokens,
      stopReason: item.output?.stop_reason,
    }))
    .filter((item) => item.thinking);

  const timeline = useMemo(() => {
    if (!selectedRun) return [];
    return [
      { name: "collect_pre_telemetry", status: "COMPLETED" },
      { name: "agent_step", status: selectedRun.status || "COMPLETED" },
      { name: "generate_report", status: selectedRun.status || "COMPLETED" },
    ];
  }, [selectedRun]);

  useEffect(() => {
    const list = thinkingListRef.current;
    if (!list) return;

    const selectedChanged = previousSelectedIdRef.current !== selectedId;
    const hasNewThinkingStep =
      thinkingBlocks.length > previousThinkingCountRef.current;

    if (selectedChanged || hasNewThinkingStep) {
      window.requestAnimationFrame(() => {
        list.scrollTop = list.scrollHeight;
      });
    }

    previousSelectedIdRef.current = selectedId;
    previousThinkingCountRef.current = thinkingBlocks.length;
  }, [selectedId, thinkingBlocks.length]);

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark" />
          <div>
            <h1>Workflow Agent</h1>
            <p>Observability</p>
          </div>
        </div>
        <div className="section-title">History</div>
        {loading ? (
          <div className="loading">Loading…</div>
        ) : runsError ? (
          <div className="empty">{runsError}</div>
        ) : runs.length === 0 ? (
          <div className="empty">No runs found</div>
        ) : (
          <ul className="runs">
            {runs.map((run) => (
              <li
                key={run.workflow_id}
                className={
                  run.workflow_id === selectedId ? "run-item active" : "run-item"
                }
                onClick={() => handleSelect(run.workflow_id)}
              >
                <div className="run-row">
                  <span className="run-id">{shorten(run.workflow_id, 8)}</span>
                  <span className="run-tags">
                    {run.workflow_id === selectedId && (
                      <span className="pin">Pinned</span>
                    )}
                    <span className={statusColors[run.status] || "status neutral"}>
                      {run.status || "COMPLETED"}
                    </span>
                  </span>
                </div>
                <div className="run-meta">
                  <span>{shorten(run.scenario || "Unknown Scenario", 24)}</span>
                  <span>{run.generated_at ? formatTimeShort(run.generated_at) : "—"}</span>
                </div>
              </li>
            ))}
          </ul>
        )}
      </aside>

      <main className="main">
        {detailLoading || !selectedRun ? (
          <div className="loading">Loading run detail…</div>
        ) : (
          <div className="content">
            <section className="layout split-layout">
              <div className="left-pane">
                <div className="summary">
                  <div>
                    <div className="summary-title">Run Summary</div>
                    <h2>{selectedRun.scenario || "Scenario"}</h2>
                    <p className="summary-subtitle">
                      ID: {selectedRun.workflow_id}
                    </p>
                  </div>
                  <div className="summary-grid">
                    <div className="summary-card">
                      <span>Status</span>
                      <strong>{selectedRun.status || "COMPLETED"}</strong>
                    </div>
                    <div className="summary-card">
                      <span>Improved</span>
                      <strong>
                        {selectedRun.summary?.improved === false
                          ? "No"
                          : "Yes"}
                      </strong>
                    </div>
                  </div>
                </div>

                <div className="panel">
                  <div className="panel-header">
                    <h3>Workflow Timeline</h3>
                  </div>
                  <ul className="timeline">
                    {timeline.map((node) => (
                      <li key={node.name} className="timeline-item">
                        <div className="dot" />
                        <div>
                          <div className="timeline-name">{node.name}</div>
                          <span
                            className={
                              statusColors[node.status] || "status neutral"
                            }
                          >
                            {node.status}
                          </span>
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="panel">
                  <div className="panel-header">
                    <h3>Agent Trace</h3>
                    <span className="panel-meta">
                      {summarizeTrace(trace).toolCalls} tool calls
                    </span>
                  </div>
                  <div className="trace-list">
                    {trace
                      .filter((item) => item.log?.includes("tool"))
                      .map((item, index) => (
                        <div key={`${item.tool}-${index}`} className="trace-item">
                          <div className="trace-title">
                            <span>{item.tool}</span>
                            <span className="trace-tag">{item.log}</span>
                          </div>
                          <div className="trace-body">
                            <pre>{shorten(JSON.stringify(item.input, null, 2), 280)}</pre>
                            <pre>{shorten(JSON.stringify(item.output, null, 2), 280)}</pre>
                          </div>
                        </div>
                      ))}
                  </div>
                </div>

                <div className="panel-grid-row">
                    <div className="panel">
                      <div className="panel-header">
                        <h3>LLM Usage</h3>
                      </div>
                      <div className="usage-grid">
                        {llmStats.map((stat, index) => (
                          <div key={index} className="usage-card">
                            <span>Call {index + 1}</span>
                            <strong>{stat.output?.input_tokens || 0} → {stat.output?.output_tokens || 0}</strong>
                            <em>{stat.output?.stop_reason || ""}</em>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="panel">
                      <div className="panel-header">
                        <h3>Metrics Delta</h3>
                      </div>
                      <div className="metrics">
                        <div>
                          <span>p95</span>
                          <strong>
                            {selectedRun.metrics?.pre?.latency_ms?.p95} → {selectedRun.metrics?.post?.latency_ms?.p95}
                          </strong>
                        </div>
                        <div>
                          <span>Queue</span>
                          <strong>
                            {selectedRun.metrics?.pre?.queue_depth} → {selectedRun.metrics?.post?.queue_depth}
                          </strong>
                        </div>
                      </div>
                    </div>
                </div>
              </div>

              <div className="right-pane">
                <div className="panel full-height">
                  <div className="panel-header">
                    <h3>LLM Thinking Flow</h3>
                    <span className="panel-meta">{thinkingBlocks.length} blocks</span>
                  </div>
                  <div ref={thinkingListRef} className="thinking-list full-scroll">
                    {thinkingBlocks.map((block) => (
                      <div key={block.index} className="thinking-item">
                        <div className="thinking-step-marker">
                            <div className="step-dot"></div>
                            <div className="step-line"></div>
                        </div>
                        <div className="thinking-content-wrapper">
                            <div className="thinking-header">
                              <span>Thinking Step {block.index}</span>
                              <span className="trace-tag">
                                {block.stopReason || ""}
                              </span>
                            </div>
                            <div className="thinking-content">
                                {formatThinking(block.thinking)}
                            </div>
                        </div>
                      </div>
                    ))}
                    {!thinkingBlocks.length && (
                      <div className="empty">No thinking blocks captured.</div>
                    )}
                  </div>
                </div>
              </div>
            </section>
          </div>
        )}
      </main>
    </div>
  );
}
