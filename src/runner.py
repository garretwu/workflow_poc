from __future__ import annotations

import argparse
import asyncio
import uuid

from temporalio.client import Client
from temporalio.worker import Worker

from activities.analysis import analyze_latency
from activities.fix import apply_fix_simulation, propose_fix
from activities.report import generate_report, validate_improvement
from activities.telemetry import collect_telemetry
from workflows.latency_workflow import LatencyWorkflow

TASK_QUEUE = "vllm-latency-poc"


def _render_report(report: dict) -> None:
    summary = report["summary"]
    causes = ", ".join(summary["root_causes"])
    pre = report["metrics"]["pre"]["latency_ms"]
    post = report["metrics"]["post"]["latency_ms"]

    print("Scenario:", report["scenario"])
    print("Improved:", summary["improved"])
    print("Root causes:", causes)
    print(
        "Latency (p50/p95/p99):",
        f"{pre['p50']}/{pre['p95']}/{pre['p99']} -> {post['p50']}/{post['p95']}/{post['p99']} ms",
    )
    print("Recommendations:", ", ".join(report["recommendations"]))
    print("Report JSON:", report["report_path"])


async def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal vLLM latency POC")
    parser.add_argument(
        "--scenario",
        default="default",
        choices=["default", "oom_pressure", "tokenizer_cpu_bottleneck"],
        help="Mock scenario to simulate",
    )
    parser.add_argument(
        "--address",
        default="localhost:7233",
        help="Temporal server address",
    )
    parser.add_argument(
        "--task-queue",
        default=TASK_QUEUE,
        help="Temporal task queue",
    )
    args = parser.parse_args()

    client = await Client.connect(args.address)
    workflow_id = f"vllm-latency-{uuid.uuid4().hex[:8]}"

    async with Worker(
        client,
        task_queue=args.task_queue,
        workflows=[LatencyWorkflow],
        activities=[
            collect_telemetry,
            analyze_latency,
            propose_fix,
            apply_fix_simulation,
            validate_improvement,
            generate_report,
        ],
    ):
        report = await client.execute_workflow(
            LatencyWorkflow.run,
            args.scenario,
            id=workflow_id,
            task_queue=args.task_queue,
        )
        _render_report(report)


if __name__ == "__main__":
    asyncio.run(main())
