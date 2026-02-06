from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from activities.agent import agent_step
    from activities.report import generate_report
    from activities.telemetry import collect_telemetry


@workflow.defn
class LatencyWorkflow:
    @workflow.run
    async def run(self, scenario: str = "default") -> Dict[str, Any]:
        pre = await workflow.execute_activity(
            collect_telemetry,
            args=[scenario, "pre", None],
            start_to_close_timeout=timedelta(seconds=5),
        )
        agent_result = await workflow.execute_activity(
            agent_step,
            args=[scenario, pre],
            start_to_close_timeout=timedelta(seconds=60),
        )
        diagnosis = agent_result["diagnosis"]
        fix_plan = agent_result["fix_plan"]
        post = agent_result["post_telemetry"]
        validation = agent_result["validation"]
        report = await workflow.execute_activity(
            generate_report,
            args=[
                scenario,
                pre,
                post,
                diagnosis,
                fix_plan,
                validation,
                "./reports/latency_report.json",
                agent_result.get("agent_trace", []),
            ],
            start_to_close_timeout=timedelta(seconds=5),
        )

        return report
