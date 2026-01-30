from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict

from temporalio import workflow

from activities.analysis import analyze_latency
from activities.fix import apply_fix_simulation, propose_fix
from activities.report import generate_report, validate_improvement
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
        diagnosis = await workflow.execute_activity(
            analyze_latency,
            pre,
            start_to_close_timeout=timedelta(seconds=5),
        )
        fix_plan = await workflow.execute_activity(
            propose_fix,
            args=[pre, diagnosis],
            start_to_close_timeout=timedelta(seconds=5),
        )
        fix_result = await workflow.execute_activity(
            apply_fix_simulation,
            args=[pre["config"], fix_plan],
            start_to_close_timeout=timedelta(seconds=5),
        )
        post = await workflow.execute_activity(
            collect_telemetry,
            args=[scenario, "post", fix_result["updated_config"]],
            start_to_close_timeout=timedelta(seconds=5),
        )
        validation = await workflow.execute_activity(
            validate_improvement,
            args=[pre, post],
            start_to_close_timeout=timedelta(seconds=5),
        )
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
            ],
            start_to_close_timeout=timedelta(seconds=5),
        )

        return report
