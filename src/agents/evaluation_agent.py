"""
Evaluation Agent

Responsible for model evaluation, performance metrics, and quality
assessment of the AI-NetGuard system.
"""

from typing import Dict, Any
from .base_agent import BaseAgent


class EvaluationAgent(BaseAgent):
    """Agent specialized in evaluation and quality assessment."""

    def __init__(self, coordinator_agent=None, **kwargs):
        system_message = """
        You are the EvaluationAgent, responsible for comprehensive evaluation
        and quality assessment of AI-NetGuard's performance.
        """

        super().__init__(
            name="EvaluationAgent",
            system_message=system_message,
            coordinator_agent=coordinator_agent,
            **kwargs
        )

        self.capabilities = ["performance_evaluation", "quality_assessment", "metrics_calculation"]
        self.dependencies = ["ModelArchitectAgent", "MonitoringAgent"]

    async def _execute_task(self, task_description: str, **kwargs) -> Any:
        if "evaluate_model" in task_description.lower():
            return await self._evaluate_model(**kwargs)
        else:
            return {"status": "completed", "task": task_description}