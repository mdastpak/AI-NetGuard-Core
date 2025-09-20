"""
Optimization Agent

Responsible for hyperparameter optimization, model fine-tuning, and
performance optimization across the AI-NetGuard system.
"""

from typing import Dict, Any
from .base_agent import BaseAgent


class OptimizationAgent(BaseAgent):
    """Agent specialized in optimization and fine-tuning."""

    def __init__(self, coordinator_agent=None, **kwargs):
        system_message = """
        You are the OptimizationAgent, responsible for optimizing all aspects
        of AI-NetGuard's performance and efficiency.
        """

        super().__init__(
            name="OptimizationAgent",
            system_message=system_message,
            coordinator_agent=coordinator_agent,
            **kwargs
        )

        self.capabilities = ["hyperparameter_tuning", "performance_optimization", "resource_optimization"]
        self.dependencies = ["ModelArchitectAgent", "EvaluationAgent"]

    async def _execute_task(self, task_description: str, **kwargs) -> Any:
        if "optimize_hyperparameters" in task_description.lower():
            return await self._optimize_hyperparameters(**kwargs)
        else:
            return {"status": "completed", "task": task_description}