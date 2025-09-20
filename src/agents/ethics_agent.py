"""
Ethics Agent

Responsible for ethical decision making, bias detection, and ensuring
responsible AI practices in the AI-NetGuard system.
"""

from typing import Dict, Any
from .base_agent import BaseAgent


class EthicsAgent(BaseAgent):
    """Agent specialized in ethical AI and responsible practices."""

    def __init__(self, coordinator_agent=None, **kwargs):
        system_message = """
        You are the EthicsAgent, responsible for ensuring ethical AI practices
        and responsible decision making in AI-NetGuard.
        """

        super().__init__(
            name="EthicsAgent",
            system_message=system_message,
            coordinator_agent=coordinator_agent,
            **kwargs
        )

        self.capabilities = ["bias_detection", "ethical_decision_making", "fairness_assessment"]
        self.dependencies = ["EvaluationAgent", "PrivacyAgent"]

    async def _execute_task(self, task_description: str, **kwargs) -> Any:
        if "assess_bias" in task_description.lower():
            return await self._assess_bias(**kwargs)
        else:
            return {"status": "completed", "task": task_description}