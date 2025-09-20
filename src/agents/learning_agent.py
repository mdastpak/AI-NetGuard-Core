"""
Learning Agent

Responsible for continuous learning, adaptation, and knowledge
acquisition in the AI-NetGuard system.
"""

from typing import Dict, Any
from .base_agent import BaseAgent


class LearningAgent(BaseAgent):
    """Agent specialized in continuous learning and adaptation."""

    def __init__(self, coordinator_agent=None, **kwargs):
        system_message = """
        You are the LearningAgent, responsible for continuous learning and
        adaptation to new threats and patterns in AI-NetGuard.
        """

        super().__init__(
            name="LearningAgent",
            system_message=system_message,
            coordinator_agent=coordinator_agent,
            **kwargs
        )

        self.capabilities = ["continuous_learning", "adaptation", "knowledge_acquisition"]
        self.dependencies = ["DataSynthesisAgent", "EvaluationAgent"]

    async def _execute_task(self, task_description: str, **kwargs) -> Any:
        if "learn_patterns" in task_description.lower():
            return await self._learn_patterns(**kwargs)
        else:
            return {"status": "completed", "task": task_description}