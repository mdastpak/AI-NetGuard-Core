"""
Monitoring Agent

Responsible for real-time monitoring, performance tracking, and health
assessment of the AI-NetGuard system and its components.
"""

from typing import Dict, Any
from .base_agent import BaseAgent


class MonitoringAgent(BaseAgent):
    """Agent specialized in system monitoring and health assessment."""

    def __init__(self, coordinator_agent=None, **kwargs):
        system_message = """
        You are the MonitoringAgent, responsible for continuous monitoring
        of AI-NetGuard's performance, health, and operational status.
        """

        super().__init__(
            name="MonitoringAgent",
            system_message=system_message,
            coordinator_agent=coordinator_agent,
            **kwargs
        )

        self.capabilities = ["performance_monitoring", "health_assessment", "anomaly_detection"]
        self.dependencies = ["All agents"]

    async def _execute_task(self, task_description: str, **kwargs) -> Any:
        if "monitor_performance" in task_description.lower():
            return await self._monitor_performance(**kwargs)
        elif "health_check" in task_description.lower():
            return await self._health_check(**kwargs)
        else:
            return {"status": "completed", "task": task_description}