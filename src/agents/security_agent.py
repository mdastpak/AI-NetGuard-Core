"""
Security Agent

Responsible for implementing security measures, encryption, and protection
against cyber threats targeting the AI-NetGuard system.
"""

from typing import Dict, Any
from .base_agent import BaseAgent


class SecurityAgent(BaseAgent):
    """Agent specialized in system security and threat protection."""

    def __init__(self, coordinator_agent=None, **kwargs):
        system_message = """
        You are the SecurityAgent, responsible for implementing comprehensive
        security measures and protecting AI-NetGuard from cyber threats.
        """

        super().__init__(
            name="SecurityAgent",
            system_message=system_message,
            coordinator_agent=coordinator_agent,
            **kwargs
        )

        self.capabilities = ["encryption", "threat_detection", "access_control"]
        self.dependencies = ["MonitoringAgent", "PrivacyAgent"]

    async def _execute_task(self, task_description: str, **kwargs) -> Any:
        if "encrypt_data" in task_description.lower():
            return await self._encrypt_data(**kwargs)
        else:
            return {"status": "completed", "task": task_description}