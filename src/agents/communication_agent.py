"""
Communication Agent

Responsible for inter-agent communication, coordination, and information
sharing in the AI-NetGuard system.
"""

from typing import Dict, Any
from .base_agent import BaseAgent


class CommunicationAgent(BaseAgent):
    """Agent specialized in inter-agent communication and coordination."""

    def __init__(self, coordinator_agent=None, **kwargs):
        system_message = """
        You are the CommunicationAgent, responsible for facilitating communication
        and coordination between all agents in AI-NetGuard.
        """

        super().__init__(
            name="CommunicationAgent",
            system_message=system_message,
            coordinator_agent=coordinator_agent,
            **kwargs
        )

        self.capabilities = ["message_routing", "coordination", "information_sharing"]
        self.dependencies = ["All agents"]

    async def _execute_task(self, task_description: str, **kwargs) -> Any:
        if "route_message" in task_description.lower():
            return await self._route_message(**kwargs)
        else:
            return {"status": "completed", "task": task_description}