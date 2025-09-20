"""
Base Agent Class for AI-NetGuard Multi-Agent System

This module provides the foundation for all specialized agents in the system.
All agents inherit from this base class, which provides common functionality
for communication, coordination, and autonomous operation.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import autogen
from autogen import ConversableAgent


class BaseAgent(ConversableAgent, ABC):
    """
    Base agent class extending AutoGen's ConversableAgent.

    Provides common functionality for all specialized agents including:
    - Communication protocols
    - Coordination mechanisms
    - Logging and monitoring
    - Autonomous decision making
    - State management
    """

    def __init__(
        self,
        name: str,
        system_message: str,
        coordinator_agent: Optional['MetaCoordinatorAgent'] = None,
        **kwargs
    ):
        """
        Initialize the base agent.

        Args:
            name: Unique agent identifier
            system_message: System prompt defining agent behavior
            coordinator_agent: Reference to the meta-coordinator for oversight
            **kwargs: Additional arguments for ConversableAgent
        """
        super().__init__(
            name=name,
            system_message=system_message,
            **kwargs
        )

        self.coordinator = coordinator_agent
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{name}")
        self.logger.setLevel(logging.INFO)

        # Agent state
        self.is_active = False
        self.capabilities: List[str] = []
        self.dependencies: List[str] = []
        self.state: Dict[str, Any] = {}

        # Communication history
        self.message_history: List[Dict[str, Any]] = []

        # Consensus tracking
        self.consensus_votes: Dict[str, Any] = {}

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for the agent."""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f'%(asctime)s - {self.name} - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    async def activate(self) -> bool:
        """
        Activate the agent and register with coordinator.

        Returns:
            bool: True if activation successful
        """
        try:
            self.is_active = True
            if self.coordinator:
                await self.coordinator.register_agent(self)
            self.logger.info(f"Agent {self.name} activated successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to activate agent {self.name}: {e}")
            return False

    async def deactivate(self) -> bool:
        """
        Deactivate the agent and unregister from coordinator.

        Returns:
            bool: True if deactivation successful
        """
        try:
            self.is_active = False
            if self.coordinator:
                await self.coordinator.unregister_agent(self)
            self.logger.info(f"Agent {self.name} deactivated successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to deactivate agent {self.name}: {e}")
            return False

    async def communicate(self, target_agent: 'BaseAgent', message: str, **kwargs) -> Dict[str, Any]:
        """
        Send a message to another agent.

        Args:
            target_agent: The agent to communicate with
            message: The message content
            **kwargs: Additional communication parameters

        Returns:
            Dict containing response and metadata
        """
        try:
            # Record outgoing message
            outgoing_msg = {
                'from': self.name,
                'to': target_agent.name,
                'message': message,
                'timestamp': asyncio.get_event_loop().time(),
                'type': 'outgoing'
            }
            self.message_history.append(outgoing_msg)

            # Send message using AutoGen
            response = await self.a_send(message, target_agent, **kwargs)

            # Record incoming response
            incoming_msg = {
                'from': target_agent.name,
                'to': self.name,
                'message': response,
                'timestamp': asyncio.get_event_loop().time(),
                'type': 'incoming'
            }
            self.message_history.append(incoming_msg)

            self.logger.info(f"Communicated with {target_agent.name}")
            return {'response': response, 'success': True}

        except Exception as e:
            self.logger.error(f"Communication failed with {target_agent.name}: {e}")
            return {'response': None, 'success': False, 'error': str(e)}

    async def request_consensus(self, proposal: Any, required_votes: Optional[int] = None) -> Dict[str, Any]:
        """
        Request consensus from other agents on a proposal.

        Args:
            proposal: The proposal to vote on
            required_votes: Minimum votes required (default: majority)

        Returns:
            Dict containing consensus result
        """
        if not self.coordinator:
            return {'consensus': False, 'error': 'No coordinator available'}

        try:
            result = await self.coordinator.request_consensus(self, proposal, required_votes)
            return result
        except Exception as e:
            self.logger.error(f"Consensus request failed: {e}")
            return {'consensus': False, 'error': str(e)}

    def update_state(self, key: str, value: Any):
        """Update agent state."""
        self.state[key] = value
        self.logger.debug(f"State updated: {key} = {value}")

    def get_state(self, key: str) -> Any:
        """Get agent state value."""
        return self.state.get(key)

    async def perform_task(self, task_description: str, **kwargs) -> Dict[str, Any]:
        """
        Perform the agent's primary task.

        Args:
            task_description: Description of the task to perform
            **kwargs: Task-specific parameters

        Returns:
            Dict containing task results
        """
        try:
            self.logger.info(f"Starting task: {task_description}")
            result = await self._execute_task(task_description, **kwargs)
            self.logger.info(f"Task completed: {task_description}")
            return {'result': result, 'success': True}
        except Exception as e:
            self.logger.error(f"Task failed: {task_description} - {e}")
            return {'result': None, 'success': False, 'error': str(e)}

    @abstractmethod
    async def _execute_task(self, task_description: str, **kwargs) -> Any:
        """
        Abstract method to be implemented by specialized agents.

        Args:
            task_description: Description of the task
            **kwargs: Task parameters

        Returns:
            Task execution result
        """
        pass

    def get_capabilities(self) -> List[str]:
        """Get list of agent capabilities."""
        return self.capabilities

    def get_dependencies(self) -> List[str]:
        """Get list of agent dependencies."""
        return self.dependencies

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the agent.

        Returns:
            Dict containing health status
        """
        return {
            'agent': self.name,
            'active': self.is_active,
            'capabilities': len(self.capabilities),
            'state_size': len(self.state),
            'message_count': len(self.message_history),
            'last_activity': self.message_history[-1]['timestamp'] if self.message_history else None
        }


# Forward declaration for type hints
from .meta_coordinator_agent import MetaCoordinatorAgent