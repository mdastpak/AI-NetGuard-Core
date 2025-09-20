"""
AI-NetGuard Agent System Framework

This module provides the main framework for initializing and managing
the multi-agent AI-NetGuard system with AutoGen integration.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from autogen import GroupChat, GroupChatManager

# Import all agents
from agents.meta_coordinator_agent import MetaCoordinatorAgent
from agents.data_synthesis_agent import DataSynthesisAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.model_architect_agent import ModelArchitectAgent
from agents.adversarial_agent import AdversarialAgent
from agents.monitoring_agent import MonitoringAgent
from agents.scaling_agent import ScalingAgent
from agents.security_agent import SecurityAgent
from agents.optimization_agent import OptimizationAgent
from agents.evaluation_agent import EvaluationAgent
from agents.deployment_agent import DeploymentAgent
from agents.recovery_agent import RecoveryAgent
from agents.learning_agent import LearningAgent
from agents.privacy_agent import PrivacyAgent
from agents.ethics_agent import EthicsAgent
from agents.communication_agent import CommunicationAgent

# Import infrastructure managers
from infrastructure.distributed_manager import get_infrastructure_manager
from infrastructure.cloud_manager import get_cloud_manager

# Import foundation model manager
from models.foundation_model_manager import get_foundation_model_manager


class AINetGuardAgentSystem:
    """
    Main agent system class that manages all AI-NetGuard agents.

    Provides initialization, coordination, and communication between agents.
    """

    def __init__(self):
        self.logger = logging.getLogger("AINetGuardAgentSystem")
        self.logger.setLevel(logging.INFO)

        # Agent instances
        self.coordinator: Optional[MetaCoordinatorAgent] = None
        self.agents: Dict[str, Any] = {}
        self.group_chat: Optional[GroupChat] = None
        self.group_chat_manager: Optional[GroupChatManager] = None

        # Infrastructure managers
        self.distributed_manager = None
        self.cloud_manager = None
        self.foundation_model_manager = None

        # System state
        self.is_initialized = False
        self.is_running = False

    async def initialize_system(self) -> bool:
        """
        Initialize the complete AI-NetGuard agent system.

        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("Initializing AI-NetGuard Agent System...")

            # Create meta-coordinator first
            self.coordinator = MetaCoordinatorAgent()
            await self.coordinator.activate()

            # Create all specialized agents
            agent_classes = {
                'DataSynthesisAgent': DataSynthesisAgent,
                'FeatureEngineeringAgent': FeatureEngineeringAgent,
                'ModelArchitectAgent': ModelArchitectAgent,
                'AdversarialAgent': AdversarialAgent,
                'MonitoringAgent': MonitoringAgent,
                'ScalingAgent': ScalingAgent,
                'SecurityAgent': SecurityAgent,
                'OptimizationAgent': OptimizationAgent,
                'EvaluationAgent': EvaluationAgent,
                'DeploymentAgent': DeploymentAgent,
                'RecoveryAgent': RecoveryAgent,
                'LearningAgent': LearningAgent,
                'PrivacyAgent': PrivacyAgent,
                'EthicsAgent': EthicsAgent,
                'CommunicationAgent': CommunicationAgent
            }

            # Initialize each agent
            for agent_name, agent_class in agent_classes.items():
                agent = agent_class(coordinator_agent=self.coordinator)
                self.agents[agent_name] = agent
                await agent.activate()
                self.logger.info(f"Initialized {agent_name}")

            # Initialize infrastructure managers
            self.logger.info("Initializing infrastructure managers...")
            self.distributed_manager = await get_infrastructure_manager()
            self.cloud_manager = await get_cloud_manager()

            # Initialize foundation model manager
            self.logger.info("Initializing foundation model manager...")
            self.foundation_model_manager = await get_foundation_model_manager()

            # Set up group chat for AutoGen integration
            await self._setup_group_chat()

            self.is_initialized = True
            self.logger.info("AI-NetGuard Agent System initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize agent system: {e}")
            return False

    async def _setup_group_chat(self):
        """Set up AutoGen GroupChat for agent communication."""
        try:
            # Create list of all agents for group chat
            group_chat_agents = [self.coordinator] + list(self.agents.values())

            # Create group chat
            self.group_chat = GroupChat(
                agents=group_chat_agents,
                messages=[],
                max_round=50,
                speaker_selection_method="round_robin",
                allow_repeat_speaker=False
            )

            # Create group chat manager
            self.group_chat_manager = GroupChatManager(
                groupchat=self.group_chat,
                name="AINetGuard_GroupChatManager"
            )

            self.logger.info("Group chat setup completed")

        except Exception as e:
            self.logger.error(f"Failed to setup group chat: {e}")

    async def start_system(self) -> bool:
        """
        Start the agent system operations.

        Returns:
            bool: True if system started successfully
        """
        if not self.is_initialized:
            self.logger.error("System not initialized")
            return False

        try:
            self.is_running = True
            self.logger.info("AI-NetGuard Agent System started")

            # Start monitoring and coordination
            await self._start_background_tasks()

            return True

        except Exception as e:
            self.logger.error(f"Failed to start system: {e}")
            return False

    async def _start_background_tasks(self):
        """Start background monitoring and coordination tasks."""
        # This would start periodic health checks, consensus monitoring, etc.
        # For now, just log that background tasks would start
        self.logger.info("Background tasks initialized")

    async def execute_task(self, task_description: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a task using the agent system.

        Args:
            task_description: Description of the task
            **kwargs: Task parameters

        Returns:
            Dict with execution results
        """
        if not self.is_running:
            return {'error': 'System not running'}

        try:
            # Route task to appropriate agent via coordinator
            result = await self.coordinator.perform_task(task_description, **kwargs)
            return result

        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            return {'error': str(e)}

    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status.

        Returns:
            Dict with system status information
        """
        if not self.coordinator:
            return {'error': 'System not initialized'}

        try:
            status = await self.coordinator.get_system_status()
            # Add infrastructure status
            if self.distributed_manager:
                infra_status = await self.distributed_manager.get_resource_status()
                status['infrastructure'] = infra_status

            if self.cloud_manager:
                cloud_status = await self.cloud_manager.get_global_status()
                status['cloud'] = cloud_status

            # Add foundation models status
            if self.foundation_model_manager:
                model_status = await self.foundation_model_manager.get_model_status()
                status['foundation_models'] = model_status

            status.update({
                'system_initialized': self.is_initialized,
                'system_running': self.is_running,
                'total_agents': len(self.agents) + 1,  # +1 for coordinator
                'group_chat_active': self.group_chat is not None
            })
            return status

        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}

    async def shutdown_system(self) -> bool:
        """
        Shutdown the agent system gracefully.

        Returns:
            bool: True if shutdown successful
        """
        try:
            self.logger.info("Shutting down AI-NetGuard Agent System...")

            self.is_running = False

            # Shutdown foundation models
            if self.foundation_model_manager:
                await self.foundation_model_manager.shutdown_models()

            # Shutdown infrastructure managers
            if self.cloud_manager:
                await self.cloud_manager.shutdown_cloud_infrastructure()

            if self.distributed_manager:
                await self.distributed_manager.shutdown_infrastructure()

            # Deactivate all agents
            for agent in self.agents.values():
                await agent.deactivate()

            if self.coordinator:
                await self.coordinator.deactivate()

            self.logger.info("AI-NetGuard Agent System shutdown completed")
            return True

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            return False

    def get_agent(self, agent_name: str) -> Optional[Any]:
        """
        Get a specific agent by name.

        Args:
            agent_name: Name of the agent

        Returns:
            Agent instance or None if not found
        """
        if agent_name == "MetaCoordinatorAgent":
            return self.coordinator
        return self.agents.get(agent_name)

    def list_agents(self) -> List[str]:
        """
        List all available agents.

        Returns:
            List of agent names
        """
        agents = ["MetaCoordinatorAgent"] + list(self.agents.keys())
        return agents


# Global system instance
_agent_system: Optional[AINetGuardAgentSystem] = None


async def get_agent_system() -> AINetGuardAgentSystem:
    """
    Get or create the global agent system instance.

    Returns:
        AINetGuardAgentSystem instance
    """
    global _agent_system
    if _agent_system is None:
        _agent_system = AINetGuardAgentSystem()
        await _agent_system.initialize_system()
    return _agent_system


def get_agent_system_sync() -> AINetGuardAgentSystem:
    """
    Synchronous wrapper to get agent system.
    Note: This creates a new event loop if none exists.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, we need to handle differently
            # For now, return existing instance or create new
            global _agent_system
            if _agent_system is None:
                _agent_system = AINetGuardAgentSystem()
                # Note: Can't await here if loop is running
                # Would need to handle this differently in production
            return _agent_system
        else:
            return loop.run_until_complete(get_agent_system())
    except RuntimeError:
        # No event loop, create new one
        return asyncio.run(get_agent_system())