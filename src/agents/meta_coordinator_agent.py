"""
Meta Coordinator Agent

Responsible for overseeing and coordinating all agents in the AI-NetGuard
system using consensus-based decision making.
"""

from typing import Dict, List, Any, Optional
from .base_agent import BaseAgent


class MetaCoordinatorAgent(BaseAgent):
    """
    Meta-coordinator agent that oversees all other agents and implements
    consensus-based coordination mechanisms.
    """

    def __init__(self, **kwargs):
        system_message = """
        You are the MetaCoordinatorAgent, the central coordinator for all
        AI-NetGuard agents. Your responsibilities include:

        1. Overseeing agent registration and coordination
        2. Implementing consensus-based decision making
        3. Managing inter-agent communication
        4. Ensuring system-wide coherence and performance
        5. Handling conflict resolution and optimization

        You maintain ultimate authority over system coordination while
        respecting agent autonomy.
        """

        super().__init__(
            name="MetaCoordinatorAgent",
            system_message=system_message,
            coordinator_agent=None,  # This is the coordinator
            **kwargs
        )

        self.capabilities = [
            "agent_coordination",
            "consensus_management",
            "conflict_resolution",
            "system_oversight",
            "decision_synthesis"
        ]

        self.dependencies = []  # Coordinator has no dependencies

        # Agent registry
        self.registered_agents: Dict[str, BaseAgent] = {}
        self.active_consensus: Dict[str, Dict[str, Any]] = {}

    async def register_agent(self, agent: BaseAgent) -> bool:
        """
        Register a new agent with the coordinator.

        Args:
            agent: The agent to register

        Returns:
            bool: True if registration successful
        """
        try:
            self.registered_agents[agent.name] = agent
            self.logger.info(f"Registered agent: {agent.name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register agent {agent.name}: {e}")
            return False

    async def unregister_agent(self, agent: BaseAgent) -> bool:
        """
        Unregister an agent from the coordinator.

        Args:
            agent: The agent to unregister

        Returns:
            bool: True if unregistration successful
        """
        try:
            if agent.name in self.registered_agents:
                del self.registered_agents[agent.name]
                self.logger.info(f"Unregistered agent: {agent.name}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to unregister agent {agent.name}: {e}")
            return False

    async def request_consensus(self, requesting_agent: BaseAgent, proposal: Any,
                               required_votes: Optional[int] = None) -> Dict[str, Any]:
        """
        Request consensus from registered agents on a proposal.

        Args:
            requesting_agent: The agent making the request
            proposal: The proposal to vote on
            required_votes: Minimum votes required (default: majority)

        Returns:
            Dict containing consensus result
        """
        try:
            consensus_id = f"consensus_{len(self.active_consensus)}"

            # Initialize consensus
            total_agents = len(self.registered_agents)
            if required_votes is None:
                required_votes = (total_agents // 2) + 1  # Majority

            self.active_consensus[consensus_id] = {
                'proposal': proposal,
                'requesting_agent': requesting_agent.name,
                'votes': {},
                'required_votes': required_votes,
                'total_agents': total_agents,
                'status': 'active'
            }

            # Collect votes from all agents
            votes = {}
            for agent_name, agent in self.registered_agents.items():
                if agent_name != requesting_agent.name:  # Don't vote on own proposal
                    vote = await self._get_agent_vote(agent, proposal)
                    votes[agent_name] = vote

            # Include requesting agent's vote
            votes[requesting_agent.name] = True  # Auto-approve own proposal

            # Determine consensus
            positive_votes = sum(1 for vote in votes.values() if vote)
            consensus_achieved = positive_votes >= required_votes

            result = {
                'consensus_id': consensus_id,
                'consensus': consensus_achieved,
                'positive_votes': positive_votes,
                'total_votes': len(votes),
                'required_votes': required_votes,
                'votes': votes
            }

            self.active_consensus[consensus_id]['status'] = 'completed'
            self.logger.info(f"Consensus {consensus_id}: {consensus_achieved} ({positive_votes}/{len(votes)})")

            return result

        except Exception as e:
            self.logger.error(f"Consensus request failed: {e}")
            return {'consensus': False, 'error': str(e)}

    async def _get_agent_vote(self, agent: BaseAgent, proposal: Any) -> bool:
        """
        Get a vote from an agent on a proposal.

        Args:
            agent: The agent to vote
            proposal: The proposal

        Returns:
            bool: The agent's vote
        """
        try:
            # In practice, this would involve agent communication
            # For now, simulate voting based on agent type
            vote = True  # Default to approve
            return vote
        except Exception as e:
            self.logger.error(f"Failed to get vote from {agent.name}: {e}")
            return False

    async def broadcast_message(self, message: str, sender: BaseAgent,
                               target_agents: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Broadcast a message to multiple agents.

        Args:
            message: The message to broadcast
            sender: The sending agent
            target_agents: List of target agent names (None for all)

        Returns:
            Dict with broadcast results
        """
        try:
            targets = target_agents if target_agents else list(self.registered_agents.keys())
            results = {}

            for agent_name in targets:
                if agent_name in self.registered_agents and agent_name != sender.name:
                    agent = self.registered_agents[agent_name]
                    result = await sender.communicate(agent, message)
                    results[agent_name] = result

            return {
                'broadcast_success': True,
                'targets_reached': len(results),
                'results': results
            }

        except Exception as e:
            self.logger.error(f"Broadcast failed: {e}")
            return {'broadcast_success': False, 'error': str(e)}

    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall system status from all agents.

        Returns:
            Dict with system status
        """
        try:
            agent_statuses = {}
            for agent_name, agent in self.registered_agents.items():
                status = await agent.health_check()
                agent_statuses[agent_name] = status

            overall_status = {
                'total_agents': len(self.registered_agents),
                'active_agents': sum(1 for s in agent_statuses.values() if s.get('active', False)),
                'agent_statuses': agent_statuses,
                'active_consensus': len(self.active_consensus),
                'system_health': 'healthy' if all(s.get('active', False) for s in agent_statuses.values()) else 'degraded'
            }

            return overall_status

        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}

    async def _execute_task(self, task_description: str, **kwargs) -> Any:
        """Execute coordinator tasks."""
        if "coordinate_agents" in task_description.lower():
            return await self._coordinate_agents(**kwargs)
        elif "resolve_conflict" in task_description.lower():
            return await self._resolve_conflict(**kwargs)
        else:
            return await self._general_coordination(task_description, **kwargs)

    async def _coordinate_agents(self, task: str, **kwargs) -> Dict[str, Any]:
        """Coordinate agents for a specific task."""
        self.logger.info(f"Coordinating agents for task: {task}")

        # Broadcast task to relevant agents
        await self.broadcast_message(f"Task coordination: {task}", self)

        return {
            'coordination_task': task,
            'agents_notified': len(self.registered_agents),
            'status': 'coordinated'
        }

    async def _resolve_conflict(self, conflict_description: str, **kwargs) -> Dict[str, Any]:
        """Resolve conflicts between agents."""
        self.logger.info(f"Resolving conflict: {conflict_description}")

        # Request consensus on conflict resolution
        resolution_proposal = f"Resolve conflict: {conflict_description}"
        consensus = await self.request_consensus(self, resolution_proposal)

        return {
            'conflict': conflict_description,
            'resolution': consensus.get('consensus', False),
            'consensus_details': consensus
        }

    async def _general_coordination(self, task_description: str, **kwargs) -> Dict[str, Any]:
        """Handle general coordination tasks."""
        return {
            'task': task_description,
            'coordination_type': 'general',
            'status': 'completed'
        }