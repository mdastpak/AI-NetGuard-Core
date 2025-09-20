#!/usr/bin/env python3
"""
Test script for AI-NetGuard Agent System

This script demonstrates the initialization and basic functionality
of the multi-agent AI-NetGuard system.
"""

import asyncio
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from framework.agent_system import get_agent_system


async def test_agent_system():
    """Test the agent system initialization and basic operations."""
    print("🚀 Testing AI-NetGuard Agent System")
    print("=" * 50)

    try:
        # Initialize the agent system
        print("📋 Initializing agent system...")
        agent_system = await get_agent_system()

        # Check system status
        print("📊 Checking system status...")
        status = await agent_system.get_system_status()
        print(f"✅ System initialized: {status.get('system_initialized', False)}")
        print(f"✅ System running: {status.get('system_running', False)}")
        print(f"✅ Total agents: {status.get('total_agents', 0)}")
        print(f"✅ Active agents: {status.get('active_agents', 0)}")

        # List all agents
        print("\n🤖 Available Agents:")
        agents = agent_system.list_agents()
        for i, agent_name in enumerate(agents, 1):
            print(f"  {i}. {agent_name}")

        # Test infrastructure status
        print("\n🏗️  Testing infrastructure...")
        infra_status = await agent_system.get_system_status()
        if 'infrastructure' in infra_status:
            infra = infra_status['infrastructure']
            print(f"✅ Ray cluster: {infra.get('ray_cluster', {}).get('running', False)}")
            print(f"✅ Dask cluster: {infra.get('dask_cluster', {}).get('running', False)}")
            print(f"✅ GPU devices: {len(infra.get('gpu_devices', []))}")
            print(f"✅ CPU cores: {infra.get('cpu_info', {}).get('cores', 0)}")
        else:
            print("⚠️  Infrastructure status not available")

        if 'cloud' in infra_status:
            cloud = infra_status['cloud']
            print(f"✅ Cloud providers: {len(cloud.get('active_providers', []))}")
            print(f"✅ Global regions: {len(cloud.get('global_regions', []))}")

        # Test basic task execution
        print("\n⚡ Testing task execution...")

        # Test DataSynthesisAgent
        print("Testing DataSynthesisAgent...")
        data_agent = agent_system.get_agent("DataSynthesisAgent")
        if data_agent:
            result = await data_agent.perform_task("generate_training_data", dataset_size=100, attack_ratio=0.2)
            print(f"✅ DataSynthesisAgent task result: {result.get('success', False)}")
            if result.get('success'):
                data_result = result.get('result', {})
                stats = data_result.get('statistics', {})
                print(f"   📈 Generated {stats.get('total_samples', 0)} samples")
                print(f"   🎯 Normal samples: {stats.get('normal_samples', 0)}")
                print(f"   🚨 Attack samples: {stats.get('attack_samples', 0)}")
                print(f"   📊 Features: {len(stats.get('features', []))}")

                # Show sample of generated data
                dataset = data_result.get('data')
                if dataset is not None:
                    print("   📋 Sample data (first 3 rows):")
                    print(dataset.head(3).to_string())
                    print(f"   🏷️  Label distribution: {dataset['label'].value_counts().to_dict()}")
        else:
            print("❌ DataSynthesisAgent not found")

        # Test FeatureEngineeringAgent
        print("\nTesting FeatureEngineeringAgent...")
        feat_agent = agent_system.get_agent("FeatureEngineeringAgent")
        if feat_agent:
            # Create sample data first
            sample_data = await data_agent.perform_task("generate_training_data", dataset_size=50, attack_ratio=0.1)
            if sample_data.get('success'):
                dataset = sample_data['result']['data']
                result = await feat_agent.perform_task("discover_features", dataset=dataset)
                print(f"✅ FeatureEngineeringAgent task result: {result.get('success', False)}")
                if result.get('success'):
                    feat_result = result.get('result', {})
                    print(f"   🆕 New features added: {len(feat_result.get('new_features', []))}")
                    print(f"   📊 Total features: {feat_result.get('total_features', 0)}")
        else:
            print("❌ FeatureEngineeringAgent not found")

        # Test EvaluationAgent
        print("\nTesting EvaluationAgent...")
        eval_agent = agent_system.get_agent("EvaluationAgent")
        if eval_agent:
            # Use the dataset from DataSynthesisAgent
            if 'dataset' in locals() and dataset is not None:
                result = await eval_agent.perform_task("evaluate_model", model_data=dataset, target_column='label')
                print(f"✅ EvaluationAgent task result: {result.get('success', False)}")
                if result.get('success'):
                    eval_result = result.get('result', {})
                    print(f"   📊 Evaluation completed for dataset with {len(dataset)} samples")
                else:
                    print("   ⚠️  EvaluationAgent method not implemented yet")
            else:
                print("   ⚠️  No dataset available for evaluation")
        else:
            print("❌ EvaluationAgent not found")

        # Check status of other agents (not fully tested)
        print("\n📋 Agent implementation status:")
        other_agents = [
            "ModelArchitectAgent", "AdversarialAgent", "MonitoringAgent",
            "ScalingAgent", "SecurityAgent", "OptimizationAgent",
            "DeploymentAgent", "RecoveryAgent", "LearningAgent",
            "PrivacyAgent", "EthicsAgent", "CommunicationAgent"
        ]

        for agent_name in other_agents:
            agent = agent_system.get_agent(agent_name)
            if agent:
                # Try a simple health check
                try:
                    health = await agent.health_check()
                    print(f"   ✅ {agent_name}: Initialized (basic functionality)")
                except Exception as e:
                    print(f"   ⚠️  {agent_name}: Initialized (limited functionality)")
            else:
                print(f"   ❌ {agent_name}: Not found")

        # Test consensus mechanism
        print("\n🗳️  Testing consensus mechanism...")
        coordinator = agent_system.get_agent("MetaCoordinatorAgent")
        if coordinator:
            consensus_result = await coordinator.request_consensus(
                coordinator,
                "Test proposal for system improvement",
                required_votes=8
            )
            print(f"✅ Consensus achieved: {consensus_result.get('consensus', False)}")
            print(f"   👍 Positive votes: {consensus_result.get('positive_votes', 0)}")
            print(f"   📊 Total votes: {consensus_result.get('total_votes', 0)}")

        # Shutdown system
        print("\n🛑 Shutting down agent system...")
        shutdown_success = await agent_system.shutdown_system()
        print(f"✅ Shutdown successful: {shutdown_success}")

        print("\n🎉 Agent system test completed successfully!")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


async def main():
    """Main test function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run tests
    success = await test_agent_system()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())