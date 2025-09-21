#!/usr/bin/env python3
"""
Test script for Edge Computing functionality in ScalingAgent
"""

import asyncio
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from framework.agent_system import get_agent_system


async def test_edge_computing():
    """Test Edge Computing functionality."""
    print("🌐 Testing Edge Computing functionality")
    print("=" * 50)

    try:
        # Initialize the agent system
        print("📋 Initializing agent system...")
        agent_system = await get_agent_system()

        # Get ScalingAgent
        scaling_agent = agent_system.get_agent("ScalingAgent")
        if not scaling_agent:
            print("❌ ScalingAgent not found")
            return False

        print("✅ ScalingAgent found")

        # Test edge computing deployment
        print("\n🏭 Testing edge computing deployment...")
        edge_result = await scaling_agent.perform_task("edge_computing", num_edge_nodes=25)
        print(f"✅ Edge computing result: {edge_result.get('success', False)}")
        if edge_result.get('success'):
            data = edge_result.get('result', {})
            print(f"   🌍 Edge nodes deployed: {data.get('edge_nodes_deployed', 0)}")
            print(f"   💪 Total capacity: {data.get('total_capacity', 0)}")
            print(f"   ⚡ Average latency: {data.get('average_latency', 0):.2f}ms")
            print(f"   ✅ Active nodes: {data.get('active_nodes', 0)}")

        # Test distributed intelligence
        print("\n🧠 Testing distributed intelligence...")
        intelligence_tasks = ['threat_detection', 'anomaly_scoring', 'pattern_recognition', 'behavior_analysis']

        distributed_result = await scaling_agent.perform_task("distributed_intelligence", intelligence_tasks=intelligence_tasks)
        print(f"✅ Distributed intelligence result: {distributed_result.get('success', False)}")
        if distributed_result.get('success'):
            data = distributed_result.get('result', {})
            print(f"   📋 Tasks distributed: {data.get('tasks_distributed', 0)}")
            print(f"   🌐 Edge nodes utilized: {data.get('total_edge_nodes_utilized', 0)}")
            print(f"   ☁️  Cloud nodes utilized: {data.get('total_cloud_nodes_utilized', 0)}")
            print(f"   ⚡ Average processing latency: {data.get('average_processing_latency', 0):.2f}ms")
            print(f"   🎯 Overall accuracy: {data.get('overall_accuracy', 0):.3f}")

        # Test low-latency detection
        print("\n⚡ Testing low-latency detection...")
        detection_scenarios = ['real_time_traffic', 'behavioral_anomalies', 'signature_matching', 'ai_based_detection']

        latency_result = await scaling_agent.perform_task("low_latency_detection", detection_scenarios=detection_scenarios)
        print(f"✅ Low-latency detection result: {latency_result.get('success', False)}")
        if latency_result.get('success'):
            data = latency_result.get('result', {})
            print(f"   🎯 Scenarios optimized: {data.get('scenarios_optimized', 0)}")
            print(f"   ⚡ Average latency: {data.get('average_latency', 0):.2f}ms")
            print(f"   ✅ Latency targets met: {data.get('latency_targets_met', 0)}")
            print(f"   🚀 Total throughput: {data.get('total_throughput', 0):,} detections/sec")
            print(f"   🎯 Average accuracy: {data.get('average_accuracy', 0):.3f}")

        # Test hierarchical architecture
        print("\n🏗️  Testing hierarchical architecture...")
        hierarchy_levels = ['edge_devices', 'edge_servers', 'regional_hubs', 'cloud_datacenters']

        hierarchy_result = await scaling_agent.perform_task("hierarchical_architecture", hierarchy_levels=hierarchy_levels)
        print(f"✅ Hierarchical architecture result: {hierarchy_result.get('success', False)}")
        if hierarchy_result.get('success'):
            data = hierarchy_result.get('result', {})
            print(f"   🏢 Hierarchy levels: {data.get('hierarchy_levels', 0)}")
            print(f"   🌐 Total nodes: {data.get('total_nodes', 0)}")
            print(f"   🔄 Data flow patterns: {data.get('data_flow_patterns', 0)}")
            print(f"   ⚡ Latency optimization: {data.get('latency_optimization', 0):.1%}")
            print(f"   📈 Scalability factor: {data.get('scalability_factor', 0):.1f}")
            print(f"   🛡️  Fault tolerance: {data.get('fault_tolerance', 0):.1%}")

        # Test existing scaling capabilities (for comparison)
        print("\n📊 Testing predictive scaling (existing capability)...")
        predictive_result = await scaling_agent.perform_task("predictive_scaling")
        print(f"✅ Predictive scaling result: {predictive_result.get('success', False)}")
        if predictive_result.get('success'):
            data = predictive_result.get('result', {})
            print(f"   📈 Predicted load: {data.get('predicted_load', 0):.3f}")
            print(f"   🎯 Scaling recommendation: {data.get('scaling_recommendation', 'N/A')}")
            print(f"   📊 Confidence score: {data.get('confidence_score', 0):.3f}")

        # Shutdown system
        print("\n🛑 Shutting down agent system...")
        shutdown_success = await agent_system.shutdown_system()
        print(f"✅ Shutdown successful: {shutdown_success}")

        print("\n🎉 Edge Computing test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_edge_computing())
    sys.exit(0 if success else 1)