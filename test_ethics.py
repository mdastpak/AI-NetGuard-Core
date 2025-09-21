#!/usr/bin/env python3
"""
Test script for Ethics & Compliance functionality in EthicsAgent
"""

import asyncio
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from framework.agent_system import get_agent_system


async def test_ethics_functionality():
    """Test Ethics & Compliance functionality."""
    print("🛡️  Testing Ethics & Compliance functionality")
    print("=" * 50)

    try:
        # Initialize the agent system
        print("📋 Initializing agent system...")
        agent_system = await get_agent_system()

        # Get EthicsAgent
        ethics_agent = agent_system.get_agent("EthicsAgent")
        if not ethics_agent:
            print("❌ EthicsAgent not found")
            return False

        print("✅ EthicsAgent found")

        # Generate mock data for testing
        print("\n📊 Generating mock test data...")
        np.random.seed(42)
        n_samples = 1000

        # Mock predictions and targets
        predictions = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        targets = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])

        # Mock sensitive features (gender, age group)
        sensitive_features = {
            'gender': np.random.choice(['male', 'female'], size=n_samples, p=[0.5, 0.5]),
            'age_group': np.random.choice(['young', 'middle', 'senior'], size=n_samples, p=[0.4, 0.4, 0.2])
        }

        print(f"   📈 Generated {n_samples} samples with predictions and sensitive features")

        # Test bias assessment
        print("\n⚖️  Testing bias assessment...")
        bias_result = await ethics_agent.perform_task("assess_bias",
                                                     predictions=predictions.tolist(),
                                                     targets=targets.tolist(),
                                                     sensitive_features=sensitive_features)
        print(f"✅ Bias assessment result: {bias_result.get('success', False)}")
        if bias_result.get('success'):
            data = bias_result.get('result', {})
            print(f"   📊 Bias score: {data.get('bias_score', 0):.3f}")
            print(f"   🚨 Bias detected: {data.get('bias_detected', False)}")
            print(f"   💡 Recommendations: {len(data.get('recommendations', []))}")

        # Test fairness assessment
        print("\n⚖️  Testing fairness assessment...")
        fairness_result = await ethics_agent.perform_task("assess_fairness",
                                                         predictions=predictions.tolist(),
                                                         targets=targets.tolist(),
                                                         sensitive_features=sensitive_features)
        print(f"✅ Fairness assessment result: {fairness_result.get('success', False)}")
        if fairness_result.get('success'):
            data = fairness_result.get('result', {})
            print(f"   📊 Fairness score: {data.get('fairness_score', 0):.3f}")
            print(f"   ✅ Fairness achieved: {data.get('fairness_achieved', False)}")
            print(f"   🔧 Improvement needed: {data.get('improvement_needed', False)}")

        # Test bias mitigation
        print("\n🛠️  Testing bias mitigation...")
        mitigation_result = await ethics_agent.perform_task("bias_mitigation",
                                                           model="mock_model",
                                                           training_data={"size": n_samples},
                                                           sensitive_features=sensitive_features)
        print(f"✅ Bias mitigation result: {mitigation_result.get('success', False)}")
        if mitigation_result.get('success'):
            data = mitigation_result.get('result', {})
            print(f"   🎯 Recommended strategy: {data.get('recommended_strategy', 'N/A')}")
            print(f"   📈 Expected improvement: {data.get('expected_improvement', 0):.3f}")
            print(f"   📋 Strategies available: {len(data.get('mitigation_strategies', []))}")

        # Test ethical decision making
        print("\n🤔 Testing ethical decision making...")
        scenario = "Resource allocation in anomaly detection"
        options = ["Prioritize accuracy", "Prioritize fairness", "Balance both", "Minimize false positives"]

        decision_result = await ethics_agent.perform_task("ethical_decision",
                                                         scenario=scenario,
                                                         options=options)
        print(f"✅ Ethical decision result: {decision_result.get('success', False)}")
        if decision_result.get('success'):
            data = decision_result.get('result', {})
            print(f"   🏆 Selected option: {data.get('decision', 'N/A')}")
            print(f"   📊 Ethical score: {data.get('ethical_score', 0):.3f}")
            print(f"   🧠 Framework: {data.get('ethical_framework', 'N/A')}")

        # Test continuous monitoring
        print("\n📊 Testing continuous ethics monitoring...")
        monitoring_result = await ethics_agent.perform_task("continuous_monitoring",
                                                           monitoring_duration=1800,  # 30 minutes
                                                           check_interval=300)  # 5 minutes
        print(f"✅ Continuous monitoring result: {monitoring_result.get('success', False)}")
        if monitoring_result.get('success'):
            data = monitoring_result.get('result', {})
            print(f"   ⏱️  Monitoring duration: {data.get('monitoring_duration', 0)} seconds")
            print(f"   🔍 Checks performed: {data.get('checks_performed', 0)}")
            print(f"   📊 Average bias score: {data.get('average_bias_score', 0):.3f}")
            print(f"   ⚖️  Average fairness score: {data.get('average_fairness_score', 0):.3f}")
            print(f"   🚨 Alerts triggered: {data.get('alerts_triggered', 0)}")

        # Shutdown system
        print("\n🛑 Shutting down agent system...")
        shutdown_success = await agent_system.shutdown_system()
        print(f"✅ Shutdown successful: {shutdown_success}")

        print("\n🎉 Ethics & Compliance test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_ethics_functionality())
    sys.exit(0 if success else 1)