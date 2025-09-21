#!/usr/bin/env python3
"""
Test script for A/B Testing functionality in EvaluationAgent
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from framework.agent_system import get_agent_system


async def test_ab_testing():
    """Test A/B testing functionality."""
    print("🧪 Testing A/B Testing functionality")
    print("=" * 50)

    try:
        # Initialize the agent system
        print("📋 Initializing agent system...")
        agent_system = await get_agent_system()

        # Get EvaluationAgent
        eval_agent = agent_system.get_agent("EvaluationAgent")
        if not eval_agent:
            print("❌ EvaluationAgent not found")
            return False

        print("✅ EvaluationAgent found")

        # Test variant generation
        print("\n🆕 Testing variant generation...")
        variant_result = await eval_agent.perform_task("generate_variants", num_variants=10)
        print(f"✅ Variant generation result: {variant_result.get('success', False)}")
        if variant_result.get('success'):
            data = variant_result.get('result', {})
            print(f"   📊 Variants generated: {data.get('variants_generated', 0)}")
            print(f"   🏷️  Total variants: {data.get('total_variants', 0)}")

        # Test A/B testing
        print("\n🆚 Testing A/B testing...")
        variant_a = {'id': 'variant_A', 'score': 0.85}
        variant_b = {'id': 'variant_B', 'score': 0.87}

        ab_result = await eval_agent.perform_task("run_ab_test", variant_a=variant_a, variant_b=variant_b, sample_size=500)
        print(f"✅ A/B testing result: {ab_result.get('success', False)}")
        if ab_result.get('success'):
            data = ab_result.get('result', {})
            print(f"   📊 Variant A: {data.get('variant_a', 'N/A')}")
            print(f"   📊 Variant B: {data.get('variant_b', 'N/A')}")
            print(f"   📈 Mean A: {data.get('mean_a', 0):.3f}")
            print(f"   📈 Mean B: {data.get('mean_b', 0):.3f}")
            print(f"   📊 Difference: {data.get('difference', 0):.3f}")
            print(f"   🎯 Winner: {data.get('winner', 'tie')}")
            print(f"   📏 Statistically significant: {data.get('statistically_significant', False)}")

        # Test continuous A/B testing
        print("\n🔄 Testing continuous A/B testing...")
        continuous_result = await eval_agent.perform_task("continuous_testing", num_variants=5, iterations=3)
        print(f"✅ Continuous testing result: {continuous_result.get('success', False)}")
        if continuous_result.get('success'):
            data = continuous_result.get('result', {})
            print(f"   🔄 Total iterations: {data.get('total_iterations', 0)}")
            print(f"   🏆 Final best score: {data.get('final_best_score', 0):.3f}")
            print(f"   📈 Total improvement: {data.get('total_improvement', 0):.3f}")

        # Shutdown system
        print("\n🛑 Shutting down agent system...")
        shutdown_success = await agent_system.shutdown_system()
        print(f"✅ Shutdown successful: {shutdown_success}")

        print("\n🎉 A/B Testing test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_ab_testing())
    sys.exit(0 if success else 1)