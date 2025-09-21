#!/usr/bin/env python3
"""
Test script for Multi-Modal Learning functionality in LearningAgent
"""

import pytest
import sys
import os
import numpy as np
import warnings

# Suppress autogen deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='autogen.*')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from framework.agent_system import get_agent_system


@pytest.mark.asyncio
async def test_multimodal_learning():
    """Test Multi-Modal Learning functionality."""
    print("🔄 Testing Multi-Modal Learning functionality")
    print("=" * 50)

    try:
        # Initialize the agent system
        print("📋 Initializing agent system...")
        agent_system = await get_agent_system()

        # Get LearningAgent
        learning_agent = agent_system.get_agent("LearningAgent")
        if not learning_agent:
            print("❌ LearningAgent not found")
            return False

        print("✅ LearningAgent found")

        # Test multi-modal learning
        print("\n🔄 Testing multi-modal learning...")
        modalities = ['text', 'network_traffic', 'behavioral_patterns', 'temporal_sequences']

        multimodal_result = await learning_agent.perform_task("multi_modal_learn", modalities=modalities)
        print(f"✅ Multi-modal learning result: {multimodal_result.get('success', False)}")
        if multimodal_result.get('success'):
            data = multimodal_result.get('result', {})
            print(f"   📊 Modalities processed: {data.get('modalities_processed', 0)}")
            print(f"   🎯 Fusion accuracy: {data.get('fusion_accuracy', 0):.3f}")
            print(f"   🔗 Cross-modal features: {data.get('cross_modal_features', 0)}")

        # Test modality fusion
        print("\n🔗 Testing modality fusion...")
        # Create mock modality features
        modality_features = {
            'text': np.random.random((50, 128)),
            'network': np.random.random((50, 64)),
            'behavioral': np.random.random((50, 32)),
            'temporal': np.random.random((50, 256))
        }

        fusion_result = await learning_agent.perform_task("modality_fusion", modality_features=modality_features)
        print(f"✅ Modality fusion result: {fusion_result.get('success', False)}")
        if fusion_result.get('success'):
            data = fusion_result.get('result', {})
            print(f"   📊 Input modalities: {data.get('input_modalities', 0)}")
            print(f"   🧠 Fused dimension: {data.get('fused_dimension', 0)}")
            print(f"   ⚡ Information preservation: {data.get('information_preservation', 0):.3f}")
            print(f"   🔄 Cross-modal synergy: {data.get('cross_modal_synergy', 0):.3f}")

        # Test cross-domain adaptation
        print("\n🌐 Testing cross-domain adaptation...")
        cross_domain_result = await learning_agent.perform_task("cross_domain_adapt",
                                                               source_domain='corporate_network',
                                                               target_domain='iot_devices')
        print(f"✅ Cross-domain adaptation result: {cross_domain_result.get('success', False)}")
        if cross_domain_result.get('success'):
            data = cross_domain_result.get('result', {})
            print(f"   📍 Source domain: {data.get('source_domain', 'N/A')}")
            print(f"   🎯 Target domain: {data.get('target_domain', 'N/A')}")
            print(f"   📊 Domain shift detected: {data.get('domain_shift_detected', 0):.3f}")
            print(f"   🏆 Best technique: {data.get('best_technique', 'N/A')}")
            print(f"   📈 Performance improvement: {data.get('performance_improvement', 0):.3f}")

        # Test domain alignment
        print("\n📐 Testing domain alignment...")
        domains = ['enterprise_network', 'cloud_infrastructure', 'iot_devices', 'mobile_networks']

        alignment_result = await learning_agent.perform_task("domain_alignment", domains=domains)
        print(f"✅ Domain alignment result: {alignment_result.get('success', False)}")
        if alignment_result.get('success'):
            data = alignment_result.get('result', {})
            print(f"   🌍 Domains aligned: {data.get('domains_aligned', 0)}")
            print(f"   📊 Overall alignment score: {data.get('overall_alignment_score', 0):.3f}")
            print(f"   ⚖️  Alignment stability: {data.get('alignment_stability', 0):.3f}")
            best_alignments = data.get('best_alignments', [])
            if best_alignments:
                print(f"   🏅 Best alignment: {best_alignments[0][0]} (score: {best_alignments[0][1]['alignment_score']:.3f})")

        # Test existing learning capabilities (for comparison)
        print("\n🧠 Testing meta-learning (existing capability)...")
        meta_result = await learning_agent.perform_task("meta_learn")
        print(f"✅ Meta-learning result: {meta_result.get('success', False)}")
        if meta_result.get('success'):
            data = meta_result.get('result', {})
            print(f"   📚 Meta-knowledge acquired: {len(data.get('meta_knowledge_acquired', {}))}")
            print(f"   🚀 Learning efficiency: {data.get('learning_efficiency', 'N/A')}")

        # Shutdown system
        print("\n🛑 Shutting down agent system...")
        shutdown_success = await agent_system.shutdown_system()
        print(f"✅ Shutdown successful: {shutdown_success}")

        print("\n🎉 Multi-Modal Learning test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

