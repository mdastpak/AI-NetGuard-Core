#!/usr/bin/env python3
"""
Test script for Knowledge Management functionality in CommunicationAgent
"""

import asyncio
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from framework.agent_system import get_agent_system


async def test_knowledge_management():
    """Test Knowledge Management functionality."""
    print("📚 Testing Knowledge Management functionality")
    print("=" * 50)

    try:
        # Initialize the agent system
        print("📋 Initializing agent system...")
        agent_system = await get_agent_system()

        # Get CommunicationAgent
        communication_agent = agent_system.get_agent("CommunicationAgent")
        if not communication_agent:
            print("❌ CommunicationAgent not found")
            return False

        print("✅ CommunicationAgent found")

        # Test AI-generated documentation
        print("\n📝 Testing AI-generated documentation...")
        doc_result = await communication_agent.perform_task("generate_documentation",
                                                          topic="AI-NetGuard Threat Detection",
                                                          content_type="technical_documentation")
        print(f"✅ Documentation generation result: {doc_result.get('success', False)}")
        if doc_result.get('success'):
            data = doc_result.get('result', {})
            print(f"   📖 Topic: {data.get('topic', 'N/A')}")
            print(f"   📄 Content type: {data.get('content_type', 'N/A')}")
            print(f"   📊 Word count: {data.get('word_count', 0)}")
            print(f"   ⭐ Quality score: {data.get('quality_score', 0):.3f}")
            print(f"   📚 Sections: {len(data.get('sections', []))}")

        # Test semantic search
        print("\n🔍 Testing semantic search...")
        search_result = await communication_agent.perform_task("semantic_search",
                                                             query="machine learning algorithms for cybersecurity",
                                                             knowledge_base=['ml_algorithms', 'cybersecurity', 'threat_detection', 'anomaly_scoring'])
        print(f"✅ Semantic search result: {search_result.get('success', False)}")
        if search_result.get('success'):
            data = search_result.get('result', {})
            print(f"   🔎 Query: {data.get('query', 'N/A')}")
            print(f"   📊 Total results: {data.get('total_results', 0)}")
            print(f"   ⚡ Search time: {data.get('search_time', 'N/A')}")
            top_results = data.get('top_results', [])
            if top_results:
                print(f"   🏆 Top result: {top_results[0]['document']} (score: {top_results[0]['relevance_score']:.3f})")

        # Test knowledge base management
        print("\n🗂️  Testing knowledge base management...")
        kb_result = await communication_agent.perform_task("knowledge_base",
                                                         operation="index",
                                                         content={'type': 'algorithm_documentation', 'topic': 'neural_networks'})
        print(f"✅ Knowledge base management result: {kb_result.get('success', False)}")
        if kb_result.get('success'):
            data = kb_result.get('result', {})
            print(f"   🏗️  Operation: {data.get('operation', 'N/A')}")
            print(f"   📊 Knowledge base size: {data.get('knowledge_base_size', 0)}")
            print(f"   ⚙️  Processing time: {data.get('processing_time', 'N/A')}")
            print(f"   ✅ Consistency check: {data.get('consistency_check', False)}")

        # Test knowledge graph construction
        print("\n🕸️  Testing knowledge graph construction...")
        entities = ['threat_detection', 'anomaly_scoring', 'behavioral_analysis', 'pattern_recognition', 'neural_networks']
        relationships = [('threat_detection', 'uses', 'anomaly_scoring'),
                        ('anomaly_scoring', 'relates_to', 'behavioral_analysis'),
                        ('behavioral_analysis', 'employs', 'pattern_recognition'),
                        ('pattern_recognition', 'powered_by', 'neural_networks')]

        graph_result = await communication_agent.perform_task("knowledge_graph",
                                                            entities=entities,
                                                            relationships=relationships)
        print(f"✅ Knowledge graph construction result: {graph_result.get('success', False)}")
        if graph_result.get('success'):
            data = graph_result.get('result', {})
            print(f"   🔗 Nodes: {data.get('nodes', 0)}")
            print(f"   ➡️  Edges: {data.get('edges', 0)}")
            print(f"   📈 Average degree: {data.get('average_degree', 0):.2f}")
            print(f"   🏗️  Construction time: {data.get('construction_time', 'N/A')}")
            print(f"   🛡️  Fault tolerance: {data.get('fault_tolerance', 0):.1%}")

        # Test existing communication capabilities (for comparison)
        print("\n📡 Testing federated communication (existing capability)...")
        fed_result = await communication_agent.perform_task("federated_communication", participants=15)
        print(f"✅ Federated communication result: {fed_result.get('success', False)}")
        if fed_result.get('success'):
            data = fed_result.get('result', {})
            print(f"   👥 Participants connected: {data.get('participants_connected', 0)}")
            print(f"   🔒 Communication protocol: {data.get('communication_protocol', 'N/A')}")
            print(f"   ⚡ Latency: {data.get('latency', 'N/A')}")

        # Shutdown system
        print("\n🛑 Shutting down agent system...")
        shutdown_success = await agent_system.shutdown_system()
        print(f"✅ Shutdown successful: {shutdown_success}")

        print("\n🎉 Knowledge Management test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_knowledge_management())
    sys.exit(0 if success else 1)