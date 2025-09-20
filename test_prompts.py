#!/usr/bin/env python3
"""
Test script to verify prompts submodule access.
"""

import json
import os
import sys

def test_prompts_access():
    """Test that we can access and load prompt files."""
    prompts_dir = "prompts"

    if not os.path.exists(prompts_dir):
        print(f"❌ Prompts directory '{prompts_dir}' not found")
        return False

    # Test loading project-spec.json
    spec_file = os.path.join(prompts_dir, "project-spec.json")
    if not os.path.exists(spec_file):
        print(f"❌ {spec_file} not found")
        return False

    try:
        with open(spec_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        title = data.get('metadata', {}).get('title', 'Unknown')
        version = data.get('metadata', {}).get('version', 'Unknown')

        print(f"✅ Successfully loaded project spec: {title} v{version}")

        # Test loading phase1-mvp.json
        phase1_file = os.path.join(prompts_dir, "phase1-mvp.json")
        if os.path.exists(phase1_file):
            with open(phase1_file, 'r', encoding='utf-8') as f:
                phase1_data = json.load(f)
            print("✅ Phase 1 prompts accessible")
        else:
            print(f"⚠️  {phase1_file} not found")

        return True

    except Exception as e:
        print(f"❌ Error loading prompts: {e}")
        return False

if __name__ == "__main__":
    print("Testing prompts submodule access...")
    success = test_prompts_access()
    sys.exit(0 if success else 1)