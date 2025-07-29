#!/usr/bin/env python3
"""
Test script for the graph editor integration.
This script tests the graph editor functionality independently.
"""

import sys
import os

# Add the extension directory to the path
ext_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ext_dir)

def test_graph_editor():
    """Test the graph editor functionality."""
    try:
        print("Testing graph editor import...")
        from mit.test_extensions_1.graph_editor import launch_graph_editor, GraphEditorWindow

        print("✓ Graph editor imported successfully")

        # Test creating a window
        print("Testing graph editor window creation...")
        window = GraphEditorWindow()
        print("✓ Graph editor window created successfully")

        # Test save directory creation
        save_dir = os.path.join(ext_dir, "test_graphs")
        os.makedirs(save_dir, exist_ok=True)
        print(f"✓ Save directory created: {save_dir}")

        print("✓ All tests passed!")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_graph_editor()
    sys.exit(0 if success else 1)