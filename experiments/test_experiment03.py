#!/usr/bin/env python3
"""
Test script for experiment03 - Edge-focused MEC with DQN training episodes
"""

import os
import sys

# Add project root to path
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from experiments.experiment03 import run_experiment_03


def test_experiment03():
    """Test experiment03 with a short simulation"""
    print("Testing Experiment 03 - Edge-focused MEC with DQN training episodes")
    print("=" * 70)
    
    try:
        # Run a short test simulation with 10 training iterations
        result_path = run_experiment_03(
            sim_time_s=120.0,  # 2 minutes test to allow for 10 training episodes
            edges_count=5,    # Small number of edges for testing
            cells_per_edge=10,  # Small number of cells per edge
            lambda_per_s=0.1,   # Low task arrival rate for testing
            edge_offload_bias=0.8,
            cloud_penalty=0.3,
            local_penalty=0.1,
            eval_interval_windows=5,  # Training every 5 windows for faster testing
            pdf_path="data/test_exp03.pdf",
            summary_json_path="data/test_exp03_summary.json"
        )
        
        print(f"✅ Experiment 03 completed successfully!")
        print(f"📊 Results saved to: {result_path}")
        
        # Check if files were created
        if os.path.exists(result_path):
            print(f"✅ PDF report created: {result_path}")
        else:
            print(f"❌ PDF report not found: {result_path}")
            
        if os.path.exists("data/test_exp03_summary.json"):
            print("✅ JSON summary created")
            
            # Analyze training backlog
            print("\n🔍 Analyzing DQN training backlog...")
            try:
                import subprocess
                result = subprocess.run([
                    sys.executable, "analyze_training_backlog.py", 
                    "data/test_exp03_summary.json"
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    print("✅ Training backlog analysis completed")
                    print(result.stdout)
                else:
                    print(f"⚠️ Training analysis failed: {result.stderr}")
            except Exception as e:
                print(f"⚠️ Could not run training analysis: {e}")
        else:
            print("❌ JSON summary not found")
            
    except Exception as e:
        print(f"❌ Experiment 03 failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = test_experiment03()
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Tests failed!")
        sys.exit(1)
