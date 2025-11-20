#!/usr/bin/env python3
"""
Test the fixed analysis functionality
"""
import os
import sys
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.append('src')

try:
    from agent_core import ClinicalAgent
    
    print("🧪 Testing fixed analysis with threading timeout...")
    
    # Create simple test data
    test_data = pd.DataFrame({
        'patient_id': [1, 2, 3],
        'visit_number': [1, 1, 1], 
        'treatment': ['A', 'B', 'A'],
        'outcome_score': [85, 78, 92],
        'compliance_pct': [95, 88, 97],
        'adverse_events': [0, 1, 0]
    })
    
    print("✅ Test data created")
    
    # Initialize agent
    os.environ['VALIDATE_API_CONNECTION'] = 'false'  # Skip validation for speed
    agent = ClinicalAgent()
    print("✅ Agent initialized")
    
    # Test sync analysis method
    print("🔄 Running sync analysis...")
    results = agent.analyze_trial_data_sync(test_data, ["Test efficacy"])
    
    print("✅ Analysis completed successfully!")
    print(f"📊 Results: {len(results.get('insights', []))} insights generated")
    
except Exception as e:
    print(f"❌ Test failed: {str(e)}")
    sys.exit(1)

print("🎉 All tests passed! The signal error is fixed.")