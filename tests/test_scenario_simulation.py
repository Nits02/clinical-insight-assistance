"""
Unit tests for the Scenario Simulation module.

This module tests all functionality of the ScenarioSimulator class including:
- Initialization and configuration
- Dosage adjustment simulations
- Risk assessments  
- Prediction intervals
- Recommendation generation
- Confidence scoring
- Edge cases and error handling
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os
from datetime import datetime, timedelta

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scenario_simulation import ScenarioSimulator, SimulationResult


class TestSimulationResult(unittest.TestCase):
    """Test the SimulationResult dataclass."""
    
    def test_simulation_result_creation(self):
        """Test creating a SimulationResult instance."""
        result = SimulationResult(
            simulation_id="TEST_001",
            scenario_type="dosage_adjustment",
            patient_id="P001",
            baseline_metrics={"average_outcome": 75.0},
            simulation_parameters={"current_dosage": 50, "proposed_dosage": 75},
            predicted_outcomes={"outcome_change": 5.2},
            risk_assessment={"overall_risk": "low"},
            confidence_intervals={"outcome_score_ci": [70.0, 80.0]},
            recommendations=["Test recommendation"],
            confidence_score=0.85
        )
        
        self.assertEqual(result.simulation_id, "TEST_001")
        self.assertEqual(result.scenario_type, "dosage_adjustment")
        self.assertEqual(result.patient_id, "P001")
        self.assertEqual(result.confidence_score, 0.85)
        self.assertIsInstance(result.baseline_metrics, dict)
        self.assertIsInstance(result.recommendations, list)


class TestScenarioSimulatorInitialization(unittest.TestCase):
    """Test ScenarioSimulator initialization and configuration."""
    
    def test_initialization_default_config(self):
        """Test initialization with default configuration."""
        simulator = ScenarioSimulator()
        
        self.assertIsInstance(simulator.config, dict)
        self.assertIsInstance(simulator.simulation_history, list)
        self.assertIsInstance(simulator.models, dict)
        self.assertEqual(len(simulator.simulation_history), 0)
        
        # Check default config structure
        self.assertIn('simulation_config', simulator.config)
        self.assertIn('dosage_config', simulator.config)
        self.assertIn('risk_thresholds', simulator.config)
        self.assertIn('prediction_config', simulator.config)
    
    def test_initialization_custom_config(self):
        """Test initialization with custom configuration."""
        custom_config = {
            'simulation_config': {
                'monte_carlo_iterations': 500,
                'confidence_level': 0.90
            }
        }
        
        simulator = ScenarioSimulator(config=custom_config)
        self.assertEqual(simulator.config, custom_config)
    
    def test_default_config_structure(self):
        """Test the structure and values of default configuration."""
        simulator = ScenarioSimulator()
        config = simulator.config
        
        # Test simulation config
        sim_config = config['simulation_config']
        self.assertEqual(sim_config['monte_carlo_iterations'], 1000)
        self.assertEqual(sim_config['confidence_level'], 0.95)
        self.assertEqual(sim_config['random_seed'], 42)
        self.assertEqual(sim_config['min_historical_data_points'], 5)
        
        # Test dosage config
        dosage_config = config['dosage_config']
        self.assertEqual(dosage_config['min_dosage'], 25)
        self.assertEqual(dosage_config['max_dosage'], 200)
        self.assertEqual(dosage_config['therapeutic_window'], [50, 100])
        
        # Test risk thresholds
        risk_config = config['risk_thresholds']
        self.assertEqual(risk_config['low_risk'], 0.1)
        self.assertEqual(risk_config['medium_risk'], 0.25)
        self.assertEqual(risk_config['high_risk'], 0.5)


class TestBaselineMetricsCalculation(unittest.TestCase):
    """Test baseline metrics calculation functionality."""
    
    def setUp(self):
        """Set up test data for baseline metrics tests."""
        self.simulator = ScenarioSimulator()
        
        # Create sample patient data
        data = []
        for i in range(10):
            data.append({
                'patient_id': 'P001',
                'trial_day': i + 1,
                'dosage_mg': 50 + i,
                'compliance_pct': 85 + i,
                'adverse_event_flag': 1 if i % 3 == 0 else 0,
                'outcome_score': 75 + i * 0.5,
                'visit_date': datetime(2024, 1, 1) + timedelta(days=i)
            })
        
        self.patient_data = pd.DataFrame(data)
    
    def test_calculate_baseline_metrics_complete_data(self):
        """Test baseline metrics calculation with complete data."""
        metrics = self.simulator._calculate_baseline_metrics(self.patient_data)
        
        # Test presence of all expected metrics
        expected_metrics = [
            'average_outcome', 'outcome_trend', 'outcome_variability',
            'average_compliance', 'compliance_variability',
            'average_dosage', 'dosage_variability',
            'adverse_event_rate', 'total_adverse_events',
            'data_points', 'duration_days'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Test specific calculations
        self.assertEqual(metrics['data_points'], 10)
        self.assertEqual(metrics['duration_days'], 9)  # 9 days between first and last visit
        self.assertAlmostEqual(metrics['average_outcome'], 77.25, places=2)
        self.assertAlmostEqual(metrics['average_dosage'], 54.5, places=1)
        self.assertAlmostEqual(metrics['adverse_event_rate'], 0.4, places=1)  # 4 out of 10
    
    def test_calculate_baseline_metrics_missing_columns(self):
        """Test baseline metrics calculation with missing columns."""
        # Create data without some columns
        minimal_data = pd.DataFrame({
            'patient_id': ['P001'] * 5,
            'trial_day': [1, 2, 3, 4, 5],
            'outcome_score': [75, 76, 77, 78, 79]
        })
        
        metrics = self.simulator._calculate_baseline_metrics(minimal_data)
        
        # Should have outcome metrics but not compliance/dosage/adverse events
        self.assertIn('average_outcome', metrics)
        self.assertIn('data_points', metrics)
        self.assertNotIn('average_compliance', metrics)
        self.assertNotIn('average_dosage', metrics)
        self.assertNotIn('adverse_event_rate', metrics)


class TestDosageOutcomePrediction(unittest.TestCase):
    """Test dosage outcome prediction functionality."""
    
    def setUp(self):
        """Set up test data for dosage prediction tests."""
        self.simulator = ScenarioSimulator()
        
        # Create baseline metrics
        self.baseline_metrics = {
            'average_outcome': 75.0,
            'average_dosage': 50.0,
            'adverse_event_rate': 0.1,
            'outcome_variability': 5.0
        }
        
        # Create minimal patient history
        self.patient_history = pd.DataFrame({
            'patient_id': ['P001'] * 5,
            'trial_day': [1, 2, 3, 4, 5],
            'dosage_mg': [50, 50, 50, 50, 50],
            'outcome_score': [75, 76, 74, 77, 75]
        })
    
    def test_predict_dosage_outcomes_increase(self):
        """Test predicting outcomes for dosage increase."""
        outcomes = self.simulator._predict_dosage_outcomes(
            self.patient_history, self.baseline_metrics, 75.0, 30
        )
        
        # Check all expected outcome keys are present
        expected_keys = [
            'predicted_outcome_score', 'outcome_change',
            'predicted_adverse_event_rate', 'adverse_event_risk_change',
            'dosage_effect_magnitude', 'therapeutic_window_compliance'
        ]
        
        for key in expected_keys:
            self.assertIn(key, outcomes)
        
        # Test that increasing dosage improves outcome (for logarithmic curve)
        self.assertGreater(outcomes['outcome_change'], 0)
        self.assertGreaterEqual(outcomes['predicted_outcome_score'], 75.0)
        
        # Test that new dosage is within therapeutic window
        self.assertEqual(outcomes['therapeutic_window_compliance'], 1.0)
    
    def test_predict_dosage_outcomes_decrease(self):
        """Test predicting outcomes for dosage decrease."""
        outcomes = self.simulator._predict_dosage_outcomes(
            self.patient_history, self.baseline_metrics, 25.0, 30
        )
        
        # Test that decreasing dosage reduces outcome
        self.assertLess(outcomes['outcome_change'], 0)
        
        # Test that dosage below therapeutic window has reduced effectiveness
        self.assertEqual(outcomes['therapeutic_window_compliance'], 0.0)
    
    def test_predict_dosage_outcomes_different_curves(self):
        """Test different efficacy response curves."""
        # Test linear curve
        self.simulator.config['dosage_config']['efficacy_response_curve'] = 'linear'
        linear_outcomes = self.simulator._predict_dosage_outcomes(
            self.patient_history, self.baseline_metrics, 75.0, 30
        )
        
        # Test sigmoid curve
        self.simulator.config['dosage_config']['efficacy_response_curve'] = 'sigmoid'
        sigmoid_outcomes = self.simulator._predict_dosage_outcomes(
            self.patient_history, self.baseline_metrics, 75.0, 30
        )
        
        # Test logarithmic curve (default)
        self.simulator.config['dosage_config']['efficacy_response_curve'] = 'logarithmic'
        log_outcomes = self.simulator._predict_dosage_outcomes(
            self.patient_history, self.baseline_metrics, 75.0, 30
        )
        
        # All should predict positive outcome change for dosage increase
        self.assertGreater(linear_outcomes['outcome_change'], 0)
        self.assertGreater(sigmoid_outcomes['outcome_change'], 0)
        self.assertGreater(log_outcomes['outcome_change'], 0)
    
    def test_predict_dosage_outcomes_high_dosage_safety(self):
        """Test safety considerations for high dosage."""
        high_dosage = 175  # Above safety threshold of 150
        
        outcomes = self.simulator._predict_dosage_outcomes(
            self.patient_history, self.baseline_metrics, high_dosage, 30
        )
        
        # Should have increased adverse event risk
        self.assertGreater(
            outcomes['predicted_adverse_event_rate'], 
            self.baseline_metrics['adverse_event_rate']
        )
        self.assertGreater(outcomes['adverse_event_risk_change'], 0)


class TestRiskAssessment(unittest.TestCase):
    """Test risk assessment functionality."""
    
    def setUp(self):
        """Set up test data for risk assessment tests."""
        self.simulator = ScenarioSimulator()
        
        self.baseline_metrics = {
            'average_outcome': 75.0,
            'average_dosage': 50.0,
            'adverse_event_rate': 0.1
        }
        
        self.predicted_outcomes = {
            'outcome_change': 5.0,
            'adverse_event_risk_change': 0.02,
            'predicted_adverse_event_rate': 0.12
        }
    
    def test_assess_dosage_risks_low_risk(self):
        """Test low risk scenario assessment."""
        # Small dosage change (10%)
        risks = self.simulator._assess_dosage_risks(
            self.baseline_metrics, 50.0, 55.0, self.predicted_outcomes
        )
        
        self.assertEqual(risks['dosage_change_risk'], 'low')
        self.assertEqual(risks['safety_threshold_risk'], 'low')
        self.assertEqual(risks['adverse_event_risk'], 'low')
        self.assertEqual(risks['overall_risk'], 'low')
    
    def test_assess_dosage_risks_medium_risk(self):
        """Test medium risk scenario assessment."""
        # Medium dosage change (40%)
        risks = self.simulator._assess_dosage_risks(
            self.baseline_metrics, 50.0, 70.0, self.predicted_outcomes
        )
        
        self.assertEqual(risks['dosage_change_risk'], 'medium')
        self.assertEqual(risks['safety_threshold_risk'], 'low')
    
    def test_assess_dosage_risks_high_risk(self):
        """Test high risk scenario assessment."""
        # Large dosage change (100%) and high dosage
        high_risk_outcomes = self.predicted_outcomes.copy()
        high_risk_outcomes['adverse_event_risk_change'] = 0.15
        
        risks = self.simulator._assess_dosage_risks(
            self.baseline_metrics, 50.0, 175.0, high_risk_outcomes
        )
        
        self.assertEqual(risks['dosage_change_risk'], 'high')
        self.assertEqual(risks['safety_threshold_risk'], 'high')
        self.assertEqual(risks['adverse_event_risk'], 'high')
        self.assertEqual(risks['overall_risk'], 'high')
    
    def test_assess_dosage_risks_safety_threshold(self):
        """Test safety threshold risk assessment."""
        # Test dosage at safety threshold boundary
        risks_at_threshold = self.simulator._assess_dosage_risks(
            self.baseline_metrics, 50.0, 150.0, self.predicted_outcomes
        )
        
        risks_above_threshold = self.simulator._assess_dosage_risks(
            self.baseline_metrics, 50.0, 160.0, self.predicted_outcomes
        )
        
        # Should be different risk levels
        self.assertNotEqual(
            risks_at_threshold['safety_threshold_risk'],
            risks_above_threshold['safety_threshold_risk']
        )


class TestPredictionIntervals(unittest.TestCase):
    """Test prediction interval calculation using Monte Carlo simulation."""
    
    def setUp(self):
        """Set up test data for prediction interval tests."""
        self.simulator = ScenarioSimulator()
        
        self.baseline_metrics = {
            'average_outcome': 75.0,
            'average_dosage': 50.0,
            'adverse_event_rate': 0.1
        }
        
        self.patient_history = pd.DataFrame({
            'patient_id': ['P001'] * 10,
            'trial_day': list(range(1, 11)),
            'dosage_mg': [50] * 10,
            'outcome_score': [75 + i * 0.5 for i in range(10)]
        })
    
    def test_calculate_prediction_intervals_structure(self):
        """Test prediction interval calculation structure."""
        intervals = self.simulator._calculate_prediction_intervals(
            self.patient_history, self.baseline_metrics, 75.0, 30
        )
        
        # Check expected keys are present
        self.assertIn('outcome_score_ci', intervals)
        self.assertIn('adverse_event_rate_ci', intervals)
        
        # Check intervals are lists with two elements (lower, upper bounds)
        self.assertIsInstance(intervals['outcome_score_ci'], list)
        self.assertEqual(len(intervals['outcome_score_ci']), 2)
        self.assertIsInstance(intervals['adverse_event_rate_ci'], list)
        self.assertEqual(len(intervals['adverse_event_rate_ci']), 2)
        
        # Check lower bound <= upper bound (can be equal for very stable predictions)
        self.assertLessEqual(
            intervals['outcome_score_ci'][0],
            intervals['outcome_score_ci'][1]
        )
        self.assertLessEqual(
            intervals['adverse_event_rate_ci'][0],
            intervals['adverse_event_rate_ci'][1]
        )
    
    def test_calculate_prediction_intervals_reproducibility(self):
        """Test that prediction intervals are reproducible with same seed."""
        intervals1 = self.simulator._calculate_prediction_intervals(
            self.patient_history, self.baseline_metrics, 75.0, 30
        )
        
        intervals2 = self.simulator._calculate_prediction_intervals(
            self.patient_history, self.baseline_metrics, 75.0, 30
        )
        
        # Should be identical due to fixed random seed
        np.testing.assert_array_almost_equal(
            intervals1['outcome_score_ci'],
            intervals2['outcome_score_ci']
        )
    
    def test_calculate_prediction_intervals_different_iterations(self):
        """Test prediction intervals with different iteration counts."""
        # Test with fewer iterations
        self.simulator.config['simulation_config']['monte_carlo_iterations'] = 100
        
        intervals = self.simulator._calculate_prediction_intervals(
            self.patient_history, self.baseline_metrics, 75.0, 30
        )
        
        # Should still produce valid intervals
        self.assertIsInstance(intervals['outcome_score_ci'], list)
        self.assertEqual(len(intervals['outcome_score_ci']), 2)


class TestRecommendationGeneration(unittest.TestCase):
    """Test recommendation generation functionality."""
    
    def setUp(self):
        """Set up test data for recommendation tests."""
        self.simulator = ScenarioSimulator()
        
        self.baseline_metrics = {
            'average_outcome': 75.0,
            'average_dosage': 50.0
        }
    
    def test_generate_dosage_recommendations_positive_change(self):
        """Test recommendations for positive outcome change."""
        simulation_params = {
            'current_dosage': 50.0,
            'proposed_dosage': 75.0
        }
        
        predicted_outcomes = {
            'outcome_change': 8.0,
            'adverse_event_risk_change': 0.02
        }
        
        risk_assessment = {
            'overall_risk': 'low'
        }
        
        recommendations = self.simulator._generate_dosage_recommendations(
            self.baseline_metrics, simulation_params, predicted_outcomes, risk_assessment
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Should contain positive recommendation for outcome improvement
        rec_text = ' '.join(recommendations).lower()
        self.assertIn('improve', rec_text)
    
    def test_generate_dosage_recommendations_negative_change(self):
        """Test recommendations for negative outcome change."""
        simulation_params = {
            'current_dosage': 50.0,
            'proposed_dosage': 25.0
        }
        
        predicted_outcomes = {
            'outcome_change': -8.0,
            'adverse_event_risk_change': 0.01
        }
        
        risk_assessment = {
            'overall_risk': 'medium'
        }
        
        recommendations = self.simulator._generate_dosage_recommendations(
            self.baseline_metrics, simulation_params, predicted_outcomes, risk_assessment
        )
        
        # Should contain warning about outcome reduction
        rec_text = ' '.join(recommendations).lower()
        self.assertIn('reduce', rec_text)
    
    def test_generate_dosage_recommendations_therapeutic_window(self):
        """Test recommendations for dosage outside therapeutic window."""
        # Test below therapeutic window
        simulation_params_low = {
            'current_dosage': 50.0,
            'proposed_dosage': 25.0  # Below therapeutic window [50, 100]
        }
        
        predicted_outcomes = {
            'outcome_change': 2.0,
            'adverse_event_risk_change': 0.01
        }
        
        risk_assessment = {
            'overall_risk': 'low'
        }
        
        recommendations_low = self.simulator._generate_dosage_recommendations(
            self.baseline_metrics, simulation_params_low, predicted_outcomes, risk_assessment
        )
        
        rec_text_low = ' '.join(recommendations_low).lower()
        self.assertIn('below therapeutic window', rec_text_low)
        
        # Test above therapeutic window
        simulation_params_high = {
            'current_dosage': 50.0,
            'proposed_dosage': 125.0  # Above therapeutic window [50, 100]
        }
        
        recommendations_high = self.simulator._generate_dosage_recommendations(
            self.baseline_metrics, simulation_params_high, predicted_outcomes, risk_assessment
        )
        
        rec_text_high = ' '.join(recommendations_high).lower()
        self.assertIn('above therapeutic window', rec_text_high)
    
    def test_generate_dosage_recommendations_high_ae_risk(self):
        """Test recommendations for high adverse event risk."""
        simulation_params = {
            'current_dosage': 50.0,
            'proposed_dosage': 75.0
        }
        
        predicted_outcomes = {
            'outcome_change': 5.0,
            'adverse_event_risk_change': 0.08  # High risk change
        }
        
        risk_assessment = {
            'overall_risk': 'medium'
        }
        
        recommendations = self.simulator._generate_dosage_recommendations(
            self.baseline_metrics, simulation_params, predicted_outcomes, risk_assessment
        )
        
        rec_text = ' '.join(recommendations).lower()
        self.assertIn('adverse event risk', rec_text)
        self.assertIn('monitoring', rec_text)


class TestConfidenceScoring(unittest.TestCase):
    """Test confidence score calculation functionality."""
    
    def setUp(self):
        """Set up test data for confidence scoring tests."""
        self.simulator = ScenarioSimulator()
    
    def test_calculate_confidence_score_high_confidence(self):
        """Test confidence score calculation for high confidence scenario."""
        # Lots of data points, low risk, small change
        risk_assessment = {'overall_risk': 'low'}
        predicted_outcomes = {'outcome_change': 2.0}
        
        confidence = self.simulator._calculate_confidence_score(
            50, risk_assessment, predicted_outcomes
        )
        
        self.assertGreaterEqual(confidence, 0.8)
        self.assertLessEqual(confidence, 0.95)
    
    def test_calculate_confidence_score_low_confidence(self):
        """Test confidence score calculation for low confidence scenario."""
        # Few data points, high risk, large change
        risk_assessment = {'overall_risk': 'high'}
        predicted_outcomes = {'outcome_change': 25.0}
        
        confidence = self.simulator._calculate_confidence_score(
            3, risk_assessment, predicted_outcomes
        )
        
        self.assertLess(confidence, 0.7)
    
    def test_calculate_confidence_score_data_points_effect(self):
        """Test effect of data points on confidence score."""
        risk_assessment = {'overall_risk': 'medium'}
        predicted_outcomes = {'outcome_change': 5.0}
        
        confidence_few_points = self.simulator._calculate_confidence_score(
            5, risk_assessment, predicted_outcomes
        )
        
        confidence_many_points = self.simulator._calculate_confidence_score(
            50, risk_assessment, predicted_outcomes
        )
        
        self.assertGreater(confidence_many_points, confidence_few_points)


class TestDosageAdjustmentSimulation(unittest.TestCase):
    """Test complete dosage adjustment simulation functionality."""
    
    def setUp(self):
        """Set up test data for full simulation tests."""
        self.simulator = ScenarioSimulator()
        
        # Create comprehensive patient data
        data = []
        np.random.seed(42)
        for i in range(20):
            data.append({
                'patient_id': 'P001',
                'trial_day': i + 1,
                'dosage_mg': 50,
                'compliance_pct': 85 + np.random.normal(0, 5),
                'adverse_event_flag': 1 if np.random.random() < 0.1 else 0,
                'outcome_score': 75 + np.random.normal(0, 3),
                'cohort': 'A',
                'visit_date': datetime(2024, 1, 1) + timedelta(days=i)
            })
        
        self.patient_data = pd.DataFrame(data)
    
    def test_simulate_dosage_adjustment_complete(self):
        """Test complete dosage adjustment simulation."""
        result = self.simulator.simulate_dosage_adjustment(
            self.patient_data, 'P001', 50.0, 75.0, 30
        )
        
        # Test result structure
        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(result.scenario_type, 'dosage_adjustment')
        self.assertEqual(result.patient_id, 'P001')
        
        # Test all required attributes are present
        self.assertIsInstance(result.baseline_metrics, dict)
        self.assertIsInstance(result.simulation_parameters, dict)
        self.assertIsInstance(result.predicted_outcomes, dict)
        self.assertIsInstance(result.risk_assessment, dict)
        self.assertIsInstance(result.confidence_intervals, dict)
        self.assertIsInstance(result.recommendations, list)
        self.assertIsInstance(result.confidence_score, float)
        
        # Test confidence score is in valid range
        self.assertGreaterEqual(result.confidence_score, 0.0)
        self.assertLessEqual(result.confidence_score, 0.95)
        
        # Test simulation is stored in history
        self.assertEqual(len(self.simulator.simulation_history), 1)
        self.assertEqual(self.simulator.simulation_history[0], result)
    
    def test_simulate_dosage_adjustment_insufficient_data(self):
        """Test simulation with insufficient historical data."""
        # Create data with only 3 points (below minimum of 5)
        insufficient_data = self.patient_data.head(3)
        
        with self.assertRaises(ValueError) as context:
            self.simulator.simulate_dosage_adjustment(
                insufficient_data, 'P001', 50.0, 75.0, 30
            )
        
        self.assertIn("Insufficient historical data", str(context.exception))
    
    def test_simulate_dosage_adjustment_multiple_simulations(self):
        """Test multiple dosage adjustment simulations."""
        # First simulation
        result1 = self.simulator.simulate_dosage_adjustment(
            self.patient_data, 'P001', 50.0, 75.0, 30
        )
        
        # Second simulation
        result2 = self.simulator.simulate_dosage_adjustment(
            self.patient_data, 'P001', 50.0, 100.0, 30
        )
        
        # Check both simulations are stored
        self.assertEqual(len(self.simulator.simulation_history), 2)
        
        # Check simulation IDs are different
        self.assertNotEqual(result1.simulation_id, result2.simulation_id)
        
        # Check different dosage parameters
        self.assertNotEqual(
            result1.simulation_parameters['proposed_dosage'],
            result2.simulation_parameters['proposed_dosage']
        )


class TestSimulationSummary(unittest.TestCase):
    """Test simulation summary functionality."""
    
    def setUp(self):
        """Set up test data for simulation summary tests."""
        self.simulator = ScenarioSimulator()
        
        # Create patient data
        data = []
        for i in range(10):
            data.append({
                'patient_id': 'P001',
                'trial_day': i + 1,
                'dosage_mg': 50,
                'compliance_pct': 85,
                'adverse_event_flag': 0,
                'outcome_score': 75,
                'visit_date': datetime(2024, 1, 1) + timedelta(days=i)
            })
        
        self.patient_data = pd.DataFrame(data)
    
    def test_get_simulation_summary_empty(self):
        """Test simulation summary with no simulations."""
        summary = self.simulator.get_simulation_summary()
        
        self.assertEqual(summary['total_simulations'], 0)
    
    def test_get_simulation_summary_with_simulations(self):
        """Test simulation summary with multiple simulations."""
        # Run several simulations
        self.simulator.simulate_dosage_adjustment(
            self.patient_data, 'P001', 50.0, 75.0, 30
        )
        self.simulator.simulate_dosage_adjustment(
            self.patient_data, 'P001', 50.0, 100.0, 30
        )
        
        summary = self.simulator.get_simulation_summary()
        
        # Test summary structure
        self.assertEqual(summary['total_simulations'], 2)
        self.assertIn('by_scenario_type', summary)
        self.assertIn('average_confidence_score', summary)
        self.assertIn('high_confidence_simulations', summary)
        self.assertIn('recent_simulations', summary)
        
        # Test scenario type counting
        self.assertEqual(summary['by_scenario_type']['dosage_adjustment'], 2)
        
        # Test average confidence score
        self.assertIsInstance(summary['average_confidence_score'], float)
        self.assertGreaterEqual(summary['average_confidence_score'], 0.0)
        self.assertLessEqual(summary['average_confidence_score'], 1.0)
        
        # Test recent simulations
        self.assertEqual(len(summary['recent_simulations']), 2)
        
        recent_sim = summary['recent_simulations'][0]
        self.assertIn('simulation_id', recent_sim)
        self.assertIn('scenario_type', recent_sim)
        self.assertIn('patient_id', recent_sim)
        self.assertIn('confidence_score', recent_sim)


class TestEdgeCasesAndErrorHandling(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up test data for edge case tests."""
        self.simulator = ScenarioSimulator()
    
    def test_zero_dosage_change(self):
        """Test simulation with zero dosage change."""
        data = []
        for i in range(10):
            data.append({
                'patient_id': 'P001',
                'trial_day': i + 1,
                'dosage_mg': 50,
                'compliance_pct': 85,
                'adverse_event_flag': 0,
                'outcome_score': 75,
                'visit_date': datetime(2024, 1, 1) + timedelta(days=i)
            })
        
        patient_data = pd.DataFrame(data)
        
        # Same current and proposed dosage
        result = self.simulator.simulate_dosage_adjustment(
            patient_data, 'P001', 50.0, 50.0, 30
        )
        
        # Should complete without error
        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(result.simulation_parameters['dosage_change'], 0.0)
    
    def test_extreme_dosage_values(self):
        """Test simulation with extreme dosage values."""
        data = []
        for i in range(10):
            data.append({
                'patient_id': 'P001',
                'trial_day': i + 1,
                'dosage_mg': 50,
                'compliance_pct': 85,
                'adverse_event_flag': 0,
                'outcome_score': 75,
                'visit_date': datetime(2024, 1, 1) + timedelta(days=i)
            })
        
        patient_data = pd.DataFrame(data)
        
        # Very low dosage
        result_low = self.simulator.simulate_dosage_adjustment(
            patient_data, 'P001', 50.0, 1.0, 30
        )
        
        # Very high dosage
        result_high = self.simulator.simulate_dosage_adjustment(
            patient_data, 'P001', 50.0, 500.0, 30
        )
        
        # Both should complete without error
        self.assertIsInstance(result_low, SimulationResult)
        self.assertIsInstance(result_high, SimulationResult)
        
        # High dosage should have higher risk
        self.assertIn('high', result_high.risk_assessment['overall_risk'])
    
    def test_missing_patient_data(self):
        """Test simulation with missing patient in dataset."""
        data = []
        for i in range(10):
            data.append({
                'patient_id': 'P001',
                'trial_day': i + 1,
                'dosage_mg': 50,
                'outcome_score': 75
            })
        
        patient_data = pd.DataFrame(data)
        
        # Try to simulate for non-existent patient
        with self.assertRaises(ValueError):
            self.simulator.simulate_dosage_adjustment(
                patient_data, 'P999', 50.0, 75.0, 30
            )
    
    def test_invalid_simulation_duration(self):
        """Test simulation with various duration values."""
        data = []
        for i in range(10):
            data.append({
                'patient_id': 'P001',
                'trial_day': i + 1,
                'dosage_mg': 50,
                'outcome_score': 75,
                'visit_date': datetime(2024, 1, 1) + timedelta(days=i)
            })
        
        patient_data = pd.DataFrame(data)
        
        # Zero duration should work
        result_zero = self.simulator.simulate_dosage_adjustment(
            patient_data, 'P001', 50.0, 75.0, 0
        )
        self.assertIsInstance(result_zero, SimulationResult)
        
        # Negative duration should work (treated as absolute value internally)
        result_negative = self.simulator.simulate_dosage_adjustment(
            patient_data, 'P001', 50.0, 75.0, -10
        )
        self.assertIsInstance(result_negative, SimulationResult)


class TestMainFunction(unittest.TestCase):
    """Test the main function for basic functionality."""
    
    @patch('builtins.print')
    def test_main_function_execution(self, mock_print):
        """Test that main function executes without errors."""
        from scenario_simulation import main
        
        # Should run without raising exceptions
        main()
        
        # Should have printed some output
        self.assertTrue(mock_print.called)


if __name__ == '__main__':
    unittest.main(verbosity=2)