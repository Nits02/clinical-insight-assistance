"""
Scenario Simulation Module for Clinical Insights Assistant

This module provides simulation capabilities for clinical scenarios such as:
- Dosage adjustments and their predicted impact
- Compliance improvement scenarios
- Treatment protocol modifications
- Risk-benefit analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Data class to hold simulation results."""
    simulation_id: str
    scenario_type: str
    patient_id: str
    baseline_metrics: Dict[str, float]
    simulation_parameters: Dict[str, Any]
    predicted_outcomes: Dict[str, float]
    risk_assessment: Dict[str, Any]
    confidence_intervals: Dict[str, List[float]]
    recommendations: List[str]
    confidence_score: float


class ScenarioSimulator:
    """
    Class for simulating various clinical scenarios and predicting their outcomes.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Scenario Simulator.
        
        Args:
            config (Dict, optional): Configuration dictionary for simulation parameters.
        """
        self.config = config or self._get_default_config()
        self.simulation_history = []
        self.models = {}
        
        logger.info("Scenario Simulator initialized")
    
    def _get_default_config(self) -> Dict:
        """
        Get default configuration for scenario simulation.
        
        Returns:
            Dict: Default configuration parameters.
        """
        return {
            # Simulation parameters
            'simulation_config': {
                'monte_carlo_iterations': 1000,
                'confidence_level': 0.95,
                'random_seed': 42,
                'min_historical_data_points': 5
            },
            
            # Dosage simulation parameters
            'dosage_config': {
                'min_dosage': 25,
                'max_dosage': 200,
                'dosage_step': 25,
                'efficacy_response_curve': 'logarithmic',  # 'linear', 'logarithmic', 'sigmoid'
                'safety_threshold': 150,  # Dosage above which safety concerns increase
                'therapeutic_window': [50, 100]  # Optimal dosage range
            },
            
            # Risk assessment thresholds
            'risk_thresholds': {
                'low_risk': 0.1,
                'medium_risk': 0.25,
                'high_risk': 0.5,
                'adverse_event_multiplier': 1.5  # Factor by which AE risk increases with higher dosage
            },
            
            # Outcome prediction parameters
            'prediction_config': {
                'outcome_variance': 5.0,  # Standard deviation for outcome predictions
                'compliance_impact_factor': 0.3,  # How much compliance affects outcomes
                'dosage_impact_factor': 0.2,  # How much dosage affects outcomes
                'baseline_decay_factor': 0.95  # How baseline conditions decay over time
            }
        }
    
    def simulate_dosage_adjustment(self, patient_data: pd.DataFrame, patient_id: str,
                                 current_dosage: float, proposed_dosage: float,
                                 simulation_duration: int = 30) -> SimulationResult:
        """
        Simulate the impact of dosage adjustment on patient outcomes.
        
        Args:
            patient_data (pd.DataFrame): Historical patient data.
            patient_id (str): Patient identifier.
            current_dosage (float): Current dosage in mg.
            proposed_dosage (float): Proposed new dosage in mg.
            simulation_duration (int): Number of days to simulate.
            
        Returns:
            SimulationResult: Simulation results and predictions.
        """
        logger.info(f"Simulating dosage adjustment for patient {patient_id}: {current_dosage}mg -> {proposed_dosage}mg")
        
        # Filter patient data
        patient_history = patient_data[patient_data['patient_id'] == patient_id].copy()
        
        if len(patient_history) < self.config['simulation_config']['min_historical_data_points']:
            raise ValueError(f"Insufficient historical data for patient {patient_id}")
        
        # Calculate baseline metrics
        baseline_metrics = self._calculate_baseline_metrics(patient_history)
        
        # Prepare simulation parameters
        simulation_params = {
            'current_dosage': current_dosage,
            'proposed_dosage': proposed_dosage,
            'dosage_change': proposed_dosage - current_dosage,
            'dosage_change_percent': ((proposed_dosage - current_dosage) / current_dosage) * 100,
            'simulation_duration': simulation_duration
        }
        
        # Predict outcomes with new dosage
        predicted_outcomes = self._predict_dosage_outcomes(
            patient_history, baseline_metrics, proposed_dosage, simulation_duration
        )
        
        # Assess risks
        risk_assessment = self._assess_dosage_risks(
            baseline_metrics, current_dosage, proposed_dosage, predicted_outcomes
        )
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_prediction_intervals(
            patient_history, baseline_metrics, proposed_dosage, simulation_duration
        )
        
        # Generate recommendations
        recommendations = self._generate_dosage_recommendations(
            baseline_metrics, simulation_params, predicted_outcomes, risk_assessment
        )
        
        # Calculate overall confidence score
        confidence_score = self._calculate_confidence_score(
            len(patient_history), risk_assessment, predicted_outcomes
        )
        
        simulation_id = f"DOSAGE_{patient_id}_{len(self.simulation_history) + 1}"
        
        result = SimulationResult(
            simulation_id=simulation_id,
            scenario_type='dosage_adjustment',
            patient_id=patient_id,
            baseline_metrics=baseline_metrics,
            simulation_parameters=simulation_params,
            predicted_outcomes=predicted_outcomes,
            risk_assessment=risk_assessment,
            confidence_intervals=confidence_intervals,
            recommendations=recommendations,
            confidence_score=confidence_score
        )
        
        self.simulation_history.append(result)
        logger.info(f"Dosage simulation completed for patient {patient_id}")
        
        return result
    
    def _calculate_baseline_metrics(self, patient_history: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate baseline metrics from patient history.
        
        Args:
            patient_history (pd.DataFrame): Patient historical data.
            
        Returns:
            Dict[str, float]: Baseline metrics.
        """
        metrics = {}
        
        if 'outcome_score' in patient_history.columns:
            metrics['average_outcome'] = float(patient_history['outcome_score'].mean())
            metrics['outcome_trend'] = float(patient_history['outcome_score'].iloc[-1] - patient_history['outcome_score'].iloc[0])
            metrics['outcome_variability'] = float(patient_history['outcome_score'].std())
        
        if 'compliance_pct' in patient_history.columns:
            metrics['average_compliance'] = float(patient_history['compliance_pct'].mean())
            metrics['compliance_variability'] = float(patient_history['compliance_pct'].std())
        
        if 'dosage_mg' in patient_history.columns:
            metrics['average_dosage'] = float(patient_history['dosage_mg'].mean())
            metrics['dosage_variability'] = float(patient_history['dosage_mg'].std())
        
        if 'adverse_event_flag' in patient_history.columns:
            metrics['adverse_event_rate'] = float(patient_history['adverse_event_flag'].mean())
            metrics['total_adverse_events'] = int(patient_history['adverse_event_flag'].sum())
        
        metrics['data_points'] = len(patient_history)
        metrics['duration_days'] = (patient_history['visit_date'].max() - patient_history['visit_date'].min()).days if 'visit_date' in patient_history.columns else 30
        
        return metrics
    
    def _predict_dosage_outcomes(self, patient_history: pd.DataFrame, baseline_metrics: Dict,
                               new_dosage: float, duration: int) -> Dict[str, float]:
        """
        Predict outcomes based on dosage change.
        
        Args:
            patient_history (pd.DataFrame): Patient historical data.
            baseline_metrics (Dict): Baseline patient metrics.
            new_dosage (float): New dosage to simulate.
            duration (int): Simulation duration in days.
            
        Returns:
            Dict[str, float]: Predicted outcomes.
        """
        current_dosage = baseline_metrics.get('average_dosage', 50.0)
        current_outcome = baseline_metrics.get('average_outcome', 75.0)
        
        # Calculate dosage effect using configured response curve
        dosage_ratio = new_dosage / current_dosage
        
        if self.config['dosage_config']['efficacy_response_curve'] == 'logarithmic':
            dosage_effect = np.log(dosage_ratio) * self.config['prediction_config']['dosage_impact_factor'] * 20
        elif self.config['dosage_config']['efficacy_response_curve'] == 'sigmoid':
            # Sigmoid response with diminishing returns
            x = (new_dosage - 50) / 25  # Normalize around typical dosage
            dosage_effect = (2 / (1 + np.exp(-x)) - 1) * 10
        else:  # Linear
            dosage_effect = (dosage_ratio - 1) * self.config['prediction_config']['dosage_impact_factor'] * 15
        
        # Apply therapeutic window considerations
        therapeutic_min, therapeutic_max = self.config['dosage_config']['therapeutic_window']
        if new_dosage < therapeutic_min:
            dosage_effect *= 0.7  # Reduced effectiveness below therapeutic window
        elif new_dosage > therapeutic_max:
            dosage_effect *= 0.8  # Diminishing returns above therapeutic window
        
        # Predict new outcome
        predicted_outcome = current_outcome + dosage_effect
        predicted_outcome = np.clip(predicted_outcome, 0, 100)
        
        # Calculate adverse event risk
        ae_risk_multiplier = 1.0
        if new_dosage > self.config['dosage_config']['safety_threshold']:
            ae_risk_multiplier = self.config['risk_thresholds']['adverse_event_multiplier']
        
        current_ae_rate = baseline_metrics.get('adverse_event_rate', 0.1)
        predicted_ae_rate = min(0.5, current_ae_rate * ae_risk_multiplier)
        
        return {
            'predicted_outcome_score': float(predicted_outcome),
            'outcome_change': float(predicted_outcome - current_outcome),
            'predicted_adverse_event_rate': float(predicted_ae_rate),
            'adverse_event_risk_change': float(predicted_ae_rate - current_ae_rate),
            'dosage_effect_magnitude': float(abs(dosage_effect)),
            'therapeutic_window_compliance': float(therapeutic_min <= new_dosage <= therapeutic_max)
        }
    
    def _assess_dosage_risks(self, baseline_metrics: Dict, current_dosage: float,
                           new_dosage: float, predicted_outcomes: Dict) -> Dict[str, Any]:
        """
        Assess risks associated with dosage changes.
        
        Args:
            baseline_metrics (Dict): Baseline patient metrics.
            current_dosage (float): Current dosage.
            new_dosage (float): Proposed new dosage.
            predicted_outcomes (Dict): Predicted outcomes.
            
        Returns:
            Dict[str, Any]: Risk assessment.
        """
        risks = {}
        
        # Dosage-based risk assessment
        dosage_change_percent = abs((new_dosage - current_dosage) / current_dosage) * 100
        
        if dosage_change_percent > 50:
            risks['dosage_change_risk'] = 'high'
        elif dosage_change_percent > 25:
            risks['dosage_change_risk'] = 'medium'
        else:
            risks['dosage_change_risk'] = 'low'
        
        # Safety threshold risk
        safety_threshold = self.config['dosage_config']['safety_threshold']
        if new_dosage > safety_threshold:
            risks['safety_threshold_risk'] = 'high'
        elif new_dosage > safety_threshold * 0.8:
            risks['safety_threshold_risk'] = 'medium'
        else:
            risks['safety_threshold_risk'] = 'low'
        
        # Adverse event risk
        ae_risk_change = predicted_outcomes.get('adverse_event_risk_change', 0)
        if ae_risk_change > 0.1:
            risks['adverse_event_risk'] = 'high'
        elif ae_risk_change > 0.05:
            risks['adverse_event_risk'] = 'medium'
        else:
            risks['adverse_event_risk'] = 'low'
        
        # Overall risk assessment
        risk_scores = {'low': 1, 'medium': 2, 'high': 3}
        total_risk_score = sum(risk_scores[risk] for risk in [
            risks['dosage_change_risk'],
            risks['safety_threshold_risk'],
            risks['adverse_event_risk']
        ])
        
        if total_risk_score >= 7:
            risks['overall_risk'] = 'high'
        elif total_risk_score >= 5:
            risks['overall_risk'] = 'medium'
        else:
            risks['overall_risk'] = 'low'
        
        return risks
    
    def _calculate_prediction_intervals(self, patient_history: pd.DataFrame, baseline_metrics: Dict,
                                      new_dosage: float, duration: int) -> Dict[str, List[float]]:
        """
        Calculate prediction intervals using Monte Carlo simulation.
        
        Args:
            patient_history (pd.DataFrame): Patient historical data.
            baseline_metrics (Dict): Baseline metrics.
            new_dosage (float): New dosage.
            duration (int): Simulation duration.
            
        Returns:
            Dict[str, List[float]]: Prediction intervals.
        """
        np.random.seed(self.config['simulation_config']['random_seed'])
        
        # Monte Carlo simulation
        iterations = self.config['simulation_config']['monte_carlo_iterations']
        outcome_predictions = []
        ae_predictions = []
        
        outcome_variance = self.config['prediction_config']['outcome_variance']
        
        for _ in range(iterations):
            # Add random variation to baseline metrics
            varied_baseline = baseline_metrics.copy()
            varied_baseline['average_outcome'] += np.random.normal(0, outcome_variance)
            
            # Predict with variation
            outcomes = self._predict_dosage_outcomes(patient_history, varied_baseline, new_dosage, duration)
            outcome_predictions.append(outcomes['predicted_outcome_score'])
            ae_predictions.append(outcomes['predicted_adverse_event_rate'])
        
        # Calculate confidence intervals
        confidence_level = self.config['simulation_config']['confidence_level']
        alpha = 1 - confidence_level
        
        outcome_ci = [
            np.percentile(outcome_predictions, (alpha/2) * 100),
            np.percentile(outcome_predictions, (1 - alpha/2) * 100)
        ]
        
        ae_ci = [
            np.percentile(ae_predictions, (alpha/2) * 100),
            np.percentile(ae_predictions, (1 - alpha/2) * 100)
        ]
        
        return {
            'outcome_score_ci': outcome_ci,
            'adverse_event_rate_ci': ae_ci
        }
    
    def _generate_dosage_recommendations(self, baseline_metrics: Dict, simulation_params: Dict,
                                       predicted_outcomes: Dict, risk_assessment: Dict) -> List[str]:
        """
        Generate recommendations for dosage adjustment scenarios.
        
        Args:
            baseline_metrics (Dict): Baseline metrics.
            simulation_params (Dict): Simulation parameters.
            predicted_outcomes (Dict): Predicted outcomes.
            risk_assessment (Dict): Risk assessment.
            
        Returns:
            List[str]: List of recommendations.
        """
        recommendations = []
        
        outcome_change = predicted_outcomes.get('outcome_change', 0)
        overall_risk = risk_assessment.get('overall_risk', 'medium')
        
        # Efficacy-based recommendations
        if outcome_change > 5:
            recommendations.append(f"Dosage adjustment is predicted to improve outcomes by {outcome_change:.1f} points. Consider implementing the change.")
        elif outcome_change < -5:
            recommendations.append(f"Dosage adjustment may reduce outcomes by {abs(outcome_change):.1f} points. Reconsider this change.")
        else:
            recommendations.append("Dosage adjustment is predicted to have minimal impact on outcomes.")
        
        # Risk-based recommendations
        if overall_risk == 'high':
            recommendations.append("High risk detected. Implement change gradually with close monitoring.")
        elif overall_risk == 'medium':
            recommendations.append("Moderate risk detected. Monitor patient closely after implementation.")
        else:
            recommendations.append("Low risk detected. Change can be implemented with standard monitoring.")
        
        # Safety-specific recommendations
        ae_risk_change = predicted_outcomes.get('adverse_event_risk_change', 0)
        if ae_risk_change > 0.05:
            recommendations.append(f"Increased adverse event risk ({ae_risk_change:.1%}). Consider additional safety monitoring.")
        
        # Therapeutic window recommendations
        new_dosage = simulation_params.get('proposed_dosage', 50)
        therapeutic_min, therapeutic_max = self.config['dosage_config']['therapeutic_window']
        
        if new_dosage < therapeutic_min:
            recommendations.append(f"Proposed dosage ({new_dosage}mg) is below therapeutic window ({therapeutic_min}-{therapeutic_max}mg). Consider higher dosage.")
        elif new_dosage > therapeutic_max:
            recommendations.append(f"Proposed dosage ({new_dosage}mg) is above therapeutic window ({therapeutic_min}-{therapeutic_max}mg). Consider lower dosage.")
        
        return recommendations
    
    def _calculate_confidence_score(self, data_points: int, risk_assessment: Dict,
                                  predicted_outcomes: Dict) -> float:
        """
        Calculate overall confidence score for the simulation.
        
        Args:
            data_points (int): Number of historical data points.
            risk_assessment (Dict): Risk assessment results.
            predicted_outcomes (Dict): Predicted outcomes.
            
        Returns:
            float: Confidence score (0-1).
        """
        # Base confidence from data availability
        data_confidence = min(0.9, 0.5 + (data_points / 20))
        
        # Risk-based confidence adjustment
        overall_risk = risk_assessment.get('overall_risk', 'medium')
        risk_confidence = {'low': 0.9, 'medium': 0.7, 'high': 0.5}[overall_risk]
        
        # Prediction magnitude confidence (more conservative for larger changes)
        outcome_change = abs(predicted_outcomes.get('outcome_change', 0))
        magnitude_confidence = max(0.5, 1.0 - (outcome_change / 50))
        
        # Combined confidence score
        confidence_score = (data_confidence * 0.4 + risk_confidence * 0.3 + magnitude_confidence * 0.3)
        
        return min(0.95, confidence_score)
    
    def get_simulation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all simulations performed.
        
        Returns:
            Dict[str, Any]: Summary of simulations.
        """
        if not self.simulation_history:
            return {'total_simulations': 0}
        
        summary = {
            'total_simulations': len(self.simulation_history),
            'by_scenario_type': {},
            'average_confidence_score': 0.0,
            'high_confidence_simulations': 0,
            'recent_simulations': []
        }
        
        # Count by scenario type
        for sim in self.simulation_history:
            scenario_type = sim.scenario_type
            summary['by_scenario_type'][scenario_type] = summary['by_scenario_type'].get(scenario_type, 0) + 1
        
        # Calculate average confidence
        total_confidence = sum(sim.confidence_score for sim in self.simulation_history)
        summary['average_confidence_score'] = total_confidence / len(self.simulation_history)
        
        # Count high confidence simulations
        summary['high_confidence_simulations'] = sum(1 for sim in self.simulation_history if sim.confidence_score > 0.8)
        
        # Recent simulations (last 5)
        summary['recent_simulations'] = [
            {
                'simulation_id': sim.simulation_id,
                'scenario_type': sim.scenario_type,
                'patient_id': sim.patient_id,
                'confidence_score': sim.confidence_score
            }
            for sim in self.simulation_history[-5:]
        ]
        
        return summary


def main():
    """
    Main function for testing the Scenario Simulator.
    """
    # Create sample data for testing
    np.random.seed(42)
    
    # Generate sample patient data
    data = []
    for i in range(50):
        patient_id = "P001"  # Single patient for testing
        trial_day = i + 1
        
        # Simulate patient with declining outcomes
        outcome = 80 - (i * 0.3) + np.random.normal(0, 3)
        compliance = 85 + np.random.normal(0, 5)
        dosage = 50
        adverse_event = 1 if np.random.random() < 0.1 else 0
        
        data.append({
            'patient_id': patient_id,
            'trial_day': trial_day,
            'dosage_mg': dosage,
            'compliance_pct': np.clip(compliance, 0, 100),
            'adverse_event_flag': adverse_event,
            'outcome_score': np.clip(outcome, 0, 100),
            'cohort': 'A',
            'visit_date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=trial_day-1)
        })
    
    df = pd.DataFrame(data)
    
    # Initialize simulator
    simulator = ScenarioSimulator()
    
    # Test dosage adjustment simulation
    print("Testing Dosage Adjustment Simulation:")
    dosage_result = simulator.simulate_dosage_adjustment(df, 'P001', 50, 75)
    print(f"Predicted outcome change: {dosage_result.predicted_outcomes['outcome_change']:.2f}")
    print(f"Risk level: {dosage_result.risk_assessment['overall_risk']}")
    print(f"Confidence score: {dosage_result.confidence_score:.2f}")
    print(f"Recommendations: {len(dosage_result.recommendations)}")
    
    # Get simulation summary
    summary = simulator.get_simulation_summary()
    print(f"\nSimulation Summary:")
    print(f"Total simulations: {summary['total_simulations']}")
    print(f"Average confidence: {summary['average_confidence_score']:.2f}")


if __name__ == "__main__":
    main()