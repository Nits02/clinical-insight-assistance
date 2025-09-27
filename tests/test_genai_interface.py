"""
Unit tests for the GenAI Interface module.

This test suite provides comprehensive coverage for the GenAI Interface functionality,
including initialization, API interaction, and various analysis methods. It uses mocking
to avoid actual API calls during testing while ensuring all code paths are validated.

Test Coverage:
• GenAI Interface initialization and configuration
• API call functionality with error handling and retries
• Doctor notes analysis with structured output
• Cohort comparison summary generation
• Scenario simulation summary creation
• Regulatory summary generation
• Adverse event extraction from clinical text
• Clinical insights generation
• Error handling and edge cases
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import os
import tempfile
from dataclasses import asdict
import sys
import logging

# Add the src directory to the Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from genai_interface import GenAIInterface, AnalysisResult


class TestGenAIInterface(unittest.TestCase):
    """
    Test cases for the GenAI Interface class.
    
    This test class covers all aspects of the GenAI Interface functionality,
    from basic initialization to complex clinical analysis operations.
    """
    
    def setUp(self):
        """
        Set up test fixtures before each test method.
        
        Creates a GenAI Interface instance with a mock API key and
        prepares common test data used across multiple test methods.
        """
        # Use a test API key to avoid environment variable dependency
        self.test_api_key = "test-api-key-12345"
        self.genai = GenAIInterface(api_key=self.test_api_key, model="gpt-3.5-turbo")
        
        # Sample test data for various test scenarios
        self.sample_notes = [
            "Patient appears stable with no acute complaints today.",
            "Reports mild headache that started yesterday, rated 3/10.",
            "Some nausea after morning medication, resolved after food."
        ]
        
        self.sample_patient_context = {
            'patient_id': 'TEST001',
            'dosage_mg': 25,
            'compliance_pct': 92.5,
            'cohort': 'Treatment_A'
        }
        
        # Sample JSON response for testing structured parsing
        self.sample_analysis_response = {
            "summary": "Patient shows stable condition with minor side effects",
            "themes": ["Stability", "Mild side effects", "Good compliance"],
            "adverse_events": ["Mild headache", "Transient nausea"],
            "recommendations": ["Continue current dosage", "Monitor side effects"],
            "confidence_score": 0.85
        }
    
    def test_initialization_with_api_key_parameter(self):
        """Test GenAI Interface initialization with API key parameter."""
        genai = GenAIInterface(api_key="test-key", model="gpt-4")
        
        self.assertEqual(genai.api_key, "test-key")
        self.assertEqual(genai.model, "gpt-4")
        self.assertIsNotNone(genai.client)
        self.assertIsNotNone(genai.config)
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'env-test-key'})
    def test_initialization_with_environment_variable(self):
        """Test GenAI Interface initialization using environment variable."""
        genai = GenAIInterface()
        
        self.assertEqual(genai.api_key, "env-test-key")
        self.assertEqual(genai.model, "gpt-3.5-turbo")  # default model
    
    def test_initialization_without_api_key_raises_error(self):
        """Test that initialization without API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as context:
                GenAIInterface()
            
            self.assertIn("OpenAI API key not provided", str(context.exception))
    
    def test_default_configuration(self):
        """Test that default configuration is properly set."""
        config = self.genai._get_default_config()
        
        # Verify all expected configuration keys are present
        expected_keys = ['max_tokens', 'temperature', 'top_p', 'frequency_penalty', 
                        'presence_penalty', 'timeout', 'max_retries']
        for key in expected_keys:
            self.assertIn(key, config)
        
        # Verify configuration values are reasonable for clinical analysis
        self.assertEqual(config['temperature'], 0.3)  # Low for consistency
        self.assertEqual(config['max_tokens'], 1000)
        self.assertEqual(config['max_retries'], 3)
    
    @patch('genai_interface.OpenAI')
    def test_api_call_success(self, mock_openai_class):
        """Test successful API call with proper response handling."""
        # Mock the OpenAI client and response
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "  Test response content  "
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create fresh instance to use mocked client
        genai = GenAIInterface(api_key="test-key")
        
        # Test API call
        messages = [{"role": "user", "content": "Test message"}]
        result = genai._make_api_call(messages)
        
        # Verify response is properly stripped and returned
        self.assertEqual(result, "Test response content")
        mock_client.chat.completions.create.assert_called_once()
    
    @patch('genai_interface.OpenAI')
    def test_api_call_with_retries(self, mock_openai_class):
        """Test API call retry mechanism on failures."""
        # Mock the OpenAI client to fail twice, then succeed
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Configure mock to fail twice, then succeed on third attempt
        mock_client.chat.completions.create.side_effect = [
            Exception("Network error"),
            Exception("Rate limit error"),
            Mock(choices=[Mock(message=Mock(content="Success on retry"))])
        ]
        
        # Create fresh instance with mocked client
        genai = GenAIInterface(api_key="test-key")
        
        # Test API call with retries
        messages = [{"role": "user", "content": "Test message"}]
        result = genai._make_api_call(messages)
        
        # Verify successful result after retries
        self.assertEqual(result, "Success on retry")
        self.assertEqual(mock_client.chat.completions.create.call_count, 3)
    
    @patch('genai_interface.OpenAI')
    def test_api_call_exhausted_retries(self, mock_openai_class):
        """Test API call failure after exhausting all retries."""
        # Mock the OpenAI client to always fail
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("Persistent error")
        
        # Create fresh instance with mocked client
        genai = GenAIInterface(api_key="test-key")
        
        # Test API call that should fail after retries
        messages = [{"role": "user", "content": "Test message"}]
        
        with self.assertRaises(Exception) as context:
            genai._make_api_call(messages)
        
        self.assertIn("API call failed after 3 attempts", str(context.exception))
        self.assertEqual(mock_client.chat.completions.create.call_count, 3)
    
    @patch.object(GenAIInterface, '_make_api_call')
    def test_analyze_doctor_notes_with_valid_json_response(self, mock_api_call):
        """Test doctor notes analysis with valid JSON response."""
        # Mock API response with valid JSON
        mock_api_call.return_value = json.dumps(self.sample_analysis_response)
        
        # Perform analysis
        result = self.genai.analyze_doctor_notes(self.sample_notes, self.sample_patient_context)
        
        # Verify result structure and content
        self.assertIsInstance(result, AnalysisResult)
        self.assertEqual(result.summary, self.sample_analysis_response["summary"])
        self.assertEqual(result.themes, self.sample_analysis_response["themes"])
        self.assertEqual(result.adverse_events, self.sample_analysis_response["adverse_events"])
        self.assertEqual(result.recommendations, self.sample_analysis_response["recommendations"])
        self.assertEqual(result.confidence_score, self.sample_analysis_response["confidence_score"])
        
        # Verify API was called with correct parameters
        mock_api_call.assert_called_once()
        call_args = mock_api_call.call_args
        self.assertIn('temperature', call_args[1])
        self.assertEqual(call_args[1]['temperature'], 0.2)
    
    @patch.object(GenAIInterface, '_make_api_call')
    def test_analyze_doctor_notes_with_invalid_json_response(self, mock_api_call):
        """Test doctor notes analysis with invalid JSON response (fallback)."""
        # Mock API response with invalid JSON
        invalid_response = "This is not valid JSON but still useful text"
        mock_api_call.return_value = invalid_response
        
        # Perform analysis
        result = self.genai.analyze_doctor_notes(self.sample_notes)
        
        # Verify fallback behavior
        self.assertIsInstance(result, AnalysisResult)
        self.assertEqual(result.summary, invalid_response)
        self.assertEqual(result.themes, [])
        self.assertEqual(result.adverse_events, [])
        self.assertEqual(result.recommendations, [])
        self.assertEqual(result.confidence_score, 0.5)  # Lower confidence for unparsed
        self.assertEqual(result.raw_response, invalid_response)
    
    @patch.object(GenAIInterface, '_make_api_call')
    def test_analyze_doctor_notes_without_patient_context(self, mock_api_call):
        """Test doctor notes analysis without patient context."""
        mock_api_call.return_value = json.dumps(self.sample_analysis_response)
        
        # Perform analysis without patient context
        result = self.genai.analyze_doctor_notes(self.sample_notes)
        
        # Verify analysis succeeds without context
        self.assertIsInstance(result, AnalysisResult)
        mock_api_call.assert_called_once()
        
        # Check that the prompt doesn't include patient context
        call_args = mock_api_call.call_args[0][0]  # Get the messages parameter
        prompt_content = call_args[1]['content']
        self.assertNotIn("Patient Context:", prompt_content)
    
    @patch.object(GenAIInterface, '_make_api_call')
    def test_generate_cohort_comparison_summary(self, mock_api_call):
        """Test cohort comparison summary generation."""
        expected_summary = "Cohort A showed significant improvement compared to Cohort B..."
        mock_api_call.return_value = expected_summary
        
        # Sample cohort data
        cohort_a_data = {
            'sample_size': 150,
            'mean_outcome': 75.5,
            'mean_compliance': 88.2,
            'adverse_events': 12,
            'adverse_event_rate': 8.0
        }
        
        cohort_b_data = {
            'sample_size': 145,
            'mean_outcome': 68.3,
            'mean_compliance': 85.7,
            'adverse_events': 18,
            'adverse_event_rate': 12.4
        }
        
        statistical_results = {
            'p_value_outcome': 0.032,
            't_statistic': 2.45,
            'confidence_interval': [1.2, 13.2]
        }
        
        # Generate summary
        result = self.genai.generate_cohort_comparison_summary(
            cohort_a_data, cohort_b_data, statistical_results
        )
        
        # Verify result
        self.assertEqual(result, expected_summary)
        mock_api_call.assert_called_once()
        
        # Verify API call parameters
        call_args = mock_api_call.call_args
        self.assertEqual(call_args[1]['temperature'], 0.3)
        self.assertEqual(call_args[1]['max_tokens'], 800)
    
    @patch.object(GenAIInterface, '_make_api_call')
    def test_generate_scenario_simulation_summary(self, mock_api_call):
        """Test scenario simulation summary generation."""
        expected_summary = "Dosage adjustment from 25mg to 50mg shows positive risk-benefit profile..."
        mock_api_call.return_value = expected_summary
        
        # Sample simulation data
        simulation_params = {
            'patient_id': 'SIM001',
            'current_dosage': 25,
            'proposed_dosage': 50,
            'scenario_type': 'dosage_increase'
        }
        
        simulation_results = {
            'outcome_change': '+15.2%',
            'risk_level': 'Low',
            'confidence_interval': [10.1, 20.3],
            'additional_metrics': {'safety_score': 0.85}
        }
        
        # Generate summary
        result = self.genai.generate_scenario_simulation_summary(
            simulation_params, simulation_results
        )
        
        # Verify result
        self.assertEqual(result, expected_summary)
        mock_api_call.assert_called_once()
        
        # Verify API call parameters
        call_args = mock_api_call.call_args
        self.assertEqual(call_args[1]['temperature'], 0.3)
        self.assertEqual(call_args[1]['max_tokens'], 600)
    
    @patch.object(GenAIInterface, '_make_api_call')
    def test_generate_regulatory_summary(self, mock_api_call):
        """Test regulatory summary generation."""
        expected_summary = """Paragraph 1: Study Design...
        Paragraph 2: Efficacy Results...
        Paragraph 3: Safety Profile..."""
        mock_api_call.return_value = expected_summary
        
        # Sample regulatory data
        trial_data_summary = {
            'study_design': 'Randomized, double-blind, placebo-controlled',
            'duration_weeks': 12,
            'total_patients': 300
        }
        
        key_findings = {
            'primary_endpoint_met': True,
            'p_value': 0.005,
            'effect_size': 12.5
        }
        
        safety_data = {
            'total_adverse_events': 45,
            'serious_adverse_events': 3,
            'discontinuation_rate': 5.2
        }
        
        # Generate regulatory summary
        result = self.genai.generate_regulatory_summary(
            trial_data_summary, key_findings, safety_data
        )
        
        # Verify result
        self.assertEqual(result, expected_summary)
        mock_api_call.assert_called_once()
        
        # Verify API call parameters for regulatory compliance
        call_args = mock_api_call.call_args
        self.assertEqual(call_args[1]['temperature'], 0.2)  # High consistency for regulatory
        self.assertEqual(call_args[1]['max_tokens'], 1200)
    
    @patch.object(GenAIInterface, '_make_api_call')
    def test_extract_adverse_events_with_valid_json(self, mock_api_call):
        """Test adverse event extraction with valid JSON response."""
        # Sample adverse events response
        adverse_events = [
            {
                "event": "Headache",
                "severity": "mild",
                "relationship": "possibly_related",
                "action_taken": "No action required",
                "outcome": "resolved"
            },
            {
                "event": "Nausea",
                "severity": "moderate",
                "relationship": "related",
                "action_taken": "Medication with food",
                "outcome": "ongoing"
            }
        ]
        
        mock_api_call.return_value = json.dumps(adverse_events)
        
        # Test adverse event extraction
        clinical_text = "Patient reports headache and some nausea after taking medication."
        result = self.genai.extract_adverse_events_from_text(clinical_text)
        
        # Verify result
        self.assertEqual(result, adverse_events)
        mock_api_call.assert_called_once()
        
        # Verify API call parameters
        call_args = mock_api_call.call_args
        self.assertEqual(call_args[1]['temperature'], 0.2)  # High consistency for safety
    
    @patch.object(GenAIInterface, '_make_api_call')
    def test_extract_adverse_events_with_invalid_json(self, mock_api_call):
        """Test adverse event extraction with invalid JSON (fallback)."""
        # Mock invalid JSON response
        mock_api_call.return_value = "No adverse events found in text"
        
        # Test adverse event extraction
        clinical_text = "Patient feels great today with no complaints."
        result = self.genai.extract_adverse_events_from_text(clinical_text)
        
        # Verify fallback to empty list
        self.assertEqual(result, [])
        mock_api_call.assert_called_once()
    
    @patch.object(GenAIInterface, '_make_api_call')
    def test_extract_adverse_events_no_events_found(self, mock_api_call):
        """Test adverse event extraction when no events are found."""
        # Mock empty array response
        mock_api_call.return_value = "[]"
        
        # Test adverse event extraction
        clinical_text = "Patient shows excellent progress with no side effects."
        result = self.genai.extract_adverse_events_from_text(clinical_text)
        
        # Verify empty result
        self.assertEqual(result, [])
        mock_api_call.assert_called_once()
    
    @patch.object(GenAIInterface, '_make_api_call')
    def test_generate_clinical_insights(self, mock_api_call):
        """Test clinical insights generation."""
        expected_insights = """Strategic Clinical Insights Report:
        1. Overall Performance: Positive trends observed...
        2. Safety Profile: Acceptable risk profile...
        3. Recommendations: Continue current protocol..."""
        
        mock_api_call.return_value = expected_insights
        
        # Sample data for insights generation
        data_summary = {
            'total_patients': 500,
            'completion_rate': 94.2,
            'primary_endpoint_success': True
        }
        
        analysis_results = {
            'efficacy_score': 78.5,
            'safety_score': 92.1,
            'compliance_rate': 88.7
        }
        
        # Generate insights
        result = self.genai.generate_clinical_insights(data_summary, analysis_results)
        
        # Verify result
        self.assertEqual(result, expected_insights)
        mock_api_call.assert_called_once()
        
        # Verify API call parameters for strategic analysis
        call_args = mock_api_call.call_args
        self.assertEqual(call_args[1]['temperature'], 0.4)  # Higher creativity for insights
        self.assertEqual(call_args[1]['max_tokens'], 1000)
    
    @patch.object(GenAIInterface, '_make_api_call')
    def test_api_call_error_handling_in_analysis_methods(self, mock_api_call):
        """Test error handling in analysis methods when API calls fail."""
        # Mock API call to raise an exception
        mock_api_call.side_effect = Exception("API communication error")
        
        # Test that exceptions are properly propagated
        with self.assertRaises(Exception):
            self.genai.analyze_doctor_notes(self.sample_notes)
        
        with self.assertRaises(Exception):
            self.genai.generate_cohort_comparison_summary({}, {}, {})
        
        with self.assertRaises(Exception):
            self.genai.generate_scenario_simulation_summary({}, {})
        
        with self.assertRaises(Exception):
            self.genai.generate_regulatory_summary({}, {}, {})
        
        with self.assertRaises(Exception):
            self.genai.generate_clinical_insights({}, {})
        
        # Adverse event extraction should return empty list on error
        result = self.genai.extract_adverse_events_from_text("test text")
        self.assertEqual(result, [])


class TestGenAIInterfaceIntegration(unittest.TestCase):
    """
    Integration tests for GenAI Interface.
    
    These tests focus on end-to-end functionality and integration
    between different components of the GenAI Interface.
    """
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.genai = GenAIInterface(api_key="test-integration-key")
    
    def test_analysis_result_dataclass_functionality(self):
        """Test AnalysisResult dataclass functionality."""
        # Create an AnalysisResult instance
        result = AnalysisResult(
            summary="Test summary",
            themes=["Theme 1", "Theme 2"],
            adverse_events=["Event 1"],
            recommendations=["Rec 1", "Rec 2"],
            confidence_score=0.75,
            raw_response="Raw API response"
        )
        
        # Test dataclass functionality
        self.assertEqual(result.summary, "Test summary")
        self.assertEqual(len(result.themes), 2)
        self.assertEqual(len(result.adverse_events), 1)
        self.assertEqual(len(result.recommendations), 2)
        self.assertEqual(result.confidence_score, 0.75)
        self.assertEqual(result.raw_response, "Raw API response")
        
        # Test dataclass conversion to dict
        result_dict = asdict(result)
        self.assertIsInstance(result_dict, dict)
        self.assertIn('summary', result_dict)
        self.assertIn('themes', result_dict)
    
    def test_configuration_parameter_override(self):
        """Test that API call parameters can be overridden."""
        with patch.object(self.genai, 'client') as mock_client:
            # Mock successful response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Test response"
            mock_client.chat.completions.create.return_value = mock_response
            
            # Make API call with custom parameters
            messages = [{"role": "user", "content": "Test"}]
            result = self.genai._make_api_call(
                messages, 
                temperature=0.8, 
                max_tokens=500
            )
            
            # Verify custom parameters were used
            call_args = mock_client.chat.completions.create.call_args[1]
            self.assertEqual(call_args['temperature'], 0.8)
            self.assertEqual(call_args['max_tokens'], 500)
            self.assertEqual(result, "Test response")
    
    @patch('genai_interface.logger')
    def test_logging_functionality(self, mock_logger):
        """Test that logging is properly implemented."""
        # Test initialization logging (updated for new format)
        GenAIInterface(api_key="test-log-key", model="gpt-4")
        mock_logger.info.assert_called_with("GenAI Interface initialized with standard OpenAI - Model: gpt-4")
        
        # Test warning logging for JSON parsing failure
        with patch.object(GenAIInterface, '_make_api_call') as mock_api_call:
            mock_api_call.return_value = "Invalid JSON"
            genai = GenAIInterface(api_key="test-key")
            
            result = genai.analyze_doctor_notes(["Test note"])
            mock_logger.warning.assert_called_with("Failed to parse JSON response, returning raw text")


class TestGenAIInterfaceEdgeCases(unittest.TestCase):
    """
    Edge case tests for GenAI Interface.
    
    These tests cover unusual scenarios, boundary conditions,
    and potential error situations.
    """
    
    def setUp(self):
        """Set up edge case test fixtures."""
        self.genai = GenAIInterface(api_key="test-edge-key")
    
    def test_empty_notes_analysis(self):
        """Test doctor notes analysis with empty notes list."""
        with patch.object(self.genai, '_make_api_call') as mock_api_call:
            mock_api_call.return_value = json.dumps({
                "summary": "No notes provided",
                "themes": [],
                "adverse_events": [],
                "recommendations": [],
                "confidence_score": 0.0
            })
            
            result = self.genai.analyze_doctor_notes([])
            
            self.assertIsInstance(result, AnalysisResult)
            self.assertEqual(result.summary, "No notes provided")
            self.assertEqual(result.themes, [])
    
    def test_very_long_clinical_text(self):
        """Test adverse event extraction with very long clinical text."""
        # Create a very long clinical text
        long_text = "Patient reports mild symptoms. " * 1000
        
        with patch.object(self.genai, '_make_api_call') as mock_api_call:
            mock_api_call.return_value = "[]"
            
            result = self.genai.extract_adverse_events_from_text(long_text)
            
            # Should handle long text without issues
            self.assertEqual(result, [])
            mock_api_call.assert_called_once()
    
    def test_special_characters_in_text(self):
        """Test handling of special characters in clinical text."""
        special_text = "Patient reports symptoms: headache (5/10), nausea & vomiting, fatigue—severe."
        
        with patch.object(self.genai, '_make_api_call') as mock_api_call:
            mock_api_call.return_value = "[]"
            
            result = self.genai.extract_adverse_events_from_text(special_text)
            
            # Should handle special characters without issues
            self.assertEqual(result, [])
            mock_api_call.assert_called_once()
    
    def test_none_values_in_patient_context(self):
        """Test handling of None values in patient context."""
        patient_context = {
            'patient_id': None,
            'dosage_mg': None,
            'compliance_pct': None,
            'cohort': None
        }
        
        with patch.object(self.genai, '_make_api_call') as mock_api_call:
            mock_api_call.return_value = json.dumps({
                "summary": "Analysis with None context",
                "themes": [],
                "adverse_events": [],
                "recommendations": [],
                "confidence_score": 0.5
            })
            
            result = self.genai.analyze_doctor_notes(["Test note"], patient_context)
            
            # Should handle None values gracefully
            self.assertIsInstance(result, AnalysisResult)
            mock_api_call.assert_called_once()


if __name__ == '__main__':
    """
    Main execution block for running the tests.
    
    This block configures test execution and provides detailed output
    for debugging and validation purposes.
    """
    # Configure logging for test execution
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing
    
    # Create test suite combining all test classes
    test_suite = unittest.TestSuite()
    
    # Add all test classes to the suite
    test_classes = [
        TestGenAIInterface,
        TestGenAIInterfaceIntegration, 
        TestGenAIInterfaceEdgeCases
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run the tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    print(f"Exit code: {exit_code}")