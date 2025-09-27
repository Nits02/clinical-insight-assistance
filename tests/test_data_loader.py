"""
Unit tests for the ClinicalDataLoader module.

This test suite covers all major functionality of the ClinicalDataLoader class including:
- Data loading and validation
- Data cleaning and preprocessing
- Metadata generation
- Data filtering methods
- Synthetic data generation
- Error handling scenarios
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add the src directory to the path to import our module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import ClinicalDataLoader, DataLoader


class TestClinicalDataLoader(unittest.TestCase):
    """Test cases for the ClinicalDataLoader class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Initialize a data loader instance for testing
        self.loader = ClinicalDataLoader()
        
        # Create sample valid data for testing
        self.sample_data = pd.DataFrame({
            'patient_id': ['P001', 'P001', 'P002', 'P002'],
            'trial_day': [1, 2, 1, 2],
            'dosage_mg': [50, 50, 75, 75],
            'compliance_pct': [85.0, 90.0, 75.0, 80.0],
            'adverse_event_flag': [False, False, True, False],
            'doctor_notes': ['Patient stable', 'Good progress', 'Minor side effects', 'Improving'],
            'outcome_score': [75.0, 78.0, 65.0, 70.0],
            'cohort': ['A', 'A', 'B', 'B'],
            'visit_date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-01', '2024-01-02'])
        })
        
        # Create sample data with missing values for testing cleaning functionality
        self.data_with_missing = pd.DataFrame({
            'patient_id': ['P001', 'P001', 'P002', None],
            'trial_day': [1, 2, None, 2],
            'dosage_mg': [50, None, 75, 75],
            'compliance_pct': [85.0, 90.0, None, 80.0],
            'adverse_event_flag': [False, None, True, False],
            'doctor_notes': ['Patient stable', '', 'Minor side effects', None],
            'outcome_score': [75.0, 78.0, 65.0, None],
            'cohort': ['A', 'A', None, 'B'],
            'visit_date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-01', '2024-01-02'])
        })
    
    def tearDown(self):
        """Clean up after each test method."""
        # Reset the loader's data and metadata
        self.loader.data = None
        self.loader.metadata = {}
    
    def test_init_default_config(self):
        """Test initialization with default configuration."""
        loader = ClinicalDataLoader()
        
        # Check that default config is loaded
        self.assertIsNotNone(loader.config)
        self.assertIn('required_columns', loader.config)
        self.assertIn('numeric_columns', loader.config)
        
        # Check that data and metadata are initialized as None/empty
        self.assertIsNone(loader.data)
        self.assertEqual(loader.metadata, {})
    
    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        custom_config = {
            'required_columns': ['patient_id', 'outcome'],
            'numeric_columns': ['outcome'],
            'max_missing_percentage': 0.2
        }
        
        loader = ClinicalDataLoader(config=custom_config)
        
        # Check that custom config is used
        self.assertEqual(loader.config, custom_config)
        self.assertEqual(loader.config['max_missing_percentage'], 0.2)
    
    def test_get_default_config(self):
        """Test the default configuration structure."""
        config = self.loader._get_default_config()
        
        # Check that all required keys are present
        required_keys = [
            'required_columns', 'numeric_columns', 'categorical_columns',
            'boolean_columns', 'date_columns', 'text_columns',
            'compliance_threshold', 'outcome_threshold', 'max_missing_percentage'
        ]
        
        for key in required_keys:
            self.assertIn(key, config)
        
        # Check specific values
        self.assertEqual(config['compliance_threshold'], 70.0)
        self.assertEqual(config['outcome_threshold'], 60.0)
        self.assertEqual(config['max_missing_percentage'], 0.1)
    
    def test_load_data_success(self):
        """Test successful data loading from CSV file."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.sample_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            # Load data
            result = self.loader.load_data(temp_file)
            
            # Check that data was loaded successfully
            self.assertIsNotNone(self.loader.data)
            self.assertEqual(len(result), 4)
            self.assertIsInstance(result, pd.DataFrame)
            
            # Check that metadata was generated
            self.assertIsNotNone(self.loader.metadata)
            self.assertIn('total_records', self.loader.metadata)
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file)
    
    def test_load_data_file_not_found(self):
        """Test loading data from non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.loader.load_data('non_existent_file.csv')
    
    def test_load_data_missing_required_columns(self):
        """Test loading data that's missing required columns."""
        # Create data missing required columns
        incomplete_data = pd.DataFrame({
            'patient_id': ['P001', 'P002'],
            'trial_day': [1, 2]
            # Missing other required columns
        })
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            incomplete_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            # Should raise ValueError for missing columns
            with self.assertRaises(ValueError) as context:
                self.loader.load_data(temp_file)
            
            self.assertIn('Missing required columns', str(context.exception))
            
        finally:
            os.unlink(temp_file)
    
    def test_validate_data_structure_no_data(self):
        """Test validation when no data is loaded."""
        with self.assertRaises(ValueError) as context:
            self.loader._validate_data_structure()
        
        self.assertIn('No data loaded', str(context.exception))
    
    def test_validate_data_structure_valid_data(self):
        """Test validation with valid data structure."""
        self.loader.data = self.sample_data
        
        # Should not raise any exceptions
        try:
            self.loader._validate_data_structure()
        except Exception as e:
            self.fail(f"Validation failed unexpectedly: {e}")
    
    @patch('data_loader.logger')
    def test_validate_data_types_warnings(self, mock_logger):
        """Test data type validation warnings."""
        # Create data with invalid compliance percentages
        invalid_data = self.sample_data.copy()
        invalid_data.loc[0, 'compliance_pct'] = 150.0  # Invalid: > 100
        invalid_data.loc[1, 'outcome_score'] = -10.0   # Invalid: < 0
        
        self.loader.data = invalid_data
        self.loader._validate_data_types()
        
        # Check that warnings were logged
        mock_logger.warning.assert_called()
    
    def test_clean_data_type_conversion(self):
        """Test data type conversion during cleaning."""
        # Create data with mixed types
        mixed_data = pd.DataFrame({
            'patient_id': ['P001', 'P002'],
            'trial_day': ['1', '2'],  # String numbers
            'dosage_mg': [50.0, 75.0],
            'compliance_pct': ['85.0', '90.0'],  # String numbers
            'adverse_event_flag': [0, 1],  # Numeric booleans
            'doctor_notes': ['Note 1', 'Note 2'],
            'outcome_score': [75.0, 80.0],
            'cohort': ['A', 'B'],
            'visit_date': ['2024-01-01', '2024-01-02']  # String dates
        })
        
        self.loader.data = mixed_data
        cleaned_data = self.loader._clean_data()
        
        # Check that types were converted correctly
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_data['trial_day']))
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_data['compliance_pct']))
        self.assertTrue(pd.api.types.is_bool_dtype(cleaned_data['adverse_event_flag']))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned_data['visit_date']))
    
    def test_handle_missing_values(self):
        """Test missing value handling strategies."""
        self.loader.data = self.data_with_missing
        cleaned_data = self.loader._handle_missing_values(self.data_with_missing.copy())
        
        # Check that missing values were filled appropriately
        # Numeric columns should be filled with median
        self.assertFalse(cleaned_data['trial_day'].isnull().any())
        self.assertFalse(cleaned_data['compliance_pct'].isnull().any())
        
        # Categorical columns should be filled with mode or 'Unknown'
        self.assertFalse(cleaned_data['patient_id'].isnull().any())
        self.assertFalse(cleaned_data['cohort'].isnull().any())
        
        # Boolean columns should be filled with False
        self.assertFalse(cleaned_data['adverse_event_flag'].isnull().any())
        
        # Text columns should be filled with placeholder
        self.assertFalse(cleaned_data['doctor_notes'].isnull().any())
    
    def test_generate_metadata(self):
        """Test metadata generation."""
        self.loader.data = self.sample_data
        self.loader._generate_metadata()
        
        # Check that metadata contains expected keys
        expected_keys = [
            'total_records', 'total_patients', 'date_range', 'cohorts',
            'adverse_events', 'compliance_stats', 'outcome_stats'
        ]
        
        for key in expected_keys:
            self.assertIn(key, self.loader.metadata)
        
        # Check specific values
        self.assertEqual(self.loader.metadata['total_records'], 4)
        self.assertEqual(self.loader.metadata['total_patients'], 2)
        self.assertIn('A', self.loader.metadata['cohorts'])
        self.assertIn('B', self.loader.metadata['cohorts'])
    
    def test_get_patient_data(self):
        """Test filtering data by patient ID."""
        self.loader.data = self.sample_data
        
        # Get data for patient P001
        patient_data = self.loader.get_patient_data('P001')
        
        # Check that only P001 data is returned
        self.assertEqual(len(patient_data), 2)
        self.assertTrue((patient_data['patient_id'] == 'P001').all())
    
    def test_get_patient_data_no_data_loaded(self):
        """Test getting patient data when no data is loaded."""
        with self.assertRaises(ValueError) as context:
            self.loader.get_patient_data('P001')
        
        self.assertIn('No data loaded', str(context.exception))
    
    def test_get_cohort_data(self):
        """Test filtering data by cohort."""
        self.loader.data = self.sample_data
        
        # Get data for cohort A
        cohort_data = self.loader.get_cohort_data('A')
        
        # Check that only cohort A data is returned
        self.assertEqual(len(cohort_data), 2)
        self.assertTrue((cohort_data['cohort'] == 'A').all())
    
    def test_get_date_range_data(self):
        """Test filtering data by date range."""
        self.loader.data = self.sample_data
        
        # Get data for a specific date range
        date_range_data = self.loader.get_date_range_data('2024-01-01', '2024-01-01')
        
        # Check that only data from 2024-01-01 is returned
        self.assertEqual(len(date_range_data), 2)
        expected_date = pd.to_datetime('2024-01-01').date()
        self.assertTrue((date_range_data['visit_date'].dt.date == expected_date).all())
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        synthetic_data = self.loader.generate_synthetic_data(
            num_patients=10, 
            days_per_patient=5
        )
        
        # Check that correct amount of data was generated
        expected_records = 10 * 5
        self.assertEqual(len(synthetic_data), expected_records)
        
        # Check that all required columns are present
        for col in self.loader.config['required_columns']:
            self.assertIn(col, synthetic_data.columns)
        
        # Check that patient IDs are correctly formatted
        unique_patients = synthetic_data['patient_id'].unique()
        self.assertEqual(len(unique_patients), 10)
        self.assertTrue(all(pid.startswith('P') for pid in unique_patients))
        
        # Check that compliance percentages are within valid range
        self.assertTrue((synthetic_data['compliance_pct'] >= 50).all())
        self.assertTrue((synthetic_data['compliance_pct'] <= 100).all())
        
        # Check that outcome scores are within valid range
        self.assertTrue((synthetic_data['outcome_score'] >= 40).all())
        self.assertTrue((synthetic_data['outcome_score'] <= 100).all())
    
    def test_generate_synthetic_data_with_output_path(self):
        """Test synthetic data generation with file output."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_file = f.name
        
        try:
            synthetic_data = self.loader.generate_synthetic_data(
                num_patients=5,
                days_per_patient=3,
                output_path=temp_file
            )
            
            # Check that file was created
            self.assertTrue(os.path.exists(temp_file))
            
            # Check that file contains the correct data
            loaded_data = pd.read_csv(temp_file)
            self.assertEqual(len(loaded_data), 15)  # 5 patients * 3 days
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_get_summary_statistics(self):
        """Test getting summary statistics."""
        self.loader.data = self.sample_data
        self.loader._generate_metadata()
        
        summary = self.loader.get_summary_statistics()
        
        # Check that summary contains expected sections
        self.assertIn('metadata', summary)
        self.assertIn('column_info', summary)
        self.assertIn('numeric_summary', summary)
        
        # Check column info structure
        for col in self.sample_data.columns:
            self.assertIn(col, summary['column_info'])
            self.assertIn('dtype', summary['column_info'][col])
            self.assertIn('null_count', summary['column_info'][col])
            self.assertIn('unique_count', summary['column_info'][col])
    
    def test_get_summary_statistics_no_data(self):
        """Test getting summary statistics when no data is loaded."""
        with self.assertRaises(ValueError) as context:
            self.loader.get_summary_statistics()
        
        self.assertIn('No data loaded', str(context.exception))
    
    def test_dataloader_alias(self):
        """Test that DataLoader is an alias for ClinicalDataLoader."""
        # Test that the alias exists and points to the same class
        self.assertEqual(DataLoader, ClinicalDataLoader)
        
        # Test that we can instantiate using the alias
        loader_alias = DataLoader()
        self.assertIsInstance(loader_alias, ClinicalDataLoader)
    

class TestDataLoaderIntegration(unittest.TestCase):
    """Integration tests for the complete data loading workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = ClinicalDataLoader()
    
    def test_complete_workflow(self):
        """Test the complete data loading and processing workflow."""
        # Step 1: Generate synthetic data
        synthetic_data = self.loader.generate_synthetic_data(
            num_patients=20,
            days_per_patient=10
        )
        
        # Step 2: Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            synthetic_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            # Step 3: Load data using the loader
            loaded_data = self.loader.load_data(temp_file)
            
            # Step 4: Verify the complete process worked
            self.assertEqual(len(loaded_data), 200)  # 20 patients * 10 days
            self.assertIsNotNone(self.loader.metadata)
            
            # Step 5: Test filtering methods
            patient_data = self.loader.get_patient_data('P001')
            self.assertEqual(len(patient_data), 10)
            
            cohort_data = self.loader.get_cohort_data('A')
            self.assertGreater(len(cohort_data), 0)
            
            # Step 6: Test summary statistics
            summary = self.loader.get_summary_statistics()
            self.assertEqual(summary['metadata']['total_records'], 200)
            
        finally:
            # Clean up
            os.unlink(temp_file)
    
    def test_data_cleaning_integration(self):
        """Test data cleaning with realistic messy data."""
        # Create messy data that needs cleaning
        messy_data = pd.DataFrame({
            'patient_id': ['P001', 'P001', 'P002', 'P002', 'P001'],  # Duplicate row
            'trial_day': [1, 2, 1, 2, 1],  # Duplicate row
            'dosage_mg': [50, 75, None, 100, 50],  # Missing value, duplicate row
            'compliance_pct': [85.0, 120.0, 75.0, -5.0, 85.0],  # Invalid values, duplicate
            'adverse_event_flag': [False, True, None, False, False],  # Missing value, duplicate
            'doctor_notes': ['Good', '  ', 'Side effects', '', 'Good'],  # Empty/whitespace, duplicate
            'outcome_score': [75.0, 150.0, 65.0, 70.0, 75.0],  # Invalid value, duplicate
            'cohort': ['A', 'B', 'A', 'B', 'A'],  # Duplicate row
            'visit_date': ['2024-01-01', '2024-01-02', '2024-01-01', '2024-01-02', '2024-01-01']  # Duplicate row
        })
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            messy_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            # Load and clean the data
            cleaned_data = self.loader.load_data(temp_file)
            
            # Check that duplicates were removed
            self.assertEqual(len(cleaned_data), 4)  # Should remove 1 duplicate row
            
            # Check that missing values were handled
            self.assertFalse(cleaned_data.isnull().any().any())
            
            # Check that data types were converted properly
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned_data['visit_date']))
            self.assertTrue(pd.api.types.is_bool_dtype(cleaned_data['adverse_event_flag']))
            
        finally:
            os.unlink(temp_file)


class TestDataLoaderEdgeCases(unittest.TestCase):
    """Test edge cases and error scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = ClinicalDataLoader()
    
    def test_empty_data_file(self):
        """Test loading an empty CSV file."""
        # Create empty CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("patient_id,trial_day,dosage_mg,compliance_pct,adverse_event_flag,doctor_notes,outcome_score,cohort,visit_date\n")
            temp_file = f.name
        
        try:
            # Should handle empty data gracefully
            loaded_data = self.loader.load_data(temp_file)
            self.assertEqual(len(loaded_data), 0)
            
        finally:
            os.unlink(temp_file)
    
    def test_single_row_data(self):
        """Test loading data with only one row."""
        single_row_data = pd.DataFrame({
            'patient_id': ['P001'],
            'trial_day': [1],
            'dosage_mg': [50],
            'compliance_pct': [85.0],
            'adverse_event_flag': [False],
            'doctor_notes': ['Patient stable'],
            'outcome_score': [75.0],
            'cohort': ['A'],
            'visit_date': ['2024-01-01']
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            single_row_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            loaded_data = self.loader.load_data(temp_file)
            self.assertEqual(len(loaded_data), 1)
            self.assertEqual(self.loader.metadata['total_patients'], 1)
            
        finally:
            os.unlink(temp_file)
    
    def test_get_nonexistent_patient(self):
        """Test getting data for a patient that doesn't exist."""
        self.loader.data = pd.DataFrame({
            'patient_id': ['P001', 'P002'],
            'trial_day': [1, 1],
            'dosage_mg': [50, 75],
            'compliance_pct': [85.0, 90.0],
            'adverse_event_flag': [False, False],
            'doctor_notes': ['Note 1', 'Note 2'],
            'outcome_score': [75.0, 80.0],
            'cohort': ['A', 'B'],
            'visit_date': ['2024-01-01', '2024-01-01']
        })
        
        # Should return empty DataFrame
        result = self.loader.get_patient_data('P999')
        self.assertEqual(len(result), 0)
        self.assertIsInstance(result, pd.DataFrame)


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestClinicalDataLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoaderIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoaderEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    # Run tests when script is executed directly
    print("Running ClinicalDataLoader Unit Tests...")
    print("=" * 50)
    
    result = run_tests()
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed successfully!")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")