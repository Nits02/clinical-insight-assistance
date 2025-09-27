"""
Data Loader Module for Clinical Insights Assistant

This module handles data ingestion, cleaning, and preprocessing for clinical trial data.
It supports loading data from CSV files and provides utilities for data validation and transformation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json

# Configure logging for monitoring data loading operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClinicalDataLoader:
    """
    A class to handle loading and preprocessing of clinical trial data.
    
    This class provides comprehensive functionality for:
    - Loading clinical trial data from CSV files
    - Validating data structure and integrity
    - Cleaning and preprocessing data
    - Generating metadata and summary statistics
    - Filtering data by patient, cohort, or date range
    - Creating synthetic test data
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the ClinicalDataLoader.
        
        Args:
            config (Dict, optional): Configuration dictionary for data loading parameters.
                                   If None, default configuration will be used.
        """
        # Use provided config or get default configuration
        self.config = config or self._get_default_config()
        # Initialize data storage - will hold the loaded DataFrame
        self.data = None
        # Initialize metadata storage - will hold summary information about the data
        self.metadata = {}
        
    def _get_default_config(self) -> Dict:
        """
        Get default configuration for data loading.
        
        This method defines the expected structure of clinical trial data including:
        - Required columns that must be present
        - Data types for each column category
        - Validation thresholds and parameters
        
        Returns:
            Dict: Default configuration parameters for clinical trial data.
        """
        return {
            # Essential columns that must be present in the dataset
            'required_columns': [
                'patient_id', 'trial_day', 'dosage_mg', 'compliance_pct',
                'adverse_event_flag', 'doctor_notes', 'outcome_score', 'cohort', 'visit_date'
            ],
            # Columns that should contain numeric values
            'numeric_columns': ['trial_day', 'dosage_mg', 'compliance_pct', 'outcome_score'],
            # Columns that contain categorical data (identifiers, groups)
            'categorical_columns': ['patient_id', 'cohort'],
            # Columns that should contain boolean values (True/False)
            'boolean_columns': ['adverse_event_flag'],
            # Columns that should contain date/datetime values
            'date_columns': ['visit_date'],
            # Columns that contain free text/notes
            'text_columns': ['doctor_notes'],
            # Validation thresholds
            'compliance_threshold': 70.0,  # Minimum compliance percentage for flagging
            'outcome_threshold': 60.0,     # Minimum outcome score for efficacy assessment
            'max_missing_percentage': 0.1  # Maximum allowed missing data percentage (10%)
        }
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load clinical trial data from a CSV file.
        
        This is the main entry point for data loading. It performs the complete workflow:
        1. File existence check
        2. Data loading from CSV
        3. Structure validation
        4. Data cleaning and preprocessing
        5. Metadata generation
        
        Args:
            file_path (str): Path to the CSV file containing clinical trial data.
            
        Returns:
            pd.DataFrame: Loaded, validated, and cleaned clinical trial data.
            
        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the data does not meet validation requirements.
        """
        try:
            # Step 1: Check if file exists to provide clear error message
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            # Step 2: Load data from CSV file
            logger.info(f"Loading data from {file_path}")
            self.data = pd.read_csv(file_path)
            
            # Step 3: Validate that the data has the expected structure and content
            self._validate_data_structure()
            
            # Step 4: Clean and preprocess the data (handle missing values, convert types, etc.)
            self.data = self._clean_data()
            
            # Step 5: Generate metadata for quick data overview
            self._generate_metadata()
            
            logger.info(f"Successfully loaded {len(self.data)} records from {file_path}")
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _validate_data_structure(self) -> None:
        """
        Validate that the loaded data has the required structure.
        
        This method performs comprehensive data validation:
        - Checks for presence of required columns
        - Validates data types for each column category
        - Checks for reasonable value ranges
        - Warns about excessive missing data
        
        Raises:
            ValueError: If required columns are missing or data structure is invalid.
        """
        # Ensure data has been loaded
        if self.data is None:
            raise ValueError("No data loaded")
        
        # Check that all required columns are present
        missing_columns = set(self.config['required_columns']) - set(self.data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Validate data types and value ranges for each column
        self._validate_data_types()
        
        # Check for excessive missing data across the entire dataset
        total_cells = len(self.data) * len(self.data.columns)
        missing_cells = self.data.isnull().sum().sum()
        missing_percentage = missing_cells / total_cells
        
        if missing_percentage > self.config['max_missing_percentage']:
            logger.warning(f"High percentage of missing data: {missing_percentage:.2%}")
    
    def _validate_data_types(self) -> None:
        """
        Validate data types and value ranges for key columns.
        
        This method checks:
        - Numeric columns contain numeric values
        - Compliance percentages are within 0-100 range
        - Outcome scores are within expected range (0-100)
        """
        # Check that numeric columns actually contain numeric data
        for col in self.config['numeric_columns']:
            if col in self.data.columns:
                if not pd.api.types.is_numeric_dtype(self.data[col]):
                    logger.warning(f"Column {col} should be numeric but contains non-numeric data")
        
        # Validate compliance percentage range (should be 0-100)
        if 'compliance_pct' in self.data.columns:
            invalid_compliance = (self.data['compliance_pct'] < 0) | (self.data['compliance_pct'] > 100)
            if invalid_compliance.any():
                logger.warning(f"Found {invalid_compliance.sum()} records with invalid compliance percentages")
        
        # Validate outcome score range (assuming 0-100 scale for clinical outcomes)
        if 'outcome_score' in self.data.columns:
            invalid_outcomes = (self.data['outcome_score'] < 0) | (self.data['outcome_score'] > 100)
            if invalid_outcomes.any():
                logger.warning(f"Found {invalid_outcomes.sum()} records with invalid outcome scores")
    
    def _clean_data(self) -> pd.DataFrame:
        """
        Clean and preprocess the loaded data.
        
        This method performs comprehensive data cleaning:
        - Converts columns to appropriate data types
        - Handles missing values using appropriate strategies
        - Removes duplicate records
        - Cleans text fields
        
        Returns:
            pd.DataFrame: Cleaned and preprocessed data.
        """
        # Create a copy to avoid modifying original data
        data_cleaned = self.data.copy()
        
        # Convert numeric columns to proper numeric types, coercing errors to NaN
        for col in self.config['numeric_columns']:
            if col in data_cleaned.columns:
                data_cleaned[col] = pd.to_numeric(data_cleaned[col], errors='coerce')
        
        # Convert boolean columns to proper boolean type
        for col in self.config['boolean_columns']:
            if col in data_cleaned.columns:
                data_cleaned[col] = data_cleaned[col].astype(bool)
        
        # Convert date columns to datetime, handling parsing errors gracefully
        for col in self.config['date_columns']:
            if col in data_cleaned.columns:
                data_cleaned[col] = pd.to_datetime(data_cleaned[col], errors='coerce')
        
        # Clean text columns by trimming whitespace and handling empty strings
        for col in self.config['text_columns']:
            if col in data_cleaned.columns:
                data_cleaned[col] = data_cleaned[col].astype(str).str.strip()
                # Replace empty strings with NaN for consistent missing value handling
                data_cleaned[col] = data_cleaned[col].replace('', np.nan)
        
        # Handle missing values using appropriate strategies for each column type
        data_cleaned = self._handle_missing_values(data_cleaned)
        
        # Remove duplicate records and log the number removed
        initial_count = len(data_cleaned)
        data_cleaned = data_cleaned.drop_duplicates()
        if len(data_cleaned) < initial_count:
            logger.info(f"Removed {initial_count - len(data_cleaned)} duplicate records")
        
        return data_cleaned
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset using appropriate strategies.
        
        Strategy by column type:
        - Numeric: Fill with median (robust to outliers)
        - Categorical: Fill with mode (most common value)
        - Boolean: Fill with False (conservative assumption)
        - Text: Fill with placeholder text
        
        Args:
            data (pd.DataFrame): Data with potential missing values.
            
        Returns:
            pd.DataFrame: Data with missing values handled appropriately.
        """
        # Handle missing values in numeric columns using median imputation
        for col in self.config['numeric_columns']:
            if col in data.columns and data[col].isnull().any():
                median_value = data[col].median()
                data[col].fillna(median_value, inplace=True)
                logger.info(f"Filled missing values in {col} with median: {median_value}")
        
        # Handle missing values in categorical columns using mode imputation
        for col in self.config['categorical_columns']:
            if col in data.columns and data[col].isnull().any():
                mode_value = data[col].mode().iloc[0] if not data[col].mode().empty else 'Unknown'
                data[col].fillna(mode_value, inplace=True)
                logger.info(f"Filled missing values in {col} with mode: {mode_value}")
        
        # Handle missing values in boolean columns (assume False for adverse events if not specified)
        for col in self.config['boolean_columns']:
            if col in data.columns and data[col].isnull().any():
                data[col].fillna(False, inplace=True)
                logger.info(f"Filled missing values in {col} with False")
        
        # Handle missing values in text columns with placeholder text
        for col in self.config['text_columns']:
            if col in data.columns and data[col].isnull().any():
                data[col].fillna('No notes available', inplace=True)
                logger.info(f"Filled missing values in {col} with placeholder text")
        
        return data
    
    def _generate_metadata(self) -> None:
        """
        Generate comprehensive metadata about the loaded dataset.
        
        This method creates a summary dictionary containing:
        - Basic dataset statistics (record count, patient count)
        - Date range information
        - Cohort distribution
        - Adverse event statistics
        - Compliance and outcome score statistics
        """
        if self.data is None:
            return
        
        # Generate comprehensive metadata for quick dataset overview
        self.metadata = {
            # Basic dataset information
            'total_records': len(self.data),
            'total_patients': self.data['patient_id'].nunique() if 'patient_id' in self.data.columns else 0,
            
            # Date range information for temporal analysis
            'date_range': {
                'start': self.data['visit_date'].min().isoformat() if 'visit_date' in self.data.columns else None,
                'end': self.data['visit_date'].max().isoformat() if 'visit_date' in self.data.columns else None
            },
            
            # Cohort distribution for group analysis
            'cohorts': self.data['cohort'].value_counts().to_dict() if 'cohort' in self.data.columns else {},
            
            # Adverse event statistics for safety analysis
            'adverse_events': {
                'total': int(self.data['adverse_event_flag'].sum()) if 'adverse_event_flag' in self.data.columns else 0,
                'percentage': float(self.data['adverse_event_flag'].mean() * 100) if 'adverse_event_flag' in self.data.columns else 0
            },
            
            # Compliance statistics for adherence analysis
            'compliance_stats': {
                'mean': float(self.data['compliance_pct'].mean()) if 'compliance_pct' in self.data.columns else 0,
                'median': float(self.data['compliance_pct'].median()) if 'compliance_pct' in self.data.columns else 0,
                'std': float(self.data['compliance_pct'].std()) if 'compliance_pct' in self.data.columns else 0
            },
            
            # Outcome statistics for efficacy analysis
            'outcome_stats': {
                'mean': float(self.data['outcome_score'].mean()) if 'outcome_score' in self.data.columns else 0,
                'median': float(self.data['outcome_score'].median()) if 'outcome_score' in self.data.columns else 0,
                'std': float(self.data['outcome_score'].std()) if 'outcome_score' in self.data.columns else 0
            }
        }
    
    def get_patient_data(self, patient_id: str) -> pd.DataFrame:
        """
        Get all data records for a specific patient.
        
        This method filters the dataset to return only records for the specified patient,
        useful for individual patient analysis and tracking progress over time.
        
        Args:
            patient_id (str): Patient identifier to filter by.
            
        Returns:
            pd.DataFrame: All data records for the specified patient.
            
        Raises:
            ValueError: If no data has been loaded.
        """
        if self.data is None:
            raise ValueError("No data loaded")
        
        return self.data[self.data['patient_id'] == patient_id].copy()
    
    def get_cohort_data(self, cohort: str) -> pd.DataFrame:
        """
        Get all data records for a specific cohort.
        
        This method filters the dataset to return only records for the specified cohort,
        useful for cohort-based analysis and comparison studies.
        
        Args:
            cohort (str): Cohort identifier to filter by.
            
        Returns:
            pd.DataFrame: All data records for the specified cohort.
            
        Raises:
            ValueError: If no data has been loaded.
        """
        if self.data is None:
            raise ValueError("No data loaded")
        
        return self.data[self.data['cohort'] == cohort].copy()
    
    def get_date_range_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get data records within a specific date range.
        
        This method filters the dataset to return only records with visit dates
        within the specified range, useful for temporal analysis.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format.
            end_date (str): End date in YYYY-MM-DD format.
            
        Returns:
            pd.DataFrame: Data records within the specified date range.
            
        Raises:
            ValueError: If no data has been loaded.
        """
        if self.data is None:
            raise ValueError("No data loaded")
        
        # Convert string dates to pandas datetime objects
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Filter data within the date range (inclusive)
        return self.data[
            (self.data['visit_date'] >= start_date) & 
            (self.data['visit_date'] <= end_date)
        ].copy()
    
    def generate_synthetic_data(self, num_patients: int = 200, days_per_patient: int = 30, 
                              output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Generate synthetic clinical trial data for testing and development purposes.
        
        This method creates realistic synthetic data that follows the expected
        clinical trial data structure, including:
        - Patient progression over time
        - Realistic compliance patterns
        - Adverse event occurrence
        - Outcome score improvements
        - Varied doctor notes
        
        Args:
            num_patients (int): Number of patients to generate data for.
            days_per_patient (int): Number of trial days per patient.
            output_path (str, optional): Path to save the generated data as CSV.
            
        Returns:
            pd.DataFrame: Generated synthetic clinical trial data.
        """
        # Set random seed for reproducible synthetic data generation
        np.random.seed(42)
        
        # Generate patient IDs with zero-padded format (P001, P002, etc.)
        patient_ids = [f"P{str(i).zfill(3)}" for i in range(1, num_patients + 1)]
        cohorts = ['A', 'B']  # Two treatment cohorts
        records = []
        
        # Template notes for realistic doctor observations
        notes_templates = [
            "Patient stable, no complaints.",
            "Mild headache reported, advised rest.",
            "Fatigue noted, monitoring ongoing.",
            "Symptoms improving with current dosage.",
            "Adverse reaction observed, dosage adjustment needed.",
            "Patient reports feeling better today.",
            "Some nausea reported, will monitor closely.",
            "Blood pressure within normal range.",
            "Patient compliance excellent this visit.",
            "Minor side effects noted, continuing treatment."
        ]
        
        # Generate data for each patient
        for pid in patient_ids:
            cohort = np.random.choice(cohorts)
            # Each patient has a baseline compliance tendency
            base_compliance = np.random.normal(85, 10)  # Patient-specific baseline compliance
            
            # Generate daily records for each patient
            for day in range(1, days_per_patient + 1):
                # Dosage varies between standard amounts
                dosage = np.random.choice([50, 75, 100])
                
                # Compliance varies around patient baseline with daily variation
                compliance = np.clip(np.random.normal(base_compliance, 5), 50, 100)
                
                # Adverse events occur in ~10% of visits
                adverse_event = np.random.choice([0, 1], p=[0.9, 0.1])
                
                # Simulate outcome score based on multiple factors:
                # - Higher dosage generally improves outcomes
                # - Better compliance improves outcomes
                # - Adverse events worsen outcomes
                base_score = 80 + (dosage - 50) * 0.2 + (compliance - 90) * 0.3 - adverse_event * 15
                
                # Add temporal trend (slight improvement over time in trial)
                temporal_bonus = day * 0.1
                
                # Final outcome with some random variation
                outcome = np.clip(np.random.normal(base_score + temporal_bonus, 5), 40, 100)
                
                # Select appropriate notes based on patient condition
                if adverse_event:
                    notes = np.random.choice([
                        "Adverse reaction observed, dosage adjustment needed.",
                        "Some nausea reported, will monitor closely.",
                        "Minor side effects noted, continuing treatment."
                    ])
                elif outcome > 85:
                    notes = np.random.choice([
                        "Patient reports feeling better today.",
                        "Symptoms improving with current dosage.",
                        "Patient stable, no complaints."
                    ])
                else:
                    notes = np.random.choice(notes_templates)
                
                # Generate visit date based on trial day
                visit_date = pd.Timestamp('2024-01-01') + pd.Timedelta(days=day-1)
                
                # Add record to the dataset
                records.append([
                    pid, day, dosage, compliance, adverse_event, 
                    notes, outcome, cohort, visit_date
                ])
        
        # Create DataFrame with the expected column structure
        df = pd.DataFrame(records, columns=[
            'patient_id', 'trial_day', 'dosage_mg', 'compliance_pct',
            'adverse_event_flag', 'doctor_notes', 'outcome_score', 
            'cohort', 'visit_date'
        ])
        
        # Save to file if output path is provided
        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"Synthetic data saved to {output_path}")
        
        logger.info(f"Generated synthetic dataset with {len(df)} records for {num_patients} patients")
        return df
    
    def get_summary_statistics(self) -> Dict:
        """
        Get comprehensive summary statistics for the loaded data.
        
        This method provides detailed information about the dataset including:
        - Metadata summary
        - Column information (data types, null counts, unique values)
        - Descriptive statistics for numeric columns
        
        Returns:
            Dict: Comprehensive summary statistics and metadata.
            
        Raises:
            ValueError: If no data has been loaded.
        """
        if self.data is None:
            raise ValueError("No data loaded")
        
        return {
            # Previously generated metadata
            'metadata': self.metadata,
            
            # Detailed information about each column
            'column_info': {
                col: {
                    'dtype': str(self.data[col].dtype),
                    'null_count': int(self.data[col].isnull().sum()),
                    'unique_count': int(self.data[col].nunique())
                }
                for col in self.data.columns
            },
            
            # Descriptive statistics for all numeric columns
            'numeric_summary': self.data.select_dtypes(include=[np.number]).describe().to_dict()
        }


def main():
    """
    Main function for testing the ClinicalDataLoader independently.
    
    This function demonstrates the complete workflow:
    1. Initialize the data loader
    2. Generate synthetic test data
    3. Load and process the data
    4. Display summary statistics
    
    This serves as both a test and an example of how to use the module.
    """
    # Initialize data loader with default configuration
    loader = ClinicalDataLoader()
    
    # Generate synthetic data for testing (50 patients, 14 days each)
    synthetic_data = loader.generate_synthetic_data(
        num_patients=50, 
        days_per_patient=14,
        output_path='data/clinical_trial_data.csv'
    )
    
    # Load the generated data to test the complete pipeline
    loaded_data = loader.load_data('data/clinical_trial_data.csv')
    
    # Display summary statistics to verify data loading and processing
    summary = loader.get_summary_statistics()
    print("Data Summary:")
    print(f"Total records: {summary['metadata']['total_records']}")
    print(f"Total patients: {summary['metadata']['total_patients']}")
    print(f"Adverse events: {summary['metadata']['adverse_events']['total']} ({summary['metadata']['adverse_events']['percentage']:.1f}%)")
    print(f"Mean compliance: {summary['metadata']['compliance_stats']['mean']:.1f}%")
    print(f"Mean outcome score: {summary['metadata']['outcome_stats']['mean']:.1f}")


# Create alias for backward compatibility with streamlit_app.py
# This allows existing code that imports DataLoader to continue working
DataLoader = ClinicalDataLoader


# Execute main function when script is run directly
if __name__ == "__main__":
    main()

