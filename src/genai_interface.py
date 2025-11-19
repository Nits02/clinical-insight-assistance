"""
GenAI Interface Module for Clinical Insights Assistant

This module acts as the bridge between the application and large language models (LLMs),
specifically OpenAI's GPT models. It encapsulates the logic for making API calls, handling
prompts, and processing responses for various GenAI tasks.

Purpose:
• Interact with OpenAI's API for text generation and understanding.
• Abstract away API details, providing a clean interface for other modules.
• Handle prompt engineering for specific clinical tasks.

This module provides the essential interface for all GenAI-powered functionality that
the application can leverage the power of LLMs for complex tasks like summarization
and insight generation.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import openai
from openai import OpenAI, AzureOpenAI
import re

# Configure logging for monitoring GenAI operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """
    Data class to hold analysis results from GenAI operations.
    
    This structured approach ensures consistent return types across different
    GenAI analysis functions and makes it easier to handle results in the application.
    """
    summary: str              # Concise summary of the analysis
    themes: List[str]         # Main themes or patterns identified
    adverse_events: List[str] # List of adverse events found
    recommendations: List[str] # Clinical recommendations
    confidence_score: float   # Confidence level of the analysis (0.0-1.0)
    raw_response: str         # Raw LLM response for debugging/audit purposes


class GenAIInterface:
    """
    Interface class for interacting with Large Language Models for clinical data analysis.
    
    This class serves as the central hub for all GenAI operations, providing:
    - Clean abstraction over OpenAI API complexity
    - Consistent error handling and retry logic
    - Specialized methods for different clinical analysis tasks
    - Prompt engineering optimized for clinical contexts
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo", provider: Optional[str] = None):
        """
        Initialize the GenAI Interface with support for both OpenAI and Azure OpenAI.
        
        Sets up the appropriate OpenAI client (standard or Azure) and configuration 
        for clinical data analysis. The initialization handles API key management 
        and establishes default parameters optimized for medical/clinical text analysis.
        
        Args:
            api_key (str, optional): API key. If not provided, will use environment variables.
            model (str): Model to use for analysis (default: gpt-3.5-turbo for standard OpenAI).
            provider (str, optional): Provider type ('openai' or 'azure'). If not provided, uses OPENAI_PROVIDER env var.
        
        Raises:
            ValueError: If required configuration is missing for the selected provider.
        """
        # Determine provider from parameter or environment variable
        self.provider = provider or os.getenv('OPENAI_PROVIDER', 'openai').lower()
        
        if self.provider == 'azure':
            # Azure OpenAI configuration
            self.api_key = api_key or os.getenv('AZURE_OPENAI_API_KEY')
            self.api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01')
            self.azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
            self.deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4o-mini-2024-07-18')
            self.model = self.deployment_name  # For Azure, model is the deployment name
            
            # Validate Azure configuration
            if not self.api_key:
                raise ValueError("Azure OpenAI API key not provided. Set AZURE_OPENAI_API_KEY environment variable or pass api_key parameter.")
            if not self.azure_endpoint:
                raise ValueError("Azure OpenAI endpoint not provided. Set AZURE_OPENAI_ENDPOINT environment variable.")
            
            # Initialize Azure OpenAI client
            self.client = AzureOpenAI(
                api_key=self.api_key,  # DIAL API Key
                api_version=self.api_version,
                azure_endpoint=self.azure_endpoint  # https://ai-proxy.lab.epam.com
            )
            
            logger.info(f"GenAI Interface initialized with Azure OpenAI - Deployment: {self.deployment_name}")
            
        else:
            # Standard OpenAI configuration
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            self.model = model
            
            # Validate OpenAI configuration
            if not self.api_key:
                raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
            
            # Initialize standard OpenAI client
            self.client = OpenAI(api_key=self.api_key)
            
            logger.info(f"GenAI Interface initialized with standard OpenAI - Model: {self.model}")
        
        # Load default configuration optimized for clinical analysis
        self.config = self._get_default_config()
        
        # Test connection after configuration is loaded
        self._validate_connection()
    
    def _validate_connection(self) -> bool:
        """
        Validate API connection immediately after initialization.
        
        This method performs a quick test call to ensure the API credentials
        and endpoint are working correctly, preventing issues later during analysis.
        
        Returns:
            bool: True if connection is successful
            
        Raises:
            Exception: If connection validation fails
        """
        try:
            logger.info("Validating API connection...")
            
            # Simple test message to validate connection
            test_messages = [{"role": "user", "content": "Hello, respond with 'API connection successful'"}]
            
            # Make a quick test call with reduced timeout
            response = self._make_api_call(test_messages, max_tokens=50, timeout=15)
            
            if "successful" in response.lower() or "hello" in response.lower():
                logger.info("✅ API connection validation successful")
                return True
            else:
                logger.warning(f"⚠️ Unexpected response during validation: {response}")
                return True  # Still consider it successful if we got a response
                
        except Exception as e:
            error_msg = f"API connection validation failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            raise Exception(f"Failed to connect to OpenAI API. {error_msg}")
    
    def _get_default_config(self) -> Dict:
        """
        Get default configuration for GenAI operations.
        
        These parameters are optimized for clinical text analysis:
        - Lower temperature for more consistent medical analysis
        - Appropriate token limits for clinical summaries
        - Error handling and retry configuration
        
        Returns:
            Dict: Default configuration parameters optimized for clinical analysis.
        """
        return {
            'max_tokens': 1000,           # Sufficient for clinical summaries
            'temperature': 0.3,           # Lower temperature for more consistent medical analysis
            'top_p': 0.9,                # Nucleus sampling parameter
            'frequency_penalty': 0.0,     # No frequency penalty for medical terminology
            'presence_penalty': 0.0,      # No presence penalty for clinical terms
            'timeout': 30,                # 30-second timeout for API calls
            'max_retries': 3              # Retry failed API calls up to 3 times
        }
    
    def _make_api_call(self, messages: List[Dict], **kwargs) -> str:
        """
        Make an API call to OpenAI with error handling and retries.
        
        This method abstracts away the complexity of API communication and provides
        robust error handling with automatic retries. It ensures reliable communication
        with the OpenAI API even in case of temporary network issues or rate limiting.
        
        Args:
            messages (List[Dict]): List of message dictionaries for the conversation context.
            **kwargs: Additional parameters for the API call (overrides default config).
            
        Returns:
            str: Response content from the API, cleaned and ready for processing.
            
        Raises:
            Exception: If API call fails after all retry attempts are exhausted.
        """
        import time
        
        # Merge default configuration with any custom parameters provided
        api_params = {**self.config, **kwargs}
        api_params['model'] = self.model
        api_params['messages'] = messages
        
        # Extract retry configuration before making the API call
        max_retries = api_params.pop('max_retries', 3)
        
        # Implement retry logic with exponential backoff for robust API communication
        for attempt in range(max_retries):
            try:
                # Add timeout to prevent hanging
                api_params['timeout'] = 30  # 30 second timeout
                
                # Make the actual API call to OpenAI
                response = self.client.chat.completions.create(**api_params)
                
                # Validate response
                if not response.choices or not response.choices[0].message.content:
                    raise Exception("Empty response received from API")
                
                # Extract and return the response content
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"API call attempt {attempt + 1} failed: {error_msg}")
                
                # Check for specific error types that shouldn't be retried
                if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
                    raise Exception(f"Authentication error: {error_msg}. Please check your DIAL API key.")
                
                if "not found" in error_msg.lower() or "404" in error_msg:
                    raise Exception(f"Model or endpoint not found: {error_msg}. Please check your deployment configuration.")
                
                # If this was the last attempt, raise the exception
                if attempt == max_retries - 1:
                    raise Exception(f"API call failed after {max_retries} attempts: {error_msg}")
                
                # Exponential backoff: wait longer between retries
                wait_time = (2 ** attempt) + 1  # 2, 5, 9 seconds
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
    
    def analyze_doctor_notes(self, notes: List[str], patient_context: Optional[Dict] = None) -> AnalysisResult:
        """
        Analyze doctor notes to extract themes, adverse events, and insights.
        
        This method performs comprehensive analysis of clinical notes using NLP techniques
        powered by LLMs. It extracts structured information from unstructured clinical text,
        including safety signals, treatment effectiveness, and compliance issues.
        
        Args:
            notes (List[str]): List of doctor notes to analyze (chronological order preferred).
            patient_context (Dict, optional): Additional patient context for better analysis:
                - patient_id: Patient identifier
                - dosage_mg: Current medication dosage
                - compliance_pct: Patient compliance rate
                - cohort: Treatment group assignment
            
        Returns:
            AnalysisResult: Structured analysis results including summary, themes, and recommendations.
            
        Raises:
            Exception: If the analysis fails due to API errors or other issues.
        """
        # Prepare contextual information to enhance analysis quality
        context_str = ""
        if patient_context:
            context_str = f"""
Patient Context:
- Patient ID: {patient_context.get('patient_id', 'Unknown')}
- Current Dosage: {patient_context.get('dosage_mg', 'Unknown')} mg
- Compliance Rate: {patient_context.get('compliance_pct', 'Unknown')}%
- Cohort: {patient_context.get('cohort', 'Unknown')}
"""
        
        # Combine all notes into a structured format for analysis
        combined_notes = "\n".join([f"Note {i+1}: {note}" for i, note in enumerate(notes)])
        
        # Construct specialized prompt for clinical note analysis
        # This prompt is engineered to extract specific clinical information
        prompt = f"""
You are a clinical data analyst specializing in pharmaceutical trials. Analyze the following doctor notes and provide a comprehensive analysis.

{context_str}

Doctor Notes:
{combined_notes}

Please provide your analysis in the following JSON format:
{{
    "summary": "A concise summary of the patient's overall condition and progress",
    "themes": ["List of main themes or patterns identified in the notes"],
    "adverse_events": ["List of any adverse events or side effects mentioned"],
    "recommendations": ["List of clinical recommendations based on the analysis"],
    "confidence_score": 0.85
}}

Focus on:
1. Patient safety and adverse events
2. Treatment efficacy and patient response
3. Compliance issues
4. Any concerning patterns or trends
5. Clinical significance of observations

Ensure your analysis is objective, evidence-based, and clinically relevant.
"""
        
        # Create message structure for the conversation with the LLM
        messages = [
            {"role": "system", "content": "You are an expert clinical data analyst with extensive experience in pharmaceutical trials and patient safety."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Make API call with lower temperature for consistent medical analysis
            response = self._make_api_call(messages, temperature=0.2)
            
            # Parse JSON response and create structured result
            try:
                # Clean response by removing markdown code blocks if present
                cleaned_response = response.strip()
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]  # Remove ```json
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]  # Remove ```
                cleaned_response = cleaned_response.strip()
                
                parsed_response = json.loads(cleaned_response)
                return AnalysisResult(
                    summary=parsed_response.get('summary', ''),
                    themes=parsed_response.get('themes', []),
                    adverse_events=parsed_response.get('adverse_events', []),
                    recommendations=parsed_response.get('recommendations', []),
                    confidence_score=parsed_response.get('confidence_score', 0.0),
                    raw_response=response
                )
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails - return raw response
                logger.warning("Failed to parse JSON response, returning raw text")
                return AnalysisResult(
                    summary=response,
                    themes=[],
                    adverse_events=[],
                    recommendations=[],
                    confidence_score=0.5,  # Lower confidence for unparsed response
                    raw_response=response
                )
                
        except Exception as e:
            logger.error(f"Error analyzing doctor notes: {str(e)}")
            raise
    
    def generate_cohort_comparison_summary(self, cohort_a_data: Dict, cohort_b_data: Dict, 
                                         statistical_results: Dict) -> str:
        """
        Generate a natural language summary of cohort comparison results.
        
        This method transforms statistical analysis results into comprehensive,
        clinically-relevant narratives suitable for regulatory submissions or
        medical publications. It interprets statistical significance in clinical context.
        
        Args:
            cohort_a_data (Dict): Summary statistics for cohort A (control or treatment group).
            cohort_b_data (Dict): Summary statistics for cohort B (comparison group).
            statistical_results (Dict): Statistical test results (p-values, confidence intervals, etc.).
            
        Returns:
            str: Natural language summary suitable for clinical and regulatory audiences.
            
        Raises:
            Exception: If summary generation fails due to API errors.
        """
        # Construct specialized prompt for cohort comparison analysis
        # This prompt is designed to generate regulatory-quality summaries
        prompt = f"""
You are a clinical biostatistician analyzing pharmaceutical trial data. Generate a comprehensive summary of the cohort comparison results.

Cohort A Statistics:
- Sample Size: {cohort_a_data.get('sample_size', 'N/A')}
- Mean Outcome Score: {cohort_a_data.get('mean_outcome', 'N/A')}
- Mean Compliance: {cohort_a_data.get('mean_compliance', 'N/A')}%
- Adverse Events: {cohort_a_data.get('adverse_events', 'N/A')} ({cohort_a_data.get('adverse_event_rate', 'N/A')}%)

Cohort B Statistics:
- Sample Size: {cohort_b_data.get('sample_size', 'N/A')}
- Mean Outcome Score: {cohort_b_data.get('mean_outcome', 'N/A')}
- Mean Compliance: {cohort_b_data.get('mean_compliance', 'N/A')}%
- Adverse Events: {cohort_b_data.get('adverse_events', 'N/A')} ({cohort_b_data.get('adverse_event_rate', 'N/A')}%)

Statistical Test Results:
{json.dumps(statistical_results, indent=2)}

Please provide a clinical interpretation that includes:
1. Key differences between cohorts
2. Statistical significance of findings
3. Clinical significance and implications
4. Safety considerations
5. Recommendations for further analysis or action

Write in a professional, clinical tone suitable for regulatory submission or medical publication.
"""
        
        # Create conversation context with appropriate system role
        messages = [
            {"role": "system", "content": "You are an expert clinical biostatistician with extensive experience in pharmaceutical trials and regulatory submissions."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Generate summary with moderate temperature for balanced creativity and consistency
            response = self._make_api_call(messages, temperature=0.3, max_tokens=800)
            return response
        except Exception as e:
            logger.error(f"Error generating cohort comparison summary: {str(e)}")
            raise
    
    def generate_scenario_simulation_summary(self, simulation_params: Dict, 
                                           simulation_results: Dict) -> str:
        """
        Generate a summary of scenario simulation results.
        
        This method creates clinical decision-support summaries for dosage adjustment
        scenarios, helping healthcare providers understand the potential impact of
        treatment modifications on patient outcomes and safety.
        
        Args:
            simulation_params (Dict): Parameters used in the simulation:
                - patient_id: Patient identifier
                - current_dosage: Current medication dosage
                - proposed_dosage: Proposed new dosage
                - scenario_type: Type of simulation performed
            simulation_results (Dict): Results of the simulation:
                - outcome_change: Predicted change in outcomes
                - risk_level: Risk assessment
                - confidence_interval: Statistical confidence bounds
                
        Returns:
            str: Clinical decision-support summary for healthcare providers.
            
        Raises:
            Exception: If summary generation fails due to API errors.
        """
        # Construct prompt for scenario simulation analysis
        # Focus on clinical decision-making and patient safety
        prompt = f"""
You are a clinical pharmacologist analyzing dosage adjustment scenarios. Provide a comprehensive summary of the simulation results.

Simulation Parameters:
- Patient ID: {simulation_params.get('patient_id', 'N/A')}
- Current Dosage: {simulation_params.get('current_dosage', 'N/A')} mg
- Proposed Dosage: {simulation_params.get('proposed_dosage', 'N/A')} mg
- Simulation Type: {simulation_params.get('scenario_type', 'N/A')}

Simulation Results:
- Predicted Outcome Change: {simulation_results.get('outcome_change', 'N/A')}
- Risk Assessment: {simulation_results.get('risk_level', 'N/A')}
- Confidence Interval: {simulation_results.get('confidence_interval', 'N/A')}
- Additional Metrics: {json.dumps(simulation_results.get('additional_metrics', {}), indent=2)}

Please provide an analysis that includes:
1. Expected clinical impact of the dosage change
2. Risk-benefit assessment
3. Patient safety considerations
4. Monitoring recommendations
5. Alternative scenarios to consider

Write in a clinical decision-support format suitable for healthcare providers.
"""
        
        # Create conversation with clinical pharmacology expertise
        messages = [
            {"role": "system", "content": "You are an expert clinical pharmacologist specializing in dosage optimization and patient safety."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Generate analysis with moderate creativity for comprehensive recommendations
            response = self._make_api_call(messages, temperature=0.3, max_tokens=600)
            return response
        except Exception as e:
            logger.error(f"Error generating scenario simulation summary: {str(e)}")
            raise
    
    def generate_regulatory_summary(self, trial_data_summary: Dict, 
                                  key_findings: Dict, safety_data: Dict) -> str:
        """
        Generate a regulatory-ready summary aligned with FDA expectations.
        
        This method creates formal clinical study summaries that meet regulatory
        standards for FDA submissions. It follows the standard 3-paragraph structure
        commonly used in regulatory documents.
        
        Args:
            trial_data_summary (Dict): Summary of trial data and demographics.
            key_findings (Dict): Key efficacy and safety findings from the trial.
            safety_data (Dict): Detailed safety analysis results and adverse events.
            
        Returns:
            str: FDA-style regulatory summary in standard 3-paragraph format.
            
        Raises:
            Exception: If regulatory summary generation fails.
        """
        # Construct regulatory-specific prompt following FDA guidelines
        # This prompt ensures compliance with regulatory submission standards
        prompt = f"""
You are a regulatory affairs specialist preparing a clinical study summary for FDA submission. Generate a comprehensive 3-paragraph summary following FDA guidelines.

Trial Data Summary:
{json.dumps(trial_data_summary, indent=2)}

Key Findings:
{json.dumps(key_findings, indent=2)}

Safety Data:
{json.dumps(safety_data, indent=2)}

Generate exactly 3 paragraphs following this structure:

Paragraph 1: Study Design and Demographics
- Study design, duration, and objectives
- Patient population and demographics
- Primary and secondary endpoints

Paragraph 2: Efficacy Results
- Primary endpoint results with statistical significance
- Secondary endpoint results
- Clinical significance and interpretation

Paragraph 3: Safety Profile
- Adverse event summary and rates
- Serious adverse events
- Safety conclusions and risk-benefit assessment

Use precise medical terminology, include statistical measures where appropriate, and maintain the formal tone expected in regulatory submissions. Ensure all claims are supported by the provided data.
"""
        
        # Create conversation with regulatory affairs expertise
        messages = [
            {"role": "system", "content": "You are an expert regulatory affairs specialist with extensive experience in FDA submissions and clinical study reports."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Generate regulatory summary with high consistency (low temperature)
            response = self._make_api_call(messages, temperature=0.2, max_tokens=1200)
            return response
        except Exception as e:
            logger.error(f"Error generating regulatory summary: {str(e)}")
            raise
    
    def extract_adverse_events_from_text(self, text: str) -> List[Dict]:
        """
        Extract adverse events mentioned in clinical text.
        
        This method uses NLP techniques to identify and classify adverse events
        from unstructured clinical text. It provides structured extraction of
        safety signals that can be used for pharmacovigilance and safety analysis.
        
        Args:
            text (str): Clinical text to analyze (doctor notes, patient reports, etc.).
            
        Returns:
            List[Dict]: List of extracted adverse events with structured details:
                - event: Name of the adverse event
                - severity: Severity classification (mild/moderate/severe)
                - relationship: Causal relationship assessment
                - action_taken: Any corrective actions taken
                - outcome: Current status of the event
        """
        # Construct specialized prompt for adverse event extraction
        # This prompt is designed for pharmacovigilance and safety analysis
        prompt = f"""
You are a clinical safety specialist. Analyze the following clinical text and extract any adverse events or side effects mentioned.

Clinical Text:
{text}

For each adverse event found, provide the following information in JSON format:
[
    {{
        "event": "Name of the adverse event",
        "severity": "mild/moderate/severe",
        "relationship": "related/possibly_related/unrelated",
        "action_taken": "Description of any action taken",
        "outcome": "resolved/ongoing/unknown"
    }}
]

If no adverse events are found, return an empty array [].

Focus on:
- Explicit mentions of side effects or adverse reactions
- Symptoms that could be drug-related
- Any safety concerns mentioned
- Actions taken in response to adverse events
"""
        
        # Create conversation with clinical safety expertise
        messages = [
            {"role": "system", "content": "You are an expert clinical safety specialist with extensive experience in adverse event identification and classification."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Extract adverse events with high consistency
            response = self._make_api_call(messages, temperature=0.2)
            
            # Parse JSON response and validate structure
            try:
                # Clean response by removing markdown code blocks if present
                cleaned_response = response.strip()
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]  # Remove ```json
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]  # Remove ```
                cleaned_response = cleaned_response.strip()
                
                adverse_events = json.loads(cleaned_response)
                return adverse_events if isinstance(adverse_events, list) else []
            except json.JSONDecodeError:
                logger.warning("Failed to parse adverse events JSON, returning empty list")
                return []
                
        except Exception as e:
            logger.error(f"Error extracting adverse events: {str(e)}")
            return []
    
    def generate_clinical_insights(self, data_summary: Dict, analysis_results: Dict) -> str:
        """
        Generate high-level clinical insights from analysis results.
        
        This method synthesizes multiple analysis results into strategic insights
        suitable for senior management, clinical teams, and decision-makers. It
        provides a comprehensive view of trial performance and strategic recommendations.
        
        Args:
            data_summary (Dict): Summary of the clinical data and key metrics.
            analysis_results (Dict): Results from various analyses performed on the data.
            
        Returns:
            str: Comprehensive clinical insights report with strategic recommendations.
            
        Raises:
            Exception: If insight generation fails due to API errors.
        """
        # Construct prompt for high-level strategic analysis
        # This prompt focuses on actionable insights and strategic recommendations
        prompt = f"""
You are a senior clinical researcher and data scientist. Based on the following clinical trial data and analysis results, provide strategic insights and recommendations.

Data Summary:
{json.dumps(data_summary, indent=2)}

Analysis Results:
{json.dumps(analysis_results, indent=2)}

Please provide insights covering:
1. Overall trial performance and key trends
2. Patient safety profile and risk assessment
3. Efficacy signals and clinical significance
4. Operational insights (compliance, data quality)
5. Strategic recommendations for next steps
6. Areas requiring further investigation

Structure your response as a comprehensive clinical insights report suitable for senior management and clinical teams.
"""
        
        # Create conversation with senior clinical research expertise
        messages = [
            {"role": "system", "content": "You are a senior clinical researcher with extensive experience in pharmaceutical development and clinical trial management."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Generate insights with moderate creativity for comprehensive recommendations
            response = self._make_api_call(messages, temperature=0.4, max_tokens=1000)
            return response
        except Exception as e:
            logger.error(f"Error generating clinical insights: {str(e)}")
            raise
    
    def generate_insights(self, prompt: str) -> str:
        """
        Generate insights from any given text prompt using AI analysis.
        
        This is a general-purpose method for analyzing any clinical text and generating
        insights, recommendations, and analysis. It's designed to be flexible and work
        with various types of clinical documentation.
        
        Args:
            prompt (str): The text prompt or clinical content to analyze.
            
        Returns:
            str: AI-generated insights and analysis of the provided text.
            
        Raises:
            Exception: If insight generation fails due to API errors.
        """
        try:
            # Create a comprehensive analysis prompt
            analysis_prompt = [
                {
                    "role": "system", 
                    "content": """You are an expert clinical researcher and data analyst specializing in clinical trials and medical data analysis. 
                    Your task is to analyze clinical text and provide comprehensive insights including:
                    - Key findings and observations
                    - Potential issues or concerns
                    - Clinical recommendations
                    - Risk assessments
                    - Next steps or follow-up actions
                    
                    Provide structured, actionable insights that would be valuable for clinical teams and researchers."""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Generate insights using the API call
            response = self._make_api_call(analysis_prompt)
            
            logger.info("Successfully generated insights from text analysis")
            return response
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            raise Exception(f"Failed to generate insights: {str(e)}")
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models from the provider.
        
        For Azure OpenAI, this makes a GET request to the models endpoint.
        For standard OpenAI, it returns common models (since the API requires billing to list models).
        
        Returns:
            List[str]: List of available model names/deployment names.
        """
        if self.provider == 'azure':
            try:
                import requests
                models_url = f"{self.azure_endpoint}/openai/models"
                headers = {"Authorization": f"Bearer {self.api_key}"}
                response = requests.get(models_url, headers=headers)
                if response.status_code == 200:
                    models_data = response.json()
                    return [model['id'] for model in models_data.get('data', [])]
                else:
                    logger.warning(f"Failed to fetch models: {response.status_code}")
                    return [self.deployment_name]  # Return current deployment as fallback
            except Exception as e:
                logger.error(f"Error fetching available models: {str(e)}")
                return [self.deployment_name]
        else:
            # Return common OpenAI models
            return ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini"]


def main():
    """
    Main function for testing the GenAI Interface.
    
    This function provides a simple test of the GenAI interface functionality,
    demonstrating how to use the doctor notes analysis feature. It serves as
    both a test and an example of how to integrate the GenAI interface into
    larger applications.
    
    The main function tests:
    - Interface initialization with API key handling
    - Doctor notes analysis with patient context
    - Error handling for missing API keys
    """
    # Load environment variables from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Environment variables loaded from .env file")
    except ImportError:
        print("⚠️  python-dotenv not installed, using system environment variables")
    
    # Test the GenAI interface initialization and basic functionality
    try:
        # Initialize GenAI interface (will automatically detect provider from environment)
        genai = GenAIInterface()
        
        # Show provider information
        print(f"🔗 Using provider: {genai.provider.upper()}")
        if genai.provider == 'azure':
            print(f"📍 Azure endpoint: {genai.azure_endpoint}")
            print(f"🚀 Deployment: {genai.deployment_name}")
        else:
            print(f"🤖 Model: {genai.model}")
        
        # Prepare sample clinical data for testing
        sample_notes = [
            "Patient stable, no complaints.",
            "Mild headache reported, advised rest.",
            "Some nausea reported, will monitor closely."
        ]
        
        # Sample patient context to enhance analysis quality
        patient_context = {
            'patient_id': 'P001',
            'dosage_mg': 50,
            'compliance_pct': 85.0,
            'cohort': 'A'
        }
        
        # Test doctor notes analysis functionality
        print("\n🧪 Testing doctor notes analysis...")
        analysis_result = genai.analyze_doctor_notes(sample_notes, patient_context)
        
        # Display results for verification
        print(f"\n📋 Analysis Results:")
        print(f"Summary: {analysis_result.summary}")
        print(f"Themes: {analysis_result.themes}")
        print(f"Adverse Events: {analysis_result.adverse_events}")
        print(f"Confidence: {analysis_result.confidence_score}")
        
    except Exception as e:
        print(f"❌ Error testing GenAI Interface: {str(e)}")
        provider = os.getenv('OPENAI_PROVIDER', 'openai').lower()
        if provider == 'azure':
            print("Make sure AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables are set.")
        else:
            print("Make sure OPENAI_API_KEY environment variable is set.")


# Execute main function when script is run directly
if __name__ == "__main__":
    main()