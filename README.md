# 🏥 Clinical Insight Assistance

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-green.svg)](https://pandas.pydata.org)
[![Tests](https://img.shields.io/badge/Tests-54%20Total-brightgreen.svg)](tests/)
[![GenAI](https://img.shields.io/badge/GenAI-Azure%20OpenAI-blue.svg)](src/genai_interface.py)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive AI-powered project for providing clinical insights and assistance through advanced data processing, analysis, and GenAI-powered recommendations for clinical trial data.

## � **Project Statistics**

| Component | Count | Status |
|-----------|-------|--------|
| 🧪 **Total Tests** | 54 | ✅ All Passing |
| 📊 **Data Loader Tests** | 26 | ✅ All Passing |
| 🤖 **GenAI Interface Tests** | 25 | ✅ All Passing |
| 🔗 **Integration Tests** | 3 | ✅ All Passing |
| 📁 **Source Modules** | 2 | ✅ Production Ready |
| 🌐 **AI Providers** | 2 | ✅ Azure + OpenAI |
| 📋 **Dependencies** | 30+ | ✅ Latest Versions |

## �📋 Table of Contents

- [🚀 Features](#-features)
- [🏗️ Project Structure](#️-project-structure)
- [⚙️ Installation](#️-installation)
- [🔧 Quick Start](#-quick-start)
- [📊 Data Loader Module](#-data-loader-module)
- [� GenAI Interface Module](#-genai-interface-module)
- [�🧪 Testing](#-testing)
- [📈 Usage Examples](#-usage-examples)
- [🔍 Data Analysis](#-data-analysis)
- [🛠️ Development](#️-development)
- [📚 API Reference](#-api-reference)
- [🤝 Contributing](#-contributing)

## 🚀 Features

### 🔬 **Clinical Data Processing**
- ✅ **Multi-format Data Loading** - Support for CSV, Excel, and JSON files
- ✅ **Intelligent Data Validation** - Comprehensive structure and type validation
- ✅ **Smart Data Cleaning** - Automated missing value handling and outlier detection
- ✅ **Metadata Generation** - Automatic statistical summaries and insights

### 📊 **Advanced Analytics**
- ✅ **Patient Tracking** - Individual patient progress monitoring
- ✅ **Cohort Analysis** - Treatment group comparisons and analysis
- ✅ **Temporal Analysis** - Time-based data filtering and trends
- ✅ **Statistical Insights** - Comprehensive summary statistics

### 🧪 **Testing & Quality**
- ✅ **Synthetic Data Generation** - Realistic test data creation
- ✅ **Comprehensive Testing** - Full unit and integration test coverage
- ✅ **Error Handling** - Robust exception handling and logging
- ✅ **Production Ready** - Scalable and maintainable architecture

### 🤖 **AI & GenAI Integration**
- ✅ **Azure OpenAI Integration** - Enterprise-grade AI with EPAM proxy support
- ✅ **Clinical Text Analysis** - Doctor notes analysis and adverse event extraction
- ✅ **Regulatory Summaries** - FDA-style clinical study summaries
- ✅ **GenAI Interface** - Comprehensive AI-powered clinical insights
- ✅ **Data Visualization** - Plotly, Matplotlib, and Seaborn integration
- ✅ **Statistical Analysis** - SciPy and Statsmodels support

## 🏗️ Project Structure

```
clinical-insight-assistance/
├── 📁 data/                    # Data files and datasets
│   ├── .gitkeep               # Keeps directory in version control
│   └── clinical_trial_data.csv # Generated synthetic clinical data
├── 📁 notebooks/              # Jupyter notebooks for analysis
│   └── .gitkeep               # Keeps directory in version control
├── 📁 src/                    # Source code files
│   ├── data_loader.py         # 🔧 Core data loading and processing module
│   └── genai_interface.py     # 🤖 GenAI interface for AI-powered analysis
├── 📁 tests/                  # Test files
│   ├── .gitkeep               # Keeps directory in version control
│   ├── test_data_loader.py    # 🧪 Data loader unit tests (26 tests)
│   ├── test_genai_interface.py # 🤖 GenAI interface unit tests (25 tests)
│   └── test_azure_integration.py # 🔗 Azure OpenAI integration tests (3 tests)
├── 📄 .env                    # Environment configuration (API keys, settings)
├── 📄 requirements.txt        # Project dependencies
├── 📄 .gitignore             # Git ignore rules
└── 📄 README.md              # Project documentation
```

## ⚙️ Installation

### 📋 Prerequisites
- 🐍 Python 3.12+ 
- 📦 pip (Python package manager)
- 🔄 Git (for cloning the repository)

### 🛠️ Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Nits02/clinical-insight-assistance.git
   cd clinical-insight-assistance
   ```

2. **Activate Virtual Environment**
   ```bash
   # Activate the virtual environment
   source .venv/bin/activate
   
   # Verify activation (you should see (.venv) in your prompt)
   which python
   ```

3. **Install Dependencies**
   ```bash
   # Install all required packages
   pip install -r requirements.txt
   
   # Verify installation
   pip list | grep -E "(pandas|numpy|streamlit|openai)"
   ```

4. **Configure Environment Variables**
   ```bash
   # Copy and configure environment variables
   cp .env.example .env  # If you have a template
   
   # Edit .env file with your API keys
   nano .env
   ```

   **Required Configuration:**
   ```bash
   # For Azure OpenAI (EPAM Company Keys)
   OPENAI_PROVIDER=azure
   AZURE_OPENAI_API_KEY=your_dial_api_key_here
   AZURE_OPENAI_ENDPOINT=https://ai-proxy.lab.epam.com
   AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini-2024-07-18
   
   # For Standard OpenAI (Alternative)
   OPENAI_PROVIDER=openai
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## 🔧 Quick Start

### 🚀 **Run the Data Loader Demo**

```bash
# Navigate to project directory
cd clinical-insight-assistance

# Run the data loader module directly
.venv/bin/python src/data_loader.py
```

**Expected Output:**
```
INFO:__main__:Generated synthetic dataset with 700 records for 50 patients
INFO:__main__:Loading data from data/clinical_trial_data.csv
INFO:__main__:Successfully loaded 700 records from data/clinical_trial_data.csv

Data Summary:
Total records: 700
Total patients: 50
Adverse events: 77 (11.0%)
Mean compliance: 86.0%
Mean outcome score: 82.9
```

### 🤖 **Test GenAI Integration**

```bash
# Test Azure OpenAI integration
.venv/bin/python tests/test_azure_integration.py

# Test GenAI interface directly  
.venv/bin/python src/genai_interface.py
```

### 📊 **Interactive Testing**

```bash
# Test data processing functionality
.venv/bin/python -c "
import sys; sys.path.append('src')
from data_loader import ClinicalDataLoader

loader = ClinicalDataLoader()
data = loader.generate_synthetic_data(num_patients=10, days_per_patient=5)
print(f'✅ Generated {len(data)} records for clinical analysis')
"

# Test AI-powered analysis (requires API key configuration)
.venv/bin/python -c "
from dotenv import load_dotenv
load_dotenv()
import sys; sys.path.append('src')
from genai_interface import GenAIInterface

try:
    genai = GenAIInterface()
    print(f'✅ GenAI Interface initialized with {genai.provider.upper()} provider')
    print(f'🚀 Model/Deployment: {genai.model}')
except Exception as e:
    print(f'⚠️  GenAI setup needed: {str(e)}')
"
```

## 📊 Data Loader Module

### 🔧 **ClinicalDataLoader Class**

The core module (`src/data_loader.py`) provides comprehensive clinical data processing capabilities:

#### **🚀 Key Features:**
- **📥 Data Loading**: CSV, Excel, JSON file support
- **🔍 Data Validation**: Structure and type checking
- **🧹 Data Cleaning**: Missing value handling and preprocessing
- **📈 Analytics**: Statistical summaries and metadata generation
- **🔎 Filtering**: Patient, cohort, and date-based filtering
- **🎲 Synthetic Data**: Realistic test data generation

#### **📋 Required Data Structure:**
```python
required_columns = [
    'patient_id',        # 👤 Patient identifier (P001, P002, etc.)
    'trial_day',         # 📅 Day of trial (1, 2, 3, ...)
    'dosage_mg',         # 💊 Medication dosage in mg
    'compliance_pct',    # 📊 Patient compliance percentage (0-100)
    'adverse_event_flag',# ⚠️  Boolean flag for adverse events
    'doctor_notes',      # 📝 Clinical observations and notes
    'outcome_score',     # 🎯 Treatment outcome score (0-100)
    'cohort',           # 👥 Treatment group (A, B, etc.)
    'visit_date'        # 📆 Visit date (YYYY-MM-DD)
]
```

#### **💡 Usage Example:**
```python
from src.data_loader import ClinicalDataLoader

# Initialize the loader
loader = ClinicalDataLoader()

# Generate synthetic data
synthetic_data = loader.generate_synthetic_data(
    num_patients=50, 
    days_per_patient=14,
    output_path='data/clinical_trial_data.csv'
)

# Load and process data
data = loader.load_data('data/clinical_trial_data.csv')

# Get summary statistics
summary = loader.get_summary_statistics()
print(f"Total patients: {summary['metadata']['total_patients']}")
```

## � GenAI Interface Module

### 🚀 **AI-Powered Clinical Analysis**

The GenAI Interface module (`src/genai_interface.py`) provides comprehensive AI-powered clinical analysis capabilities using Azure OpenAI and standard OpenAI APIs.

#### **🎯 Key Features:**
- **🏥 Doctor Notes Analysis** - AI-powered analysis of clinical notes
- **📊 Cohort Comparisons** - Natural language summaries of statistical comparisons
- **📋 Regulatory Summaries** - FDA-style clinical study summaries
- **⚠️ Adverse Event Extraction** - Automated safety signal detection
- **🧪 Scenario Simulations** - Dosage adjustment impact analysis
- **🔍 Clinical Insights** - Strategic recommendations for decision-makers

#### **🔧 Provider Support:**
- ✅ **Azure OpenAI** - Enterprise integration with EPAM proxy
- ✅ **Standard OpenAI** - Direct OpenAI API integration
- ✅ **Automatic Detection** - Environment-based provider selection
- ✅ **Error Handling** - Robust retry logic and fallback mechanisms

#### **💡 Quick Start:**
```python
from src.genai_interface import GenAIInterface

# Initialize GenAI interface (auto-detects provider from .env)
genai = GenAIInterface()

# Analyze doctor notes
notes = [
    "Patient stable, no complaints.",
    "Mild headache reported, advised rest.",
    "Some nausea reported, will monitor closely."
]

patient_context = {
    'patient_id': 'P001',
    'dosage_mg': 50,
    'compliance_pct': 85.0,
    'cohort': 'A'
}

# Get AI-powered analysis
analysis = genai.analyze_doctor_notes(notes, patient_context)
print(f"Summary: {analysis.summary}")
print(f"Adverse Events: {analysis.adverse_events}")
print(f"Recommendations: {analysis.recommendations}")
```

#### **🧪 Test the GenAI Interface:**
```bash
# Test with your configured API keys
.venv/bin/python src/genai_interface.py

# Expected output:
# ✅ Environment variables loaded from .env file
# 🔗 Using provider: AZURE
# 📍 Azure endpoint: https://ai-proxy.lab.epam.com
# 🚀 Deployment: gpt-4o-mini-2024-07-18
# 🧪 Testing doctor notes analysis...
# 📋 Analysis Results: [Comprehensive clinical analysis]
```

#### **🔍 Available Methods:**
- **`analyze_doctor_notes()`** - Comprehensive clinical note analysis
- **`generate_cohort_comparison_summary()`** - Statistical comparison narratives
- **`generate_scenario_simulation_summary()`** - Dosage adjustment analysis
- **`generate_regulatory_summary()`** - FDA-style clinical summaries
- **`extract_adverse_events_from_text()`** - Safety signal extraction
- **`generate_clinical_insights()`** - Strategic decision support

#### **⚙️ Configuration:**
The GenAI interface automatically detects your provider configuration from `.env`:

```bash
# Azure OpenAI Configuration (Recommended for Enterprise)
OPENAI_PROVIDER=azure
AZURE_OPENAI_API_KEY=your_dial_api_key_here
AZURE_OPENAI_ENDPOINT=https://ai-proxy.lab.epam.com
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini-2024-07-18

# Standard OpenAI Configuration (Alternative)
OPENAI_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
```

## �🧪 Testing

### 🚀 **Run All Tests**

#### **🧪 Complete Test Suite (54 Tests Total)**
```bash
# Run all tests with pytest
.venv/bin/python -m pytest tests/ -v

# Or run individual test suites
.venv/bin/python tests/test_data_loader.py      # 26 tests
.venv/bin/python tests/test_genai_interface.py  # 25 tests  
.venv/bin/python tests/test_azure_integration.py # 3 tests
```

#### **📊 Individual Test Results:**

**1. Data Loader Tests (26 tests)**
```bash
.venv/bin/python tests/test_data_loader.py
```
```
Running ClinicalDataLoader Unit Tests...
==================================================
test_clean_data_type_conversion ... ok
test_dataloader_alias ... ok
test_generate_metadata ... ok
...
----------------------------------------------------------------------
Ran 26 tests in 0.185s

OK
✅ All tests passed successfully!
```

**2. GenAI Interface Tests (25 tests)**
```bash
.venv/bin/python tests/test_genai_interface.py
```
```
test_analyze_doctor_notes_with_valid_json_response ... ok
test_api_call_success ... ok
test_extract_adverse_events_with_valid_json ... ok
...
----------------------------------------------------------------------
Ran 25 tests in 0.149s

OK
✅ All tests passed successfully!
```

**3. Azure OpenAI Integration Tests (3 tests)**
```bash
.venv/bin/python tests/test_azure_integration.py
```
```
🧪 Azure OpenAI Integration Test Suite
============================================================

1️⃣ Testing Environment Configuration... ✅
2️⃣ Testing Azure OpenAI Connectivity... ✅  
3️⃣ Testing GenAI Interface Integration... ✅

🎉 All integration tests passed! Azure OpenAI is fully operational.
✅ Your clinical insight assistance system is ready for production use.
============================================================
Exit code: 0
```

### 🧪 **Test Categories**

#### **📊 Data Loader Tests (test_data_loader.py - 26 tests)**

**1. 🔧 Core Functionality (TestClinicalDataLoader)**
- ✅ **Initialization**: Default and custom configuration
- ✅ **Data Loading**: Success cases and error handling
- ✅ **Data Validation**: Structure and type validation
- ✅ **Data Cleaning**: Type conversion and missing values
- ✅ **Metadata Generation**: Statistical summaries
- ✅ **Filtering**: Patient, cohort, and date filtering
- ✅ **Synthetic Data**: Data generation with file output

**2. 🔄 Integration Tests (TestDataLoaderIntegration)**
- ✅ **End-to-End Workflow**: Complete data processing pipeline
- ✅ **Data Cleaning**: Realistic messy data processing
- ✅ **File I/O**: CSV reading and writing operations

**3. ⚠️ Edge Case Tests (TestDataLoaderEdgeCases)**
- ✅ **Empty Files**: Handling empty datasets
- ✅ **Single Records**: Minimal data processing
- ✅ **Non-existent Data**: Error scenarios

#### **🤖 GenAI Interface Tests (test_genai_interface.py - 25 tests)**

**1. 🔧 Core GenAI Functionality (TestGenAIInterface)**
- ✅ **Initialization**: API key handling, provider detection
- ✅ **API Communication**: Success, retries, error handling
- ✅ **Doctor Notes Analysis**: JSON parsing, fallback behavior
- ✅ **Analysis Methods**: All clinical analysis functions
- ✅ **Adverse Event Extraction**: Safety signal detection
- ✅ **Error Handling**: Comprehensive error scenarios

**2. 🔄 Integration Tests (TestGenAIInterfaceIntegration)**
- ✅ **AnalysisResult Dataclass**: Functionality and conversion
- ✅ **Parameter Overrides**: Custom API call parameters
- ✅ **Logging**: Proper logging implementation

**3. ⚠️ Edge Cases (TestGenAIInterfaceEdgeCases)**
- ✅ **Empty Inputs**: Handling of empty data
- ✅ **Long Text**: Processing of very long clinical text
- ✅ **Special Characters**: Unicode and special character handling
- ✅ **None Values**: Graceful handling of missing data

#### **🔗 Azure Integration Tests (test_azure_integration.py - 3 tests)**

**1. 🔧 Environment Configuration Test**
- ✅ **Config Validation**: All Azure OpenAI environment variables
- ✅ **Provider Detection**: Automatic provider selection
- ✅ **Error Diagnostics**: Clear feedback on missing configuration

**2. 🌐 Azure OpenAI Connectivity Test**  
- ✅ **Real API Calls**: Live connection to EPAM endpoint
- ✅ **Authentication**: API key validation
- ✅ **Response Validation**: Successful AI response processing

**3. 🧪 GenAI Interface Integration Test**
- ✅ **End-to-End**: Complete workflow with Azure OpenAI
- ✅ **Clinical Analysis**: Real AI-powered doctor notes analysis
- ✅ **Production Readiness**: Full system validation

### 🧪 **Alternative Testing Methods**

```bash
# Using unittest module
.venv/bin/python -m unittest tests.test_data_loader -v
.venv/bin/python -m unittest tests.test_genai_interface -v

# Using pytest (if installed) 
pytest tests/ -v                    # All tests
pytest tests/test_data_loader.py -v # Data loader tests only
pytest tests/test_genai_interface.py -v # GenAI tests only

# Run specific test classes
.venv/bin/python -m unittest tests.test_data_loader.TestClinicalDataLoader -v
.venv/bin/python -m unittest tests.test_genai_interface.TestGenAIInterface -v

# Integration and connectivity tests
.venv/bin/python tests/test_azure_integration.py  # Azure OpenAI integration
```

### 🚀 **Quick Test Commands Summary**

```bash
# 🎯 Essential Tests (run these for validation)
.venv/bin/python tests/test_data_loader.py      # Data processing (26 tests)
.venv/bin/python tests/test_azure_integration.py # AI integration (3 tests)

# 🧪 Unit Tests (development and debugging)  
.venv/bin/python tests/test_genai_interface.py  # GenAI mocks (25 tests)

# 📊 Complete Test Suite (CI/CD and comprehensive validation)
.venv/bin/python -m pytest tests/ -v            # All 54 tests
```

## 📈 Usage Examples

### 📊 **Data Analysis Workflow**

```bash
# Complete data analysis example
.venv/bin/python -c "
import sys; sys.path.append('src')
from data_loader import ClinicalDataLoader

# Initialize loader
loader = ClinicalDataLoader()

# Load existing data or generate synthetic data
try:
    data = loader.load_data('data/clinical_trial_data.csv')
    print('📊 Loaded existing clinical trial data')
except FileNotFoundError:
    data = loader.generate_synthetic_data(num_patients=50, days_per_patient=14)
    print('🎲 Generated new synthetic data')

# Patient-specific analysis
patient_data = loader.get_patient_data('P001')
print(f'👤 Patient P001 has {len(patient_data)} visits')

# Cohort comparison
cohort_a = loader.get_cohort_data('A')
cohort_b = loader.get_cohort_data('B')
print(f'👥 Cohort A: {len(cohort_a)} records, Cohort B: {len(cohort_b)} records')

# Date range analysis
recent_data = loader.get_date_range_data('2024-01-01', '2024-01-07')
print(f'📅 First week data: {len(recent_data)} records')

# Summary statistics
summary = loader.get_summary_statistics()
metadata = summary['metadata']
print(f'📈 Analysis Summary:')
print(f'   • Total Patients: {metadata[\"total_patients\"]}')
print(f'   • Adverse Events: {metadata[\"adverse_events\"][\"total\"]} ({metadata[\"adverse_events\"][\"percentage\"]:.1f}%)')
print(f'   • Mean Compliance: {metadata[\"compliance_stats\"][\"mean\"]:.1f}%')
print(f'   • Mean Outcome: {metadata[\"outcome_stats\"][\"mean\"]:.1f}')
"
```

### 🔍 **Error Handling Testing**

```bash
# Test error handling capabilities
.venv/bin/python -c "
import sys; sys.path.append('src')
from data_loader import ClinicalDataLoader

loader = ClinicalDataLoader()
print('🧪 Testing error handling...')

# Test file not found
try:
    loader.load_data('non_existent_file.csv')
except FileNotFoundError as e:
    print(f'✅ FileNotFoundError: {e}')

# Test no data loaded
try:
    loader.get_summary_statistics()
except ValueError as e:
    print(f'✅ ValueError: {e}')

print('✅ Error handling works correctly!')
"
```

## 🔍 Data Analysis

### 📊 **Generated Data Quality**

The synthetic data generator creates realistic clinical trial data with:

- **👥 Patient Management**: 50 patients (P001-P050)
- **📅 Temporal Tracking**: 14-day trial periods
- **💊 Dosage Variation**: Realistic dosage levels (50mg, 75mg, 100mg)
- **📈 Compliance Patterns**: Patient-specific adherence behaviors (50-100%)
- **⚠️ Adverse Events**: ~10% occurrence rate with appropriate notes
- **🎯 Outcome Progression**: Treatment effectiveness scoring (40-100)
- **👥 Cohort Distribution**: Balanced treatment groups (A/B)
- **📝 Clinical Notes**: Contextual doctor observations

### 📈 **Data Insights**

```bash
# View generated data structure
head -5 data/clinical_trial_data.csv
```

Sample output:
```csv
patient_id,trial_day,dosage_mg,compliance_pct,adverse_event_flag,doctor_notes,outcome_score,cohort,visit_date
P001,1,50,82.07,0,Blood pressure within normal range.,73.14,A,2024-01-01
P001,2,50,78.88,0,Patient stable no complaints.,73.96,A,2024-01-02
P001,3,50,76.87,0,Symptoms improving with current dosage.,63.30,A,2024-01-03
```

## 🛠️ Development

### 🔧 **Development Environment**

```bash
# Activate development environment
source .venv/bin/activate

# Install development dependencies (if needed)
pip install black flake8 pytest

# Run code formatting
black src/ tests/

# Run linting
flake8 src/ tests/
```

### 📝 **Adding New Features**

1. **🔧 Modify Source Code**: Update `src/data_loader.py`
2. **🧪 Add Tests**: Create tests in `tests/test_data_loader.py`
3. **🏃 Run Tests**: Verify all tests pass
4. **📚 Update Documentation**: Update this README.md

### 🧪 **Testing New Features**

```bash
# Run tests after changes
.venv/bin/python tests/test_data_loader.py

# Run specific test methods
.venv/bin/python -m unittest tests.test_data_loader.TestClinicalDataLoader.test_your_new_feature
```

## 📚 API Reference

### 🔧 **ClinicalDataLoader Methods**

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `__init__(config)` | Initialize loader | `config`: Optional Dict | `None` |
| `load_data(file_path)` | Load data from CSV | `file_path`: str | `pd.DataFrame` |
| `generate_synthetic_data()` | Create test data | `num_patients`, `days_per_patient`, `output_path` | `pd.DataFrame` |
| `get_patient_data(patient_id)` | Filter by patient | `patient_id`: str | `pd.DataFrame` |
| `get_cohort_data(cohort)` | Filter by cohort | `cohort`: str | `pd.DataFrame` |
| `get_date_range_data()` | Filter by date range | `start_date`, `end_date`: str | `pd.DataFrame` |
| `get_summary_statistics()` | Get data summary | None | `Dict` |

### 📊 **Configuration Options**

```python
default_config = {
    'required_columns': [...],      # Essential columns
    'numeric_columns': [...],       # Numeric data columns
    'categorical_columns': [...],   # Category columns
    'boolean_columns': [...],       # True/False columns
    'date_columns': [...],          # Date/time columns
    'text_columns': [...],          # Free text columns
    'compliance_threshold': 70.0,   # Minimum compliance %
    'outcome_threshold': 60.0,      # Minimum outcome score
    'max_missing_percentage': 0.1   # Max missing data (10%)
}
```

## 🤝 Contributing

### 📋 **Contribution Guidelines**

1. **🍴 Fork** the repository
2. **🌿 Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **💻 Make** your changes
4. **🧪 Add** tests for new functionality
5. **✅ Ensure** all tests pass
6. **📝 Update** documentation
7. **🚀 Commit** your changes (`git commit -m 'Add amazing feature'`)
8. **📤 Push** to the branch (`git push origin feature/amazing-feature`)
9. **🔄 Open** a Pull Request

### 🧪 **Testing Requirements**

- ✅ All existing tests must pass
- ✅ New features must include tests
- ✅ Code coverage should not decrease
- ✅ Follow existing code style

### 📋 **Code Style**

- 🐍 Follow PEP 8 Python style guidelines
- 📝 Include comprehensive docstrings
- 🧪 Write meaningful test names
- 💬 Add comments for complex logic

---

## �‍💻 Author & Contact

**Nitesh Sharma**
- 📧 **Email**: [nitesh.sharma@live.com](mailto:nitesh.sharma@live.com)
- 🌐 **Blog**: [The Data Arch](https://thedataarch.com/)
- 💼 **GitHub**: [Nits02](https://github.com/Nits02)
- 🏥 **Specialization**: Clinical Data Analysis & AI Solutions

### 📝 **About The Author**
Passionate about leveraging data science and AI to improve healthcare outcomes. Visit [The Data Arch](https://thedataarch.com/) for insights on data architecture, machine learning, and clinical analytics.

---

## �📞 Support

### 🐛 **Issues & Bug Reports**
- 📋 Use GitHub Issues for bug reports
- 🏷️ Label issues appropriately
- 📝 Provide detailed reproduction steps
- 📧 For urgent issues, contact: [nitesh.sharma@live.com](mailto:nitesh.sharma@live.com)

### 💡 **Feature Requests**
- 🚀 Submit feature requests via GitHub Issues
- 🎯 Describe the use case and expected behavior
- 📊 Include examples if applicable
- 💬 Discuss ideas on [The Data Arch](https://thedataarch.com/)

### 📚 **Documentation**
- 📖 Check this README for usage examples
- 🧪 Review test files for additional examples
- 💻 Explore the source code for detailed implementation
- 🌐 Read more clinical data insights at [The Data Arch](https://thedataarch.com/)

### 🤝 **Professional Inquiries**
- 📧 **Business Contact**: [nitesh.sharma@live.com](mailto:nitesh.sharma@live.com)
- 💼 **Consulting**: Clinical data analysis and AI implementation
- 🎓 **Training**: Data science workshops and clinical analytics training

---

**🎉 Happy Clinical Data Analysis!** 🏥📊🔬

*Developed with ❤️ by [Nitesh Sharma](mailto:nitesh.sharma@live.com) | Visit [The Data Arch](https://thedataarch.com/) for more insights*