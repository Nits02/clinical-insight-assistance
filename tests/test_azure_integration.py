"""
Integration test for Azure OpenAI configuration.

This test script provides validation of Azure OpenAI configuration
and connectivity with the EPAM company API keys. Unlike unit tests,
this performs actual API calls to verify real integration.

Usage:
    python tests/test_azure_integration.py

Purpose:
    - Validates Azure OpenAI environment configuration
    - Tests real API connectivity with EPAM's Azure OpenAI proxy
    - Provides diagnostic information for troubleshooting
    - Performs live API calls to verify integration works
"""

import os
import sys
from dotenv import load_dotenv

# Add the src directory to the Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_environment_configuration():
    """Test that all required environment variables are properly configured."""
    print("🔧 Azure OpenAI Configuration Test")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Check provider configuration
    provider = os.getenv('OPENAI_PROVIDER', 'openai')
    print(f"Provider: {provider}")
    
    if provider.lower() != 'azure':
        print("⚠️  Provider is set to 'openai'. Change OPENAI_PROVIDER to 'azure' to test Azure OpenAI.")
        return False
    
    # Check Azure-specific configuration
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
    api_version = os.getenv('AZURE_OPENAI_API_VERSION')
    
    print(f"API Key: {'✅ Set' if api_key else '❌ Missing'}")
    print(f"Endpoint: {endpoint if endpoint else '❌ Missing'}")
    print(f"Deployment: {deployment if deployment else '❌ Missing'}")
    print(f"API Version: {api_version if api_version else '❌ Missing'}")
    
    # Validate all required configuration is present
    if not all([api_key, endpoint, deployment]):
        print("\n❌ Missing required configuration. Please update your .env file with:")
        print("- AZURE_OPENAI_API_KEY (your DIAL API key)")
        print("- AZURE_OPENAI_ENDPOINT (should be set to https://ai-proxy.lab.epam.com)")
        print("- AZURE_OPENAI_DEPLOYMENT_NAME (should be set to gpt-4o-mini-2024-07-18)")
        return False
    
    return True


def test_azure_openai_connectivity():
    """Test actual connectivity to Azure OpenAI endpoint."""
    print("\n🧪 Testing Azure OpenAI connection...")
    
    try:
        from openai import AzureOpenAI
        
        # Initialize Azure OpenAI client with environment configuration
        client = AzureOpenAI(
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
        )
        
        # Test with a simple message to verify connectivity
        response = client.chat.completions.create(
            model=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
            temperature=0,
            max_tokens=50,
            timeout=15,  # 15 second timeout
            messages=[
                {
                    "role": "user",
                    "content": "Hello! Please respond with 'Azure OpenAI connection successful!'",
                },
            ],
        )
        
        print("✅ Connection successful!")
        print(f"Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        return False


def test_genai_interface_integration():
    """Test GenAI Interface integration with Azure OpenAI."""
    print("\n🔬 Testing GenAI Interface with Azure OpenAI...")
    
    try:
        from genai_interface import GenAIInterface
        
        # Initialize GenAI interface (should auto-detect Azure configuration)
        genai = GenAIInterface()
        
        # Verify it's using Azure provider
        if genai.provider != 'azure':
            print(f"❌ Expected Azure provider, got: {genai.provider}")
            return False
        
        print(f"✅ GenAI Interface initialized with Azure provider")
        print(f"📍 Endpoint: {genai.azure_endpoint}")
        print(f"🚀 Deployment: {genai.deployment_name}")
        
        # Test a simple analysis
        sample_notes = ["Patient stable with no complaints."]
        analysis_result = genai.analyze_doctor_notes(sample_notes)
        
        print("✅ GenAI Interface integration successful!")
        print(f"📊 Analysis summary (first 100 chars): {analysis_result.summary[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ GenAI Interface integration failed: {str(e)}")
        return False


def main():
    """Main test execution function."""
    print("🧪 Azure OpenAI Integration Test Suite")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Test 1: Environment Configuration
    print("\n1️⃣ Testing Environment Configuration...")
    config_test = test_environment_configuration()
    all_tests_passed &= config_test
    
    if not config_test:
        print("\n❌ Configuration test failed. Please fix configuration before proceeding.")
        return False
    
    # Test 2: Azure OpenAI Connectivity
    print("\n2️⃣ Testing Azure OpenAI Connectivity...")
    connectivity_test = test_azure_openai_connectivity()
    all_tests_passed &= connectivity_test
    
    # Test 3: GenAI Interface Integration
    print("\n3️⃣ Testing GenAI Interface Integration...")
    integration_test = test_genai_interface_integration()
    all_tests_passed &= integration_test
    
    # Final summary
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 All integration tests passed! Azure OpenAI is fully operational.")
        print("✅ Your clinical insight assistance system is ready for production use.")
    else:
        print("❌ Some integration tests failed. Please review the errors above.")
        print("🔧 Check your .env configuration and API key validity.")
    
    print("=" * 60)
    return all_tests_passed


if __name__ == "__main__":
    """Execute integration tests when run directly."""
    success = main()
    exit_code = 0 if success else 1
    print(f"Exit code: {exit_code}")
    sys.exit(exit_code)