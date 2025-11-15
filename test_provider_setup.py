#!/usr/bin/env python
"""
Test script to verify multi-provider LLM setup for TaxoAdapt

Usage:
    python test_provider_setup.py
    python test_provider_setup.py --provider openai
    python test_provider_setup.py --provider azure
    python test_provider_setup.py --provider claude
    python test_provider_setup.py --provider huggingface
"""

import os
import sys
import argparse

def test_imports():
    """Test if all required packages are installed"""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)
    
    packages = {
        'dotenv': 'python-dotenv',
        'openai': 'openai',
        'anthropic': 'anthropic',
        'huggingface_hub': 'huggingface_hub',
    }
    
    missing = []
    for module, package in packages.items():
        try:
            __import__(module)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is NOT installed")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    print("\n✓ All required packages are installed")
    return True


def test_env_file():
    """Test if .env file exists and is configured"""
    print("\n" + "=" * 60)
    print("Testing .env file...")
    print("=" * 60)
    
    if not os.path.exists('.env'):
        print("✗ .env file not found")
        print("Create it with: cp .env.example .env")
        return False
    
    print("✓ .env file exists")
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✓ .env file loaded successfully")
    except Exception as e:
        print(f"✗ Error loading .env file: {e}")
        return False
    
    return True


def test_provider(provider_name=None):
    """Test a specific provider"""
    from dotenv import load_dotenv
    load_dotenv()
    
    if provider_name is None:
        provider_name = os.getenv('DEFAULT_LLM_PROVIDER', 'openai')
    
    print("\n" + "=" * 60)
    print(f"Testing provider: {provider_name}")
    print("=" * 60)
    
    # Check for required credentials
    credentials = {
        'openai': ['OPENAI_API_KEY'],
        'azure': ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_DEPLOYMENT_NAME'],
        'claude': ['ANTHROPIC_API_KEY'],
        'anthropic': ['ANTHROPIC_API_KEY'],
        'huggingface': ['HUGGINGFACE_API_KEY'],
        'hf': ['HUGGINGFACE_API_KEY']
    }
    
    if provider_name not in credentials:
        print(f"✗ Unknown provider: {provider_name}")
        print(f"Valid providers: {', '.join(set(credentials.keys()))}")
        return False
    
    # Check credentials
    missing_creds = []
    for cred in credentials[provider_name]:
        value = os.getenv(cred)
        if not value:
            print(f"✗ {cred} not set in .env")
            missing_creds.append(cred)
        else:
            # Mask the key for security
            masked = value[:8] + "..." if len(value) > 8 else "***"
            print(f"✓ {cred} is set: {masked}")
    
    if missing_creds:
        print(f"\nMissing credentials: {', '.join(missing_creds)}")
        print(f"Add them to your .env file")
        return False
    
    # Try to initialize the provider
    try:
        from api.llm_provider import get_llm_provider
        provider = get_llm_provider(provider_name)
        print(f"✓ Provider initialized successfully")
        print(f"✓ Using model: {provider.get_model_name()}")
    except Exception as e:
        print(f"✗ Error initializing provider: {e}")
        return False
    
    # Try a simple chat (optional - comment out if you don't want to make an API call)
    try:
        print("\nTesting API call (this will use a small amount of API credits)...")
        messages = [{"role": "user", "content": "Say 'test successful' and nothing else."}]
        response = provider.chat(messages, max_new_tokens=20, temperature=0.1)
        print(f"✓ API call successful")
        print(f"  Response: {response[:100]}")
    except Exception as e:
        print(f"⚠ API call failed: {e}")
        print(f"  (Provider initialized correctly, but API call failed)")
        print(f"  Check your API key, credits, and internet connection")
        return False
    
    print(f"\n✓ All tests passed for {provider_name}!")
    return True


def main():
    parser = argparse.ArgumentParser(description='Test TaxoAdapt LLM provider setup')
    parser.add_argument('--provider', type=str, help='Provider to test (openai, azure, claude, huggingface)')
    parser.add_argument('--skip-api-call', action='store_true', help='Skip actual API call test')
    args = parser.parse_args()
    
    print("TaxoAdapt Multi-Provider Setup Test")
    print("=" * 60)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please install missing packages.")
        sys.exit(1)
    
    # Test .env file
    if not test_env_file():
        print("\n❌ .env file test failed. Please create and configure .env file.")
        sys.exit(1)
    
    # Test provider
    if not test_provider(args.provider):
        print(f"\n❌ Provider test failed.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Your setup is ready to use.")
    print("=" * 60)
    print("\nYou can now run: python main.py")


if __name__ == "__main__":
    main()

