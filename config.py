import os
from pathlib import Path

def load_api_keys():
    """Load API keys from .env file"""
    env_file = Path(__file__).parent / '.env'
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
    
    # Set the keys if not already in environment
    if "GROQ_API_KEY" not in os.environ:
        os.environ["GROQ_API_KEY"] = ""
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = ""