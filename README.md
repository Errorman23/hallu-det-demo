# Hallucination Detection Demo Applications

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B.svg)](https://streamlit.io)
[![LLM](https://img.shields.io/badge/LLM-Groq%20%7C%20OpenAI-green.svg)](https://console.groq.com)

This repository contains two Streamlit applications that implement hallucination detection in Large Language Models (LLMs) using consistency checking methodology. The system generates multiple responses and evaluates their consistency at the claim level to detect potential hallucinations.

## Overview

### Applications

| App | File | Description |
|-----|------|-------------|
| **Preset Data** | `app_preset_data.py` | Uses pre-defined questions from domain packs |
| **Live Generation** | `app_live_gen_data.py` | Dynamically generates domain-specific questions |

### Key Features

- **Consistency Checking**: Compares main response against multiple stochastic samples
- **Claim-Level Analysis**: Intelligently splits responses into verifiable claims
- **Multi-Model Support**: Test with various models (Llama, GPT-OSS, Kimi)
- **Visual Charts**: Per-question inconsistency score bar charts
- **Export Results**: Download detailed JSON reports of detection results
- **Color-Coded Feedback**: Risk indicators and detailed scoring

### Detection Method: Temperature Variation

Both applications use the temperature variation method:
- **Main response**: Generated deterministically (temp=0.0)
- **N Samples**: Generated stochastically (temp=1.0) 
- **Same prompt**: Used for all queries
- **Claim extraction**: Uses GPT-5-mini to split response into verifiable claims
- **Consistency check**: Each claim verified against all samples

## Prerequisites

- Python 3.9 or higher
- API keys for:
  - **Groq**: For running LLMs (Llama, GPT-OSS, Kimi models)
  - **OpenAI**: For GPT-5-mini judge model

## Installation

### Step 1: Navigate to the Directory

```bash
cd hallu-det-demo
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up API Keys

Create a `.env` file with your API keys:

```bash
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
OPENAI_API_KEY=sk-your_actual_openai_api_key_here
```

**Important**: 
- Get your Groq API key from: https://console.groq.com/keys
- Get your OpenAI API key from: https://platform.openai.com/api-keys
- Never commit the `.env` file to version control

## Running the Applications

### App 1: Preset Data

Uses pre-defined question sets for consistent testing.

```bash
streamlit run app_preset_data.py --server.port 8501
```

Access at: http://localhost:8501

**How to use:**
1. Select a domain pack (currently Singapore Insurance)
2. Choose target model for testing
3. Configure sampling parameters (main temp, sample temp, N samples)
4. Select number of questions to process
5. Run detection and analyze results
6. View per-question inconsistency chart
7. Download results as JSON

### App 2: Live Generation

Allows you to test any domain/use case by generating questions dynamically.

```bash
streamlit run app_live_gen_data.py --server.port 8502
```

Access at: http://localhost:8502

**How to use:**
1. Enter a domain/use case description (e.g., "Medical diagnosis", "Legal compliance")
2. Click "Use Case Check" to validate suitability
3. Generate domain-specific questions
4. Configure model and sampling parameters
5. Run hallucination detection
6. View per-question inconsistency chart
7. Download results as JSON

### Running Both Apps Simultaneously

```bash
# Terminal 1
source venv/bin/activate && streamlit run app_preset_data.py --server.port 8501

# Terminal 2
source venv/bin/activate && streamlit run app_live_gen_data.py --server.port 8502
```

## Configuration Options

### Target Models (via Groq)
- `llama-3.1-8b-instant`: Fast, lightweight model
- `llama-3.3-70b-versatile`: More capable, larger model
- `moonshotai/kimi-k2-instruct`: Alternative model option
- `openai/gpt-oss-20b`: OpenAI's open-source 20B model
- `openai/gpt-oss-120b`: OpenAI's open-source 120B model

### Judge Model
- `gpt-5-mini`: OpenAI GPT-5-mini (used for claim extraction and verification)

### Sampling Parameters
- **Main Response Temperature** (0.0-1.0): Controls determinism of main answer (default: 0.0)
- **Sample Temperature** (0.0-1.5): Controls variation in consistency samples (default: 1.0)
- **Number of Samples** (3-40): How many stochastic samples to generate (default: 20)
- **Max Tokens** (64-1024): Maximum tokens per response (default: 512)

### Scoring Interpretation
- **0.0-0.3**: ✅ Low hallucination risk (green)
- **0.3-0.6**: ⚠️ Medium hallucination risk (yellow)
- **0.6-1.0**: ❌ High hallucination risk (red)

## Troubleshooting

### Common Issues

1. **Port Already in Use**
```bash
# Find and kill process on port
lsof -i :8501  # Find PID
kill -9 <PID>  # Kill process
```

2. **API Key Errors**
- Verify your `.env` file exists and contains valid keys
- Check API key quotas and rate limits
- Ensure keys have proper permissions

3. **Module Not Found Errors**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

4. **Rate Limiting Issues**
- The apps include retry logic with exponential backoff
- For OpenAI OSS models, requests are automatically staggered
- Consider reducing the number of samples or questions

5. **Python 3.9 Compatibility**
- Ensure you're using Python 3.9+
- The apps have been tested with Python 3.9

## Output Format

Results are exported as JSON with the following structure:

```json
{
  "metadata": {
    "version": "1.0.0",
    "export_timestamp": 1234567890,
    "test_type": "hallucination_detection",
    "method": "temperature_variation"
  },
  "configuration": {
    "target_model": { "model": "...", "provider": "...", "temperature": 0.0 },
    "judge_model": { "model": "gpt-5-mini", "temperature": 0.0 }
  },
  "summary": {
    "overall_average_inconsistency_score": 0.25,
    "risk_level": "low|medium|high",
    "total_claims_analyzed": 50
  },
  "results": [
    {
      "question": "Question text",
      "base_answer": "Main response",
      "samples": ["Sample 1", "Sample 2"],
      "overall_score": 0.3,
      "claims": [
        {
          "claim": "Individual claim",
          "inconsistency_score": 0.2,
          "support_rate": 0.8
        }
      ]
    }
  ]
}
```

## Technical Details

### Methodology

1. **Main Response Generation**: Creates deterministic response (temp=0.0)
2. **Sample Generation**: Creates N stochastic responses (temp=1.0)
3. **Claim Extraction**: Uses GPT-5-mini to split response into verifiable claims
4. **Consistency Checking**: Each claim is verified against all samples by the judge model
5. **Scoring**: Aggregates claim-level scores for overall assessment

### Architecture
- **Frontend**: Streamlit web interface
- **LLM Integration**: LiteLLM for unified API access
- **Async Processing**: Parallel API calls for efficiency
- **Session State**: Preserves results across UI interactions
- **Visualization**: Plotly for interactive charts

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for the full license text.

```
Copyright 2026 hallu-det-demo Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

This demo is provided for educational and research purposes. Please ensure compliance with API terms of service when using third-party models.
