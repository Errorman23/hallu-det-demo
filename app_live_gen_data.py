import os
import time
import json
import hashlib
import random
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import base64
from io import BytesIO

import litellm
import numpy as np
import streamlit as st
import plotly.express as px
import pandas as pd

# Load API keys from config
from config import load_api_keys
load_api_keys()

litellm.set_verbose = False  # Set to True for debugging

# -------------------------------
# Data structures
# -------------------------------
@dataclass
class LLMConfig:
    provider: str      # "groq" (all models are on groq)
    model: str
    temperature: float = 0.0  # Default for main response (deterministic)
    top_p: float = 0.9
    max_tokens: int = 512  # Shared max tokens
    reasoning_effort: Optional[str] = None  # For GPT-5 models

# -------------------------------
# Page header
# -------------------------------
def get_page_header():
    """Return the page header HTML"""
    header_html = """
    <div style="margin-bottom: 20px;">
        <h1 style="margin: 0;">Dynamic Hallucination Detection with Live Question Generation</h1>
        <p style="margin: 5px 0 0 0; color: gray; font-size: 14px;">Generate domain-specific questions and detect inconsistencies in LLM responses</p>
    </div>
    """
    return header_html

# -------------------------------
# Helpers
# -------------------------------
def normalize_text(text: str) -> str:
    if not text:
        return ""
    # Follow original implementation - replace newlines with spaces
    return text.replace("\r\n", " ").replace("\n", " ").replace("\t", " ").replace("  ", " ").strip()

def parse_model_selection(model_string: str) -> Tuple[str, str]:
    """Parse model string like 'groq/llama-3.1-8b-instant' into provider and model"""
    if "/" in model_string:
        parts = model_string.split("/", 1)
        return parts[0], parts[1]
    # Default to groq if no provider specified
    return "groq", model_string

# -------------------------------
# Async LLM functions with liteLLM
# -------------------------------
def get_litellm_model(cfg: LLMConfig) -> str:
    """Convert our config to litellm model string"""
    if cfg.provider == "groq":
        return f"groq/{cfg.model}"
    elif cfg.provider == "openai":
        return cfg.model
    elif cfg.provider == "moonshotai":
        return f"moonshotai/{cfg.model}"
    else:
        return f"{cfg.provider}/{cfg.model}"

async def async_chat_complete(cfg: LLMConfig, messages: List[Dict], temperature: Optional[float] = None, seed: Optional[int] = None, retry_count: int = 0) -> str:
    """Async chat completion using liteLLM with retry logic"""
    try:
        model = get_litellm_model(cfg)
        
        # Build kwargs based on model requirements
        kwargs = {
            "model": model,
            "messages": messages,
            "timeout": 90,  # Increased timeout
        }
        
        # Handle different model requirements
        if cfg.provider == "openai" and cfg.model == "gpt-5-mini":
            # GPT-5-mini specific settings - always use high token limit
            kwargs["max_completion_tokens"] = 4000  # Doubled from 2000 for better context handling
            # GPT-5-mini doesn't support custom temperature setting
            # but we can still pass it (will use default)
        elif cfg.provider == "openai" and cfg.model == "gpt-5":
            # GPT-5 with reasoning effort - doesn't use temperature
            kwargs["max_completion_tokens"] = cfg.max_tokens if cfg.max_tokens else 2000
            if cfg.reasoning_effort:
                kwargs["reasoning_effort"] = cfg.reasoning_effort
            # GPT-5 doesn't support temperature parameter
        elif cfg.provider == "groq" and cfg.model in ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
            # OpenAI OSS models on Groq - standard configuration
            kwargs["temperature"] = temperature if temperature is not None else cfg.temperature
            kwargs["top_p"] = cfg.top_p
            kwargs["max_tokens"] = cfg.max_tokens
            if seed is not None:
                kwargs["seed"] = seed
        else:
            # Standard models (including moonshotai)
            kwargs["temperature"] = temperature if temperature is not None else cfg.temperature
            kwargs["top_p"] = cfg.top_p
            kwargs["max_tokens"] = cfg.max_tokens
            if seed is not None:
                kwargs["seed"] = seed
        
        response = await litellm.acompletion(**kwargs)
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Received empty response from model")
        return content
    except Exception as e:
        error_msg = str(e)
        
        # Check for rate limiting errors
        if "rate" in error_msg.lower() or "429" in error_msg or retry_count < 2:
            if retry_count < 2:
                # Wait before retrying (exponential backoff)
                wait_time = (retry_count + 1) * 2
                await asyncio.sleep(wait_time)
                return await async_chat_complete(cfg, messages, temperature, seed, retry_count + 1)
        
        # Log error to console for debugging
        print(f"LLM Error (attempt {retry_count + 1}): {error_msg}")
        print(f"Model: {model}, Provider: {cfg.provider}")
        # Don't raise exception for better error handling - return empty string
        # The calling function should handle empty responses appropriately
        return ""

async def intelligent_clause_split(text: str) -> List[str]:
    """Use GPT-5-mini to intelligently split text into verifiable claims with improved structured prompt"""
    # Structured claim splitting with clear formatting like judge prompt
    system_prompt = """You are a precise claim extraction assistant. Your task is to break down text into atomic, verifiable claims.

You must:
- Extract each factual statement as a separate claim
- Ensure each claim can be independently verified
- Preserve the original meaning
- Return ONLY a JSON array of strings"""
    
    user_prompt = f"""=== TEXT TO ANALYZE ===
{text}

=== YOUR TASK ===
Extract individual claims from the text above. Each claim must be:
- A complete, standalone statement
- Independently fact-checkable
- Specific and unambiguous

=== OUTPUT FORMAT ===
Return ONLY a JSON array of strings, where each string is one claim.

=== JSON ARRAY ===
"""
    
    try:
        cfg = LLMConfig(provider="openai", model="gpt-5-mini", max_tokens=2000)
        # Use system message for better structured extraction
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = await async_chat_complete(cfg, messages, temperature=0.0)
        
        # Extract JSON from response
        if "[" in response and "]" in response:
            json_str = response[response.find("["):response.rfind("]")+1]
            claims = json.loads(json_str)
            # Filter out empty or very short claims
            return [c.strip() for c in claims if isinstance(c, str) and len(c.strip()) > 10][:20]  # Max 20 claims
        else:
            # Fallback to simple splitting
            return simple_sentence_split(text)
    except Exception as e:
        st.warning(f"Intelligent splitting failed, using simple method: {str(e)}")
        return simple_sentence_split(text)

def simple_sentence_split(text: str) -> List[str]:
    """Simple fallback sentence splitter"""
    import re
    text = normalize_text(text)
    # Split by sentence endings
    sentences = re.split(r'[.!?]+\s+', text)
    # Filter and clean
    return [s.strip() for s in sentences if len(s.strip()) > 10][:20]

async def async_generate_main_response(cfg: LLMConfig, user_prompt: str, main_temp: float) -> str:
    """Generate the main response deterministically (as per paper setting)"""
    messages = [{"role": "user", "content": user_prompt}]
    
    # Use specified temperature for main response (typically 0.0 for deterministic)
    response = await async_chat_complete(cfg, messages, temperature=main_temp)
    return response

async def async_sample_model_answers(cfg: LLMConfig, user_prompt: str, n: int, sample_temp: float) -> List[str]:
    """Sample multiple answers from model asynchronously with stochastic sampling"""
    messages = [{"role": "user", "content": user_prompt}]
    
    # Generate n samples with specified temperature (typically 1.0 for stochastic)
    tasks = []
    for i in range(n):
        # Use random seed for variation in stochastic sampling
        seed = random.randint(1, 9_999_999)
        tasks.append(async_chat_complete(cfg, messages, temperature=sample_temp, seed=seed))
    
    # Run all sampling tasks in parallel
    outputs = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Collect all outputs
    all_outputs = []
    for i, out in enumerate(outputs):
        if isinstance(out, str) and out:
            all_outputs.append(out)
        else:
            # If sampling failed, try once more
            try:
                retry = await async_chat_complete(cfg, messages, temperature=sample_temp, seed=random.randint(1, 9_999_999))
                if retry:
                    all_outputs.append(retry)
            except:
                pass
    
    # Ensure we return exactly n samples (pad with retries if needed)
    while len(all_outputs) < n:
        try:
            retry = await async_chat_complete(cfg, messages, temperature=sample_temp, seed=random.randint(1, 9_999_999))
            if retry:
                all_outputs.append(retry)
            else:
                all_outputs.append(all_outputs[0] if all_outputs else "Failed to generate sample")
        except:
            all_outputs.append(all_outputs[0] if all_outputs else "Failed to generate sample")
    
    return all_outputs[:n]  # Return exactly n samples

async def async_judge_single(sentence: str, sample: str) -> float:
    """
    Judge if a sentence is supported by a sample passage.
    Using improved structured prompt.
    Returns: 0.0 if supported/consistent (Yes), 1.0 if not supported/inconsistent (No), 0.5 if unclear
    """
    
    # System prompt for better clarity
    system_prompt = """You are a precise fact-checking assistant. Your task is to determine if a given sentence is supported by a context passage.
You must respond with ONLY 'Yes' or 'No'.
- Answer 'Yes' if the sentence is clearly supported by the context
- Answer 'No' if the sentence contradicts the context or makes claims not found in the context
- Only answer 'Yes' or 'No', nothing else"""
    
    # User prompt with clear structure
    sample_cleaned = sample.replace('\n', ' ').strip()
    user_prompt = f"""=== CONTEXT PASSAGE ===
{sample_cleaned}

=== SENTENCE TO VERIFY ===
{sentence}

=== QUESTION ===
Is the sentence above supported by the context passage? Answer Yes or No.

=== YOUR ANSWER ===
"""
    
    try:
        # Use GPT-5-mini as judge with deterministic setting
        cfg = LLMConfig(provider="openai", model="gpt-5-mini", max_tokens=100)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Temperature 0 for deterministic as in original
        response = await async_chat_complete(cfg, messages, temperature=0.0)
        
        # Process response as in original
        response_lower = response.lower().strip()
        
        if response_lower.startswith('yes'):
            return 0.0  # Supported = no hallucination
        elif response_lower.startswith('no'):
            return 1.0  # Not supported = hallucination
        else:
            return 0.5  # Unclear
            
    except Exception as e:
        st.warning(f"Judge error: {str(e)}")
        return 0.5  # Default to unclear on error

async def async_evaluate_sentence(sentence: str, samples: List[str], progress_text=None) -> Tuple[float, List[float]]:
    """
    Evaluate a sentence against all samples in parallel.
    Returns: (mean_score, individual_scores)
    """
    if progress_text:
        progress_text.text(f"Evaluating: {sentence[:50]}...")
    
    # Create tasks for parallel evaluation
    tasks = [async_judge_single(sentence, sample) for sample in samples]
    scores = await asyncio.gather(*tasks)
    
    # Calculate mean score (hallucination risk)
    mean_score = np.mean(scores)
    
    return mean_score, scores

async def process_question_with_delay(q: str, cfg: LLMConfig, n_samples: int, main_temp: float, sample_temp: float, progress_container, delay: float):
    """Process a question with initial delay to avoid rate limiting"""
    await asyncio.sleep(delay)
    return await process_question_async(q, cfg, n_samples, main_temp, sample_temp, progress_container)

async def process_question_async(q: str, cfg: LLMConfig, n_samples: int, main_temp: float, sample_temp: float, progress_container):
    """Process a single question with consistency checking methodology"""
    with progress_container.container():
        # Don't show the question here since it's already in the expander title
        status = st.empty()
        
        # Step 1: Generate main response deterministically
        status.text(f"Generating main response (temp={main_temp:.1f})...")
        base_answer = await async_generate_main_response(cfg, q, main_temp)
        
        if not base_answer:
            st.error("Failed to generate main response")
            return None
        
        # Step 2: Generate N stochastic samples  
        status.text(f"Generating {n_samples} stochastic samples (temp={sample_temp:.1f})...")
        samples = await async_sample_model_answers(cfg, q, n_samples, sample_temp)
        
        if len(samples) < 1:
            st.error("Failed to generate sufficient samples")
            return None
        
        status.text(f"Using {len(samples)} samples for consistency check")
        
        # Step 3: Split base answer into claims using intelligent splitting
        status.text("Extracting claims from response...")
        claims = await intelligent_clause_split(base_answer)
        
        if not claims:
            claims = [base_answer]  # Fallback to whole answer
        
        status.text(f"Found {len(claims)} claims to evaluate")
        
        # Step 4: Evaluate each claim against all samples
        claim_results = []
        for i, claim in enumerate(claims):
            status.text(f"Evaluating claim {i+1}/{len(claims)}...")
            mean_score, individual_scores = await async_evaluate_sentence(claim, samples, status)
            
            claim_results.append({
                "claim": claim,
                "inconsistency_score": mean_score,
                "individual_scores": individual_scores,
                "support_rate": 1.0 - mean_score  # Convert to support rate for display
            })
        
        # Calculate overall score
        overall_score = np.mean([c["inconsistency_score"] for c in claim_results])
        
        status.success(f"✓ Completed: Overall inconsistency score = {overall_score:.2f}")
        
        return {
            "question": q,
            "base_answer": base_answer,
            "samples": samples,
            "claims": claim_results,
            "overall_score": overall_score,
            "main_temp": main_temp,
            "sample_temp": sample_temp
        }

async def process_all_questions_async(questions: List[str], cfg: LLMConfig, n_samples: int, main_temp: float, sample_temp: float):
    """Process all questions in parallel with staggered start for rate limiting"""
    st.write("### Processing Questions in Parallel")
    
    # Create containers for each question
    containers = []
    for i, q in enumerate(questions, 1):
        with st.expander(f"Q{i}: {q}", expanded=True):  # Show full question
            containers.append(st.empty())
    
    # Create tasks for all questions with staggered start for OpenAI OSS models
    tasks = []
    for idx, (q, container) in enumerate(zip(questions, containers)):
        # Add delay for OpenAI OSS models to avoid rate limiting
        if cfg.model in ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
            # Stagger the start by 1 second per question for OSS models
            task = process_question_with_delay(q, cfg, n_samples, main_temp, sample_temp, container, idx * 1.0)
        else:
            # No delay for other models
            task = process_question_async(q, cfg, n_samples, main_temp, sample_temp, container)
        tasks.append(task)
    
    # Process all questions in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter successful results
    valid_results = []
    for r in results:
        if isinstance(r, dict):
            valid_results.append(r)
        elif isinstance(r, Exception):
            st.error(f"Question processing error: {str(r)}")
    
    return valid_results

# -------------------------------
# Domain checking and question generation
# -------------------------------
async def check_domain_suitability(domain_description: str) -> Tuple[bool, str]:
    """Check if a domain/task is suitable for hallucination detection"""
    
    system_prompt = """You are an expert at evaluating whether a domain, task, or use case is suitable for hallucination detection through consistency checking.

=== CRITERIA FOR SUITABLE DOMAINS ===
Domains SUITABLE for hallucination detection must have:
• Factual, objective information
• Verifiable answers from authoritative sources
• Clear right/wrong distinctions
• Testable claims and statements

Examples of SUITABLE domains:
• Science and technology
• History and historical events
• Medicine and healthcare
• Law and regulations
• Technical documentation
• Product specifications
• Insurance policies and compliance
• Geographic facts

=== CRITERIA FOR UNSUITABLE DOMAINS ===
Domains NOT SUITABLE for hallucination detection have:
• Creative or subjective content
• Opinion-based answers
• No clear right/wrong answers
• Personal preferences or tastes

Examples of UNSUITABLE domains:
• Creative writing and storytelling
• Poetry and artistic expression
• Personal opinions and beliefs
• Philosophical discussions
• Artistic interpretation
• Subjective experiences

=== YOUR TASK ===
Analyze the provided domain and determine if it's suitable for consistency-based hallucination detection.

=== RESPONSE FORMAT ===
You must respond with ONLY this JSON structure:
{
    "suitable": true or false,
    "reason": "A clear, concise explanation of why this domain is or isn't suitable for hallucination detection"
}"""

    user_prompt = f"""=== DOMAIN TO EVALUATE ===
{domain_description}

=== INSTRUCTIONS ===
1. Analyze if this domain has factual, verifiable information
2. Determine if questions in this domain would have objective answers
3. Respond with ONLY the JSON object specified above
4. Do not include any text before or after the JSON"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        # Use GPT-5-mini for evaluation - Must use temperature=1.0!
        cfg = LLMConfig(provider="openai", model="gpt-5-mini", temperature=1.0, max_tokens=500)
        response = await async_chat_complete(cfg, messages)
        
        if not response:
            return False, "Failed to get response from GPT-5-mini. Please check API key and connection."
        
        # Parse JSON response
        try:
            result = json.loads(response)
            return result.get("suitable", False), result.get("reason", "No reason provided")
        except json.JSONDecodeError:
            # Try to extract JSON from response if it contains extra text
            import re
            json_match = re.search(r'\{.*?"suitable".*?\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result.get("suitable", False), result.get("reason", "No reason provided")
            print(f"Could not parse JSON from response: {response}")
            return False, f"Could not parse response: {response[:200]}..."
    except Exception as e:
        print(f"Domain check error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, f"Error evaluating domain: {str(e)}"

async def generate_domain_questions(domain_description: str, num_questions: int = 10) -> List[str]:
    """Generate factual questions for a given domain using GPT-5 with reasoning"""
    
    # More structured prompt for GPT-5
    user_prompt = f"""You are tasked with generating exactly {num_questions} factual, verifiable questions about {domain_description}.

Requirements:
1. Each question must test specific factual knowledge about {domain_description}
2. Questions should have clear, objective answers that can be verified
3. Cover different aspects and complexity levels of the domain
4. Questions should be suitable for testing if an AI model might hallucinate

Return your response as a JSON array containing exactly {num_questions} question strings.

Example format:
["What is the regulatory body for insurance in Singapore?", "When was the Insurance Act first enacted in Singapore?", "What are the minimum capital requirements for general insurers in Singapore?", "Which types of insurance are mandatory in Singapore?", "What is the role of MAS in insurance supervision?"]

Generate the {num_questions} questions for {domain_description} now:"""

    messages = [
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        # Use GPT-5-mini for question generation with high token limit
        cfg = LLMConfig(provider="openai", model="gpt-5-mini", max_tokens=8000)
        
        # Log the request for debugging
        print(f"Generating questions for domain: {domain_description}")
        
        response = await async_chat_complete(cfg, messages)
        
        if not response:
            print("ERROR: Received empty response from GPT-5-mini for question generation")
            return []
        
        # Log the response for debugging
        print(f"Raw response received (length={len(response)}): {response[:500]}...")
        
        # Try to parse the response as JSON directly first
        try:
            questions = json.loads(response.strip())
            if isinstance(questions, list):
                # Validate and clean questions
                valid_questions = []
                for q in questions:
                    if isinstance(q, str) and len(q.strip()) > 10:
                        q = q.strip()
                        if not q.endswith("?"):
                            q += "?"
                        valid_questions.append(q)
                
                print(f"Successfully generated {len(valid_questions)} valid questions")
                return valid_questions[:num_questions]
        except json.JSONDecodeError:
            # Try to extract JSON array from response
            if "[" in response and "]" in response:
                json_str = response[response.find("["):response.rfind("]")+1]
                try:
                    questions = json.loads(json_str)
                    valid_questions = []
                    for q in questions:
                        if isinstance(q, str) and len(q.strip()) > 10:
                            q = q.strip()
                            if not q.endswith("?"):
                                q += "?"
                            valid_questions.append(q)
                    print(f"Extracted {len(valid_questions)} valid questions from response")
                    return valid_questions[:num_questions]
                except json.JSONDecodeError as e:
                    print(f"Failed to parse extracted JSON: {str(e)}")
                    print(f"Extracted string: {json_str[:500]}...")
                    return []
            else:
                print(f"Response doesn't contain JSON array: {response[:500]}...")
                return []
                
    except Exception as e:
        print(f"Error in generate_domain_questions: {str(e)}")
        import traceback
        print(traceback.format_exc())
        # Don't use st.error here as it might not display in spinner context
        return []

# -------------------------------
# Utilities (Domain packs removed - using dynamic generation)
# -------------------------------

# -------------------------------
# UI
# -------------------------------
def main():
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .sample-header {
            font-weight: bold;
            color: #0080FF;
            margin-top: 10px;
            margin-bottom: 5px;
            font-family: monospace;
        }
        .sample-content {
            background-color: #e8e8e8;
            color: #1a1a1a;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 15px;
            font-family: sans-serif;
            border: 1px solid #d0d0d0;
            line-height: 1.5;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Display page header
    st.markdown(get_page_header(), unsafe_allow_html=True)
    
    # Initialize session state
    if 'domain_validated' not in st.session_state:
        st.session_state.domain_validated = False
    if 'domain_suitable' not in st.session_state:
        st.session_state.domain_suitable = False
    if 'domain_reason' not in st.session_state:
        st.session_state.domain_reason = ""
    if 'generated_questions' not in st.session_state:
        st.session_state.generated_questions = []
    if 'domain_description' not in st.session_state:
        st.session_state.domain_description = ""
    if 'detection_results' not in st.session_state:
        st.session_state.detection_results = None
    if 'detection_elapsed' not in st.session_state:
        st.session_state.detection_elapsed = None
    
    # Domain Input Section
    st.header("Step 1: Define Your Domain/Task/Use Case")
    st.markdown("""Enter a description of the domain, task, or use case you want to test for hallucinations. 
    The system will check if it's suitable for consistency-based detection.""")
    
    domain_input = st.text_area(
        "Domain/Task Description",
        value=st.session_state.domain_description,
        placeholder="Example: Medical diagnosis information, Legal compliance questions, Technical documentation for cloud services, etc.",
        height=100
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        check_btn = st.button("🔍 Use Case Check", type="primary")
    with col2:
        if st.session_state.domain_suitable or st.session_state.domain_validated:
            if st.button("🔄 Reset Domain"):
                st.session_state.domain_validated = False
                st.session_state.domain_suitable = False
                st.session_state.domain_reason = ""
                st.session_state.generated_questions = []
                st.session_state.domain_description = ""
                st.rerun()
    
    # Domain validation
    if check_btn and domain_input:
        with st.spinner("Evaluating domain suitability..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            is_suitable, reason = loop.run_until_complete(check_domain_suitability(domain_input))
            
            if is_suitable:
                st.success(f"✅ Domain is suitable for hallucination detection: {reason}")
                # Save domain validation results
                st.session_state.domain_suitable = True
                st.session_state.domain_reason = reason
                st.session_state.domain_description = domain_input
                st.rerun()
            else:
                st.error(f"❌ Domain not suitable: {reason}")
                st.info("Please enter a domain with factual, objective information that can be verified.")
                st.session_state.domain_suitable = False
    elif check_btn:
        st.warning("Please enter a domain description first.")
    
    # Show question generation section if domain is suitable
    if st.session_state.domain_suitable and not st.session_state.domain_validated:
        st.markdown("---")
        st.header("Step 2: Generate Test Questions")
        st.success(f"✅ Domain validated: {st.session_state.domain_description}")
        st.info(f"Reason: {st.session_state.domain_reason}")
        
        if st.button("🎯 Generate Questions", type="primary", key="gen_questions_btn"):
            with st.spinner("Generating domain-specific questions..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    questions = loop.run_until_complete(generate_domain_questions(st.session_state.domain_description, 10))
                    
                    if questions and len(questions) > 0:
                        st.session_state.domain_validated = True
                        st.session_state.generated_questions = questions
                        st.success(f"Generated {len(questions)} questions for testing")
                        st.rerun()
                    else:
                        st.error("Failed to generate questions. The API returned an empty or invalid response.")
                        st.info("Tips: Try clicking 'Generate Questions' again. The model may need another attempt.")
                        st.caption("Check the console output for debugging information.")
                except Exception as e:
                    st.error(f"Error generating questions: {str(e)}")
                    st.info("This may be a temporary API issue. Please try again.")
                    # Show more details about the error
                    with st.expander("Error Details"):
                        st.code(str(e))
        return
    
    # Only show the rest of the UI if questions are generated
    elif not st.session_state.domain_validated:
        st.markdown("---")
        st.info("💡 **Examples of suitable domains:**\n"
                "- Medical information and diagnosis\n"
                "- Legal regulations and compliance\n"
                "- Historical facts and events\n"
                "- Scientific concepts and theories\n"
                "- Technical specifications and documentation\n\n"
                "**Not suitable:**\n"
                "- Creative writing\n"
                "- Personal opinions\n"
                "- Artistic interpretation\n"
                "- Philosophical discussions")
        return
    
    # Show generated questions
    st.markdown("---")
    st.header("Step 2: Review Generated Questions")
    st.success(f"✅ Domain validated: {st.session_state.domain_description}")
    
    with st.expander("View Generated Questions", expanded=True):
        for i, q in enumerate(st.session_state.generated_questions, 1):
            st.markdown(f"**Q{i}.** {q}")
    
    # Sidebar configuration (only show after domain validation)
    with st.sidebar:
        st.header("Configuration")
        
        st.markdown("---")
        st.subheader("1) Active Domain")
        st.info(st.session_state.domain_description[:100] + "..." if len(st.session_state.domain_description) > 100 else st.session_state.domain_description)
        
        st.markdown("---")
        st.subheader("2) Target Model")
        
        # Multi-model support with dropdown
        model_options = [
            "groq/llama-3.1-8b-instant",
            "groq/llama-3.3-70b-versatile",
            "groq/moonshotai/kimi-k2-instruct",
            "groq/openai/gpt-oss-20b",
            "groq/openai/gpt-oss-120b"
        ]
        
        selected_model = st.selectbox("Model", model_options, index=0)
        provider, model = parse_model_selection(selected_model)
        
        # Show model details
        st.caption(f"Provider: {provider}, Model: {model}")
        
        st.markdown("---")
        st.subheader("3) Judge Model")
        st.text_input("Judge Model", value="gpt-5-mini", disabled=True)
        st.caption("Using OpenAI GPT-5-mini for claim verification")
        
        st.markdown("---")
        st.subheader("4) Sampling Configuration")
        
        # Two different temperatures for response generation
        main_temp = st.slider("Main Response Temperature", 0.0, 1.0, 0.0, 0.1,
                             help="Temperature for generating the main response (0.0 = deterministic)")
        
        sample_temp = st.slider("Sample Temperature", 0.0, 1.5, 1.0, 0.1,
                               help="Temperature for generating N stochastic samples (1.0 = high variation)")
        
        n_samples = st.slider("Number of Samples (N)", 3, 40, 20, 1, 
                             help="How many stochastic samples to generate for consistency checking")
        
        # Shared max tokens
        max_tokens = st.slider("Max Tokens (shared)", 64, 1024, 512, 64)

    # Main area
    st.header("Step 3: Run Hallucination Detection")
    
    cols = st.columns(3)
    with cols[0]:
        q_count = st.slider("Questions to process", 1, len(st.session_state.generated_questions), len(st.session_state.generated_questions), 1)
    with cols[2]:
        run_btn = st.button("🚀 Run Detection", type="primary")

    selected_qs = st.session_state.generated_questions[:q_count]

    if run_btn:
        # Validate questions (all should be valid since we generated them)
        valid_qs = selected_qs
        
        if valid_qs:
            # Create config with parsed provider and model
            cfg = LLMConfig(provider=provider, model=model, temperature=main_temp, max_tokens=max_tokens)
            
            # Run async processing
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            start_time = time.time()
            results = loop.run_until_complete(
                process_all_questions_async(valid_qs, cfg, n_samples, main_temp, sample_temp)
            )
            elapsed = time.time() - start_time
            
            # Store results in session state
            st.session_state.detection_results = results
            st.session_state.detection_elapsed = elapsed
            st.session_state.selected_model = selected_model
            st.session_state.provider = provider
            st.session_state.main_temp = main_temp
            st.session_state.sample_temp = sample_temp
            st.session_state.n_samples = n_samples
            st.session_state.selected_judge_model = 'gpt-5-mini'
            st.session_state.judge_temp = 1.0
            st.session_state.judge_max_tokens = 8192
            
            st.success(f"✓ Completed {len(results)} questions in {elapsed:.1f} seconds")
    
    # Display results from session state if available
    if st.session_state.detection_results:
        results = st.session_state.detection_results
        elapsed = st.session_state.detection_elapsed
        
        # Use stored values from session state
        if 'selected_model' in st.session_state:
            selected_model = st.session_state.selected_model
            provider = st.session_state.provider
            main_temp = st.session_state.main_temp
            sample_temp = st.session_state.sample_temp
            n_samples = st.session_state.n_samples
        
        overall_avg = np.mean([r["overall_score"] for r in results])
        
        # Calculate overall average inconsistency score
        all_claim_scores = []
        for r in results:
            for c in r['claims']:
                all_claim_scores.append(c['inconsistency_score'])
        
        if all_claim_scores:
            overall_avg_score = np.mean(all_claim_scores)
        else:
            overall_avg_score = overall_avg
        
        # Create bar chart for per-question scores
        st.markdown("---")
        st.subheader("📊 Per-Question Inconsistency Scores")
        
        # Prepare data for bar chart
        question_labels = []
        question_scores = []
        for i, r in enumerate(results, 1):
            question_labels.append(f"Q{i}")
            question_scores.append(r['overall_score'])
        
        # Create DataFrame for plotting
        chart_data = pd.DataFrame({
            'Question': question_labels,
            'Inconsistency Score': question_scores
        })
        
        # Create bar chart using plotly
        fig = px.bar(
            chart_data, 
            x='Question', 
            y='Inconsistency Score',
            title='Inconsistency Score by Question',
            labels={'Inconsistency Score': 'Score (0=consistent, 1=inconsistent)'},
            color='Inconsistency Score',
            color_continuous_scale=['green', 'yellow', 'red'],
            range_color=[0, 1]
        )
        
        # Add horizontal lines for thresholds
        fig.add_hline(y=0.3, line_dash="dot", line_color="orange")
        fig.add_hline(y=0.6, line_dash="dot", line_color="red")
        
        # Update layout with fixed y-axis range
        fig.update_layout(
            yaxis_range=[0, 1],  # Fixed range from 0 to 1
            height=400,
            showlegend=False,
            yaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=0.1  # Show ticks at 0.1 intervals
            )
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Display overall score prominently
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.metric(
                "🎯 Overall Inconsistency Score",
                f"{overall_avg_score:.3f}",
                help="Average of all claim-level scores across all questions (0=consistent, 1=highly inconsistent)"
            )
            # Color indicator
            if overall_avg_score < 0.3:
                st.success("✅ Low hallucination risk")
            elif overall_avg_score < 0.6:
                st.warning("⚠️ Moderate hallucination risk")
            else:
                st.error("❌ High hallucination risk")
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Results")
        
        for r in results:
            score_emoji = "🔴" if r['overall_score'] > 0.6 else "🟡" if r['overall_score'] > 0.3 else "🟢"
            
            with st.expander(f"{score_emoji} {r['question']} (Score: {r['overall_score']:.2f})", expanded=False):
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write("**Response:**")
                    st.write(r['base_answer'])
                    st.caption(f"Generated with temperature={r['main_temp']:.1f}")
                with col2:
                    st.metric("Inconsistency Score", f"{r['overall_score']:.2f}")
                    st.caption("0 = Consistent\n1 = Inconsistent")
                
                st.write("**Claim-level Analysis:**")
                
                # Create a dataframe for better display
                claims_data = []
                for c in r['claims']:
                    emoji = "🔴" if c['inconsistency_score'] > 0.6 else "🟡" if c['inconsistency_score'] > 0.3 else "🟢"
                    claims_data.append({
                        "Status": emoji,
                        "Claim": c['claim'],
                        "Score": f"{c['inconsistency_score']:.2f}",
                        "Support": f"{c['support_rate']:.1%}"
                    })
                
                if claims_data:
                    df = st.dataframe(claims_data, use_container_width=True, hide_index=True)
                
                # Show sample responses with fixed styling
                with st.expander("View sample responses", expanded=False):
                    st.caption(f"Samples generated with temperature={r['sample_temp']:.1f}")
                    for i, sample in enumerate(r['samples'], 1):
                        st.markdown(f'<div class="sample-header">Sample {i}:</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="sample-content">{sample}</div>', unsafe_allow_html=True)
        
        # Export results
        st.markdown("---")
        
        # Create production-grade export data structure
        export_data = {
            "metadata": {
                "version": "1.0.0",
                "export_timestamp": time.time(),
                "export_datetime": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "test_type": "hallucination_detection",
                "method": "temperature_variation",
                "domain": st.session_state.domain_description,
                "elapsed_seconds": elapsed,
                "total_questions": len(results)
            },
            "configuration": {
                "target_model": {
                    "model": selected_model,
                    "provider": provider,
                    "main_temperature": main_temp,
                    "sample_temperature": sample_temp,
                    "n_samples": n_samples
                },
                "judge_model": {
                    "model": st.session_state.get('selected_judge_model', 'gpt-5-mini'),
                    "temperature": st.session_state.get('judge_temp', 1.0),
                    "max_tokens": st.session_state.get('judge_max_tokens', 8192)
                },
                "question_generation": {
                    "domain": st.session_state.domain_description,
                    "num_questions": st.session_state.get('num_questions', 5),
                    "dynamic": True
                }
            },
            "summary": {
                "overall_average_inconsistency_score": overall_avg_score if all_claim_scores else None,
                "risk_level": "low" if overall_avg_score < 0.3 else "medium" if overall_avg_score < 0.6 else "high" if overall_avg_score else None,
                "total_claims_analyzed": sum(len(r.get("claims", [])) for r in results),
                "questions_processed": len(results)
            },
            "results": []
        }
        
        for r in results:
            result_export = {
                "question_id": results.index(r) + 1,
                "original_question": r["question"],
                "base_response": {
                    "answer": r["base_answer"],
                    "temperature": r["main_temp"]
                },
                "sample_responses": {
                    "responses": r["samples"],
                    "temperature": r["sample_temp"]
                },
                "analysis": {
                    "overall_inconsistency_score": r["overall_score"],
                    "risk_level": "low" if r["overall_score"] < 0.3 else "medium" if r["overall_score"] < 0.6 else "high",
                    "total_claims": len(r.get("claims", [])),
                    "claims_detail": []
                }
            }
            
            for c in r["claims"]:
                claim_export = {
                    "claim_text": c["claim"],
                    "inconsistency_score": c["inconsistency_score"],
                    "support_rate": c["support_rate"],
                    "individual_support_scores": c["individual_scores"],
                    "risk_level": "low" if c["inconsistency_score"] < 0.3 else "medium" if c["inconsistency_score"] < 0.6 else "high"
                }
                result_export["analysis"]["claims_detail"].append(claim_export)
            
            export_data["results"].append(result_export)
        
        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
        st.download_button(
            "📥 Download Results (JSON)",
            json_str,
            file_name=f"hallucination_detection_dynamic_{selected_model.replace('/', '_')}_{int(time.time())}.json",
            mime="application/json"
        )
    
    # Information section
    st.markdown("---")
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        **Dynamic Hallucination Detection Process:**
        
        1. **Domain Validation**: The system checks if your domain has objective, verifiable answers
        2. **Question Generation**: 10 domain-specific questions are automatically generated
        3. **Response Generation**: For each question, the model generates a main response and N samples
        4. **Claim Extraction**: Responses are split into individual verifiable claims
        5. **Consistency Checking**: Each claim is verified against all samples
        6. **Scoring**: Inconsistency scores indicate hallucination risk
        
        **Score Interpretation:**
        - **0.0 - 0.3**: Low hallucination risk (green) - Claims are consistent
        - **0.3 - 0.6**: Medium hallucination risk (yellow) - Some inconsistencies detected
        - **0.6 - 1.0**: High hallucination risk (red) - Highly inconsistent
        
        **Supported Models (all on Groq):**
        - llama-3.1-8b-instant
        - llama-3.3-70b-versatile
        - moonshotai/kimi-k2-instruct
        - openai/gpt-oss-20b
        - openai/gpt-oss-120b
        
        The overall inconsistency score averages all claim-level scores across all tested questions.
        """)

if __name__ == "__main__":
    main()