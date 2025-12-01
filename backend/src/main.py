"""
ML Model Evaluation System - Main Entry Point
Evaluates machine learning models from HuggingFace with comprehensive metrics.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

# ============================================================================
# Environment Variable Setup and Validation
# ============================================================================

def setup_logging():
    """
    Setup logging based on LOG_FILE and LOG_LEVEL environment variables.
    
    LOG_LEVEL:
        0 = Silent (only errors)
        1 = Info messages
        2 = Debug messages
    
    LOG_FILE: Path to log file. If invalid path, exit with error.
    """
    log_file = os.getenv('LOG_FILE')
    log_level_str = os.getenv('LOG_LEVEL', '0')
    
    # Parse log level
    try:
        log_level = int(log_level_str)
    except ValueError:
        log_level = 0
    
    # Map log level to logging constants
    if log_level == 0:
        level = logging.CRITICAL  # Silent mode
    elif log_level == 1:
        level = logging.INFO
    elif log_level == 2:
        level = logging.DEBUG
    else:
        level = logging.CRITICAL
    
    # Validate log file path if provided
    if log_file:
        log_path = Path(log_file)
        
        # Get parent directory - handle both absolute and relative paths
        parent_dir = log_path.parent
        
        # For relative paths or paths with just filename, parent might be empty
        if str(parent_dir) == '.' or str(parent_dir) == '':
            parent_dir = Path.cwd()
        
        # Convert to absolute path for checking
        try:
            parent_dir = parent_dir.resolve()
        except Exception:
            print(f"Error: Invalid log file path: {log_file}", file=sys.stderr)
            sys.exit(1)
        
        # Check if parent directory exists
        if not parent_dir.exists():
            print(f"Error: Log file directory does not exist: {parent_dir}", 
                  file=sys.stderr)
            sys.exit(1)
        
        # Check if it's actually a directory
        if not parent_dir.is_dir():
            print(f"Error: Log file parent path is not a directory: {parent_dir}", 
                  file=sys.stderr)
            sys.exit(1)
        
        # Check if parent directory is writable
        if not os.access(parent_dir, os.W_OK):
            print(f"Error: Log file directory is not writable: {parent_dir}", 
                  file=sys.stderr)
            sys.exit(1)
        
        # Try to create/open the log file to verify we can write to it
        try:
            # Test if we can open the file for writing
            with open(log_file, 'a') as f:
                pass
        except (OSError, IOError, PermissionError) as e:
            print(f"Error: Cannot write to log file {log_file}: {e}", 
                  file=sys.stderr)
            sys.exit(1)
        
        # Setup file logging
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stderr)
            ]
        )
    else:
        # Setup console-only logging
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stderr)]
        )
    
    return logging.getLogger(__name__)


def validate_github_token():
    """
    Validate GitHub token if provided.
    If token is invalid format, print error and exit.
    """
    github_token = os.getenv('GITHUB_TOKEN')
    
    if github_token:
        # GitHub tokens have specific prefixes
        valid_prefixes = ['ghp_', 'github_pat_', 'gho_', 'ghu_', 'ghs_', 'ghr_']
        
        if not any(github_token.startswith(prefix) for prefix in valid_prefixes):
            print("Error: Invalid GitHub token format", file=sys.stderr)
            sys.exit(1)
    
    return github_token


def validate_environment():
    """
    Validate all environment variables before starting.
    """
    # Validate GitHub token
    validate_github_token()
    
    # Setup logging (will exit if LOG_FILE path is invalid)
    logger = setup_logging()
    
    return logger


# ============================================================================
# Import after environment validation setup
# ============================================================================

from Controllers.Controller import Controller
from Services.Metric_Model_Service import ModelMetricService
from lib.Metric_Result import MetricResult


# ============================================================================
# Input Parsing
# ============================================================================

def parse_input(file_path: str) -> List[Dict[str, Optional[str]]]:
    """
    Parse input file containing URLs.
    
    Format: code_link,dataset_link,model_link
    Or just: model_link
    
    Returns list of dicts with keys: model_link, dataset_link, code_link
    """
    jobs = []
    
    # Resolve relative paths
    if not os.path.isabs(file_path):
        project_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), '..', '..')
        )
        file_path = os.path.join(project_root, file_path)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Parse CSV format: code_link,dataset_link,model_link
                parts = [p.strip() for p in line.split(',')]
                
                # Ensure we have at least 3 parts
                while len(parts) < 3:
                    parts.insert(0, '')
                
                code_link = parts[0] if parts[0] else None
                dataset_link = parts[1] if parts[1] else None
                model_link = parts[2] if parts[2] else None
                
                # Must have at least a model link
                if model_link:
                    jobs.append({
                        'model_link': model_link,
                        'dataset_link': dataset_link,
                        'code_link': code_link
                    })
    
    except FileNotFoundError:
        logging.error(f"Input file not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error parsing input file: {e}")
        sys.exit(1)
    
    return jobs


# ============================================================================
# Model Name Extraction
# ============================================================================

def extract_model_name(model_link: str) -> str:
    """
    Extract model name from HuggingFace URL.
    
    Examples:
        https://huggingface.co/google-bert/bert-base-uncased -> bert-base-uncased
        https://huggingface.co/gpt2 -> gpt2
    """
    try:
        if not model_link or not isinstance(model_link, str):
            return "unknown_model"
        
        # Remove query parameters
        if '?' in model_link:
            model_link = model_link.split('?')[0]
        
        # Remove trailing slashes and /tree/main
        model_link = model_link.rstrip('/')
        if '/tree/' in model_link:
            model_link = model_link.split('/tree/')[0]
        
        # Extract model name from URL
        if 'huggingface.co/' in model_link:
            parts = model_link.split('/')
            if len(parts) >= 4:
                # Return the last part (model name)
                return parts[-1]
        
        return "unknown_model"
    except Exception:
        return "unknown_model"


# ============================================================================
# Link Discovery
# ============================================================================

def find_missing_links(model_link: str, dataset_link: Optional[str], 
                       code_link: Optional[str]) -> Tuple[List[str], Optional[str]]:
    """
    Discover missing dataset and code links from model card.
    Returns: (dataset_links, code_link)
    """
    import re
    from lib.HuggingFace_API_Manager import HuggingFaceAPIManager
    
    dataset_links = []
    discovered_code = None
    
    # Add provided links
    if dataset_link:
        dataset_links.append(dataset_link)
    
    if code_link:
        discovered_code = code_link
    
    # Try to discover from model card
    try:
        hf_manager = HuggingFaceAPIManager()
        model_id = hf_manager.model_link_to_id(model_link)
        model_info = hf_manager.get_model_info(model_id)
        
        if model_info:
            # Search in card data
            if hasattr(model_info, 'cardData') and model_info.cardData:
                card_text = str(model_info.cardData)
                
                # Find dataset links
                dataset_patterns = [
                    r'https://huggingface\.co/datasets/([^\s\)]+)',
                    r'datasets/([^\s\)]+)',
                ]
                
                for pattern in dataset_patterns:
                    matches = re.findall(pattern, card_text, re.IGNORECASE)
                    for match in matches:
                        if match.startswith('http'):
                            dataset_url = match
                        else:
                            dataset_url = f"https://huggingface.co/datasets/{match}"
                        
                        if dataset_url not in dataset_links:
                            dataset_links.append(dataset_url)
                
                # Find code links (GitHub)
                if not discovered_code:
                    code_patterns = [
                        r'https://github\.com/([^\s\)]+)',
                        r'github\.com/([^\s\)]+)',
                        r'repo:\s*([^\s]+)',
                        r'code:\s*([^\s]+)',
                    ]
                    
                    for pattern in code_patterns:
                        matches = re.findall(pattern, card_text, re.IGNORECASE)
                        if matches:
                            match = matches[0]
                            if match.startswith('http'):
                                discovered_code = match
                            else:
                                discovered_code = f"https://github.com/{match}"
                            break
            
            # Check tags for dataset info
            if hasattr(model_info, 'tags') and model_info.tags:
                for tag in model_info.tags:
                    if tag.startswith('dataset:'):
                        dataset_name = tag.replace('dataset:', '')
                        dataset_url = f"https://huggingface.co/datasets/{dataset_name}"
                        if dataset_url not in dataset_links:
                            dataset_links.append(dataset_url)
            
            # Check modelId for potential GitHub repo
            if not discovered_code and hasattr(model_info, 'modelId') and model_info.modelId:
                # Many models follow pattern: org/model-name -> github.com/org/model-name
                if '/' in model_info.modelId:
                    potential_repo = f"https://github.com/{model_info.modelId}"
                    discovered_code = potential_repo
        
        # Log discoveries
        if len(dataset_links) > 3:
            logging.info(f"Discovered {len(dataset_links)} datasets: "
                        f"{', '.join(dataset_links[:3])}, ...")
        elif dataset_links:
            logging.info(f"Discovered datasets: {', '.join(dataset_links)}")
        
        if discovered_code:
            logging.info(f"Discovered code repository: {discovered_code}")
    
    except Exception as e:
        logging.warning(f"Could not discover additional links: {e}")
    
    return dataset_links, discovered_code


# ============================================================================
# Evaluation Timing
# ============================================================================

def time_evaluation(eval_func, *args, **kwargs) -> Tuple[MetricResult, float]:
    """
    Time the execution of an evaluation function.
    Returns: (result, execution_time_seconds)
    """
    start_time = time.time()
    result = eval_func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    
    return result, execution_time


# ============================================================================
# Evaluation Runners
# ============================================================================

def run_evaluations_sequential(model_data) -> Dict[str, Tuple[MetricResult, float]]:
    """Run all evaluations sequentially."""
    service = ModelMetricService()
    results = {}
    
    evaluations = [
        ("Performance Claims", service.EvaluatePerformanceClaims),
        ("Bus Factor", service.EvaluateBusFactor),
        ("Size", service.EvaluateSize),
        ("Ramp-Up Time", service.EvaluateRampUpTime),
        ("Availability", service.EvaluateDatasetAndCodeAvailabilityScore),
        ("Code Quality", service.EvaluateCodeQuality),
        ("Dataset Quality", service.EvaluateDatasetsQuality),
        ("License", service.EvaluateLicense),
    ]
    
    for name, eval_func in evaluations:
        result, exec_time = time_evaluation(eval_func, model_data)
        results[name] = (result, exec_time)
    
    return results


def run_evaluations_parallel(model_data, max_workers: int = 4) -> Dict[str, Tuple[MetricResult, float]]:
    """Run evaluations in parallel for better performance."""
    service = ModelMetricService()
    results = {}
    
    evaluations = [
        ("Performance Claims", service.EvaluatePerformanceClaims),
        ("Bus Factor", service.EvaluateBusFactor),
        ("Size", service.EvaluateSize),
        ("Ramp-Up Time", service.EvaluateRampUpTime),
        ("Availability", service.EvaluateDatasetAndCodeAvailabilityScore),
        ("Code Quality", service.EvaluateCodeQuality),
        ("Dataset Quality", service.EvaluateDatasetsQuality),
        ("License", service.EvaluateLicense),
    ]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_name = {
            executor.submit(time_evaluation, eval_func, model_data): name
            for name, eval_func in evaluations
        }
        
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                result, exec_time = future.result()
                results[name] = (result, exec_time)
            except Exception as e:
                logging.error(f"Evaluation '{name}' failed: {e}")
    
    return results


# ============================================================================
# Score Calculation
# ============================================================================

def calculate_net_score(results: Dict[str, Tuple[MetricResult, float]]) -> float:
    """
    Calculate net score as weighted average of all metrics.
    
    Weights based on Sarah's priorities from Phase 1 spec.
    """
    weights = {
        "License": 0.20,
        "Ramp-Up Time": 0.15,
        "Bus Factor": 0.15,
        "Performance Claims": 0.15,
        "Size": 0.10,
        "Availability": 0.10,
        "Dataset Quality": 0.075,
        "Code Quality": 0.075
    }
    
    weighted_sum = 0.0
    total_weight = 0.0
    
    for name, (result, _) in results.items():
        if name in weights and result and hasattr(result, 'value'):
            value = max(0.0, min(1.0, float(result.value)))
            weighted_sum += value * weights[name]
            total_weight += weights[name]
    
    return weighted_sum / total_weight if total_weight > 0 else 0.0


# ============================================================================
# Output Formatting
# ============================================================================

def format_size_score(result: MetricResult) -> Dict[str, float]:
    """Format size score as dict with hardware targets."""
    if result and hasattr(result, 'value'):
        size_value = max(0.0, min(1.0, float(result.value)))
        return {
            "raspberry_pi": round(size_value * 0.2, 2),
            "jetson_nano": round(size_value * 0.4, 2),
            "desktop_pc": round(size_value * 0.8, 2),
            "aws_server": round(size_value, 2)
        }
    return {
        "raspberry_pi": 0.0,
        "jetson_nano": 0.0,
        "desktop_pc": 0.0,
        "aws_server": 0.0
    }


def handle_missing_model_data(model_link: str) -> str:
    """Generate output for a model that failed to fetch."""
    model_name = extract_model_name(model_link)
    
    output = {
        "name": model_name,
        "category": "MODEL",
        "net_score": 0.0,
        "net_score_latency": 1,
        "ramp_up_time": 0.0,
        "ramp_up_time_latency": 1,
        "bus_factor": 0.0,
        "bus_factor_latency": 1,
        "performance_claims": 0.0,
        "performance_claims_latency": 1,
        "license": 0.0,
        "license_latency": 1,
        "size_score": {
            "raspberry_pi": 0.0,
            "jetson_nano": 0.0,
            "desktop_pc": 0.0,
            "aws_server": 0.0
        },
        "size_score_latency": 1,
        "dataset_and_code_score": 0.0,
        "dataset_and_code_score_latency": 1,
        "dataset_quality": 0.0,
        "dataset_quality_latency": 1,
        "code_quality": 0.0,
        "code_quality_latency": 1
    }
    
    return json.dumps(output)


# ============================================================================
# Timing Summary
# ============================================================================

def print_timing_summary(results: Dict[str, Tuple[MetricResult, float]], 
                        total_time: float):
    """Print timing summary for all evaluations."""
    logging.info("=" * 60)
    logging.info("EVALUATION TIMING SUMMARY")
    logging.info("=" * 60)
    
    for name, (result, exec_time) in results.items():
        logging.info(f"{name:30s}: {exec_time:6.3f}s (score: {result.value:.2f})")
    
    logging.info("-" * 60)
    logging.info(f"{'Total Time':30s}: {total_time:6.3f}s")
    logging.info("=" * 60)


# ============================================================================
# Batch Evaluation
# ============================================================================

def run_batch_evaluation(input_file_path: str):
    """Run batch evaluation on URLs from input file."""
    # Validate environment FIRST
    logger = validate_environment()
    
    # Parse input file
    jobs = parse_input(input_file_path)
    
    if not jobs:
        logger.error("No valid URLs found in input file")
        sys.exit(1)
    
    logger.info(f"Processing {len(jobs)} model(s)")
    
    # Initialize controller
    controller = Controller()
    
    # Process each job
    for job in jobs:
        model_link = job['model_link']
        dataset_links = [job['dataset_link']] if job['dataset_link'] else []
        code_link = job['code_link']
        
        # Find any missing links
        dataset_links, code_link = find_missing_links(
            model_link, 
            dataset_links[0] if dataset_links else None,
            code_link
        )
        
        try:
            # Fetch model data
            logger.info(f"Fetching data for: {model_link}")
            model_data = controller.fetch(model_link, dataset_links, code_link)
            
            if not model_data:
                logger.warning(f"Failed to fetch data for {model_link}")
                print(handle_missing_model_data(model_link))
                continue
            
            # Run evaluations in parallel
            logger.info("Running evaluations...")
            start_time = time.time()
            results = run_evaluations_parallel(model_data)
            total_time = time.time() - start_time
            
            # Print timing summary to log
            print_timing_summary(results, total_time)
            
            # Extract model name
            model_name = extract_model_name(model_link)
            
            # Calculate net score
            net_score = calculate_net_score(results)
            total_latency = sum(t * 1000 for _, (_, t) in results.items())
            
            # Build output dict
            output = {
                "name": model_name,
                "category": "MODEL",
                "net_score": round(max(0.0, min(1.0, net_score)), 2),
                "net_score_latency": max(1, int(round(total_latency)))
            }
            
            # Add all metrics
            field_mapping = [
                ("Ramp-Up Time", "ramp_up_time"),
                ("Bus Factor", "bus_factor"),
                ("Performance Claims", "performance_claims"),
                ("License", "license"),
                ("Size", "size_score"),
                ("Availability", "dataset_and_code_score"),
                ("Dataset Quality", "dataset_quality"),
                ("Code Quality", "code_quality")
            ]
            
            for name, field_name in field_mapping:
                if name in results:
                    result, exec_time = results[name]
                    
                    if field_name == "size_score":
                        output["size_score"] = format_size_score(result)
                    else:
                        value = result.value if hasattr(result, 'value') else 0.0
                        output[field_name] = round(max(0.0, min(1.0, value)), 2)
                    
                    output[f"{field_name}_latency"] = max(1, int(round(exec_time * 1000)))
                else:
                    # Missing metric, use defaults
                    if field_name == "size_score":
                        output["size_score"] = {
                            "raspberry_pi": 0.0,
                            "jetson_nano": 0.0,
                            "desktop_pc": 0.0,
                            "aws_server": 0.0
                        }
                    else:
                        output[field_name] = 0.0
                    output[f"{field_name}_latency"] = 1
            
            # Print JSON to stdout (autograder reads this)
            print(json.dumps(output))
            
        except Exception as e:
            logger.error(f"Error processing {model_link}: {e}", exc_info=True)
            print(handle_missing_model_data(model_link))


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <url_file>", file=sys.stderr)
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Resolve relative paths
    if not os.path.isabs(input_file):
        project_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), '..', '..')
        )
        input_file = os.path.join(project_root, input_file)
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)
    
    run_batch_evaluation(input_file)