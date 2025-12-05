#!/usr/bin/env python3
"""
ML Model Evaluation System - Run Script
Cross-platform version that works on Windows, Mac, and Linux
"""

import sys
import subprocess
import os

def run_install():
    """Install dependencies"""
    print("Installing ML Model Evaluation System dependencies...", file=sys.stderr)
    
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--user"],
        capture_output=True
    )
    
    if result.returncode == 0:
        print("✓ All dependencies installed successfully!", file=sys.stderr)
        sys.exit(0)
    else:
        print("Error: Failed to install dependencies", file=sys.stderr)
        sys.exit(1)

def run_tests():
    """Run test suite and output in required format"""
    # Run pytest
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "backend/src/Testing/",
         "--cov=backend/src", "--cov-report=term", "-q"],
        capture_output=True,
        text=True
    )
    
    # Show full output to stderr
    print(result.stdout, file=sys.stderr, end='')
    print(result.stderr, file=sys.stderr, end='')
    
    # Parse the output
    output = result.stdout + result.stderr
    passed = 0
    failed = 0
    coverage = 0
    
    # Extract passed tests
    import re
    passed_match = re.search(r'(\d+) passed', output)
    if passed_match:
        passed = int(passed_match.group(1))
    
    # Extract failed tests
    failed_match = re.search(r'(\d+) failed', output)
    if failed_match:
        failed = int(failed_match.group(1))
    
    # Extract coverage
    coverage_match = re.search(r'TOTAL.*?(\d+)%', output)
    if coverage_match:
        coverage = int(coverage_match.group(1))
    
    total = passed + failed
    
    # Output in EXACT required format to stdout
    print(f"{passed}/{total} test cases passed. {coverage}% line coverage achieved.")
    
    sys.exit(result.returncode)

def run_evaluation(url_file):
    """Run evaluation on URL file"""
    if not os.path.exists(url_file):
        print(f"Error: URL file not found: {url_file}", file=sys.stderr)
        sys.exit(1)
    
    # Change to backend directory and run main.py
    os.chdir("backend")
    result = subprocess.run([sys.executable, "src/main.py", url_file])
    sys.exit(result.returncode)

def main():
    if len(sys.argv) < 2:
        print("Usage: ./run [install|test|URL_FILE]", file=sys.stderr)
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "install":
        run_install()
    elif command == "test":
        run_tests()
    elif command in ["-h", "--help", "help"]:
        print("Usage: ./run [install|test|URL_FILE]", file=sys.stderr)
        print("", file=sys.stderr)
        print("Commands:", file=sys.stderr)
        print("  install     Install all required dependencies", file=sys.stderr)
        print("  test        Run the test suite", file=sys.stderr)
        print("  URL_FILE    Run evaluation on URLs in the specified file", file=sys.stderr)
        sys.exit(0)
    else:
        # Assume it's a URL file
        run_evaluation(command)

if __name__ == "__main__":
    main()