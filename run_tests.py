#!/usr/bin/env python3
"""
Run all unit tests for the Anthropic-powered Agent
"""

import os
import sys
import unittest
import coverage
import argparse

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_tests(coverage_report=False):
    """Run all unit tests with optional coverage report."""
    
    if coverage_report:
        # Start coverage measurement
        cov = coverage.Coverage(
            source=["anthropic_agent", "rag", "image_processing", "web_api"],
            omit=["*/__pycache__/*", "*/tests/*"]
        )
        cov.start()
    
    # Discover and run tests
    loader = unittest.TestLoader()
    tests_dir = os.path.abspath(os.path.dirname(__file__))
    suite = loader.discover(tests_dir)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if coverage_report:
        # Generate coverage report
        cov.stop()
        cov.save()
        
        print("\nCoverage Report:")
        cov.report()
        
        # Generate HTML report
        cov.html_report(directory="coverage_html")
        print(f"\nHTML coverage report generated in: {os.path.abspath('coverage_html')}")
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run unit tests for the Anthropic-powered Agent")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    args = parser.parse_args()
    
    result = run_tests(coverage_report=args.coverage)
    
    # Exit with non-zero code if tests failed
    sys.exit(not result.wasSuccessful())