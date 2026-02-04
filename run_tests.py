#!/usr/bin/env python3
"""
Test Runner for ACOC
====================
Comprehensive test runner that executes all test suites and provides a summary.
"""

import sys
import subprocess
from pathlib import Path


def run_all_tests():
    """Run all test suites."""
    print("=" * 70)
    print("ACOC Test Suite Runner")
    print("=" * 70)
    print()

    # Get project root
    project_root = Path(__file__).parent

    # Run pytest with verbose output
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(project_root / "tests"),
        "-v",
        "--tb=short",
        "-ra"
    ]

    print(f"Running command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, cwd=project_root)
        return result.returncode
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def run_specific_suite(suite_name):
    """Run a specific test suite."""
    project_root = Path(__file__).parent
    test_file = project_root / "tests" / f"test_{suite_name}.py"

    if not test_file.exists():
        print(f"Error: Test file {test_file} not found")
        return 1

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(test_file),
        "-v"
    ]

    print(f"Running {suite_name} tests...")
    print()

    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode


def run_with_coverage():
    """Run tests with coverage report."""
    project_root = Path(__file__).parent

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(project_root / "tests"),
        "--cov=src/acoc",
        "--cov-report=html",
        "--cov-report=term",
        "-v"
    ]

    print("Running tests with coverage...")
    print()

    try:
        result = subprocess.run(cmd, cwd=project_root)
        if result.returncode == 0:
            print()
            print("Coverage report generated in htmlcov/index.html")
        return result.returncode
    except Exception as e:
        print(f"Error: {e}")
        print("Note: pytest-cov may not be installed. Install with: pip install pytest-cov")
        return 1


def list_test_suites():
    """List all available test suites."""
    project_root = Path(__file__).parent
    tests_dir = project_root / "tests"

    if not tests_dir.exists():
        print("No tests directory found")
        return

    print("Available test suites:")
    print()

    test_files = sorted(tests_dir.glob("test_*.py"))
    for test_file in test_files:
        suite_name = test_file.stem.replace("test_", "")
        print(f"  - {suite_name:20s} ({test_file.name})")

    print()
    print(f"Total: {len(test_files)} test suites")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ACOC Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py              # Run all tests
  python run_tests.py --suite router    # Run router tests only
  python run_tests.py --coverage   # Run with coverage report
  python run_tests.py --list       # List all test suites
        """
    )

    parser.add_argument(
        "--suite",
        type=str,
        help="Run a specific test suite (e.g., router, model, experts)"
    )

    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run tests with coverage report"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available test suites"
    )

    args = parser.parse_args()

    if args.list:
        list_test_suites()
        return 0

    if args.coverage:
        return run_with_coverage()

    if args.suite:
        return run_specific_suite(args.suite)

    return run_all_tests()


if __name__ == "__main__":
    sys.exit(main())
