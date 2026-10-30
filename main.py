
"""
main.py

Project: Intelligent Healthcare Support System

Purpose:
    Provide a central menu to run each project module.

Modules:
    1. Neural Network
    2. Mamdani Fuzzy Logic
    3. Genetic Algorithm
    4. Ant Colony Optimization
    5. Genetic-Neural Network
    6. Neuro-Fuzzy System
"""

import subprocess
import sys
from pathlib import Path


# Get the directory where main.py is stored.
PROJECT_DIR = Path(__file__).resolve().parent


# Map menu choices to module names and filenames.
MODULES = {
    "1": ("Neural Network", "neural_network.py"),
    "2": ("Mamdani Fuzzy Logic", "fuzzy_logic.py"),
    "3": ("Genetic Algorithm", "genetic_algorithm.py"),
    "4": ("Ant Colony Optimization", "aco_scheduling.py"),
    "5": ("Genetic-Neural Network", "genetic_neural.py"),
    "6": ("Neuro-Fuzzy System", "neuro_fuzzy.py"),
}


def run_module(module_name, filename):
    """
    Execute a selected Python module as a separate process.

    Running modules separately keeps their execution independent
    and allows each module to use its own main entry point.
    """
    file_path = PROJECT_DIR / filename

    # Check whether the requested Python file exists.
    if not file_path.is_file():
        print(f"\nError: {filename} was not found.")
        return False

    print("\n" + "=" * 60)
    print(f"Running: {module_name}")
    print("=" * 60)

    try:
        result = subprocess.run(
            [sys.executable, str(file_path)],
            cwd=str(PROJECT_DIR),
            check=False
        )

        if result.returncode == 0:
            print(f"\n{module_name} completed successfully.")
            return True

        print(
            f"\n{module_name} failed with exit code "
            f"{result.returncode}."
        )
        print("Check the error output above.")
        return False

    except OSError as error:
        print(f"\nUnable to execute {filename}: {error}")
        return False


def main():
    """Display the menu and handle the user's selection."""

    while True:
        print("\n" + "=" * 60)
        print("       INTELLIGENT HEALTHCARE SUPPORT SYSTEM")
        print("=" * 60)
        print("1. Run Neural Network")
        print("2. Run Mamdani Fuzzy Logic")
        print("3. Run Genetic Algorithm")
        print("4. Run Ant Colony Optimization")
        print("5. Run Genetic-Neural Network")
        print("6. Run Neuro-Fuzzy System")
        print("7. Run All Modules")
        print("0. Exit")
        print("=" * 60)

        choice = input("Enter your choice: ").strip()

        if choice == "0":
            print("Exiting the application.")
            break

        elif choice in MODULES:
            name, filename = MODULES[choice]
            run_module(name, filename)

        elif choice == "7":
            results = {}

            # Run every module even if one module fails.
            for name, filename in MODULES.values():
                results[name] = run_module(name, filename)

            print("\n" + "=" * 60)
            print("                 EXECUTION SUMMARY")
            print("=" * 60)

            for name, success in results.items():
                status = "SUCCESS" if success else "FAILED"
                print(f"{name}: {status}")

        else:
            print("Invalid choice. Please enter a number from 0 to 7.")


if __name__ == "__main__":
    main()
