# Intelligent Healthcare Support System

An academic Computational Intelligence project integrating machine learning, neural networks, fuzzy logic, evolutionary computation, and optimization techniques for healthcare prediction and resource management.

## Overview

The Intelligent Healthcare Support System explores how computational intelligence techniques can support healthcare-related prediction, risk estimation, patient scheduling, and hospital resource allocation.

The project combines data-driven learning with fuzzy reasoning and nature-inspired optimization algorithms to demonstrate different approaches to healthcare decision-support problems.

**Note:** This project is an educational prototype and is not clinically validated. It must not be used for medical diagnosis or treatment decisions.

## Objectives

- Predict the presence or absence of heart disease using clinical attributes.
- Estimate illustrative health-risk scores using fuzzy logic.
- Explore optimization techniques for hospital resource allocation.
- Simulate patient scheduling using Ant Colony Optimization.
- Demonstrate genetic algorithms for neural-network parameter optimization.
- Explore neuro-fuzzy learning using an Adaptive Neuro-Fuzzy Inference System (ANFIS).

## Technologies and Techniques

| Technique | Application |
|---|---|
| Neural Network | Binary heart-disease classification |
| Mamdani Fuzzy Logic | Rule-based health-risk estimation |
| Genetic Algorithm | Optimization of hospital resource allocation |
| Ant Colony Optimization (ACO) | Patient scheduling and waiting-time optimization |
| Genetic-Neural System | Exploring genetic optimization of neural-network parameters |
| Neuro-Fuzzy System (ANFIS) | Combining fuzzy inference with learning |

## Dataset

The project presentation describes the use of the **UCI Heart Disease dataset**, with 920 records and 12 selected clinical features.

Example features include:
- Age
- Sex
- Chest-pain type
- Blood pressure
- Cholesterol
- Maximum heart rate
- ST depression

The original target values (`num`, ranging from 0 to 4) are mapped to a binary classification label:

- `0` — No heart disease
- `1` — Heart disease present

The dataset is preprocessed using the following steps:

1. Remove identifier columns and handle missing values.
2. Impute numerical and categorical values.
3. One-hot encode categorical features.
4. Scale numerical inputs for neural-network training.
5. Split the data into stratified training, validation, and test sets.

The presentation specifies a 70/15/15 split with random seed 42.

**Dataset note:** The repository includes `data/heart.csv`. Confirm its source, licensing, and consistency with the dataset described above before relying on the stated dataset statistics.

## Project Modules

### 1. Neural Network for Heart-Disease Prediction

A supervised binary-classification model that learns patterns from clinical features to predict whether heart disease is present.

Key concepts:
- Forward propagation
- Activation functions
- Loss calculation
- Backpropagation
- Weight and bias updates
- Early stopping

**File:** `neural_network.py`

### 2. Mamdani Fuzzy Logic

A rule-based system that estimates an illustrative health-risk score from age, blood pressure, and cholesterol.

The workflow consists of:
1. Fuzzification
2. Fuzzy rule evaluation
3. Implication
4. Aggregation
5. Centroid defuzzification
6. Risk-category assignment

The output is an educational risk estimate, not a medically validated assessment.

**File:** `fuzzy_logic.py`

### 3. Genetic Algorithm for Resource Allocation

Explores the allocation of hospital resources across departments using an evolutionary search process.

The algorithm demonstrates:
- Population initialization
- Fitness evaluation
- Selection
- Crossover
- Mutation
- Candidate-solution evaluation

**File:** `genetic_algorithm.py`

### 4. Ant Colony Optimization for Patient Scheduling

Simulates patient scheduling with the objective of reducing severity-weighted waiting time while accounting for scheduling constraints.

Key concepts:
- Pheromone initialization and updates
- Heuristic desirability
- Probabilistic selection
- Waiting-cost evaluation
- Iterative solution improvement

The scheduling results are based on simulation, not evidence of real-world hospital impact.

**File:** `aco_scheduling.py`

### 5. Genetic-Neural System

Explores the use of genetic algorithms to optimize neural-network parameters such as weights and biases.

The approach illustrates candidate encoding, fitness evaluation, selection, crossover, mutation, and the selection of promising parameter configurations.

**File:** `genetic_neural.py`

### 6. Neuro-Fuzzy System

Demonstrates the combination of fuzzy inference and parameterized learning through a neuro-fuzzy approach.

The illustrative workflow includes:
- Input processing
- Membership-function evaluation
- Rule firing strengths
- Normalized firing strengths
- Rule outputs
- Weighted output calculation

**File:** `neuro_fuzzy.py`

## Repository Structure

```text
intelligent-healthcare-support/
├── data/
│   └── heart.csv
├── aco_scheduling.py
├── data_preprocessing.py
├── fuzzy_logic.py
├── genetic_algorithm.py
├── genetic_neural.py
├── main.py
├── neural_network.py
├── neuro_fuzzy.py
├── requirements.txt
├── ci ppt final (1) (1).pptx
└── README.md
```

## Installation

### Prerequisites

- Python 3.10 or a compatible version supported by the dependencies
- pip

### 1. Clone the repository

```bash
git clone https://github.com/sanketyadav2411/Intelligent-Healthcare-Support.git
cd Intelligent-Healthcare-Support
```

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

Start with the main entry point:

```bash
python main.py
```

Individual modules can be explored separately:

```bash
python neural_network.py
python fuzzy_logic.py
python genetic_algorithm.py
python aco_scheduling.py
python genetic_neural.py
python neuro_fuzzy.py
```

The available entry points and required input files may vary by module. Run each script according to its implementation.

## Limitations

- This project is an academic prototype.
- Fuzzy membership functions and rules are illustrative and are not clinically validated.
- Patient scheduling results use synthetic or simulated data.
- Performance depends on the dataset, preprocessing, model configuration, and evaluation methodology.
- Results should not be interpreted as evidence of clinical effectiveness.

## Future Improvements

- Add reproducible model evaluation and detailed performance metrics.
- Improve input validation and missing-data handling.
- Compare models using consistent training and test splits.
- Add visualizations for prediction performance and optimization progress.
- Develop a unified interface for exploring the different modules.
- Validate algorithms using appropriate datasets and domain-expert review before any clinical application.

## Author

**Sanket Yadav**

Computational Intelligence | Machine Learning | Neural Networks | Fuzzy Logic | Optimization

## Disclaimer

This project is intended for educational and research purposes only. It is not a medical device and must not be used to diagnose diseases, recommend treatments, or replace professional medical advice.
