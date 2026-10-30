
"""
genetic_neural.py

Project: Intelligent Healthcare Support System
Technique: Genetic-Neural Network (Neuro-GA)

Purpose:
    Use a Genetic Algorithm to optimize the weights and biases
    of a small neural network for heart disease classification.

How it works:
    1. Create a population of candidate neural network parameters.
    2. Evaluate each candidate using training data.
    3. Select better candidates.
    4. Apply crossover and mutation to create new candidates.
    5. Repeat for several generations.
    6. Evaluate the best candidate on validation and test data.

Important:
    - Genetic operations are implemented manually.
    - NumPy is used for numerical calculations.
    - The data comes from data_preprocessing.py.
    - This is an academic experiment, not a medical diagnostic tool.
"""

import random
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data_preprocessing import load_and_preprocess


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

random.seed(42)
np.random.seed(42)

POPULATION_SIZE = 30
GENERATIONS = 50
MUTATION_RATE = 0.10
TOURNAMENT_SIZE = 3

# Neural network architecture:
# Number of input features -> 8 hidden neurons -> 1 output neuron.
HIDDEN_NEURONS = 8


# ---------------------------------------------------------
# 2. NEURAL NETWORK PARAMETER REPRESENTATION
# ---------------------------------------------------------

def get_parameter_count(input_size):
    """
    Calculate the total number of weights and biases.

    Input -> Hidden:
        input_size * HIDDEN_NEURONS weights
        HIDDEN_NEURONS biases

    Hidden -> Output:
        HIDDEN_NEURONS weights
        1 output bias
    """
    input_to_hidden = input_size * HIDDEN_NEURONS
    hidden_biases = HIDDEN_NEURONS

    hidden_to_output = HIDDEN_NEURONS
    output_bias = 1

    return (
        input_to_hidden
        + hidden_biases
        + hidden_to_output
        + output_bias
    )


def create_individual(input_size):
    """
    Create one candidate solution.

    An individual is a one-dimensional array containing
    all weights and biases of the neural network.
    """
    number_of_parameters = get_parameter_count(input_size)

    # Small random values are used for initial parameters.
    return np.random.uniform(
        low=-1.0,
        high=1.0,
        size=number_of_parameters
    )


# ---------------------------------------------------------
# 3. DECODE AN INDIVIDUAL
# ---------------------------------------------------------

def decode_individual(individual, input_size):
    """
    Convert the one-dimensional chromosome into neural
    network weight matrices and bias vectors.
    """
    index = 0

    # Input-to-hidden weights
    number_of_weights = input_size * HIDDEN_NEURONS

    W1 = individual[
        index:index + number_of_weights
    ].reshape(input_size, HIDDEN_NEURONS)

    index += number_of_weights

    # Hidden layer biases
    b1 = individual[
        index:index + HIDDEN_NEURONS
    ]

    index += HIDDEN_NEURONS

    # Hidden-to-output weights
    W2 = individual[
        index:index + HIDDEN_NEURONS
    ].reshape(HIDDEN_NEURONS, 1)

    index += HIDDEN_NEURONS

    # Output bias
    b2 = individual[index:index + 1]

    return W1, b1, W2, b2


# ---------------------------------------------------------
# 4. FORWARD PROPAGATION
# ---------------------------------------------------------

def sigmoid(values):
    """
    Convert values into numbers between 0 and 1.

    Clipping prevents numerical overflow for very large values.
    """
    values = np.clip(values, -500, 500)
    return 1.0 / (1.0 + np.exp(-values))


def predict_probabilities(individual, X):
    """
    Run the neural network forward.

    Hidden layer:
        Weighted sum -> ReLU activation

    Output layer:
        Weighted sum -> Sigmoid activation

    Returns the probability of the positive class.
    """
    input_size = X.shape[1]

    W1, b1, W2, b2 = decode_individual(
        individual,
        input_size
    )

    # First layer: input features to hidden neurons.
    hidden_values = np.dot(X, W1) + b1

    # ReLU activation: negative values become zero.
    hidden_activations = np.maximum(0, hidden_values)

    # Second layer: hidden neurons to output neuron.
    output_values = np.dot(hidden_activations, W2) + b2

    # Sigmoid converts the output to a probability.
    probabilities = sigmoid(output_values).ravel()

    return probabilities


# ---------------------------------------------------------
# 5. FITNESS FUNCTION
# ---------------------------------------------------------

def calculate_fitness(individual, X_train, y_train):
    """
    Evaluate how well a candidate neural network performs.

    Binary cross-entropy loss is used as the objective.
    Lower loss means a better candidate.

    Fitness = 1 / (1 + loss)

    This converts loss into a positive fitness value,
    where a higher value indicates a better candidate.
    """
    probabilities = predict_probabilities(individual, X_train)

    # Avoid log(0) when calculating cross-entropy.
    probabilities = np.clip(
        probabilities,
        1e-7,
        1.0 - 1e-7
    )

    y = np.asarray(y_train).astype(float).ravel()

    loss = -np.mean(
        y * np.log(probabilities)
        + (1.0 - y) * np.log(1.0 - probabilities)
    )

    fitness = 1.0 / (1.0 + loss)

    return fitness, loss


# ---------------------------------------------------------
# 6. SELECTION
# ---------------------------------------------------------

def select_parent(population, fitness_values):
    """
    Tournament selection:

    1. Randomly choose a few individuals.
    2. Compare their fitness values.
    3. Return the individual with the highest fitness.
    """
    candidate_indices = random.sample(
        range(len(population)),
        min(TOURNAMENT_SIZE, len(population))
    )

    winner_index = max(
        candidate_indices,
        key=lambda i: fitness_values[i]
    )

    return population[winner_index].copy()


# ---------------------------------------------------------
# 7. CROSSOVER
# ---------------------------------------------------------

def crossover(parent1, parent2):
    """
    Single-point crossover.

    The child receives the first section of parameters
    from parent1 and the remaining section from parent2.
    """
    if len(parent1) < 2:
        return parent1.copy()

    crossover_point = random.randint(1, len(parent1) - 1)

    child = np.concatenate([
        parent1[:crossover_point],
        parent2[crossover_point:]
    ])

    return child


# ---------------------------------------------------------
# 8. MUTATION
# ---------------------------------------------------------

def mutate(individual):
    """
    Randomly modify some neural network parameters.

    Each parameter has a MUTATION_RATE chance of mutation.
    A small random value is added to a mutated parameter.
    """
    child = individual.copy()

    for index in range(len(child)):
        if random.random() < MUTATION_RATE:
            child[index] += np.random.normal(
                loc=0.0,
                scale=0.2
            )

    return child


# ---------------------------------------------------------
# 9. TRAIN THE GENETIC-NEURAL NETWORK
# ---------------------------------------------------------

def train_genetic_neural_network(X_train, y_train):
    """
    Evolve a population of neural networks and return
    the best individual found during training.
    """
    input_size = X_train.shape[1]

    # Step 1: Create the initial population.
    population = [
        create_individual(input_size)
        for _ in range(POPULATION_SIZE)
    ]

    best_individual = None
    best_fitness = -float("inf")

    for generation in range(GENERATIONS):

        # Step 2: Evaluate every individual.
        fitness_values = []
        losses = []

        for individual in population:
            fitness, loss = calculate_fitness(
                individual,
                X_train,
                y_train
            )

            fitness_values.append(fitness)
            losses.append(loss)

        # Step 3: Identify the best individual this generation.
        generation_best_index = int(np.argmax(fitness_values))
        generation_best_fitness = fitness_values[
            generation_best_index
        ]

        # Preserve the best solution found across generations.
        if generation_best_fitness > best_fitness:
            best_fitness = generation_best_fitness
            best_individual = population[
                generation_best_index
            ].copy()

        # Step 4: Print training progress.
        if (
            generation == 0
            or (generation + 1) % 5 == 0
        ):
            best_loss = losses[generation_best_index]

            print(
                f"Generation {generation + 1:02d}/{GENERATIONS} | "
                f"Best Loss: {best_loss:.4f} | "
                f"Best Fitness: {generation_best_fitness:.4f}"
            )

        # Step 5: Elitism.
        # Keep the best individual unchanged in the next generation.
        next_population = [best_individual.copy()]

        # Step 6: Generate the rest of the population.
        while len(next_population) < POPULATION_SIZE:

            parent1 = select_parent(
                population,
                fitness_values
            )

            parent2 = select_parent(
                population,
                fitness_values
            )

            child = crossover(parent1, parent2)
            child = mutate(child)

            next_population.append(child)

        population = next_population

    return best_individual


# ---------------------------------------------------------
# 10. EVALUATION
# ---------------------------------------------------------

def evaluate_model(individual, X, y, dataset_name):
    """
    Evaluate the trained model on a dataset.

    A threshold of 0.5 is used for classification.
    The test set is not used during genetic optimization.
    """
    probabilities = predict_probabilities(individual, X)
    predictions = (probabilities >= 0.5).astype(int)

    y = np.asarray(y).astype(int).ravel()

    accuracy = accuracy_score(y, predictions)
    f1 = f1_score(y, predictions, zero_division=0)

    # ROC AUC requires both classes to be present.
    if len(np.unique(y)) == 2:
        auc = roc_auc_score(y, probabilities)
    else:
        auc = float("nan")

    print(f"\n===== {dataset_name} Results =====")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")

    return {
        "accuracy": accuracy,
        "f1": f1,
        "roc_auc": auc,
    }


# ---------------------------------------------------------
# 11. MAIN PROGRAM
# ---------------------------------------------------------

def main():
    """
    Load the project's preprocessed data, train the model,
    and evaluate it on validation and test partitions.
    """
    print("Loading preprocessed healthcare data...")

    data = load_and_preprocess()

    # Use standardized neural network inputs.
    X_train = np.asarray(data["nn"]["X_train"], dtype=float)
    X_val = np.asarray(data["nn"]["X_val"], dtype=float)
    X_test = np.asarray(data["nn"]["X_test"], dtype=float)

    y_train = np.asarray(data["y_train"], dtype=int)
    y_val = np.asarray(data["y_val"], dtype=int)
    y_test = np.asarray(data["y_test"], dtype=int)

    print(f"Training samples: {len(y_train)}")
    print(f"Validation samples: {len(y_val)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Input features: {X_train.shape[1]}")

    print("\nStarting Genetic-Neural Network training...")

    best_individual = train_genetic_neural_network(
        X_train,
        y_train
    )

    # Validation data is used for intermediate evaluation.
    evaluate_model(
        best_individual,
        X_val,
        y_val,
        "Validation"
    )

    # Test data is held out from the optimization process.
    evaluate_model(
        best_individual,
        X_test,
        y_test,
        "Test"
    )

    print("\nTraining and evaluation completed.")
    print("This is an academic experiment, not a medical diagnosis.")


if __name__ == "__main__":
    main()
