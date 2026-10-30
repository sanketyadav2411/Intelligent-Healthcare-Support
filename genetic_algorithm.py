
# ============================================================
# FILE: genetic_algorithm.py
# PROJECT: Intelligent Healthcare Support System
#
# PURPOSE:
# Demonstrate Genetic Algorithm (GA) for allocating
# hospital resources across departments.
#
# IMPLEMENTATION:
# Selection, crossover, mutation, and evolution are coded
# manually. No GA library is used.
#
# NOTE:
# This is a simulated optimization example, not a validated
# hospital staffing or resource-allocation recommendation.
# ============================================================

import random


# ------------------------------------------------------------
# 1. PROBLEM CONFIGURATION
# ------------------------------------------------------------

# Departments that need resource units.
DEPARTMENTS = [
    "Emergency",
    "Cardiology",
    "Neurology",
    "General Medicine"
]

# Total resource units available.
TOTAL_RESOURCES = 100

# Number of candidate solutions in each generation.
POPULATION_SIZE = 30

# Maximum number of generations to evolve.
MAX_GENERATIONS = 100

# Probability that a candidate gene will mutate.
MUTATION_RATE = 0.10

# Make random operations reproducible between runs.
random.seed(42)


# ------------------------------------------------------------
# 2. CREATE A CANDIDATE SOLUTION
# ------------------------------------------------------------

def create_individual():
    """
    Create one candidate resource allocation.

    Each gene represents the resources allocated to one
    department. The allocation always sums to TOTAL_RESOURCES.
    """

    # Generate random weights for the four departments.
    weights = [
        random.uniform(0.1, 1.0)
        for _ in DEPARTMENTS
    ]

    # Convert weights into resource allocations.
    allocations = [
        weight / sum(weights) * TOTAL_RESOURCES
        for weight in weights
    ]

    # Round to integers.
    allocations = [round(value) for value in allocations]

    # Correct rounding differences so the total is exactly 100.
    difference = TOTAL_RESOURCES - sum(allocations)
    allocations[0] += difference

    return allocations


# ------------------------------------------------------------
# 3. DEFINE THE FITNESS FUNCTION
# ------------------------------------------------------------

def fitness(individual):
    """
    Score a candidate solution.

    In this demonstration, the hypothetical demand shares are:
        Emergency        : 40%
        Cardiology       : 25%
        Neurology        : 15%
        General Medicine: 20%

    Higher fitness means the allocation is closer to these
    assumed demand shares.

    These shares are illustrative, not real hospital data.
    """

    demand_shares = [0.40, 0.25, 0.15, 0.20]

    # Calculate the target number of resources per department.
    target = [
        share * TOTAL_RESOURCES
        for share in demand_shares
    ]

    # Total absolute difference from the target allocation.
    total_error = sum(
        abs(individual[i] - target[i])
        for i in range(len(DEPARTMENTS))
    )

    # Convert error into a fitness score.
    # Smaller error produces higher fitness.
    return 1.0 / (1.0 + total_error)


# ------------------------------------------------------------
# 4. SELECT A PARENT
# ------------------------------------------------------------

def select_parent(population):
    """
    Tournament selection:
    1. Randomly select three candidates.
    2. Return the candidate with the highest fitness.
    """

    tournament_size = min(3, len(population))

    candidates = random.sample(
        population,
        tournament_size
    )

    return max(candidates, key=fitness)


# ------------------------------------------------------------
# 5. CROSSOVER
# ------------------------------------------------------------

def crossover(parent1, parent2):
    """
    Single-point crossover combines two parents.

    Genes before the crossover point come from parent 1.
    Remaining genes come from parent 2.
    """

    # Select a crossover point between the genes.
    point = random.randint(1, len(parent1) - 1)

    child = (
        parent1[:point]
        + parent2[point:]
    )

    # Restore the fixed total resource constraint.
    child = repair_allocation(child)

    return child


# ------------------------------------------------------------
# 6. REPAIR A CANDIDATE ALLOCATION
# ------------------------------------------------------------

def repair_allocation(individual):
    """
    Ensure resource values are non-negative integers and
    the total allocation equals TOTAL_RESOURCES.
    """

    # Convert values to non-negative integers.
    repaired = [
        max(0, int(round(value)))
        for value in individual
    ]

    current_total = sum(repaired)

    # Handle the unlikely case where all genes are zero.
    if current_total == 0:
        return create_individual()

    # Rescale the allocation to the available total.
    scaled = [
        value * TOTAL_RESOURCES / current_total
        for value in repaired
    ]

    repaired = [int(value) for value in scaled]

    # Distribute remaining units based on fractional parts.
    remaining = TOTAL_RESOURCES - sum(repaired)

    fractions = [
        scaled[i] - repaired[i]
        for i in range(len(repaired))
    ]

    order = sorted(
        range(len(fractions)),
        key=lambda i: fractions[i],
        reverse=True
    )

    for i in range(remaining):
        repaired[order[i % len(order)]] += 1

    return repaired


# ------------------------------------------------------------
# 7. MUTATION
# ------------------------------------------------------------

def mutate(individual):
    """
    Randomly modify a candidate allocation.

    A mutation transfers resources from one department
    to another, preserving the total resource count.
    """

    mutated = individual.copy()

    # Mutate with the configured probability.
    if random.random() < MUTATION_RATE:

        # Select two different departments.
        donor, receiver = random.sample(
            range(len(mutated)),
            2
        )

        # Transfer a random number of units.
        if mutated[donor] > 0:
            maximum_transfer = min(10, mutated[donor])

            amount = random.randint(
                1,
                maximum_transfer
            )

            mutated[donor] -= amount
            mutated[receiver] += amount

    return mutated


# ------------------------------------------------------------
# 8. RUN THE GENETIC ALGORITHM
# ------------------------------------------------------------

def run_genetic_algorithm():
    """
    Evolve candidate allocations over multiple generations.

    Each generation:
    1. Sort candidates by fitness.
    2. Preserve the best candidate (elitism).
    3. Select parents.
    4. Perform crossover.
    5. Apply mutation.
    6. Build the next generation.
    """

    # Create the initial population of random solutions.
    population = [
        create_individual()
        for _ in range(POPULATION_SIZE)
    ]

    best_overall = None
    best_overall_fitness = float("-inf")

    print("Starting Genetic Algorithm...")
    print(f"Population size: {POPULATION_SIZE}")
    print(f"Maximum generations: {MAX_GENERATIONS}")
    print(f"Total resources: {TOTAL_RESOURCES}\n")

    # Main evolutionary loop.
    for generation in range(1, MAX_GENERATIONS + 1):

        # Rank candidates from best to worst.
        population.sort(
            key=fitness,
            reverse=True
        )

        # Current generation's best solution.
        generation_best = population[0]
        generation_best_fitness = fitness(generation_best)

        # Preserve the best solution found so far.
        if generation_best_fitness > best_overall_fitness:
            best_overall = generation_best.copy()
            best_overall_fitness = generation_best_fitness

        # Print progress every 10 generations.
        if generation == 1 or generation % 10 == 0:
            print(
                f"Generation {generation:3d} | "
                f"Best fitness: {generation_best_fitness:.6f} | "
                f"Best allocation: {generation_best}"
            )

        # Elitism: retain the best candidate in the next generation.
        next_population = [
            population[0].copy()
        ]

        # Create the rest of the next generation.
        while len(next_population) < POPULATION_SIZE:

            # Choose two parents using tournament selection.
            parent1 = select_parent(population)
            parent2 = select_parent(population)

            # Combine the parents to produce a child.
            child = crossover(parent1, parent2)

            # Randomly introduce variation into the child.
            child = mutate(child)

            # Add the child to the next generation.
            next_population.append(child)

        # Replace the old population with the new one.
        population = next_population

    # --------------------------------------------------------
    # 9. DISPLAY THE FINAL SOLUTION
    # --------------------------------------------------------

    print("\n" + "=" * 45)
    print("GENETIC ALGORITHM COMPLETED")
    print("=" * 45)

    print(f"Best fitness: {best_overall_fitness:.6f}")
    print(f"Total allocated resources: {sum(best_overall)}")

    print("\nOptimized Resource Allocation:")

    for department, resources in zip(
        DEPARTMENTS,
        best_overall
    ):
        print(f"{department:20s}: {resources}")

    return best_overall


# ------------------------------------------------------------
# 10. PROGRAM ENTRY POINT
# ------------------------------------------------------------

if __name__ == "__main__":
    best_allocation = run_genetic_algorithm()
