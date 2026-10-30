
"""
aco_scheduling.py

Project: Intelligent Healthcare Support System
Technique: Ant Colony Optimization (ACO)

Purpose:
    Generate a patient appointment schedule using ACO.

How it works:
    1. Each ant constructs a possible appointment order.
    2. Pheromone values represent how desirable scheduling choices are.
    3. A heuristic favours urgent patients and shorter waiting times.
    4. Better schedules deposit more pheromone.
    5. Pheromone evaporates to reduce the influence of old choices.
    6. The algorithm repeats and keeps the best schedule found.

This is a simulated educational example. It does not use real patient data.
"""

import random
import math


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

random.seed(42)

NUM_ANTS = 20
NUM_ITERATIONS = 50

# Pheromone influence
ALPHA = 1.0

# Heuristic influence
BETA = 2.0

# Fraction of pheromone evaporated after each iteration
EVAPORATION_RATE = 0.3

# Pheromone added by a completed schedule
PHEROMONE_CONSTANT = 100.0

# Simulated clinic starts at 9:00 AM.
CLINIC_START_MINUTES = 9 * 60

# Each appointment takes 20 minutes.
APPOINTMENT_DURATION = 20


# ---------------------------------------------------------
# 2. SAMPLE PATIENTS
# ---------------------------------------------------------

# Priority:
# 3 = High
# 2 = Medium
# 1 = Low
#
# Arrival time is represented as minutes after midnight.
# These patients and times are fictional.

PATIENTS = [
    {"id": "P1", "priority": 3, "arrival": 9 * 60},
    {"id": "P2", "priority": 1, "arrival": 9 * 60 + 10},
    {"id": "P3", "priority": 2, "arrival": 9 * 60 + 5},
    {"id": "P4", "priority": 3, "arrival": 9 * 60 + 15},
    {"id": "P5", "priority": 1, "arrival": 9 * 60 + 20},
    {"id": "P6", "priority": 2, "arrival": 9 * 60 + 10},
]


# ---------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------

def format_time(minutes):
    """Convert minutes after midnight into a readable 12-hour time."""
    hour = minutes // 60
    minute = minutes % 60

    suffix = "AM" if hour < 12 else "PM"
    display_hour = hour % 12

    if display_hour == 0:
        display_hour = 12

    return f"{display_hour}:{minute:02d} {suffix}"


def calculate_schedule_cost(schedule):
    """
    Calculate the cost of a schedule.

    Lower cost is better.

    Cost includes:
        - Waiting time for patients.
        - A penalty for scheduling high-priority patients later.

    This is a simplified demonstration, not a validated hospital policy.
    """
    total_waiting_time = 0
    priority_penalty = 0

    for position, patient_index in enumerate(schedule):
        patient = PATIENTS[patient_index]

        appointment_start = (
            CLINIC_START_MINUTES
            + position * APPOINTMENT_DURATION
        )

        # A patient cannot be seen before arriving.
        waiting_time = max(
            0,
            appointment_start - patient["arrival"]
        )

        total_waiting_time += waiting_time

        # Penalize delaying higher-priority patients.
        # A high-priority patient receives a larger penalty.
        priority_penalty += (
            (patient["priority"] - 1) * position * 15
        )

    return total_waiting_time + priority_penalty


def calculate_heuristic(patient_index, current_position):
    """
    Estimate how desirable it is to schedule a patient next.

    Higher heuristic value means a more attractive choice.

    The heuristic favours:
        - Higher priority.
        - Patients who have already arrived.
    """
    patient = PATIENTS[patient_index]

    appointment_start = (
        CLINIC_START_MINUTES
        + current_position * APPOINTMENT_DURATION
    )

    waiting_time = max(
        0,
        appointment_start - patient["arrival"]
    )

    # Higher priority increases desirability.
    priority_score = patient["priority"]

    # A small positive value avoids division by zero.
    waiting_score = 1.0 / (1.0 + waiting_time)

    # Give priority more weight while still considering waiting.
    return priority_score * (1.0 + waiting_score)


# ---------------------------------------------------------
# 4. CONSTRUCT A SCHEDULE USING AN ANT
# ---------------------------------------------------------

def construct_schedule(pheromone):
    """
    One ant builds a complete appointment order.

    At each position, the ant selects one of the patients
    who have not yet been scheduled.

    Selection probability depends on:
        pheromone ** ALPHA
        multiplied by
        heuristic ** BETA
    """
    remaining_patients = list(range(len(PATIENTS)))
    schedule = []

    while remaining_patients:
        current_position = len(schedule)

        # Calculate a selection weight for every available patient.
        weights = []

        for patient_index in remaining_patients:
            pheromone_value = pheromone[current_position][patient_index]

            heuristic_value = calculate_heuristic(
                patient_index,
                current_position
            )

            weight = (
                pheromone_value ** ALPHA
                * heuristic_value ** BETA
            )

            weights.append(weight)

        # Convert weights into probabilities.
        total_weight = sum(weights)

        if total_weight == 0:
            probabilities = [
                1.0 / len(remaining_patients)
            ] * len(remaining_patients)
        else:
            probabilities = [
                weight / total_weight
                for weight in weights
            ]

        # Randomly select a patient using the calculated probabilities.
        selected_patient = random.choices(
            remaining_patients,
            weights=probabilities,
            k=1
        )[0]

        schedule.append(selected_patient)
        remaining_patients.remove(selected_patient)

    return schedule


# ---------------------------------------------------------
# 5. UPDATE PHEROMONE
# ---------------------------------------------------------

def update_pheromone(pheromone, schedules_and_costs):
    """
    Update pheromone values.

    Step 1: Evaporate existing pheromone.
    Step 2: Let every ant deposit pheromone along its schedule.

    Better schedules have lower cost and deposit more pheromone.
    """

    # Step 1: Evaporation
    for position in range(len(pheromone)):
        for patient_index in range(len(PATIENTS)):
            pheromone[position][patient_index] *= (
                1.0 - EVAPORATION_RATE
            )

            # Keep pheromone above zero.
            pheromone[position][patient_index] = max(
                pheromone[position][patient_index],
                0.0001
            )

    # Step 2: Pheromone deposit
    for schedule, cost in schedules_and_costs:

        # Lower cost means a larger pheromone deposit.
        deposit = PHEROMONE_CONSTANT / (1.0 + cost)

        for position, patient_index in enumerate(schedule):
            pheromone[position][patient_index] += deposit


# ---------------------------------------------------------
# 6. MAIN ACO ALGORITHM
# ---------------------------------------------------------

def run_aco_scheduling():
    """Run ACO and return the best schedule found."""

    number_of_patients = len(PATIENTS)

    # Initially, all possible position-patient assignments
    # receive the same pheromone value.
    pheromone = [
        [1.0 for _ in range(number_of_patients)]
        for _ in range(number_of_patients)
    ]

    best_schedule = None
    best_cost = math.inf

    print("\n===== Ant Colony Optimization: Patient Scheduling =====")
    print(f"Number of patients: {number_of_patients}")
    print(f"Ants per iteration: {NUM_ANTS}")
    print(f"Iterations: {NUM_ITERATIONS}")

    for iteration in range(NUM_ITERATIONS):
        schedules_and_costs = []

        # Every ant constructs a candidate appointment order.
        for _ in range(NUM_ANTS):
            schedule = construct_schedule(pheromone)
            cost = calculate_schedule_cost(schedule)

            schedules_and_costs.append((schedule, cost))

            if cost < best_cost:
                best_cost = cost
                best_schedule = schedule.copy()

        # Update pheromone based on the schedules found.
        update_pheromone(pheromone, schedules_and_costs)

        # Print progress occasionally.
        if (iteration + 1) % 10 == 0:
            print(
                f"Iteration {iteration + 1:02d}: "
                f"Best cost = {best_cost:.2f}"
            )

    # Display the best appointment order.
    print("\n===== Best Schedule Found =====")

    for position, patient_index in enumerate(best_schedule):
        patient = PATIENTS[patient_index]

        appointment_start = (
            CLINIC_START_MINUTES
            + position * APPOINTMENT_DURATION
        )

        waiting_time = max(
            0,
            appointment_start - patient["arrival"]
        )

        print(
            f"{format_time(appointment_start)} | "
            f"Patient: {patient['id']} | "
            f"Priority: {patient['priority']} | "
            f"Arrival: {format_time(patient['arrival'])} | "
            f"Waiting: {waiting_time} minutes"
        )

    print(f"\nFinal schedule cost: {best_cost:.2f}")
    print("\nNote: This is a simulated academic scheduling example.")
    print("It must not replace clinical triage or hospital scheduling policy.")

    return best_schedule, best_cost


# ---------------------------------------------------------
# 7. RUN THE PROGRAM
# ---------------------------------------------------------

if __name__ == "__main__":
    run_aco_scheduling()
