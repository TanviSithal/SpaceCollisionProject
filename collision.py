import numpy as np


def calculate_distance(position_a, position_b):

    position_a = np.array(position_a)
    position_b = np.array(position_b)

    return np.linalg.norm(position_b - position_a)


def calculate_relative_velocity(velocity_a, velocity_b):

    velocity_a = np.array(velocity_a)
    velocity_b = np.array(velocity_b)

    return velocity_b - velocity_a


def calculate_closest_approach(
    position_a,
    velocity_a,
    position_b,
    velocity_b
):

    position_a = np.array(position_a, dtype=float)
    position_b = np.array(position_b, dtype=float)

    velocity_a = np.array(velocity_a, dtype=float)
    velocity_b = np.array(velocity_b, dtype=float)

    relative_position = position_b - position_a
    relative_velocity = velocity_b - velocity_a

    velocity_squared = np.dot(
        relative_velocity,
        relative_velocity
    )

    if velocity_squared == 0:
        return 0, np.linalg.norm(relative_position)

    time_to_cpa = -np.dot(
        relative_position,
        relative_velocity
    ) / velocity_squared

    if time_to_cpa < 0:
        time_to_cpa = 0

    closest_position = (
        relative_position +
        relative_velocity * time_to_cpa
    )

    minimum_distance = np.linalg.norm(
        closest_position
    )

    return time_to_cpa, minimum_distance


def calculate_risk(minimum_distance, time_to_cpa):

    if minimum_distance < 2 and time_to_cpa < 5:

        return "CRITICAL", 95

    elif minimum_distance < 5 and time_to_cpa < 10:

        return "HIGH", 80

    elif minimum_distance < 10:

        return "MEDIUM", 45

    else:

        return "LOW", 10


def analyze_collision(
    position_a,
    velocity_a,
    position_b,
    velocity_b
):

    distance = calculate_distance(
        position_a,
        position_b
    )

    relative_velocity = calculate_relative_velocity(
        velocity_a,
        velocity_b
    )

    relative_speed = np.linalg.norm(
        relative_velocity
    )

    time_to_cpa, minimum_distance = calculate_closest_approach(
        position_a,
        velocity_a,
        position_b,
        velocity_b
    )

    risk, probability = calculate_risk(
        minimum_distance,
        time_to_cpa
    )

    return {
        "distance": distance,
        "relative_velocity": relative_velocity,
        "relative_speed": relative_speed,
        "time_to_cpa": time_to_cpa,
        "minimum_distance": minimum_distance,
        "risk": risk,
        "probability": probability
    }