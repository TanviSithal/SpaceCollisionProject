import numpy as np


# ============================================================
# DISTANCE
# ============================================================

def calculate_distance(position_a, position_b):

    position_a = np.asarray(position_a, dtype=float)
    position_b = np.asarray(position_b, dtype=float)

    return float(
        np.linalg.norm(position_b - position_a)
    )


# ============================================================
# RELATIVE VELOCITY
# ============================================================

def calculate_relative_velocity(velocity_a, velocity_b):

    velocity_a = np.asarray(velocity_a, dtype=float)
    velocity_b = np.asarray(velocity_b, dtype=float)

    return velocity_b - velocity_a


# ============================================================
# CLOSEST APPROACH
# ============================================================

def calculate_closest_approach(
    position_a,
    velocity_a,
    position_b,
    velocity_b
):

    position_a = np.asarray(position_a, dtype=float)
    position_b = np.asarray(position_b, dtype=float)

    velocity_a = np.asarray(velocity_a, dtype=float)
    velocity_b = np.asarray(velocity_b, dtype=float)

    relative_position = position_b - position_a
    relative_velocity = velocity_b - velocity_a

    velocity_squared = np.dot(
        relative_velocity,
        relative_velocity
    )

    if velocity_squared == 0:

        return (
            0.0,
            float(np.linalg.norm(relative_position))
        )

    time_to_cpa = (
        -np.dot(
            relative_position,
            relative_velocity
        )
        / velocity_squared
    )

    if time_to_cpa < 0:
        time_to_cpa = 0.0

    closest_position = (
        relative_position
        + relative_velocity * time_to_cpa
    )

    minimum_distance = float(
        np.linalg.norm(closest_position)
    )

    return (
        float(time_to_cpa),
        minimum_distance
    )


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def calculate_risk(
    minimum_distance,
    time_to_cpa
):

    if minimum_distance < 2 and time_to_cpa < 5:

        return "CRITICAL", 95

    elif minimum_distance < 5 and time_to_cpa < 10:

        return "HIGH", 80

    elif minimum_distance < 10:

        return "MEDIUM", 45

    else:

        return "LOW", 10


# ============================================================
# COMPLETE COLLISION ANALYSIS
# ============================================================

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

    relative_speed = float(
        np.linalg.norm(relative_velocity)
    )

    time_to_cpa, minimum_distance = calculate_closest_approach(
        position_a,
        velocity_a,
        position_b,
        velocity_b
    )

    risk, risk_score = calculate_risk(
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

        "probability": risk_score
    }