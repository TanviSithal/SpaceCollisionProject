import numpy as np
from energy import calculate_energy_cost


def simulate_manoeuvre(
    position_a,
    velocity_a,
    position_b,
    velocity_b,
    manoeuvre,
    duration=5,
    dt=0.1
):

    position_a = np.array(position_a, dtype=float)
    velocity_a = np.array(velocity_a, dtype=float)

    position_b = np.array(position_b, dtype=float)
    velocity_b = np.array(velocity_b, dtype=float)

    if manoeuvre == "Reduce speed":

        velocity_a = velocity_a * 0.6

    elif manoeuvre == "Increase speed":

        velocity_a = velocity_a * 1.4

    elif manoeuvre == "Move left":

        velocity_a = velocity_a + np.array(
            [0, 1.5, 0]
        )

    elif manoeuvre == "Move right":

        velocity_a = velocity_a + np.array(
            [0, -1.5, 0]
        )

    minimum_distance = float("inf")

    steps = int(duration / dt)

    for _ in range(steps):

        position_a = position_a + velocity_a * dt
        position_b = position_b + velocity_b * dt

        distance = np.linalg.norm(
            position_b - position_a
        )

        minimum_distance = min(
            minimum_distance,
            distance
        )

    energy = calculate_energy_cost(
        np.linalg.norm(velocity_a),
        duration
    )

    return minimum_distance, energy


def find_best_manoeuvre(
    position_a,
    velocity_a,
    position_b,
    velocity_b
):

    manoeuvres = [
        "No action",
        "Reduce speed",
        "Increase speed",
        "Move left",
        "Move right"
    ]

    results = []

    for manoeuvre in manoeuvres:

        minimum_distance, energy = simulate_manoeuvre(
            position_a,
            velocity_a,
            position_b,
            velocity_b,
            manoeuvre
        )

        results.append({
            "manoeuvre": manoeuvre,
            "minimum_distance": minimum_distance,
            "energy": energy
        })

    # Prefer manoeuvres that maintain safe separation
    safe_results = [
        r for r in results
        if r["minimum_distance"] >= 5
    ]

    if safe_results:

        best = min(
            safe_results,
            key=lambda x: x["energy"]
        )

    else:

        best = max(
            results,
            key=lambda x: x["minimum_distance"]
        )

    return best, results