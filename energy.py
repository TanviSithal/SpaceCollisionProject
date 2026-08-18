def calculate_energy_cost(speed, duration, mass=1.0):

    # Simplified prototype energy model
    energy = mass * speed ** 2 * duration

    return energy


def compare_manoeuvres(
    current_speed,
    duration=5
):

    normal = calculate_energy_cost(
        current_speed,
        duration
    )

    reduced_speed = current_speed * 0.65

    reduce_speed_energy = calculate_energy_cost(
        reduced_speed,
        duration
    )

    turn_energy = normal * 0.85

    return {
        "Normal movement": normal,
        "Reduce speed": reduce_speed_energy,
        "Change direction": turn_energy
    }