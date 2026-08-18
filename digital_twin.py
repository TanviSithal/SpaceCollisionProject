import numpy as np


def generate_trajectory(
    position,
    velocity,
    duration=10,
    points=100
):

    position = np.array(position, dtype=float)
    velocity = np.array(velocity, dtype=float)

    time = np.linspace(
        0,
        duration,
        points
    )

    trajectory = (
        position[:, None]
        +
        velocity[:, None] * time
    )

    return time, trajectory


def create_digital_twin(
    position_a,
    velocity_a,
    position_b,
    velocity_b
):

    time_a, trajectory_a = generate_trajectory(
        position_a,
        velocity_a
    )

    time_b, trajectory_b = generate_trajectory(
        position_b,
        velocity_b
    )

    return {
        "time": time_a,
        "trajectory_a": trajectory_a,
        "trajectory_b": trajectory_b
    }