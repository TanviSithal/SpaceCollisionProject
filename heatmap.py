import numpy as np


def generate_risk_heatmap(
    position_a,
    position_b,
    grid_size=30,
    area=20
):

    x = np.linspace(
        min(position_a[0], position_b[0]) - area,
        max(position_a[0], position_b[0]) + area,
        grid_size
    )

    y = np.linspace(
        min(position_a[1], position_b[1]) - area,
        max(position_a[1], position_b[1]) + area,
        grid_size
    )

    X, Y = np.meshgrid(x, y)

    # Risk centered between the two objects
    center_x = (
        position_a[0] + position_b[0]
    ) / 2

    center_y = (
        position_a[1] + position_b[1]
    ) / 2

    distance_squared = (
        (X - center_x) ** 2
        +
        (Y - center_y) ** 2
    )

    risk = np.exp(
        -distance_squared / 100
    )

    return X, Y, risk