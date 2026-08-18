import numpy as np


def simulate_reentry(
    initial_altitude=400,
    decay_rate=2,
    duration=200
):

    time = np.linspace(
        0,
        duration,
        200
    )

    altitude = (
        initial_altitude
        -
        decay_rate * time
    )

    altitude = np.maximum(
        altitude,
        0
    )

    reentry_index = np.where(
        altitude <= 100
    )[0]

    if len(reentry_index) > 0:

        reentry_time = time[
            reentry_index[0]
        ]

    else:

        reentry_time = None

    return time, altitude, reentry_time