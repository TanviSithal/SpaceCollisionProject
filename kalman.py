import numpy as np


class SimpleKalmanFilter:

    def __init__(self, initial_position, process_noise=0.01,
                 measurement_noise=0.3):

        self.estimate = np.array(initial_position, dtype=float)

        self.error = np.ones(3)

        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def update(self, measurement):

        measurement = np.array(measurement, dtype=float)

        # Prediction
        predicted = self.estimate

        predicted_error = (
            self.error + self.process_noise
        )

        # Kalman gain
        kalman_gain = predicted_error / (
            predicted_error + self.measurement_noise
        )

        # Update
        self.estimate = predicted + kalman_gain * (
            measurement - predicted
        )

        self.error = (
            (1 - kalman_gain) * predicted_error
        )

        return self.estimate


def create_filters():

    filter_a = SimpleKalmanFilter([0, 0, 2])
    filter_b = SimpleKalmanFilter([30, 3, 2])

    return filter_a, filter_b