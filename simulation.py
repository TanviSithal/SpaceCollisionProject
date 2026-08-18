import numpy as np


class VirtualCar:
    def __init__(self, name, position, velocity, direction=0):
        self.name = name
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.direction = direction

    def update(self, dt=0.1):
        self.position = self.position + self.velocity * dt

    def get_true_state(self):
        return {
            "x": self.position[0],
            "y": self.position[1],
            "z": self.position[2],
            "vx": self.velocity[0],
            "vy": self.velocity[1],
            "vz": self.velocity[2],
        }

    def get_noisy_measurement(self, noise=0.3):
        noisy_position = self.position + np.random.normal(
            0, noise, 3
        )

        return {
            "x": noisy_position[0],
            "y": noisy_position[1],
            "z": noisy_position[2]
        }


def create_cars():

    car_a = VirtualCar(
        "Object A",
        position=[0, 0, 2],
        velocity=[1.5, 0.1, 0]
    )

    car_b = VirtualCar(
        "Object B",
        position=[30, 3, 2],
        velocity=[-1.2, -0.05, 0]
    )

    return car_a, car_b