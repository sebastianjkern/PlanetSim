# %% 

ITERATIONS = 1000000

WIDTH = 250
HEIGHT = 250

DT = 0.001
G = 6.67430 * 10 **-11

import random
import numpy as np
from vectors import Vector, Point
from functools import reduce
import matplotlib.pyplot as plt

def uniqueid():
    seed = random.getrandbits(32)
    while True:
       yield seed
       seed += 1

id_gen = uniqueid()

class Planet:
    def __init__(self, velo = Vector(0, 0, 0), mass = 100000, position = Point(0, 0, 0)):
        self.id = next(id_gen)
        self.velocity = velo
        self.mass = mass
        self.position = position

        self.history = []

objects = [
    Planet(position=Point(0, 0.2, 0), mass=100000, velo=Vector(2, 0, 0)),
    Planet(position=Point(0, 0.4, 0), mass=100000, velo=Vector(2.25, 0, 0)),
    Planet(position=Point(-0.25, 0.5, 0), mass=100000, velo=Vector(2, 0, 0)),
    Planet(position=Point(-0.5, 0.6, 0), mass=100000, velo=Vector(1.75, 0, 0)),
    Planet(position=Point(-1, 0.7, 0), mass=100000, velo=Vector(1.3, 0, 0)),
    ]
stationary_objects = [Planet(mass=20000000000, position=Point(0, 0)) for _ in range(1)]

resulting_forces = []

fig, ax = plt.subplots()
ax.set_title('Planet plot')

for iteration in range(ITERATIONS):
    if iteration % 1000 == 0:
        print(iteration, "/", ITERATIONS)

    resulting_forces = []

    for obj1 in objects:
        forces = []

        for obj2 in objects:
            if obj1.id == obj2.id:
                continue

            vec_between_objects = Vector.from_points(obj1.position, obj2.position)
            f = G * obj1.mass * obj2.mass / vec_between_objects.magnitude()**2
            direction = vec_between_objects

            vec_between_objects = vec_between_objects.multiply(1 / vec_between_objects.magnitude())
            vec_between_objects = vec_between_objects.multiply(f)

            forces.append(vec_between_objects)

        for obj2 in stationary_objects:
            vec_between_objects = Vector.from_points(obj1.position, obj2.position)
            f = G * obj1.mass * obj2.mass / vec_between_objects.magnitude()**2
            direction = vec_between_objects

            vec_between_objects = vec_between_objects.multiply(1 / vec_between_objects.magnitude())
            vec_between_objects = vec_between_objects.multiply(f)

            forces.append(vec_between_objects)

        resulting_forces.append(reduce(lambda f1, f2: f1.sum(f2), forces))

    for obj, force in zip(objects, resulting_forces):
        diff_impulse = force.multiply(DT)
        new_impulse = obj.velocity.multiply(obj.mass).sum(diff_impulse)
        new_velocity = new_impulse.multiply(1 / obj.mass)
        new_position = Point(obj.position.x + new_velocity.x*DT, obj.position.y + new_velocity.y*DT, obj.position.z + new_velocity.z*DT)

        obj.history.append(obj.position)

        obj.position = new_position
        obj.velocity = new_velocity

for obj in objects:
    x = [c.x for c in obj.history]
    y = [c.y for c in obj.history]
    ax.plot(x, y)
    ax.scatter(x[-1], y[-1], c="red")
    ax.scatter(x[0], y[0], c="green", s=4.0)

for stat_obj in stationary_objects:
    ax.scatter(stat_obj.position.x, stat_obj.position.y, c="green")

ax.set_ylim([-7.5, 7.5])
ax.set_xlim([-7.5, 7.5])

plt.show()

# %%
