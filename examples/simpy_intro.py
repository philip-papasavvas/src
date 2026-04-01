""""
Created on 31 Jan 2021
Introduction to SimPy python package
https://simpy.readthedocs.io/en/latest/simpy_intro/basic_concepts.html
"""

'''
INTRO
discrete event simulation library modelling behaviour of active
components (vehicles, customers, messages), modelled with processes.
Processes live in an environment, which interact with the environment
and each other via events.

processes are described by python generators - during their lifetime,
they create events and yield them to wait for them to be triggered

when a process yields an event, the process gets suspended. this is
resumed on SimPy when the event occurs (the event is triggered). 
Multiple processes can wait for the same event. SimPy resumes them
in the same order in which they yielded the event. 

Timeout is an event type triggered after a certain amount of simulated time
has passed. It allows a process to sleep (or hold state) for a given time.
'''

# let's start our first process, a CAR process, which will alternatively
# drive and park for a while. when it starts driving/parking, it
# will print the current simulation time


import simpy


def car(environment):
    while True:
        print('Start parking at %d' % environment.now)
        parking_duration = 5
        yield environment.timeout(parking_duration)

        print('Start driving at %d' % environment.now)
        trip_duration = 2
        yield environment.timeout(trip_duration)


env = simpy.Environment()  # create instance of the environment
env.process(car(env))  # pass the instance to the car function
env.run(until=15)

# Start parking at 0
# Start driving at 5
# Start parking at 7
# Start driving at 12
# Start parking at 14

# Process interaction
'''
the Process instance returns by Environment.process() can be utilised
for process interactions -most common examples being waiting for another
process to finish, and interrupting another process whilst waiting for 
an event

WAITING FOR A PROCESS
now we imagine the car to be electric, which takes a while to charge,
so we write an addition charge() process (as a class method).
So, the new class will have two process methods: run() and charge()
'''


class Car(object):
    def __init__(self, env):
        self.env = env
        # Start the run process everytime an instance is created.
        self.action = env.process(self.run())

    def run(self):
        while True:
            print('Start parking and charging at %d' % self.env.now)
            charge_duration = 5
            # We yield the process that process() returns
            # to wait for it to finish
            yield self.env.process(self.charge(charge_duration))

            # The charge process has finished and
            # we can start driving again.
            print('Start driving at %d' % self.env.now)
            trip_duration = 2
            yield self.env.timeout(trip_duration)

    def charge(self, duration):
        yield self.env.timeout(duration)


# start the environment again
env_new = simpy.Environment()
car = Car(env_new)
env_new.run(until=15)

# Start parking and charging at 0
# Start driving at 5
# Start parking and charging at 7
# Start driving at 12
# Start parking and charging at 14

'''
INTERUPTING ANOTHER PROCESS
You don't want to wait until the EV is fully charged but instead 
want to interrupt the charging process and drive instead
Can interrupt using the interrupt method
'''


def driver(env, car):
    yield env.timeout(3)
    car.action.interrupt()


class Car(object):
    def __init__(self, env):
        self.env = env
        self.action = env.process(self.run())

    def run(self):
        while True:
            print('Start parking and charging at %d' % self.env.now)
            charge_duration = 5
            # We may get interrupted while charging the battery
            try:
                yield self.env.process(self.charge(charge_duration))
            except simpy.Interrupt:
                # When we received an interrupt, we stop charging and
                # switch to the "driving" state
                print('Was interrupted. Hope, the battery is full enough ...')

            print('Start driving at %d' % self.env.now)
            trip_duration = 2
            yield self.env.timeout(trip_duration)

    def charge(self, duration):
        yield self.env.timeout(duration)


env_interrupt = simpy.Environment()
car = Car(env_interrupt)
env_interrupt.process(driver(env_interrupt, car))
env_interrupt.run(until=20)