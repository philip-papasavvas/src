# 1
def f_to_c(f_temp):
    c_temp = (f_temp - 32) * 5 / 9

    return c_temp

# print(c_temp) #use print to get the result

# 2
# f100_in_celsius = f_to_c(100)
# returns 37.78

# solution printed:  print("100F is " + str(f100_in_celsius) + " in Celsius")

# 3
def c_to_f(c_temp):
    f_temp = c_temp * (9 / 5) + 32
    # print(f_temp) #used print to get the result. call the result using: return f_temp


# solution printed:  print("0C is " + str(c0_in_fahrenheit) + " in Fahrenheit")


# 4
# c0_in_farenheit = c_to_f(0)
# returns 32


# 5
def get_force(mass, acceleration):
    force = mass * acceleration
    return force  # give the result which can be obtained by using str(get_force(argument1,argument2))


# 6
train_mass = 22680
train_acceleration = 10

train_force = get_force(train_mass, train_acceleration)
# train_force

# 7
c = 3 * 10 ** 8


def get_energy(mass, c):
    energy = mass * (c ** 2)
    return energy


# 8
bomb_mass = 1
get_energy(bomb_mass, c)


# 9
def get_work(mass, acceleration, distance):
    force = get_force(mass, acceleration)
    return force * distance
    # print(force*distance)


# get_work(15,10,8060)

# 10
train_mass = 22000
train_acceleration = 10
train_distance = 1800

train_work = get_work(train_mass, train_acceleration, train_distance)
train_work

# 11
print("The GE train does " + str(train_work) + " Joules of work over " + str(train_distance) + " meters.")

train_work
# train_distance = 100

# bomb_mass = 1