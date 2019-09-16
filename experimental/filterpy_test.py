from filterpy.kalman import predict
import numpy as np

dt = 0.1

"""
implementica basic KF
"""
x = np.array([10.0, 4.5])
P = np.diag([500, 49])
F = np.array([[1, dt], [0, 1]]) # this state transition matrix will define our model - like velocity model etc

# Q is the process noise

x, P = predict(x=x, P=P, F=F, Q=0)
x, P = predict(x=x, P=P, F=F, Q=0)
print('x =', x)
