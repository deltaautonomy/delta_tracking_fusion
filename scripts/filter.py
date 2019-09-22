#########################################################################
#####                         Delta Autonomy                        #####
#####                   Written By: Karmesh Yadav                   #####
#####                         Date: 09/09/19                        #####
#########################################################################

# Python 2/3 compatibility
from __future__ import print_function, absolute_import, division
import numpy as np
import time
import math
import pdb

# A class for Kalman Filter which can take more than 1 sensor as input
class KalmanFilterRADARCamera():
    def __init__(self, vehicle_id, state_dim, camera_dim, radar_dim, control_dim,
        dt, first_call_time, motion_model='velocity', verbose=False):
        """
        Initializes the kalman filter class 
        @param: vehicle_id - the id given to the vehicle track
        @param: state_dim - number of states
        @param: camera_dim - number of observations returned by camera
        @param: radar_dim - number of observations returned by radar
        @param: control_dim - number of control variables
        @param: dt - timestep at which the vehicle state should update
        """
        # KalmanFilter.__init__(state_dim, sensor_dim, 0)
        self.dt = dt
        self.state_dim = state_dim
        self.camera_dim = camera_dim
        self.radar_dim = radar_dim
        self.control_dim = control_dim
        self.motion_model = motion_model
        self.verbose = verbose

        # Filter State estimate [x, y, vx, vy]
        self.x = np.zeros((state_dim, 1))
        # State Transition matrix
        self.F = np.eye(state_dim)
        # Control Transition Matix
        self.B = np.zeros((state_dim, control_dim))
        # State Covariance matrix
        self.P = np.eye(state_dim)
        # Process Noise
        self.Q = np.eye(state_dim)
        # Camera Measurement Model [x, y]
        self.camera = SensorMeasurementModel(state_dim, camera_dim)
        # Radar Measurement Model [x, y, vx, vy]
        self.radar = SensorMeasurementModel(state_dim, radar_dim)
        # self.radar = RadarMeasurementModel(state_dim, radar_dim)
        self.initialize_filter(sigma_acc=8.8)

        self.id = vehicle_id
        self.time_since_update = 0
        self.age = 0
        self.last_call_time = first_call_time

    def initialize_filter(self, sigma_acc=8.8):
        """
        Internal function to initialize the filter

        @param: sigma_acc - a constant to specify the process noise
        """

        T = self.dt

        if self.motion_model == 'velocity':
            assert (self.F.shape == (4, 4)), "The state dimension is incorrect"
            self.x +=0.000001
            self.F, self.P, self.B, G = self.constant_velocity_motion_model(self.dt)

        else:
            assert (self.F.shape == (6, 6)), "The state dimension is incorrect"
            self.x += 1
            self.F, self.P, self.B, G = self.constant_acceleration_motion_model(self.dt)

        self.Q = np.matmul(G.T, G) * sigma_acc**2

        H_camera = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0]])
        self.camera.set_H(H_camera)

        R_camera = np.eye(2)
        self.camera.set_R(R_camera)

        H_radar = np.eye(4)
        self.radar.set_H(H_radar)
        
        R_radar = np.eye(4)
        self.radar.set_R(R_radar)
        
        if self.verbose:
            print("==========================Predict Function==========================")
            print("_____State_____ \n", self.x )
            print("_____covariance_____\n", self.P)
            print("_____State Transition_____\n", self.F)
            print("_____Process Noise_____\n", self.Q)
            print("_____Control Transition_____\n", self.B)

    def predict(self, time_step, u=None):
        """
        The predict function of the EKF

        @param: time_step - the timestep to update the equations by, usually equal to self.dt
        @param: u - the control signal to update the state (not use currently)
        """
        self.age += 1
        self.time_since_update += 1 

        if self.motion_model == 'velocity':
            self.F, _, _, _ = self.constant_velocity_motion_model(time_step)
        else:
            self.F, _, _, _ = self.constant_acceleration_motion_model(time_step)

        if u is None:
            self.x = np.matmul(self.F,self.x)
            self.P = np.matmul(self.F, np.matmul(self.P, self.F.T)) + self.Q
        else:
            assert (u.shape == (self.control_dim, 1)), "Shape of the control input is incorrect"
            self.x = np.matmul(self.F,self.x) + np.matmul(self.B, u)
            # TODO: Write the process noise equation correctly for the case when control input is given 
            self.P = np.matmul(self.F, np.matmul(self.P, self.F.T)) + self.Q # THe Q part of equation will be different

        if self.verbose:
            print("==========================Predict Function==========================")
            print("_____State_____ \n", self.x )
            print("_____covariance_____\n", self.P)

    def constant_velocity_motion_model(self, time_step):
        """
        Fuction to calculate the matrices of the filter. Uses constant velocity model

        @param: time_step - the time step to calculate the matrices at
        """
        T = time_step
        F = np.array([[1, T, 0, 0], 
                      [0, 1, 0, 0], 
                      [0, 0, 1, T], 
                      [0, 0, 0, 1]])
        
        P = np.array([[1000.0,      0,      0,      0], 
                      [     0, 1000.0,      0,      0],
                      [     0,      0, 1000.0,      0],
                      [     0,      0,      0, 1000.0]])

        B = np.array([[0, 1, 0, 0],
                      [0, 0, 0, 1]])

        G = np.array([0.5*T**2, T, 0.5*T**2, T])

        return F, P, B, G

    def constant_acceleration_motion_model(self, time_step):
        """
        Fuction to calculate the matrices of the filter. Uses constant acceleration model

        @param: time_step - the time step to calculate the matrices at
        """
        T = time_step
        F = np.array([[1, T, 0.5*T**2, 0, 0,        0], 
                      [0, 1,        T, 0, 0,        0],
                      [0, 0,        1, 0, 0,        0],
                      [0, 0,        0, 1, T, 0.5*T**2],
                      [0, 0,        0, 0, 1,        T], 
                      [0, 0,        0, 0, 0,        1]])

        P = np.array([[1000.0,      0,      0,      0,      0,      0], 
                      [     0, 1000.0,      0,      0,      0,      0],
                      [     0,      0, 1000.0,      0,      0,      0],
                      [     0,      0,      0, 1000.0,      0,      0],
                      [     0,      0,      0,      0, 1000.0,      0],
                      [     0,      0,      0,      0,      0, 1000.0]])
    
        B = np.array([[0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1]])

        G = np.array([0.5*T**2, T, 1, 0.5*T**2, T, 1])

        return F, P, B, G

    def update(self, z, R_current=None, sensor='radar'):
        """
        The update step for Kalman Filter. It assumes that R=None means that the value
        for R is constant and set during initialization.
        """
        self.time_since_update = 0

        if sensor == 'radar':
            assert (z.shape == (self.radar_dim,1)), "The data input dimension is incorrect"
            H = self.radar.H
            if R_current is not None:
                self.radar.set_R(R_current)
                R = R_current
            else:
                R = self.radar.R
            Y = z - np.matmul(H, self.x)
            
        else:
            assert (z.shape == (self.camera_dim,1)), "The data input dimension is incorrect"
            H = self.camera.H
            if R_current is not None:
                self.camera.set_R(R_current)
                R = R_current
            else:
                R = self.camera.R
            Y = z - np.matmul(H, self.x)
        
        covariance_sum = np.matmul(np.matmul(H, self.P), H.T) + R
        K = np.matmul(np.matmul(self.P, H.T), np.linalg.pinv(covariance_sum))

        self.x = self.x + np.matmul(K, Y)
        KH = np.matmul(K, H)
        self.P = np.matmul((np.eye(KH.shape[0]) - KH), self.P)
        
        if self.verbose:
            print("==========================Update Function [{}] ==========================".format(sensor))
            print("_____State_____ \n", self.x )
            print("_____covariance_____\n", self.P)
            print("_____Error_____\n", Y)
            print("_____Kalman_Gain_____\n", K)
            
    def predict_step(self, current_time):
        """
        Carries out both predict and update step
        @param: current_time - the time since the last update
        """
        time_step = self.last_call_time - current_time
        while(time_step > self.dt):
            time_step -= self.dt
            self.predict(self.dt)
        if (time_step%self.dt > 0):
            self.predict(time_step%self.dt)
        self.last_call_time = current_time
        return self.x
    
    def update_step(self, z_camera=None, z_radar=None, R_camera=None, R_radar=None):
        if z_camera is not None:
            self.update(z_camera, R_camera, sensor='camera')
        if z_radar is not None:
            self.update(z_radar, R_radar, sensor='radar')
        return self.x

    def get_state(self):
        return self.x


class SensorMeasurementModel():
    def __init__(self, state_dim, sensor_dim):
        self.state_dim = state_dim
        self.sensor_dim = sensor_dim

        # The measurement 
        self.z = np.zeros((sensor_dim, 1))

        # Time of measurement
        self.time = 0

        # measurement uncertainity
        self.R = np.zeros((sensor_dim, sensor_dim))

        # measurement function. Maps measurement values to state
        self.H = np.zeros((sensor_dim, state_dim))

    def set_H(self, H):
        """
        The function works for simple Kalman Filter
        Need to linearize the non linear equations for EKF
        """
        assert (H.shape == self.H.shape), "Shape of the H matrix is incorrect"
        self.H = H

    def set_R(self, R):
        assert (R.shape == self.R.shape), "Shape of the R matrix is incorrect"
        self.R = R

    def get_H(self, state):
        """
        Gets the linearized H matrix for EKF
        """
        pass

    def measurement_function(self, state):
        """
        Define a measurement function for EKF
        Note: Not required for Kalman Filter
        """
        pass

    def set_z(self, z, t, R=None):
        assert (z.shape == self.z.shape), "Shape of the Z matrix is incorrect"
        self.z = z
        self.time = t

        if R is not None:
            self.set_R(R)


class RadarMeasurementModel(SensorMeasurementModel):
    def __init__(self, state_dim, sensor_dim):
        SensorMeasurementModel.__init__(self, state_dim, sensor_dim)

    def measurement_function(self, state):
        px = state[0][0]
        py = state[2][0]
        vx = state[1][0]
        vy = state[3][0]
        r = math.sqrt(px**2 + py**2)
        theta = math.atan2(py, px)
        if abs(r) > 0.000001:
            r_dot = (px*vx + py*vy)/r
        else:
            r_dot = 0
        return np.array([[r], [theta], [r_dot]])

    def get_H(self, state):
        px = state[0][0]
        py = state[2][0]
        vx = state[1][0]
        vy = state[3][0]
        c1 = (px**2 + py**2)
        c2 = vx*py - px*vy

        if math.sqrt(c1) < 0.000001:
            c1 = 0.000001

        H = np.array([[      px/math.sqrt(c1),                0,        py/math.sqrt(c1),                0],  ##this matrix is definitely wrong
                      [                -py/c1,                0,                   px/c1,                0],
                      [py*c2/math.sqrt(c1)**3, px/math.sqrt(c1), -px*c2/math.sqrt(c1)**3, py/math.sqrt(c1)]])
        return H


if __name__ == "__main__":
    KF = KalmanFilterRADARCamera(vehicle_id=1,
                                 state_dim=4, 
                                 camera_dim=2, 
                                 radar_dim=4, 
                                 control_dim=0, 
                                 dt=0.1, 
                                 first_call_time=0)

    R = np.array([[10,  0,  0,  0],
                  [ 0, 10,  0,  0], 
                  [ 0,  0, 10,  0],
                  [ 0,  0,  0, 10]])

    KF.predict_step(1.05)
    KF.update_step(z_camera=np.array([[1],[3]]), z_radar=np.array([[3.1],[1.3],[1.5],[1.5]]), R_radar=R)
