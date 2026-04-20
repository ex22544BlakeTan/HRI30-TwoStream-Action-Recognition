import math
import numpy as np

class chameleon:
	def __init__(self):
		"""
		Initialises the plant model with its physical properties and initial states.
		
		Parameters:
		m    (float): mass of the tongue
		q    (array): joint angles
		qdot (array): joint velocities
		tau  (array): joint generalised forces
		g    (float): acceleration due to gravity
		"""
		# Plant properties
		self.dimq         = 2 # dimensionality of joint space (tongue angle and length)
		self.dimr         = 2 # dimensionality of task space (tongue tip coordinates)
		self.m            = 0.1 # mass of tongue
		
		# States
		self.q            = np.array([0,0]).T # initial joint configuration
		self.qdot         = np.array([0,0]).T # initial joint velocities
		self.tau          = np.array([0,0]).T # initial joint generalised forces
		
		# Environmental properties
		self.g = 9.81 # acceleration due to gravity
		
	def set_m(self, m):
		""" Sets the plant mass to a non-default value """
		self.m = m

	def forward_kinematics(self, q):
		"""
		Computes the end effector (tongue tip) position using forward kinematics.
		
		The forward kinematics equations are:
		
		r0 = q1 * cos(q0) 
		r1 = q1 * sin(q0) 
		
		Arguments:
		q (array): joint angles
		Returns:
		r (array): tongue tip position
		"""
		r = np.full((self.dimr,),math.nan) # initialise r
		r[0] = q[1] * math.cos(q[0]) 
		r[1] = q[1] * math.sin(q[0]) 
		return r
	    
	def get_MCG(self, q, qdot):
		"""
		Calculates the inertial matrix, the Coriolis and centrifugal effects vector, and the gravity vector using the given robot state and kinematic parameters.
		Arguments:
		    q (numpy array): joint angles
		    q_dot (numpy array): joint angular velocity
		Returns:
		    M (numpy array): the inertial matrix
		    c (numpy array): the Coriolis and centrifugal effects vector
		    G (numpy array): the gravity vector
		"""
		m = self.m # mass of tongue
		g = self.g # 
		
		M = np.array([[m*q[1]**2, 0], 
		              [        0, m]])
		
		C = np.array([[               0, 2*m*q[1]*qdot[0]], 
		              [ -m*q[1]*qdot[0],               0]])
		
		G = np.array([m*g*q[1]*math.cos(q[0]), m*g*math.sin(q[0])]).T
		
		return M, C, G
    
	def get_joint_acceleration(self, q, qdot, tau):
		"""
		Calculates the joint accelerations required to produce the given torques using the given robot
		state and kinematic parameters.
		
		qddot = inv(M) (tau - C qdot - G)
		
		Arguments:
		    q     (numpy array): joint angles
		    qdot  (numpy array): joint angular velocity
		    tau   (numpy array): joint torques
		Returns:
		    qddot (numpy array): joint accelerations
		"""
		M, C, G = self.get_MCG(q, qdot)
		qddot = (np.linalg.solve(M,(tau - C @ qdot - G))).ravel()
		return qddot
