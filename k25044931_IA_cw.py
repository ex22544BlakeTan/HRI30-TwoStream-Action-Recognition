#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7CCEMIAA Intelligence and Autonomy

Coursework 1: Chameleon's Tongue
"""
import math
import random
import numpy as np
import sklearn.linear_model
import sklearn.neural_network 
import sklearn.metrics
from chameleon import chameleon
# ---- DO NOT MODIFY ABOVE THIS LINE ---- #
def estimate_model(R,Q):
	model = sklearn.neural_network.MLPRegressor()
# ---- Begin Answer to Question 1 ---- #
	R_train = R.T
	Q_train = Q.T

	#hidden_layer_sizes=(200, 200)
	model = sklearn.neural_network.MLPRegressor(
        	hidden_layer_sizes=(200,200, 200),
       		activation='tanh',
        	solver='lbfgs',
        	max_iter=10000,
        	random_state=1
    	)
	model.fit(R_train, Q_train)
# ---- End Answer to Question 1 ---- #
	return model

def estimate_rmse(Rd,model):
	rmse = -1 # initialise return value
# ---- Begin Answer to Question 2 ---- #
# 1. recienve joint paramenters
	Rd_test = Rd.T
	Q_pred = model.predict(Rd_test)
    
# 2. Call out the dynamic function
	cham = chameleon()
	R_actual = []
    
# 3. calculate corresponding actual location
	for q in Q_pred:
		r = cham.forward_kinematics(q)
		R_actual.append(r)
	
	R_actual = np.array(R_actual)
    
    # 4. Calculate RMSE using sklearn.metrics
    # squared=False for RMSE not MSE
	rmse = np.sqrt(sklearn.metrics.mean_squared_error(Rd_test, R_actual))
# ---- End Answer to Question 2 ---- #
	return rmse

def estimate_probability(Rd,model):
	probability = -1 # initialise return value
# ---- Begin Answer to Question 3 ---- #
	Rd_test = Rd.T
	Q_pred = model.predict(Rd_test)
	
	cham = chameleon()
	success_count = 0
	radius = 0.01  
    
	for i in range(len(Rd_test)):
        # Calculate expection
		r_actual = cham.forward_kinematics(Q_pred[i])
        # Calculate Euclidian distance
		distance = np.linalg.norm(r_actual - Rd_test[i])
		if distance <= radius:
			success_count += 1
	probability = success_count / len(Rd_test)
	
# ---- End Answer to Question 3 ---- #
	return probability

def estimate_mass(Q,Qdot,Qddot,Tau):
	mass = -1 # initialise return value
# ---- Begin Answer to Question 4 ---- #
#  Reshape the sample
	if Q.shape[0] == 2:
		Q = Q.T
		Qdot = Qdot.T
		Qddot = Qddot.T
		Tau = Tau.T
	g_const = 9.81 #
	Phi = []
	T_flat = []
    

	for i in range(len(Q)):
		q1, q2 = Q[i]        # q1 and q2
		dq1, dq2 = Qdot[i]   # velocity
		ddq1, ddq2 = Qddot[i]# acc
		tau1, tau2 = Tau[i]  
        
        # take out the m coefficient
        # q2^2 * ddq1 + 2*q2*dq1*dq2 + g*q2*cos(q1)
		p1 = (q2**2 * ddq1) + (2 * q2 * dq1 * dq2) + (g_const * q2 * math.cos(q1))
        # ddq2 - q2*dq1^2 + g*sin(q1)
		p2 = ddq2 - (q2 * (dq1**2)) + (g_const * math.sin(q1))
		
		Phi.append([p1])
		Phi.append([p2])
		T_flat.append(tau1)
		T_flat.append(tau2)
        
    # linear regression solution
	reg = sklearn.linear_model.LinearRegression(fit_intercept=False)
	reg.fit(np.array(Phi), np.array(T_flat))
	
	mass = float(reg.coef_[0])
# ---- End Answer to Question 4 ---- #
# ---- DO NOT MODIFY BELOW THIS LINE ---- #
	return mass

def main():
	random.seed(1)
	np.random.seed(1)

	# load training data
	R     = np.loadtxt('R.csv'    ,  delimiter=',') # tongue tip locations
	Q     = np.loadtxt('Q.csv'    ,  delimiter=',') # tongue configurations (angle and length)
	Qdot  = np.loadtxt('Qdot.csv' , delimiter=',') # tongue velocities
	Qddot = np.loadtxt('Qddot.csv', delimiter=',') # tongue accelerations
	Tau   = np.loadtxt('Tau.csv'  , delimiter=',') # tongue generalised forces
	Rd    = np.loadtxt('Rd.csv'   , delimiter=',') # insect locations
	Rp    = np.full(Rd.shape,math.nan) # initialise an array of NaNs

	# estimate model
	model = estimate_model(R,Q)

	# compute RMSE in model
	rmse = estimate_rmse(Rd,model)

	# compute probability of catching prey
	probability = estimate_probability(Rd,model)
	
	# estimate mass of tongue
	mass = estimate_mass(Q,Qdot,Qddot,Tau)
if __name__ == "__main__":
    main()
