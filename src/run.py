import numpy as np
from implementations import *
from helpers import *
import matplotlib.pyplot as plt
from run_functions import *
from validation import *


def main():

	############### DATA LOADING ################
	print("Data loading")
	yb_train, input_data_train, ids_train = load_csv_data('../train.csv', sub_sample=True)
	yb_test, input_data_test, ids_test = load_csv_data('../test.csv', sub_sample=True)


	############### FEATURE PROCESSING ################
	print("Feature processing")

	# Remove -999 values
	input_data_train, input_data_test = remove999(input_data_train, yb_train, ids_train, input_data_test, ids_test)
	
	# Remove selected features
	#input_data_train, input_data_test = removecols(input_data_train, input_data_test, [14,15,17,18,24,25,27,28])

	# Turn positive columns into logarithm
	input_data_train, input_data_test = logpositive(input_data_train, input_data_test)

	# Standardize and sentralize data
	x_train = standardize(input_data_train)
	x_test = standardize(input_data_test)

	# Build model test data
	#y_test, tx_test = build_model_data(x_test,yb_test)


	########### RUN FUNCTIONS #############
	#gd_w, gd_loss = run_gradient_descent(yb_train, x_train)

	#sgd_w, sgd_loss = run_stochastic_gradient_descent(yb_train, x_train)

	rr_w, rr_loss, degree = run_ridge_regression(yb_train,x_train)
	tx_test = build_poly(x_test,degree)

	#ls_w, ls_loss, degree = run_least_square(yb_train,x_train)
	#print(ls_w.shape)
	#tx_test = build_poly(x_test,degree)

	#lr_w, lr_loss = run_logistic_regression(yb_train, x_train)
	#print(lr_w, lr_loss)

	#lr_w, lr_loss = run_logistic_regression_hessian(yb_train, x_train)
	#print(lr_w, lr_loss)

	#rlr_w, rlr_loss = run_reg_logistic_regression(yb_train, x_train)
	#print("w", rlr_w, "\n\n", "loss",rlr_loss)

	

	############# VALIDATIONS ###############
	#gradientdescent_gamma(yb_train, x_train)
	
	#leastsquares_degree(yb_train, x_train)
	
	#ridgeregression_lambda(yb_train, x_train)

	#ridgeregression_degree_lambda(yb_train, x_train)

	#logregression_gamma(yb_train, x_train)

	#logregression_gamma_hessian(yb_train, x_train)

	#logregression_lambda(yb_train, x_train)

	#logregression_gamma_degree(yb_train, x_train)

	#reglogregression_gamma(yb_train, x_train)

	stacking_cross(yb_train, x_train)
	
	#y_pred = stacking(yb_train,x_train,yb_test,x_test)
	


	############# MAKE PREDICTIONS ###############
	
	#y_pred = predict_labels(rr_w, tx_test)
	#create_csv_submission(ids_test, y_pred, 'results') #lager prediction-fila i Rolex-mappa med det navnet
	print("Finished")
	return 0;


### Run main function
main()
