import numpy as np
from implementations import *
from helpers import *
import matplotlib.pyplot as plt
from run_functions import *
from validation import *


def main():

	####################### DATA LOADING #######################
	print("Data loading")
	yb_train, input_data_train, ids_train = load_csv_data('../train.csv', sub_sample=True)
	yb_test, input_data_test, ids_test = load_csv_data('../test.csv', sub_sample=True)


	###################### FEATURE PROCESSING ##################
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


	###################### RUN FUNCTIONS #####################
	#w, loss = run_gradient_descent(yb_train, x_train)
	# Build model test data must be applied when running run_gradient_descent
	#y_test, tx_test = build_model_data(x_test,yb_test)

	#w, loss = run_stochastic_gradient_descent(yb_train, x_train)
	#Build model test data must be applied when running run_stochastic_gradient_descent
	#y_test, tx_test = build_model_data(x_test,yb_test)

	#w, loss, degree = run_least_square(yb_train,x_train)
	# Build model poly data, has to be done wen running run_least_square
	#tx_test = build_poly(x_test,degree)

	#w, loss, degree = run_ridge_regression(yb_train,x_train)
	# Build poly data, has to be done when running run_ridge_regression
	#tx_test = build_poly(x_test,degree)

	#w, loss = run_logistic_regression(yb_train, x_train)
	#Build model test data must be applied when running run_logistic_regression
	#y_test, tx_test = build_model_data(x_test,yb_test)

	#w, loss = run_reg_logistic_regression(yb_train, x_train)
	#Build model test data must be applied when running run_reg_logistic_regression
	#y_test, tx_test = build_model_data(x_test,yb_test)

	# When performing stacking, the predicted labels are given directly #
	#y_pred = stacking(yb_train,x_train,yb_test,x_test)


	

	###################### VALIDATIONS ########################
	#gradientdescent_gamma(yb_train, x_train)

	#stochastic_gradientdescent_gamma(yb_train, x_train)
	
	#leastsquares_degree(yb_train, x_train)
	
	#ridgeregression_lambda(yb_train, x_train)

	ridgeregression_degree_lambda(yb_train, x_train)

	#logregression_gamma(yb_train, x_train)

	#logregression_gamma_degree(yb_train, x_train)

	#reglogregression_gamma_lambda(yb_train, x_train)

	#stacking_crossvalidation(yb_train, x_train)
	
	


	################## MAKE PREDICTIONS #####################
	
	#y_pred = predict_labels(w, tx_test)
	#create_csv_submission(ids_test, y_pred, 'results') 
	print("Finished")
	return 0;


### Run main function
main()
