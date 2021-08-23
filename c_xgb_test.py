import c_xgb_test

print ("Linear Regression")
print ("**********************************************")
c_xgb_test.test_xgb_regression(n_samples = 10000, n_features = 20, n_estimators = 10, depth = 5)
#print ("**********************************************")
#c_xgb_test.test_xgb_regression(n_samples = 50000, n_features = 25, n_estimators = 10, depth = 10)
#print ("**********************************************")
#c_xgb_test.test_xgb_regression(n_samples = 100000, n_features = 20, n_estimators = 10, depth = 5)
#print ("**********************************************")
#c_xgb_test.test_xgb_regression(n_samples = 500000, n_features = 35, n_estimators = 10, depth = 10)

