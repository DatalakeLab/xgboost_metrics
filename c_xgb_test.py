import c_xgb_test

print ("Linear Regression")
print ("**********************************************")
c_xgb_test.test_xgb_regression(n_samples = 10000, n_features = 20, n_estimators = 10, depth = 5)
#print ("**********************************************")
#c_xgb_test.test_xgb_regression(n_samples = 10000, n_features = 20, n_estimators = 10, depth = 10)
#print ("**********************************************")
#c_xgb_test.test_xgb_regression(n_samples = 10000, n_features = 20, n_estimators = 100, depth = 5)
#print ("**********************************************")
#c_xgb_test.test_xgb_regression(n_samples = 10000, n_features = 20, n_estimators = 100, depth = 10)

