#required extern class constructors and functions
cdef extern from "c_xgb/c_xgb.cpp":
    cdef cppclass CXgboost:
        CXgboost()
        CXgboost(int depth, int n_features, int n_trees_ , int objective_, float base_score_)
        float predict(float *features, int ntree_limit)

import numpy as np
import xgboost as xgb
import numpy as np
import pickle
import time
from sklearn.datasets import make_regression, make_classification
from libc.stdio cimport printf
import psutil

def test_xgb_regression(n_samples = 10000, n_features = 20, n_estimators = 3, depth = 11):
    '''
    correctness and performance test for C-xgb for regression tasks
    '''
    cdef float base_score = 0.5

    #creating dataset
    x, y = make_regression(n_samples = n_samples, n_features = n_features, n_informative = n_features - 1, random_state = 1)
    
    #training xgb classifier on random dataset
    def create_xgb():
            
        model = xgb.XGBRegressor(max_depth = depth, learning_rate = 0.1, n_estimators = n_estimators, silent = True, 
                                 objective = 'reg:linear', n_jobs = 8, min_child_weight = 1, 
                                  subsample = 0.8, colsample_bytree = 0.8, random_state = 5, missing = np.nan, base_score = base_score)
        
        model.fit(x, y)

        return model

    model = create_xgb()
    booster = model.get_booster()

    #dumping xgb to json files
    tree_data = booster.get_dump(dump_format='json')
    cdef i
    for i in xrange(len(tree_data)):
        f = open("trees/tree_%d.json" % i, 'w')
        f.write(tree_data[i])
        f.close()


    #creating instance of CXgboost xgboost class
    # 0 in parameters means objective 'reg:linear'
    cdef CXgboost model_c = CXgboost(depth, n_features, n_estimators, 0, base_score)
    cdef float x_cython[20], time_c_xgb = 0.0, time_xgb = 0.0
    cdef int j, q, N = 10

    #performing tests
    total_p=0
    total_c=0
    for i in xrange(n_samples):
        for j in xrange(n_features): 
            x_cython[j] = x[i][j]#np.around(x[i][j], 3)

        reshaped_sample = x[i].reshape(1, n_features)

        preds_xgb = model.predict(reshaped_sample, ntree_limit = n_estimators)[0] #xgb prediction
        preds_c_xgb = model_c.predict(x_cython, n_estimators) #C-xgb prediction
        
        #comparing predictions
        assert(abs(preds_xgb - preds_c_xgb) < 1e-3)
        
        #time measurement for CythonXGB
        memory_c_xgp = psutil.Process().memory_info().rss / (1024 * 1024)
        start = time.time()
        for q in xrange(N):
            model_c.predict(x_cython, n_estimators)
            memory_c_xgp += (psutil.Process().memory_info().rss / (1024 * 1024))
        time_c_xgb += (time.time() - start)
        memory_c_xgp = memory_c_xgp/N
        total_c += memory_c_xgp

        #time measurement for XGBoost
        memory_p_xgp = psutil.Process().memory_info().rss / (1024 * 1024)
        start = time.time()
        for q in xrange(N):
            model.predict(reshaped_sample)
            memory_p_xgp += (psutil.Process().memory_info().rss / (1024 * 1024))
        time_xgb += (time.time() - start)
        memory_p_xgp = memory_p_xgp/N
        total_p += memory_p_xgp

    total_c = total_c/n_samples
    total_p = total_p/n_samples
    print 'n_samples = %d | n_estimators = %d | max_depth = %d | objective = %s' % (n_samples, n_estimators, depth, 'reg:linear')
    print "XGBoost mean time in ms: %f" % (time_xgb*1000)
    print "Python XGBoost Memory: %f" % (total_p)

    print "C_XGBoost mean time in ms: %f" % (time_c_xgb*1000)
    print "Cython XGBoost Memory: %f" % (total_c)

    print "ACCELERATION IS %f TIMES\n" % (time_xgb / time_c_xgb)


