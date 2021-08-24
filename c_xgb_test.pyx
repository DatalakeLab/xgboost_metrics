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
import resource
import memory_profiler
from memory_profiler import profile



def uso(mensagem):
    #usage = resource.getrusage(resource.RUSAGE_SELF)
    print ("**********************************************")
    print (mensagem)
    #print ("CPU Usage: " + str(psutil.cpu_times()))
    print ("***CPU Percent:None: " + str(psutil.cpu_percent(interval=None, percpu=True)))
    #print ("CPU Percent:1: " + str(psutil.cpu_percent(interval=1, percpu=True)))
    #print ("CPU Percent:0.1: " + str(psutil.cpu_percent(interval=0.1, percpu=True)))
    mem_usage = memory_profiler.memory_usage()[0]
    print ("***Memory Usage: " + str(mem_usage))
    #for name, desc in [
    # ('ru_utime', 'User time'),
    # ('ru_stime', 'System time'),
    # ('ru_maxrss', 'Max. Resident Set Size'),
    # ('ru_ixrss', 'Shared Memory Size'),
    # ('ru_idrss', 'Unshared Memory Size'),
    # ('ru_isrss', 'Stack Size'),
    # ('ru_inblock', 'Block inputs'),
    # ('ru_oublock', 'Block outputs'),
    # ]:
    # print '%-25s (%-10s) = %s' % (desc, name, getattr(usage, name))
   
def test_xgb_regression(n_samples = 10000, n_features = 20, n_estimators = 3, depth = 11):
    '''
    correctness and performance test for C-xgb for regression tasks
    '''
    cdef float base_score = 0.5

    #creating dataset
    ##uso("Creating DataSet: Begin")
    x, y = make_regression(n_samples = n_samples, n_features = n_features, n_informative = n_features - 1, random_state = 1)
    ##uso("Creating DataSet: End")

    #training xgb classifier on random dataset
    def create_xgb():
            
        uso("Training XGB Python: Begin")
        
        model = xgb.XGBRegressor(max_depth = depth, learning_rate = 0.1, n_estimators = n_estimators, silent = True, 
                                 objective = 'reg:linear', n_jobs = 8, min_child_weight = 1, 
                                  subsample = 0.8, colsample_bytree = 0.8, random_state = 5, missing = np.nan, base_score = base_score)

        model.fit(x, y)
        
        uso("Training XGB Python: END")

        return model

    model = create_xgb()
    
    #uso("Model Booster: Begin")
    booster = model.get_booster()
    #uso("Model Booster: End")

    #dumping xgb to json files
    tree_data = booster.get_dump(dump_format='json')
    cdef i
    for i in xrange(len(tree_data)):
        f = open("trees/tree_%d.json" % i, 'w')
        f.write(tree_data[i])
        f.close()


    #creating instance of CXgboost xgboost class
    # 0 in parameters means objective 'reg:linear'
    
    uso("Training C_XGB Cython: Begin")
    cdef CXgboost model_c = CXgboost(depth, n_features, n_estimators, 0, base_score)
    uso("Training C_XGB Cython: END")

    cdef float x_cython[50], time_c_xgb = 0.0, time_xgb = 0.0
    cdef int j, q, N = 10

    #performing tests
    #total_c= memory_profiler.memory_usage()[0]
    #total_p= memory_profiler.memory_usage()[0]
    uso("Predicting C_XGB Cython: Begin")
    for i in xrange(n_samples):
        for j in xrange(n_features): 
            x_cython[j] = x[i][j]#np.around(x[i][j], 3)
            
        preds_c_xgb = model_c.predict(x_cython, n_estimators) #C-xgb prediction
        
        #time measurement for CythonXGB
        start = time.time()
        #mem_c = memory_profiler.memory_usage()[0]
        for q in xrange(N):
           model_c.predict(x_cython, n_estimators)
           #if ((i % 5000) == 0) and ((q % 5) == 0):
                #mem_usage = memory_profiler.memory_usage()[0]
                #print ("**** Memory Usage: " + str(mem_usage))
                #uso (">>>Predicting C_XGB...")
        #mem_c += memory_profiler.memory_usage()[0]
        #total_c += mem_c/N
        time_c_xgb += (time.time() - start)
    uso("Predicting C_XGB Cython: END")
    
    uso("Predicting XGB Python: BEGIN")
    for i in xrange(n_samples):
        reshaped_sample = x[i].reshape(1, n_features)

        preds_xgb = model.predict(reshaped_sample, ntree_limit = n_estimators)[0] #xgb prediction

        #time measurement for XGBoost
        start = time.time()
        #mem_p = memory_profiler.memory_usage()[0]
        for q in xrange(N):
            model.predict(reshaped_sample)
            #if ((i % 5000) == 0) and ((q % 5) == 0):
                #mem_usage = memory_profiler.memory_usage()[0]
                #print ("**** Memory Usage: " + str(mem_usage))
                #uso (">>>Predicting XGB...")

        #mem_p += memory_profiler.memory_usage()[0]
        #total_p += mem_p/N
        time_xgb += (time.time() - start)     
    uso("Predicting XGB Python: END")

    
    print 'n_samples = %d | n_estimators = %d | max_depth = %d | objective = %s' % (n_samples, n_estimators, depth, 'reg:linear')
    print "XGBoost mean time in ms: %f" % (time_xgb*1000)
    #print "XGBoost mean Memory in ms: %f" % (total_p/n_samples)

    print "C_XGBoost mean time in ms: %f" % (time_c_xgb*1000)
    #print "C_XGBoost mean Memory in ms: %f" % (total_c/n_samples)

    print "ACCELERATION IS %f TIMES\n" % (time_xgb / time_c_xgb)


                
