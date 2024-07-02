# Calculate error spaces

# Import libraries
import numpy as np
import pygimli as pg
#import matplotlib.pyplot as plt
#import matplotlib.tri as tri
from sklearn.metrics import root_mean_squared_error
#from mpl_toolkits import mplot3d
from joblib import Parallel, delayed
import time
import sys
sys.path.insert(1, '../../src')

# Import functions
from EM1D import EMf_3Lay_HVP, GlobalSearch_3Lay, EMf_3Lay_Opt_HVP_1D

# Load model and data
model_B1_1 = np.load('models/model_synth_B1_1.npy')
model_B1_2 = np.load('models/model_synth_B1_2.npy')
model_B1_3 = np.load('models/model_synth_B1_3.npy')
model_B1_4 = np.load('models/model_synth_B1_4.npy')

data_B1_1 = np.load('data/data_synth_B1_1.npy')
data_B1_2 = np.load('data/data_synth_B1_2.npy')
data_B1_3 = np.load('data/data_synth_B1_3.npy')
data_B1_4 = np.load('data/data_synth_B1_4.npy')

data_B1_1_GS = np.load('data/data_GS_B1_1.npy')

#LUT = np.load('../data/LUTable_3Lay.npy')
conds = np.load('../data/conds.npy')
thicks = np.load('../data/thicks.npy')

# Load survey parameters
survey = np.load('../data/survey_3Lay.npy', allow_pickle=True).item()
offsets = survey['offsets']
height = survey['height']
freq = survey['freq']
lambd = survey['lambd']
filt = survey['filt']

# Load estimated model from Global Search
model_GS_B1_1 = np.load('results/model_GS_B1_1.npy')
model_GS_B1_2 = np.load('results/model_GS_B1_2.npy')
model_GS_B1_3 = np.load('results/model_GS_B1_3.npy')
model_GS_B1_4 = np.load('results/model_GS_B1_4.npy')

# Load estimated models from Optimization
model_Opt_B1_1 = np.load('results/model_Opt_B1_1.npy')
model_Opt_B1_2 = np.load('results/model_Opt_B1_2.npy')
model_Opt_B1_3 = np.load('results/model_Opt_B1_3.npy')
model_Opt_B1_4 = np.load('results/model_Opt_B1_4.npy')

# For example let's check the 1D model following position
pos = 10
nsl = len(conds)

def nrmse(obs, pred):
    
    nrse = root_mean_squared_error(obs, pred)/np.abs(np.max(obs)-np.min(obs))
    return nrse

def Error_analysis_parallel(data_true, max_err, h1, h2, s1, s2, s3):
    """ change later rmse"""

    mod = [h1, h2, s1, s2, s3]
    dat = EMf_3Lay_HVP(lambd, s1, s2, s3, h1, h2, height, offsets, freq, filt)
    nrse = nrmse(data_true, dat)

    if nrse < max_err:
        model_err = np.hstack((mod, nrse))
        return model_err

# Analyze error space

n_workers = 48
max_err = 0.3 # Testing larger noise
print('Start calculating error B1-1 ...')
startTime = time.time()

models_err_B1_1 = Parallel(n_jobs=n_workers,verbose=0)(delayed(Error_analysis_parallel)(data_B1_1[pos], max_err, h1, h2, s1, s2, s3)
                                           for h1 in thicks for h2 in thicks for s1 in conds for s2 in conds for s3 in conds)

executionTime = ((time.time() - startTime))/60
print('Execution time in seconds: ', f"{executionTime:.3}", ' minutes')

mod_err = [i for i in models_err_B1_1 if i is not None]
models_err_B1_1_snip = np.array(mod_err)
err_B1_1_snip = models_err_B1_1_snip[:,-1]
models_err_B1_1_snip = models_err_B1_1_snip[:,:5]

np.save('results/models_err_B1_1_0.3', models_err_B1_1_snip)
np.save('results/err_B1_1_0.3', err_B1_1_snip)

print('Start calculating error B1-2 ...')
startTime = time.time()

models_err_B1_2 = Parallel(n_jobs=n_workers,verbose=0)(delayed(Error_analysis_parallel)(data_B1_2[pos], max_err, h1, h2, s1, s2, s3)
                                           for h1 in thicks for h2 in thicks for s1 in conds for s2 in conds for s3 in conds)

executionTime = ((time.time() - startTime))/60
print('Execution time in seconds: ', f"{executionTime:.3}", ' minutes')

mod_err = [i for i in models_err_B1_2 if i is not None]
models_err_B1_2_snip = np.array(mod_err)
err_B1_2_snip = models_err_B1_2_snip[:,-1]
models_err_B1_2_snip = models_err_B1_2_snip[:,:5]

np.save('results/models_err_B1_2_0.3', models_err_B1_2_snip)
np.save('results/err_B1_2_0.3', err_B1_2_snip)

print('Start calculating error B1-3 ...')
startTime = time.time()

models_err_B1_3 = Parallel(n_jobs=n_workers,verbose=0)(delayed(Error_analysis_parallel)(data_B1_3[pos], max_err, h1, h2, s1, s2, s3)
                                           for h1 in thicks for h2 in thicks for s1 in conds for s2 in conds for s3 in conds)

executionTime = ((time.time() - startTime))/60
print('Execution time in seconds: ', f"{executionTime:.3}", ' minutes')

mod_err = [i for i in models_err_B1_3 if i is not None]
models_err_B1_3_snip = np.array(mod_err)
err_B1_3_snip = models_err_B1_3_snip[:,-1]
models_err_B1_3_snip = models_err_B1_3_snip[:,:5]

np.save('results/models_err_B1_3_0.3', models_err_B1_3_snip)
np.save('results/err_B1_3_0.3', err_B1_3_snip)

print('Start calculating error B1-4 ...')
startTime = time.time()

models_err_B1_4 = Parallel(n_jobs=n_workers,verbose=0)(delayed(Error_analysis_parallel)(data_B1_4[pos], max_err, h1, h2, s1, s2, s3)
                                           for h1 in thicks for h2 in thicks for s1 in conds for s2 in conds for s3 in conds)

executionTime = ((time.time() - startTime))/60
print('Execution time in seconds: ', f"{executionTime:.3}", ' minutes')

mod_err = [i for i in models_err_B1_4 if i is not None]
models_err_B1_4_snip = np.array(mod_err)
err_B1_4_snip = models_err_B1_4_snip[:,-1]
models_err_B1_4_snip = models_err_B1_4_snip[:,:5]

np.save('results/models_err_B1_4_0.3', models_err_B1_4_snip)
np.save('results/err_B1_4_0.3', err_B1_4_snip)






