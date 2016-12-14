
from theano import config
import numpy as np
import time
"""
Functions for evaluating a DKF object
"""
def infer(dkf, dataset):
    """ Posterior Inference using recognition network 
    Returns: z,mu,logcov (each a 3D tensor) Remember to multiply each by the mask of the dataset before
    using the latent variables
    """
    assert len(dataset.shape)==3,'Expecting 3D tensor for data' 
    return dkf.posterior_inference(dataset)

# TSBN: Consider each different time length of sample. Divide by time. Compare to train bound!!
# Bound: Just average per sample lower bound. Not consider time.
def evaluateBound(dkf, dataset, indicators, actions, mask, batch_size):
    """ Evaluate ELBO """
    start_time = time.time()

    N = dataset.shape[0]
    dkf.resetDataset(dataset, indicators, actions, mask, quiet=True)

    tsbn_bound = 0
    for bnum, st_idx in enumerate(range(0, N, batch_size)):
        end_idx = min(st_idx+batch_size, N)
        idx_data = np.arange(st_idx, end_idx)

        batch_vec = dkf.evaluate(idx=idx_data)

        M = mask[idx_data]
        tsbn_bound += (batch_vec / M.sum(axis=1, keepdims=True)).sum()
    end_time   = time.time()
    dkf._p(('(Evaluate) Validation Bound: %.4f [Took %.4f seconds]')
           %(tsbn_bound, end_time-start_time))
    return tsbn_bound

def modelCfac(dkf, actions):
    N = actions.shape[0]
    z_init = np.random.randn(N, 1, dkf.params['dim_stochastic'])
    x_drug, z_drug     = sample(dkf, actions, z =  z_init)
    x_nodrug, z_nodrug = sample(dkf, np.zeros_like(actions), z = z_init) 
    modelCfac          = {}
    modelCfac['x_drug']= x_drug
    modelCfac['a_drug']= actions
    modelCfac['z_drug']= z_drug
    modelCfac['x_nodrug']= x_nodrug
    modelCfac['a_nodrug']= np.zeros_like(actions)
    modelCfac['z_nodrug']= z_nodrug
    return modelCfac

def genderCfac(dkf, dataset, actions):
    dataCfac= {}
    for idx in range(dataset.shape[0]):
        pat_data = dataset[[idx],:,:] #1x5*183
        act_data = actions[[idx],:,:] #1x5*2

        gender = 'male' if act_data[0][0][0] == 1 and act_data[0][0][1] == 0 else 'female'
        print 'Now doing gender: ', gender
        # Conditions:
        # Make first point fixed and make Cfac at time=2 (22, 27, 32, 37)

        # Do inference w/ the patient data up to st_idx
        _, mu, _    = dkf.posterior_inference(pat_data[:,:1,:])
        remaining_act = act_data[:,1:,:]
        remaining_opposite_act = 1-remaining_act
        x_gender, z_gender      = sample(dkf, remaining_act, z = np.copy(mu[:,[-1],:]))
        x_opposite_gender, z_opposite_gender  = sample(dkf, remaining_opposite_act, z = np.copy(mu[:,[-1],:]))
        dataCfac[idx] = {}
        dataCfac[idx]['z_before_cfac'] = mu
        dataCfac[idx]['x'] = pat_data
        dataCfac[idx]['z_opposite_gender'] = z_opposite_gender
        dataCfac[idx]['x_opposite_gender'] = x_opposite_gender
        dataCfac[idx]['a_opposite_gender'] = remaining_opposite_act
        dataCfac[idx]['z_gender']   = z_gender
        dataCfac[idx]['x_gender']   = x_gender
        dataCfac[idx]['a_gender']   = remaining_act
    print 'Processed: ',len(dataCfac),' out of ',dataset.shape[0],' patients for data based cfac inference'
    return dataCfac

def sample(dkf, actions, z = None):
    #Sample (fake patients) from the distribution
    #Since we don't have a model for actions, use the emperical distribution over actions, for each one
    #sample more than one patient to check the ability of the model to generalize. Use all 1s as indicators 
    N              = actions.shape[0]
    if z is None:
        z          = np.random.randn(actions.shape[0],1,dkf.params['dim_stochastic'])
    else:
        assert z.shape[0] == actions.shape[0], 'Check if we at least have as many z as actions'

    all_z          = [np.copy(z)]
    for t in range(actions.shape[1]-1):
        #Use the transition means during sampling -Could vary this
        z,_        = dkf.transition_fxn(z= z, actions = actions[:,[t],:]) 
        all_z.append(np.copy(z))
    all_z    = np.concatenate(all_z,axis=1)
    x        = dkf.emission_fxn(all_z)
    return x, all_z

#How to store counterfactual results?
#Use pickle -> probably the best since it will be a bunch of hashmaps
