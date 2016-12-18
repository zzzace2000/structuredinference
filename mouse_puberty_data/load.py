import random

from utils.misc import getPYDIR,loadHDF5
import numpy as np
import os


def getActions(action_settings, numSample, T):
    def onehotEncodeTime(t, T):
        result = [0]*T
        result[t] = 1
        return result
    def oneHotEncodeGenderTime(t, T, gender):
        result = [0]*T*2 # Female and male with time T
        gender_map = {'male': 0, 'female': 1}
        result[gender_map[gender]*T + t] = 1
        return result

    N = numSample*T
    # N * T * dim(onehot)
    if action_settings == 'time':
        actions = np.array([[onehotEncodeTime(t, T) for t in xrange(T)]]*numSample, dtype=int)
    elif action_settings == 'gender':
        assert N == 60
        # First 30 is male, last 30 is female
        actions = np.array( ([[[1, 0]]*T]*(numSample/2) + [[[0, 1]]*T]*(numSample/2)), dtype=int)
    elif action_settings == 'time-gender':
        assert N == 60
        # First 30 is male, last 30 is female
        male_action_arr = [[oneHotEncodeGenderTime(t, T, 'male') for t in xrange(T)]]*(numSample/2)
        female_action_arr = [[oneHotEncodeGenderTime(t, T, 'female') for t in xrange(T)]]*(numSample/2)
        actions = np.array(male_action_arr+female_action_arr, dtype=int)
    return actions


def loadMedicalData(setting = 'male_H', action_settings='time'):
    """
    Need a good way to consider different configurations (with and without actions)
    Settings:
    A: X (observations) I (indicators) A (actions)      #
    B: X (observations|indicators) I (None) A (actions) #Actions on entire observations
    C: X (observations|indicators) I (None) A (None)    #Only for density estimation
    """
    # suffix = 'unnormalized_merged.txt'
    data_suffix = 'M_F_H_N_P_data.txt'
    ind_suffix = 'M_F_H_N_P_indicators.txt'
    DIR = os.path.dirname(os.path.realpath(__file__)).split('mouse_puberty_data')[0]+'mouse_puberty_data/data/'

    raw_data = np.loadtxt(DIR+data_suffix)
    indicators = np.loadtxt(DIR+ind_suffix)

    dataset = {}
    if setting == 'male_H':
        dataset['observation'] = raw_data[0:30, 0:183]
        dataset['indicators'] = indicators[0:30, 0:183]
    elif setting == 'male_N':
        dataset['observation'] = raw_data[:30, 183:2*183]
        dataset['indicators'] = indicators[:30, 183:2*183]
    elif setting == 'male_P':
        dataset['observation'] = raw_data[:30, 2*183:3*183]
        dataset['indicators'] = indicators[:30, 2*183:3*183]
    elif setting == 'female_H':
        dataset['observation'] = raw_data[30:, :183]
        dataset['indicators'] = indicators[30:, :183]
    elif setting == 'female_N':
        dataset['observation'] = raw_data[30:, 183:2 * 183]
        dataset['indicators'] = indicators[30:, 183:2 * 183]
    elif setting == 'female_P':
        dataset['observation'] = raw_data[30:, 2 * 183:3 * 183]
        dataset['indicators'] = indicators[30:, 2 * 183:3 * 183]
    elif setting == 'H':
        dataset['observation'] = raw_data[:, :183]
        dataset['indicators'] = indicators[:, :183]
    elif setting == 'N':
        dataset['observation'] = raw_data[:, 183:2*183]
        dataset['indicators'] = indicators[:, 183:2*183]
    elif setting == 'P':
        dataset['observation'] = raw_data[:, 2*183:3*183]
        dataset['indicators'] = indicators[:, 2*183:3*183]
    else:
        assert False, 'Wrong setting! '+setting

    # Arbitrary mapping each reading...
    def transformToTimeSeries(observation, total_num, individual_nums, time_points):
        assert observation.shape[0] == total_num, \
            'Observation shape is not equal to total_num, '+observation.shape[0]+', '+total_num
        assert total_num == individual_nums*time_points

        result = np.zeros((individual_nums, time_points, observation.shape[1]))
        for individual_index in xrange(individual_nums):
            for t in xrange(time_points):
                result[individual_index, t, :] = observation[individual_index+6*t]
        return result
    # N * T * Dim
    N = dataset['observation'].shape[0]
    T = 5
    numSample = N/T
    dataset['observation'] = transformToTimeSeries(dataset['observation'],
                                                   total_num=N,
                                                   individual_nums=numSample,
                                                   time_points=T)
    dataset['indicators'] = transformToTimeSeries(dataset['indicators'],
                                                   total_num=N,
                                                   individual_nums=numSample,
                                                   time_points=T)
    dataset['actions'] = getActions(action_settings, numSample, T)
    dataset['dim_actions'] = dataset['actions'].shape[2]

    dataset['dim_observations'] = dataset['observation'].shape[2]
    dataset['dim_indicators'] = dataset['indicators'].shape[2]
    # Only pick 1 as a validation set
    random_valid_index = random.randint(0, numSample-1)
    rest = range(numSample)[0:random_valid_index]+range(numSample)[random_valid_index+1:]
    dataset['train_obs'] = dataset['observation'][rest,:,:]
    dataset['train_act'] = dataset['actions'][rest,:,:]
    dataset['train_ind'] = dataset['indicators'][rest,:,:]

    dataset['valid_obs'] = dataset['observation'][random_valid_index:random_valid_index+1,:,:]
    dataset['valid_act'] = dataset['actions'][random_valid_index:random_valid_index+1,:,:]
    dataset['valid_ind'] = dataset['indicators'][random_valid_index:random_valid_index+1,:,:]
    # Do a testing. Make validation as a random vector. Observe its cost.
    # dataset['valid_obs'] = np.random.rand(2, T, dataset['dim_observations'])
    # dataset['valid_act'] = dataset['actions'][0:2,:,:]

    # Some dummy variable
    dim1, dim2 = dataset['train_obs'].shape[0], dataset['train_obs'].shape[1]
    dataset['train_mask'] = np.ones((dim1, dim2))
    dim1, dim2 = dataset['valid_obs'].shape[0], dataset['valid_obs'].shape[1]
    dataset['valid_mask'] = np.ones((dim1, dim2))

    dataset['data_type'] = 'gaussian'
    return dataset

    # elif setting=='B':
    #     dataset['train_obs']        = np.concatenate([dataset['train_obs'],dataset['train_ind']],axis=2)
    #     dataset['valid_obs']        = np.concatenate([dataset['valid_obs'],dataset['valid_ind']],axis=2)
    #     dataset['test_obs']         = np.concatenate([dataset['test_obs'],dataset['test_ind']],axis=2)
    #     dataset['dim_observations'] = dataset['train_obs'].shape[2]
    #     dataset['train_ind']        *= 0.
    #     dataset['valid_ind']        *= 0.
    #     dataset['test_ind']         *= 0.
    #     dataset['dim_indicators']   = 0
    #     dataset['obs_dict']         = dataset['obs_dict']+dataset['ind_dict']
    #     dataset['ind_dict']         = []
    #     dataset['dim_actions']      = dataset['train_act'].shape[2]
    # elif setting=='C':
    #     dataset['train_obs']        = np.concatenate([dataset['train_obs'],dataset['train_ind']],axis=2)
    #     dataset['valid_obs']        = np.concatenate([dataset['valid_obs'],dataset['valid_ind']],axis=2)
    #     dataset['test_obs']         = np.concatenate([dataset['test_obs'],dataset['test_ind']],axis=2)
    #     dataset['dim_observations'] = dataset['train_obs'].shape[2]
    #     dataset['train_ind']        *= 0.
    #     dataset['valid_ind']        *= 0.
    #     dataset['test_ind']         *= 0.
    #     dataset['train_act']        *= 0.
    #     dataset['valid_act']        *= 0.
    #     dataset['test_act']         *= 0.
    #     dataset['dim_indicators']   = 0
    #     dataset['dim_actions']      = 0
    #     dataset['obs_dict']         = dataset['obs_dict']+dataset['ind_dict']
    #     dataset['ind_dict']         = []
    #     dataset['act_dict']         = []
    # else:
    #     assert False,'Invalid setting :'+str(setting)
    # #All 1's mask
    # dataset['train_mask'] = np.ones_like(dataset['train_obs'][:,:,0])
    # dataset['valid_mask'] = np.ones_like(dataset['valid_obs'][:,:,0])
    # dataset['test_mask']  = np.ones_like(dataset['test_obs'][:,:,0])
    # dataset['data_type']  = 'binary'
    # return dataset

if __name__=='__main__':
    dset = loadMedicalData() 
    import ipdb;ipdb.set_trace()
