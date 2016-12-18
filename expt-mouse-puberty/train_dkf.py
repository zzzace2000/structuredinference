import os,time,sys
import fcntl,errno
import socket

import cPickle

sys.path.append('../')
from datasets.load import loadDataset
from parse_args_dkf_mouse_puberty import params
from utils.misc import removeIfExists,createIfAbsent,mapPrint,saveHDF5,displayTime,getLowestError
from utils.misc import savePickle
from mouse_puberty_data.load import loadMedicalData

# dataset = loadMedicalData(setting=params['dataset'])
dataset = loadMedicalData(setting=params['dataset'], action_settings=params['actionset'])
params['savedir'] += '-'+params['dataset']+'-'+params['actionset']
if params['extrafoldername'] != '':
    params['savedir'] += '-'+params['extrafoldername']

createIfAbsent(params['savedir'])

#Saving/loading
for k in ['dim_observations','dim_actions','data_type','dim_indicators']:
    params[k] = dataset[k]
mapPrint('Options: ',params)

start_time = time.time()
from stinfmodel_mouse_puberty.dkf import DKF
import stinfmodel_mouse_puberty.learning as DKF_learn
import stinfmodel_mouse_puberty.evaluate as DKF_evaluate
displayTime('import DKF',start_time, time.time())
dkf    = None

#Remove from params
start_time = time.time()
removeIfExists('./NOSUCHFILE')
reloadFile = params.pop('reloadFile')
if os.path.exists(reloadFile):
    pfile=params.pop('paramFile')
    assert os.path.exists(pfile),pfile+' not found. Need paramfile'
    print 'Reloading trained model from : ',reloadFile
    print 'Assuming ',pfile,' corresponds to model'
    dkf  = DKF(params, paramFile = pfile, reloadFile = reloadFile) 
else:
    pfile= params['savedir']+'/'+params['unique_id']+'-config.pkl'
    print 'Training model from scratch. Parameters in: ',pfile
    dkf  = DKF(params, paramFile = pfile)
displayTime('Building dkf',start_time, time.time())

savef     = os.path.join(params['savedir'],params['unique_id']) 
print 'Savefile: ',savef
start_time= time.time()
train_bound, valid_bound = DKF_learn.learn(dkf, dataset['train_obs'],
                                indicators=dataset['train_ind'],
                                actions=dataset['train_act'],
                                mask=dataset['train_mask'],
                                # mask=None,
                                epoch_start =0,
                                epoch_end = params['epochs'], 
                                batch_size = params['batch_size'],
                                savefreq   = params['savefreq'],
                                savefile   = savef,
                                dataset_eval=dataset['valid_obs'],
                                indicators_eval=dataset['valid_ind'],
                                actions_eval=dataset['valid_act'],
                                mask_eval  = dataset['valid_mask'],
                                replicate_K= params['replicate_K'],
                                shuffle    = True
                                )
displayTime('Running DKF', start_time, time.time())
saveHDF5(params['savedir']+'/'+params['unique_id']+'-final-stats.h5',
         {'train_bound': train_bound,
            'valid_bound': valid_bound})

# Do inference
eval_z_q, eval_mu_q, eval_logcov_q = DKF_evaluate.infer(dkf, dataset['observation'])
saveHDF5(params['savedir']+'/'+params['unique_id']+'-final-infer.h5',
         { 'eval_z_q': eval_z_q, 'eval_mu_q': eval_mu_q, 'eval_logcov_q': eval_logcov_q })

# gender_each_time_cfac = DKF_evaluate.genderEachTimeCfac(dkf, dataset['observation'], dataset['actions'], dataset['indicators'])
# cPickle.dump({'gender_each_time_cfac': gender_each_time_cfac},
#              open('%s/%s-gender-each-time-cfac.pkl' % (params['savedir'], params['unique_id']), 'wb'))

# Do time inference at both gender
# time_cfac = DKF_evaluate.timeCfac(dkf, dataset['observation'], dataset['actions'])
# cPickle.dump({'time_cfac': time_cfac}, open('%s/%s-cfac.pkl' % (model, params['unique_id']), 'wb'))

# Do Counterfactual analysis
# Put male and make into female
# male_cfac = DKF_evaluate.genderCfac(dkf, dataset['observation'][:6], dataset['actions'][:6])
# # Put female and make into male
# female_cfac = DKF_evaluate.genderCfac(dkf, dataset['observation'][6:], dataset['actions'][6:])
# model_weights = dkf.getModelWeights()
#
# cPickle.dump({'male_cfac': male_cfac, 'female_cfac': female_cfac, 'model_weights': model_weights},
#              open(params['savedir']+'/'+params['unique_id']+'-cfac.pkl', 'wb'))


# Use val as test
# fname = pfile.replace('-config.pkl','-cfac.pkl')
# x_sampled, z_sampled  = DKF_evaluate.sample(dkf, dataset['valid_act'])
# tosave        = {}
# tosave['x_s'] = x_sampled
# tosave['a_s'] = dataset['valid_act']
# dataCfac      = DKF_evaluate.dataCfac(dkf, dataset['valid_obs'], dataset['valid_act'], dataset['act_dict'])
# modelCfac     = DKF_evaluate.modelCfac(dkf, dataset['valid_act'])
# savePickle([tosave, dataCfac, modelCfac], fname)

print 'Done evaluation'
import pdb; pdb.set_trace()