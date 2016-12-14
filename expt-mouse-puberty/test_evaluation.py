import os,time,sys
import numpy as np
import fcntl,errno
import socket

import cPickle

from utils.misc import removeIfExists,createIfAbsent,mapPrint,saveHDF5,displayTime,getLowestError
from utils.misc import readPickle, loadHDF5, getConfigFile, savePickle
from mouse_puberty_data.load import loadMedicalData
from parse_args_dkf_mouse_puberty import params

model = 'chkpt-%s-%s' % (params['dataset'], params['actionset'])
dataset    = loadMedicalData(params['dataset'], params['actionset'])
# reloadFile = 'chkpt-male_H/DKF_lr-8_0000e-04-vm-R-inf-structured-dh-50-ds-20-nl-relu-bs-256-ep-2000-rs-100-ar-1_0000e+01-rv-5_0000e-02-uid-EP1999-params.npz'
reloadFile = '%s/%s-EP2999-params.npz' % (model, params['unique_id'])

pfile      = getConfigFile(reloadFile)
params     = readPickle(pfile)[0]

from stinfmodel_mouse_puberty.dkf import DKF
import stinfmodel_mouse_puberty.learning as DKF_learn
import stinfmodel_mouse_puberty.evaluate as DKF_evaluate
dkf  = DKF(params, paramFile = pfile, reloadFile = reloadFile)

# eval_z_q, eval_mu_q, eval_logcov_q = DKF_evaluate.infer(dkf, dataset['observation'])
#
# saveHDF5('%s/%s-final-infer.h5' % (model, params['unique_id']),
#          { 'eval_z_q': eval_z_q, 'eval_mu_q': eval_mu_q, 'eval_logcov_q': eval_logcov_q })

# Put male and make into female
male_cfac = DKF_evaluate.genderCfac(dkf, dataset['observation'][:6], dataset['actions'][:6])
# Put female and make into male
female_cfac = DKF_evaluate.genderCfac(dkf, dataset['observation'][6:], dataset['actions'][6:])

cPickle.dump({'male_cfac': male_cfac, 'female_cfac': female_cfac}, open('%s/%s-cfac.pkl' % (model, params['unique_id']), 'wb'))

#Visualize in ipynb - display samples, display cfac on test, display 
# fname= 'check_evaluation.pkl'
# x_sampled, z_sampled  = DKF_evaluate.sample(dkf, dataset['test_act'])
#
# tosave = {}
# tosave['x_s'] = x_sampled
# tosave['a_s'] = dataset['test_act']
#
# dataCfac     = DKF_evaluate.dataCfac(dkf, dataset['test_obs'], dataset['test_act'], dataset['act_dict'])
# modelCfac = DKF_evaluate.modelCfac(dkf, dataset['test_act'])
# savePickle([tosave, dataCfac, modelCfac], fname)
print 'Done evaluation'
