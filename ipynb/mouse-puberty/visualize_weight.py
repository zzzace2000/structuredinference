import sys
import numpy as np
import cPickle
import h5py

sys.path.append('../..')
sys.path.append('../../..')
from tsne_python.tsne import tsne
import matplotlib.pyplot as plt

np.random.seed(7)
from sklearn.decomposition import PCA


def getHDF5Data(f):
    ff = h5py.File(f, mode='r')
    alldata = {}
    for k in ff.keys():
        alldata[k] = ff[k].value
    return alldata


def getPKLData(f):
    with open(f, 'rb') as f:
        data = cPickle.load(f)
    return data


def plotWithOriginalAndCfac(x_original, x_gender, x_opposite_gender, x_original_opposite,
                            gender='male', transform='None', title=''):
    # N(1) * T(5) * Obs(183)
    assert x_original.shape == x_gender.shape == x_opposite_gender.shape == x_original_opposite.shape
    assert len(x_original.shape) == 3
    assert x_original.shape[0] == 1

    def calAbsDifference(x_original, x_gender):
        a = np.abs((x_original[0] - x_gender[0]))
        a[a >= 1E308] = 0

        test = []
        for t in xrange(5):
            tmp = a[t]
            test += [(element, org, pred, t, ind) for element, isBigger, org, pred, ind in
                     zip(tmp, tmp>1, x_original[0][t], x_gender[0][t], xrange(len(tmp))) if isBigger]

        for t in xrange(a.shape[0]):
            plt.hist(a[t])
            plt.title('t = ' + str(t))
            plt.show()

        return a.mean()

    print 'abs diff. btw x_original, x_gender:', calAbsDifference(x_original, x_gender)
    print 'abs diff. btw x_original_opposite, x_opopsite_gender:', calAbsDifference(x_original_opposite,
                                                                                    x_opposite_gender)

    T = x_original.shape[1]
    # 3T * Obs(183)
    X = np.concatenate((x_original[0], x_gender[0], x_opposite_gender[0], x_original_opposite[0]), axis=0)

    if transform == 'PCA':
        pca = PCA(n_components=2)
        Y = pca.fit_transform(X)
        axis1_ratio = pca.explained_variance_ratio_[0]
        axis2_ratio = pca.explained_variance_ratio_[1]
        axis1_text = 'PC 1 (%.2f%%)' % (axis1_ratio * 100)
        axis2_text = 'PC 2 (%.2f%%)' % (axis2_ratio * 100)
    elif transform == 'tsne':
        Y = tsne(X, 2, 50, 30.0, quiet=True)
        axis1_text = 'tsne axis 1'
        axis2_text = 'tsne axis 2'
    elif transform == 'None':
        Y = X
        axis1_text = 'hidden state 1'
        axis2_text = 'hidden state 2'
    print 'Y shape:', Y.shape
    print 'T', T

    label = np.array(range(1, 1 + T), dtype=int)
    # Orange is original
    # Purples is original opposite
    plt_original = plt.scatter(Y[0:T, 0], Y[0:T, 1], s=20, c=label, marker='^', cmap='Oranges')
    cmap = 'Blues' if gender == 'male' else 'Reds'
    plt_gender = plt.scatter(Y[T:2 * T, 0], Y[T:2 * T, 1], s=20, c=label, marker='o',
                             cmap=cmap)
    cmap = 'Reds' if gender == 'male' else 'Blues'
    plt_opposite_gender = plt.scatter(Y[2 * T:3 * T, 0], Y[2 * T:3 * T, 1], s=20, c=label, marker='x',
                                      cmap=cmap)

    legend = plt.legend((plt_original, plt_gender, plt_opposite_gender),
                        ('x_observe', 'x_gender', 'x_opposite_gender'),
                        scatterpoints=1,
                        loc='upper right',
                        ncol=3,
                        fontsize=8)
    legend.legendHandles[0].set_color('orange')
    legend.legendHandles[1].set_color('blue' if gender == 'male' else 'red')
    legend.legendHandles[2].set_color('red' if gender == 'male' else 'blue')

    cbar = plt.colorbar()
    labels = np.arange(1, 6, 1)
    loc = labels
    cbar.set_ticks(loc)
    cbar.set_ticklabels(labels)
    cbar.ax.set_ylabel('time growth (from light to dark)', rotation=270, labelpad=20)

    plt.title('%s %s counterfactual' % (title, gender))
    plt.xlabel(axis1_text)
    plt.ylabel(axis2_text)
    plt.show()
    print Y


def plotAllCfac(male_cfac, opposite_example, gender='male'):
    for index in male_cfac:
        x_original = male_cfac[index]['x']
        x_gender = male_cfac[index]['x_gender']
        x_opposite_gender = male_cfac[index]['x_opposite_gender']
        x_original_opposite = opposite_example['x']

        x_gender = np.concatenate((x_original[0:1, 0:1, :], x_gender), axis=1)
        x_opposite_gender = np.concatenate((x_original[0:1, 0:1, :], x_opposite_gender), axis=1)

        plotWithOriginalAndCfac(x_original, x_gender, x_opposite_gender, x_original_opposite, gender=gender,
                                transform='PCA', title='x')

        z_gender = male_cfac[index]['z_gender']
        z_opposite_gender = male_cfac[index]['z_opposite_gender']

        if z_gender.shape[1] == 2:
            transform = 'None'
        elif z_gender.shape[1] > 2:
            transform = 'PCA'

        plotCfac(z_gender, z_opposite_gender, gender=gender, transform=transform, title='z')

        #     print x_original[0][2]
        #     print x_gender[0][2]
        #     ratio = np.absolute(x_original[0][2]-x_gender[0][2]) / np.absolute(x_original[0][2])
        #     print np.ma.masked_invalid(ratio).sum()
        #     print ratio
        #     break


def plotCfac(x_gender, x_opposite_gender, gender='male', transform='None', title=''):
    # N(1) * T(5) * Obs(183)
    assert x_opposite_gender.shape == x_gender.shape
    assert len(x_gender.shape) == 3
    assert x_gender.shape[0] == 1

    T = x_gender.shape[1]
    # 3T * Obs(183)
    X = np.concatenate((x_gender[0], x_opposite_gender[0]), axis=0)

    if transform == 'PCA':
        pca = PCA(n_components=2)
        Y = pca.fit_transform(X)
        axis1_ratio = pca.explained_variance_ratio_[0]
        axis2_ratio = pca.explained_variance_ratio_[1]
        axis1_text = 'PC 1 (%f%%)' % (axis1_ratio * 100)
        axis2_text = 'PC 2 (%f%%)' % (axis2_ratio * 100)
    elif transform == 'tsne':
        Y = tsne(X, 2, 50, 30.0, quiet=True)
        axis1_text = 'tsne axis 1'
        axis2_text = 'tsne axis 2'
    elif transform == 'None':
        Y = X
        axis1_text = 'hidden state 1'
        axis2_text = 'hidden state 2'
    print 'Y shape:', Y.shape
    print 'T', T

    label = np.array(range(2, 2 + T), dtype=int)
    # Green is original
    cmap = 'Blues' if gender == 'male' else 'Reds'
    plt_gender = plt.scatter(Y[:T, 0], Y[:T, 1], s=20, c=label, marker='o', cmap=cmap)
    cmap = 'Reds' if gender == 'male' else 'Blues'
    plt_opposite_gender = plt.scatter(Y[T:2 * T, 0], Y[T:2 * T, 1], s=20, c=label, marker='x', cmap=cmap)

    legend = plt.legend((plt_gender, plt_opposite_gender),
                        ('z_gender', 'z_opposite_gender'),
                        scatterpoints=1,
                        loc='upper right',
                        ncol=3,
                        fontsize=8)
    legend.legendHandles[0].set_color('blue' if gender == 'male' else 'red')
    legend.legendHandles[1].set_color('red' if gender == 'male' else 'blue')

    cbar = plt.colorbar()
    labels = np.arange(2, 6, 1)
    loc = labels
    cbar.set_ticks(loc)
    cbar.set_ticklabels(labels)
    cbar.ax.set_ylabel('time growth (from light to dark)', rotation=270, labelpad=20)

    plt.xlabel(axis1_text)
    plt.ylabel(axis2_text)
    plt.title('%s %s counterfactual' % (title, gender))
    plt.show()

# For time gender. Looks better.
expt = 'H-time-gender-linear'
uid = 'DKF_lr-5_0000e-03-vm-R-inf-structured-dh-10-ds-10-nl-relu-bs-256-ep-3000-rs-20-ar-1_0000e+01-rv-1_0000e-01-uid'

# expt = 'H-time-gender'
# uid = 'DKF_lr-8_0000e-04-vm-R-inf-structured-dh-10-ds-10-nl-relu-bs-256-ep-3000-rs-20-ar-1_0000e+01-rv-5_0000e-02-uid'

cfac_file = '../../expt-mouse-puberty/chkpt-%s/%s-cfac.pkl' % (expt, uid)
cfac = getPKLData(cfac_file)

male_cfac = cfac['male_cfac']
female_cfac = cfac['female_cfac']

model_weights = cfac['model_weights']
trans = model_weights['p_trans_W_mu']

a_trans = trans[:10]
z_trans = trans[10:]

a_trans_norm = np.linalg.norm(a_trans)
z_trans_norm = np.linalg.norm(z_trans)

plotAllCfac(male_cfac, opposite_example=female_cfac[0])

# Do it with Gaussian and pick those important weights.
# for i in xrange(i):
#     dim_vector = cfac['model_weights']['p_emis_W_mu'][i][:10]
#     print dim_vector

trans = cfac['model_weights']['p_trans_W_mu'][:10]
for i in xrange(10):
    print np.linalg.norm(trans[i])
    print trans[i]
# plt.hist(dim_vector)
# plt.show()