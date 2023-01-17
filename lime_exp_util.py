from copy import copy
import numpy as np
import pandas as pd
from scipy.stats.stats import ttest_1samp
from sklearn.isotonic import IsotonicRegression
from math import (log,exp)
from scipy.special import expit
from scipy.special import xlogy
from scipy.optimize import fmin_bfgs

class Platt_Scaler():
    """Sigmoid regression model.
    Attributes
    ----------
    a_ : float
        The slope.
    b_ : float
        The intercept.
    """
    def __init__(self, clf) -> None:
        self.clf = clf


    def fit(self, probs, targets):
        self.a_, self.b_ = platt_calibration(probs,targets)
        return self

    def predict(self, prob):
        #return 1/(1+np.exp(-self.a_*test_prob-self.b_))
        # return np.exp(-(self.a_ * prob + self.b_))
        return expit(-(self.a_ * prob + self.b_))

    def predict_proba(self, X):
        #return 1/(1+np.exp(-self.a_*test_prob-self.b_))
        # return np.exp(-(self.a_ * prob + self.b_))
        prob = self.clf.predict_proba(X)
        proba = expit(-(self.a_ * prob + self.b_))
        norm_proba = (proba - np.min(proba, axis=1)) /  (np.max(proba, axis=1) - np.min(proba, axis=1))
        return norm_proba


def platt_calibration(predictions, y):
    """Probability Calibration with sigmoid method (Platt 2000)
    Parameters
    ----------
    predictions : ndarray of shape (n_samples,)
        The decision function or predict proba for the samples.
    y : ndarray of shape (n_samples,)
        The targets.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. If None, then samples are equally weighted.
    Returns
    -------
    a : float
        The slope.
    b : float
        The intercept.
    References
    ----------
    Platt, "Probabilistic Outputs for Support Vector Machines"
    """
   
    F = predictions  # F follows Platt's notations

    # Bayesian priors (see Platt end of section 2.2)
    prior0 = float(np.sum(y <= 0))
    prior1 = y.shape[0] - prior0
    T = np.zeros(y.shape)
    T[y > 0] = (prior1 + 1.) / (prior1 + 2.)
    T[y <= 0] = 1. / (prior0 + 2.)
    T1 = 1. - T

    def objective(AB):
        # From Platt (beginning of Section 2.2)
        P = expit(-(AB[0] * F + AB[1]))
        loss = -(xlogy(T, P) + xlogy(T1, 1. - P))
        return (loss).sum()
        

    def grad(AB):
        # gradient of the objective function
        P = expit(-(AB[0] * F + AB[1]))
        TEP_minus_T1P = T - P          
        dA = np.dot(TEP_minus_T1P, F)
        dB = np.sum(TEP_minus_T1P)
        return np.array([dA, dB])

    AB0 = np.array([0., log((prior0 + 1.) / (prior1 + 1.))])
    AB_ = fmin_bfgs(objective, AB0, fprime=grad, disp=False)
    return AB_[0], AB_[1]    

def predict_proba_laplace(treemodel, X):
    proba = treemodel.predict_proba(X)
    #treemodel.predict_proba(X)
    c = treemodel.n_classes_
    proba = proba[:, :treemodel.n_classes_]
    normalizer = proba.sum(axis=1)[:, np.newaxis]
    normalizer[normalizer == 0.0] = 1.0
    proba = (proba + 1) / (normalizer + c)

    return proba


def predict_proba_mestimate(treemodel, X, prior, m):
    proba = treemodel.predict_proba(X)
    c = treemodel.n_classes_
    proba = proba[:, :treemodel.n_classes_]
    normalizer = proba.sum(axis=1)[:, np.newaxis]
    normalizer[normalizer == 0.0] = 1.0
    proba = (proba + m*prior) / (normalizer + m)

    return proba


def bin_total(y_true, y_prob, n_bins):
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    # In sklearn.calibration.calibration_curve,
    # the last value in the array is always 0.
    binids = np.digitize(y_prob, bins) - 1
    return np.bincount(binids, minlength=len(bins))

def ece(y_true, y_prob, fop, mpv, n_bins=10):
    bins = bin_total(y_true, y_prob, n_bins)
    bins = bins[bins != 0]
    w = bins / np.sum(bins)
    return np.sum(w * abs(fop - mpv))

def vennAbers(cprobs, cpreds, ctargets, tprobs, tpreds):
    testInstances = len(tpreds)
    l = np.zeros(testInstances)
    h = np.zeros(testInstances)
    vap = np.zeros(testInstances)
    classes = np.unique(ctargets)
    ll = np.zeros(len(classes))
    hh = np.zeros(len(classes))
    iso = IsotonicRegression(out_of_bounds="clip")

    for i in range(testInstances):
        for c in range(len(classes)):
            class_predicted = classes[c]
            ix = np.searchsorted(classes, class_predicted)
            test_prob = tprobs[i, ix]
            cY = (ctargets == (class_predicted)).astype(int)
            tcalX = np.append(cprobs[:, ix], [test_prob])
            tcalY = np.append(cY, 0)
            iso.fit(tcalX, tcalY)
            ll[c] = iso.predict([test_prob])
            tcalY = np.append(cY, 1)
            iso.fit(tcalX, tcalY)
            hh[c] = iso.predict([test_prob])
            tmp=0
        chosen = np.argmax((hh+ll)/2, axis=0)
        l[i] = ll[chosen]
        h[i] = hh[chosen]
        vap[i] = classes[chosen]  # Must be fixed
    return l, h, vap

def fix_class_missing(proba, class_missing, unique_y, unique_train_y):
    if not class_missing:
        return proba
    new_proba = np.zeros([proba.shape[0], len(unique_y)])
    for i in range(len(unique_train_y)):
        idx = np.searchsorted(unique_y, unique_train_y[i])
        new_proba[:, idx] = proba[:, i]
    return new_proba

def venn_abers(cprobs, cpreds, ctargets, tprobs, tpreds):
        testInstances = len(tpreds)
        l = np.zeros(testInstances)
        h = np.zeros(testInstances)
        vap = np.zeros(testInstances)
        classes = np.unique(ctargets)
        ll = np.zeros(len(classes))
        hh = np.zeros(len(classes))
        iso = IsotonicRegression(out_of_bounds="clip")


        for i in range(testInstances):
            for c in range(len(classes)):
                class_predicted = classes[c]
                ix = np.searchsorted(classes, class_predicted)
                test_prob = tprobs[i, ix]
                cY = (ctargets == (class_predicted)).astype(int)
                tcalX = np.append(cprobs[:, ix], [test_prob])
                tcalY = np.append(cY, 0)
                iso.fit(tcalX, tcalY)
                ll[c] = iso.predict([test_prob])
                tcalY = np.append(cY, 1)
                iso.fit(tcalX, tcalY)
                hh[c] = iso.predict([test_prob])
                tmp=0
            chosen = np.argmax((hh+ll)/2, axis=0)
            l[i] = ll[chosen]
            h[i] = hh[chosen]
            vap[i] = classes[chosen]  # Must be fixed
        return l, h, vap

import numpy as np

from sklearn.isotonic import IsotonicRegression

def VennABERS_by_def(ds,test):
    p0,p1 = [],[]
    for x in test:
        ds0 = ds+[(x,0)]
        iso0 = IsotonicRegression().fit(*zip(*ds0))
        p0.append(iso0.predict([x]))
        
        ds1 = ds+[(x,1)]
        iso1 = IsotonicRegression().fit(*zip(*ds1))
        p1.append(iso1.predict([x]))
    return np.array(p0).flatten(),np.array(p1).flatten()

class VennAbers:
    iso = IsotonicRegression(out_of_bounds="clip")

    def __init__(self, calX, calY, model):
        self.cprobs = model.predict_proba(calX)
        self.ctargets = calY
        self.model = model

    def predict_proba_all(self, test_X):
        tprobs = self.model.predict_proba(test_X)
        classes = np.unique(self.ctargets)
        low,high = VennABERS_by_def(list(zip(self.cprobs[:,1],self.ctargets)),tprobs[:,1])
        l2 = [1-high[i] if low[i]<=0.5 else low[i] for i in range(len(low))]
        h2 = [1-low[i] if high[i]<=0.5 else high[i] for i in range(len(low))]
        ll1 = [[l2[i],1-h2[i]] for i in range(len(low))]
        hh1 = [[h2[i],1-l2[i]] for i in range(len(low))]
        va_proba1 = [[hh1[i][j] / (1-ll1[i][j]+hh1[i][j]) for j in range(2)] for i in range(len(l2))]
        chosen = np.argmax(va_proba1, axis=1)        
        vap1 = classes[chosen] 
        return l2, h2, vap1, va_proba1, ll1, hh1

    def predict(self, test_X):
        cprobs = self.cprobs[:,1]
        tprobs = self.model.predict_proba(test_X)[:,1]
        low,high = VennABERS_by_def(list(zip(cprobs,self.ctargets)),tprobs)
        tmp = high / (1-low + high)
        return np.asarray(np.round(tmp))

    def predict_proba(self, test_X):        
        cprobs = self.cprobs[:,1]
        va_proba = self.model.predict_proba(test_X)
        tprobs = va_proba[:,1]
        low,high = VennABERS_by_def(list(zip(cprobs,self.ctargets)),tprobs)
        tmp = high / (1-low + high)
        va_proba[:,0] = 1-tmp
        va_proba[:,1] = tmp
        return np.asarray(va_proba)

    def predict_proba_low(self, test_X):
        l, h, vap, va_proba, ll1, hh1 = self.predict_proba_all(test_X)
        return np.asarray(ll1)

    def predict_proba_high(self, test_X):
        l, h, vap, va_proba, ll1, hh1 = self.predict_proba_all(test_X)
        return np.asarray(hh1)

def lime_extractor(lal, nr_of_features): #lime_as_list
    new_lal = np.zeros(nr_of_features)
    bounds = np.ones([nr_of_features, 2])*np.nan
    
    for c in range(len(lal)):
        txt = lal[c][0]
        pos = txt.find('X')
        id = int(txt[pos+1:pos+txt[pos:].find(' ')])-1
        new_lal[id] = lal[c][1]

        less_than = txt.find('<=')
        greater_than = txt.find('< ')
        greater_than2 = txt.find('> ')
        if less_than != -1:
            bounds[id][1] = float(txt[less_than+3:])
        if greater_than != -1:
            bounds[id][0] = float(txt[:greater_than])
        if greater_than2 != -1:
            bounds[id][0] = float(txt[greater_than2+2:])

    most_important = np.argmax(np.abs(new_lal))

    return new_lal, bounds, most_important
        
def get_lime_change(exp, no_of_features, proba, test, step, min_, max_, calibrator):
    lst = exp.as_list()
    lal, bounds, MI  = lime_extractor(lst, no_of_features)
    tmp = test.copy() 
    pred_p = pred_tmp = 0
    if not(np.isnan(bounds[MI,0])):
        tmp[MI] = np.maximum(bounds[MI,0] - step[MI], min_[MI]) 
        new_p = calibrator.predict_proba([tmp])[0][0]
        pred_p = proba[0] + lal[MI]
    
    if not(np.isnan(bounds[MI,1])):
        if pred_p != 0:
            pred_tmp = pred_p         
            new_tmp = new_p       
        tmp[MI] = np.minimum(bounds[MI,1] + step[MI], max_[MI])
        new_p = calibrator.predict_proba([tmp])[0][0]
        pred_p = proba[0] + lal[MI]
    if pred_tmp != 0:
        pred_p = (pred_p + pred_tmp) / 2
        new_p = (new_p + new_tmp) / 2
    change = pred_p - new_p
    abs = np.abs(pred_p - new_p)
    return lal, change, abs
        
def get_change(shap_val, idx, proba, test, base, calibrator):
    shap_value = shap_val.values[idx]
    attr = shap_val.attributes[idx]
    abs = change = 0
    for val, att in zip(shap_value, attr):
        pred_p = new_p = 0
        tmp = test.copy() 
        
        tmp[att] = base[att] 
        new_p = calibrator.predict_proba([tmp])[0][0]
        pred_p = proba[0] + val  
        
        change += pred_p - new_p
        abs += np.abs(pred_p - new_p)
    return change/len(attr), abs/len(attr)



# def shap_fidelity(exp, explainer, model, instances):
#     no_features = len(instances[0])
#     no_instances = len(instances[:,0])
#     average_pone = np.zeros((no_instances, no_features))
#     fidelity = np.zeros((no_instances, no_features))
#     proba_exp = np.zeros((no_instances, no_features))
#     weight = explanation[0].values
#     pred = model.predict_proba(instances)[:,1]
#     average_pone = np.array([[pred[i] for j in range(no_features)] for i in range(no_instances)] )
#     assert not np.any(np.isnan(average_pone)),'finns nan'
#     from shap.utils import MaskedModel
#     from shap.explainers import Exact, Permutation
#     for n in range(no_instances):
#         # print('.')
#         instance = instances[n,:].copy()
#         fm = MaskedModel(explainer.model, explainer.masker, explainer.link, instances[n,:])

#         if issubclass(type(explainer), Permutation):
#             max_evals = 3 * 2 * len(fm)
#             # loop over many permutations
#             inds = fm.varying_inputs()
#             inds_mask = np.zeros(len(fm), dtype=bool)
#             inds_mask[inds] = True
#             masks = np.zeros(2*len(inds)+1, dtype=int)
#             masks[0] = MaskedModel.delta_mask_noop_value
#             npermutations = max_evals // (2*len(inds)+1)
#             outputs = []
#             changed = None
#             if len(inds) > 0:
#                 for _ in range(npermutations):
#                     np.random.shuffle(inds)

#                     # create a large batch of masks to evaluate
#                     i = 1
#                     for ind in inds:
#                         masks[i] = ind
#                         i += 1
#                     for ind in inds:
#                         masks[i] = ind
#                         i += 1
#                     masked_inputs, varying_rows = explainer.masker(masks, *fm.args)

#                     subset_masked_inputs = [arg[varying_rows.reshape(-1)] for arg in masked_inputs]
#                     subset_masked_inputs = subset_masked_inputs[0][np.random.rand(varying_rows.sum()) < 0.3,:]
#                     # evaluate the masked model 
#                     for i in inds:
#                         p_one = 0
#                         instance = instances[n,:].copy()    
#                         for v in np.unique(subset_masked_inputs[:,i]):
#                             instance[i] = v
#                             p_one += model.predict_proba(instance)[:,1]
#                         average_pone[n,i] = p_one/len(np.unique(subset_masked_inputs[:,i]))
#                     outputs = np.append(outputs, model.predict_proba(subset_masked_inputs)[:,1])  
#                     if changed is None:                  
#                         changed = subset_masked_inputs != instances[n,:]
#                     else:
#                         changed = np.append(changed, subset_masked_inputs != instances[n,:], axis=0)
#                 average_pone[n,inds] = [np.mean(outputs[changed[:,i]]) for i in inds]


#         elif issubclass(type(explainer), Exact):

#             inds = fm.varying_inputs()        
#             delta_indexes = explainer._cached_gray_codes(len(inds))
#             extended_delta_indexes = np.zeros(2**len(inds), dtype=int)
#             for i in range(2**len(inds)):
#                 if delta_indexes[i] == MaskedModel.delta_mask_noop_value:
#                     extended_delta_indexes[i] = delta_indexes[i]
#                 else:
#                     extended_delta_indexes[i] = inds[delta_indexes[i]]

#             # run the model
#             masked_inputs, varying_rows = explainer.masker(extended_delta_indexes, *fm.args)
#             num_varying_rows = varying_rows.sum(1)

#             subset_masked_inputs = [arg[varying_rows.reshape(-1)] for arg in masked_inputs]
#             subset_masked_inputs = subset_masked_inputs[0][np.random.rand(varying_rows.sum()) < 0.3,:]


#             outputs = model.predict_proba(subset_masked_inputs)[:,1]
#             changed = subset_masked_inputs != instances[n,:]
#             average_pone[n,inds] = [np.mean(outputs[changed[:,i]]) for i in inds]
#     for feature in range(no_features):
#         proba_exp[:,feature] = average_pone[:,feature] - weight[:,feature]
#         fidelity[:,feature] = 1 - (pred - proba_exp[:,feature])

#     return fidelity, average_pone, weight, proba_exp