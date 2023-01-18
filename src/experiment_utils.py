import numpy as np
from sklearn.isotonic import IsotonicRegression

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

        


def clip(val, min_=0, max_=1):
    if len(val) == 1:
        return min_ if val < min_ else max_ if val > max_ else val
    return [min_ if v < min_ else max_ if v > max_ else v for v in val]

def find_bin(rules, rule):
    return rules.index(rule)

def lime_fidelity(exp, explainer, model, instance, feature=None):
    if feature is None:
        fidelity = np.zeros(len(instance))
        res = {'p_one':[], 'weight':[], 'pw':[]}
        for i in range(len(instance)):
            fidelity[i], tmp = lime_fidelity(exp, explainer, model, instance.copy(), i)
            for m in ['p_one','weight','pw']:
                res[m].append(tmp[m])
        return fidelity, res
    feature_idx = exp.local_exp[1][feature][0]
    bin = find_bin(explainer.discretizer.names[feature_idx], exp.as_list()[feature][0])
    pred = exp.predict_proba[1]
    #non_foul = np.delete(explainer.categorical_names[odor_idx], foul_idx)
    normalized_frequencies = explainer.feature_frequencies[feature_idx].copy()
    if len(normalized_frequencies) > 3:
        normalized_frequencies[bin] = 0
        normalized_frequencies /= normalized_frequencies.sum()
    elif len(normalized_frequencies) > 1:
        if bin == len(normalized_frequencies):
            bin -= 1 # if the rule is X > Y when fewer than four bins, then it always correspond to the last of the bins.
        normalized_frequencies[bin] = 0
        normalized_frequencies /= normalized_frequencies.sum()

    average_pone=0
    weight = exp.local_exp[1][feature][1]
    for j in range(len(normalized_frequencies)):
        instance[feature_idx]=explainer.discretizer.means[feature_idx][j]
        p_one = model.predict_proba(instance.reshape(1, -1))[0,1]
        average_pone += p_one * normalized_frequencies[j]
    #     print('P(one | x=%f): %.2f' % (explainer.discretizer.means[feature_idx][j], p_one))
    # print ('P(one) = %.2f' % average_pone)
    
    # print ('Err: Pred - avgPreds - weight = %.2f - %.2f - %.2f = %.2f' % (pred, average_pone, weight, pred - average_pone - weight))
    return 1 - (pred - average_pone - weight), {'p_one':average_pone, 'weight':weight, 'pw':average_pone + weight}

def debug_print(message, debug=True):
    if debug:
        print(message)