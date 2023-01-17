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