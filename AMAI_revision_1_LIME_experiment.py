"""

"""

# from joblib import delayed
from lime_exp_util import (VennAbers, ece, get_lime_change, get_change)
import numpy as np
import pandas as pd
import xgboost as xgb
# from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
# from sklearn.isotonic import IsotonicRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
# from sklearn import tree
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (brier_score_loss, log_loss, accuracy_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import calibration_curve
from matplotlib import pyplot as plt
from lime import lime_tabular
from shap import Explainer
# from BeXAI.target_models import model
# from BeXAI.explanations.explainer import Explanation, shap, result_values
# from BeXAI.evaluation.metrics import metrics_cls
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import time
from shap.utils import show_progress
# from joblib import Parallel,delayed
import copy
import pickle
from scipy.stats import spearmanr

def odds_to_proba(proba, shap_values):
    # new_proba = copy.deepcopy(proba)
    shap_proba = copy.deepcopy(shap_values)
    shap_proba_values = copy.deepcopy(shap_values)
    starting_odds = np.asarray(proba[:,1] / (1 - proba[:,1]))
    num_features = len(shap_values[0,:])
    for f in range(num_features):
        shift_in_odds = np.asarray(np.exp(shap_values[:,f]))
        shifted_odds = starting_odds * shift_in_odds
        shap_proba[:,f] = shifted_odds / (1 + shifted_odds)
        shap_proba_values[:,f] = proba[:,1] - shap_proba[:,f]
    # new_proba[:,0] = 1-shap_proba[:,0]
    # new_proba[:,1] = shap_proba[:,0]
    return shap_proba_values, shap_proba#, new_proba, proba - new_proba

def topRanked(series1, series2, metric='difference', order_importance='decreasing'):
    """
    Compute the difference between two series of ranked feauters

    Returns the difference between two series of ranked features.
    Positions (can) have decreasing importance, with the highest ranked feature being most important.
    
    Parameters
    ----------
    series1 : array-like
        matrix of ranked features per instance
    series2 : array-like
        matrix of ranked features per instance
    metric : 'difference' (deafult), 'spearman', optional
        'difference' calculates the proprtion of ranks that are the same and 'spearman' calculates the spearman correlation
    order_importance : 'decreasing' (deafult), 'identical', optional
        'decreasing' values the first position with 1, the second with 1/2 etc, whereas 'identical' does not consider rank position 
    """    
    
    assert(series1.size == series2.size)
    if len(series1.shape) == 1:
        if metric == 'difference':
            return np.mean([int(item) for item in series1 == series2])
        elif metric == 'spearman':
            return spearmanr(series1, series2).correlation
    m = 1
    m_sum = 0
    result = 0
    for col in range(series1.shape[1]):    
        m_sum += m   
        if metric == 'difference':
            result += m * np.mean([int(item) for item in series1[:,col] == series2[:,col]])
        elif metric == 'spearman':
            result += m * spearmanr(series1[:,col], series2[:,col]).correlation
        if order_importance == 'decreasing':
            m = m/2 

    return result/m_sum


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

def shap_fidelity(exp, explainer, model, instances, trainX=None):
    no_features = len(instances[0])
    no_instances = len(instances[:,0])
    average_pone = np.zeros((no_instances, no_features))
    fidelity = np.zeros((no_instances, no_features))
    proba_exp = np.zeros((no_instances, no_features))
    weight = explanation[0].values
    pred = model.predict_proba(instances)[:,1]
    average_pone = np.array([[pred[i] for j in range(no_features)] for i in range(no_instances)] )

    assert not np.any(np.isnan(average_pone)),'finns nan'
    from shap.utils import MaskedModel
    from shap.explainers import Exact, Permutation
    if trainX is not None:
        for i in range(no_features):
            p_one = 0 
            values = np.unique(trainX[:,i])
            for n in range(no_instances):
                val = values[values != instances[n,i]]
                if len(val) > 0:
                    instance = instances[n,:].copy() 
                    n_instances = np.array([instance for j in range(len(val))])
                    n_instances[:,i] = val
                    average_pone[n,i] = np.average(model.predict_proba(n_instances)[:,1])
    else:
        for n in range(no_instances):
            # print('.')
            instance = instances[n,:].copy()
            fm = MaskedModel(explainer.model, explainer.masker, explainer.link, instances[n,:])

            if issubclass(type(explainer), Permutation):
                max_evals = 3 * 2 * len(fm)
                # loop over many permutations
                inds = fm.varying_inputs()
                inds_mask = np.zeros(len(fm), dtype=bool)
                inds_mask[inds] = True
                masks = np.zeros(2*len(inds)+1, dtype=int)
                masks[0] = MaskedModel.delta_mask_noop_value
                npermutations = 1#max_evals // (2*len(inds)+1)
                outputs = []
                changed = None
                if len(inds) > 0:
                    p_one = np.zeros(no_features)
                    count = np.zeros(no_features)
                    for _ in range(npermutations):
                        np.random.shuffle(inds)

                        # create a large batch of masks to evaluate
                        i = 1
                        for ind in inds:
                            masks[i] = ind
                            i += 1
                        for ind in inds:
                            masks[i] = ind
                            i += 1
                        masked_inputs, varying_rows = explainer.masker(masks, *fm.args)

                        subset_masked_inputs = [arg[varying_rows.reshape(-1)] for arg in masked_inputs]
                        subset_masked_inputs = subset_masked_inputs[0]#[np.random.rand(varying_rows.sum()) < 0.3,:]
                        # evaluate the masked model 
                        for i in inds:
                            instance = instances[n,:].copy()  
                            subtract = 0  
                            for v in np.unique(subset_masked_inputs[:,i]):
                                if v == instances[n,i]:
                                    subtract = 1
                                    continue
                                instance[i] = v
                                p_one[i] += model.predict_proba([instance])[:,1]
                            count[i] += len(np.unique(subset_masked_inputs[:,i])) - subtract
                    for i in inds:
                        average_pone[n,i] = p_one[i]/count[i]
                    #     outputs = np.append(outputs, model.predict_proba(subset_masked_inputs)[:,1])  
                    #     if changed is None:                  
                    #         changed = subset_masked_inputs != instances[n,:]
                    #     else:
                    #         changed = np.append(changed, subset_masked_inputs != instances[n,:], axis=0)
                    # average_pone[n,inds] = [np.mean(outputs[changed[:,i]]) for i in inds]

            elif issubclass(type(explainer), Exact):

                inds = fm.varying_inputs()        
                delta_indexes = explainer._cached_gray_codes(len(inds))
                extended_delta_indexes = np.zeros(2**len(inds), dtype=int)
                for i in range(2**len(inds)):
                    if delta_indexes[i] == MaskedModel.delta_mask_noop_value:
                        extended_delta_indexes[i] = delta_indexes[i]
                    else:
                        extended_delta_indexes[i] = inds[delta_indexes[i]]

                # run the model
                masked_inputs, varying_rows = explainer.masker(extended_delta_indexes, *fm.args)
                num_varying_rows = varying_rows.sum(1)

                subset_masked_inputs = [arg[varying_rows.reshape(-1)] for arg in masked_inputs]
                subset_masked_inputs = subset_masked_inputs[0]#[np.random.rand(varying_rows.sum()) < 0.3,:]

                for i in inds:
                    p_one = 0
                    instance = instances[n,:].copy()  
                    subtract = 0  
                    for v in np.unique(subset_masked_inputs[:,i]):
                        if v == instances[n,i]:
                            subtract = 1
                            continue
                        instance[i] = v
                        p_one += model.predict_proba([instance])[:,1]
                    average_pone[n,i] = p_one/(len(np.unique(subset_masked_inputs[:,i])) - subtract)

                # outputs = model.predict_proba(subset_masked_inputs)[:,1]
                # changed = subset_masked_inputs != instances[n,:]
                # average_pone[n,inds] = [np.mean(outputs[changed[:,i]]) for i in inds]
    for feature in range(no_features):
        proba_exp[:,feature] = average_pone[:,feature] - weight[:,feature]
        fidelity[:,feature] = 1 - (pred - proba_exp[:,feature])

    return fidelity, average_pone, weight, proba_exp

def debug_print(message, debug=True):
    if debug:
        print(message)

num_attr = 2 # attribut att ta fram regler för
outerloop = 1 # antal upprepningar
k=10 # Antal foldar
divider = 1 #M 
number_of_bins = 10
plot_to_file = True
instance_to_file = False
eval_matrix = []
import sys
original_stdout = sys.stdout
is_debug = True
use_lime = True
if use_lime:
    result_metrics = ['proba','preds','fidelity_lime','monotonicity_lime','fidelity_shap','monotonicity_shap','change_shap','abs_shap','change_lime','abs_lime','change_lime2','abs_lime2','lal']
else:
    result_metrics = ['proba','preds','fidelity_shap','monotonicity_shap','change_shap','abs_shap']

descriptors = ['uncal','platt','va',] #['uncal','platt','va',] #  'platt','uncal',
Descriptors = {'uncal':'Uncal','platt': 'Platt','va': 'VA'}
models = ['xGB','RF',] # ['xGB','RF','DT','SVM',] # 'NN',
# models = ['RF',] # ['xGB','RF','DT','SVM',] # 'NN',
# models = ['xGB',] # ['xGB','RF','DT','SVM',] # 'NN',
explainers = ['shap',] # 'lime',

datasets = {1:"pc1req",2:"haberman",3:"hepati",4:"transfusion",5:"spect",6:"heartS",7:"heartH",8:"heartC",9:"je4243",10:"vote",11:"kc2",12:"wbc",
            13:"kc3",14:"creditA",15:"diabetes",16:"iono",17:"liver",18:"je4042",19:"sonar", 20:"spectf",21:"german",22:"ttt",23:"colic",24:"pc4",25:"kc1",}
klara = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]#7,21
rf_klara = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25]
tic_all = time.time()
for dataset in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,]:
    dataSet = datasets[dataset]

    tic_data = time.time()
    print(dataSet)
    fileName = '../Data/' + dataSet + ".csv"
    df = pd.read_csv(fileName, delimiter=';')
    Xn, y = df.drop('Y',axis=1), df['Y'] #Dela upp datamängden i inputattribut och targetattribut

    no_of_classes = len(np.unique(y))
    no_of_features = Xn.shape[1]
    no_of_instances = Xn.shape[0]

    t1 = DecisionTreeClassifier(min_weight_fraction_leaf=0.15) #Changed from min_leaf=4
    t2 = DecisionTreeClassifier(min_weight_fraction_leaf=0.15)
    s1 = SVC(probability=True) # Skala även input
    s2 = SVC(probability=True)
    r1 = RandomForestClassifier(n_estimators=100)
    r2 = RandomForestClassifier(n_estimators=100)
    h1 = HistGradientBoostingClassifier()
    h2 = HistGradientBoostingClassifier()
    g1 = xgb.XGBClassifier(objective='binary:logistic',use_label_encoder=False)
    g2 = xgb.XGBClassifier(objective='binary:logistic',use_label_encoder=False)
    
    model_dict = {'xGB':(g1,g2,"xGB",Xn),'RF':(r1,r2,"RF",Xn),'SVM': (s1,s2,"SVM",Xn),'DT': (t1,t2,"DT",Xn),'HGB': (h1,h2,"HGB",Xn)}#,'NN': (a1,a2,"NN",Xn)
    model_struct = [model_dict[model] for model in models]

    has_pickle = False
    for c1, c2, alg, X in model_struct:
        try:
            all_results = pickle.load(open(dataSet +' '+ alg + "- fidelity3.pkl", 'rb'))
            results = all_results['results']
            explanations = all_results['explanations']
            print('results unpickled')
            has_pickle = True
        except FileNotFoundError:
            if len(explainers) == 1:
                try:
                    all_results = pickle.load(open(dataSet +' '+ alg + ' '+ explainers[0] + "- fidelity3.pkl", 'rb'))
                    results = all_results['results']
                    explanations = all_results['explanations']
                    print('results unpickled')
                    has_pickle = True
                except FileNotFoundError:
                    has_pickle = False

        if not(has_pickle):
            tic_algorithm = time.time()
            debug_print(dataSet+' '+alg)
            results = {}
            results['calibrators']=[]
            results['yall']=np.array([])
            results['low']=np.array([])
            results['high']=np.array([])
            for desc in descriptors:
                results[desc] = {}
                for explain in explainers:
                    results[desc][explain] = {}
                    for metric in ['preds',]:
                        results[desc][explain][metric] = np.array([])
                    results[desc][explain]['proba'] = np.empty(shape=[0,no_of_classes])


            for x in range(outerloop):
                local = {}
                for metric in ['low','high']:
                    local[metric] = np.zeros(len(y))
                for desc in descriptors:
                    local[desc] = {}
                    for explain in explainers:    
                        local[desc][explain] = {}
                        local[desc][explain]['proba'] =  np.ones((len(y),no_of_classes), dtype=np.float32 ) 
                        for metric in ['preds',]:#,'change_shap','abs_shap','change_lime','abs_lime','change_lime2','abs_lime2','prob1','fidelity_lime','monotonicity_lime','fidelity_shap','monotonicity_shap']:
                            local[desc][explain][metric] = np.zeros(len(y))
                        
                explanations = {}
                # explanations['test_index'] = []    
                for desc in descriptors:              
                    explanations[desc] = {}
                    for explain in explainers:                    
                        explanations[desc][explain] = {}
                        # for metric in ['fidelity','proba_exp','avg_prob1','weight']:
                        #     explanations[desc][explain][metric] = np.zeros(len(y))
                        for metric in ['explanation']:
                            explanations[desc][explain][metric] = {}
                        for metric in ['abs_rank','values','fidelity','proba_exp','avg_prob1','weight']:
                            explanations[desc][explain][metric] = np.zeros((len(y),no_of_features))

                debug = {}
                kf = StratifiedKFold(n_splits=k)
                kn = 0

            
                for train_index, test_index in kf.split(X,y):  
                    calibrators = {} 
                    for desc in descriptors:
                        calibrators[desc] = {}
                        calibrators[desc]['lime'] = []
                        calibrators[desc]['shap'] = []
                    kn += 1
                    trainCalX, testX = X.iloc[train_index].values, X.iloc[test_index].values
                    trainCalY, testY = y.iloc[train_index].values, y.iloc[test_index].values
                    trainX, calX, trainY, calY = train_test_split(trainCalX, trainCalY, test_size=0.33,random_state=42)
                    
                    if any([alg == cmp for cmp in ['NN','SVM']]):
                        sc = StandardScaler()
                        trainX = sc.fit_transform(trainX)
                        calX = sc.transform(calX)
                        testX = sc.transform(testX)
                    c2.fit(trainX,trainY)

                    explainer = lime_tabular.LimeTabularExplainer(
                        training_data=np.array(trainX),
                        feature_names=X.columns,
                        class_names=np.unique(y),
                        mode='classification', 
                        random_state=42
                    )
                    calibrators['uncal']['model'] = c2
                    for explain in explainers:                  
                        local['uncal'][explain]['proba'][test_index,:] = calibrators['uncal']['model'].predict_proba(testX)
                        local['uncal'][explain]['preds'][test_index] = calibrators['uncal']['model'].predict(testX)
                    if descriptors.__contains__('platt'):
                        calibrators['platt']['model'] = CalibratedClassifierCV(base_estimator = c2, cv="prefit")
                        calibrators['platt']['model'].fit(calX,calY)
                        for explain in explainers:                 
                            local['platt'][explain]['proba'][test_index,:] = calibrators['platt']['model'].predict_proba(testX)
                            local['platt'][explain]['preds'][test_index] = np.round(calibrators['platt']['model'].predict(testX))
                    if descriptors.__contains__('va'):                        
                        calibrators['va']['model'] = VennAbers(calX, calY, c2)
                        for explain in explainers:                                    
                            local['va'][explain]['proba'][test_index,:] = calibrators['va']['model'].predict_proba(testX)
                            local['va'][explain]['preds'][test_index] = calibrators['va']['model'].predict(testX)
                    calibrators['data'] = {'trainX':trainX,'trainY':trainY,'calX':calX,'calY':calY,'testX':testX,'testY':testY,'test_index':test_index,}

                    debug_print(str(kn)  + ': ' + dataSet + ' ' + alg + ' ' + desc , is_debug)
                    np.random.seed(1337)
                    for explain in explainers:
                        for desc in descriptors:
                            calibrator = calibrators[desc]['model']
                            
                            if explain == 'shap':
                                predict_fn = lambda x:calibrator.predict_proba(x)[:,1]

                                explainer = Explainer(predict_fn, calX)
                                explanation = explainer(testX)          
                                explanations[desc][explain]['explanation'][str(kn)] = explanation
                                # Sort features and values to get identical structures from lime and shap for easier evaluation
                                explanations[desc][explain]['abs_rank'][test_index,:] = np.flip(np.argsort(np.abs(explanation.values), axis=1), axis=1)
                                explanations[desc][explain]['values'][test_index,:] = np.array([[explanation.values[i, int(j)] for j in explanations[desc][explain]['abs_rank'][ii]] for i, ii in enumerate(test_index)])  
                                calibrators[desc][explain].append(explanation) 
                            

                            if explain == 'lime':
                                predict_fn = lambda x:calibrator.predict_proba(x)
                                        
                                res_struct = {}
                                for m in ['explanation', 'fidelity', 'proba_exp', 'avg_prob1', 'weight','abs_rank','values']:
                                    res_struct[m] = [] 
                                explainer = lime_tabular.LimeTabularExplainer(
                                                    training_data=np.array(calX),
                                                    # feature_names=calX.columns,
                                                    class_names=np.unique(calY),
                                                    mode='classification', 
                                                    random_state=1337, 
                                                )
        #             
                                for i, j in enumerate(test_index):
                                    x = testX[i]
                                    exp = explainer.explain_instance(x, predict_fn = predict_fn, num_features=len(x))
                                    fidelity, tmp = lime_fidelity(exp, explainer, calibrator, x)
                                    res_struct['fidelity'].append(list(fidelity)) 
                                    res_struct['proba_exp'].append(tmp['pw']) 
                                    res_struct['avg_prob1'].append(tmp['p_one']) 
                                    res_struct['weight'].append(tmp['weight']) 
                                    res_struct['explanation'].append(exp) 
                                    # Sort features and values to get identical structures from lime and shap for easier evaluation
                                    res_struct['abs_rank'].append([res_struct['explanation'][i].local_exp[1][j][0] for j in range(len(x))])                                
                                    res_struct['values'].append([res_struct['explanation'][i].local_exp[1][j][1] for j in range(len(x))])    
                                    explanations[desc][explain]['explanation'][str(j)] = res_struct['explanation'][i]

                                for metric in ['abs_rank','values','fidelity','proba_exp','avg_prob1','weight']:
                                    explanations[desc][explain][metric][test_index,:] = res_struct[metric] 
                                calibrators[desc][explain].append(res_struct) 
                    # explanations['test_index'] = np.append(explanations['test_index'], test_index) 
                    
                    results['calibrators'].append(calibrators)

                for explain in explainers:
                    for desc in descriptors:
                        for metric in ['preds', 'proba']:
                    # isnan = np.isnan(local[desc][metric])
                    # local[desc][metric][isnan] = 0
                            results[desc][explain][metric]=np.append(results[desc][explain][metric], local[desc][explain][metric],axis=0)
                            
                for metric in ['low','high']:
                    results[metric]=np.append(results[metric],local[metric])
                results['yall']=np.append(results['yall'],y)         
            all_results = {'explanations':explanations, 'results':results}
            if len(explainers) == 1:
                a_file = open(dataSet +' '+  alg +' '+  explainers[0] + "- fidelity3.pkl", "wb")
            else:
                a_file = open(dataSet +' '+  alg + "- fidelity3.pkl", "wb")
            pickle.dump(all_results, a_file)
            a_file. close() 
        for explain in explainers:
            if explain == 'shap':
                try:
                    all_results = pickle.load(open(dataSet +' '+ alg + "- fidelity5.pkl", 'rb'))
                    results = all_results['results']
                    explanations = all_results['explanations']
                    print('results 5 unpickled')
                except FileNotFoundError:                    
                    for kn in range(k):
                        calibrators = results['calibrators'][kn]
                        data = calibrators['data']
                        trainX = data['trainX']
                        trainY = data['trainY']
                        calX = data['calX']
                        calY = data['calY']
                        testX = data['testX']
                        testY = data['testY']
                        test_index = data['test_index']   
                                        
                        for desc in descriptors:  
                            calibrator = calibrators[desc]['model']
                            predict_fn = lambda x:calibrator.predict_proba(x)[:,1]
                            explainer = Explainer(predict_fn, calX)  
                            explanation = calibrators[desc][explain]
                            fidelity, average_pone, weight, proba_exp = shap_fidelity(explanation, explainer, calibrator, testX, trainX=trainX)
                            explanations[desc][explain]['fidelity'][test_index,:] = fidelity
                            explanations[desc][explain]['proba_exp'][test_index,:] = [proba_exp[i,explanations[desc][explain]['abs_rank'][test_index[i],:].astype(int)] for i in range(len(testY))]
                            explanations[desc][explain]['avg_prob1'][test_index,:] = average_pone
                            explanations[desc][explain]['weight'][test_index,:] = weight
                            
                        debug_print(str(kn+1) + ': ' + dataSet + ' ' + alg + ' ' + str(explainers), is_debug)
                               
                    all_results = {'explanations':explanations, 'results':results}
                    a_file = open(dataSet +' '+  alg + "- fidelity5.pkl", "wb")
                    pickle.dump(all_results, a_file)
                    a_file. close() 

            # Evaluation lime/shap
            for calib in descriptors:                              
                idx = 0
                fop, mpv = calibration_curve(results['yall'], clip(np.nanmean(explanations[calib][explain]['proba_exp'],axis=1)), n_bins=number_of_bins)
                eval_matrix.append([dataSet, alg,'Brier', explain, '', '', calib, brier_score_loss(results['yall'], clip(np.nanmean(explanations[calib][explain]['proba_exp'],axis=1)))])
                eval_matrix.append([dataSet, alg,'Log', explain, '', '', calib, log_loss(results['yall'], clip(np.nanmean(explanations[calib][explain]['proba_exp'],axis=1)))])
                eval_matrix.append([dataSet, alg,'ECE', explain, '', '', calib, ece(results['yall'], clip(np.nanmean(explanations[calib][explain]['proba_exp'],axis=1)), fop, mpv)])
                fop, mpv = calibration_curve(results['yall'], clip([explanations[calib][explain]['proba_exp'][i][0] for i in range(no_of_instances)]), n_bins=number_of_bins)
                eval_matrix.append([dataSet, alg,'Brier', explain, '', 1, calib, brier_score_loss(results['yall'], clip([explanations[calib][explain]['proba_exp'][i][0] for i in range(no_of_instances)]))])
                eval_matrix.append([dataSet, alg,'Log', explain, '', 1, calib, log_loss(results['yall'], clip([explanations[calib][explain]['proba_exp'][i][0] for i in range(no_of_instances)]))])
                eval_matrix.append([dataSet, alg,'ECE', explain, '', 1, calib, ece(results['yall'], clip([explanations[calib][explain]['proba_exp'][i][0] for i in range(no_of_instances)]), fop, mpv)])

                # eval_matrix.append([dataSet, alg, 'fidelity_score',explain, '', '', calib,np.mean([explanations[calib][explain]['explanation'][str(i)].score for i in range(no_of_instances)])]) #
                eval_matrix.append([dataSet, alg, 'fidelity',explain, '', 1, calib,np.mean(explanations[calib][explain]['fidelity'])]) #
                eval_matrix.append([dataSet, alg, 'ma_fidelity',explain, '', 1, calib,np.mean(np.abs(explanations[calib][explain]['fidelity']))]) #
                
        evaluation_matrix = pd.DataFrame(data=eval_matrix, columns=['DataSet', 'Algorithm', 'Metric', 'Explainer', 'Type', 'NumFeatures', 'Comparison','Value'])
        evaluation_matrix.to_csv('AMAI_Revision_1_fidelity3.csv', index=True, header=True, sep=';')
        toc_algorithm = time.time()
        debug_print(dataSet + "-" + alg,is_debug )


    toc_data = time.time()  
    debug_print(dataSet + ': ' +str(toc_data-tic_data),is_debug )  


toc_all = time.time()    
debug_print(str(toc_data-tic_data),is_debug ) 