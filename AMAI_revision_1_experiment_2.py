from experiment_utils import (VennAbers, ece)
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (brier_score_loss, log_loss)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import calibration_curve
from lime import lime_tabular
from sklearn.calibration import CalibratedClassifierCV
import time
import pickle


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

descriptors = ['uncal','platt','va',] 
Descriptors = {'uncal':'Uncal','platt': 'Platt','va': 'VA'}
models = ['xGB','RF',] 
explainers = ['lime',]

datasets = {1:"pc1req",2:"haberman",3:"hepati",4:"transfusion",5:"spect",6:"heartS",7:"heartH",8:"heartC",9:"je4243",10:"vote",11:"kc2",12:"wbc",
            13:"kc3",14:"creditA",15:"diabetes",16:"iono",17:"liver",18:"je4042",19:"sonar", 20:"spectf",21:"german",22:"ttt",23:"colic",24:"pc4",25:"kc1",}
tic_all = time.time()
for dataset in datasets.keys():
    dataSet = datasets[dataset]

    tic_data = time.time()
    print(dataSet)
    fileName = 'data/' + dataSet + ".csv"
    df = pd.read_csv(fileName, delimiter=';')
    Xn, y = df.drop('Y',axis=1), df['Y'] #Dela upp datamängden i inputattribut och targetattribut

    no_of_classes = len(np.unique(y))
    no_of_features = Xn.shape[1]
    no_of_instances = Xn.shape[0]

    r1 = RandomForestClassifier(n_estimators=100)
    r2 = RandomForestClassifier(n_estimators=100)
    g1 = xgb.XGBClassifier(objective='binary:logistic',use_label_encoder=False)
    g2 = xgb.XGBClassifier(objective='binary:logistic',use_label_encoder=False)
    
    model_dict = {'xGB':(g1,g2,"xGB",Xn),'RF':(r1,r2,"RF",Xn),}
    model_struct = [model_dict[model] for model in models]

    has_pickle = False
    for c1, c2, alg, X in model_struct:
        try:
            all_results = pickle.load(open(dataSet +' '+ alg + " - experiment 2.pkl", 'rb'))
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
                        for metric in ['preds',]:
                            local[desc][explain][metric] = np.zeros(len(y))
                        
                explanations = {}    
                for desc in descriptors:              
                    explanations[desc] = {}
                    for explain in explainers:                    
                        explanations[desc][explain] = {}
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
                    kn += 1
                    trainCalX, testX = X.iloc[train_index].values, X.iloc[test_index].values
                    trainCalY, testY = y.iloc[train_index].values, y.iloc[test_index].values
                    trainX, calX, trainY, calY = train_test_split(trainCalX, trainCalY, test_size=0.33,random_state=42)
                    
                    c2.fit(trainX,trainY)
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
                    
                    results['calibrators'].append(calibrators)

                for explain in explainers:
                    for desc in descriptors:
                        for metric in ['preds', 'proba']:
                            results[desc][explain][metric]=np.append(results[desc][explain][metric], local[desc][explain][metric],axis=0)
                            
                for metric in ['low','high']:
                    results[metric]=np.append(results[metric],local[metric])
                results['yall']=np.append(results['yall'],y)         
            all_results = {'explanations':explanations, 'results':results}
            a_file = open(dataSet +' '+  alg + " - experiment 2.pkl", "wb")
            pickle.dump(all_results, a_file)
            a_file. close() 
        for explain in explainers:
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
                
        evaluation_matrix = pd.DataFrame(data=eval_matrix, columns=['DataSet', 'Algorithm', 'Metric', 'Explainer', 'Type', 'NumFeatures', 'Comparison','Value'])
        evaluation_matrix.to_csv('AMAI_Revision_1_experiment_2.csv', index=True, header=True, sep=';')
        toc_algorithm = time.time()
        debug_print(dataSet + "-" + alg,is_debug )

    toc_data = time.time()  
    debug_print(dataSet + ': ' +str(toc_data-tic_data),is_debug )  


toc_all = time.time()    
debug_print(str(toc_data-tic_data),is_debug ) 