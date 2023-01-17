from experiment_util import (VennAbers, ece)
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (brier_score_loss, log_loss, accuracy_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import calibration_curve
from matplotlib import pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
import time
import sys

def debug_print(message, debug=True):
    if debug:
        print(message)

num_attr = 2 # attribut att ta fram regler för
outerloop = 1 # antal upprepningar
k=2 # Antal foldar
divider = 1 #M 
number_of_bins = 10
plot_to_file = True
eval_matrix = []
original_stdout = sys.stdout
is_debug = True
result_metrics = ['proba','preds',]

descriptors = ['uncal','platt','va',] 
Descriptors = {'uncal':'Uncal','platt': 'Platt','va': 'VA'}
models = ['xGB','RF',] 


datasets = {1:"pc1req",2:"haberman",3:"hepati",4:"transfusion",5:"spect",6:"heartS",7:"heartH",8:"heartC",9:"je4243",10:"vote",11:"kc2",12:"wbc",
            13:"kc3",14:"creditA",15:"diabetes",16:"iono",17:"kc1",18:"je4042",19:"sonar", 20:"spectf",21:"german",22:"ttt",23:"colic",24:"pc4",25:"liver",}

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
    g1 = xgb.XGBClassifier(objective='binary:logistic')
    g2 = xgb.XGBClassifier(objective='binary:logistic')
    
    model_dict = {'xGB':(g1,g2,"xGB",Xn),'RF':(r1,r2,"RF",Xn)}
    model_struct = [model_dict[model] for model in models]

    for c1, c2, alg, X in model_struct:
        tic_algorithm = time.time()
        debug_print(dataSet+' '+alg)
        results = {}
        results['yall']=np.array([])
        for desc in descriptors:
            results[desc] = {}
            results[desc]['preds'] = np.array([])
            results[desc]['proba'] = np.empty(shape=[0,no_of_classes])

        for x in range(outerloop):
            local = {}
            for desc in descriptors:
                local[desc] = {}
                local[desc]['proba'] =  np.ones((len(y),no_of_classes), dtype=np.float32 ) 
                local[desc]['preds'] = np.zeros(len(y))

            debug = {}
            for desc in descriptors:
                kf = StratifiedKFold(n_splits=k)

                for train_index, test_index in kf.split(X,y):
                    trainCalX, testX = X.iloc[train_index].values, X.iloc[test_index].values
                    trainCalY, testY = y.iloc[train_index].values, y.iloc[test_index].values
                    trainX, calX, trainY, calY = train_test_split(trainCalX, trainCalY, test_size=0.33,random_state=42)
                    if desc == 'uncal':
                        c1.fit(trainCalX,trainCalY)

                        local[desc]['preds'][test_index] = c1.predict(testX)
                        local[desc]['proba'][test_index,:] = c1.predict_proba(testX)
                        calibrator = c1
                    else:

                        c2.fit(trainX,trainY)
                        local[desc]['preds'][test_index] = c2.predict(testX)
                        if desc == 'platt':
                            calibrator = CalibratedClassifierCV(base_estimator = c2, cv="prefit")
                            calibrator.fit(calX,calY)
                            local[desc]['proba'][test_index,:] = calibrator.predict_proba(testX)
                            local[desc]['preds'][test_index] = np.round(calibrator.predict(testX))
                        elif desc == 'va':                                  
                            calibrator = VennAbers(calX, calY, c2)                     
                            local[desc]['proba'][test_index,:] = calibrator.predict_proba(testX)
                            local[desc]['preds'][test_index] = calibrator.predict(testX)

                    debug_print(dataSet + ' ' + alg + ' ' + desc, is_debug)

                for metric in result_metrics:
                    results[desc][metric]=np.append(results[desc][metric], local[desc][metric],axis=0)

            for metric in ['low','high']:
                results[metric]=np.append(results[metric],local[metric])
            results['yall']=np.append(results['yall'],y)        
        
        bool_to_int = [0,1]
        for desc in descriptors:            
            results[desc]['proba_predicted'] = np.amax(results[desc]['proba'],1)
            results[desc]['prob1'] = results[desc]['proba'][:,1]
            results[desc]['correct'] = np.asarray(results[desc]['preds'] == results['yall']).astype(int)
            results[desc]['diff'] = np.mean(results[desc]['proba_predicted'])-accuracy_score(results['yall'], results[desc]['preds'])
            results[desc]['fop'], results[desc]['mpv'] = calibration_curve(results['yall'], results[desc]['prob1'], n_bins=number_of_bins)
            results[desc]['mean_abs'] = np.mean(np.abs(results[desc]['lal']),axis=0)
            
        debug_print(dataSet + "-" + alg + ': klar')

        if plot_to_file: 
            plt.figure(figsize=(10, 10))
            ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
            ax2 = plt.subplot2grid((3, 1), (2, 0))
            ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            min_x = 1
            for desc in descriptors:
                name = Descriptors[desc]
                y_test = results['yall']
                prob_pos = results[desc]['prob1']
                ec = "%.3f" % (ece(results['yall'], results[desc]['prob1'], results[desc]['fop'], results[desc]['mpv']))

                fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=number_of_bins)

                line_new = name + ": ECE=" + ec
                ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                            label="%s" % (line_new,))
                ax2.hist(prob_pos, range=(0, 1), bins=int(number_of_bins), label=name,
                            histtype="step", lw=2)

                if min(fraction_of_positives) < min_x:
                    min_x = min(fraction_of_positives) - 0.05
            if min_x > 0.45:
                    min_x = 0.45

            ax1.set_ylabel("Fraction of positives")
            ax1.set_ylim([-0.05, 1.05])
            ax1.set_xlim([-0.022, 1.022])
            ax1.legend(loc="lower right")
            ax1.set_title(dataSet+' '+alg)

            ax2.set_xlabel("Mean predicted value")
            ax2.set_ylabel("Count")
            ax2.legend(loc="upper left", ncol=1)

            plt.tight_layout()
            plt.savefig('plots/' + dataSet + '_' + alg +'.png')
            # plt.show()
            plt.close() 

        for desc in descriptors:
            eval_matrix.append([dataSet, alg,'Acc', '',desc, accuracy_score(results['yall'], results[desc]['preds'])])
            eval_matrix.append([dataSet, alg,'AUC', '',desc, roc_auc_score(results[desc]['correct'], results[desc]['proba_predicted'])])
            eval_matrix.append([dataSet, alg,'Log', '',desc, log_loss(results[desc]['correct'], results[desc]['proba_predicted'])])
            eval_matrix.append([dataSet, alg,'ECE', '',desc, ece(results['yall'], results[desc]['prob1'], results[desc]['fop'], results[desc]['mpv'])])
            
        evaluation_matrix = pd.DataFrame(data=eval_matrix, columns=['DataSet', 'Algorithm', 'Metric', 'Explainer', 'Criteria', 'Value'])
        evaluation_matrix.to_csv('AMAI revision 1 experiment 1.csv', index=True, header=True, sep=';')
        toc_algorithm = time.time()
        debug_print(dataSet + "-" + alg + ': ' +str(toc_algorithm-tic_algorithm),is_debug )

    toc_data = time.time()  
    debug_print(dataSet + ': ' +str(toc_data-tic_data),is_debug )  


toc_all = time.time()    
debug_print(str(toc_data-tic_data),is_debug ) 