import sys 
import pandas as pd 
from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score
from oboe import AutoLearner, error
import numpy as np 
from autosklearn.classification import AutoSklearnClassifier
import pickle 


def read_abalone_dataframe() :
    """ Read abalone dataset to pandas dataframe """
    dataframe = pd.read_csv("abalone.data", header=None)
    dataframe.columns = ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', "shell_weight", "rings"]

    # Convet to numerical categories 
    def replace_sex(sex):
        return {
            'M' : 0, 
            'F' : 1, 
            'I' : 2
        }[sex]

    dataframe["sex"] = dataframe["sex"].apply(replace_sex)
    return dataframe  

def read_yeast_dataframe() :
    """ Read yeast dataset to pandas dataframe """
    yeast_dataset = pd.read_csv("yeast.data", header=None)

    yeast_dataset.iloc[:,-1] = yeast_dataset.iloc[:,-1].apply(lambda x : {
        "CYT" : 0,
        "NUC" : 1,
        "MIT" : 2,
        "ME3" : 3,
        "ME2" : 4,
        "ME1" : 5, 
        "EXC" : 6,
        "VAC" : 7,
        "POX" : 8,
        "ERL" : 9
    }[x])

    dataframe = yeast_dataset.iloc[:,1:]
    
    return dataframe 



# TODO Shuffle data before reading it
# TODO Save models (date et heure !!)  
# Pas de German 


def split_k_folds(k, dataframe) :

    """ Split data to k folds """
    # Shuffle data 
    dataframe = dataframe.sample(frac=1, random_state=42)
    # Split data 
    length = int(len(dataframe)/k) #length of each fold
    folds = []
    for i in range(k-1):
        folds += [dataframe[i*length:(i+1)*length]]
    folds += [dataframe[(k-1)*length:len(dataframe)]]
    
    return folds 

def launch_tpot(x_train, y_train, duration) :
    tpot_clf = TPOTClassifier(verbosity=2, max_time_mins=duration )
    tpot_clf.fit(x_train, y_train)
    return tpot_clf 

def launch_oboe(x_train, y_train, duration) :
    method = 'Oboe'  
    problem_type = 'classification'

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    oboe_clf = AutoLearner(p_type=problem_type, runtime_limit=duration, method=method, verbose=False)
    oboe_clf.fit(x_train, y_train)

    return oboe_clf


def launch_autosklearn(x_train, y_train, duration) :
    autosklearn_clf = AutoSklearnClassifier(time_left_for_this_task=duration*60, memory_limit= None )
    autosklearn_clf.fit(x_train, y_train)
    return autosklearn_clf 

def launch_classifier(tool, x_train, y_train, duration):
    if tool == "oboe" :
        return launch_oboe(x_train, y_train, duration)
    elif tool == "tpot" : 
        return launch_tpot(x_train, y_train, duration) 
    elif tool == "autosklearn" :
        launch_autosklearn(x_train, y_train, duration)
    else :
        raise Exception( "Not a valid tool")

def get_train_val(train_dataset, val_datatset, dataset, tool):
    if dataset not in ["abalone", "yeast"] :
        raise Exception( "Not a valid dataset")
    
    if dataset == "abalone" :
        x_train = train_dataset.drop(columns=['sex'])
        y_train = train_dataset['sex']
        x_val = val_datatset.drop(columns=['sex'])
        y_val = val_datatset['sex']

    else  : 
        x_train = train_dataset.iloc[:,:-1]
        y_train = train_dataset.iloc[:,-1]

        x_val = val_datatset.iloc[:,:-1]
        y_val = val_datatset.iloc[:,-1]
    
    if tool == "oboe":
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_val = np.array(x_val)
        y_val = np.array(y_val)
    
    return x_train, y_train, x_val, y_val  
    
    
        
    


def main() :
    tool = sys.argv[1]
    dataset = sys.argv[2]
    duration = int(sys.argv[3])
    k_folds = int(sys.argv[4])
    
    accuracy = []
    # Read data
    if dataset == "abalone" :
        dataframe = read_abalone_dataframe()
    elif dataset == "yeast" :
        dataframe = read_yeast_dataframe() 
    else :
        raise Exception("Not valid dataset")
    
    print(dataframe.head())
    # Split data 
    folds = split_k_folds(k_folds, dataframe)
    print(folds[0].head())
    # launch training 
    for i in range(k_folds) :
        print(i)
        val_datatset  = folds[i]
        train_dataset = pd.DataFrame()

        for j in range(k_folds) :
            if j != i :
                train_dataset = pd.concat([train_dataset, folds[j]])

        print(train_dataset.head())
        x_train, y_train, x_val, y_val = get_train_val(train_dataset, val_datatset, dataset, tool)
        
        print(f"launch classifier {tool} for the {i+1} eme split")
        clf = launch_classifier(tool, x_train, y_train, duration)
        print("training completed")
        
        # Save classifier
        if  tool == "tpot" :
            clf.export(f"{tool}_clf_{dataset}_{duration}_{i+1}.sav")
        else :
            pickle.dump(clf, open(f"{tool}_clf_{dataset}_{duration}_{i+1}.sav", 'wb'))
        print("model saved")
        
        # Get accuracy 
        if tool == "oboe" :
            y_predicted = clf.predict(x_val)[0]
        else :
            y_predicted = clf.predict(x_val)
        
        value = accuracy_score(y_val, y_predicted)
        accuracy.append(value) 
        print(f"accuracy : {value}")
    
    # print accuracy list 
    print(accuracy)
    # print mean accuracy 
    print(np.mean(np.array(accuracy)))
        
        


if __name__ == "__main__" :
    main()