import pandas as pd
import numpy as np

def report(lst, df):
    lst_unq = []
    lst_mode = []
    lst_max = []
    lst_min = []
    lst_na = []
    lst_ct_mode = []
    for x in lst:
        if df[x].dtype == 'object':
            lst_unq.append(df[x].nunique())
            lst_na.append(df[x].isna().sum())
        else:
            lst_unq.append(df[x].nunique())
            lst_max.append(df[x].max()) 
            lst_min.append(df[x].min())
            lst_na.append(df[x].isna().sum())
            md = df[x].mode().values[0]
            lst_mode.append(md)
            lst_ct_mode.append(f'{round(df[x].value_counts().values[0]/len(df)*100, 2)}%')
    report = pd.DataFrame([lst, lst_unq, lst_na, lst_max, lst_min, lst_mode, lst_ct_mode], index=['name', 'Number of Unique Values', 'Number of NaN', 'Max', 'Min', 'Mode', 'Frequency of Mode'])
    return report

def actual_vs_predicted_counts(yhat, ytest):
    
    predicted_fraud = len(yhat[yhat == True])
    predicted_nf = len(yhat[yhat == False])

    actual_fraud = len(ytest[ytest == True])
    actual_nf = len(ytest[ytest == False])
    
    return predicted_fraud, predicted_nf, actual_fraud, actual_nf

