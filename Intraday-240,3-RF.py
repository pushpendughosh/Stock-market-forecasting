import pandas as pd
import numpy as np
import random
import time
import pickle
from sklearn.ensemble import RandomForestClassifier
from Statistics import Statistics

import os
SEED = 9
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)

SP500_df = pd.read_csv('data/SPXconst.csv')
all_companies = list(set(SP500_df.values.flatten()))
all_companies.remove(np.nan)

constituents = {'-'.join(col.split('/')[::-1]):set(SP500_df[col].dropna()) 
                for col in SP500_df.columns}

constituents_train = {} 
for test_year in range(1993,2016):
    months = [str(t)+'-0'+str(m) if m<10 else str(t)+'-'+str(m) 
              for t in range(test_year-3,test_year) for m in range(1,13)]
    constituents_train[test_year] = [list(constituents[m]) for m in months]
    constituents_train[test_year] = set([i for sublist in constituents_train[test_year] 
                                         for i in sublist])
    
def trainer(train_data,test_data):
    random.seed(SEED)
    np.random.seed(SEED)
    
    train_x,train_y = train_data[:,2:-2],train_data[:,-1]
    train_y = train_y.astype('int')

    print('Started training')
    clf = RandomForestClassifier(n_estimators=1000, 
        max_depth=10, 
        random_state = SEED, 
        n_jobs=-1)
    clf.fit(train_x,train_y)
    print('Completed ',clf.score(train_x,train_y))

    dates = list(set(test_data[:,0]))
    predictions = {}
    for day in dates:
        test_d = test_data[test_data[:,0]==day]
        test_d = test_d[:,2:-2] 
        predictions[day] = clf.predict_proba(test_d)[:,1]
    return predictions


def simulate(test_data,predictions):
    rets = pd.DataFrame([],columns=['Long','Short'])
    k = 10
    for day in sorted(predictions.keys()):
        preds = predictions[day]
        test_returns = test_data[test_data[:,0]==day][:,-2]
        top_preds = predictions[day].argsort()[-k:][::-1] 
        trans_long = test_returns[top_preds]
        worst_preds = predictions[day].argsort()[:k][::-1] 
        trans_short = -test_returns[worst_preds]
        rets.loc[day] = [np.mean(trans_long),np.mean(trans_short)] 
    return rets   
    
def create_label(df_open,df_close,perc=[0.5,0.5]):
    if not np.all(df_close.iloc[:,0]==df_open.iloc[:,0]):
        print('Date Index issue')
        return
    perc = [0.]+list(np.cumsum(perc))
    label = (df_close.iloc[:,1:]/df_open.iloc[:,1:]-1).apply(
            lambda x: pd.qcut(x.rank(method='first'),perc,labels=False), axis=1)
    return label

def create_stock_data(df_close,df_open,st):
    st_data = pd.DataFrame([])
    st_data['Date'] = list(df_close['Date'])
    st_data['Name'] = [st]*len(st_data)
    
    daily_change = df_close[st]/df_open[st]-1
    m = list(range(1,20))+list(range(20,241,20))
    for k in m:
        st_data['IntraR'+str(k)] = daily_change.shift(k)
    for k in m:
        st_data['CloseR'+str(k)] = df_close[st].pct_change(k).shift(1)
    for k in m:
        st_data['OverNR'+str(k)] = df_open[st]/df_close[st].shift(k)-1
        
    st_data['R-future'] = daily_change 
    st_data['label'] = list(label[st]) 
    st_data['Month'] = list(df_close['Date'].str[:-3])
    st_data = st_data.dropna()
    
    trade_year = st_data['Month'].str[:4]
    st_data = st_data.drop(columns=['Month'])
    st_train_data = st_data[trade_year<str(test_year)]
    st_test_data = st_data[trade_year==str(test_year)]
    return np.array(st_train_data),np.array(st_test_data) 

result_folder = 'results2'
for directory in [result_folder]:
    if not os.path.exists(directory):
        os.makedirs(directory)

for test_year in range(1993,2020):
    
    print('-'*40)
    print(test_year)
    print('-'*40)
    
    filename = 'data/Open-'+str(test_year-3)+'.csv'
    df_open = pd.read_csv(filename)
    filename = 'data/Close-'+str(test_year-3)+'.csv'
    df_close = pd.read_csv(filename)
    
    label = create_label(df_open,df_close)
    stock_names = sorted(list(constituents[str(test_year-1)+'-12']))
    train_data,test_data = [],[]
    
    start = time.time()
    for st in stock_names:
        st_train_data,st_test_data = create_stock_data(df_close,df_open,st)
        train_data.append(st_train_data)
        test_data.append(st_test_data)

    train_data = np.concatenate([x for x in train_data])
    test_data = np.concatenate([x for x in test_data])
    
    print('Created :',train_data.shape,test_data.shape,time.time()-start)
    
    predictions = trainer(train_data,test_data)
    returns = simulate(test_data,predictions)
    result = Statistics(returns.sum(axis=1))
    print('\nAverage returns prior to transaction charges')
    result.shortreport() 
    
    with open(result_folder+'/predictions-'+str(test_year)+'.pickle', 'wb') as handle:
        pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    returns.to_csv(result_folder+'/avg_daily_rets-'+str(test_year)+'.csv')
    with open(result_folder+"/avg_returns.txt", "a") as myfile:
        res = '-'*30 + '\n' 
        res += str(test_year) + '\n'
        res += 'Mean = ' + str(result.mean()) + '\n'
        res += 'Sharpe = '+str(result.sharpe()) + '\n'
        res += '-'*30 + '\n'
        myfile.write(res)
        
