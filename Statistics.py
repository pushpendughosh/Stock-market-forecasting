import pandas as pd
import numpy as np
import scipy.stats

class Statistics:
    def __init__(self,series):
        self.series = np.array(series)
        self.n = len(series)
    def mean(self):
        return np.mean(self.series)
    def std(self):
        return np.std(self.series)
    def stderr(self):
        return scipy.stats.sem(self.series)
    def percentiles(self,p=[.25,.5,.75]):
        return pd.Series(self.series).describe(percentiles=p)
    def pos_perc(self):
        return 100*sum(self.series>0)/self.n
    def skewness(self):
        return scipy.stats.skew(self.series)
    def kurtosis(self):
        return scipy.stats.kurtosis(self.series)
    def VaR(self,confidence):
        indx = int(confidence*self.n/100)
        return sorted(self.series)[indx-1]
    def CVaR(self,confidence):
        indx = int(confidence*self.n/100)
        return sum(sorted(self.series)[:indx])/indx
    def MDD(self):
        money = np.cumprod(1+self.series/100)
        maximums = np.maximum.accumulate(money)
        drawdowns = 1 - money/maximums
        return np.max(drawdowns)
    def sharpe(self,risk_free_rate = 0.0003):
        mu = self.mean()
        sig = self.std()
        sharpe_d = (mu-risk_free_rate)/sig
        return (252**0.5)*sharpe_d 
    def shortreport(self):
        print('Mean \t\t',self.mean())
        print('Standard dev \t',self.std())
        print('Sharpe ratio \t',self.sharpe())       
    def report(self):
        print('Mean \t\t',self.mean())
        print('Standard dev \t',self.std())
        print('Sharpe ratio \t',self.sharpe())
        print('Standard Error \t',self.stderr())
        print('Share>0 \t',self.pos_perc())
        print('Skewness \t',self.skewness())
        print('Kurtosis \t',self.kurtosis())
        print('VaR_1 \t\t',self.VaR(1))
        print('VaR_2 \t\t',self.VaR(2))
        print('VaR_5 \t\t',self.VaR(5))
        print('CVaR_1 \t\t',self.CVaR(1))
        print('CVaR_2 \t\t',self.CVaR(2))
        print('CVaR_5 \t\t',self.CVaR(5))
        print('MDD \t\t',self.MDD())
        print(self.percentiles())
        
