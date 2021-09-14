import numpy as np
import pandas as pd

class Strategy():
    def __init__(self):
        self.date = None
        self.cache = None 

    def compute_target(self, universe_list):
        target_weight = {}
        for ticker in universe_list:
            target_weight[ticker] = 1
            
        target_weight = self.normalize(target_weight)

        return target_weight
    
    def custom_factor(self, ticker, ftype):
        assert False

    def default_factor(self, ticker, ftype):
        assert False
    
    def compute_factor(self, ticker, ftype):
        try:
            factor = self.default_factor(ticker, ftype)
        except:
            factor = self.custom_factor(ticker, ftype)
        return factor
    
    def compute_factor_zscore(self, universe_list, ftype, groupby_sector=False):
        ticker_to_factor = {}
        ticker_to_sector = {}
        for ticker in universe_list:
            try:
                factor = self.compute_factor(ticker, ftype)
                sector = self.get_value('tickerinfo', ticker, 'sector')
            except:
                factor = np.nan
            if np.isnan(factor):
                pass
            else:
                ticker_to_factor[ticker] = factor
                ticker_to_sector[ticker] = sector
        factor_series = pd.Series(ticker_to_factor).dropna().sort_values(ascending=False)
        sector_series = pd.Series(ticker_to_sector).dropna()
        
        
        if groupby_sector:
            df = pd.concat([factor_series, sector_series], axis=1).dropna()
            df.columns=['factor','sector']
            df = df.groupby('sector').transform(lambda x:(x-x.mean())/x.std(ddof=0)).dropna()
            factor_series = df['factor'].sort_values(ascending=False)
        else:
            factor_series = (factor_series-factor_series.mean())/factor_series.std()
        
        return factor_series
    
    def normalize(self, target_weight):
        target_sum = sum([np.abs(x) for x in target_weight.values()])
        target_weight = {ticker:target_weight[ticker]/target_sum for ticker in target_weight}
        assert np.abs(sum(target_weight.values())-1) < 1e-6
        return target_weight
    
    def get_value(self, table, ticker, value, lag=0):
        try:
            if table == 'tickerinfo':
                x = self.cache[table][ticker][value].iloc[0]
            else:
                x = self.cache[table][ticker][value].iloc[-1-lag]
        except:
            x = np.nan
        return x
    
    def get_value_list(self, table, ticker, value, lag='max'):
        try:
            if lag == 'max':
                x = self.cache[table][ticker][value]
            else:
                x = self.cache[table][ticker][value].iloc[-1-lag:]
        except:
            x = np.nan
        return x