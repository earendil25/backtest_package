import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
import datetime as dt
from dateutil.relativedelta import relativedelta
import os, zipfile,sqlite3, time, requests, json

import yfinance as yf
import quandl 

def timeis(func):  
    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print('[{}] is executed in {:.2f} seconds'.format(func.__name__, end-start))
        return result
    return wrap

DOWNLOAD_PATH = './data'
class DataLoader():
    def __init__(self, fred_key=None, quandl_key=None, fred_list=[], yfinance_list=[],
        sdate=None, edate=None, size=100, db_name='MyDB.db', is_update=False, use_bulk=False):

        if sdate is None: sdate = '1900-01-01'
        if edate is None: edate = dt.datetime.today().strftime('%Y-%m-%d')
        
        if is_update:
            db = sqlite3.connect(db_name)
            date_list = list(pd.read_sql('SELECT * FROM market', db).date.sort_values().unique())
            sdate = date_list[-1]
            db.close()

        print('Downloading data in {}--{}'.format(sdate.replace('-','/'), edate.replace('-','/')))
        
        self.use_bulk = use_bulk
        self.update_data(sdate, edate, size, fred_key, quandl_key, fred_list, yfinance_list)

        self.save_db(db_name=db_name, is_update=is_update)

    def make_download_folder(self):
        try:
            os.mkdir(DOWNLOAD_PATH)
        except:
            pass

    @timeis
    def update_data(self, sdate, edate, size, fred_key, quandl_key, fred_list, yfinance_list):
        if fred_key is not None:
            self.macro_df = self.update_macro(fred_key, fred_list)
        else:
            self.macro_df = pd.DataFrame({'datekey':[], 'value':[], 'ticker':[]})
        
        if quandl_key is not None:
            quandl.ApiConfig.api_key = quandl_key
            self.universe_df = self.update_universe(sdate, edate, size)
            universe = list(set(self.universe_df.ticker))
            universe = list(set(yfinance_list + universe))
            self.market_df = self.update_market(sdate, edate, universe)
            self.ticker_df = self.update_ticker(universe)
            self.fundamental_df = self.update_fundamentals(sdate, edate, universe)
            self.metric_df = self.update_metric(sdate, edate, universe)

            df_quandl = self.market_df
            df_yf = self.update_market_yf([])
            self.market_df = pd.concat([df_quandl, df_yf]).drop_duplicates(['date','ticker'])
        else:
            self.universe_df = self.update_universe_yf(size)
            universe = list(set(self.universe_df.ticker))
            universe = list(set(yfinance_list + universe))
            self.fundamental_df = self.update_fundamentals_yf(universe)
            self.market_df = self.update_market_yf(universe)
            self.ticker_df, ticker_to_info = self.update_ticker_yf(universe)
            self.metric_df = self.update_metric_yf(universe, ticker_to_info)
            
        

    @timeis
    def update_universe(self, sdate, edate, size):
        date_list = mcal.get_calendar('NYSE').valid_days(start_date=sdate, end_date=edate)

        monthly_date_list = [date_list[0].strftime('%Y-%m-%d')]
        for idx in range(len(date_list)-1):
            today, tomorrow = date_list[idx], date_list[idx+1]
            if today.month != tomorrow.month:
                monthly_date_list.append(today.strftime('%Y-%m-%d'))
                
        try:
            assert self.use_bulk == False
            print('Trying API call for universe')
            df = quandl.get_table('SHARADAR/DAILY',  
                                  date = monthly_date_list, 
                                  qopts={"columns":['date', 'marketcap','ticker']},
                                paginate=True).sort_values('marketcap', ascending=False
                            ).groupby('date').head(size).set_index('date').sort_index()
        except:
            print('Trying bulk download for universe')
            self.make_download_folder()
            filename = DOWNLOAD_PATH+'/universe.zip'
            quandl.export_table('SHARADAR/DAILY',  
                                  date = monthly_date_list, 
                                  qopts={"columns":['date', 'marketcap','ticker']},
                                filename=filename)
            
            zf = zipfile.ZipFile(filename, 'r')
            zf.extractall(DOWNLOAD_PATH)
            df = pd.read_csv(DOWNLOAD_PATH+'/'+zf.namelist()[0]).sort_values('marketcap', ascending=False
                            ).groupby('date').head(size).set_index('date').sort_index()

        df = df.reset_index()

        return df
    
    @timeis
    def update_universe_yf(self, size):
        universe = self.get_default_universe(size)
        df = pd.DataFrame({'ticker':universe, 
            'date':[pd.to_datetime('1900-01-01')]*len(universe),
            'marketcap':[0]*len(universe)})
        df = df.set_index('date').sort_index().reset_index()
        return df

    @timeis
    def update_ticker(self, universe):
        try:
            print('Trying API call for tickers')
            df = quandl.get_table('SHARADAR/TICKERS',ticker=universe, table='SF1', 
                                     paginate=True).set_index('permaticker').sort_index()
        except:
            print('Trying bulk download for tickers')
            self.make_download_folder()
            filename = DOWNLOAD_PATH+'/tickers.zip'
            quandl.export_table('SHARADAR/TICKERS',ticker=universe, table='SF1', filename=filename)
            zf = zipfile.ZipFile(filename, 'r')
            zf.extractall(DOWNLOAD_PATH)
            df = pd.read_csv(DOWNLOAD_PATH+'/'+zf.namelist()[0]).set_index('permaticker').sort_index()

        df = df.reset_index()

        return df

    @timeis
    def update_ticker_yf(self, universe):
        df = pd.DataFrame({'ticker':[], 'permaticker':[], 'sector':[]})
        idx = 0
        ticker_to_info = {}
        for ticker in universe:
            idx += 1
            try:
                tickerinfo = yf.Ticker(ticker.replace('.','-')).info
                sector = tickerinfo['sector']
            except:
                sector = 'unknown'
                print('{} has no sector information'.format(ticker))
            df = df.append({'ticker':ticker, 'permaticker':idx, 'sector':sector}, ignore_index=True)
            ticker_to_info[ticker] = tickerinfo

        df = df.set_index('permaticker').sort_index().reset_index()

        return df, ticker_to_info
    
    @timeis
    def update_fundamentals(self, sdate, edate, universe):
        try:
            print('Trying API call for fundamentals')
            df = quandl.get_table('SHARADAR/SF1', 
                          ticker = universe,
                          datekey = {'gte':sdate,'lte':edate},
                          dimension = 'ART', paginate=True).set_index('datekey').sort_index()
        except:
            print('Trying bulk download for fundamentals')
            self.make_download_folder()
            filename = DOWNLOAD_PATH+'/fundamentals.zip'
            quandl.export_table('SHARADAR/SF1', 
                          ticker = universe,
                          datekey = {'gte':sdate,'lte':edate},
                          dimension = 'ART', filename=filename)
            zf = zipfile.ZipFile(filename, 'r')
            zf.extractall(DOWNLOAD_PATH)
            df = pd.read_csv(DOWNLOAD_PATH+'/'+zf.namelist()[0]).set_index('datekey').sort_index()

        df = df.reset_index()

        return df
    
    @timeis
    def update_fundamentals_yf(self, universe):
        df = None
        for ticker in universe:
            try:
                ticker_yf = yf.Ticker(ticker.replace('.','-'))

                financials = ticker_yf.financials.T
                balance_sheet = ticker_yf.balance_sheet.T
                cashflow = ticker_yf.cashflow.T

                df_add = pd.concat([financials, balance_sheet, cashflow], axis=1)
                df_add['ticker'] = ticker
                df_add = df_add.loc[:,~df_add.columns.duplicated()]

                df_add.index = pd.to_datetime(df_add.index.rename('datekey'))

                df = pd.concat([df, df_add]).sort_index()
            except:
                print('{} has no fundamental information'.format(ticker))

        #df.index = pd.to_datetime(df.index.rename('datekey'))
        df = df.reset_index()

        return df

    @timeis
    def update_metric(self, sdate, edate, universe):
        try:
            assert self.use_bulk == False
            print('Trying API call for metric')
            df = quandl.get_table('SHARADAR/DAILY', 
                          ticker = universe,
                          date = {'gte':sdate, 'lte':edate}, paginate=True
                          ).set_index('date').sort_index()
        except:
            print('Trying bulk download for metric')
            self.make_download_folder()
            filename = DOWNLOAD_PATH+'/metric.zip'
            quandl.export_table('SHARADAR/DAILY', 
                          ticker = universe,
                          date = {'gte':sdate, 'lte':edate}, filename=filename)
            zf = zipfile.ZipFile(filename, 'r')
            zf.extractall(DOWNLOAD_PATH)
            df = pd.read_csv(DOWNLOAD_PATH+'/'+zf.namelist()[0]).set_index('date').sort_index()
        
        df = df.reset_index()

        return df
    
    @timeis
    def update_metric_yf(self, universe, ticker_to_info):
        df = None
        market_df = self.market_df
        for ticker in universe:
            try:
                tickerinfo = ticker_to_info[ticker]
                df_add = market_df[market_df.ticker==ticker][['date', 'close']]
                total_stocks = tickerinfo['marketCap']/df_add.iloc[-1].close

                df_add['close'] = df_add['close']*total_stocks
                df_add.columns = ['date','marketcap']
                df_add['ticker'] = ticker
            except:
                print('{} has no marketcap data'.format(ticker))

            df = pd.concat([df, df_add])

        return df

    @timeis
    def update_market(self, sdate, edate, universe):
        try:
            assert self.use_bulk == False
            print('Trying API call for market')
            df = quandl.get_table('SHARADAR/SEP', 
                          ticker = universe,
                          date = {'gte':sdate, 'lte':edate}, paginate=True).set_index('date').sort_index()
        except:
            self.make_download_folder()
            filename = DOWNLOAD_PATH+'/market.zip'
            print('Trying bulk download for market in {}'.format(filename))
            quandl.export_table('SHARADAR/SEP', 
                          ticker = universe,
                          date = {'gte':sdate, 'lte':edate}, filename=filename)
            print('done')
            zf = zipfile.ZipFile(filename, 'r')
            zf.extractall(DOWNLOAD_PATH)
            df = pd.read_csv(DOWNLOAD_PATH+'/'+zf.namelist()[0]).set_index('date').sort_index()
        
        df = df.reset_index()

        df = df[['date','open','high','low','closeadj','volume','ticker']]
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']
        df.date = pd.to_datetime(df.date)
        return df
        
    @timeis
    def update_market_yf(self, yfinance_list):
        df = None
        yf_ticker_list = [
            '^GSPC','^IXIC','^DJI','^RUT','^VIX','^TNX','^SP500TR',
            'GC=F', 'CL=F']

        tickers = [x.replace('.','-') for x in set(yf_ticker_list + yfinance_list)]
        df_all = yf.download(tickers=tickers, auto_adjust=True, period='max', group_by='ticker')

        df = None
        for ticker in tickers:
            df_add = df_all[ticker].dropna()
            df_add['ticker'] = ticker.replace('-','.')
            df = pd.concat([df,df_add])
            
        df = df.reset_index()
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'ticker']]
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'ticker',]
        df.date = pd.to_datetime(df.date)
        return df

    @timeis
    def update_macro(self, fred_key, fred_list):
        macro_list = ['T10Y2Y']
        macro_list = set(macro_list + fred_list)
        df = None
        for ticker in macro_list:
            try:
                df_add = self._get_PIT_df(ticker, fred_key)
                df_add['ticker'] = [ticker for _ in df_add.index]
                df = pd.concat([df, df_add],axis=0)
            except:
                print('{} is not retrieved from ALFRED'.format(ticker))

        df = df.reset_index()        
        return df

    def _get_PIT_df(self, ID, fred_key):
        API_KEY = fred_key
        REAL_TIME_START, REAL_TIME_END = '1900-01-01', '9999-12-31'
        
        url = 'https://api.stlouisfed.org/fred/series/observations?series_id={}'.format(ID)
        url += '&realtime_start={}&realtime_end={}&api_key={}&file_type=json'.format(
                                        REAL_TIME_START, REAL_TIME_END, API_KEY)
        
        response = requests.get(url)
        observations = json.loads(response.text)['observations']
        
        df = pd.DataFrame(observations).sort_values(['date','realtime_start']
            ).groupby('date').first()
        df.index = pd.to_datetime(df.index)
        df.realtime_start = pd.to_datetime(df.realtime_start)

        df['datekey'] = df.realtime_start
        df['is_inferred'] = (df.datekey == df.datekey.shift(1))|(
            df.datekey == df.datekey.shift(-1))

        non_inferred_df = df[df['is_inferred']==False]
        lag_list = [(y-x).days for x,y in 
                        zip(non_inferred_df.index, non_inferred_df.datekey)]
        mean_lag, max_lag = int(np.mean(lag_list)+1), int(np.max(lag_list)+1)
        
        df.datekey = [
            date + relativedelta(days=mean_lag) if df.loc[date].is_inferred
            else df.loc[date].datekey
            for date in df.index]

        df = df[['value','datekey','is_inferred']]
        df['cdate'] = df.index
        df = df.set_index('datekey')

        return df
    
    @timeis
    def save_db(self, db_name, is_update=False):

        db = sqlite3.connect(db_name)

        if_exists = 'append' if is_update else 'replace'
        try:
            self.universe_df.to_sql(name='universe', con=db, if_exists=if_exists, index=False)
        except:
            print('Universe is not saved')
        try:
            self.ticker_df.to_sql(name='ticker', con=db, if_exists=if_exists, index=False)
        except:
            print('Ticker is not saved')
        try:
            self.fundamental_df.to_sql(name='fundamentals', con=db, if_exists=if_exists, index=False)
        except:
            print('Fundamental is not saved')
        try:
            self.metric_df.to_sql(name='metric', con=db, if_exists=if_exists, index=False)
        except:
            print('Metric is not saved')
        try:
            self.market_df.to_sql(name='market', con=db, if_exists=if_exists, index=False)
        except:
            print('Market is not saved')
        try:
            self.macro_df.to_sql(name='macro', con=db, if_exists=if_exists, index=False)
        except:
            print('Macro is not saved')
        
        
        if is_update:
            def drop_duplicates(table, subset):
                qry = 'SELECT * FROM {}'.format(table)
                pd.read_sql(sql=qry, con=db).drop_duplicates(subset, keep='last'
                    ).sort_values(subset[-1]
                    ).to_sql(name=table, con=db, index=False, if_exists='replace')

            drop_duplicates('universe', ['ticker', 'date'],)
            drop_duplicates('ticker', ['ticker', 'permaticker'])
            drop_duplicates('fundamentals', ['ticker', 'datekey'])
            drop_duplicates('metric', ['ticker', 'date'])
            drop_duplicates('market', ['ticker', 'date'])
            drop_duplicates('macro', ['ticker', 'datekey'])

        try:
            date_list = pd.read_sql('SELECT * FROM market', db).date.sort_values().unique()

            print('DB has data from {} to {}'.format(
                date_list[0], date_list[-1]))
        except:
            pass

        db.close()
        
        print('Data saved in {}'.format(db_name))


    def get_default_universe(self, size):
        sp500 = ['AAPL',
            'MSFT',
            'AMZN',
            'FB',
            'GOOGL',
            #'GOOG',
            'NVDA',
            'BRK.B',
            'TSLA',
            'JPM',
            'JNJ',
            'UNH',
            'V',
            'PG',
            'HD',
            'PYPL',
            'DIS',
            'BAC',
            'ADBE',
            'MA',
            'CMCSA',
            'PFE',
            'CRM',
            'CSCO',
            'NFLX',
            'XOM',
            'VZ',
            'ABT',
            'TMO',
            'KO',
            'INTC',
            'PEP',
            'NKE',
            'ABBV',
            'ACN',
            'LLY',
            'WFC',
            'WMT',
            'DHR',
            'COST',
            'AVGO',
            'MRK',
            'T',
            'CVX',
            'MDT',
            'MCD',
            'TXN',
            'NEE',
            'LIN',
            'ORCL',
            'QCOM',
            'HON',
            'PM',
            'MS',
            'INTU',
            'BMY',
            'C',
            'UNP',
            'LOW',
            'GS',
            'UPS',
            'SBUX',
            'BLK',
            'AMD',
            'AMT',
            'RTX',
            'AMGN',
            'IBM',
            'ISRG',
            'NOW',
            'TGT',
            'MRNA',
            'AMAT',
            'BA',
            'DE',
            'CAT',
            'GE',
            'MMM',
            'SCHW',
            'CHTR',
            'CVS',
            'AXP',
            'SPGI',
            'ZTS',
            'PLD',
            'BKNG',
            'ANTM',
            'MO',
            'GILD',
            'TJX',
            'SYK',
            'ADP',
            'LMT',
            'MDLZ',
            'LRCX',
            'CB',
            'CCI',
            'MU',
            'PNC',
            'TMUS',
            'DUK',
            'FIS',
            'MMC',
            'EL',
            'USB',
            'COF',
            'TFC',
            'CSX',
            'COP',
            'EQIX',
            'EW',
            'SHW',
            'BDX',
            'CME',
            'CI',
            'REGN',
            'FISV',
            'SO',
            'ILMN',
            'ADSK',
            'ETN',
            'ITW',
            'HCA',
            'ICE',
            'CL',
            'FDX',
            'NSC',
            'AON',
            'BSX',
            'D',
            'ATVI',
            'EMR',
            'GM',
            'ADI',
            'NXPI',
            'MCO',
            'WM',
            'APD',
            'IDXX',
            'PGR',
            'ECL',
            'NOC',
            'JCI',
            'CMG',
            'DG',
            'A',
            'HUM',
            'BIIB',
            'VRTX',
            'MSCI',
            'KLAC',
            'F',
            'FCX',
            'ROP',
            'ALGN',
            'TWTR',
            'TEL',
            'TROW',
            'SNPS',
            'DXCM',
            'IQV',
            'EBAY',
            'LHX',
            'PSA',
            'GPN',
            'EXC',
            'TT',
            'DOW',
            'CARR',
            'AIG',
            'KMB',
            'MET',
            'GD',
            'NEM',
            'APH',
            'BK',
            'DLR',
            'CDNS',
            'INFO',
            'AEP',
            'SPG',
            'MCHP',
            'ROST',
            'ORLY',
            'FTNT',
            'PRU',
            'APTV',
            'SRE',
            'RMD',
            'MSI',
            'CTSH',
            'ALL',
            'EA',
            'TRV',
            'SYY',
            'DFS',
            'DD',
            'YUM',
            'EOG',
            'SLB',
            'PH',
            'SBAC',
            'PPG',
            'IFF',
            'OTIS',
            'BAX',
            'ROK',
            'XLNX',
            'MPC',
            'CNC',
            'XEL',
            'MTD',
            'STZ',
            'PAYX',
            'MNST',
            'AFL',
            'MAR',
            'WELL',
            'NUE',
            'CMI',
            'GIS',
            'HPQ',
            'HLT',
            'FRC',
            'CTAS',
            'AZO',
            'KR',
            'WBA',
            'PXD',
            'ADM',
            'WST',
            'SIVB',
            'AWK',
            'CTVA',
            'KEYS',
            'STT',
            'TDG',
            'PEG',
            'VRSK',
            'EFX',
            'FAST',
            'DHI',
            'AMP',
            'CBRE',
            'KMI',
            'MCK',
            'AME',
            'AVB',
            'ANSS',
            'ZBH',
            'SWK',
            'ES',
            'ZBRA',
            'BLL',
            'GLW',
            'PSX',
            'SWKS',
            'WEC',
            'CPRT',
            'LUV',
            'LH',
            'WMB',
            'AJG',
            'LEN',
            'EQR',
            'PCAR',
            'ARE',
            'MXIM',
            'CDW',
            'WLTW',
            'FITB',
            'SYF',
            'ODFL',
            'ETSY',
            'ALB',
            'GNRC',
            'VLO',
            'KSU',
            'O',
            'IT',
            'BBY',
            'WY',
            'LYB',
            'DAL',
            'RSG',
            'GRMN',
            'ED',
            'HSY',
            'WAT',
            'URI',
            'DOV',
            'VMC',
            'FTV',
            'NTRS',
            'VFC',
            'EXR',
            'VIAC',
            'XYL',
            'HIG',
            'MLM',
            'TRMB',
            'PAYC',
            'ENPH',
            'OKE',
            'KHC',
            'DTE',
            'CERN',
            'IP',
            'TSN',
            'ETR',
            'HBAN',
            'AEE',
            'PPL',
            'CTLT',
            'TSCO',
            'NDAQ',
            'DLTR',
            'COO',
            'EIX',
            'CRL',
            'MAA',
            'FLT',
            'ULTA',
            'VRSN',
            'MKC',
            'TDY',
            'QRVO',
            'FE',
            'CZR',
            'EXPD',
            'CLX',
            'STE',
            'ESS',
            'MPWR',
            'PKI',
            'KMX',
            'VTR',
            'EXPE',
            'ANET',
            'CHD',
            'OXY',
            'HOLX',
            'KEY',
            'DPZ',
            'AMCR',
            'RF',
            'BR',
            'HPE',
            'DGX',
            'TER',
            'DRI',
            'IR',
            'TYL',
            'POOL',
            'NTAP',
            'WDC',
            'PEAK',
            'GWW',
            'CCL',
            'DRE',
            'AVY',
            'CFG',
            'AKAM',
            'HES',
            'CMS',
            'CINF',
            'TTWO',
            'MKTX',
            'CE',
            'TFX',
            'BBWI',
            'MTB',
            'GPC',
            'RCL',
            'NVR',
            'HAL',
            'J',
            'VTRS',
            'MGM',
            'DVN',
            'ABC',
            'BIO',
            'IEX',
            'RJF',
            'STX',
            'PFG',
            'TXT',
            'BKR',
            'ABMD',
            'K',
            'CAG',
            'AES',
            'BXP',
            'MAS',
            'EVRG',
            'WAB',
            'UDR',
            'NLOK',
            'OMC',
            'EMN',
            'LNT',
            'CAH',
            'UAL',
            'CNP',
            'JBHT',
            'LKQ',
            'PHM',
            'IPG',
            'PKG',
            'LVS',
            'PWR',
            'INCY',
            'WHR',
            'FBHS',
            'PTC',
            'AAP',
            'SJM',
            'CBOE',
            'WRK',
            'JKHY',
            'XRAY',
            'FANG',
            'IRM',
            'LDOS',
            'KIM',
            'PNR',
            'AAL',
            'BF.B',
            'ALLE',
            'ATO',
            'L',
            'HWM',
            'HRL',
            'HAS',
            'LNC',
            'CTXS',
            'FOXA',
            'FFIV',
            'SNA',
            'LYV',
            'FMC',
            'CHRW',
            'UHS',
            'PENN',
            'LUMN',
            'MHK',
            'TPR',
            'RHI',
            'HST',
            'NRG',
            'MOS',
            'RE',
            'DISH',
            'HSIC',
            'REG',
            'WRB',
            'WYNN',
            'CMA',
            'BWA',
            'AIZ',
            'AOS',
            'JNPR',
            'NI',
            'NWL',
            'CF',
            'LW',
            'IVZ',
            'SEE',
            'DVA',
            'DXC',
            'NCLH',
            'ZION',
            'MRO',
            'GL',
            'TAP',
            'WU',
            'BEN',
            'NWSA',
            'PNW',
            'OGN',
            'ROL',
            'FRT',
            'HII',
            'CPB',
            'DISCK',
            'NLSN',
            'PVH',
            'ALK',
            'PBCT',
            'APA',
            'HBI',
            'VNO',
            'LEG',
            'IPGP',
            'COG',
            'RL',
            'GPS',
            'UNM',
            'PRGO',
            'FOX',
            'NOV',
            'DISCA',
            'UAA',
            'UA',
            'NWS']
            
        return sp500[:size]
