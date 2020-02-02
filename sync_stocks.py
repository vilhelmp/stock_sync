from yahoofinancials import YahooFinancials
import pandas as pd
from lxml import html
import requests as r
from time import sleep
import json
import pystore
import numpy as np
import datetime as dt
import sys
import holidays
import logging
import os
import matplotlib.dates as mdates

# holidays will show Sundays and public holidays as holidays
# NOT Saturdays
swe_holidays = holidays.Sweden()

# move to config file
STOCKMARKET_CLOSES = dt.time(16,00,00)
# in minutes
WAITFORMARKET = 30              
# attempt to sync this far back
START_DATE = dt.date(2000,1,1)

STOREPATH = 'path/to/pystore'

LOGPATH = '/path/to/logs/'

SYNC_STORES = ['NASDAQ-STO',
               'NASDAQ-FirstNorthPremier', 
               'NASDAQ-FirstNorth',
               'INDEXES']

SYNC_URLS = ['http://www.nasdaqomxnordic.com/shares/listed-companies/stockholm',
             'http://www.nasdaqomxnordic.com/shares/listed-companies/first-north-premier',
             'http://www.nasdaqomxnordic.com/shares/listed-companies/first-north',
             '']

listofindexes = [
            {'index': '^OMX',
             'name': 'Stockholm OMX',
             },
            {'index': '^OMXSPI',
             'name': 'Stockholm all-share',
             },
            {'index': '^OMXSBGI',
             'name': 'Stockhoolm Generalindex',
             },
            {'index': '^SX2000PI',
             'name': 'Stockholm Industrials Index',
             },
            ]


# Some stocks is some kind of weird type?
# usually an existing stock with some added name 
# for exampel both TAGM-B.ST and SPEC.ST works, 
# but TAGM-BTA-B.ST and SPEC-TO3.ST doesn't
# 
tickers_to_skip = []


logging.basicConfig(filename=os.path.join(LOGPATH,'stock_syncing-{0}.log'.format(dt.datetime.now().strftime('%y%m%d-%H%M%S')) 
                                          ), 
                    filemode='w', format='%(asctime)s-%(levelname)s: %(message)s',
                    level=logging.INFO)

date_today = dt.date.today()
datetime_today = dt.datetime.today()
timeconstraint = dt.datetime.combine(date_today,STOCKMARKET_CLOSES) + dt.timedelta(minutes=WAITFORMARKET)
if (datetime_today.time() <= timeconstraint.time()):
    logging.warning( 'Warning: Stock market closes around *{0}* during weekdays.'.format(STOCKMARKET_CLOSES) )




# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()


def get_start_date(collection, key):
    # try to get the last date stored, if there
    try:
        last_date = collection.item(key).tail(1).index[0]
        # 
        # If you are not storing Timestamps
        # you have to run last_date = dt.date( *[int(i) for i in last_date.split('-')] )
        # first to get last date into a datetime date object.
        last_date = last_date.date()
        # starting date is last date + one day
        sdate = last_date + dt.timedelta(days=1)
    except(FileNotFoundError):
        # if the ticker doesn't exist, start syncing the full thing
        sdate = START_DATE
    return sdate

def write_to_database(collection, key, data_frame, metadata=dict(source='unknown')):

    try:
        collection.write(key, data_frame, 
                            metadata=metadata
                            )
    except ValueError as e: 
        if 'exist' in str(e).lower(): 
            try:
                # test to append it, if this doesn't work 
                # then there's something else wrong.
                collection.append( key, data_frame )
            except Exception as e:
                logging.error( str(e) )
                raise StandardError
        return True
    

def get_dates_to_sync(collection, key):
    
    date_today = dt.date.today()
    datetime_today = dt.datetime.today()
    # get last date synced, if any, else use global param START_DATE
    sdate = get_start_date(collection, key)
    
    timeconstraint = dt.datetime.combine(date_today,STOCKMARKET_CLOSES) + dt.timedelta(minutes=WAITFORMARKET)
    # always sync until today unless
    # the market hasn't closed yet, sync until yesterday
    if (datetime_today.time() <= timeconstraint.time()):
        edate = date_today - dt.timedelta(days=1)        
    else:
        edate = date_today
    # check that last sync date is not a (bank) holiday
    # NOTE that this will not take care of Saturdays
    while edate in swe_holidays:
        if edate in swe_holidays:
            logging.info('Last sync day is a public holiday ({0}), testing the day before...'.format(edate.strftime('%Y-%m-%d')) )
        edate -= dt.timedelta(days=1)
    # number of days to sync
    daystosync = (edate-sdate).days + 1
    # if number of days to sync is negative it means last sync date is today.
    if daystosync <= 0:
        logging.info('*No data (dates) to sync for ticker: {0}*'.format(ticker))
        return False
    # If start is on saturday/sunday, and we are trying to sync sat, or sat+sun
    # we have to wait.
    if sdate.weekday() > 4 and daystosync < 2:
        logging.info('*Stock market not open on weekends!*')
        return False

    if sdate>edate:
        logging.info('{0}: No data to sync.'.format(key))
        return False
    # format input for Yahoo
    sdate_str = sdate.strftime('%Y-%m-%d')
    edate_str = edate.strftime('%Y-%m-%d')
    return (sdate,edate),(sdate_str,edate_str)


def hsp_empty(hsp, ticker):
    
    if not hsp[ticker] or 'prices' not in hsp[ticker].keys():
        # if hsp[ticker] empty, or if it doesn't list prices
        return True
    else: # if it has 'prices' but it's empty
        try:
            if not hsp[ticker]['prices']:
                return True
            else:
                return False
        except Exception as e:
            #~ print(e)
            logging.error(str(e))
            raise Exception

# prepare local storage/database
pystore.set_path(STOREPATH)

for sync_store, sync_url in zip(SYNC_STORES, SYNC_URLS):
    logging.info('###############  Syncing {0} ###############'.format(sync_store) )
    
    store = pystore.store(sync_store)
    
    # the various "tables" to store things in
    logging.info('Syncing {0}'.format(sync_store) )
    collection_prices = store.collection('prices')
    if sync_store == 'INDEXES':
        for ind,n in zip(listofindexes, range(len(listofindexes))):
            printProgressBar(n,len(listofindexes), prefix='Estimated ', suffix = 'of *indexes* synced', length=30)
            ticker = ind['index']
            name = ind['name']
            yahoo_financials = YahooFinancials( ticker )
            dates_to_sync = get_dates_to_sync(collection_prices, ticker) 
            if not dates_to_sync:
                continue
            else:
                (sdate, edate), (sdate_str, edate_str) = dates_to_sync
            # get historical prices
            hsp = yahoo_financials.get_historical_price_data(sdate_str, edate_str, 'daily')
            # Create DataFrame and write to store
            # NOTE: 'date' and 'volume' should ideally be 'int64',
            #       but it doesn't support NaN values. Pandas >0.24 
            #       has an update using 'Int64' (not uppercase), dtype
            #       BUT parquet file storage doesn't support this (yet?).
            data_types = {'date'            :'float',   
                          'formatted_date'  :'str',
                          'open'            :'float',
                          'high'            :'float',
                          'low'             :'float',
                          'close'           :'float',
                          'adjclose'        :'float',
                          'volume'          :'float'}
            if hsp_empty(hsp,ticker):
                empty_ticker_counter += 1
                empty_tickers.append(ticker)
                logging.info('   ***{0}***   '.format(ticker))
                logging.info('No data in this reply...')
                logging.info('If this happens often, perhaps wait a couple of hours to sync.')
                continue
            price_data = pd.DataFrame( hsp[ticker]['prices'] )
            price_data = price_data.astype( dtype=data_types )
            if price_data.empty:
                logging.info('   ***{0}***   '.format(ticker))
                logging.info('No data in this reply...')
                logging.info('If this happens often, perhaps wait a couple of hours to sync.')
                continue
            price_data['formatted_date'] = pd.to_datetime(price_data['formatted_date'])
            price_data.set_index('formatted_date', inplace=True)
            if not price_data['open'].any(): # i.e. if any is not false=nan=none
                logging.info('   ***{0}***   '.format(ticker))
                logging.info('Data is empty, try later.')
                continue
            write_to_database(collection_prices, 
                                ticker, 
                                price_data, 
                                metadata={'source': 'Yahoo', 
                                        'name': str(name), 
                                        }
                                )
            logging.info('   ***{0}***   '.format(ticker))
            logging.info('Synced from {0} to {1}'.format(sdate_str,edate_str))
            price_data = []
            w8 = 1.5/np.random.randint(1,high=20)
            sleep(w8)


    else:

        ################## FIRST SYNC
        # 1. Get stock list

        # first get stock symbols, anonymously through TOR!
        page = r.get(sync_url, 
            )
        tree = html.fromstring(page.content)
        page.close
        # fix links
        tree.make_links_absolute('http://www.nasdaqomxnordic.com')
        # get table rows with stocks
        trs = tree.xpath('//tbody//tr')
        # get the data
        data = pd.DataFrame(
                [[j.text_content() for j in i.getchildren()[:-1]] for i in trs],
                columns = ['name', 'symbol', 'currency', 'isin', 'sector', 'icb']
                )
        
        # 2. Data gathering

        ## 2a. Prepare to fetch stock data

        tickers = ["-".join(i.split(" "))+".ST" for i in data['symbol'].values]
        sectors = data['sector'].values
        names = data['name'].values
        isins = data['isin'].values
        currencies = data['currency'].values

        
        ## 2b. Get data for each stock, one at a time, since we might have 
        #      different sync intervals for every stock, and minimize the 
        #      payload (fly under the radar).
        empty_tickers = []
        empty_ticker_counter = 0
        for ticker, sector, name, isin, currency,n in zip(tickers, sectors, names, isins, currencies, range(len(tickers)) ):
            logging.info('   ***{0}***   '.format(ticker))
            printProgressBar(n, len(tickers), prefix='Estimated ', suffix = 'of {0} *stocks* synced'.format(sync_store), length=30)
            # lets try to find the ticker first,
            # if it doesn't work, catch the exception (FileNotFoundError)
            # and proceed
            yahoo_financials = YahooFinancials(ticker)
            
            ### SYNC STOCK PRICES
            # Fist check the dates to be synced
            dates_to_sync = get_dates_to_sync(collection_prices, ticker) 
            if not dates_to_sync:
                continue
            else:
                (sdate, edate), (sdate_str, edate_str) = dates_to_sync
            # get historical prices
            hsp = yahoo_financials.get_historical_price_data(sdate_str, edate_str, 'daily')
            # Create DataFrame and write to store
            # NOTE: 'date' and 'volume' should ideally be 'int64',
            #       but it doesn't support NaN values. Pandas >0.24 
            #       has an update using 'Int64' (not uppercase), dtype
            #       BUT parquet file storage doesn't support this (yet?).
            data_types = {'date'            :'float',   
                          'formatted_date'  :'str',
                          'open'            :'float',
                          'high'            :'float',
                          'low'             :'float',
                          'close'           :'float',
                          'adjclose'        :'float',
                          'volume'          :'float'}
            if hsp_empty(hsp,ticker):
                empty_ticker_counter += 1
                empty_tickers.append(ticker)
                logging.info('No data in this reply...')
                logging.info('If this happens often, perhaps wait a couple of hours to sync.')
                continue
            price_data = pd.DataFrame( hsp[ticker]['prices'] )
            price_data = price_data.astype( dtype=data_types )
            if price_data.empty:
                logging.info('No data in this reply...')
                logging.info('If this happens often, perhaps wait a couple of hours to sync.')
                continue
            price_data['formatted_date'] = pd.to_datetime(price_data['formatted_date'])
            price_data.set_index('formatted_date', inplace=True)
            if not price_data['open'].any(): # i.e. if any is not false=nan=none
                logging.info('Data is empty, try later.')
                continue
            write_to_database(collection_prices, 
                                ticker, 
                                price_data, 
                                metadata={'source': 'Yahoo', 
                                        'sector':str(sector), 
                                        'name': str(name), 
                                        'isin': isin,
                                        'currency': str(currency),
                                        }
                                )
            logging.info('   ***{0}***   '.format(ticker))
            logging.info('Synced from {0} to {1}'.format(sdate_str,edate_str))
            price_data = []
            ### SYNC financial statements
            # Currently deactivated!
            #all_statement_data_qt 
            #~ asd_qt = yahoo_financials.get_financial_stmts('quarterly', ['income', 'cash', 'balance'])
            #~ asd_data = []
            #~ # get all financial info in order
            #~ key_order = list(asd_qt['incomeStatementHistoryQuarterly'][ticker][0][list( asd_qt['incomeStatementHistoryQuarterly'][ticker][0].keys())[0]] )
            #~ for line in asd_qt['incomeStatementHistoryQuarterly'][ticker]:
                #~ d = list( line.keys() )[0]
                #~ temp_d = line[d]
                #~ temp_d = [temp_d[i] for i in key_order]
                #~ #d.concatenate(temp_d)
                #~ asd_data.append( [d] + temp_d )
            #~ asd_data = pd.DataFrame(asd_data,columns=['formatted_date']+key_order )
            #~ asd_data['formatted_date'] = pd.to_datetime(asd_data['formatted_date'])
            #~ asd_data.set_index('formatted_date', inplace=True)
            #~ ### SYNC earnings data
            #~ #eri_earnings_data 
            #~ eed = yahoo_financials.get_stock_earnings_data()
            
            
            #~ ### SYNC  eri net income
            #~ #eri_net_income 
            #~ eni = yahoo_financials.get_net_income()  
            # sleep for a random time 0.25~5 seconds
            w8 = 1.5/np.random.randint(1,high=20)
            sleep(w8)
        logging.debug('Couldn\'t sync these tickers: {0}'.format(empty_tickers) )



