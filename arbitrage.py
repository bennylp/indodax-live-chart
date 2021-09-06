#!/usr/bin/env python
from collections import defaultdict
from datetime import datetime
from doctest import master
import json
import os
import sys
import time
import time

import matplotlib
import requests

import matplotlib.pyplot as plt
import pandas as pd


if sys.platform=='win32':
    DIR = "C:/Users/bennylp/Desktop/GoogleDrive-Stosia/Work/Projects/coin-applet"
elif sys.platorm=='linux':
    DIR = '/home/bennylp/Desktop/GoogleDrive-Stosia/Work/Projects/coin-applet'
else:
    assert False, "Unknown platform"


FILENAME = os.path.join(DIR, 'market.parquet')


class Quote:
    def __init__(self, xchange, pair, t, h, l, c, value):
        self.xchange = xchange
        self.pair = pair
        self.t = t
        self.h = float(h)
        self.l = float(l)
        self.c = float(c)
        self.value = value
        

class Indodax:
    def get_all_quotes(self, dtime):
        req = requests.get('https://indodax.com/api/summaries')
        doc = req.json()
        quotes = []
        for xpair, data in doc['tickers'].items():
            pair = [p.upper() for p in xpair.split('_')]
            quote = {'exchange': self.__class__.__name__, 
                     'pair': f'{pair[0]}-{pair[1]}',
                     'dtime': dtime, 
                     #'high': float(data['high']), 
                     #'low': float(data['low']), 
                     'close': float(data['last']), 
                     #'value': float(data[f'vol_{pair[1].lower()}'])
                     }
            quotes.append(quote)
        
        return pd.DataFrame(quotes).set_index(['exchange', 'pair', 'dtime'])


class Coinbase:
    def get_all_quotes(self, dtime):
        req = requests.get('https://www.coinbase.com/api/v2/assets/prices/?base=IDR')
        doc = req.json()
        quotes = []
        for data in doc['data']:
            pair = (data['base'], data['currency'])
            quote = {'exchange': self.__class__.__name__, 
                     'pair': f'{pair[0]}-{pair[1]}',
                     'dtime': dtime, 
                     #'high': float(data['high']), 
                     #'low': float(data['low']), 
                     'close': float(data['prices']['latest']), 
                     #'value': float(data[f'vol_{pair[1].lower()}'])
                     }
            quotes.append(quote)
        
        return pd.DataFrame(quotes).set_index(['exchange', 'pair', 'dtime'])


exchanges = {inst.__class__.__name__: inst for inst in [Indodax(), Coinbase()]}


def get_all_quotes(dtime):
    list_of_quotes = []
    for exchange in exchanges.values():
        quotes = exchange.get_all_quotes(dtime)
        list_of_quotes.append(quotes)
    
    return pd.concat(list_of_quotes)
    

def update_quotes(dtime):
    if os.path.exists(FILENAME):
        df = pd.read_parquet(FILENAME)
    else:
        df = None
    
    update = get_all_quotes(dtime)
    if df is None:
        df = update
    else:
        df = df.append(update)
    
    df.to_parquet(FILENAME)
    return df
        

def wait_update():
    ts = time.time()
    
    while int(ts) % 60:
        dtime = pd.Timestamp.fromtimestamp(ts)
        print(f"{dtime.strftime('%H:%M:%S')}..", end=' \r')
        next_sec = round(ts, 0) + 1
        time.sleep(next_sec - ts)
        
        ts = time.time()
    print('')

    return int(ts)


def plot_pairs(pairs, interval=1):
    nrows = len(pairs)
    fig, axs = plt.subplots(nrows, 1, figsize=(12, 3*nrows))
    master = pd.read_parquet(FILENAME)
    
    for pair, ax in zip(pairs, axs):
        df = master
        df = df[ df.index.get_level_values('pair')==pair ]
        df = df.droplevel('pair')
        exchanges = sorted(df.index.get_level_values('exchange').unique())
        for exchange in exchanges:
            df1 = df[ df.index.get_level_values('exchange')==exchange]
            df1 = df1.droplevel('exchange')
            
            if interval != 1:
                df1 = df1.resample(f'{interval}min', closed='right', label='right').last()
                
            df1 = df1.iloc[:120]
            
            x = df1.index.get_level_values('dtime')
            y = df1['close']
            ax.plot(x, y, label=f'{exchange}')
            
        ax.grid()
        ax.legend()
        ax.set_title(pair)
        ax.get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    fig.tight_layout()
    plt.savefig(f'arbitrage{interval:02d}.png')
    #plt.show()


def last_price(pairs):
    df = pd.read_parquet(FILENAME)
    df = df[ df.index.get_level_values('pair').isin(pairs) ]
    df = df.groupby(['exchange', 'pair']).last()
    print(df)
    
    
def poll():
    pairs = ['BTC-IDR', 'ETH-IDR']
    ts = wait_update()

    while True:
        df = update_quotes(pd.Timestamp.fromtimestamp(ts))
        #plot_pairs(pairs, 1)
        #plot_pairs(pairs, 5)
        #plot_pairs(pairs, 15)
        #plot_pairs(pairs, 60)
        ts = wait_update()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: arbitrage.py (poll|plot|last)')
        sys.exit(1)
        
    if sys.argv[1]=='poll':
        poll()
    elif sys.argv[1]=='plot':
        plot_pairs(['BTC-IDR', 'ETH-IDR'], 1)
    elif sys.argv[1]=='last':
        last_price(['BTC-IDR', 'ETH-IDR'])
    else:
        assert False
        
    