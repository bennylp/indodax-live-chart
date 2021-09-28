#!/usr/bin/env python
from collections import defaultdict
from datetime import datetime
from doctest import master
import json
import os
import sys
import time

from Cython.Tempita._tempita import url
from dash import dcc
from dash import html
import dash
from dash.dependencies import Input, Output
import matplotlib
import requests

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


#import dash_bootstrap_components as dbc
if sys.platform=='win32':
    DIR = "C:/Users/bennylp/Desktop/GoogleDrive-Stosia/Work/Projects/coin-applet"
elif sys.platform=='linux':
    DIR = '/home/bennylp/Desktop/GoogleDrive-Stosia/Work/Projects/coin-applet'
else:
    assert False, "Unknown platform"


FILENAME = os.path.join(DIR, 'market.parquet')
HIST_FILENAME = os.path.join(DIR, 'history.parquet')
all_pairs = None


class Indodax:
    def get_all_quotes(self, dtime):
        try:
            req = requests.get('https://indodax.com/api/summaries', timeout=30)
            req.raise_for_status()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(str(e))
            return None
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
        try:
            req = requests.get('https://www.coinbase.com/api/v2/assets/prices/?base=IDR', timeout=30)
            req.raise_for_status()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(str(e))
            return None
        doc = req.json()
        quotes = []
        for data in doc['data']:
            pair = (data['base'], data['currency'])
            quote = {'exchange': self.__class__.__name__, 
                     'pair': f'{pair[0]}-{pair[1]}',
                     'dtime': dtime, 
                     'close': float(data['prices']['latest']), 
                     }
            quotes.append(quote)
        
        return pd.DataFrame(quotes).set_index(['exchange', 'pair', 'dtime'])

    def get_historical(self, pair_info):
        end = pd.Timestamp('2021-09-06 00:00:00')
        
        pair = (pair_info[0], pair_info[1])
        url = f'https://www.coinbase.com/api/v2/assets/prices/{pair_info[2]}?base=IDR'
        
        try:
            req = requests.get(url, timeout=30)
            req.raise_for_status()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(str(e))
            return None

        doc = req.json()
        assert doc['data']['base'] == pair[0]
        assert doc['data']['currency'] == pair[1]
        datas = []
        
        for field in ['month', 'year', 'all']: # 
            prices = doc['data']['prices'][field]['prices']
            for item in prices:
                data = {'exchange': self.__class__.__name__,
                        'pair': f'{pair[0]}-{pair[1]}', 
                        'dtime': pd.Timestamp.fromtimestamp(item[1]),
                        'close': float(item[0])}
                datas.append( data )
                
        indices = ['exchange', 'pair', 'dtime']
        df = pd.DataFrame(datas).sort_values(indices)
        df = df.drop_duplicates(indices)
        df = df.set_index(indices)
        return df
        

exchanges = {inst.__class__.__name__: inst for inst in [Indodax(), Coinbase()]}

def get_all_quotes(dtime):
    list_of_quotes = []
    for exchange in exchanges.values():
        quotes = exchange.get_all_quotes(dtime)
        if quotes is None:
            return None
        list_of_quotes.append(quotes)
    
    return pd.concat(list_of_quotes)
    

def update_quotes(dtime):
    if os.path.exists(FILENAME):
        df = pd.read_parquet(FILENAME)
    else:
        df = None
    
    update = get_all_quotes(dtime)
    if update is None:
        return
    if df is None:
        df = update
    else:
        df = df.append(update)
    
    df.to_parquet(FILENAME)
    return df
        

def wait_update():
    ts = time.time()
    
    while True:
        dtime = pd.Timestamp.fromtimestamp(ts)
        print(f"{dtime.strftime('%H:%M:%S')}..", end=' \r')
        if int(ts) % 60 == 0:
            break
        next_sec = round(ts, 0) + 1
        time.sleep(next_sec - ts)
        
        ts = time.time()
    print('')

    return int(ts)


def last_price(pairs):
    df = pd.read_parquet(FILENAME)
    df = df[ df.index.get_level_values('pair').isin(pairs) ]
    df = df.groupby(['exchange', 'pair']).last()
    print(df)
    
    
def poll():
    ts = wait_update()

    while True:
        update_quotes(pd.Timestamp.fromtimestamp(ts))
        ts = wait_update()


def update_historical():
    if os.path.exists(HIST_FILENAME):
        old_hist = pd.read_parquet(HIST_FILENAME)
        dfs = [old_hist]
        old_len = len(old_hist)
    else:
        dfs = []
        old_len = 0
    
    pair_infos = [('BTC', 'bitcoin'), 
                  ('ETH', 'ethereum'), 
                  ('ADA', 'cardano'), 
                  ('BNB', 'binance-coin'),
                  ('SOL', 'solana'),
                  ('USDT', 'tether'),
                  ('DOGE', 'dogecoin'),
                  ('USDC', 'usdc'),
                  ('XRP', 'xrp'),
                  ('DOT', 'polkadot'),
                  ('BCH', 'bitcoin-cash'),
                  ('LTC', 'litecoin'),
                  ('ALGO', 'algorand'),
                  ('FTT', 'ftx-token'),
                  ('MATIC', 'polygon'),
                  ('THETA', 'theta'),
                  ('WAVES', 'waves'),
                  ('XTZ', 'tezos'),
                  ('UNI', 'uniswap'),
                  ('LINK', 'chainlink'),
                  ('ATOM', 'cosmos'),
                  ('XLM', 'stellar'),
                  ('DAI', 'dai'),
                  ('TRX', 'tron'),
                  ('ETC', 'ethereum-classic'),
                  ('FIL', 'filecoin'),
                  ('AAVE', 'aave'),
                  ('ALT', 'alitas'),
                  ('AXS', 'axie-infinity'),
                  ('BSV', 'bitcoin-sv'),
                  ('BUSD', 'binance-usd'),
                  ('CAKE', 'pancakeswap'),
                  ]
    
    exchange = Coinbase()
    for pair in pair_infos:
        pair = (pair[0], 'IDR', pair[1])
        print(f'{pair[0]}-{pair[1]}..', end=' ')
        sys.stdout.flush()
        df = exchange.get_historical(pair)
        if df is not None and len(df):
            dfs.append(df)
        time.sleep(0.5)

    print('\nDone')
    df = pd.concat(dfs).reset_index()
    indices = ['exchange', 'pair', 'dtime']
    df = df.drop_duplicates(indices)
    df = df.sort_values(indices).set_index(indices)
    new_len = len(df)
    print(f'Added {new_len-old_len} new rows')
    if new_len-old_len > 0:
        diff = df.drop(index=old_hist.index)
        oldest = diff.index.get_level_values('dtime').min()
        newest = diff.index.get_level_values('dtime').max()
        diff_pairs = diff.index.get_level_values('pair').unique()
        print('Oldest time:', oldest)
        print('Newest time:', newest)
        print('Pairs:', diff_pairs)

        for pair, subset in diff.groupby(level='pair'):
            print(pair,':')
            print(f'{pair} ({len(subset)} rows):')
            print(subset.head())
            print('')

    df.to_parquet(HIST_FILENAME)
    return df
        
    
app = dash.Dash(__name__, #external_stylesheets=[dbc.themes.COSMO],
                title='Arbitrage',
                #suppress_callback_exceptions=True
                )

def serve():
    global all_pairs
    if all_pairs is None:
        df = pd.read_parquet(FILENAME)
        df = df.reset_index()[['exchange', 'pair']].drop_duplicates()
        d = defaultdict(set)
        for _, row in df.iterrows():
            d[ row['exchange'] ].add( row['pair'] )
        for exchange, supported_pairs in d.items():
            if all_pairs is None:
                all_pairs = set(supported_pairs)
            else:
                all_pairs = all_pairs.intersection(supported_pairs)
    
    pairs = ['ALGO-IDR', 'BTC-IDR', 'ETH-IDR', 'DOGE-IDR']
    pairs += [p for p in sorted(all_pairs) if p not in pairs]

    intervals = ['1 min', '3 min', '5 min', '10 min', '15 min', '30 min', '1h', '3h', '6h', '1d', '3d', '7d', '30d']
    children = html.Div([
        html.Div([
                html.Div(
                        # RadioItems
                        dcc.Dropdown(id='input_pair',
                                       options=[{'label': i, 'value': i} for i in pairs],
                                       value='ETH-IDR',
                                       className='auto',
                                       searchable=False,
                                       #labelStyle={"padding-right": "10px",
                                       #            },
                                       style={'width': '150px', 'vertical-align': 'middle'}
                        ),
                    style={ 'display': 'inline-block', '*display': 'inline', 'vertical-align': 'middle'}
                ),
                html.Div(
                        dcc.RadioItems(id='input_interval',
                                       options=[{'label': i, 'value': i} for i in intervals],
                                       value='5 min',
                                       #searchable=False,
                                       className='auto',
                                       labelStyle={"padding-right": "10px",
                                                   },
                        ),
                    style={ 'display': 'inline-block', '*display': 'inline', 'vertical-align': 'middle'}
                )
            ],  
            style={'vertical-align': 'middle'},   
        ),
        html.Div(id="the_graph"),
        dcc.Input(id="latest_price", type=_, value='', readOnly=True, disabled=True,),
        html.Div(id='blank-output'),
        dcc.Interval(id='interval-component', interval=60*1000, )        
    ],
    )

    #app.layout = dbc.Container(children)
    app.layout = html.Div(children)
    app.run_server(host='0.0.0.0', debug=False, port=8050)
    
    
@app.callback(
    Output("the_graph", "children"),
    Output('latest_price', 'value'),
    Input("input_interval", "value"),
    Input("input_pair", "value"),
    Input('interval-component', 'n_intervals')
)
def render_graph(input_interval, input_pair, n_intervals):
    global app
    children = []
    
    interval_min = pd.Timedelta(input_interval).total_seconds() // 60
    
    master = pd.read_parquet(FILENAME)

    if interval_min >= 6*60:
        historical = pd.read_parquet(HIST_FILENAME)
        master = pd.concat([master, historical])
        master = master[ master.index.get_level_values('pair')==input_pair ]
        master = master.reset_index()
        indices = ['exchange', 'pair', 'dtime']
        master = master.sort_values(indices).drop_duplicates(indices)
        master = master.set_index(indices)

    last_price = None

    pair = input_pair
    df = master
    df = df[ df.index.get_level_values('pair')==pair ]
    df = df.droplevel('pair')
    
    cs_datas = []
    cs_data = None
    
    max_rows = 120
    for exchange, df in df.groupby(level='exchange'):
        if exchange == 'Indodax':
            last_price = df.iloc[-1]['close']
        
        if interval_min != 1:
            df = df.resample(input_interval, level='dtime',closed='left', label='left')['close'] \
                   .agg(['first', 'last', 'max', 'min'])
            df = df.iloc[-max_rows:]
            df = df.rename(columns={'first': 'open', 'last': 'close', 'max': 'high', 'min': 'low'})
            df = df.reset_index()
            df['exchange'] = exchange
        else:
            df['open'] = df['close'].shift()
            df['high'] = df[['open', 'close']].max(axis=1)
            df['low'] = df[['open', 'close']].min(axis=1)
            df = df.reset_index().iloc[-max_rows:]
        
        df['dtime_diff'] = df['dtime'].diff().dt.total_seconds()
        df.loc[ df['dtime_diff'] > (interval_min+3)*60, 'open'] = np.NaN
        
        cs_data = go.Candlestick(x=df['dtime'],
                                 open=df['open'],
                                 high=df['high'],
                                 low=df['low'],
                                 close=df['close'], 
                                 opacity=0.9 if exchange=='Indodax' else 1,
                                 increasing_line_color= 'green' if exchange=='Indodax' else 'lightgray', 
                                 decreasing_line_color= 'red' if exchange=='Indodax' else 'darkgray',
                                 name=exchange)
        cs_datas.append(cs_data)
    
    latest_price = f'{input_pair} {int(last_price):,}' if last_price else '-err-'
    fig = go.Figure(data=cs_datas)
    fig.update_layout(xaxis_rangeslider_visible=False,
                      title=latest_price,)
    
    t = int(time.time())
    children = [dcc.Graph(figure=fig, 
                          config=dict(displayModeBar=False, ),
                          id=f'the_dcc_graph{t}',
                          clear_on_unhover=True,
                         ),
                html.Div('Latest is: ' + str(master.index.get_level_values('dtime')[-1]))
               ]
    
    return children, latest_price


app.clientside_callback(
    """
    function(latest_price) {
        document.title = latest_price;
    }
    """,
    Output('blank-output', 'children'),
    Input('latest_price', 'value')
)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: arbitrage.py (poll|serve|hist)')
        sys.exit(1)
        
    if sys.argv[1]=='poll':
        poll()
    elif sys.argv[1]=='plot':
        plot_pairs(['BTC-IDR', 'ETH-IDR'], 1)
    elif sys.argv[1]=='last':
        last_price(['BTC-IDR', 'ETH-IDR'])
    elif sys.argv[1]=='serve':
        serve()
    elif sys.argv[1]=='hist':
        update_historical()
    else:
        assert False

