#!/usr/bin/env python
from collections import defaultdict
from datetime import datetime
from doctest import master
import json
import os
import sys
import time

from dash import dcc
from dash import html
import dash
from dash.dependencies import Input, Output
import matplotlib
import requests

#import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


if sys.platform=='win32':
    DIR = "C:/Users/bennylp/Desktop/GoogleDrive-Stosia/Work/Projects/coin-applet"
elif sys.platform=='linux':
    DIR = '/home/bennylp/Desktop/GoogleDrive-Stosia/Work/Projects/coin-applet'
else:
    assert False, "Unknown platform"


FILENAME = os.path.join(DIR, 'market.parquet')
all_pairs = None


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
        try:
            req = requests.get('https://indodax.com/api/summaries')
        except:
            print('Error')
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
            req = requests.get('https://www.coinbase.com/api/v2/assets/prices/?base=IDR')
        except:
            print('Error')
            return None
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
            df = df[ df.index.get_level_values('exchange')==exchange]
            df = df.droplevel('exchange')
            
            if interval != 1:
                df = df.resample(f'{interval}min', closed='right', label='right').last()
                
            df = df.iloc[:120]
            
            x = df.index.get_level_values('dtime')
            y = df['close']
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
        update_quotes(pd.Timestamp.fromtimestamp(ts))
        ts = wait_update()


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
    
    pairs = ['BTC-IDR', 'ETH-IDR', 'USDC-IDR']
    pairs += [p for p in sorted(all_pairs) if p not in pairs]

    intervals = ['1 min', '3 min', '5 min', '10 min', '15 min', '30 min', '60 min', '1 d']
    children = html.Div([
        html.Div([
                html.Div(
                        # RadioItems
                        dcc.Dropdown(id='input_pair',
                                       options=[{'label': i, 'value': i} for i in pairs],
                                       value='ETH-IDR',
                                       className='auto',
                                       #searchable=False,
                                       #labelStyle={"padding-right": "10px",
                                       #            },
                                       style={'width': '150px', 'vertical-align': 'middle'}
                        ),
                    style={ 'display': 'inline-block', '*display': 'inline', 'vertical-align': 'middle'}
                ),
                html.Div(
                        dcc.RadioItems(id='input_interval',
                                       options=[{'label': i, 'value': i} for i in intervals],
                                       value='1 min',
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
    
    master = pd.read_parquet(FILENAME)

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
        
        interval_min = pd.Timedelta(input_interval).total_seconds() // 60
        if interval_min != 1:
            df = df.resample(input_interval, level='dtime',closed='right', label='right')['close'] \
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
    
    latest_price = f'{input_pair} {int(last_price):,}'
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
        print('Usage: arbitrage.py (poll|plot|last)')
        sys.exit(1)
        
    if sys.argv[1]=='poll':
        poll()
    elif sys.argv[1]=='plot':
        plot_pairs(['BTC-IDR', 'ETH-IDR'], 1)
    elif sys.argv[1]=='last':
        last_price(['BTC-IDR', 'ETH-IDR'])
    elif sys.argv[1]=='serve':
        serve()
    else:
        assert False
        
    
