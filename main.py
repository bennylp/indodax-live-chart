#!/usr/bin/env python
import time

import requests

import pandas as pd


class Quote:
    def __init__(self, pair, h, l, c, value):
        self.pair = pair
        self.t = pd.Timestamp.now()
        self.h = float(h)
        self.l = float(l)
        self.c = float(c)
        self.value = value
        

class Indodax:
    def __init__(self):
        pass
    
    def p2x(self, pair):
        return f'{pair[0].lower()}_{pair[1].lower()}'
    
    def x2p(self, pair):
        pair = pair.split('_')
        return (pair[0].upper(), pair[1].upper())
    
    def get_quotes(self, pairs):
        req = requests.get('https://indodax.com/api/summaries')
        doc = req.json()
        quotes = []
        for pair in pairs:
            xpair = self.p2x(pair)
            data = doc['tickers'][xpair]
            quote = Quote(pair, data['high'], data['low'], data['last'], 
                          data['vol_idr'])
            quotes.append(quote)
        
        return quotes


def main(interval=600):
    pairs = [('BTC', 'IDR'), ('ETH', 'IDR')]
    exchange = Indodax()
    
    while True:
        quotes = exchange.get_quotes(pairs)
        now = pd.Timestamp.now()
        
        print(f'{now.strftime("%d-%b %H:%M")} ', end='')
        for quote in quotes:
            print(f' {quote.pair[0]}-{quote.pair[1]}: {quote.c:,} ', end='')
        print('')
        time.sleep(interval)


if __name__ == '__main__':
    main()
    