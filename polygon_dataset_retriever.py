from polygon import RESTClient
from pprint import pprint
import csv

client = RESTClient(api_key="XjabqthVtsWoU6Yen5gkeoG_dijvyceK")

class Aggregate:
    def __init__(self, open, close, high, low, volume, vwap, timestamp, transactions):
        self.open = open
        self.close = close
        self.high = high
        self.low = low
        self.volume = volume
        self.vwap = vwap
        self.timestamp = timestamp
        self.transactions = transactions
    def __repr__(self):
        return f"Aggregate(open={self.open}, close={self.close}, high={self.high}, low={self.low}, volume={self.volume}, vwap={self.vwap}, timestamp={self.timestamp}, transactions={self.transactions})"

# convert timestamp 
def convert_timestamp(timestamp):
    from datetime import datetime
    return datetime.fromtimestamp(timestamp / 1000)

# save to CSV
def save_aggs_to_csv(aggs, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Open', 'Close', 'High', 'Low', 'Volume', 'VWAP', 'Timestamp', 'Transactions'])
        for agg in aggs:
            writer.writerow([
                agg.open, agg.close, agg.high, agg.low, agg.volume,
                agg.vwap, agg.timestamp.isoformat(), agg.transactions
            ])

def generate_aggs_csv(from_date, to_date, date_expansion, ticker="X:BTCUSD"):
    aggs = []
    for a in client.list_aggs(ticker=ticker, multiplier=1, timespan="day", from_=from_date, to=date_expansion):
        aggs.append(Aggregate(
            open=a.open,
            close=a.close,
            high=a.high,
            low=a.low,
            volume=a.volume,
            vwap=a.vwap,
            timestamp=convert_timestamp(a.timestamp),
            transactions=a.transactions,
        ))

    save_aggs_to_csv(aggs, f'{from_date}_to_{date_expansion}_{ticker}.csv')


    #omit results before from_date
    from_date = from_date.replace("-", "")
    to_date = to_date.replace("-", "")
    aggs = [agg for agg in aggs if from_date <= agg.timestamp.strftime('%Y%m%d') <= to_date]
    #omit results after to_date
    aggs = [agg for agg in aggs if agg.timestamp.strftime('%Y%m%d') <= to_date]


    return aggs


generate_aggs_csv("2025-05-29", "2025-05-29", "2025-07-01")

#day = "2025-06-12"
#d1 = get_aggs_csv(day, day, "2025-06-13")
#save_aggs_to_csv(d1, f'{day}_aggs_btc.csv')
#pprint(d1)






