from math import isinf
from operator import index

import pyfolio as pf
import pytz
import datetime

from utils import serialize_series


def get_transactions(round_trip):
    round_trip['filledQty'] *= round_trip['side']
    df = round_trip[['filledQty', 'filledPrice', 'asset']]
    df.columns = ['amount', 'price', 'symbol']
    df['symbol'] = df.symbol.apply(lambda x: x['ticker'])
    df.index = df.index.tz_localize('utc')
    return df


async def txn_tear_sheets(orders, positions, base_tf, bin_minutes, tz):
    txn_dict = {}
    transactions = get_transactions(orders)
    df_turnover = pf.txn.get_turnover(positions, transactions, 'AGB')

    txn_dict['df_turnover'] = serialize_series(df_turnover)
    txn_dict['df_turnover_mean'] = df_turnover.mean()
    txn_dict['df_turnover_by_month'] = serialize_series(df_turnover.resample("M").mean())

    # daily volume
    daily_txn = pf.txn.get_txn_vol(transactions)
    txn_dict['txn_shares'] = serialize_series(daily_txn.txn_shares)
    txn_dict['txn_shares_mean'] = daily_txn.txn_shares.mean()

    # daily_turnover_hist

    # daily txn_time_hist
    if  base_tf == '1T':
        txn_time = transactions

        txn_time.index = txn_time.index.tz_convert(pytz.timezone(tz))
        txn_time.index = txn_time.index.map(lambda x: x.hour * 60 + x.minute)
        txn_time["trade_value"] = (txn_time.amount * txn_time.price).abs()
        txn_time = (
            txn_time.groupby(level=0).sum(numeric_only=True).reindex(index=range(570, 961))
        )
        txn_time.index = (txn_time.index / bin_minutes).astype(int) * bin_minutes
        txn_time = txn_time.groupby(level=0).sum(numeric_only=True)

        txn_time["time_str"] = txn_time.index.map(
            lambda x: str(datetime.time(int(x / 60), x % 60))[:-3]
        )

        trade_value_sum = txn_time.trade_value.sum()
        txn_time.trade_value = txn_time.trade_value.fillna(0) / trade_value_sum

        txn_dict['txn_time_dist'] = txn_time[['time_str', 'trade_value']].to_records(index=False).tolist()
    else:
        txn_dict['txn_time_dist'] = []
    return txn_dict
