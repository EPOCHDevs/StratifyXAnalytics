import msgpack
import numpy as np
import aiohttp
import logging
import pandas as pd
from io import StringIO
import pyfolio as pf
import yaml


async def async_get_campaign(server, campaign_id):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{server}/campaign/{campaign_id}") as response:
                response.raise_for_status()
                return await response.json()
    except aiohttp.ClientError as e:
        logging.error(f"Request failed: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    return None


async def async_parse_req(server, campaign_id, key):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{server}/{campaign_id}/{key}") as response:
                response.raise_for_status()
                msg_pack_data = msgpack.unpackb(await response.read())

                d = pd.DataFrame([{'t': row['t'], **row['data']} for row in msg_pack_data]).set_index('t')
                d.index = pd.to_datetime(d.index)
                return d
    except aiohttp.ClientError as e:
        logging.error(f"Request failed: {e}")
    except msgpack.exceptions.ExtraData as e:
        logging.error(f"MessagePack parsing failed: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")


async def parse_market_data_req(base_tf, server, campaign_id):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{server}/{campaign_id}/market_data") as response:
                response.raise_for_status()
                msg_pack_data = msgpack.unpackb(await response.read())

                data = {ele['data']['asset']['ticker']: pd.read_json(StringIO(ele['data']['dataBlob']),
                                                                     orient='records').set_index('t')
                        for ele in msg_pack_data if ele['data']['timeframe'] == base_tf}
                for key in data:
                    data[key].index = pd.to_datetime(data[key].index)
                return data
    except aiohttp.ClientError as e:
        logging.error(f"Request failed: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")


def get_daily_returns(equity, base_tf):
    if base_tf == '1D':
        return equity.pct_change().dropna()
    equity['date'] = equity.index.date
    x = equity.groupby('date').last().pct_change().dropna()
    x.index = pd.to_datetime(x.index)
    return x


def get_returns(account, base_tf, factor_returns):
    equity = pd.DataFrame({'equity': account['netLiquidationValue']}, index=account.index)
    returns = (get_daily_returns(equity, base_tf))['equity']
    returns.index = pd.to_datetime(returns.index, utc=True)
    returns.name = 'returns'
    if factor_returns is None:
        return returns
    return pf.utils.clip_returns_to_benchmark(returns, factor_returns)


async def async_get_benchmark_returns(period, benchmark):
    start = pd.to_datetime(period['start'])
    end = pd.to_datetime(period['end'])
    benchmark = pd.read_parquet(f's3://epoch-db/DailyBars/Stocks/{benchmark}.parquet.gzip', index='t')
    benchmark = benchmark[((benchmark.index >= start) & (benchmark.index <= end))]
    benchmark.index = pd.to_datetime(benchmark.index, utc=True)
    benchmark.name = 'benchmark_returns'
    return benchmark['c'].pct_change().dropna()


async def _get_asset_specs(server, asset_ids):
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{server}/reference/asset_specs/filter", params={'ids': asset_ids}) as response:
            response.raise_for_status()
            return await response.json()


def load_campaign_config(campaign):
    try:
        campaign_config = yaml.load(campaign['config'], yaml.CLoader)
        base_tf = '1D' if campaign_config['SimpleBacktest'] else '1T'
        return base_tf, campaign_config, ''
    except Exception as e:
        return None, None, str(e)


async def async_get_asset_specs(assets, stratifyx_server_url):
    try:
        asset_ids = set(assets.apply(lambda x: x['id']))
        asset_req_param = ','.join(asset_ids)
        return await _get_asset_specs(stratifyx_server_url, asset_req_param), ""
    except Exception as e:
        return None, str(e)


def serialize_series(values):
    values.replace([np.inf, -np.inf], np.nan, inplace=True)
    return [[int(date.value // 1_000_000), value] for date, value in values.dropna().items()]


def serialize_df(values):
    values.index = values.index.astype(int) // 1_000_000
    return values.reset_index().to_dict()


def serialize_regular_series(values):
    return [[i, value] for i, value in values.dropna().items()]
