from utils import serialize_series, serialize_regular_series
import pyfolio as pf
import pandas as pd
import numpy as np


def replace_asset(df):
    ticker = df['asset'].apply(lambda x: x['ticker'])
    df['asset'] = ticker

def make_positions(positions_res):
    positions_res['calculatedValue'] = positions_res['marketValue'] * positions_res['fxRate']
    positions_res.reset_index(inplace=True)
    replace_asset(positions_res)
    return positions_res.pivot_table(index='t', columns='asset', values='calculatedValue')

def positions_tear_sheet(positions_res, asset_specs, cash, base_tf):
    result = {}

    pos_no_cash = make_positions(positions_res)
    positions = pd.concat([cash, pos_no_cash], axis=1, sort=True)
    if base_tf == '1T':
        # Normalize the index to ensure all timestamps are at midnight
        positions.index = positions.index.normalize()

        # Group by the normalized date and take the last entry for each group
        positions = positions.groupby(positions.index).last()
    positions.index = positions.index.tz_localize('utc')

    positions_alloc = pf.pos.get_percent_alloc(positions)

    pos_no_cash = positions.drop("cash", axis=1)
    result['l_exp'] = serialize_series(pos_no_cash[pos_no_cash > 0].sum(axis=1) / positions.sum(axis=1))
    result['s_exp'] = serialize_series(pos_no_cash[pos_no_cash < 0].sum(axis=1) / positions.sum(axis=1))
    result['net_exp'] = serialize_series(pos_no_cash.sum(axis=1) / positions.sum(axis=1))

    df_top_long, df_top_short, df_top_abs = pf.pos.get_top_long_short_abs(positions_alloc)

    result['top_10_long'] = serialize_regular_series(df_top_long * 100)
    result['top_10_short'] = serialize_regular_series(df_top_short * 100)
    result['top_10'] = serialize_regular_series(df_top_abs * 100)

    palloc_over_time = positions_alloc[df_top_abs.index]
    palloc_over_time.index = palloc_over_time.index.astype(int) // 1_000_000
    result['portfolio_alloc_over_time'] = palloc_over_time.fillna(0).to_records().tolist()

    max_median_pos_concentration = pf.pos.get_max_median_position_concentration(positions_alloc)
    max_median_pos_concentration.index = max_median_pos_concentration.index.astype(int) // 1_000_000
    result['alloc_summary']  = max_median_pos_concentration.fillna(0).to_records().tolist()

    pos_no_cash = pos_no_cash.replace(0, np.nan)
    df_holdings = pos_no_cash.count(axis=1)
    result['df_holdings_by_month_mean'] = serialize_series(df_holdings.resample("1M").mean())
    result['df_holdings_by_month'] = serialize_series(df_holdings)
    result['df_holdings_mean'] = df_holdings.values.mean()

    df_longs = pos_no_cash[pos_no_cash > 0].count(axis=1)
    df_shorts = pos_no_cash[pos_no_cash < 0].count(axis=1)
    result['df_longs'] = serialize_series(df_longs)
    result['df_shorts'] = serialize_series(df_shorts)
    result['df_longs_max'] = df_longs.max()
    result['df_shorts_max'] = df_shorts.max()
    result['df_longs_min'] = df_longs.min()
    result['df_shorts_min'] = df_shorts.min()

    result['gross_leverage'] = serialize_series(pf.timeseries.gross_lev(positions))

    sector_mappings = { asset_spec['symbol']: asset_spec['industry'] for asset_spec in asset_specs}
    sector_exposures = pf.pos.get_sector_exposures(positions, sector_mappings)
    if len(sector_exposures.columns) > 1:
        sector_alloc = pf.pos.get_percent_alloc(sector_exposures)
        df = sector_alloc.drop("cash", axis="columns")
        for c in df.columns:
            # "Sector allocation over time"
            result['sector_alloc'] = { "sector": c, "data": serialize_series(df[c]) }

    return result, positions, sector_mappings