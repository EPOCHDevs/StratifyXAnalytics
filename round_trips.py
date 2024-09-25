import warnings

from utils import serialize_regular_series, serialize_series
import pyfolio as pf
import numpy as np
import pandas as pd
import scipy as sp


def extract_round_trips(returns, positions, round_trip):
    df = round_trip[['netReturn', 'openDateTime', 'closeDateTime', 'side', 'returnPercent', 'asset']]
    df.columns = ['pnl', 'open_dt', 'close_dt', 'long', 'rt_returns', 'symbol']
    df['open_dt'] = pd.to_datetime(df['open_dt'], utc=True)
    df['close_dt'] = pd.to_datetime(df['close_dt'], utc=True)
    df['long'] = df['long'] == 1
    df['symbol'] = df['symbol'].apply(lambda x: x['ticker'])
    df["duration"] = df["close_dt"].sub(df["open_dt"])

    portfolio_value = positions.sum(axis="columns") / (1 + returns)

    pv = pd.DataFrame(portfolio_value, columns=["portfolio_value"]).assign(
        date=portfolio_value.index
    )

    df["date"] = df.close_dt.apply(
        lambda x: x.replace(hour=0, minute=0, second=0)
    )

    tmp = (
        df.set_index("date")
        .join(pv.set_index("date"), lsuffix="_")
        .reset_index()
    )

    returns = tmp.pnl / tmp.portfolio_value
    df.loc[:, "returns"] = returns.values
    df = df.drop("date", axis="columns")

    return df


def get_pnl_attributions(trades):
    total_pnl = trades["pnl"].sum()
    pnl_attribution = trades.groupby("symbol")["pnl"].sum() / total_pnl
    pnl_attribution.name = ""

    pnl_attribution.index = pnl_attribution.index.map(pf.utils.format_asset)
    pnl_attribution.sort_values(
        inplace=False,
        ascending=False,
    )
    return pnl_attribution

async def round_trips_tear_sheet(round_trip, returns, positions, sector_mappings):
    if len(round_trip) >= 2:
        round_trip.index = round_trip.index.tz_localize('utc')
        trades = extract_round_trips(returns, positions, round_trip)

        round_trips =pf.round_trips.gen_round_trip_stats(trades)
        round_trips['duration'] =  round_trips['duration'].transform(lambda x: x.dt.total_seconds() * 1000)

        result = dict(stats={k : df.to_records().tolist() for k, df in round_trips.items()})
        result['stats']['columns'] = list(round_trips['duration'].columns)

        # Profitability (PnL / PnL total) per name

        result['pnl_attribution'] = serialize_regular_series(get_pnl_attributions(trades))
        result['pnl_attribution_by_sector'] = serialize_regular_series(get_pnl_attributions(pf.round_trips.apply_sector_mappings_to_round_trips(
            trades, sector_mappings
        )))

        disp_amount = 16
        symbols_sample = trades.symbol.unique()
        np.random.seed(1)
        sample = np.random.choice(
            trades.symbol.unique(),
            replace=False,
            size=min(disp_amount, len(symbols_sample)),
        )
        sample_round_trips = trades[trades.symbol.isin(sample)]

        symbol_idx = pd.Series(np.arange(len(sample)), index=sample)

        result['round_trip_lifetimes'] = {
            'symbols': [],
            'data': []
        }
        for symbol, sym_round_trips in sample_round_trips.groupby("symbol"):
            result['round_trip_lifetimes']['symbols'].append(symbol)
            for _, row in sym_round_trips.iterrows():
                y_ix = symbol_idx[symbol] + 0.05
                result['round_trip_lifetimes']['data'].append({
                    "x" : int(row["open_dt"].value // 1_000_000),
                    "x2": int(row["close_dt"].value  // 1_000_000),
                    "y": y_ix,
                    "color": 'red' if row.long else 'blue'
                })

        x = np.linspace(0, 1.0, 500)
        trades["profitable"] = trades.pnl > 0
        dist = sp.stats.beta(trades.profitable.sum(), (~trades.profitable).sum())

        result['profitability'] = {
            'stat': [[x_, y_] for x_, y_ in zip(x, dist.pdf(x))],
            'lower_perc': dist.ppf(0.025),
            'upper_perc': dist.ppf(0.975),
            'lower_plot': dist.ppf(0.001),
            'upper_plot': dist.ppf(0.999)
        }

        # Holding time in days
        result['holding_times'] = [x.days for x in trades["duration"]]
        trades["pnl"] = trades.pnl
        result['returns'] = serialize_series(trades.returns.dropna() * 100)

        return result

    warnings.warn(
        """Fewer than 5 round-trip trades made.
           Skipping round trip tearsheet.""",
        UserWarning,
    )
    return {}