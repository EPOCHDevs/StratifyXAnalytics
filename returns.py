import pyfolio as pf
import empyrical as ep
import numpy as np
import pandas as pd
from pandas._libs import NaTType

from utils import serialize_series, serialize_regular_series


def convert_to_milliseconds(x):
    if isinstance(x, pd.Timestamp):
        if pd.isna(x):  # Check if the datetime is NaT
            return 0  # or float('nan') if you prefer NaN
        else:
            return int(x.timestamp() * 1000)  # Convert to milliseconds
    if isinstance(x, NaTType):
        return 0

    return x

async def returns_tear_sheet(returns,
                      factor_returns,
                      top_draw_downs,
                      rolling_vol_rolling_window,
                      rolling_sharpe_rolling_window):
    result = {'perf_stat': serialize_regular_series(pf.timeseries.perf_stats(returns, factor_returns).fillna(0))}

    drawdown_table = pf.timeseries.gen_drawdown_table(returns, top=top_draw_downs)
    drawdown_table = drawdown_table.applymap(convert_to_milliseconds)
    result['drawdown_table'] = drawdown_table.fillna(0).to_records(False).tolist()
    # rolling returns
    # Cumulative returns
    cum_rets = ep.cum_returns(returns, 1.0)
    result['cum_returns'] = serialize_series(cum_rets)
    result['cum_factor_returns'] = serialize_series(ep.cum_returns(factor_returns.loc[cum_rets.index], 1.0))

    # Cumulative returns volatility matched to benchmark
    bmark_vol = factor_returns.loc[returns.index].std()
    vol_returns = (returns / returns.std()) * bmark_vol
    result['cum_returns_vol'] = serialize_series(ep.cum_returns(vol_returns, 1.0))
    # add cum_factor_returns

    # Cumulative returns on logarithmic scale
    # same data, different scale

    # Returns
    # plot returns
    result['returns'] = serialize_series(returns)

    rb_1 = pf.timeseries.rolling_beta(
        returns, factor_returns, rolling_window=pf.APPROX_BDAYS_PER_MONTH * 6
    )
    rb_2 = pf.timeseries.rolling_beta(
        returns, factor_returns, rolling_window=pf.APPROX_BDAYS_PER_MONTH * 12
    )
    result['rolling_beta_1'] = serialize_series(rb_1)
    result['rolling_beta_2'] = serialize_series(rb_2)
    result['rolling_beta_mean'] = rb_1.mean()

    # Rolling volatility (6-month)
    rolling_vol_ts = pf.timeseries.rolling_volatility(returns, rolling_vol_rolling_window)
    result['rolling_vol_ts'] = serialize_series(rolling_vol_ts)
    result['rolling_vol_ts_mean'] = rolling_vol_ts.mean()
    result['rolling_vol_ts_factor'] = serialize_series(pf.timeseries.rolling_volatility(factor_returns, rolling_vol_rolling_window))

    # Rolling Sharpe-ratio(6 months)
    rolling_sharpe_ts = pf.timeseries.rolling_sharpe(returns, rolling_sharpe_rolling_window)
    result['rolling_sharpe_ts'] = serialize_series(rolling_sharpe_ts)
    result['rolling_vol_ts_mean'] = rolling_sharpe_ts.mean()
    result['rolling_sharpe_ts_factor'] = serialize_series(pf.timeseries.rolling_sharpe(factor_returns, rolling_sharpe_rolling_window))

    # "Top %i drawdown periods" % top
    # uses cum_returns and draw_down

    # Underwater plot
    df_cum_rets = ep.cum_returns(returns, starting_value=1.0)
    running_max = np.maximum.accumulate(df_cum_rets)
    result['underwater'] = serialize_series(-100 * ((running_max - df_cum_rets) / running_max))

    # monthly_ret
    monthly_ret_table = ep.aggregate_returns(returns, "monthly").unstack().round(3)
    monthly_ret_table.rename(
        columns={i: m for i, m in enumerate(pf.calendar.month_abbr)}, inplace=True
    )
    result['monthly_ret_table'] = (monthly_ret_table.fillna(0) * 100.0).to_records().tolist()

    # annual ret
    ann_ret_df = pd.DataFrame(ep.aggregate_returns(returns, "yearly"))

    result['ann_returns_mean'] = 100 * ann_ret_df.values.mean()
    result['ann_returns'] = serialize_regular_series((100 * ann_ret_df.sort_index(ascending=False))['returns'])

    result['monthly_returns_mean'] =  serialize_regular_series(100 * monthly_ret_table.mean())
    result['monthly_returns'] =  (100 * monthly_ret_table).to_records().tolist()

    # Return quantiles box plot
    result['return_quantile'] = {
        'is_weekly':  ep.aggregate_returns(returns, "weekly").values.tolist(),
        'is_monthly': ep.aggregate_returns(returns, "monthly").values.tolist(),
        'returns': returns.values.tolist()
    }

    # kurtosis
    # result['bootstrap']  = serialize_series(pf.timeseries.perf_stats_bootstrap(
    #     returns, factor_returns, return_stats=False
    # ).reset_index())

    return result
