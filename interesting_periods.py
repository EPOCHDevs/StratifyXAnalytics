import warnings

import pyfolio as pf
import empyrical as ep
import pandas as pd
from utils import serialize_series


async def interesting_periods(returns, factor_returns):

    rets_interesting = pf.timeseries.extract_interesting_date_ranges(returns, None)
    if rets_interesting:
        bmark_interesting = pf.timeseries.extract_interesting_date_ranges(
            factor_returns, None
        )
        periods = []
        for i, (name, rets_period) in enumerate(rets_interesting.items()):
            periods.append({
                'event': name,
                'strategy': serialize_series(ep.cum_returns(rets_period)),
                'benchmark': serialize_series(ep.cum_returns(bmark_interesting[name])),
            })

        return {
            "stat":  (pd.DataFrame(rets_interesting).describe().transpose().loc[:, ["mean", "min", "max"]] * 100).to_records().tolist(),
            "period": periods }
    else:
        warnings.warn(
            "Passed returns do not overlap with any" "interesting times.",
            UserWarning,
        )

    return []