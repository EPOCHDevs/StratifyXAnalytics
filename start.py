import typing
import warnings
warnings.filterwarnings('ignore')

import os
import logging
import atexit
import pyfolio as pf
import utils
import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from interesting_periods import interesting_periods
from positions import positions_tear_sheet
from returns import returns_tear_sheet
from transactions import txn_tear_sheets
from round_trips import round_trips_tear_sheet
import orjson

DEFAULT_ROUND_TRIPS = {
    "stats": {
        "columns": [],
        "summary": [],
        "pnl": [],
        "duration": [],
        "returns": [],
        "symbols": []
    },
    "pnl_attribution": [],
    "pnl_attribution_by_sector": [],
    "round_trip_lifetimes": {
        "symbols": [],
        "data": [],
    },
    "profitability": {
        "stat": [],
        "lower_perc": 0,
        "lower_plot": 0,
        "upper_perc": 0,
        "upper_plot": 0,
    },
    "holding_times": [],
    "pnl": [],
    "returns": [],
}


class ORJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content: typing.Any) -> bytes:
        return orjson.dumps(content, option=orjson.OPT_SERIALIZE_NUMPY)


app = FastAPI(default_response_class=ORJSONResponse)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


async def fetch_account(stratifyx_server_url, campaign_id):
    return await utils.async_parse_req(stratifyx_server_url, campaign_id, 'account')


async def fetch_factor_returns(period, benchmark):
    return await utils.async_get_benchmark_returns(period, benchmark)


async def fetch_positions(stratifyx_server_url, campaign_id):
    return await utils.async_parse_req(stratifyx_server_url, campaign_id, 'position')


async def fetch_orders(stratifyx_server_url, campaign_id):
    return await utils.async_parse_req(stratifyx_server_url, campaign_id, 'order')


async def fetch_round_trip(stratifyx_server_url, campaign_id):
    return await utils.async_parse_req(stratifyx_server_url, campaign_id, 'round_trip')


@app.get("/{campaign_id}/returns")
async def returns_and_periods(campaign_id: str, request: Request):
    logging.debug(f'Received request for {campaign_id}')

    benchmark = request.query_params.get('benchmark', 'SPY')
    top_draw_downs = int(request.query_params.get('top_dd', 5))
    roll_window = int(request.query_params.get('roll_window', 6))

    rolling_vol_rolling_window = pf.APPROX_BDAYS_PER_MONTH * roll_window
    rolling_sharpe_rolling_window = pf.APPROX_BDAYS_PER_MONTH * roll_window

    stratifyx_server_url = os.environ.get('STRATIFYX_SERVER_URL', "http://localhost:9001")
    logging.debug(f'STRATIFYX_SERVER_URL: {stratifyx_server_url}')

    campaign = await utils.async_get_campaign(stratifyx_server_url, campaign_id)
    if not campaign:
        logging.error('Campaign not found.')
        raise HTTPException(status_code=404, detail="Campaign not found")

    base_tf, campaign_config, error_msg = utils.load_campaign_config(campaign)
    if error_msg:
        logging.error(f"Unexpected error: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

    factor_returns = await fetch_factor_returns(campaign_config['Period'], benchmark)

    account = await fetch_account(stratifyx_server_url, campaign_id)
    daily_returns = utils.get_returns(account, base_tf, factor_returns)

    future_returns = returns_tear_sheet(
        daily_returns, factor_returns, top_draw_downs,
        rolling_vol_rolling_window, rolling_sharpe_rolling_window
    )

    futures_interesting_periods = await interesting_periods(daily_returns, factor_returns)

    return ORJSONResponse(content=dict(
        returns=await future_returns,
        interesting_periods=futures_interesting_periods
    ))


@app.get("/{campaign_id}/analytics")
async def analytics(campaign_id: str, request: Request):
    logging.debug(f'Received request for {campaign_id}')

    tz = request.query_params.get('tz', "America/New_York")
    bin_minutes = int(request.query_params.get('bin_minutes', 5))

    stratifyx_server_url = os.environ.get('STRATIFYX_SERVER_URL', "http://localhost:9001")
    logging.debug(f'STRATIFYX_SERVER_URL: {stratifyx_server_url}')

    campaign = await utils.async_get_campaign(stratifyx_server_url, campaign_id)
    if not campaign:
        logging.error('Campaign not found.')
        raise HTTPException(status_code=404, detail="Campaign not found")

    base_tf, campaign_config, error_msg = utils.load_campaign_config(campaign)
    if error_msg:
        logging.error(f"Unexpected error: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

    positions_res = await fetch_positions(stratifyx_server_url, campaign_id)
    account = await fetch_account(stratifyx_server_url, campaign_id)

    asset_specs, error_msg = await utils.async_get_asset_specs(positions_res['asset'], stratifyx_server_url)
    if error_msg:
        logging.error(f"Unexpected error: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

    cash = account['cashBalance'].to_frame('cash')
    position_result, position, sector_mappings = positions_tear_sheet(positions_res, asset_specs, cash, base_tf)

    orders = await fetch_orders(stratifyx_server_url, campaign_id)
    futures_txn = txn_tear_sheets(orders, position, base_tf, bin_minutes, tz)

    daily_returns = utils.get_returns(account, base_tf, None)
    round_trip = await fetch_round_trip(stratifyx_server_url, campaign_id)

    futures_round_trip = None
    if round_trip is not None:
        futures_round_trip = round_trips_tear_sheet(round_trip, daily_returns, position, sector_mappings)

    content = dict(
        position=position_result,
        txn=await futures_txn,
        round_trip=await futures_round_trip if round_trip is not None else DEFAULT_ROUND_TRIPS
    )
    return ORJSONResponse(content=content)



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=9006, log_level="info")
