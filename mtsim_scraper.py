import pytz
from datetime import datetime, timedelta
from gym_mtsim import MtSimulator, OrderType, Timeframe, STOCKS_DATA_PATH

sim = MtSimulator(
    unit='USD',
    balance=10000.,
    hedge=False,
)

sim.download_data(
    symbols=['SPY'],
    time_range=(
        datetime(2021, 4, 14, tzinfo=pytz.UTC),
        datetime(2023, 4, 13, tzinfo=pytz.UTC)
    ),
    timeframe=Timeframe.H1
)
sim.save_symbols(STOCKS_DATA_PATH)