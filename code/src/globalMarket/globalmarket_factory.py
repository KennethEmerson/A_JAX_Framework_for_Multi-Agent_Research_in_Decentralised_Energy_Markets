""" 
-------------------------------------------------------------------------------------------
A JAX FRAMEWORK FOR MULTI-AGENT RESEARCH IN DECENTRALISED ENERGY MARKETS
-------------------------------------------------------------------------------------------
author: K. Emerson
Department of Computer Science
Faculty of Sciences and Bioengineering Sciences
Vrije Universiteit Brussel
-------------------------------------------------------------------------------------------
File contains the factory functions to create a specific Global Market object based on 
the given 
configuration dataclasses.
"""


import math
from datetime import datetime, timedelta

import jax
import jax.numpy as jnp
import pandas as pd
from globalMarket.globalmarket import GlobalMarket
from globalMarket.globalmarket_config import (
    DummyDataGlobalMarketConfig,
    GlobalMarketConfig,
    SyntheticDataGlobalMarketConfig,
)


def _generate_dummy_sine_wave(
        nbr_of_samples:int,
        hourly_phase_offset:int,
        mean_day_ahead_price:float,
        yearly_day_ahead_amplitude:float,
        daily_day_ahead_amplitude:float,
        noise_amplitude:float,
        seed:float) -> jax.Array:
    """
    Generate a synthetic sine wave representing day-ahead energy prices 
    with added noise.

    This function creates a series of samples using a combination of cosine and sine 
    functions to model yearly and daily variations. The generated samples are 
    centered around a specified mean value and include random noise to 
    mimic real-world fluctuations.

    Args:
        nbr_of_samples (int): the total number of samples to generate.
        hourly_phase_offset (int): The phase offset in hours to adjust 
                                   the starting point of the samples. 
        mean_day_ahead_price (float): The baseline mean price around which 
                                      the samples oscillate.
        yearly_day_ahead_amplitude (float):  amplitude of the yearly variation 
        daily_day_ahead_amplitude (float): amplitude of the daily variation 
        noise_amplitude (float): amplitude of the noise added. 
        seed (float): The seed value for the random number generator used to create 
            noise

    Returns:
        jax.Array: An array of generated samples with length equal to nbr_of_samples
    """

    year_divider = 365*24/(2*math.pi)
    day_divider = 24/(2*math.pi)

    year_cos = jax.numpy.cos(
        jax.numpy.arange(hourly_phase_offset,nbr_of_samples+hourly_phase_offset) / year_divider ) * yearly_day_ahead_amplitude
    day_sin = jax.numpy.sin(
        jax.numpy.arange(hourly_phase_offset-6,nbr_of_samples+hourly_phase_offset-6) / day_divider) * daily_day_ahead_amplitude
    noise = jax.random.uniform(
        jax.random.PRNGKey(seed),(nbr_of_samples,),minval=-noise_amplitude,maxval=noise_amplitude)
    
    return mean_day_ahead_price + year_cos + day_sin + noise



def create_dummy_global_market(config:DummyDataGlobalMarketConfig) -> GlobalMarket:
    """ creates a DummyGlobalMarket object based on the provided configuration

    Args:
        config (DummyDataGlobalMarketConfig): the dataclass containing the global market parameters

    Returns:
        GlobalMarket: a GlobalMarket NamedTuple containing all Global market data
    """
    
    base = datetime.strptime(config.START_DATE, "%Y-%m-%d")
    date_list = pd.Series([base + timedelta(hours=x) for x in range(config.NBR_OF_SAMPLES)])

    day_ahead_prices_Wh = _generate_dummy_sine_wave(
                                    nbr_of_samples = config.NBR_OF_SAMPLES,
                                    hourly_phase_offset = config.HOURLY_PHASE_OFFSET,
                                    mean_day_ahead_price = config.MEAN_DAY_AHEAD_PRICE,
                                    yearly_day_ahead_amplitude = config.YEARLY_DAY_AHEAD_PRICE_AMPLITUDE,
                                    daily_day_ahead_amplitude = config.DAILY_DAY_AHEAD_PRICE_AMPLTITUDE,
                                    noise_amplitude = config.NOISE_AMPLITUDE,
                                    seed = config.NOISE_SEED                                    
                                    )

    global_market = GlobalMarket(
        nbr_of_samples = config.NBR_OF_SAMPLES,
        feed_in_cte_cost = config.FEED_IN_CTE_COST,
        feed_in_cte_Wh_cost = config.FEED_IN_CTE_WH_COST,
        feed_in_perc_Wh_cost = config.FEED_IN_PERC_WH_COST,
        time_of_use_cte_cost = config.TIME_OF_USE_CTE_COST,
        time_of_use_cte_Wh_cost = config.TIME_OF_USE_CTE_WH_COST,
        time_of_use_perc_Wh_cost = config.TIME_OF_USE_PERC_WH_COST,
        years = jnp.array(date_list.dt.isocalendar().year.to_numpy()),
        months = jnp.array(date_list.dt.month.to_numpy()-1), # -1 to get months from 0 to 11
        weeknumbers = jnp.array(date_list.dt.isocalendar().week.to_numpy()-1), # -1 to get weeknumbers from 0 to 52
        weekdays= jnp.array(date_list.dt.weekday.to_numpy()),
        hours= jnp.array(date_list.dt.hour.to_numpy()),
        day_ahead_prices_Wh = day_ahead_prices_Wh,
        day_ahead_window_hour_trigger = config.DAY_AHEAD_WINDOW_HOUR_TRIGGER,
        day_ahead_window_size = config.DAY_AHEAD_WINDOW_SIZE,
        )
    
    return global_market



def create_global_market_from_file(config:SyntheticDataGlobalMarketConfig) -> GlobalMarket:
    """ creates a GlobalMarket NamedTuple instance containing data extracted from an existing CSV file. 

    Args:
        config (SyntheticDataGlobalMarketConfig): the dataclass containing the global market parameters

    Returns:
        GlobalMarket: a GlobalMarket NamedTuple instance containing the loaded data from the file
    """
    filepath = config.FILEPATH
    global_market_data = pd.read_csv(filepath, sep=';', header=0, usecols=['Datetime','Day_ahead_price_Wh'])
    global_market_data["Datetime"] = pd.to_datetime(global_market_data["Datetime"])
    global_market_data["Weekday"] = global_market_data["Datetime"].dt.weekday
    global_market_data["WeekNumber"] = global_market_data["Datetime"].dt.isocalendar().week - 1  # -1 to get weeknumbers from 0 to 52
    global_market_data["Year"] = global_market_data["Datetime"].dt.isocalendar().year 
    global_market_data["Month"] = global_market_data["Datetime"].dt.month - 1  # -1 to get months from 0 to 11
    global_market_data["Hour"] = global_market_data["Datetime"].dt.hour
    global_market_data = global_market_data.drop(columns=["Datetime"])

    nbr_of_samples = len(global_market_data)
    global_market = GlobalMarket(
        nbr_of_samples=nbr_of_samples,
        feed_in_cte_cost = config.FEED_IN_CTE_COST,
        feed_in_cte_Wh_cost = config.FEED_IN_CTE_WH_COST,
        feed_in_perc_Wh_cost = config.FEED_IN_PERC_WH_COST,
        time_of_use_cte_cost = config.TIME_OF_USE_CTE_COST,
        time_of_use_cte_Wh_cost = config.TIME_OF_USE_CTE_WH_COST,
        time_of_use_perc_Wh_cost = config.TIME_OF_USE_PERC_WH_COST,
        
        years = jnp.array(global_market_data["Year"].to_numpy()),
        months = jnp.array(global_market_data["Month"].to_numpy()), 
        weeknumbers = jnp.array(global_market_data["WeekNumber"].to_numpy()), 
        weekdays= jnp.array(global_market_data["Weekday"].to_numpy()),
        hours= jnp.array(global_market_data["Hour"].to_numpy()),
        day_ahead_prices_Wh = jnp.array(global_market_data['Day_ahead_price_Wh']),
        day_ahead_window_hour_trigger = config.DAY_AHEAD_WINDOW_HOUR_TRIGGER,
        day_ahead_window_size = config.DAY_AHEAD_WINDOW_SIZE
        )
    
    return global_market


 
def create_global_market(global_market_config:GlobalMarketConfig) -> GlobalMarket:
    """
    Create a global market instance based on the provided configuration.

    Args:
        global_market_config (GlobalMarketConfig): instance of a GlobalMarketConfig child dataclass containing 
                                                   the parameters for creating the global market. 
    Returns:
        GlobalMarket: The created global market namedTuple.

    Raises:
        Exception: If the provided configuration dataclass is not recognized or 
            implemented.
    """
    if isinstance(global_market_config,SyntheticDataGlobalMarketConfig):
        return create_global_market_from_file(global_market_config) # type:ignore
    
    
    elif isinstance(global_market_config,DummyDataGlobalMarketConfig):
        return create_dummy_global_market(global_market_config) # type:ignore
    
    else:
        raise Exception("The global market config dataclass is not implemented in the global market factory")
