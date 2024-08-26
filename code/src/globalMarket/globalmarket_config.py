""" 
-------------------------------------------------------------------------------------------
A JAX FRAMEWORK FOR MULTI-AGENT RESEARCH IN DECENTRALISED ENERGY MARKETS
-------------------------------------------------------------------------------------------
author: K. Emerson
Department of Computer Science
Faculty of Sciences and Bioengineering Sciences
Vrije Universiteit Brussel
-------------------------------------------------------------------------------------------
file contains all configuration dataclasses containing the parameters to create a 
GlobalMarket NamedTuple
"""

from dataclasses import dataclass
from abstract_dataclass import AbstractDataclass

@dataclass(frozen=True)
class GlobalMarketConfig(AbstractDataclass):
    """
    Configuration settings for the global market.

    Args:
        FEED_IN_CTE_COST (float): Constant cost to be paid each time energy is injected into the 
                                  global market.
        FEED_IN_CTE_WH_COST (float): Constant cost per Wh for each Wh that is injected 
                                     into the global market.
        FEED_IN_PERC_WH_COST (float): Percentage cost of the Wh price for each Wh that is injected 
                                      into the global market.
        TIME_OF_USE_CTE_COST (float): Constant cost to be paid each time energy is consumed from the 
                                      global market.
        TIME_OF_USE_CTE_WH_COST (float): Constant cost per Wh for each Wh that is consumed 
                                         from the global market.
        TIME_OF_USE_PERC_WH_COST (float): Percentage cost of the Wh price for each Wh that is consumed 
                                          from the global market.
        DAY_AHEAD_WINDOW_HOUR_TRIGGER (int): The hour at which the day-ahead window is updated
        DAY_AHEAD_WINDOW_SIZE (int): The size of the day-ahead window
    """
    FEED_IN_CTE_COST:float 
    FEED_IN_CTE_WH_COST:float 
    FEED_IN_PERC_WH_COST:float 

    TIME_OF_USE_CTE_COST:float 
    TIME_OF_USE_CTE_WH_COST:float 
    TIME_OF_USE_PERC_WH_COST:float 

    DAY_AHEAD_WINDOW_HOUR_TRIGGER:int
    DAY_AHEAD_WINDOW_SIZE:int
    


@dataclass(frozen=True)
class DummyDataGlobalMarketConfig(GlobalMarketConfig):
    """
    Configuration settings for the dummy global market which uses 
    a double sine wave to generate dummy day ahead prices 

    This dataclass extends the `GlobalMarketConfig` class and includes 
    additional parameters specifically for generating synthetic data 
    that simulates energy prices over time using a double sine wave 
    model.

    Args:
        NBR_OF_SAMPLES (int): number of samples to generate for the double sine wave.
        START_DATE (str): The starting date for the generated data in a string format (YYYY-MM-DD)
        HOURLY_PHASE_OFFSET (int): The phase offset in hours for the sine wave.
        MEAN_DAY_AHEAD_PRICE (float): The mean price serving as the baseline around which the sine wave oscillates.
        YEARLY_DAY_AHEAD_PRICE_AMPLITUDE (float): The amplitude of the yearly variation in day-ahead prices
        DAILY_DAY_AHEAD_PRICE_AMPLITUDE (float):  The amplitude of the daily variation in day-ahead prices 
        NOISE_AMPLITUDE (float): The amplitude of the noise added to the generated data 
        NOISE_SEED (int): The seed value for the random number generator used to create noise
    """
    NBR_OF_SAMPLES:int
    START_DATE:str

    HOURLY_PHASE_OFFSET:int
    MEAN_DAY_AHEAD_PRICE:float
    YEARLY_DAY_AHEAD_PRICE_AMPLITUDE:float
    DAILY_DAY_AHEAD_PRICE_AMPLTITUDE:float
    NOISE_AMPLITUDE:float
    NOISE_SEED:int
    


@dataclass(frozen=True)
class SyntheticDataGlobalMarketConfig(GlobalMarketConfig):
    """
    Configuration settings for the synthetic global market which uses 
    data stored in a CSV file which must:
        - use a ";" as delimiter
        - contain two columns namely "Datetime" and "Day_ahead_price_Wh"
        - contain a header with the above mentioned column names
    This dataclass extends the `GlobalMarketConfig` class.

    Args:
        FILEPATH(str): the filepath of the file containing the data
    """
    FILEPATH:str