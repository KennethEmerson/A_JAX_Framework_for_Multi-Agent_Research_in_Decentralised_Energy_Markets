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
specific Prosumer NamedTuple type
"""
from dataclasses import dataclass

from abstract_dataclass import AbstractDataclass


@dataclass(frozen=True)
class ProsumerConfig(AbstractDataclass):
    """
    Abstract Configuration settings for a prosumer in the energy market.

    This dataclass encapsulates the parameters that define the characteristics 
    of a prosumer, which is an entity that can both consume and produce energy. 

    Args:
        AGENT_ID (str): A unique identifier for the prosumer agent
        MAX_BATTERY_CAPACITY_WH (float): maximum energy storage capacity of the prosumer's battery 
                                         measured in Wh.
        MAX_BATTERY_PEAK_CAPACITY_WH (float): maximum peak power capacity of the prosumer's battery 
                                              measured in Wh, indicating the highest rate at 
                                              which energy can be drawn from or supplied to the battery
                                              at any hour.
    """
    AGENT_ID:str
    MAX_BATTERY_CAPACITY_WH:float
    MAX_BATTERY_PEAK_CAPACITY_WH:float


   
@dataclass(frozen=True)
class DummyProsumerConfig(ProsumerConfig):
    """
    Configuration settings for generating dummy data for a prosumer agent.

    This dataclass extends the `ProsumerConfig` class and includes 
    additional parameters specifically for generating synthetic energy consumption data 
    using a double sine wave model.

    Args:
        NBR_OF_SAMPLES (int): total number of samples to generate for the double sine wave.
        START_DATE (str): The starting date for the generated data in format (YYYY-MM-DD)
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
    MEAN_ENERGY_CONSUMPTION_WH:float
    YEARLY_ENERGY_CONSUMPTION_WH_AMPLITUDE:float
    DAILY_ENERGY_CONSUMPTION_WH_AMPLITUDE:float
    NOISE_AMPLITUDE:float
    NOISE_SEED:int



@dataclass(frozen=True)
class SyntheticProsumerConfig(ProsumerConfig):
    """
    Configuration settings for the synthetic prosumer which uses 
    data stored in a predefined CSV file. These files are stored in the synthetic data folder
    and are linked to the agent based on the agents index (in the current implementation), so no file can be selected.
    This dataclass extends the `GlobalMarketConfig` class.

    Args:
        CONSUMER_ONLY(bool): If true only the synthetic energy consumption data will be loaded and the energy production will
                             remain zero in every timestep, thus simulating an market actor which does not have any means of 
                             energy production   
    """
    CONSUMER_ONLY:bool