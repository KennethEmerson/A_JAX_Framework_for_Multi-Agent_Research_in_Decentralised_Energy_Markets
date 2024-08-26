""" 
-------------------------------------------------------------------------------------------
A JAX FRAMEWORK FOR MULTI-AGENT RESEARCH IN DECENTRALISED ENERGY MARKETS
-------------------------------------------------------------------------------------------
author: K. Emerson
Department of Computer Science
Faculty of Sciences and Bioengineering Sciences
Vrije Universiteit Brussel
-------------------------------------------------------------------------------------------
file contains the functions used to create Prosumer NamedTuple instances
"""
import math
import os
import sys
from datetime import datetime, timedelta

import jax
import jax.numpy as jnp
import pandas as pd
from prosumer.prosumer import Prosumer
from prosumer.prosumer_config import (
    DummyProsumerConfig,
    ProsumerConfig,
    SyntheticProsumerConfig,
)

sys.path.append(os.getcwd())
from data_synthetic_preprocess import consumption_helpers as synthcons  # noqa: E402
from data_synthetic_preprocess import production_helpers as synthprod  # noqa: E402


def _generate_dummy_sine_wave(
        nbr_of_samples:int,
        hourly_phase_offset:int,
        mean_energy_consumption_Wh:float,
        yearly_energy_consumption_Wh_amplitude:float,
        daily_energy_consumption_Wh_amplitude:float,
        noise_amplitude:float,
        seed:float) -> jax.Array:
    """
    Generate a synthetic sine wave representing energy consumption
    with added noise.

    This function creates a series of samples using a combination of cosine and sine 
    functions to model yearly and daily variations. The generated samples are 
    centered around a specified mean value and include random noise to 
    mimic real-world fluctuations.

    Args:
        nbr_of_samples (int): the total number of samples to generate.
        hourly_phase_offset (int): The phase offset in hours to adjust 
                                   the starting point of the samples. 
        mean_energy_consumption_Wh (float): The baseline mean price around which 
                                      the samples oscillate.
        yearly_energy_consumption_Wh_amplitude (float):  amplitude of the yearly variation 
        daily_energy_consumption_Wh_amplitude (float): amplitude of the daily variation 
        noise_amplitude (float): amplitude of the noise added. 
        seed (float): The seed value for the random number generator used to create 
            noise

    Returns:
        jax.Array: An array of generated samples with length equal to nbr_of_samples
    """
    year_divider = 365*24/(2*math.pi)
    day_divider = 24/(2*math.pi)

    year_cos = jax.numpy.cos(
        jax.numpy.arange(hourly_phase_offset,nbr_of_samples+hourly_phase_offset) / year_divider ) * yearly_energy_consumption_Wh_amplitude
    day_sin = jax.numpy.sin(
        jax.numpy.arange(hourly_phase_offset-6,nbr_of_samples+hourly_phase_offset-6) / day_divider) * daily_energy_consumption_Wh_amplitude
    noise = jax.random.uniform(
        jax.random.PRNGKey(seed),(nbr_of_samples,),minval=-noise_amplitude,maxval=noise_amplitude)
    
    return mean_energy_consumption_Wh + year_cos + day_sin + noise



def create_dummy_prosumer(agent_index:int,config:DummyProsumerConfig) -> Prosumer:
    """ creates a Prosumer NamedTuple containing the relevant dummy data of the Prosumer 
        for all timesteps

    Args:
        agent_index (int): the index of the agent as known in the environment
        config (DummyProsumerConfig): the dataclass containing the prosumer parameters

    Returns:
        Prosumer: a Prosumer NamedTuple instantiation based on the provided parameters
    """
    agent_id= config.AGENT_ID
    start_date = config.START_DATE
    nbr_of_samples = config.NBR_OF_SAMPLES
    max_battery_capacity_Wh = config.MAX_BATTERY_CAPACITY_WH
    max_battery_peak_Wh = config.MAX_BATTERY_PEAK_CAPACITY_WH

    hourly_phase_offset = config.HOURLY_PHASE_OFFSET
    mean_energy_consumption_Wh = config.MEAN_ENERGY_CONSUMPTION_WH
    yearly_energy_consumption_Wh_amplitude = config.YEARLY_ENERGY_CONSUMPTION_WH_AMPLITUDE
    daily_energy_consumption_Wh_amplitude = config.DAILY_ENERGY_CONSUMPTION_WH_AMPLITUDE
    noise_amplitude = config.NOISE_AMPLITUDE
    seed = config.NOISE_SEED         

    base = datetime.strptime(start_date, "%Y-%m-%d")
    date_list = pd.Series([base + timedelta(hours=x) for x in range(nbr_of_samples)])           

    energy_consumption_Wh = _generate_dummy_sine_wave(
                                    nbr_of_samples=nbr_of_samples,
                                    hourly_phase_offset=hourly_phase_offset,
                                    mean_energy_consumption_Wh=mean_energy_consumption_Wh,
                                    yearly_energy_consumption_Wh_amplitude=yearly_energy_consumption_Wh_amplitude,
                                    daily_energy_consumption_Wh_amplitude=daily_energy_consumption_Wh_amplitude,
                                    noise_amplitude=noise_amplitude,
                                    seed=seed                                
                                    )

    # if energy consumption in negative= find the max energy produced
    max_energy_production = jnp.min(energy_consumption_Wh)
    max_energy_production = jnp.where(max_energy_production<0,-max_energy_production,0)

    prosumer = Prosumer(
        agent_index = agent_index,
        agent_id= agent_id,
        nbr_of_samples = nbr_of_samples,
        max_energy_production_Wh = max_energy_production,
        max_battery_capacity_Wh =  jnp.full(nbr_of_samples,max_battery_capacity_Wh),
        max_battery_peak_Wh = jnp.full(nbr_of_samples,max_battery_peak_Wh),
        years = jnp.array(date_list.dt.isocalendar().year.to_numpy()),
        months = jnp.array(date_list.dt.month.to_numpy()-1),
        weeknumbers = jnp.array(date_list.dt.isocalendar().week.to_numpy()-1),
        weekdays= jnp.array(date_list.dt.weekday.to_numpy()),
        hours= jnp.array(date_list.dt.hour.to_numpy()),
        energy_consumption_Wh = energy_consumption_Wh,
    )
    return prosumer



def create_prosumer_from_synthetic_data(agent_index:int,config:SyntheticProsumerConfig) -> Prosumer:
    """ creates a Prosumer NamedTuple containing the relevant synthetic data of the Prosumer for all timesteps

    Args:
        agent_index (int): the index of the agent as known in the environment
        config (SyntheticProsumerConfig): the dataclass containing the prosumer parameters

    Returns:
        Prosumer: a Prosumer NamedTuple instantiation based on the provided parameters
    """
    agent_id= config.AGENT_ID
    consumer_only = config.CONSUMER_ONLY

    # load and transform all synthetic consumption data
    agent_consumptions = synthcons.load_agent_consumption(agent_id=agent_id)
    cons_data = agent_consumptions[agent_id]
    cons_data["Datetime"] = pd.to_datetime(cons_data["Datetime"])
    cons_data["Weekday"] = cons_data["Datetime"].dt.weekday
    cons_data["WeekNumber"] = cons_data["Datetime"].dt.isocalendar().week - 1
    cons_data["Year"] = cons_data["Datetime"].dt.isocalendar().year # guarantees that the year is conform the weeknumber
    cons_data["Month"] = cons_data["Datetime"].dt.month - 1
    cons_data["Hour"] = cons_data["Datetime"].dt.hour
    cons_data = cons_data.drop(columns=["Datetime"])

    # load and transform all synthetic production data
    agent_productions = synthprod.load_agent_production(agent_id=agent_id)
    prod_data = agent_productions[agent_id]
    prod_data["Datetime"] = pd.to_datetime(prod_data["Datetime"])
    prod_data["Weekday"] = prod_data["Datetime"].dt.weekday
    prod_data["WeekNumber"] = prod_data["Datetime"].dt.isocalendar().week - 1
    prod_data["Year"] = prod_data["Datetime"].dt.isocalendar().year # guarantees that the year is conform the weeknumber
    prod_data["Month"] = prod_data["Datetime"].dt.month - 1
    prod_data["Hour"] = prod_data["Datetime"].dt.hour
    prod_data = prod_data.drop(columns=["Datetime"])
    
    # merge both datasets
    data = pd.merge(prod_data,cons_data,on=["Year","WeekNumber","Month","Weekday","Hour"])
    nbr_of_samples = len(data)

    # calculate the actual energy consumption which is the difference between consumption and production. A negative energy consumption 
    # thus means that there is more energy produced than consumed.
    energy_consumption_Wh = jnp.array(
        (data["Energy_Consumed_Wh"].to_numpy())) if consumer_only else jnp.array(
                                        (data["Energy_Consumed_Wh"]-data["Energy_Produced_Wh"]).to_numpy())

    # if energy consumption in negative= find the max energy produced
    max_energy_production = jnp.min(energy_consumption_Wh)
    max_energy_production = jnp.where(max_energy_production<0,-max_energy_production,0)

    prosumer = Prosumer(
        agent_index = agent_index,
        agent_id= agent_id,
        nbr_of_samples = nbr_of_samples,
        max_energy_production_Wh = max_energy_production,
        max_battery_capacity_Wh =  jnp.full(nbr_of_samples,config.MAX_BATTERY_CAPACITY_WH),
        max_battery_peak_Wh = jnp.full(nbr_of_samples,config.MAX_BATTERY_PEAK_CAPACITY_WH),
        years = jnp.array(data["Year"].to_numpy()),
        months = jnp.array(data["Month"].to_numpy()),
        weeknumbers = jnp.array(data["WeekNumber"].to_numpy()),
        weekdays= jnp.array(data["Weekday"].to_numpy()),
        hours= jnp.array(data["Hour"].to_numpy()),
        energy_consumption_Wh = energy_consumption_Wh
    )
    return prosumer



def create_prosumer(agent_index:int,prosumer_config:ProsumerConfig) -> Prosumer:
    """
    Create a global market instance based on the provided configuration.

    Args:
        agent_index (int): 
        prosumer_config (ProsumerConfig): instance of a ProsumerConfig child dataclass containing 
                                          the parameters for creating the prosumer 
    Returns:
        Prosumer: The created Prosumer namedTuple

    Raises:
        Exception: If the provided configuration dataclass is not recognized or 
            implemented.
    """
    if isinstance(prosumer_config, SyntheticProsumerConfig):
        return create_prosumer_from_synthetic_data(agent_index,prosumer_config) 
    
    
    elif isinstance(prosumer_config,DummyProsumerConfig):
        return create_dummy_prosumer(agent_index,prosumer_config) 
    
    else:
        raise Exception(f"The prosumer config for {prosumer_config.AGENT_ID} does not comply with any of the available options")




