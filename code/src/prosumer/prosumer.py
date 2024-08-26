""" 
-------------------------------------------------------------------------------------------
A JAX FRAMEWORK FOR MULTI-AGENT RESEARCH IN DECENTRALISED ENERGY MARKETS
-------------------------------------------------------------------------------------------
author: K. Emerson
Department of Computer Science
Faculty of Sciences and Bioengineering Sciences
Vrije Universiteit Brussel
-------------------------------------------------------------------------------------------
file contains the Prosumer NamedTuple definition and functions to manipulate 
the Prosumer instances
"""
from typing import NamedTuple, Tuple

import jax


class Prosumer(NamedTuple):
    """
    NamedTuple representing the data and parameters for a prosumer.

    Args:
        agent_id (str): the agent id
        agent_index (int): the unique index the agent will use in the system (starting from zero for the first agent)
        nbr_of_samples (int): Number of data samples in the arrays.
        max_energy_production_Wh (jax.Array[int, 1]): max amount of energy the prosumer can produce
        max_battery_capacity_Wh (jax.Array[int, 1]): max amount of energy the prosumer can store
        max_battery_peak_Wh (jax.Array[int, 1]): max amount of energy the prosumer can inject/retrieve to/from the battery
        months (jax.Array[int, 1]): Array of months (1-12) corresponding to the data.
        hour (jax.Array[int, 1]): Array of hours (0-23) corresponding to the data.
        weekday (jax.Array[int, 1]): Array of weekdays (0-6) corresponding to the data.
        energy_consumption_Wh (jax.Array[float, 1]): Array of energy consumption values (negative values means that there is more energy produced than consumed).
       
    
    """
    agent_id:str
    agent_index:int
    nbr_of_samples:int
    max_energy_production_Wh:jax.Array
    max_battery_capacity_Wh:jax.Array
    max_battery_peak_Wh:jax.Array
    years:jax.Array
    months:jax.Array
    weeknumbers:jax.Array
    weekdays:jax.Array
    hours:jax.Array
    energy_consumption_Wh:jax.Array
   


def print_prosumer(prosumer:Prosumer) -> None:
    """Prints the elements of a Prosumer namedtuple.

    Args:
        prosumer (Prosumer): A Prosumer namedtuple instance.
    """
    print("\n")
    print("-"*50)
    print("Prosumer Data:")
    print("-"*50)
    print(f"Agent ID: {prosumer.agent_id}")
    print(f"Agent Index: {prosumer.agent_index}")
    print(f"Number of Samples: {prosumer.nbr_of_samples}")
    print(f"Max energy production: {prosumer.max_energy_production_Wh}")
    print(f"Max battery capacity: {prosumer.max_battery_capacity_Wh}")
    print(f"Max battery peak: {prosumer.max_battery_peak_Wh}")
    print(f"Years: {prosumer.years}")
    print(f"Months: {prosumer.months}")
    print(f"Weeknumbers: {prosumer.weeknumbers}")
    print(f"Weekdays: {prosumer.weekdays}")
    print(f"Hours: {prosumer.hours}")
    print(f"Weekdays: {prosumer.weekdays}")
    print(f"Energy Production: {prosumer.energy_consumption_Wh}")
    


def get_statistics(prosumer:Prosumer)-> Tuple[float,float,float,float,float]:
    """ calculates various statistical metrics regarding the day ahead price

    Args:
        prosumer (Prosumer): The namedTuple defining the prosumer

    Returns:
        Tuple[float,float,float,float,float]: min, Q1,Q2,Q3, max value of the energy consumption
    """
    min= float(jax.numpy.min(prosumer.energy_consumption_Wh))
    q1 = float(jax.numpy.quantile(prosumer.energy_consumption_Wh,0.25))
    q2 = float(jax.numpy.quantile(prosumer.energy_consumption_Wh,0.5))
    q3 = float(jax.numpy.quantile(prosumer.energy_consumption_Wh,0.75))
    max =float(jax.numpy.max(prosumer.energy_consumption_Wh))
    return min,q1,q2,q3,max
