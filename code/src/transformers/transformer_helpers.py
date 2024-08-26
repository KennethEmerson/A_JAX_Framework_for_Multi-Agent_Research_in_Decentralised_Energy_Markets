""" 
-------------------------------------------------------------------------------------------
A JAX FRAMEWORK FOR MULTI-AGENT RESEARCH IN DECENTRALISED ENERGY MARKETS
-------------------------------------------------------------------------------------------
author: K. Emerson
Department of Computer Science
Faculty of Sciences and Bioengineering Sciences
Vrije Universiteit Brussel
-------------------------------------------------------------------------------------------
file contains specific helper functions used by the Transformer class objects
"""

from typing import Callable

import jax
import jax.numpy as jnp

###########################################
# SCALER FUNCTIONS
###########################################

def robust_scaler(q1:float,q2:float,q3:float) -> Callable[[jax.Array],jax.Array]:
    """
    Creates a robust scaling function based on specified quantiles.
    This function generates a scaling function that normalizes input data 
    using a robust scaling approach, which is less sensitive to outliers 
    compared to standard scaling methods. The scaling is performed based 
    on the first (q1), second (q2), and third (q3) quantiles of the data.

    Args:
        q1 (float): first quantile (25th percentile) of the data
        q2 (float): second quantile (50th percentile or median) of the data 
        q3 (float): third quantile (75th percentile) of the data

    Returns:
        Callable[[jax.Array], jax.Array]: 
            A function that takes a JAX array as input and returns a 
            normalized JAX array.
    """
    denom = jnp.where((q3-q1) != 0, q3 -q1, q2)
    
    def func(array:jax.Array) -> jax.Array:
        return (array - q2)/ denom
    return func



def inverse_robust_scaler(q1:float,q2:float,q3:float) -> Callable[[jax.Array],jax.Array]:
    """
    Creates a inverse robust scaling function based on specified quantiles.
    This function inverses the normalisation of the array using the robust scaling approach

    Args:
        q1 (float): first quantile (25th percentile) of the original unnormalised data
        q2 (float): second quantile (50th percentile or median) of the original unnormalised data 
        q3 (float): third quantile (75th percentile) of the original unnormalised data

    Returns:
        Callable[[jax.Array], jax.Array]: 
            A function that takes a Robust scaling normalised JAX array as input and returns the 
            unnormalised JAX array.
    """
    nom = jnp.where((q3-q1) != 0, q3-q1, q2)
    
    def func(array:jax.Array) -> jax.Array:
        return nom * array + q2
    return func



def min_max_scaler(min:float,max:float,) -> Callable[[jax.Array],jax.Array]:
    """
    Creates a min-max scaling function for normalizing data between -1 and 1.
    This function generates a scaling function that normalizes input data 
    to a specified range using the min-max scaling technique. 
    
    Args:
        min (float): minimum value of the data range
        max (float): maximum value of the data range

    Returns:
        Callable[[jax.Array], jax.Array]: 
            A function that takes a JAX array as input and returns a 
            normalized JAX array. 
    """
    def func(array:jax.Array) -> jax.Array:
        nom = jnp.where((max - min) != 0, max - min, 1)
        return 2*((array - min) / nom) -1
    return func



def inverse_min_max_scaler(min:float,max:float,) -> Callable[[jax.Array],jax.Array]:
    """
    Creates a inverse min-max scaling function based on the specified original min max values.
    This function inverses the normalisation of the array using the min-max scaling approach
    
    Args:
        min (float): minimum value of the original unnormalised data range
        max (float): maximum value of the original unnormalised data range
    
    Returns:
        Callable[[jax.Array], jax.Array]: 
            A function that takes a min-max scaling normalised JAX array as input and returns the 
            unnormalised JAX array.
    """
    def func(array:jax.Array) -> jax.Array:
        nom = jnp.where((max - min) != 0, max - min, 1)
        return ((array+1)/2) * nom + min
    return func



def ignore_scaling() -> Callable[[jax.Array],jax.Array]:
    """ creates a dummy scaling function which just returns the original array
        This function can be used when no normalisation is required
    
    Returns:
        Callable[[jax.Array],jax.Array]: the array that was given as input
    """
    def func(array:jax.Array) -> jax.Array:
        return array
    return func



###########################################
# ACTION AMOUNT MODIFIERS
###########################################

def dont_clip_action_amount(max_battery_peak_Wh:jax.Array,
                         max_battery_capacity_Wh:jax.Array,
                         battery_level_Wh:jax.Array,
                         energy_bought_Wh:jax.Array,
                         energy_demand_Wh:jax.Array) -> jax.Array:
    """ creates a dummy clipping function which just returns the original array of energy bought
        This function can be used when no clipping is required
    
    Returns:
        jax.Array: the array with the amount bought that was given as input
    """
    return energy_bought_Wh



def clip_action_amount_to_battery(max_battery_peak_Wh:jax.Array,
                         max_battery_capacity_Wh:jax.Array,
                         battery_level_Wh:jax.Array,
                         energy_bought_Wh:jax.Array,
                         energy_demand_Wh:jax.Array) -> jax.Array:
    """will clip tha raw action amount to a value that cannot differ from the actual energy demand than what the battery can absorb

    Args:
        max_battery_peak_Wh (jax.Array): array with size 1 containing the max energy that can be injected/extracted in or out 
                                         of the battery within one hour in Wh. 
        max_battery_capacity_Wh (jax.Array): array with size 1 containing the max battery capacity in Wh
        battery_level_Wh (jax.Array): array with size 1 containing the actual battery level in Wh
        energy_bought_Wh (jax.Array): array with size 1 containing the amount the agent wants to buy in that timestep in Wh, 
                                      in which a negative value means the agent wants to sell
        energy_demand_Wh (jax.Array): array with size 1 containing the actual energy consumption in that timestep in Wh,
                                      in which a negative value means the agent produced more energy than it consumed

    Returns:
        jax.Array: the array with the amount bought clipped to conform to the battery limits
    """
        
    energy_delta = energy_bought_Wh - energy_demand_Wh

    peak_clipped_energy_delta = jnp.where(
            max_battery_peak_Wh <  energy_delta,
            max_battery_peak_Wh,
            jnp.where(
                -max_battery_peak_Wh > energy_delta,
                -max_battery_peak_Wh,
                energy_delta
            )
        )
    
    capacity_clipped_energy_delta = jnp.where(
            max_battery_capacity_Wh <  battery_level_Wh + peak_clipped_energy_delta,
            max_battery_capacity_Wh - battery_level_Wh,
            jnp.where(
                0 > battery_level_Wh + peak_clipped_energy_delta,
                - battery_level_Wh,
                peak_clipped_energy_delta
            )
        )
                  
    clipped_action = capacity_clipped_energy_delta + energy_demand_Wh
    return clipped_action


###########################################
# ACTION PRICE MODIFIERS
###########################################

def no_action_price_clipping(
        time_of_use_price:jax.Array,
        feed_in_price:jax.Array,
        price:jax.Array
        ) -> jax.Array:
    """ dummy clipping function that returns the given action price as output

    Args:
        time_of_use_price (jax.Array): the time of use price provided by the global market
        feed_in_price (jax.Array): the time of use price provided by the global market
        price (jax.Array): the unclipped action price chosen by the agent

    Returns:
        jax.Array: the original price that was given as input
    """
    return price



def clip_action_price_to_globalmarket(
        time_of_use_price:jax.Array,
        feed_in_price:jax.Array,
        price:jax.Array
        ) -> jax.Array:
    """ clipping function that clips the action price as chosen by the agent in such that it
        cannot go beyond the limits of the global market time_of_use_price and feed_in_price.

    Args:
        time_of_use_price (jax.Array): the time of use price provided by the global market
        feed_in_price (jax.Array): the time of use price provided by the global market
        price (jax.Array): the unclipped action price chosen by the agent

    Returns:
        jax.Array: the clipped price
    """
    return jnp.clip(price,feed_in_price,time_of_use_price)



def tanh_normalise_action_price_to_globalmarket(
        time_of_use_price:jax.Array,
        feed_in_price:jax.Array,
        price:jax.Array
        ) -> jax.Array:
    """ function that tanh normalises the action price as chosen by the agent in such that it
        cannot go beyond the limits of the global market time_of_use_price and feed_in_price.

    Args:
        time_of_use_price (jax.Array): the time of use price provided by the global market
        feed_in_price (jax.Array): the time of use price provided by the global market
        price (jax.Array): the unclipped action price chosen by the agent

    Returns:
        jax.Array: the clipped price
    """
    range= (time_of_use_price-feed_in_price)
    value = (((price - feed_in_price)/range)) 
    value = ((jnp.tanh(value) + 1)*range/2 + feed_in_price)
    return value



def clip_negative_action_price_to_zero(
        time_of_use_price:jax.Array,
        feed_in_price:jax.Array,
        price:jax.Array
        ) -> jax.Array:
    """ function that clips a negative action price as chosen by the agent in such that it
        negative action prices will always be set to zero.

    Args:
        time_of_use_price (jax.Array): the time of use price provided by the global market
        feed_in_price (jax.Array): the time of use price provided by the global market
        price (jax.Array): the unclipped action price chosen by the agent

    Returns:
        jax.Array: the clipped price
    """
    return jnp.where(price >= 0,price,jnp.array([0]))



def make_action_price_absolute(
        time_of_use_price:jax.Array,
        feed_in_price:jax.Array,
        price:jax.Array
        ) -> jax.Array:
    """ function that makes a negative action price as chosen by the agent, absolute.

    Args:
        time_of_use_price (jax.Array): the time of use price provided by the global market
        feed_in_price (jax.Array): the time of use price provided by the global market
        price (jax.Array): the unclipped action price chosen by the agent

    Returns:
        jax.Array: the clipped price
    """
    return jnp.abs(price)