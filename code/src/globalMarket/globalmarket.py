""" 
-------------------------------------------------------------------------------------------
A JAX FRAMEWORK FOR MULTI-AGENT RESEARCH IN DECENTRALISED ENERGY MARKETS
-------------------------------------------------------------------------------------------
author: K. Emerson
Department of Computer Science
Faculty of Sciences and Bioengineering Sciences
Vrije Universiteit Brussel
-------------------------------------------------------------------------------------------
file contains the GlobalMarket NamedTuple definition and functions to manipulate the 
GlobalMarket instances
"""

from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from auction.doubleAuction import ClearedMarket, MarketStatus
from ledger.ledger import OfferElemIndex


class GlobalMarket(NamedTuple):
    """
    a NamedTuple encapsulating the configuration and data related to the global market

    Args:
        nbr_of_samples (int): the total number of samples 
        feed_in_cte_cost (float): constant cost required each time energy is injected into the 
                                  global market.
        feed_in_cte_Wh_cost (float): constant cost per Wh for each Wh that is injected 
                                     into the global market.
        feed_in_perc_Wh_cost (float): percentage cost of the Wh price for each Wh that is injected 
                                      into the global market.
        time_of_use_cte_cost (float): constant cost required each time energy is consumed from the 
                                      global market.
        time_of_use_cte_Wh_cost (float): constant cost per Wh for each Wh that is consumed 
                                         from the global market.
        time_of_use_perc_Wh_cost (float): percentage cost of the Wh price for each Wh that is consumed 
                                          from the global market.
        years (jax.Array): an array containing the actual year of each sample, size equals nbr_of_samples
        months (jax.Array): array containing the actual month of each sample, size equals nbr_of_samples
        weeknumbers (jax.Array): array containing the week number of each sample, size equals nbr_of_samples
        weekdays (jax.Array): array containing the weekday of each sample, size equals nbr_of_samples
        hours (jax.Array): array containing the hour of each sample, size equals nbr_of_samples
        day_ahead_prices_Wh (jax.Array): array containing the day-ahead price in Wh of each sample, size equals nbr_of_samples
        day_ahead_window_hour_trigger (int): The hour at which the day-ahead window refresh is triggered 
        day_ahead_window_size (int): The size of the day-ahead window containing the future day ahead prices within the window
    """
    nbr_of_samples:int
    feed_in_cte_cost:float
    feed_in_cte_Wh_cost:float
    feed_in_perc_Wh_cost:float
    time_of_use_cte_cost:float
    time_of_use_cte_Wh_cost:float
    time_of_use_perc_Wh_cost:float
    years:jax.Array
    months:jax.Array
    weeknumbers:jax.Array
    weekdays:jax.Array
    hours:jax.Array
    day_ahead_prices_Wh: jax.Array
    day_ahead_window_hour_trigger:int
    day_ahead_window_size:int
    


def get_statistics(global_market:GlobalMarket)-> Tuple[float,float,float,float,float]:
    """ calculates various statistical metrics regarding the day ahead price

    Args:
        global_market (GlobalMarket): The namedTuple defining the global market

    Returns:
        Tuple[float,float,float,float,float]: min, Q1,Q2,Q3, max value of the day ahead price
    """
    min= float(jnp.min(global_market.day_ahead_prices_Wh))
    q1 = float(jnp.quantile(global_market.day_ahead_prices_Wh,0.25))
    q2 = float(jnp.quantile(global_market.day_ahead_prices_Wh,0.5))
    q3 = float(jnp.quantile(global_market.day_ahead_prices_Wh,0.75))
    max = float(jnp.max(global_market.day_ahead_prices_Wh))
    return min,q1,q2,q3,max



def get_day_ahead_window(global_market:GlobalMarket,timestep:int) -> jax.Array:
    """returns the day ahead window starting from the given timestep. The window size 
       is determined by the parameter set in the GlobalMarket NamedTuple     

    Args:
        global_market (GlobalMarket): The namedTuple defining the global market
        timestep (int): the timestep from where to start the day ahead window

    Returns:
        jax.Array: array containing the window with day ahead prices
    """
    day_ahead_window_size = global_market.day_ahead_window_size 
    day_ahead_prices_Wh = global_market.day_ahead_prices_Wh
   
    actual_window = jax.lax.dynamic_slice(day_ahead_prices_Wh,(timestep,),(day_ahead_window_size,))
    return actual_window



def calc_variable_time_of_use_price(global_market:GlobalMarket,timestep:int)-> jax.Array:
    """calculates the time of use price based on the actual day ahead price and the time of use 
       cost parameters as set in the global_market tuple. 

    Args:
        global_market (GlobalMarket): The namedTuple defining the global market
        timestep (int): the timestep for which to calculate the time of use price

    Returns:
        jax.Array: array of size 1 containing the time of use price 
    """
    day_ahead_price = global_market.day_ahead_prices_Wh[timestep]
    cte_cost_Wh = global_market.time_of_use_cte_Wh_cost
    perc_cost_Wh = global_market.time_of_use_perc_Wh_cost

    # take abs of day ahead price so that cost always increases feed in price even when prices are negative
    total_cost = day_ahead_price + cte_cost_Wh + (jnp.abs(day_ahead_price)*perc_cost_Wh/100) 
    return total_cost   



def calc_variable_feed_in_price(global_market:GlobalMarket,timestep:int)-> jax.Array:
    """calculates the feed in price based on the actual day ahead price and the feed in price 
       cost parameters as set in the global_market tuple. 

    Args:
        global_market (GlobalMarket): The namedTuple defining the global market
        timestep (int): the timestep for which to calculate the time of use price

    Returns:
        jax.Array: array of size 1 containing the feed in price 
    """
    day_ahead_price = global_market.day_ahead_prices_Wh[timestep]
    cte_cost_Wh = global_market.feed_in_cte_Wh_cost
    perc_cost_Wh = global_market.feed_in_perc_Wh_cost

    # take abs of day ahead price so that cost always decreases feed in price even when prices are negative
    total_cost = day_ahead_price - cte_cost_Wh - (jnp.abs(day_ahead_price)*perc_cost_Wh/100)  
    return total_cost      

    

def clear_non_cleared_offers_on_global_market(
        global_market:GlobalMarket,
        timestep:int,
        cleared_market:ClearedMarket) -> ClearedMarket:
    """ clears the offers of the agents that were not cleared on the internal market by buying/selling them
        on the global market. 

    Args:
        global_market (GlobalMarket): The namedTuple defining the global market
        timestep (int): the timestep ro use to retrieve the global market data
        cleared_market (ClearedMarket): the ClearedMarket NamedTuple containing the 
                                        internally cleared offers and the uncleared offers to be cleared on the global market

    Returns:
        ClearedMarket: the ClearedMarket NamedTuple wherein all agent offers are cleared
    """
    time_of_use_price = calc_variable_time_of_use_price(global_market,timestep)                                    
    feed_in_price = calc_variable_feed_in_price(global_market,timestep)
 
    # vmap functions to set the prices for the non cleared offer matrices    
    set_time_of_use_price = jax.vmap(lambda row:row.at[OfferElemIndex.PRICE].set(jnp.where(row[OfferElemIndex.AMOUNT] > 0 , time_of_use_price, 0)))
    set_feed_in_price = jax.vmap(lambda row:row.at[OfferElemIndex.PRICE].set(jnp.where(row[OfferElemIndex.AMOUNT] > 0 , feed_in_price, 0)))
    
    final_cleared_market = ClearedMarket(
        amount= cleared_market.amount,
        price= cleared_market.price,
        market_status= MarketStatus.GLOBAL,
        cleared_asks= cleared_market.cleared_asks,
        cleared_bids= cleared_market.cleared_bids,
        non_cleared_asks= set_time_of_use_price(cleared_market.non_cleared_asks),
        non_cleared_bids= set_feed_in_price(cleared_market.non_cleared_bids),
        battery_shortage_asks = cleared_market.battery_shortage_asks,
        battery_overflow_bids = cleared_market.battery_overflow_bids
    )
    return final_cleared_market



def add_battery_level_out_of_bounds(
        global_market:GlobalMarket,
        timestep:int,
        cleared_market:ClearedMarket,
        agent_indices:jax.Array,
        battery_level_shortage:jax.Array,
        battery_level_overflow:jax.Array) -> ClearedMarket:
    """ adds the battery level overflows to the non cleared bid amounts and the battery level shortages to the non cleared 
        asks amounts in order to clear these values on the global market

    Args:
        global_market (GlobalMarket): the GlobalMarket instance to use
        timestep (int): the timestep to take the time-of-use and feed-in prices from
        cleared_market (ClearedMarket): the ClearedMarket instance to which the battery level out of bounds should be 
                                        cleared on the global market
        agent_indices (jax.array): a 1D array with size equal to the nbr of agents which contains the agent indices
        battery_level_overflow (jax.Array): a 1D array with size equal to the nbr of agents which contains the excess energy 
                                            that is produced(or bought) and that cannot be stored in the battery
        battery_level_shortage (jax.Array): a 1D array with size equal to the nbr of agents which contains the energy 
                                            shortage that is used (or sold) and that cannot be retrieved from the battery

    Returns:
        ClearedMarket: a new ClearedMarket instances containing the modified non-cleared offer amounts
    """    
    time_of_use_price = calc_variable_time_of_use_price(global_market,timestep)                                    
    feed_in_price = calc_variable_feed_in_price(global_market,timestep)

    def _update_battery_offer(orig:jax.Array, amount:jax.Array,index:jax.Array,price:jax.Array) -> jax.Array:
        return jnp.where(amount>0,jnp.ravel(jnp.array([index,price,amount])), orig)

    update_battery_offers = jax.vmap(_update_battery_offer, (0,0,0, None))
    final_cleared_market = ClearedMarket(
        amount= cleared_market.amount,
        price= cleared_market.price,
        market_status= MarketStatus.BATTERY_GLOBAL,
        cleared_asks= cleared_market.cleared_asks,
        cleared_bids= cleared_market.cleared_bids,
        non_cleared_asks= cleared_market.non_cleared_asks,
        non_cleared_bids= cleared_market.non_cleared_bids,
        
        battery_shortage_asks = update_battery_offers(
                                        cleared_market.battery_shortage_asks,
                                        battery_level_shortage,
                                        agent_indices,time_of_use_price),
        
        battery_overflow_bids = update_battery_offers(
                                        cleared_market.battery_overflow_bids,
                                        battery_level_overflow,
                                        agent_indices,feed_in_price)
        )
    return final_cleared_market



def print_global_market(market:GlobalMarket) -> None:
    """Prints the day ahead prices of a GlobalMarket namedtuple.

    Args:
        market: A GlobalMarket namedtuple instance.
    """
    print("\n")
    print("-"*50)
    print("Global Market Data:")
    print("-"*50)
    print(f"Number of Samples: {market.nbr_of_samples}")
    print(f"day_ahead_prices_Wh: {market.day_ahead_prices_Wh}")
    