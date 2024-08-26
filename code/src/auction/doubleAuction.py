""" 
-------------------------------------------------------------------------------------------
A JAX FRAMEWORK FOR MULTI-AGENT RESEARCH IN DECENTRALISED ENERGY MARKETS
-------------------------------------------------------------------------------------------
author: K. Emerson
Department of Computer Science
Faculty of Sciences and Bioengineering Sciences
Vrije Universiteit Brussel
-------------------------------------------------------------------------------------------
File contains the complete logic for clearing a market with ask and bid offers through 
a double auction.
The functions in this file will be used by the market environment to clear the internal market  
"""

from enum import IntEnum
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from ledger.ledger import OfferElemIndex, Offers

###############################################################################
# Type Definitions
###############################################################################

class MarketStatus(IntEnum):
    """
    Enum representing indices within `Offers` arrays for specific offer elements.

    Values:
        UNCLEARED (int): market is uncleared.
        INTERNAL (int): market is cleared internally.
        GLOBAL (int): market is cleared internally and on global market.
        BATTERY_GLOBAL (int): market is cleared internally and on global market, including battery excess.
    """
    UNCLEARED      = 0
    INTERNAL       = 1
    GLOBAL         = 2
    BATTERY_GLOBAL = 3


class ClearedMarket(NamedTuple):
    """ NamedTuple containing all relevant info from a cleared market

    Args:
        price (jax.Array): scalar value representing the clearing price 
        amount (jax.Array): scalar value representing the clearing amount of energy 
        market_status (MarketStatus): enumeration defining the status of the market
        cleared_asks (jax.Array): an array containing all cleared asks where the row index represents the agents index
        cleared_bids (jax.Array): an array containing all cleared bids where the row index represents the agents index
        non_cleared_asks (jax.Array): an array containing all non or globally cleared asks where the row index represents the agents index
        non_cleared_bids (jax.Array): an array containing all non or globally cleared asks where the row index represents the agents index
        battery_shortage_asks (jax.Array): an array containing all battery related asks where the row index represents the agents index
        battery_overflow_bids (jax.Array): an array containing all battery related bids where the row index represents the agents index

    """
    price: jax.Array
    amount: jax.Array
    market_status:MarketStatus
    cleared_asks:jax.Array
    cleared_bids:jax.Array
    non_cleared_asks:jax.Array
    non_cleared_bids:jax.Array
    battery_shortage_asks:jax.Array
    battery_overflow_bids:jax.Array
    

class _IntermittentClearingValues(NamedTuple):
     bid_iter:int # iteration index for going through the bidding offers
     ask_iter:int # iteration index for going through the asking offers
     ordered_offers:Offers # the ordered offers which still need to be processed
     cleared_market:ClearedMarket # the cleared market with the intermediate or final clearing results


###############################################################################
# Helper functions
###############################################################################


def _create_empty_cleared_market(offers:Offers,nbr_of_agents:int) -> ClearedMarket:
    """ internal function used to create a new market
    """
     
    default_cleared_offer = jnp.array([-1.,0.,0.])
    empty_offers =  jnp.full((nbr_of_agents,len(OfferElemIndex)),default_cleared_offer)
    return ClearedMarket(
          price=jnp.array(0),
          amount=jnp.array(0),
          market_status = MarketStatus.UNCLEARED,
          cleared_bids= empty_offers,
          cleared_asks= empty_offers,
          non_cleared_bids= offers.bids,
          non_cleared_asks= offers.asks,
          battery_shortage_asks = empty_offers,
          battery_overflow_bids = empty_offers
     )



def _sort_offer(
          offers:jax.Array,
          offer_elem_index:OfferElemIndex,
          descending:bool=False) -> jax.Array:
    """ internal function to sort an array of offers 
    """
    return jnp.where(
         descending,
         offers[offers[:, offer_elem_index].argsort()[::-1]],
         offers[offers[:, offer_elem_index].argsort()])



def _create_ordered_offers(offers:Offers)-> Offers:
    """ internal function to sort the offers in the Offers named tuple  
    """      
    # sort the bids in ascending order 
    sorted_bids = _sort_offer(offers.bids,OfferElemIndex.PRICE,False)
    bid_curve = jnp.full(sorted_bids.shape,jnp.nan)
    bid_curve = bid_curve.at[:, OfferElemIndex.AMOUNT].set((sorted_bids[:, OfferElemIndex.AMOUNT]))
    bid_curve = bid_curve.at[:, OfferElemIndex.AGENT_ID].set(sorted_bids[:, OfferElemIndex.AGENT_ID])
    bid_curve = bid_curve.at[:, OfferElemIndex.PRICE].set(sorted_bids[:, OfferElemIndex.PRICE])
        
    # sort the asks in decending order 
    sorted_asks = _sort_offer(offers.asks,OfferElemIndex.PRICE,True)
    ask_curve = jnp.full(sorted_asks.shape,jnp.nan)
    ask_curve = ask_curve.at[:, OfferElemIndex.AMOUNT].set(sorted_asks[:, OfferElemIndex.AMOUNT])
    ask_curve = ask_curve.at[:, OfferElemIndex.AGENT_ID].set(sorted_asks[:, OfferElemIndex.AGENT_ID])
    ask_curve = ask_curve.at[:, OfferElemIndex.PRICE].set(sorted_asks[:, OfferElemIndex.PRICE])
    
    return Offers(bid_curve, ask_curve)



def _is_curve_non_exhausted(params:_IntermittentClearingValues) -> jax.Array:
    """ internal function to test if there are any more offers eligible for further clearing 
    """  
    bids = params.ordered_offers.bids
    asks = params.ordered_offers.asks
    bid_iter = params.bid_iter
    ask_iter = params.ask_iter

    return (
        (params.bid_iter < len(bids)) &
        (bids[bid_iter,OfferElemIndex.AGENT_ID] >= 0) & # no more bids are present
        (ask_iter < len(asks)) &
        (bids[bid_iter,OfferElemIndex.AGENT_ID] >=0) & # no more asks are present
        (asks[ask_iter, OfferElemIndex.PRICE] >= bids[bid_iter, OfferElemIndex.PRICE]) # bid price exceeds ask price
    )



def _ask_amount_exceeds_bid_amount(params:_IntermittentClearingValues) -> _IntermittentClearingValues:
    """ internal function to match a bid and ask offer where the amount of the ask is larger than the bidding amount 
    """  
    # get all required variables
    ordered_bids = params.ordered_offers.bids
    ordered_asks = params.ordered_offers.asks
    bid_agent_id = ordered_bids[params.bid_iter][OfferElemIndex.AGENT_ID].astype(int)
    ask_agent_id = ordered_asks[params.ask_iter][OfferElemIndex.AGENT_ID].astype(int)
    bid_price =  ordered_bids[params.bid_iter][OfferElemIndex.PRICE]
    ask_price =  ordered_asks[params.ask_iter][OfferElemIndex.PRICE]
    bid_amount =  ordered_bids[params.bid_iter][OfferElemIndex.AMOUNT]
    ask_amount =  ordered_asks[params.ask_iter][OfferElemIndex.AMOUNT]
    
    new_cleared_bid_amount = params.cleared_market.cleared_bids[bid_agent_id][OfferElemIndex.AMOUNT] + bid_amount
    new_cleared_ask_amount = params.cleared_market.cleared_asks[ask_agent_id][OfferElemIndex.AMOUNT] + bid_amount # only bid amount gets cleared

    # replace the remaining amount back in the remaining offers to be cleared
    amount_difference = ask_amount - bid_amount
    updated_ordered_offers = Offers(
         bids = ordered_bids.at[params.bid_iter,OfferElemIndex.AMOUNT].set(0), # bid amount is resolved thus set to zero 
         asks = ordered_asks.at[params.ask_iter,OfferElemIndex.AMOUNT].set(amount_difference) # remaining amount is set to actual ask
    )

    # update the cleared market NamedTuple
    cleared_market = ClearedMarket(
        price = ask_price,
        amount = params.cleared_market.amount + bid_amount,
        market_status= params.cleared_market.market_status,
        cleared_bids = params.cleared_market.cleared_bids.at[bid_agent_id,:].set(jnp.array([bid_agent_id,bid_price,new_cleared_bid_amount])),
        cleared_asks = params.cleared_market.cleared_asks.at[ask_agent_id,:].set(jnp.array([ask_agent_id,bid_price,new_cleared_ask_amount])),
        non_cleared_bids= params.cleared_market.non_cleared_bids.at[bid_agent_id,OfferElemIndex.AMOUNT].set(0),
        non_cleared_asks= params.cleared_market.non_cleared_asks.at[ask_agent_id,OfferElemIndex.AMOUNT].set(amount_difference),
        battery_shortage_asks = params.cleared_market.battery_shortage_asks,
        battery_overflow_bids = params.cleared_market.battery_overflow_bids
    )

    return _IntermittentClearingValues(
         bid_iter= params.bid_iter + 1, 
         ask_iter= params.ask_iter,
         ordered_offers= updated_ordered_offers,
         cleared_market= cleared_market)



def _bid_amount_exceeds_ask_amount(params:_IntermittentClearingValues) -> _IntermittentClearingValues:
    """ internal function to match a bid and ask offer where the amount of the ask is smaller than the bidding amount 
    """ 
    # get all required variables
    ordered_bids = params.ordered_offers.bids
    ordered_asks = params.ordered_offers.asks
    bid_agent_id = ordered_bids[params.bid_iter][OfferElemIndex.AGENT_ID].astype(int)
    ask_agent_id = ordered_asks[params.ask_iter][OfferElemIndex.AGENT_ID].astype(int)
    bid_price =  ordered_bids[params.bid_iter][OfferElemIndex.PRICE]
    ask_price =  ordered_asks[params.ask_iter][OfferElemIndex.PRICE]
    bid_amount =  ordered_bids[params.bid_iter][OfferElemIndex.AMOUNT]
    ask_amount =  ordered_asks[params.ask_iter][OfferElemIndex.AMOUNT]
    
    new_cleared_bid_amount = params.cleared_market.cleared_bids[bid_agent_id][OfferElemIndex.AMOUNT] + ask_amount # only ask amount gets cleared
    new_cleared_ask_amount = params.cleared_market.cleared_asks[ask_agent_id][OfferElemIndex.AMOUNT] + ask_amount 

    # replace the remaining amount back in the remaining offers to be cleared
    amount_difference = bid_amount - ask_amount
    updated_ordered_offers = Offers(
        bids = ordered_bids.at[params.bid_iter,OfferElemIndex.AMOUNT].set(amount_difference), # remaining amount is set to actual bid
        asks = ordered_asks.at[params.ask_iter,OfferElemIndex.AMOUNT].set(0.), # ask amount is resolved thus set to zero 
    )

    # update the cleared market NamedTuple
    cleared_market = ClearedMarket(
        price = bid_price,
        amount = params.cleared_market.amount + ask_amount,
        market_status= params.cleared_market.market_status,
        cleared_bids = params.cleared_market.cleared_bids.at[bid_agent_id,:].set(jnp.array([bid_agent_id,ask_price,new_cleared_bid_amount])),
        cleared_asks =  params.cleared_market.cleared_asks.at[ask_agent_id,:].set(jnp.array([ask_agent_id,ask_price,new_cleared_ask_amount])),
        non_cleared_bids= params.cleared_market.non_cleared_bids.at[bid_agent_id,OfferElemIndex.AMOUNT].set(amount_difference),
        non_cleared_asks= params.cleared_market.non_cleared_asks.at[ask_agent_id,OfferElemIndex.AMOUNT].set(0),
        battery_shortage_asks = params.cleared_market.battery_shortage_asks,
        battery_overflow_bids = params.cleared_market.battery_overflow_bids
    )

    return _IntermittentClearingValues(
        bid_iter= params.bid_iter, 
        ask_iter= params.ask_iter + 1,
        ordered_offers= updated_ordered_offers,
        cleared_market= cleared_market)



def _bid_amount_equals_ask_amount(params:_IntermittentClearingValues) -> _IntermittentClearingValues:
    """ internal function to match a bid and ask offer where the amount of the ask is smaller than the bidding amount 
    """ 
    # get all required variables
    ordered_bids = params.ordered_offers.bids
    ordered_asks = params.ordered_offers.asks
    bid_agent_id = ordered_bids[params.bid_iter][OfferElemIndex.AGENT_ID].astype(int)
    ask_agent_id = ordered_asks[params.ask_iter][OfferElemIndex.AGENT_ID].astype(int)
    bid_price =  ordered_bids[params.bid_iter][OfferElemIndex.PRICE]
    ask_price =  ordered_asks[params.ask_iter][OfferElemIndex.PRICE]
    amount =  ordered_bids[params.bid_iter][OfferElemIndex.AMOUNT]
    
    new_cleared_bid_amount = params.cleared_market.cleared_bids[bid_agent_id][OfferElemIndex.AMOUNT] + amount 
    new_cleared_ask_amount = params.cleared_market.cleared_asks[ask_agent_id][OfferElemIndex.AMOUNT] + amount 

    # set the processed offers in the remaining offers to zero because both are fully processed
    updated_ordered_offers = Offers(
        bids = ordered_bids.at[params.bid_iter,OfferElemIndex.AMOUNT].set(0), # bid amount is resolved thus set to zero 
        asks = ordered_asks.at[params.ask_iter,OfferElemIndex.AMOUNT].set(0.), # ask amount is resolved thus set to zero 
    )

    # update the cleared market NamedTuple
    cleared_market = ClearedMarket(
        price = bid_price,
        amount = params.cleared_market.amount + amount,
        market_status= params.cleared_market.market_status,
        cleared_bids = params.cleared_market.cleared_bids.at[bid_agent_id,:].set(jnp.array([bid_agent_id,bid_price,new_cleared_bid_amount])),
        cleared_asks =  params.cleared_market.cleared_asks.at[ask_agent_id,:].set(jnp.array([ask_agent_id,ask_price,new_cleared_ask_amount])),
        non_cleared_bids= params.cleared_market.non_cleared_bids.at[bid_agent_id,OfferElemIndex.AMOUNT].set(0),
        non_cleared_asks= params.cleared_market.non_cleared_asks.at[ask_agent_id,OfferElemIndex.AMOUNT].set(0),
        battery_shortage_asks = params.cleared_market.battery_shortage_asks,
        battery_overflow_bids = params.cleared_market.battery_overflow_bids
    )

    return _IntermittentClearingValues(
        bid_iter= params.bid_iter + 1, 
        ask_iter= params.ask_iter + 1,
        ordered_offers= updated_ordered_offers,
        cleared_market= cleared_market)



def _perform_unequal_clearing_step(param:_IntermittentClearingValues) -> _IntermittentClearingValues:
    """ internal function that checks whether the ask amount is larger than the bid amount or vice versa and 
        processes the offers accordingly
    """ 
    bids = param.ordered_offers.bids
    asks = param.ordered_offers.asks
    bid_iter = param.bid_iter
    ask_iter = param.ask_iter

    return lax.cond(asks[ask_iter, OfferElemIndex.AMOUNT] >= bids[bid_iter, OfferElemIndex.AMOUNT],
                        _ask_amount_exceeds_bid_amount,
                        _bid_amount_exceeds_ask_amount,
                        param
                        ) 



def _perform_clearing_step(param:_IntermittentClearingValues) -> _IntermittentClearingValues:
    """ internal function that checks whether the ask amount and bid amount or equal or not and 
        processes the offers accordingly
    """ 
    bids = param.ordered_offers.bids
    asks = param.ordered_offers.asks
    bid_iter = param.bid_iter
    ask_iter = param.ask_iter

    return lax.cond(asks[ask_iter, OfferElemIndex.AMOUNT] == bids[bid_iter, OfferElemIndex.AMOUNT],
                        _bid_amount_equals_ask_amount,
                        _perform_unequal_clearing_step,
                        param
                        )



###############################################################################
# Main functions
###############################################################################


@partial(jax.jit, static_argnames=['nbr_of_agents'])
def clear_market(offers:Offers,nbr_of_agents:int) -> ClearedMarket:
    """ clears the market given a Offers NamedTuple object containing the asks and bids as 2D matrix  

    Args:
        offers (Offers): namedTuple containing the asks and bids as Jax 2D arrays 
        nbr_of_agents (int): the number of agents that placed an offer

    Returns:
        ClearedMarket: NamedTuple containing the clearing price and amount and the cleared and non cleared asks and bids as jax Arrays
    """

    ordered_cumul_offers = _create_ordered_offers(offers)
    cleared_market = _create_empty_cleared_market(offers,nbr_of_agents)
    params = _IntermittentClearingValues(
        bid_iter = 0, 
        ask_iter = 0,
        ordered_offers= ordered_cumul_offers,
        cleared_market= cleared_market
    )

    params = lax.while_loop(
         _is_curve_non_exhausted,
         _perform_clearing_step,
         params
    )
    
    clearing_price = params.cleared_market.price
    clearing_amount = params.cleared_market.amount
    
    set_cleared_offers_price = jax.vmap(lambda row:row.at[OfferElemIndex.PRICE].set(jnp.where(row[OfferElemIndex.AGENT_ID] >= 0 , clearing_price, 0)))
    
    final_cleared_market = ClearedMarket(
        price = clearing_price,
        amount = clearing_amount,
        market_status= MarketStatus.INTERNAL,
        cleared_bids = set_cleared_offers_price(params.cleared_market.cleared_bids),
        cleared_asks = set_cleared_offers_price(params.cleared_market.cleared_asks),
        non_cleared_bids = params.cleared_market.non_cleared_bids,
        non_cleared_asks = params.cleared_market.non_cleared_asks,
        battery_shortage_asks = params.cleared_market.battery_shortage_asks,
        battery_overflow_bids = params.cleared_market.battery_overflow_bids
    )
  
    return final_cleared_market


def print_market(cleared_market:ClearedMarket) -> None:
    """Prints the elements of a ClearedMarket namedtuple.
    
    Args:
        cleared_market (ClearedMarket): A ClearedMarket namedtuple instance.
    """
    print("\n")
    print("-"*50)
    print("Cleared Market Data:")
    print("-"*50)
    print(f"Clearing Price: {cleared_market.price}")
    print(f"Clearing Amount: {cleared_market.amount}")
    print(f"Cleared Asks: (shape: {cleared_market.cleared_asks.shape})\n{cleared_market.cleared_asks}")
    print(f"Cleared Bids: (shape: {cleared_market.cleared_bids.shape})\n{cleared_market.cleared_bids}")
    print(f"Non-Cleared Asks: (shape: {cleared_market.non_cleared_asks.shape})\n{cleared_market.non_cleared_asks}")
    print(f"Non-Cleared Bids: (shape: {cleared_market.non_cleared_bids.shape})\n{cleared_market.non_cleared_bids}")
    print(f"Battery-Cleared Asks: (shape: {cleared_market.battery_shortage_asks.shape})\n{cleared_market.battery_shortage_asks}")
    print(f"Battery-Cleared Bids: (shape: {cleared_market.battery_overflow_bids.shape})\n{cleared_market.battery_overflow_bids}")

