""" 
-------------------------------------------------------------------------------------------
A JAX FRAMEWORK FOR MULTI-AGENT RESEARCH IN DECENTRALISED ENERGY MARKETS
-------------------------------------------------------------------------------------------
author: K. Emerson
Department of Computer Science
Faculty of Sciences and Bioengineering Sciences
Vrije Universiteit Brussel
-------------------------------------------------------------------------------------------
file contains the Offers NamedTuple definition and functions to manipulate the Offers instances
"""
from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp


class OfferElemIndex(IntEnum):
    """
    Enum representing indices within `Offers` arrays for specific offer elements.
    """
    AGENT_ID = 0
    """Index for the agent identifier."""
    PRICE    = 1
    """Index for the offer price."""
    AMOUNT   = 2
    """Index for the offer amount"""

_DEFAULT_ASK = jnp.array([-1., -jnp.inf, 0.])
_DEFAULT_BID = jnp.array([-1., jnp.inf, 0.])



class Offers(NamedTuple):
    """
    NamedTuple representing offer data for bids and asks.

    Args:
        bids (jax.Array[float,3]): 2D array of bids, where:
            - First dimension indexes offers for each agent based on the agents index (rows).
            - Second dimension indexes offer elements (see OfferElemIndex).
        asks (jax.Array[float, 3]): Similar to `bids` but for ask offers.
    """
    bids: jax.Array
    asks:jax.Array



def create_empty_auction(nbr_of_agents:int) -> Offers:
    """creates an empty Offers Named Tuple for a new auction
       Warning: this function is not JAX.JIT compliant

    Args:
        nbr_of_agents (int): the number of agents

    Returns:
        Offers: a namedTuple instance containing the asks and bids arrays filled with:
        - -1 for the agent index, 
        - zeros for all prices, 
        - zeros for the ask amounts and inf for the bid amounts.   
    """
    bids =  jnp.full((nbr_of_agents,len(OfferElemIndex)),_DEFAULT_BID)
    asks =  jnp.full((nbr_of_agents,len(OfferElemIndex)),_DEFAULT_ASK)
    return Offers(bids=bids,asks=asks)


          
def add_new_offer(offers:Offers,agent_id:int,price:float,amount:float) -> Offers:
    """adds a new offer to the ledger, offers with a negative value are considered bids for which the agent wants to sell the energy

    Args:
        offers (Offers): the object containing the current offers (bids and asks)
        agent_id (int): the numerical id of the agent
        amount (float): the amount of energy the agent wishes to bid/sell (negative value) or ask/buy (positive value)
        price (float): the price for which the agent wants to sell/buy the bid/ask

    Returns:
        Offers: the object containing all offers including the newly added offer
    """
    bids = jnp.where(amount < 0, offers.bids.at[agent_id, :].set(jnp.array([agent_id,price,jnp.absolute(amount)])),offers.bids)
    asks = jnp.where(amount > 0, offers.asks.at[agent_id, :].set(jnp.array([agent_id,price,amount])),offers.asks)
    return Offers(bids=bids,asks=asks)



def add_all_offers(unfiltered_offers:jax.Array) -> Offers:
    """ creates an Offer object containing asks and bids based on a 2D-matrix of offers

    Args:
        unfiltered_offers (jax.Array): 2D matrix where the columns comply with the OfferElemIndex

    Returns:
        Offers: 
    """
    filter_asks = jax.vmap(lambda offer: jnp.where(offer[OfferElemIndex.AMOUNT]>0,jnp.array(offer),_DEFAULT_ASK),(0,))
    filter_bids = jax.vmap(lambda offer: jnp.where(offer[OfferElemIndex.AMOUNT]< 0,jnp.array([offer[OfferElemIndex.AGENT_ID],offer[OfferElemIndex.PRICE],jnp.abs(offer[OfferElemIndex.AMOUNT])]),_DEFAULT_BID),(0,))

    return Offers(
        asks= filter_asks(unfiltered_offers),
        bids = filter_bids(unfiltered_offers)
    )