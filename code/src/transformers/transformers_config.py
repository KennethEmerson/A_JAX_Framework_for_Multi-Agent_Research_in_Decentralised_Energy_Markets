""" 
-------------------------------------------------------------------------------------------
A JAX FRAMEWORK FOR MULTI-AGENT RESEARCH IN DECENTRALISED ENERGY MARKETS
-------------------------------------------------------------------------------------------
author: K. Emerson
Department of Computer Science
Faculty of Sciences and Bioengineering Sciences
Vrije Universiteit Brussel
-------------------------------------------------------------------------------------------
File contains the factory functions to create a specific agent object based on the given 
configuration dataclasses.
"""
from dataclasses import dataclass
from enum import StrEnum, auto


class AgentActionPriceLimiter(StrEnum):
    """
    Enum representing price limiters that are implemented and can thus be chosen to use.
    Price limiters will take the action price as chosen by the agent and limit it in some way
    to enforce the final action price to adhere to certain limitations before sending it to the env.
    """
    IGNORE = auto()
    """does not perform any clipping on the action price"""
    CLIP_TO_ZERO = auto()
    """clips negative prices to zero """
    CLIP_TO_GLOB_MARKET = auto()
    """clips action prices in such that they cannot exceed the global market time-of-use and feed in price"""
    TANH_NORMALIZE_TO_GLOB_MARKET = auto()
    """normalises action prices using an tanh function in such that they cannot exceed the global market 
        time-of-use and feed in price"""
    MAKE_ABS = auto()
    """returns the absolute value of the action price chosen by the agent"""



class AgentValueScaler(StrEnum):
    """
    Enum representing scaler functions that are implemented and can thus be chosen to use.
    """
    IGNORE = auto()
    """does not rescale the value"""
    ROBUST_SCALER = auto()
    """uses a robust scaler normalisation based on the Quantiles of the underlying data"""
    MINMAX_SCALER = auto()
    """uses a min-max scaler normalisation based on the min max values of the underlying data"""



@dataclass(frozen=True)
class TransformerConfig:
    """
    Configuration settings for the transformer.

    This dataclass encapsulates the parameters that define the behavior 
    and settings of a transformer which is responsible for managing 
    the interactions between the environment and the agent.

    Attributes:
        CLIP_ACTION_AMOUNTS_TO_BATTERY (bool): flag indicating whether to clip action amounts to the battery 
                                               capacity, ensuring that actions do not exceed the available 
                                               battery storage.
        VALUE_SCALER (AgentValueScaler): instance of the `AgentValueScaler` enum 
        ACTION_PRICE_LIMITER (AgentActionPriceLimiter): instance of the `AgentActionPriceLimiter` enum 
        WITH_DAY_AHEAD_WINDOW (bool): flag indicating whether the transformer agent has access to the 
                                      day-ahead window
        DATA_SHARING_AGENTS (int): The number of agents (starting from agent_00) that share their 
                                   energy demand and battery level information with the following other agents
    """
    CLIP_ACTION_AMOUNTS_TO_BATTERY: bool
    VALUE_SCALER:AgentValueScaler
    ACTION_PRICE_LIMITER: AgentActionPriceLimiter 
    WITH_DAY_AHEAD_WINDOW: bool
    DATA_SHARING_AGENTS:int 