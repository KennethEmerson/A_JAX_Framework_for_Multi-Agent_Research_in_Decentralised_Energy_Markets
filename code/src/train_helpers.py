""" 
-------------------------------------------------------------------------------------------
A JAX FRAMEWORK FOR MULTI-AGENT RESEARCH IN DECENTRALISED ENERGY MARKETS
-------------------------------------------------------------------------------------------
author: K. Emerson
Department of Computer Science
Faculty of Sciences and Bioengineering Sciences
Vrije Universiteit Brussel
-------------------------------------------------------------------------------------------
file contains helper functions for train.py
"""
from typing import Dict, Tuple

import jax
from environments.energymarket_env import (
    Actions,
    DoubleAuctionEnv,
    EnvironmentState,
    Observations,
)
from transformers.transformers import Statistics, Transformer
from transformers.transformers_config import TransformerConfig


def batchify(agent_dict: dict, agent_list:Tuple[str,...], num_actors:int) -> jax.Array:
    """extract the per agent data out of the dictionary mapping the agent 
       identifier to the array with observations and transform it to a 2D jax Array
       
       function is used when collecting all agent observations during selfplay

    Args:
        agent_dict (dict): dict mapping agent identifiers to the array of observations as received by the transformers
        agent_list (Tuple[str,...]): list of agent identifiers
        num_actors (int): total number of actors/agents

    Returns:
        jax.Array: a 2D array containing all agents observations
    """    
    stack = jax.numpy.stack([agent_dict[a] for a in agent_list])
    return stack.reshape((num_actors, -1))



def unbatchify(agents_array:jax.numpy.ndarray, agent_list:Tuple[str,...], num_actors:int) -> Dict[str,jax.Array]:
    """ transform the batched 2D data action array back to a dictionary mapping agent 
        identifiers to an array of actions 
        
        function is used when collecting all agent actions during selfplay
    Args:
        agents_array (Array): a 2D array containing all agents observations
        agent_list (Tuple[str,...]): list of agent identifiers
        num_actors (int): total number of actors/agents

    Returns:
        Dict[str,jax.Array]: dict mapping agent identifiers to the array of actions of that agent
    """    
    x = agents_array.reshape((num_actors,1, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}



def create_stats(day_ahead_prices_Wh:jax.Array,prosumer_demand:jax.Array) -> Statistics:
    """calculates the statistics as required by each of the transformers (one transformer per agent)

    Args:
        day_ahead_prices_Wh (jax.Array): the day ahead prices of the global market on which to calculate the statistics
        prosumer_demand (jax.Array): the energy demand values of the global market on which to calculate the statistics

    Returns:
        Statistics: a NamedTuple containing all statistics required by the agent's transformer
    """
    return Statistics(
                min_day_ahead_price= float(jax.numpy.min(day_ahead_prices_Wh)),
                q1_day_ahead_price = float(jax.numpy.quantile(day_ahead_prices_Wh,0.25)),
                q2_day_ahead_price=  float(jax.numpy.quantile(day_ahead_prices_Wh,0.5)),
                q3_day_ahead_price=  float(jax.numpy.quantile(day_ahead_prices_Wh,0.75)),
                max_day_ahead_price= float(jax.numpy.max(day_ahead_prices_Wh)),
                min_prosumer_demand= float(jax.numpy.min(prosumer_demand)),
                q1_prosumer_demand=  float(jax.numpy.quantile(prosumer_demand,0.25)),
                q2_prosumer_demand=  float(jax.numpy.quantile(prosumer_demand,0.5)),
                q3_prosumer_demand=  float(jax.numpy.quantile(prosumer_demand,0.75)),
                max_prosumer_demand= float(jax.numpy.max(prosumer_demand))
            )



def create_multiagent_transformers(
        transformer_config:Dict[str,TransformerConfig],
        env:DoubleAuctionEnv) -> Dict[str,Transformer]:
    """create a dict mapping agent identifiers to Transformer objects based on a 
       dict mapping agent identifiers to TransformerConfig objects

    Args:
        transformer_config (Dict[str,TransformerConfig]): dict mapping agent identifiers to TransformerConfig dataclasses
        env (DoubleAuctionEnv): the environment object

    Returns:
        Dict[str,Transformer]: a dict mapping agent identifiers to Transformer objects
    """
    agent_ids = env.agent_ids

    transformers = {}
    for agent_id in agent_ids:

        stats = create_stats(
            day_ahead_prices_Wh = env.global_market.day_ahead_prices_Wh,
            prosumer_demand= env.agent_prosumers[agent_id].energy_consumption_Wh)

        transformer = Transformer(
            agent_id = agent_id,
            agent_index = env.agent_ids_to_index[agent_id],
            clip_action_amounts_to_battery= transformer_config[agent_id].CLIP_ACTION_AMOUNTS_TO_BATTERY,
            value_scaler = transformer_config[agent_id].VALUE_SCALER,
            action_price_limiter = transformer_config[agent_id].ACTION_PRICE_LIMITER,
            with_day_ahead_window =  transformer_config[agent_id].WITH_DAY_AHEAD_WINDOW,
            nbr_data_sharing_agents = transformer_config[agent_id].DATA_SHARING_AGENTS, 
            globalmarket = env.global_market,
            statistics = stats,
            )
        transformers[agent_id] = transformer
    return transformers



def create_selfplay_transformers(
        transformer_config:TransformerConfig,
        env:DoubleAuctionEnv) -> Dict[str,Transformer]:
    """create a dict mapping agent identifiers to (identical) Transformer objects based on one TransformerConfig dataclass
        which is used during selfplay (requires all agents to use the same Transformer)

    Args:
        transformer_config (TransformerConfig): TransformerConfig dataclass containing the configuration for all Transformers
        env (DoubleAuctionEnv): the environment object

    Returns:
        Dict[str,Transformer]: a dict mapping agent identifiers to (identical) Transformer objects 
    """
    agent_ids = env.agent_ids
    prosumer_demands = jax.numpy.array([])
    
    for prosumer in env.agent_prosumers.values():
        prosumer_demands = jax.numpy.append(prosumer_demands,prosumer.energy_consumption_Wh) 

    stats = create_stats(
        day_ahead_prices_Wh = env.global_market.day_ahead_prices_Wh,
        prosumer_demand= prosumer_demands)

    
    transformers = {}
    for agent_id in agent_ids:
        transformer = Transformer(
            agent_id = agent_id,
            agent_index = env.agent_ids_to_index[agent_id],
            clip_action_amounts_to_battery= transformer_config.CLIP_ACTION_AMOUNTS_TO_BATTERY,
            value_scaler = transformer_config.VALUE_SCALER,
            action_price_limiter = transformer_config.ACTION_PRICE_LIMITER,
            with_day_ahead_window =  transformer_config.WITH_DAY_AHEAD_WINDOW,
            nbr_data_sharing_agents = transformer_config.DATA_SHARING_AGENTS, 
            globalmarket = env.global_market,
            statistics = stats,
            )
        transformers[agent_id] = transformer

    return transformers



def transform_observations(
        agent_id_list:Tuple[str,...],
        transformers:Dict[str,Transformer],
        state:EnvironmentState,
        observations:Observations) -> Dict[str,jax.Array]:
    """transforms the observations for all agents using the designated transformer of the agent

    Args:
        agent_id_list (Tuple[str,...]): list of the agent identifiers (e.g agent_00)
        transformers (Dict[str,Transformer]): a dict mapping agent identifiers to their Transformer objects
        state (EnvironmentState): the environment State object
        observations (Observations): the actual observations dict mapping agent identifiers to their observations

    Returns:
        Dict[str,jax.Array]: a dict mapping the agent identifiers to the array with transformed observations
    """
    return {agent_id:transformers[agent_id].transform_observations(state,observations) for agent_id in agent_id_list}



def transform_actions(
        agent_id_list:Tuple[str,...],
        transformers:Dict[str,Transformer],
        state:EnvironmentState,
        actions:Dict[str,jax.Array]) -> Actions:
    """transforms the actions of all agents using the designated transformer of the agent

    Args:
        agent_id_list (Tuple[str,...]): ist of the agent identifiers (e.g agent_00)
        transformers (Dict[str,Transformer]): a dict mapping agent identifiers to their Transformer objects
        state (EnvironmentState): the environment State object
        actions (Dict[str,jax.Array]): the actual actions dict mapping agent identifiers to their action arrays

    Returns:
        Actions: a dict mapping the agent identifiers to the array with transformed actions
    """
    return {agent_id:transformers[agent_id].transform_actions(state,actions[agent_id]) for agent_id in agent_id_list}

