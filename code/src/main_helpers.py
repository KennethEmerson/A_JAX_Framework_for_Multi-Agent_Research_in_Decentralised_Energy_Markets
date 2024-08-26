""" 
-------------------------------------------------------------------------------------------
A JAX FRAMEWORK FOR MULTI-AGENT RESEARCH IN DECENTRALISED ENERGY MARKETS
-------------------------------------------------------------------------------------------
author: K. Emerson
Department of Computer Science
Faculty of Sciences and Bioengineering Sciences
Vrije Universiteit Brussel
-------------------------------------------------------------------------------------------
file contains all helper function used in the main.py training orchestration
"""
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from experiments.experiment_config import ExperimentConfig
from globalMarket.globalmarket import GlobalMarket
from prosumer import prosumer


def check_prosumers_and_glob_market_data_correctness(
        prosumers:List[prosumer.Prosumer],
        global_market:GlobalMarket) -> None:
    """performs integrity check on all the data stored in the GlobalMarket and the Prosumer NamedTuples   

    Args:
        prosumers (List[prosumer.Prosumer]): List of Prosumer NamedTuples
        global_market (GlobalMarket): GlobalMarket NamedTuple

    Raises:
        Exception: raised when one of the tests fail

    """
    all = prosumers + [global_market]
    if not len(set(map(lambda x:int(x.years[0]),all))) == 1:
        raise Exception("start years are not equal for all prosumers and or global market")
    if not len(set(map(lambda x:int(x.months[0]),all))) == 1:
        raise Exception("start months are not equal for all prosumers and or global market")
    if not len(set(map(lambda x:int(x.weeknumbers[0]),all))) == 1:
        raise Exception("start weeknumbers are not equal for all prosumers and or global market")
    if not len(set(map(lambda x:int(x.weekdays[0]),all))) == 1:
        raise Exception("start weekdays are not equal for all prosumers and or global market")
    if not len(set(map(lambda x:int(x.hours[0]),all))) == 1:
        raise Exception("start hours are not equal for all prosumers and or global market")
    if not len(set(map(lambda x:int(x.years[-1]),all))) == 1:
        raise Exception("end years are not equal for all prosumers and or global market")
    if not len(set(map(lambda x:int(x.months[-1]),all))) == 1:
        raise Exception("end months are not equal for all prosumers and or global market")
    if not len(set(map(lambda x:int(x.weeknumbers[-1]),all))) == 1:
        raise Exception("end weeknumbers are not equal for all prosumers and or global market")
    if not len(set(map(lambda x:int(x.weekdays[-1]),all))) == 1:
        raise Exception("end weekdays are not equal for all prosumers and or global market")
    if not len(set(map(lambda x:int(x.hours[-1]),all))) == 1:
        raise Exception("end hours are not equal for all prosumers and or global market")
    if not len(set(map(lambda x: x.nbr_of_samples,all))) == 1:
        raise Exception("nbr of samples are not equal for all prosumers and or global market")



def create_log_dataframes(
        metrics:Dict[str,Any],
        config:ExperimentConfig) -> pd.DataFrame:
    """creates a Pandas DataFrame containing all relevant logging metrics as received when the training is finalised 

    Args:
        metrics (Dict[str,Any]): the logged metrics as received after the training
        config (ExperimentConfig): the Configuration Dataclass containing all the settings used in the experiment

    Returns:
        pd.DataFrame: a Pandas DataFrame containing the logged metrics
    """
    config_log = config.LOG
    log_interval = config_log.LOG_INTERVAL
    num_agents = len(config.AGENT_IDS)
    
    # define all columns
    metrics_columns  = [
            "agent_index",
            "sample",
            "epoch",
            "step",
            "energy_demand",
            "battery_level",
            "max_energy_production_Wh",
            "max_battery_capacity_Wh",
            "agent_raw_amount",
            "agent_ask",
            "agent_bid",
            "cleared_ask",
            "cleared_bid",
            "global_cleared_ask",
            "global_cleared_bid",
            "battery_shortage_ask",
            "battery_overflow_bid",
            "agent_raw_price",
            "agent_price",
            "cleared_price",
            "market_cleared_price",
            "time_of_use_price",
            "feed_in_price",
            "rewards"
            ]
    
    logging_df = pd.DataFrame(columns= metrics_columns)

    # extract all the data per agent
    for i in range(num_agents):
        def _transform_logs(input:jax.Array) -> jax.Array:
            return input.reshape(-1,num_agents)[:,i][0::log_interval]

        sample = jnp.arange(len(metrics["energy_demand"].reshape(-1,num_agents)[:,i]))[0::log_interval]
        epoch = _transform_logs(metrics["epoch"])
        step = _transform_logs(metrics["epoch_step"])
        
        energy_demand = _transform_logs(metrics["energy_demand"])
        battery_level = _transform_logs(metrics["battery_level"])
        max_energy_production_Wh = _transform_logs(metrics["max_energy_production_Wh"])
        max_battery_capacity_Wh = _transform_logs(metrics["max_battery_capacity_Wh"])
         
        agent_raw_amount = _transform_logs(metrics["agent_raw_amount"])
        agent_ask_amount = _transform_logs(metrics["agent_ask_amount"])
        agent_bid_amount = _transform_logs(metrics["agent_bid_amount"])
        cleared_ask_amount = _transform_logs(metrics["cleared_ask_amount"])
        cleared_bid_amount = _transform_logs(metrics["cleared_bid_amount"])
        global_cleared_ask_amount = _transform_logs(metrics["global_cleared_ask_amount"])
        global_cleared_bid_amount = _transform_logs(metrics["global_cleared_bid_amount"])
        battery_shortage_ask_amount = _transform_logs(metrics["battery_shortage_ask_amount"])
        battery_overflow_bid_amount = _transform_logs(metrics["battery_overflow_bid_amount"])
        
        agent_raw_price = _transform_logs(metrics["agent_raw_price"])
        agent_price = _transform_logs(metrics["agent_price"])
        cleared_price = _transform_logs(metrics["cleared_price"])
        
        market_cleared_price = _transform_logs(metrics["market_cleared_price"])
        time_of_use_price = _transform_logs(metrics["time_of_use_price"])
        feed_in_price = _transform_logs(metrics["feed_in_price"])

        rewards = _transform_logs(metrics["rewards"])
        agent_index = np.full(len(sample),i)
        
        # collect all agent metrics
        metrics_data = np.array([
            agent_index,
            sample,
            epoch,
            step,
            energy_demand,
            battery_level,
            max_energy_production_Wh,
            max_battery_capacity_Wh,
            agent_raw_amount,
            agent_ask_amount,
            agent_bid_amount,
            cleared_ask_amount,
            cleared_bid_amount,
            global_cleared_ask_amount,
            global_cleared_bid_amount,
            battery_shortage_ask_amount,
            battery_overflow_bid_amount,
            agent_raw_price,
            agent_price,
            cleared_price,
            market_cleared_price,
            time_of_use_price,
            feed_in_price,
            rewards
        ]).T.tolist()
        
        # aad the metrics to the dataframe
        temp_df = pd.DataFrame(metrics_data,columns=metrics_columns)
        if i == 0:
            logging_df = temp_df
        else:
            logging_df = pd.concat([logging_df,temp_df])

    return logging_df
