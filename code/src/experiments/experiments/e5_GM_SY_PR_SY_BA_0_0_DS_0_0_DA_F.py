""" 
-------------------------------------------------------------------------------------------
A JAX FRAMEWORK FOR MULTI-AGENT RESEARCH IN DECENTRALISED ENERGY MARKETS
-------------------------------------------------------------------------------------------
author: K. Emerson
Department of Computer Science
Faculty of Sciences and Bioengineering Sciences
Vrije Universiteit Brussel
-------------------------------------------------------------------------------------------
global market: Synthetic
prosumers: Synthetic
battery: 0Wh, peak :0Wh
Day-ahead window : False
Data sharing: no

"""
import os
from pathlib import Path

import globalMarket.globalmarket_config as gconf
import prosumer.prosumer_config as pconf
import transformers.transformers_config as tconf
from train_config import TrainConfig

import experiments.experiment_config as expconf
import experiments.experiments.variations_agents as avar

####################################################################
# Experiment config
####################################################################

_log_config = expconf.LogConfig(
    LOG_TO_WANDB=False,
    LOG_LOCAL=True,
    LOG_INTERVAL=100
    )

####################################################################
# prosumers config
####################################################################


synthetic_prosumers = [
        pconf.SyntheticProsumerConfig(
            AGENT_ID=agent_id,
            MAX_BATTERY_CAPACITY_WH=0,
            MAX_BATTERY_PEAK_CAPACITY_WH=0,
            CONSUMER_ONLY=False,
            
        ) for agent_id in  expconf.AGENT_ID_LIST]



####################################################################
# transformers config
####################################################################

_multiagent_rule_transformers = {agent_id:tconf.TransformerConfig(
            CLIP_ACTION_AMOUNTS_TO_BATTERY=False, 
            ACTION_PRICE_LIMITER=tconf.AgentActionPriceLimiter.IGNORE, 
            WITH_DAY_AHEAD_WINDOW=False, 
            DATA_SHARING_AGENTS=0, 
            VALUE_SCALER=tconf.AgentValueScaler.IGNORE 
            )for agent_id in  expconf.AGENT_ID_LIST} 

_multiagent_transformers = {agent_id:tconf.TransformerConfig(
            CLIP_ACTION_AMOUNTS_TO_BATTERY=False, 
            ACTION_PRICE_LIMITER=tconf.AgentActionPriceLimiter.IGNORE, 
            WITH_DAY_AHEAD_WINDOW=False, 
            DATA_SHARING_AGENTS=0, 
            VALUE_SCALER=tconf.AgentValueScaler.ROBUST_SCALER 
            )for agent_id in  expconf.AGENT_ID_LIST} 


####################################################################
# global market config
####################################################################

_synth_global_market = gconf.SyntheticDataGlobalMarketConfig(
        FEED_IN_CTE_COST=0, # NEVER change
        FEED_IN_CTE_WH_COST=0., # NEVER change
        FEED_IN_PERC_WH_COST=0, # NEVER change
        TIME_OF_USE_CTE_COST=0, # NEVER change
        TIME_OF_USE_CTE_WH_COST=0.000045, # NEVER change
        TIME_OF_USE_PERC_WH_COST=0, # NEVER change
        DAY_AHEAD_WINDOW_HOUR_TRIGGER=11, # NEVER change
        DAY_AHEAD_WINDOW_SIZE=24, # NEVER change
        FILEPATH= expconf.GLOBAL_MARKET_FILE
    )    


####################################################################
# Experiment config
####################################################################

# rule based

_NBR_ITER = 25
experiment = Path(os.path.basename(__file__)).stem

config_rule_agent =[expconf.ExperimentConfig(
    PROJECT_NAME="A JAX Framework for Multi-Agent Research in Decentralised Energy Markets",
    EXPERIMENT_DESCRIPTION= f"{experiment}_rule",
    TAGS=["rule-based agent"],
    
    GLOBALMARKET=_synth_global_market,
    AGENT_IDS=expconf.AGENT_ID_LIST,
    PROSUMERS=synthetic_prosumers,
    AGENTS= avar.rule_agent_list,
    TRANSFORMER=_multiagent_rule_transformers,
    TRAIN=TrainConfig(
            SELF_PLAY=False,
            SEED=seed, # NEVER change
            NUM_STEPS=128, # NEVER change
            EPOCH_STEPS=expconf.EPOCH_STEPS, # NEVER change
            TOTAL_TIMESTEPS=expconf.EPOCH_STEPS * _NBR_ITER # NEVER change
            ),
    LOG=_log_config,
    ) for seed in expconf.SEED_LIST]

# ppo

config_ppo_agent =[expconf.ExperimentConfig(
    PROJECT_NAME="A JAX Framework for Multi-Agent Research in Decentralised Energy Markets",
    EXPERIMENT_DESCRIPTION=f"{experiment}_ppo",
    TAGS=["ppo agent"],
    
    GLOBALMARKET=_synth_global_market,
    AGENT_IDS=expconf.AGENT_ID_LIST,
    PROSUMERS=synthetic_prosumers,
    AGENTS= avar.ppo_agent_list,
    TRANSFORMER=_multiagent_transformers,
    TRAIN=TrainConfig(
            SELF_PLAY=False,
            SEED=seed,
            NUM_STEPS=128,
            EPOCH_STEPS=expconf.EPOCH_STEPS,
            TOTAL_TIMESTEPS=expconf.EPOCH_STEPS * _NBR_ITER
            ),
    LOG=_log_config,
    ) for seed in expconf.SEED_LIST]


# selfplay

config_selfplay_agent =[expconf.ExperimentConfig(
    PROJECT_NAME="A JAX Framework for Multi-Agent Research in Decentralised Energy Markets",
    EXPERIMENT_DESCRIPTION=f"{experiment}_selfplay",
    TAGS=["selfplay ppo agent"],
    
    GLOBALMARKET=_synth_global_market,
    AGENT_IDS=expconf.AGENT_ID_LIST,
    PROSUMERS=synthetic_prosumers,
    AGENTS= avar.selfplay_ppo_agent,
    TRANSFORMER=_multiagent_transformers["agent_00"],
    TRAIN=TrainConfig(
            SELF_PLAY=True,
            SEED=seed,
            NUM_STEPS=128,
            EPOCH_STEPS=expconf.EPOCH_STEPS,
            TOTAL_TIMESTEPS=expconf.EPOCH_STEPS * _NBR_ITER
            ),
    LOG=_log_config,
    ) for seed in expconf.SEED_LIST]