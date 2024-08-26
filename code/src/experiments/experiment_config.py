import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Union

from agents.agent_config import AgentConfig
from globalMarket.globalmarket_factory import GlobalMarketConfig
from prosumer.prosumer_config import ProsumerConfig
from train_config import TrainConfig
from transformers.transformers_config import TransformerConfig

GLOBAL_MARKET_FILE = os.path.join("data_synthetic","globalmarket","globalmarket.csv")
EPOCH_STEPS = 52512
AGENT_ID_LIST = [f"agent_{i:02d}" for i in range(0,8)]
SEED_LIST = [0,10,20,30,40]


@dataclass(frozen=True)
class LogConfig:
    """ contains all logging specific configurations """
    LOG_TO_WANDB : bool # log to wandb
    LOG_LOCAL: bool # log to a local file 
    LOG_INTERVAL : int # sample rate to log values

@dataclass(frozen=True)
class ExperimentConfig:
    """ main configuration """
    PROJECT_NAME : str  # name of the experiment project used by Wandb
    EXPERIMENT_DESCRIPTION : str # description 
    TAGS:List[str]
    GLOBALMARKET: GlobalMarketConfig
    AGENT_IDS:List[str]
    PROSUMERS: Sequence[ProsumerConfig]
    AGENTS:Union[AgentConfig,Sequence[AgentConfig]]
    TRANSFORMER:Union[TransformerConfig,Dict[str,TransformerConfig]]
    TRAIN:TrainConfig
    LOG:LogConfig


####################################################################################################







       
