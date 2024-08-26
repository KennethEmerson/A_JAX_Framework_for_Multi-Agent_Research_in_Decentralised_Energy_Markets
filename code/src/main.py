""" 
-------------------------------------------------------------------------------------------
A JAX FRAMEWORK FOR MULTI-AGENT RESEARCH IN DECENTRALISED ENERGY MARKETS
-------------------------------------------------------------------------------------------
author: K. Emerson
Department of Computer Science
Faculty of Sciences and Bioengineering Sciences
Vrije Universiteit Brussel
-------------------------------------------------------------------------------------------
File contains the main function for initialising all required elements used in a specific
experiment, executing the training experiment and saving the logged metrics after the 
experiment ends.
"""

import gc
import json
import os
import traceback
from dataclasses import asdict
from datetime import datetime

import jax
import main_helpers as ex_helpers
import wandb
from environments.energymarket_env import DoubleAuctionEnv
from experiments.experiment_config import ExperimentConfig
from globalMarket import globalmarket_factory
from prosumer import prosumer_factory
from train import train


def training_init_and_execute(config:ExperimentConfig,log_folder:str,disable_jit:bool=False) -> None:
    """initialises all required elements to start a training based on the experiment configuration

    Args:
        config (ExperimentConfig): dataclass object that contains all required configuration parameters for the experiment
        log_folder (str): the folder in which to store the configuration as Json and the logging as CSV
        disable_jit (bool, optional): disables the JIT compilation of JAX, useful when debugging. Defaults to False.
    """

    print("\n" + "*"*100)
    print(f"{config.EXPERIMENT_DESCRIPTION}, seed: {config.TRAIN.SEED}")
    print("*"*100)
    
    if disable_jit:
        print(f"[LOG][{datetime.now()}]: GOOGLE JAX JIT is disabled")
    
   

    ############################################################################################
    # LOAD CONFIGURATION
    ############################################################################################
    config_train = config.TRAIN
    config_log = config.LOG
    config_global_market = config.GLOBALMARKET
    config_prosumer_list = config.PROSUMERS
    
   
    ############################################################################################
    # INITIALISE LOGGING    
    ############################################################################################
    if config_log.LOG_TO_WANDB:
        wandb.init(
            project="my-awesome-project",
            name= datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
            config= config, # type: ignore
            notes=config.EXPERIMENT_DESCRIPTION,
            tags=config.TAGS)
        print(f"[LOG][{datetime.now()}]: WANDB initialized")

    
    ############################################################################################    
    # CREATE GLOBALMARKET, PROSUMERS, ENV
    ############################################################################################

    global_market = globalmarket_factory.create_global_market(config_global_market)
    print(f"[LOG][{datetime.now()}]: Global market created")
    
    prosumers = [prosumer_factory.create_prosumer(index,config) for index,config in enumerate(config_prosumer_list)]
    print(f"[LOG][{datetime.now()}]: {len(prosumers)} prosumers created")
    
    ex_helpers.check_prosumers_and_glob_market_data_correctness(prosumers,global_market)
    print(f"[LOG][{datetime.now()}]: Prosumers and globalmarket integrity check completed")

    env = DoubleAuctionEnv(
            global_market= global_market,
            agent_prosumers= prosumers,
            epoch_nbr_of_steps= config_train.EPOCH_STEPS
     )
    print(f"[LOG][{datetime.now()}]: environment created")

    
    ############################################################################################
    # ACTUAL TRAINING
    ############################################################################################
   
    if config.TRAIN.SELF_PLAY:
        print(f"[LOG][{datetime.now()}]: selfplay configuration detected")
    print(f"[LOG][{datetime.now()}]: Start training")        

    with jax.disable_jit(disable_jit):
        rng = jax.random.PRNGKey(config_train.SEED)
        out = train(
            train_config = config.TRAIN,
            transformer_config = config.TRANSFORMER,
            env = env,
            agent_configs = config.AGENTS,
            rng = rng
        )
    print(f"[LOG][{datetime.now()}]: training finished")


    ############################################################################################
    # LOG METRICS
    ############################################################################################
    
    metrics = out["metrics"]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') 
    
    if config_log.LOG_TO_WANDB or config_log.LOG_LOCAL:
        logging_df = ex_helpers.create_log_dataframes(metrics,config)
        
        if config_log.LOG_TO_WANDB:
            wandb.log({"logging": wandb.Table(dataframe=logging_df)})
            wandb.finish()
            print(f"[LOG][{datetime.now()}]: rewards and logging logged in WANDB")    

        if config_log.LOG_LOCAL:
            seed_string = "seed_" + str(config.TRAIN.SEED)
            log_path = os.path.join(log_folder,config.EXPERIMENT_DESCRIPTION,f"{timestamp}_{seed_string}")
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            
            with open(os.path.join(log_path,'logging.csv'), "w") as csv_file:
                logging_df.to_csv(csv_file, index=False)
                csv_file.close()
            
            with open(os.path.join(log_path,'config.json'), 'w' )  as json_file:
                json.dump(asdict(config),json_file)
                json_file.close()
            print(f"[LOG][{datetime.now()}]: rewards and logging logged in folder {log_path}")  
    
    print("*"*100 + "\n")
    gc.collect()

############################################################################################
# MAIN 
############################################################################################

if __name__ == "__main__":
    LOG_FOLDER = "experiment_logs_thesis"
  
    ##################################################
    # choose the list of experiments to run
    ##################################################
    # import experiments.experiments.exp_debug as debug
    # import experiments.experiments.e1_GM_siFFF_PR_siFFF_BA_0_0_DS_0_0_DA_F as e1
    # import experiments.experiments.e2_GM_siTFF_PR_siFFF_BA_0_0_DS_0_0_DA_F as e2
    # import experiments.experiments.e3_GM_siTTF_PR_siFFF_BA_0_0_DS_0_0_DA_F as e3
    # import experiments.experiments.e4_GM_siFFF_PR_siTFF_BA_0_0_DS_0_0_DA_F as e4
    # import experiments.experiments.e5_GM_SY_PR_SY_BA_0_0_DS_0_0_DA_F as e5
    # import experiments.experiments.e6_GM_siFFF_PR_siFFF_BA_13500_5000_DS_0_0_DA_F as e6
    # import experiments.experiments.e7_GM_siTFF_PR_siFFF_BA_13500_5000_DS_0_0_DA_F as e7
    # import experiments.experiments.e8_GM_siTTF_PR_siFFF_BA_13500_5000_DS_0_0_DA_F as e8
    # import experiments.experiments.e9_GM_siFFF_PR_siTFF_BA_13500_5000_DS_0_0_DA_F as e9
    # import experiments.experiments.e10_GM_SY_PR_SY_BA_13500_5000_DS_0_0_DA_F as e10
    # import experiments.experiments.e12_GM_siTFF_PR_siFFF_BA_13500_5000_DS_0_0_DA_T as e12
    # import experiments.experiments.e13_GM_siTTF_PR_siFFF_BA_13500_5000_DS_0_0_DA_T as e13
    # import experiments.experiments.e15_GM_SY_PR_SY_BA_13500_5000_DS_0_0_DA_T as e15
    # import experiments.experiments.e17_GM_siTFF_PR_siFFF_BA_13500_5000_DS_4_4_DA_T as e17
    # import experiments.experiments.e18_GM_siTTF_PR_siFFF_BA_13500_5000_DS_4_4_DA_T as e18
    #import experiments.experiments.e19_GM_siFFF_PR_siTFF_BA_13500_5000_DS_4_4_DA_T as e19
    import experiments.experiments.e20_GM_SY_PR_SY_BA_13500_5000_DS_4_4_DA_T as e20
    
    #experiment_list = se1.config_rule_agent + se1.config_ppo_agent + se1.config_selfplay_agent
    experiment_list =   e20.config_ppo_agent + e20.config_selfplay_agent
    
    
    
    for exp_config in experiment_list:
        log_path = os.path.join(LOG_FOLDER,exp_config.EXPERIMENT_DESCRIPTION)
        if not os.path.exists(log_path):
            os.mkdir(log_path)

        try:
            training_init_and_execute(
                config=exp_config,
                log_folder=LOG_FOLDER,
                disable_jit=False)
        except Exception:
           print(f"[ERROR][{datetime.now()}]{traceback.format_exc()}")
    