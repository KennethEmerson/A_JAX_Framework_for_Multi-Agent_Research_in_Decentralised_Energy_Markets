from agents.agent_config import PpoAgentConfig, RuleBasedAgentConfig,ActorCriticNeurNetConfig
from experiments.experiment_config import AGENT_ID_LIST

model_config = ActorCriticNeurNetConfig(
    ACTOR_LAYERS = [128,128,128],
    CRITIC_LAYERS = [128,128,128]
)

ppo_agent_list = [
    PpoAgentConfig(
        ID=agent_id,
        NN_MODEL = model_config,
        ACTIVATION= "relu", 
        MAX_GRAD_NORM= 0.5, 
        GAMMA= 0.97,  
        GAE_LAMBDA= 0.95, 
        CLIP_EPS= 0.2, 
        VF_COEF= 0.5, 
        ENT_COEF= 0.01,   
        LR= 2.8e-4,    
        ANNEAL_LR= True, 
        UPDATE_EPOCHS= 4,
        NUM_MINIBATCHES=4
        ) for agent_id in  AGENT_ID_LIST
    ]


selfplay_ppo_agent = PpoAgentConfig(
        ID='selfplay_agent',
        NN_MODEL = model_config,
        ACTIVATION= "relu", 
        MAX_GRAD_NORM= 0.5, 
        GAMMA= 0.97,  
        GAE_LAMBDA= 0.95, 
        CLIP_EPS= 0.2, 
        VF_COEF= 0.5, 
        ENT_COEF= 0.01,   
        LR= 2.8e-4,    
        ANNEAL_LR= True, 
        UPDATE_EPOCHS= 4,
        NUM_MINIBATCHES=4
        ) 

rule_agent_list = [
    RuleBasedAgentConfig(
        ID=agent_id,
        USE_DATA_SHARING_IF_AVAILABLE=False,
        USE_DAY_AHEAD_WINDOW_IF_AVAILABLE=False
    ) for agent_id in  AGENT_ID_LIST
    ]