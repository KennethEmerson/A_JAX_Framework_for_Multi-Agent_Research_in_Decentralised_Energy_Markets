""" 
-------------------------------------------------------------------------------------------
A JAX FRAMEWORK FOR MULTI-AGENT RESEARCH IN DECENTRALISED ENERGY MARKETS
-------------------------------------------------------------------------------------------
author: K. Emerson
Department of Computer Science
Faculty of Sciences and Bioengineering Sciences
Vrije Universiteit Brussel
-------------------------------------------------------------------------------------------
File contains the implementation of the PPO agent class
The implementation used is based upon the JAXMARL Implementation of PPO:
https://github.com/FLAIROx/JaxMARL/blob/main/baselines/IPPO/ippo_ff_overcooked.py
and 
https://colab.research.google.com/github/google/flax/blob/main/docs/linen_intro.ipynb
"""

import functools
from typing import Any, Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from agents.agent_abstract import Agent
from agents.agent_config import ActorCriticNeurNetConfig, PpoAgentConfig
from chex import Numeric
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from train_config import TrainConfig
from train_data_objects import _BatchInfo, _Loss, _Transition, _UpdateState
from transformers.transformers_config import TransformerConfig


class _ActorCritic(nn.Module):
    """ The implementation of the Actor Critic NN networks
    """
    model_config : ActorCriticNeurNetConfig
    action_dim: int
    activation: str = "tanh"

    # setup is automatically called during initialization of the object
    # see: https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html 
    def setup(self) -> None:
         self.actor_layers = [
                            nn.Dense(feat, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)) 
                            for feat in self.model_config.ACTOR_LAYERS
                            ]
         
         self.critic_layers = [
                            nn.Dense(feat, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)) 
                            for feat in self.model_config.CRITIC_LAYERS
                            ]
    

    @nn.compact
    def __call__(self, inputs:jax.Array)->Tuple[distrax.MultivariateNormalDiag,jax.Array]:
        
        # choose activation function
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        
        # Actor
        x = inputs
        for i, lyr in enumerate(self.actor_layers):
            x = lyr(x)
            x = activation(x)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        
        # Critic
        y = inputs
        for i, lyr in enumerate(self.critic_layers):
            y = lyr(y)
            y = activation(y)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(y)

        return pi, jnp.squeeze(critic, axis=-1)
    


class PpoAgent(Agent):
    """ Agent Class implementing a Proximal policy optimization agent
    """
     
    def __init__(
              self,
              agent_config:PpoAgentConfig,
              train_config:TrainConfig,
              transformer_config:TransformerConfig,
              action_space_size:int) -> None:
        """ instantiates the Agent object

        Args:
            agent_config (PpoAgentConfig): a AgentConfig dataclass for a PPO agent containing all agent specific parameters
            train_config (TrainConfig): the TrainConfig dataclass containing all training specific parameters
            transformer_config (TransformerConfig): the TransformerConfig dataclass containing all transformer specific parameters
            action_space_size (int): the size of the action space
        """
        
        self.agent_id =agent_config.ID
        self.agent_config = agent_config
        self.train_config = train_config
        self.network = _ActorCritic(agent_config.NN_MODEL,action_space_size, activation=agent_config.ACTIVATION)  
        self.num_updates = train_config.TOTAL_TIMESTEPS // train_config.NUM_STEPS
        self.minibatch_size = self.train_config.NUM_STEPS // self.agent_config.NUM_MINIBATCHES
        self.batch_size = self.minibatch_size * self.agent_config.NUM_MINIBATCHES



    ##############################################################################
    # HELPER FUNCTIONS
    ##############################################################################  
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def _calculate_gae(self,traj_batch:_Transition, last_val:float)-> Tuple[jax.Array,jax.Array]:
        """calculates the Generalized Advantage Estimation"""
                    
        def _get_advantages(gae_and_next_value:Tuple[jax.Array,float], transition:_Transition)->Tuple[Tuple[jax.Array,float],jax.Array]:
            gae, next_value = gae_and_next_value
            done, value, reward = (
                transition.done,
                transition.value,
                transition.reward,
            )
            delta = reward + self.agent_config.GAMMA * next_value * (1 - done) - value
            gae = (delta + self.agent_config.GAMMA * self.agent_config.GAE_LAMBDA * (1 - done) * gae)
            return (gae, value), gae

        _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_val), last_val),
            traj_batch,
            reverse=True,
            unroll=16, 
        )
        return advantages, advantages + traj_batch.value
        
    

    @functools.partial(jax.jit, static_argnums=(0,))
    def _calc_advantages_and_targets(
         self,
         train_state:TrainState,
         last_obs_batch:jax.Array,
         traj_batch:_Transition) -> Tuple[jax.Array,Tuple[jax.Array,jax.Array,jax.Array]]:
        
        _, last_critic_val = self.network.apply(train_state.params, last_obs_batch)   
        advantages, targets = self._calculate_gae(traj_batch, last_critic_val)
        return advantages, targets



    ##############################################################################
    # MAIN METHODS
    ##############################################################################  

    @functools.partial(jax.jit, static_argnames=("self","observation_space_size",))
    def create_init_train_state(self,observation_space_size:int,rng:jax.Array) -> TrainState:
        """ initializes the flax.training.train_state of the agent

        Args:
            observation_space_size (int): the size of the observation space
            rng (jax.Array): a JAX random number

        Returns:
            TrainState: the flax.training.train_state of the agent
        """
         
        def _linear_schedule(count:Numeric) -> Numeric:
            """function used to set the trajectory for the learning rate based on the given count 
            """
            
            frac = 1.0 - (count // (self.agent_config.NUM_MINIBATCHES * self.agent_config.UPDATE_EPOCHS)) / self.num_updates
            return self.agent_config.LR * frac
         
        if self.agent_config.ANNEAL_LR:
            tx = optax.chain(
                 optax.clip_by_global_norm(self.agent_config.MAX_GRAD_NORM),
                 optax.adam(learning_rate=_linear_schedule, eps=1e-5)
                 )
        else:
            tx = optax.chain(
                 optax.clip_by_global_norm(self.agent_config.MAX_GRAD_NORM), optax.adam(self.agent_config.LR, eps=1e-5)
                 )

        
        init_x = jnp.zeros(observation_space_size)      
        init_x = init_x.flatten()
        
        network_params = self.network.init(rng, init_x)

        train_state = TrainState.create(
            apply_fn=self.network.apply,
            params=network_params,
            tx=tx,
            )
        return train_state



    @functools.partial(jax.jit, static_argnums=(0,))
    def select_action(
         self,
         train_state:TrainState,
         observations:jax.Array,
         rng:jax.Array) -> Tuple[jax.Array,jax.Array,jax.Array,jax.Array]:
        """returns the selected action by the agent based on its trainstate and the given observations

        Args:
            train_state (TrainState): the current flax.training.train_state of the agent
            observations (jax.Array): the observations on which to select the new action
            rng (jax.Array): a JAX random number

        Returns:
            Tuple[jax.Array,jax.Array,jax.Array,jax.Array]: a tuple containing the following jax arrays:
                                                            policy function pi, critic value, action, action log_probability 
        """
       
        pi, value = train_state.apply_fn(train_state.params, observations)   
        action = pi.sample(seed=rng)
        log_prob = pi.log_prob(action)

        return pi, value, action, log_prob



    @functools.partial(jax.jit, static_argnums=(0,))
    def update_networks(self,update_state:_UpdateState) -> Tuple[_UpdateState,_Loss]:
        """ updates the neural network weights of the agent

        Args:
            update_state (_UpdateState): The actual update state of the agent, 
                                         containing the actual trainstate, the trajectory batch, targets...

        Returns:
            Tuple[_UpdateState,_Loss]: returns the new update state and the calculated total loss
        """
        
        def _update_minbatch(
                  train_state:TrainState, 
                  batch_info:_BatchInfo) ->Tuple[TrainState,_Loss]:
            
            traj_batch, advantages, targets = batch_info

            # calculate the loss function
            def _loss_fn(
                      params:dict, 
                      traj_batch:_Transition, 
                      gae:jax.Array, 
                      targets:jax.Array) -> Tuple[jax.Array,_Loss]:
                
                # regenerate the actions based on the observations recorded during the _env_step
                pi, value = self.network.apply(params, traj_batch.obs)
                log_prob = pi.log_prob(traj_batch.action)

                # calculate the critic value loss
                value_pred_clipped = traj_batch.value + (
                    value - traj_batch.value
                ).clip(-self.agent_config.CLIP_EPS, self.agent_config.CLIP_EPS)
                value_losses = jnp.square(value - targets)
                value_losses_clipped = jnp.square(value_pred_clipped - targets)
                value_loss:jax.Array = (
                    0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                )

                # calculate the actor loss
                ratio = jnp.exp(log_prob - traj_batch.log_prob)
                gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                loss_actor1 = ratio * gae
                loss_actor2 = (
                    jnp.clip(
                        ratio,
                        1.0 - self.agent_config.CLIP_EPS,
                        1.0 + self.agent_config.CLIP_EPS,
                        )
                    * gae
                    )
                loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                loss_actor = loss_actor.mean()
                entropy:jax.Array = pi.entropy().mean()

                total_loss:jax.Array = (
                    loss_actor
                    + self.agent_config.VF_COEF * value_loss
                    - self.agent_config.ENT_COEF * entropy
                    )
                loss = _Loss(value_loss, loss_actor, entropy)
                return total_loss, loss

            grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
            total_loss, grads = grad_fn(
                train_state.params, traj_batch, advantages, targets
                )
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, total_loss

        train_state, traj_batch, advantages, targets, rng = update_state
        
        rng, _rng = jax.random.split(rng)                
        permutation = jax.random.permutation(_rng, self.batch_size)
        batch = (traj_batch, advantages, targets)
     
        shuffled_batch = jax.tree_util.tree_map(
            lambda x: jnp.take(x, permutation, axis=0), batch
            )
        minibatches = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, [self.agent_config.NUM_MINIBATCHES, -1] + list(x.shape[1:])),
            shuffled_batch,
            )
        
        train_state, total_loss = jax.lax.scan(
            _update_minbatch, train_state, minibatches
            )
        update_state = _UpdateState(train_state, traj_batch, advantages, targets, rng)
        return update_state, total_loss
    


    @functools.partial(jax.jit, static_argnums=(0,))
    def update_networks_for_one_epoch(
            self,
            train_state:TrainState,
            last_obs_batch_per_agent:jax.Array,
            traj_batch:_Transition,
            rng:jax.Array) -> Tuple[_UpdateState,_Loss]:
        """update the Neural network weights based on one epoch

        Args:
            train_state (TrainState): the flax.training.train_state of the agent
            last_obs_batch_per_agent (jax.Array): a batch containing the last observations used for the training of the NN
            traj_batch (_Transition): a batch containing the observations trajectory used for the training of the NN
            rng (jax.Array): a JAX random number

        Returns:
            Tuple[_UpdateState,_Loss]: returns the new update state and the calculated total loss
        """

        def _update_networks_for_one_batch(update_state:_UpdateState,unused:Any) -> Tuple[_UpdateState,_Loss]:
            update_state, total_loss = self.update_networks(update_state)
            # update the network based on one minibatch
            return update_state, total_loss
        
        advantages, targets = self._calc_advantages_and_targets(train_state,last_obs_batch_per_agent,traj_batch)
        update_state = _UpdateState(train_state, traj_batch, advantages, targets, rng)
        update_state, loss_info = jax.lax.scan(
                    _update_networks_for_one_batch, update_state, None, self.agent_config.UPDATE_EPOCHS
            )
        return update_state, loss_info