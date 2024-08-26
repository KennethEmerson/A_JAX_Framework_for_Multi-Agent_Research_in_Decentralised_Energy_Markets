'''
THIS FILE IS A COPY FROM THE JAXMARL LIBRARY
https://github.com/FLAIROx/JaxMARL/blob/main/jaxmarl/environments/spaces.py

'''

from collections import OrderedDict
from typing import Any, Sequence, Tuple, Union  # type: ignore

import chex
import jax
import jax.numpy as jnp


class Space(object):
    """
    Minimal jittable class for abstract jaxmarl space.
    """

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        raise NotImplementedError

    def contains(self, x: jnp.int_) -> bool:
        raise NotImplementedError

class Discrete(Space):
	"""
	Minimal jittable class for discrete gymnax spaces.
	TODO: For now this is a 1d space. Make composable for multi-discrete.
	"""

	def __init__(self, num_categories: int, dtype=jnp.int32) -> None:
		assert num_categories >= 0
		self.n = num_categories
		self.shape = ()
		self.dtype = dtype

	def sample(self, rng: chex.PRNGKey) -> chex.Array:
		"""Sample random action uniformly from set of categorical choices."""
		return jax.random.randint(
			rng, shape=self.shape, minval=0, maxval=self.n
		).astype(self.dtype)

	def contains(self, x: jnp.int_) -> bool:
		"""Check whether specific object is within space."""
		# type_cond = isinstance(x, self.dtype)
		# shape_cond = (x.shape == self.shape)
		range_cond = jnp.logical_and(x >= 0, x < self.n)
		return range_cond # type: ignore


class MultiDiscrete(Space):
    """
    Minimal jittable class for multi-discrete gymnax spaces.
    """

    def __init__(self, num_categories: Sequence[int]) -> None:
        """Num categories is the number of cat actions for each dim, [2,2,2]=2 actions x 3 dim"""
        self.num_categories = jnp.array(num_categories)
        self.shape = (len(num_categories),)
        self.dtype = jnp.int_

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        """Sample random action uniformly from set of categorical choices."""
        return jax.random.randint(
            rng, 
            shape=self.shape, 
            minval=0, 
            maxval=self.num_categories,
            dtype=self.dtype
        )

    def contains(self, x: jnp.int_) -> bool:
        """Check whether specific object is within space."""
        range_cond = jnp.logical_and(x >= 0, x < self.num_categories)
        return jnp.all(range_cond)  # type: ignore


class Box(Space):
	"""
	Minimal jittable class for array-shaped gymnax spaces.
	TODO: Add unboundedness - sampling from other distributions, etc.
	"""
	def __init__(
		self,
		low: float,
		high: float,
		shape: Tuple[int], # type: ignore
		dtype: jnp.dtype = jnp.float32,
	) -> None:
		self.low = low
		self.high = high
		self.shape = shape
		self.dtype = dtype

	def sample(self, rng: chex.PRNGKey) -> chex.Array:
		"""Sample random action uniformly from 1D continuous range."""
		return jax.random.uniform(
			rng, shape=self.shape, minval=self.low, maxval=self.high # type: ignore
		).astype(self.dtype)

	def contains(self, x: jnp.int_) -> bool:
		"""Check whether specific object is within space."""
		# type_cond = isinstance(x, self.dtype)
		# shape_cond = (x.shape == self.shape)
		range_cond = jnp.logical_and(
			jnp.all(x >= self.low), jnp.all(x <= self.high)
		)
		return range_cond # type: ignore


class Dict(Space):
    """Minimal jittable class for dictionary of simpler jittable spaces."""
    def __init__(self, spaces: dict) -> None:
        self.spaces = spaces
        self.num_spaces = len(spaces)
    
    def __getitem__(self,item:str) -> Space:
        return self.spaces[item]

    def sample(self, rng: chex.PRNGKey) -> dict:
        """Sample random action from all subspaces."""
        key_split = jax.random.split(rng, self.num_spaces)
        return OrderedDict(
			[
				(k, self.spaces[k].sample(key_split[i]))
				for i, k in enumerate(self.spaces)
			]
		)

    def contains(self, x: jnp.int_) -> bool:
        """Check whether dimensions of object are within subspace."""
		# type_cond = isinstance(x, dict)
		# num_space_cond = len(x) != len(self.spaces)
		# Check for each space individually
        out_of_space = 0
        for k, space in self.spaces.items():
            out_of_space += 1 - space.contains(getattr(x, k))
        return out_of_space == 0
	


class Tuple(Space):
	"""Minimal jittable class for tuple (product) of jittable spaces."""
	def __init__(self, spaces: Union[tuple, list]) -> None:
		self.spaces = spaces
		self.num_spaces = len(spaces)

	def sample(self, rng: chex.PRNGKey) -> Tuple[chex.Array]: # type: ignore
		"""Sample random action from all subspaces."""
		key_split = jax.random.split(rng, self.num_spaces)
		return tuple(
			[
				space.sample(key_split[i])
				for i, space in enumerate(self.spaces)
			]
		) # type: ignore

	def contains(self, x: jnp.int_) -> bool:
		"""Check whether dimensions of object are within subspace."""
		# type_cond = isinstance(x, tuple)
		# num_space_cond = len(x) != len(self.spaces)
		# Check for each space individually
		out_of_space = 0
		for space in self.spaces:
			out_of_space += 1 - space.contains(x)
		return out_of_space == 0