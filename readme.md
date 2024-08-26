# A JAX FRAMEWORK FOR MULTI-AGENT RESEARCH IN DECENTRALISED ENERGY MARKETS

***Author: K. Emerson*** \
***Department of Computer Science*** \
***Faculty of Sciences and Bioengineering Sciences*** \
***Vrije Universiteit Brussel***


## Introduction:

This repository contains a framework, designed to facilitate the investigation of the behavior of multiple agents within a decentralized energy market, wherein each agent represents a market participant capable of producing and/or consuming energy. The framework was implemented as part of a master's thesis at the Computer Science department of the [Vrije Universiteit Brussel](https://www.vub.be/en).

The framework is implemented in Python and [JAX](https://github.com/google/jax) [[1]](#1), a high-performance numerical computing library which supports XLA just-in-time compilation, vectorization and autograd, making the framework suitable for Reinforcement Learning experiments. The implementation of the framework was inspired by the [JaxMARL](https://github.com/FLAIROx/JaxMARL) [[2]](#2) multi-agent Reinforcement Learning Environment library.

## Contents:
[Prerequisites](#Prerequisites)   
[Working Principles](#Working-Principles)  
[Implementation and Folder Structure](#Implementation-and-Folder-Structure)    
[How To Use The Framework](#How-To-Use-The-Framework)  
[Generating Synthetic Actor Energy Consumption DataSets](#Generating-Synthetic-Actor-Energy-Consumption-DataSets)  
[References](#References)

## Prerequisites:
The codebase uses [Poetry](https://python-poetry.org/) v1.7.1 as a dependency management tool and is therefore required to install all dependencies before using the framework.
With Poetry, the dependencies, used by the framework, can be installed by using the following command from within the *src/code* folder:

```
poetry install
``` 

If so desired, the option to run the framework inside a [Visual Studio Code Development Container](https://code.visualstudio.com/docs/devcontainers/containers) is also included. This uses the files *.devcontainer/devcontainer.json* and [*Dockerfile*](Dockerfile) as configuration files to run the framework inside a Docker container. To run the codebase from within a development container, [Microsoft Visual Studio Code](https://code.visualstudio.com/), the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension and [Docker Desktop](https://www.docker.com/products/docker-desktop/) needs to be installed on your local machine.


## Working Principles:

The framework consists of a simulated decentralised energy market in which two or more agents can participate.
At each time step, the agents can submit an offer to buy or sell energy. These offers subsequently serve as the input for clearing an internal market. In the current implementation, the principles of a [double auction](https://en.wikipedia.org/wiki/Double_auction) are used to execute this clearing of the market, but the framework can be extended with other auction mechanisms.

Offers that remain unfulfilled among participants in the internal market will subsequently be addressed within a global market, which encompasses a broader regional or national energy grid. It is posited that this global market possesses an infinite capacity to accommodate all offers that were not cleared in the internal market. To establish the pricing mechanism for these offers, the global market employs a reference day-ahead price, which is subject to variation at each time step. This day-ahead price, in conjunction with a predetermined pricing formula, is utilized to ascertain both a Time-of-Use price, which must be paid for any energy procured from the global market, and a Feed-in price, which is disbursed to any agent supplying energy to the global market.

As stated above, each agent engages with both markets on behalf of a market participant or prosumer, who may function as either a net producer or consumer of energy at any given time. The framework incorporates a distinct entity to model this prosumer behavior, which can be defined through a configurable mathematical expression or by utilising historical data related to production (for instance, from a photovoltaic installation) or consumption patterns. Additionally, the framework includes the provision for individual prosumers to be equipped with a private battery installation of a predetermined size.

To facilitate multiple variations within the experiments, a specialised Transformer class is employed, serving as an interface between the environment and the agents representing the market participants. In addition to its primary technical function of transforming observation and action data types to enable effective communication and processing of inputs between agents and the environment, the Transformer class also permits the manipulation of observations and actions. This capability allows for the modification of what agents perceive, as well as the imposition of constraints on the actions that agents are permitted to undertake. By assigning a distinct and configurable Transformer to each agent, a high level of granularity in control per agent can be achieved. Examples of such variations implemented are: 

* the option to normalise the observations if so required by the agent's underlying learning algorithm.
* the option to clip the agent action amounts within a valid interval to prevent overselling or overbuying with respect to the available energy stored in the agent's batteries.
* the option to clip or normalise the agent action prices within the interval of the current Time-Of_Use and Feed-In price as used by the global market. 
* the option to include a window of day-ahead prices for the coming hours into the agent's observations
* the option to share the actual energy demand and battery level between one or more agents within the market to test the posibility of agent collaboration of market collusion.

Throughout the implementation process, thorough attention was given to ensuring that all modules were adequately decoupled, thereby facilitating expansion or adaptation of the source code to meet the specific requirements of the experiment.


## Implementation and Folder Structure:

The complete codebase of the framework is stored in the *code/src* folder and is organised using the following subfolder structure:


| Folder Name| Description|
|------------|------------|
| **agents**       | Contains the implementation of the agents that will interact with the market environment. A separate factory function and configuration data structure are included to allow creating agents based on a predefined configuration. |
| **auction**      | Contains the actual implementation of the auction as used by the market environment to clear the internal market.                                                                                          |
| **environments** | Contains the implementation of the environment with which the agents interact. The environment uses and manages the global market and prosumer entities and provides each agent with their own observations which in turn, will enable the agent to decide on which offer to make. |
| **experiments**  | This folder stores various experiment configurations, used to test the framework and investigate the agents' behavior under varying conditions.                                                              |
| **globalmarket** | Contains the implementation of the global market. A separate factory function and configuration data structure are included to allow the creation of a global market based on a predefined configuration.     |
| **ledger**       | This folder contains a single file implementing various helper functions to manage and manipulate the market ledger used to collect the agents' offers.                                                      |
| **prosumer**     | Contains the implementation of the prosumer. A separate factory function and configuration data structure are included to allow the creation of a global market based on a predefined configuration.         |
| **transformers** | Contains the implementation of the global market. A separate factory function and configuration data structure are included to allow the creation of a global market based on a predefined configuration.     |


Besides the above mentioned subfolders, the main *code/src* folder also contains the implementation of the actual [training loop](code/src/train.py) and the [main function](code/src/main.py).

To enhance the runtime performance of the framework, the source code is constructed around the utilisation of the [JAX: High-Performance Array Computing](https://jax.readthedocs.io/en/latest/index.html) library. Within this architecture, the main function and training loop serve as pure Python orchestrators, coordinating the invocation of various JAX Just-In-Time (JIT) compatible functions to execute the complete training cycle.

For a more extensive introduction regarding the context, objective and implementation, please consult the accompanying thesis document, added to this repository.

## How To Use The Framework:

In the current setup, each experiment can be constructed by creating a [*ExperimentConfig*](code/src/experiments/experiment_config.py) Dataclass. This contains all parameters to instantiate the different elements required for running the experiment and is used by the [*training_init_and_execute*](code/src/main.py) function to initialize and execute the complete training loop and store all logged metrics within an *experiment_log* folder (which is not included to limit the overall repository size).

To run one or more specific experiments, define the list of *ExperimentConfig* dataclasses in the *experiment_list* variable within [*main.py*](code/src/main.py) and run the the experiment with the following command from within the *code* folder:
```
python src/main.py
```

## Auxiliary Scripts

As previously mentioned, the framework facilitates the utilization of both historical and synthetically generated data to regulate the behavior of prosumers and the global market. To generate a dataset comprising such data, a series of Python scripts have been included in the *data_synthetic_preprocess* folder, enabling the creation of a (synthetic) dataset for both the energy production and consumption of the prosumers and the day-ahead prices of the global market. An example of such datasets is also included in the *data_synthetic* folder.

All these scripts accept various command line arguments to specify how to create the data, as described in the documentation provided within the source code.

### Generating a Historical Day-Ahead Prices Dataset:
To construct a dataset of day-ahead prices, the script titled implemented in [*globalmarket_create_data.py*](code/data_synthetic_preprocess/globalmarket_create_data.py) can be employed. This script utilizes the [ENTSO-E](https://github.com/EnergieID/entsoe-py) library to retrieve historical day-ahead market prices from the [ENTSO-E API](https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html) and thus requires a free account and API-key to execute.

to fetch the actual data use the command from within the *code* folder replacing \<API-key> with your actual API-key: 
```
python data_synthetic_preprocess/globalmarket_create_data.py -a <API-key>
```

### Generating Synthetic Actor Energy Production Datasets:
The framework facilitates the integration of energy-producing installations, such as photovoltaic (PV) systems, for producers. To achieve this, two distinct scripts have been incorporated. The first script, implemented in [*production_create_profiles.py*](code/data_synthetic_preprocess/production_create_profiles.py), generates an arbitrary PV installation profile by randomly selecting parameters, including the number of PV cells, their geographical location, vertical tilt, azimuth, and other relevant factors, all within predefined limits specified in the [*config.ini*](code/data_synthetic_preprocess/config.ini) file. These profiles can subsequently be utilized in conjunction with a second script in [*production_create_data.py*](code/data_synthetic_preprocess/production_create_data.py) that uses the [PVLib](https://github.com/pvlib/pvlib-python) library, which retrieves historical weather data and employs this information to calculate the anticipated energy production for each hour within the designated timeframe. Examples of such energy production data files are included in the folder *code/data_synthetic/producing_actors_hours* folder.

To create the random production profiles, a command such as the following can be used from within the *code* folder, where \<NBR OF ACTORS> is the number of actors for which profiles should be created:

```
python data_synthetic_preprocess/production_create_profiles.py -n <NBR OF ACTORS> 
```
To create the production data based on the above created profiles, a command such as the following can be used from within the *code* folder, where \<NBR OF ACTORS> is the number of actors for which profiles should be created:

```
python data_synthetic_preprocess/production_create_data.py -n <NBR OF ACTORS> 
```

### Generating Synthetic Actor Energy Consumption DataSets:
Finally, two additional scripts are included to generate synthetic consumption data. These scripts are primarily dependent on the [ANTgen](https://gitlab.com/nunovelosa/antgen) library. The library contains a set of appliance and activity profiles which can be assigned to user profiles. Additionally, household profiles can be defined by aggregating multiple users as inhabitants within a single residence. These household profiles are subsequently utilized by the ANTgen library to synthetically generate energy consumption signatures for the home during a preconfigured time window. Examples of such household profiles are included in the folder *code/data_synthetic/consuming_profiles* folder.

To randomly generate the household profiles, a command such as the following can be used from within the *code* folder, where \<NBR OF ACTORS> is the number of actors for which profiles should be created:

```
python data_synthetic_preprocess/consumption_create_profiles.py -n <NBR OF ACTORS> 
```

These household profiles can subsequently be used as a basis for generating random synthetic energy consumption signatures by means of the ANTgen library.  Examples of such household energy consumption signatures are included in the folder *code/data_synthetic/consuming_actors_hours* folder.

To generate such a signature for a specific actor, a command such as the following can be used from within the *code* folder, where \<ACTOR_IDENTIFIER> is the identifier (e.g. actor_00) for which the signature should be created:
```
python data_synthetic_preprocess/consumption_create_data.py <ACTOR_IDENTIFIER> 
````

## References

<a id="1">[1]</a> 
Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C., Maclaurin, D., Necula, G., Paszke, A.,
VanderPlas, J., Wanderman-Milne, S., & Zhang, Q. (2018). ***JAX: Composable transformations of Python+NumPy programs***

<a id="2">[2]</a> 
Rutherford, A., Ellis, B., Gallici, M., Cook, J., Lupu, A., Ingvarsson, G., Willi, T., Khan, A., de Witt,
C. S., Souly, A., Bandyopadhyay, S., Samvelyan, M., Jiang, M., Lange, R. T., Whiteson, S.,
Lacerda, B., Hawes, N., Rocktaschel, T., Lu, C., & Foerster, J. N. (2023, December). ***JaxMARL:
Multi-Agent RL Environments in JAX.***

