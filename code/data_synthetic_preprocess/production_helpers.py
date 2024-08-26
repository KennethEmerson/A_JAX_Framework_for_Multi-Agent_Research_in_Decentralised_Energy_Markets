"""
-------------------------------------------------------------------------------------------
A JAX FRAMEWORK FOR MULTI-AGENT RESEARCH IN DECENTRALISED ENERGY MARKETS
-------------------------------------------------------------------------------------------
author: K. Emerson
Department of Computer Science
Faculty of Sciences and Bioengineering Sciences
Vrije Universiteit Brussel
-------------------------------------------------------------------------------------------
This file contains several helpers functions to create the production profiles and data.
It uses the PVLib library to simulate the performance of the Photovoltaic installations
of the energy production actors based on actual historic weather data.
(For more info on PVLib, please consult: https://pvlib-python.readthedocs.io/en/stable/index.html)

"""
import configparser
import os
import random
from typing import Dict, List, NamedTuple

import pandas as pd
import pvlib
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.pvsystem import PVSystem

###################################################################
# retrieve contants for config.ini
###################################################################
config = configparser.ConfigParser()
config.read(os.path.join(os.getcwd(),'data_synthetic_preprocess','config.ini'))

ACTOR_NAME_PREFIX = config['GENERAL']['actor_prefix']
OUTPUT_PATH = os.path.join(os.getcwd(),config['SYNTHETIC_DATA']['synthetic_data_folder'])
PRODUCTION_PATH = config['SYNTHETIC_DATA']['production_data_subfolder']
PRODUCING_ACTORS_FILENAME = config['SYNTHETIC_DATA']['production_actors_filename']

LATITUDE = float(config['SYNTHETIC_DATA.LOCALISATION']['latitude'])
LONGITUDE = float(config['SYNTHETIC_DATA.LOCALISATION']['longitude'])
ALTITUDE = int(config['SYNTHETIC_DATA.LOCALISATION']['altitude'])
TIMEZONE = config['SYNTHETIC_DATA.LOCALISATION']['timezone']

SANDIA_MODULES = pvlib.pvsystem.retrieve_sam('SandiaMod')
SAPM_INVERTERS = pvlib.pvsystem.retrieve_sam('cecinverter')
PV_MODULE = SANDIA_MODULES[config['SYNTHETIC_DATA.PV_INSTALLATION']['pv_module']]
PV_INVERTER = SAPM_INVERTERS[config['SYNTHETIC_DATA.PV_INSTALLATION']['pv_inverter']]
TEMP_MODEL_PARAMETERS = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

TILT_MIN = int(config['SYNTHETIC_DATA.PV_INSTALLATION']['tilt_min'])
TILT_MAX = int(config['SYNTHETIC_DATA.PV_INSTALLATION']['tilt_max'])
AZIMUTH_MIN = int(config['SYNTHETIC_DATA.PV_INSTALLATION']['azimuth_min'])
AZIMUTH_MAX = int(config['SYNTHETIC_DATA.PV_INSTALLATION']['azimuth_max'])
MOD_PER_STING_MIN = int(config['SYNTHETIC_DATA.PV_INSTALLATION']['modules_per_string_min'])
MOD_PER_STING_MAX = int(config['SYNTHETIC_DATA.PV_INSTALLATION']['modules_per_string_max'])
STRINGS_PER_INVERTER_MIN = int(config['SYNTHETIC_DATA.PV_INSTALLATION']['strings_per_inverter_min'])
STRINGS_PER_INVERTER_MAX = int(config['SYNTHETIC_DATA.PV_INSTALLATION']['strings_per_inverter_max'])
BATTERY_CAPACITY_MIN_Wh = int(config['SYNTHETIC_DATA.PV_INSTALLATION']['battery_capacity_min_Wh'])
BATTERY_CAPACITY_MAX_Wh = int(config['SYNTHETIC_DATA.PV_INSTALLATION']['battery_capacity_max_Wh'])
BATTERY_PEAK_MIN_Wh = int(config['SYNTHETIC_DATA.PV_INSTALLATION']['battery_peak_min_Wh'])
BATTERY_PEAK_MAX_Wh = int(config['SYNTHETIC_DATA.PV_INSTALLATION']['battery_peak_max_Wh'])


class ProductionProfile(NamedTuple):
    """ NamedTuple containing all required production actor profile info
        required to generate the production data. 
    """
    name: str
    latitude: float # coordinate of actor on earth
    longitude:float # coordinate of actor on earth
    altitude:int # altitude of actor on earth starting from sea level
    tilt:int # the vertical tilt in degrees of the solar panels (0°: horizontal)
    azimuth:int # the angle of the solar panels in relation to due south
    modules_per_string:int # number of solar panels on one electrical circuit
    strings_per_inverter:int # number of electrical circuits per inverter
    timezone:str # timezone of the actor.



###################################################################
# Helper functions
###################################################################

def create_random_profiles(nbr_of_actors:int,start_nbr_actor:int=0)->List[ProductionProfile]:
    """ create random production actor profiles within the limits set in config.ini 

    Args:
        nbr_of_actors (int): number of actors for which to generate a profile
        start_nbr_actor (int, optional): index of first actor to create a profile. Defaults to 0.

    Returns:
        List[ProductionProfile]: creates a list of profiles
    """
    actor_list = []

    # randomly create a profile for each production agent
    for i in range(start_nbr_actor,start_nbr_actor + nbr_of_actors):
        actor = ProductionProfile(
            name = f"{ACTOR_NAME_PREFIX}_{i:02d}",
            latitude = LATITUDE,
            longitude = LONGITUDE,
            altitude = ALTITUDE,
            tilt = random.randrange(TILT_MIN,TILT_MAX+1), # noqa: S311
            azimuth = random.randrange(AZIMUTH_MIN,AZIMUTH_MAX+1), # noqa: S311
            modules_per_string = random.randrange(MOD_PER_STING_MIN,MOD_PER_STING_MAX+1), # noqa: S311
            strings_per_inverter = random.randrange(STRINGS_PER_INVERTER_MIN,STRINGS_PER_INVERTER_MAX+1), # noqa: S311
            timezone = TIMEZONE,
            
        )   
        actor_list.append(actor)
    return actor_list     



def save_profiles_to_file(actors:List[ProductionProfile],append:bool=False) -> None:
    """saves the list of profiles in a file

    Args:
        actors (List[ProductionProfile]): list of actor profiles
        append (bool, optional): if true the given list will be appendend to the existing file, if false, the old file will 
                                 be overwritten. Defaults to False.
    """
    
    # create a dataframe with all the profiles
    df =pd.DataFrame(actors, columns =[
        "name",
        "latitude",
        "longitude",
        "altitude",
        "tilt",
        "azimuth",
        "modules_per_string",
        "strings_per_inverter",
        "timezone",
    ])
    
    # store data
    outname = os.path.join(OUTPUT_PATH, f"{PRODUCING_ACTORS_FILENAME}.csv")
    append_file = os.path.exists(outname) and append
    mode = 'a' if append_file else 'w'
    with open(outname, mode) as csv_file:
                df.to_csv(csv_file, sep=';',index=False)



def load_profiles_from_file() -> List[ProductionProfile]:
    """ loads existing production actor profile from file

    Returns:
        List[ProductionProfile]: a list of production actor profiles
    """
    
    with open(os.path.join(OUTPUT_PATH, f"{PRODUCING_ACTORS_FILENAME}.csv"), 'r') as csv_file:
        df = pd.read_csv(csv_file,header=0,sep=';',dtype={
            "name":str,
            "latitude":float,
            "longitude":float,
            "altitude":float,
            "tilt":int,
            "azimuth":int,
            "modules_per_string":int,
            "strings_per_inverter":int,
            "timezone":str,
        })
    
    # read and process line by line
    profile_list = []
    for i in df.index:
        actor = ProductionProfile(
            name= str(df['name'][i]),
            latitude= df['latitude'][i], # type: ignore
            longitude=df['longitude'][i], # type: ignore
            altitude=df['altitude'][i], # type: ignore
            tilt=df['tilt'][i], # type: ignore
            azimuth=df['azimuth'][i], # type: ignore
            modules_per_string=df['modules_per_string'][i], # type: ignore
            strings_per_inverter=df['strings_per_inverter'][i], # type: ignore
            timezone=df['timezone'][i], # type: ignore
        )
        profile_list.append(actor)
    return profile_list
    


def get_weather_data(latitude:float,
                     longitude:float,
                     start_year:int,
                     end_year:int)->pd.DataFrame:
    """ get the weather data using the PVLib library

    Args:
        latitude (float): Latitude for which to fetch the weather data
        longitude (float): Longitude for which to fetch the weather data
        start_year (int): startyear (YYYY) from which to fetch the weather data
        end_year (int): end year (YYYY) upon which to fetch the weather data

    Returns:
        pd.DataFrame: a dataframe containing the weather data
    """

    print("Collect weather data from api")
    api_data = pvlib.iotools.get_pvgis_hourly(
        latitude, longitude, start=start_year,end=end_year,url='https://re.jrc.ec.europa.eu/api/v5_2/',map_variables=False)
    weather = api_data[0]
    weather['G(i)'] = weather['Gb(i)'] + weather['Gd(i)']+ weather['Gr(i)']
    weather = weather.rename(columns={
                                'Gb(i)': 'dni',
                                'Gd(i)': 'dhi',
                                'T2m' : "temp_air",
                                'WS10m': "wind_speed",
                                'G(i)': 'ghi'
                            })
    return weather



def calculate_PV_production(weather_data:pd.DataFrame,actor:ProductionProfile) -> Dict[str,pd.DataFrame]: 
    """ simulate the expected energy production for the PhotoVoltaic cells of the actor based on the weather data
        and the actor production profile    

    Args:
        weather_data (pd.DataFrame): the dataframe containing the weather data
        actor (ProductionProfile): the prodcution profile of the actor

    Raises:
        Exception: when soemthing went wrong during the calculation of the data

    Returns:
        Dict[str,pd.DataFrame]: a dict with the actor name as key and the dataframe 
                                with the calculated energy production as value
    """

    print(f"Calculating PV energy production for {actor.name}")
    location = Location(
        actor.latitude,
        actor.longitude,
        name=actor.name,
        altitude=actor.altitude,
        tz=actor.timezone,
    )
     
    system = PVSystem(
        surface_tilt= actor.tilt,
        surface_azimuth=actor.azimuth,
        module_parameters= PV_MODULE,
        temperature_model_parameters=TEMP_MODEL_PARAMETERS,
        inverter_parameters = PV_INVERTER,
        modules_per_string = actor.modules_per_string,
        strings_per_inverter = actor.strings_per_inverter        
    )
    mc = ModelChain(system, location)
    mc.run_model(weather_data)
    calc_energy = mc.results.ac
    if calc_energy is None:
        raise Exception(f"{actor.name} energy production could not be calculated")
    else:
        df = pd.DataFrame(calc_energy.to_frame().reset_index()).rename(columns={'time':"Datetime",0:"Energy_Produced_Wh"})
        return {actor.name:df}



def calc_multi_actor_PV_production(weather:pd.DataFrame,profiles:List[ProductionProfile])-> Dict[str,pd.DataFrame]:
    """ simulate the expected energy production for the PhotoVoltaic cells of multiple actors based on the weather data
        and the actors production profiles    

    Args:
        weather (pd.DataFrame): the dataframe containing the weather data
        profiles (List[ProductionProfile]): A list of prodcution profiles of the actors

    Returns:
        Dict[str,pd.DataFrame]: a dict with the actor names as keys and a dataframe 
                                with their respective calculated energy production as values
    """
    
    energies = {}
    for profile in profiles:
        energy_prod = calculate_PV_production(weather,profile)
        energies.update(energy_prod)
    
    return energies



def save_actor_production(actors_pv_production:Dict[str,pd.DataFrame]) -> None:
    """ save the generated energy production data of an actor

    Args:
        actors_pv_production (Dict[str,pd.DataFrame]): the dict with the actor name as key 
                                                       and the dataframe with the calculated energy production as value
    """
    
    output_directory = os.path.join(OUTPUT_PATH,PRODUCTION_PATH)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    
    for actor, energy  in actors_pv_production.items():
        print(f"Saving PV energy production for {actor}")
        outname = os.path.join(output_directory, f"{actor}.csv")
        energy["Datetime"] =energy["Datetime"].dt.tz_localize(None)
        with open(outname, "w") as csv_file:
            energy.to_csv(csv_file, columns=["Datetime", "Energy_Produced_Wh"], sep=";", index=False)



def load_actor_production(actor_list:List[str]=[]) -> Dict[str,pd.DataFrame]: 
    """ load the energy production data for a list of actors

    Args:
        actor_list (List[str], optional): the list of actors for which to load the data, 
                                            if not included, all actor data is loaded. Defaults to [].

    Returns:
        Dict[str,pd.DataFrame]: a dict with the actor names as keys and a dataframe 
                                with their respective calculated energy production as values
    """
    
    files_to_load = list(map(lambda x: f"{x}.csv",actor_list))
    actors_production_dict = {}
    
    if not files_to_load:
        files_list = [f for f in os.listdir(os.path.join(OUTPUT_PATH,PRODUCTION_PATH)) 
                      if os.path.isfile(os.path.join(OUTPUT_PATH,PRODUCTION_PATH, f)) and f.endswith(".csv")]
    else:
        files_list = files_to_load
    
    for file in files_list:
        with open(os.path.join(OUTPUT_PATH,PRODUCTION_PATH,file),"r") as csv_file:
            df = pd.read_csv(csv_file, header=0,sep=';')  
        actors_production_dict[file.rsplit( ".", 1 )[ 0 ]] = df
    
    return actors_production_dict



def load_agent_production(agent_id:str) -> Dict[str,pd.DataFrame]:
    """ loads the energy production for one specific agent with its corresponding actor data

    Args:
        agent_id (str): the name of the agent. must be of type "agent_XX" where XX is 
                        the index that corresponds to the actor index

    Returns:
        Dict[str,pd.DataFrame]: a dict with the agent_id as key and a dataframe as production values
    """
  
    index = agent_id.split("_")[1]
    actor_name = "actor_" + index
    result = load_actor_production([actor_name])
    return {agent_id:result[actor_name]}
