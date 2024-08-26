""" 
-------------------------------------------------------------------------------------------
A JAX FRAMEWORK FOR MULTI-AGENT RESEARCH IN DECENTRALISED ENERGY MARKETS
-------------------------------------------------------------------------------------------
author: K. Emerson
Department of Computer Science
Faculty of Sciences and Bioengineering Sciences
Vrije Universiteit Brussel
-------------------------------------------------------------------------------------------

File contains the script to create new energy production data files (in Wh) for one or more 
energy production actors based on existing energy production actor profiles and the predefined 
variables stored in the config.ini file.
    
The script uses the PVLib library to simulate the performance of the Photovoltaic installations
of the energy production actors based on actual historic weather data.
(For more info on PVLib, please consult: https://pvlib-python.readthedocs.io/en/stable/index.html)

The script uses some predetermined configuration variables stored in the config.ini file.
To use the script run:
'python data_synthetic_preprocess/production_create_data.py -n <N> -i <I> -s <S> -e <E>'
The command line arguments do the following:
-n <N> where <N> is the number of actors for which to create the production data
-i <I> where <I> is index of the first actor for which to create the data
-s <S> where <S> is the startyear (YYYY) from which to start the data (optional)
-e <E> where <E> is the endyear (YYYY) upon which to create the data (optional)
"""

import argparse
import configparser
import os

import pandas as pd
import production_helpers as ph

if __name__ == "__main__":
    
    # get all the configurations from config.ini
    config = configparser.ConfigParser()
    config.read(os.path.join(os.getcwd(),'data_synthetic_preprocess','config.ini'))
    output_path = os.path.join(os.getcwd(),config['SYNTHETIC_DATA']['synthetic_data_folder'])
    output_directory = os.path.join(output_path,config['SYNTHETIC_DATA']['production_data_subfolder'])
    latitude = float(config['SYNTHETIC_DATA.LOCALISATION']['latitude'])
    longitude = float(config['SYNTHETIC_DATA.LOCALISATION']['longitude'])
    default_startyear = (pd.Timestamp(config["SYNTHETIC_DATA"]["startdate"]).year)
    default_endyear = (pd.Timestamp(config["SYNTHETIC_DATA"]["enddate"]).year-1) # in this api year is inclusive

    # process the command line arguments
    parser = argparse.ArgumentParser(prog='Production actor creator',description='Create new random production actors')
    parser.add_argument('-n', metavar="X", type=int, required=True, dest="nbr_of_actors", help="the number of actors to create")
    parser.add_argument('-i', metavar="X", type=int, default=0, dest="init_nbr_of_actor", help="the number for the first actor")
    parser.add_argument('-s', metavar="YYYY",type=int, default=default_startyear, dest="startyear", help="the starting year to generate the data")
    parser.add_argument('-e', metavar="YYYY",type=int, default=default_endyear, dest="endyear", help="the end year to generate the data")
    args = parser.parse_args()
        
    startdate = args.startyear
    enddate = args.endyear
    actors = ph.load_profiles_from_file()
    
    # only use those actors as required by command line arguments
    nbr_of_actors = args.nbr_of_actors if not args.nbr_of_actors == 0 else len(actors) 
    start_nbr_of_actor = args.init_nbr_of_actor
    actors_filtered = actors[start_nbr_of_actor:start_nbr_of_actor + nbr_of_actors]
    
    # print overview of what will be generated in terminal
    os.system('cls' if os.name == 'nt' else 'clear')  # noqa: S605
    print("-" *120)
    print(str.upper(f"This script will create new actor production data from {startdate} to {enddate} in:"))
    for actor in actors_filtered:
        outname = os.path.join(output_directory, f"{actor.name}.csv")
        print(f"\t{outname}")
    print("-" *120)
    
    # request confirmation from user
    validate = input("Are you sure you want to continue [YES/NO]:")
    if validate == "YES":

        weather = ph.get_weather_data(latitude,longitude,startdate,enddate)
        actors_prod = ph.calc_multi_actor_PV_production(weather,actors_filtered)
        ph.save_actor_production(actors_prod) 
    else:
        print("Program terminated by user.")

