""" 
-------------------------------------------------------------------------------------------
A JAX FRAMEWORK FOR MULTI-AGENT RESEARCH IN DECENTRALISED ENERGY MARKETS
-------------------------------------------------------------------------------------------
author: K. Emerson
Department of Computer Science
Faculty of Sciences and Bioengineering Sciences
Vrije Universiteit Brussel
-------------------------------------------------------------------------------------------

File contains the script to create new randomly generated actor household consumption 
data files (in Wh) based on existing household profiles and the predetermined configuration 
variables stored in the config.ini file.
    
The script uses the ANTgen library to generate the actual synthetic data per household.
(For more info on ANTgen, please consult: https://gitlab.com/nunovelosa/antgen)

The script uses some predetermined configuration variables stored in the config.ini file.

To use the script run:
'python data_synthetic_preprocess/consumption_create_data.py <ACTOR> -s <S> -e <E> -a <A>'
The command line arguments do the following:
<ACTOR> is a list of household profiles fo which data needs to be created
-s <S> where <S> is the startdate (YYY-MM-DD) of the energy consumption data
-e <E> where <E> is the enddate (YYY-MM-DD) of the energy consumption data
-a when -a is added the generated data will be appended to existing files instead of creating
    new household consumption files.      
"""

import argparse
import configparser
import datetime
import os

import consumption_helpers as ch
import pandas as pd

if __name__ == "__main__":
    
    DATE_FORMAT = "%Y-%m-%d"
    MAX_NBR_OF_DAYS = 10 # number of days to generate simultaneously

    # get all the configurations from config.ini
    config = configparser.ConfigParser()
    config.read(os.path.join(os.getcwd(),'data_synthetic_preprocess','config.ini'))
    output_path = os.path.join(os.getcwd(),config["SYNTHETIC_DATA"]["synthetic_data_folder"])
    default_startdate = config["SYNTHETIC_DATA"]["startdate"]
    default_enddate = config["SYNTHETIC_DATA"]["enddate"]

    # process the command line arguments
    parser = argparse.ArgumentParser(prog='Agent consumption data creator',description='Create new consumption data for the given actors')
    parser.add_argument('actors', nargs='*')
    parser.add_argument('-s', metavar="YYYY-MM-DD", default=default_startdate, dest="startdate", help="the starting date")
    parser.add_argument('-e', metavar="YYYY-MM-DD", default=default_enddate, dest="enddate", help="the end date")
    parser.add_argument('-a', dest="append",action="store_true", help="appends the data to the exisitng file (if one exists) instead of creating a new file",)
    args = parser.parse_args()
   
    actor_list = args.actors
    startdate = args.startdate
    enddate = args.enddate
    append=args.append

    # create a list of begin dates with an interval equal to MAX_NBR_OF_DAYS
    # then create a list of end dates that correspond to the begin dates in the begin_date_list 
    # finally create a list which calculates the actual number of dates within each interval
    # e.g. startdate = "2024-01-01", enddate = "2024-01-30", MAX_NBR_OF_DAYS = 10 gives: 
    # begin_date_list = ['2024-01-01', '2024-01-11', '2024-01-21']
    # end_date_list = ['2024-01-11', '2024-01-21', '2024-01-30']
    # nbr_of_days_list = [10, 10, 9] 
    begin_date_list = pd.date_range(startdate,enddate,freq=datetime.timedelta(MAX_NBR_OF_DAYS)).strftime(DATE_FORMAT).tolist()
    end_date_list = begin_date_list[1:] + [enddate]
    nbr_of_days_list =[]
    for i,j in zip(begin_date_list,end_date_list):
        nbr_of_days_list.append(((datetime.datetime.strptime(j,DATE_FORMAT)-datetime.datetime.strptime(i,DATE_FORMAT)).days))
     
    # print overview of the files to be generated in terminal
    os.system('cls' if os.name == 'nt' else 'clear')  # noqa: S605
    print("-" *120)
    if args.append:
        print(str.upper(f"This script will append synthetic generated consumption data from {startdate} to {enddate} to the files:"))
    else:
        print(str.upper(f"This script will create synthetic generated consumption data from {startdate} to {enddate} to the new files:"))
    for actor in actor_list:
        print(f"\t{output_path}/{ch.HOURS_DIRECTORY}/{actor}/total.csv")
    print("\n")
    print(str.upper("for the days between:"))
    print(f"\tstartdate : {startdate}")
    print(f"\tenddate (exclusive) : {enddate}")
    print("\n")
    print("-" *120)

    # request confirmation from user
    validate = input("Are you sure you want to continue [YES/NO]:")
    if validate == "YES":
        # for each interval of days as defined in the begin_date_list,nbr_of_days_list 
        for startday,nbr_of_days in zip(begin_date_list,nbr_of_days_list):
            # first create the synthetic consumption per house on a second by second basis
            # and store results in temporary files
            ch.simulate_energy_usage_per_house(nbr_of_days,startday,actor_list=actor_list)
            # convert per second data from temporary files to per hour and store the by hour results
            ch.convert_energy_usage_per_house_to_hours(actor_list=actor_list,append=append)
            append=True
    
    else:
        print("Program terminated by user.")
    
    