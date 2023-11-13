"""power-train digital twin.
"""
# To Do in this pacakge:

import functools
import itertools

import logging

import pandas as pd
import numpy as np
import scipy.optimize

import tqdm

logger = logging.getLogger(__name__)



def calculate_trip_total_energy(power_percentage, MCR_oversize, time_percentage, trip_total_time_hr):
    power_applied = np.array(power_percentage) * MCR_oversize
    time = np.array(time_percentage) * trip_total_time_hr
    
    trip_total_energy_kwh = sum(np.multiply(power_applied,time))

    return trip_total_energy_kwh

# def calculate_load_factor_tw_ave(MCR_smallsize_1, MCR_smallsize_2):
    
#     power_applied = [86*0.5,147*0.5,208*0.5,270*0.5,331*0.5,392*0.5,453*0.5,515*0.5,576*0.5,637*0.5,698*0.5,760*0.5,821*0.5,882*0.5,943*0.5,1005*0.5,1066*0.5] # exclude stationary
#     time_percentage = [0.046,0.059,0.074,0.067,0.093,0.117,0.118,0.067,0.058,0.04,0.03,0.027,0.013,0.007,0.004,0.004,0.002] # exclude stationary

#     load_factor_smallsize_1 = power_applied/ MCR_smallsize_1
#     load_factor_tw_ave_smallsize_1 = ((86*0.5*0.046)/MCR_smallsize_1 + (147*0.5*0.059)/MCR_smallsize_1 + (208*0.5*0.074)/MCR_smallsize_1 + (270*0.5*0.067)/MCR_smallsize_1 + (331*0.5*0.093)/MCR_smallsize_1 +(392*0.5*0.117)/MCR_smallsize_1 +(453*0.5*0.118)/MCR_smallsize_1 + (515*0.5*0.067)/MCR_smallsize_1 + (576*0.5*0.058)/MCR_smallsize_1 + (637*0.5*0.04)/MCR_smallsize_1 + (698*0.5*0.03)/MCR_smallsize_1 +(760*0.5*0.027)/MCR_smallsize_1 + (821*0.5*0.013)/MCR_smallsize_1 + (882*0.5*0.007)/MCR_smallsize_1 + (943*0.5*0.004)/MCR_smallsize_1 + (1005*0.5*0.004)/MCR_smallsize_1 + (1066*0.5*0.002)/MCR_smallsize_1)/ sum(time_percentage)
#     load_factor_tw_ave_smallsize_1 = np.average(int(load_factor_smallsize_1), int(time_percentage))
    
#     load_factor_smallsize_2 = power_applied/ MCR_smallsize_2
#     load_factor_tw_ave_smallsize_2 = np.average(int(load_factor_smallsize_2), int(time_percentage))
    
#     return load_factor_tw_ave_smallsize_1,load_factor_tw_ave_smallsize_2

def calculate_battery_capacity_forARA2smallICE(peakpower_load_oversize,stationarypower_load_oversize, MCR_oversize, peaktime_percentage, stationarytime_percentage, trip_total_time_hr, MCR_smallsize_1, MCR_smallsize_2):
    
    # get peak energy
    peakpower_applied = peakpower_load_oversize * MCR_oversize
    peaktime = peaktime_percentage * trip_total_time_hr
    peakpower_total_energy_kwh = peakpower_applied * peaktime        
    peak_load_factor4smallsize_1 = 0.85
    peak_load_factor4smallsize_2 = 0.85
    peak_ICE_energy_kwh = peaktime * (MCR_smallsize_1 * peak_load_factor4smallsize_1 + MCR_smallsize_2 * peak_load_factor4smallsize_2)     
    #get stationary energy
    stationarypower_applied = stationarypower_load_oversize * MCR_oversize
    stationarytime = stationarytime_percentage * trip_total_time_hr
    stationarypower_total_energy_kwh = stationarypower_applied * stationarytime      
    # get battery_capacity_kwh  
    battery_capacity_for_peakpower_kwh = peakpower_total_energy_kwh - peak_ICE_energy_kwh
    battery_capacity_for_stationarypower_kwh = stationarypower_total_energy_kwh
    battery_capacity_for_trip_kwh = (battery_capacity_for_peakpower_kwh + battery_capacity_for_stationarypower_kwh)/0.8
    
    # get load factors for various engine size choice

    time_percentage = [0.046/0.826,0.059/0.826,0.074/0.826,0.067/0.826,0.093/0.826,0.117/0.826,0.118/0.826,0.067/0.826,0.058/0.826,0.04/0.826,0.03/0.826,0.027/0.826,0.013/0.826,0.007/0.826,0.004/0.826,0.004/0.826,0.002/0.826] # exclude stationary
    

    load_factor_tw_ave_ARA_smallsize = ((86*0.046)/MCR_smallsize_1 +
                                        (147*0.059)/MCR_smallsize_1 +
                                        (208*0.074)/MCR_smallsize_1 +
                                        (270*0.067)/MCR_smallsize_1 +
                                        (331*0.093)/MCR_smallsize_1 + 
                                        (392*0.117)/(MCR_smallsize_1) +
                                        (450*0.118)/(MCR_smallsize_1) +
                                        (515*0.067)/(MCR_smallsize_1+MCR_smallsize_2) +
                                        (576*0.058)/(MCR_smallsize_1+MCR_smallsize_2) +
                                        (637*0.04)/(MCR_smallsize_1+MCR_smallsize_2) +
                                        (698*0.03)/(MCR_smallsize_1+MCR_smallsize_2) +
                                        (0.85*0.027)+
                                        (0.85*0.013)+
                                        (0.85*0.007)+
                                        (0.85*0.004)+
                                        (0.85*0.004)+
                                        (0.85*0.002))/sum(time_percentage)
        
    return battery_capacity_for_trip_kwh, battery_capacity_for_peakpower_kwh,load_factor_tw_ave_ARA_smallsize

def calculate_battery_capacity_forARA1smallICE(peakpower_load_oversize,stationarypower_load_oversize, MCR_oversize, peaktime_percentage, stationarytime_percentage, trip_total_time_hr, MCR_smallsize_1):
    
    # get peak energy
    peakpower_applied = peakpower_load_oversize * MCR_oversize
    peaktime = peaktime_percentage * trip_total_time_hr
    peakpower_total_energy_kwh = peakpower_applied * peaktime        
    peak_load_factor4smallsize = 0.85
    
    peak_ICE_energy_kwh = peaktime * (MCR_smallsize_1 * peak_load_factor4smallsize)     
    #get stationary energy
    stationarypower_applied = stationarypower_load_oversize * MCR_oversize
    stationarytime = stationarytime_percentage * trip_total_time_hr
    stationarypower_total_energy_kwh = stationarypower_applied * stationarytime      
    # get battery_capacity_kwh  
    battery_capacity_for_peakpower_kwh = peakpower_total_energy_kwh - peak_ICE_energy_kwh
    battery_capacity_for_stationarypower_kwh = stationarypower_total_energy_kwh
    battery_capacity_for_trip_kwh = (battery_capacity_for_peakpower_kwh + battery_capacity_for_stationarypower_kwh)/0.8
    
    # get load factors for various engine size choice

    time_percentage = [0.046/0.826,0.059/0.826,0.074/0.826,0.067/0.826,0.093/0.826,0.117/0.826,0.118/0.826,0.067/0.826,0.058/0.826,0.04/0.826,0.03/0.826,0.027/0.826,0.013/0.826,0.007/0.826,0.004/0.826,0.004/0.826,0.002/0.826] # exclude stationary

    load_factor_tw_ave_ARA_single_smallsize = ((86*0.046)/MCR_smallsize_1 + 
                                               (147*0.059)/MCR_smallsize_1 + 
                                               (208*0.074)/MCR_smallsize_1 + 
                                               (270*0.067)/MCR_smallsize_1 + 
                                               (331*0.093)/MCR_smallsize_1 +
                                               (392*0.117)/MCR_smallsize_1 +
                                               (453*0.118)/MCR_smallsize_1 + 
                                               (515*0.067)/MCR_smallsize_1 + 
                                               (576*0.058)/MCR_smallsize_1 + 
                                               (637*0.04)/MCR_smallsize_1 + 
                                               (698*0.03)/MCR_smallsize_1 +
                                               (0.85*0.027)+
                                               (0.85*0.013)+
                                               (0.85*0.007)+
                                               (0.85*0.004)+
                                               (0.85*0.004)+
                                               (0.85*0.002))/sum(time_percentage)
    
    return battery_capacity_for_trip_kwh, battery_capacity_for_peakpower_kwh,load_factor_tw_ave_ARA_single_smallsize

def calculate_battery_capacity_forRhine2smallICE(peakpower_load_oversize,stationarypower_load_oversize, MCR_oversize, peaktime_percentage, stationarytime_percentage, trip_total_time_hr, MCR_smallsize_1, MCR_smallsize_2):
    
    # get peak energy
    peakpower_applied = peakpower_load_oversize * MCR_oversize
    peaktime = peaktime_percentage * trip_total_time_hr
    peakpower_total_energy_kwh = peakpower_applied * peaktime        
    peak_load_factor4smallsize_1 = 0.85
    peak_load_factor4smallsize_2 = 0.85
    peak_ICE_energy_kwh = peaktime * (MCR_smallsize_1 * peak_load_factor4smallsize_1 + MCR_smallsize_2 * peak_load_factor4smallsize_2)     
    #get stationary energy
    stationarypower_applied = stationarypower_load_oversize * MCR_oversize
    stationarytime = stationarytime_percentage * trip_total_time_hr
    stationarypower_total_energy_kwh = stationarypower_applied * stationarytime      
    # get battery_capacity_kwh, use a maximum of 80% of the capacity  
    battery_capacity_for_peakpower_kwh = peakpower_total_energy_kwh - peak_ICE_energy_kwh
    battery_capacity_for_stationarypower_kwh = stationarypower_total_energy_kwh
    battery_capacity_for_trip_kwh = (battery_capacity_for_peakpower_kwh + battery_capacity_for_stationarypower_kwh)/0.8
    
    # get load factors for various engine size choice

    time_percentage = [0.021/0.856,0.015/0.856,0.039/0.856,0.036/0.856,0.037/0.856,0.056/0.856,
                       0.103/0.856,0.083/0.856,0.055/0.856,0.053/0.856,0.042/0.856,0.058/0.856,
                       0.059/0.856,0.07/0.856,0.059/0.856,
                       0.027/0.856,0.011/0.856,0.014/0.856,0.014/0.856,0.004/0.856] # exclude stationary    

    load_factor_tw_ave_Rhine_smallsize = ((86*0.021)/MCR_smallsize_1 +
                                        (147*0.015)/MCR_smallsize_1 +
                                        (208*0.039)/MCR_smallsize_1 +
                                        (270*0.036)/MCR_smallsize_1 +
                                        (331*0.037)/MCR_smallsize_1 + 
                                        (392*0.056)/(MCR_smallsize_1) +
                                        (450*0.103)/(MCR_smallsize_1) +
                                        (515*0.083)/(MCR_smallsize_1) +
                                        (576*0.055)/(MCR_smallsize_1+MCR_smallsize_2) +
                                        (637*0.053)/(MCR_smallsize_1+MCR_smallsize_2) +
                                        (698*0.042)/(MCR_smallsize_1+MCR_smallsize_2) +
                                        (760*0.058)/(MCR_smallsize_1+MCR_smallsize_2) +
                                        (821*0.059)/(MCR_smallsize_1+MCR_smallsize_2) +
                                        (882*0.07)/(MCR_smallsize_1+MCR_smallsize_2) +
                                        (943*0.059)/(MCR_smallsize_1+MCR_smallsize_2) +
                                        (0.85*0.027)+
                                        (0.85*0.011)+
                                        (0.85*0.014)+
                                        (0.85*0.014)+
                                        (0.85*0.004))/sum(time_percentage)
        
    return battery_capacity_for_trip_kwh, battery_capacity_for_peakpower_kwh,load_factor_tw_ave_Rhine_smallsize


def calculate_battery_capacity_forRhine1smallICE(peakpower_load_oversize,stationarypower_load_oversize, MCR_oversize, peaktime_percentage, stationarytime_percentage, trip_total_time_hr, MCR_smallsize_1):
    
    # get peak energy
    peakpower_applied = peakpower_load_oversize * MCR_oversize
    peaktime = peaktime_percentage * trip_total_time_hr
    peakpower_total_energy_kwh = peakpower_applied * peaktime        
    peak_load_factor4smallsize = 0.85
    
    peak_ICE_energy_kwh = peaktime * (MCR_smallsize_1 * peak_load_factor4smallsize)     
    #get stationary energy
    stationarypower_applied = stationarypower_load_oversize * MCR_oversize
    stationarytime = stationarytime_percentage * trip_total_time_hr
    stationarypower_total_energy_kwh = stationarypower_applied * stationarytime      
    # get battery_capacity_kwh  
    battery_capacity_for_peakpower_kwh = peakpower_total_energy_kwh - peak_ICE_energy_kwh
    battery_capacity_for_stationarypower_kwh = stationarypower_total_energy_kwh
    battery_capacity_for_trip_kwh = (battery_capacity_for_peakpower_kwh + battery_capacity_for_stationarypower_kwh)/0.8
    
    # get load factors for various engine size choice

    time_percentage = [0.021/0.856,0.015/0.856,0.039/0.856,0.036/0.856,0.037/0.856,0.056/0.856,
                       0.103/0.856,0.083/0.856,0.055/0.856,0.053/0.856,0.042/0.856,0.058/0.856,
                       0.059/0.856,0.07/0.856,0.059/0.856,
                       0.027/0.856,0.011/0.856,0.014/0.856,0.014/0.856,0.004/0.856] # exclude stationary  

    load_factor_tw_ave_Rhine_single_smallsize = ((86*0.021)/MCR_smallsize_1 +
                                        (147*0.015)/MCR_smallsize_1 +
                                        (208*0.039)/MCR_smallsize_1 +
                                        (270*0.036)/MCR_smallsize_1 +
                                        (331*0.037)/MCR_smallsize_1 + 
                                        (392*0.056)/(MCR_smallsize_1) +
                                        (450*0.103)/(MCR_smallsize_1) +
                                        (515*0.083)/(MCR_smallsize_1) +
                                        (576*0.055)/(MCR_smallsize_1) +
                                        (637*0.053)/(MCR_smallsize_1) +
                                        (698*0.042)/(MCR_smallsize_1) +
                                        (760*0.058)/(MCR_smallsize_1) +
                                        (821*0.059)/(MCR_smallsize_1) +
                                        (882*0.07)/(MCR_smallsize_1) +
                                        (943*0.059)/(MCR_smallsize_1) +
                                        (0.85*0.027)+
                                        (0.85*0.011)+
                                        (0.85*0.014)+
                                        (0.85*0.014)+
                                        (0.85*0.004))/sum(time_percentage)
    
    return battery_capacity_for_trip_kwh, battery_capacity_for_peakpower_kwh,load_factor_tw_ave_Rhine_single_smallsize