'''
This code implements exotic options classes.
All exotic options must have a payoff function which takes one or more price 
paths and returns the payoff and a function to set the option strike.
'''

import abc
import numpy as np
from enum import Enum

#Abstract Class which requires all child classes to implement payoff
class ExoticOption(metaclass = abc.ABCMeta):
    
    @abc.abstractmethod
    def payoff(self, price_paths: np.array) -> np.array:
        return
    
BarrierType = Enum('BarrierType', ['Down and Out', 'Down and In', 'Up and Out', 'Up and In'])
OptionType = Enum('OptionType', ['Call','Put'])
    
class BarrierOption(ExoticOption):
    
    def __init__(self, B, K, barrier_type: BarrierType, option_type: OptionType):
        
        self.B = B
        self.K = K
        self.barrier_type = barrier_type
        self.option_type = option_type
        
    def set_strike(self, K):
        
        self.K = K
        
    def payoff(self, price_paths):
        
        if self.barrier_type == BarrierType['Down and Out']:
            
            final_price = (price_paths.min(axis = 0) >= self.B) * price_paths[-1]
            
        elif self.barrier_type == BarrierType['Down and In']:
            
            final_price = (price_paths.min(axis = 0) < self.B) * price_paths[-1]
            
        elif self.barrier_type == BarrierType['Up and Out']:
            
            final_price = (price_paths.max(axis = 0) <= self.B) * price_paths[-1]
            
        else:
            
            final_price = (price_paths.max(axis = 0) > self.B) * price_paths[-1]
            
        return (final_price - self.K) * (final_price > self.K) if self.option_type == OptionType['Call'] \
            else (self.K - final_price) * (self.K > final_price)
            
class EuropeanOption(ExoticOption):
    
    def __init__(self, K, option_type: OptionType):
        
        self.K = K
        self.option_type = option_type
        
    def set_strike(self, K):
        
        self.K = K
        
    def payoff(self, price_paths):
        
        final_price = price_paths[-1]
        
        return (final_price - self.K) * (final_price > self.K) if self.option_type == OptionType['Call'] \
            else (self.K - final_price) * (self.K > final_price)