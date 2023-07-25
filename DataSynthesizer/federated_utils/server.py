from typing import Dict, List
import numpy as np
from DataSynthesizer.lib.utils import nested_dict_keys,deep_get_from_dict,nested_set
from decimal import Decimal

class FedServer:
    def __init__(self, n_clients: int, metadata: Dict = {}):
        self.number_of_clients = n_clients
        self.attributes_metadata = metadata
        self.params_aggregation = []
        self.server_round = None 
        self.conditional_probabilities = {}
        
    def get_attributes_metadata(self, metadata: Dict):
        ''' The server asks one of the clients in the federated process for the attributes names and the types'''
        self.attributes_metadata = metadata
        self.attributes_metadata['attributes_description'] = {}
        
    def compute_global_min_max_numeric_columns(self, min_max_clients: List):
        for attribute in self.attributes_metadata['numerical_attributes']:
            global_min = min(list(map(lambda d: d[attribute]['min'],  min_max_clients)))
            global_max = max(list(map(lambda d: d[attribute]['max'],  min_max_clients)))
            distribution_bins = [global_min,global_max]
            self.attributes_metadata['attributes_description'][attribute] = {'min': global_min, 'max': global_max, 'distribution_bins':distribution_bins}
        
    def align_categorical_columns(self, categories_clients: List):
        for attribute in self.attributes_metadata['categorical_attributes']:
            categories_per_attribute = list(map(lambda d: d[attribute]['categories'],  categories_clients))
            all_categories_per_attribute=set().union(*categories_per_attribute)
            self.attributes_metadata['attributes_description'][attribute] = {'distribution_bins': list(all_categories_per_attribute)}
            
    def aggregate_fit(  self, server_round: int, results: List):
        # We need to implement this in flower
        self.params_aggregation = results 
        self.server_round = server_round
        
    def aggregate_evaluation( self, results: List):
        # We need to implement this in flower
        average_result=[]
        for i in range(len(results[0])):
            average_result.append(np.mean([ res[i][0] for res in results]))
        print(max(average_result))
        best_individual_index=average_result.index(max(average_result))
        return self.params_aggregation[best_individual_index]
    
    def aggregate_conditional_probabilities(self, conditional_probabilities_clients: List):
        # We need to implement this in flower
        global_conditional_probabilities={}
        all_nested_keys = list(nested_dict_keys(conditional_probabilities_clients[0]))
        for nested_key in all_nested_keys:
            output_list=[deep_get_from_dict(i,nested_key) for i in conditional_probabilities_clients]
            output_list=list(filter(lambda item: item is not None, output_list))
            result=[sum(x) for x in zip(*output_list)]
            result=[elem/self.number_of_clients for elem in result]
            result=fix_array_to_sum_to_1(result)
            nested_set(global_conditional_probabilities,nested_key.split('.'),result)
        self.conditional_probabilities=global_conditional_probabilities
            
                        
    
from decimal import Decimal, getcontext

def fix_array_to_sum_to_1(arr):
    # Convert each element in the array to Decimal objects
    decimal_array = [Decimal(str(val)) for val in arr]
    
    # Set the desired precision (number of decimal places)
    getcontext().prec = 28
    
    # Calculate the sum of the decimal array
    sum_decimal_array = sum(decimal_array)
    
    # Normalize the decimal array to ensure it sums to exactly 1
    normalized_array = [val / sum_decimal_array for val in decimal_array]
    
    normalized_array = [ float(val) for val in decimal_array]
    
    return normalized_array        
    