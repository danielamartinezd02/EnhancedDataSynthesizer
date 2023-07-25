import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from fedlab.utils.functional import partition_report
from fedlab.utils.dataset.functional import balance_split,homo_partition,dirichlet_unbalance_split

def partition_dataset( file_path, number_clients, scenario, strategy=None, seed=0, alpha=0.5):
    '''
    Partitions a dataset in multiple clients using different distribution strategies. 
        
    Arguments:
        file_path (str): Absolute path of dataset to partition.
        number_clients (int): Total number of clients.
        scenario (str): Selects an iid or non iid partition.
        strategy (str): Indicates the type of distribution used for the partition.
        seed (int): To make experiments reproducible.

    Returns:
        dictionary: A dictionary containing the index of samples assign to each client. 
    '''
    # Set the seed to always get the same partitions
    np.random.seed(seed)
    
    dataset = pd.read_csv(file_path)    
    number_of_samples = dataset.shape[0]
    client_ids = np.arange(0,number_clients)
    
    if scenario == 'iid':
        # Quantity balanced 
        partition_size = number_of_samples // number_clients
        indexes = np.random.permutation(number_of_samples)
        splits = [(i+1)*partition_size for i in range(number_clients)]
        clients_indexes = np.split(indexes, splits)
        
        client_dict=dict(zip(client_ids,clients_indexes))
        
                    
    elif scenario == 'unbalance':      
        # quantity-skew (Dirichlet)
        client_sample_nums = dirichlet_unbalance_split(number_clients, number_of_samples,
                                                             alpha)
        client_dict = homo_partition(client_sample_nums, number_of_samples)
    
    elif scenario == 'noniid-labeldir':
        # Label Distribution Skew: means that means that the label Pi(y) distribution of different clients is different, but the Pi(x | y) situation is the same
        pass

        
        
    else:
       raise ValueError(f" The 'scenario' {(scenario)} is not valid. The supported values for 'scenario' are 'iid' or 'non-iid'.") 
                
    return client_dict


def plot_partition(clients_indexes, targets=[]):
    '''
    Plots the distribution of samples in each of the clients.
    
    Arguments:
        client_indexes (str): Dictionary containing the indexes of each of the clients.
    
    '''
    df=pd.DataFrame({'client':np.zeros(len(targets)),'class':targets})
    for key,val in clients_indexes.items():
        df.loc[val,'client']=key
    ax = sns.countplot(data=df,x='client',hue='class')
    ax.set(xlabel='Client id', ylabel='Number of Samples')
    plt.show()
    
