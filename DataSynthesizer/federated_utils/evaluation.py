import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import os
import networkx as nx



def HistogramAndHeatmapComparison(train_data, synthetic_data, path_results):
    """
    Function to plot distribution parameters of original training data against synthetic data and also
    heatmap of correlations for training data and synthetic data.

    Args:
        train_data: file of the original training dataset
        synthetic_data: file of the synthetic dataset
        path_results: path to save results
    """

    input_df = pd.read_csv(train_data, skipinitialspace=True)
    syn_df = pd.read_csv(synthetic_data)

    # Label Encoding
    header = list(input_df.columns.values)

    numeric_attributes = syn_df.select_dtypes(include=np.number).columns

    le = preprocessing.LabelEncoder()

    for i in range(len(header)):

        if header[i] not in numeric_attributes:
            input_df.iloc[:, i] = le.fit_transform(list(input_df.iloc[:, i])).astype(np.float64)
            syn_df.iloc[:, i] = le.fit_transform(list(syn_df.iloc[:, i])).astype(np.float64)

    # Check if path for saving results exists, if not creates the corresponding folders
    if not os.path.exists(path_results+'/Plots'):
        os.makedirs(path_results + '/Plots/Histograms')

    correlations_syn = syn_df.corr()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    cax = ax.matshow(correlations_syn, cmap='seismic', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(header), 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(header,rotation = 90)
    ax.set_yticklabels(header)
    plt.tight_layout()
    plt.savefig(path_results + '/Plots/heatsyn.png')
    plt.close()

    correlations_real = input_df.corr()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    cax = ax.matshow(correlations_real, cmap='seismic', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(header), 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(header,rotation = 90)
    ax.set_yticklabels(header)
    plt.tight_layout()
    plt.savefig(path_results + '/Plots/heatreal.png')
    plt.close()

    for i in range(len(header)):
        plt.hist(input_df.iloc[:, i], color='green', label='Real: ' + header[i])
        plt.legend(fontsize='x-small')
        plt.hist(syn_df.iloc[:, i], histtype='step', color='blue', label='Syn: ' + header[i])
        plt.legend(fontsize='x-small')
        plt.savefig(path_results + '/Plots/Histograms/' + header[i] + '.png')
        plt.close()
        
def TaskUtility(df_real_train, df_syn_train, df_test, label_predictor, path_results, type_task = 'Classification'):
    
    # Label Encoding
    header = list(df_real_train.columns.values)
    
    numeric_attributes = df_syn_train.select_dtypes(include=np.number).columns

    le = preprocessing.LabelEncoder()

    for i in range(len(header)):

        if header[i] not in numeric_attributes:
            df_real_train.iloc[:, i] = le.fit_transform(list(df_real_train.iloc[:, i])).astype(np.float64)
            df_test.iloc[:, i] = le.transform(list(df_test.iloc[:, i])).astype(np.float64)
            df_syn_train.iloc[:, i] = le.fit_transform(list(df_syn_train.iloc[:, i])).astype(np.float64)
            
      
    X_train_real= df_real_train.drop(columns=[label_predictor])  
    y_train_real = df_real_train[label_predictor].astype('int')
    
    X_train_syn= df_syn_train.drop(columns=[label_predictor])  
    y_train_syn = df_syn_train[label_predictor].astype('int')
    
    X_test= df_test.drop(columns=[label_predictor])  
    y_test = df_test[label_predictor].astype('int')
          
    if type_task == "Classification":
        rf_real = RandomForestClassifier()
        rf_real.fit(X_train_real,y_train_real)
        
        nb_real = GaussianNB()
        nb_real.fit(X_train_real,y_train_real)
        
        sv_real = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        sv_real.fit(X_train_real,y_train_real)
        
        knn_real = make_pipeline(StandardScaler(), KNeighborsClassifier())
        knn_real.fit(X_train_real,y_train_real)
        
        lr_real = LogisticRegression()
        lr_real.fit(X_train_real,y_train_real)
        
        rf_syn = RandomForestClassifier()
        rf_syn.fit(X_train_syn,y_train_syn)
        
        nb_syn = GaussianNB()
        nb_syn.fit(X_train_syn,y_train_syn)
        
        sv_syn = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        sv_syn.fit(X_train_syn,y_train_syn)
        
        knn_syn =  make_pipeline(StandardScaler(), KNeighborsClassifier())
        knn_syn.fit(X_train_syn,y_train_syn)
        
        lr_syn = LogisticRegression()
        lr_syn.fit(X_train_syn,y_train_syn)
                
        prediction_real_RF = rf_real.predict(X_test)
        prediction_syn_RF  = rf_syn.predict(X_test)
        
        prediction_real_NB = nb_real.predict(X_test)
        prediction_syn_NB = nb_syn.predict(X_test)
        
        prediction_real_SV = sv_real.predict(X_test)
        prediction_syn_SV  = sv_syn.predict(X_test)
        
        prediction_real_KNN = knn_real.predict(X_test)
        prediction_syn_KNN = knn_syn.predict(X_test)
        
        prediction_real_LR = lr_real.predict(X_test)
        prediction_syn_LR = lr_syn.predict(X_test)
        
        algorithms = ['RF','NB', 'SV', 'KNN', 'LR']
        index = ['Original','Synthetic']
        
        results = pd.DataFrame(columns=algorithms, index=index)
        
        for alg in algorithms:
            real_acc = accuracy_score(y_test,eval(f'prediction_real_{alg}'))
            syn_acc = accuracy_score(y_test,eval(f'prediction_syn_{alg}'))
            results.at['Original',alg] = real_acc
            results.at['Synthetic',alg] = syn_acc
            
        results.to_csv(f'{path_results}/task_utility.csv',index=False)
        
    else:
        rf_real = RandomForestRegressor()
        rf_real.fit(X_train_real,y_train_real)
        
        rf_syn = RandomForestRegressor()
        rf_syn.fit(X_train_syn,y_train_syn)


def plot_bayesian_network(bayesian_network):
    graph = nx.DiGraph()
    edges_dag = []
    for child, parents in bayesian_network:
        for parent in parents:
            edges_dag.append((parent,child))
    graph.add_edges_from(edges_dag)
    for layer, nodes in enumerate(nx.topological_generations(graph)):
        # `multipartite_layout` expects the layer as a node attribute, so add the
        # numeric layer value as a node attribute
        for node in nodes:
            graph.nodes[node]["layer"] = layer
    pos = nx.multipartite_layout(graph, subset_key="layer")
    pos_higher = {}
    y_off = -0.04  # offset on the y axis
    for k, v in pos.items():
        pos_higher[k] = (v[0], v[1]+y_off)
    nx.draw(graph, pos, arrows=True, style='dashed', width=0.5)
    nx.draw_networkx_labels(graph, pos_higher, font_size=8)
    plt.tight_layout()
    plt.show()     