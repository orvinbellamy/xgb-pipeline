### Functions for data explorations

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import pickle
import time

### Feature Selection ###
def rfe_numeric_feature_selection(df_train:pd.DataFrame, features:list, label:str, num_features:int=10, model=LogisticRegression(solver='lbfgs')):
    
	## Feature Selection RFE for numeric variables

	# Set number of columns to return
	# i.e. the top 15 columns
	num_feature_selection_numeric = 15

	# Filter out non-numerical columns
	x_train = df_train[features]
	y_train = df_train[label]

	# feature extraction for numeric variables
	rfe = RFE(estimator=model, n_features_to_select=num_feature_selection_numeric)
	fit = rfe.fit(x_train, y_train)

	# Get the indexes where the value is True
	idx_best_features = np.where(fit.support_)[0]

	# Get the best numeric columns/features
	col_best_features = x_train.iloc[:, idx_best_features].columns.to_list()
	
	return col_best_features

### Clustering ###

def plot_kmeans_inertia(df_cluster:pd.DataFrame, n_cluster:int=6, random_seed:int=123):
    
	'''
	Input df_cluster MUST only contain columns to be used in the clustering

	'''
    
	## Elbow graph for clustering
	inertia = []
	for k in range(1, n_cluster+1):
		kmeans = KMeans(n_clusters=k, random_state=random_seed)
		kmeans.fit(df_cluster)
		inertia.append(kmeans.inertia_)
		
	plt.figure(figsize=(8, 6))
	plt.plot(range(1, n_cluster+1), inertia, marker='o', linestyle='--')
	plt.xlabel('Number of clusters')
	plt.ylabel('Inertia')
	plt.title('Elbow Plot for KMeans Clustering')
	plt.xticks(range(1, n_cluster+1))
	plt.grid(True)
	plt.show()
 

class KMeansHandler():
    
	def __init__(self):
		
		pass

	def save_self(self, save_file_name:str=f'kmeans_model_{time.time()}'):
		
		self.file_name = save_file_name
		
		with open(f'models/{save_file_name}.pkl', 'wb') as file:
			pickle.dump(self, file)
    
	def create_kmeans_cluster(self, df, save_file_name:str=f'kmeans_model_{time.time()}', n_clusters:int=5, random_seed:int=123, save_cluster:bool=False):
		
		'''
		Input df_cluster MUST only contain columns to be used in the clustering

		'''
		
		## Create a new cluster

		# Initialize the K-Means model
		self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed)

		# Fit the model to your data
		self.kmeans.fit(df)
			
		# Get the cluster centers
		cluster_centers = self.kmeans.cluster_centers_

		self.df_cluster_centers = pd.DataFrame(cluster_centers, columns=df.columns)
		self.df_cluster_centers[self.df_cluster_centers.columns] = self.df_cluster_centers[self.df_cluster_centers.columns].round(4)
  
		self.col_cluster = self.df.columns
  
		if save_cluster:
			self.save_self(save_file_name=save_file_name)
   
	def define_cluster(self, dic_cluster:dict):
     
		self.dic_cluster_names = dic_cluster
  
	def apply_cluster(self, df:pd.DataFrame):
		
		df_func = df.copy()

		df_func['cluster'] = self.kmeans.predict(df_func[self.col_cluster])

		df_func['segment'] = df_func['cluster'].map(self.dic_cluster_names)
  
		return df_func

