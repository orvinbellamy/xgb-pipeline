import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt    
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score, roc_curve, auc
from scipy import stats
    
def cutoff_table(model, test, col_label:str, col_prediction:str):
	col_names = ['Accuracy', 'P-Value', 'Sensitivity', 'Specificity', 'Pos Pred Value', 'Neg Pred Value', 'Prevalence', 'Detection Rate']
	row_names = np.arange(0.05, 1, 0.05)

	con_table = pd.DataFrame(0.01, index=row_names, columns=col_names)

	for i in range(1, 20):
		predict1 = model.predict_proba(test)[:, 1]
		
		test_predict = test.copy()
		test_predict[col_prediction] = predict1
		test_predict['prediction_factor'] = 0
		test_predict.loc[test_predict[col_prediction] >= (i*0.05), 'prediction_factor'] = 1
		test_predict = test_predict[~test_predict[col_prediction].isnull()]
		
		con_matrix1 = confusion_matrix(test_predict[col_label], test_predict['prediction_factor'], labels=[1, 0])
		con_matrix1 = con_matrix1.astype(float)
		
		acc = (con_matrix1[0, 0] + con_matrix1[1, 1]) / np.sum(con_matrix1)
		acc_p_val = 1 - stats.chi2.cdf(np.sum((con_matrix1[0, :] - con_matrix1[1, :])**2 / np.sum(con_matrix1, axis=0)), 1)
		sensitivity = con_matrix1[0, 0] / np.sum(con_matrix1[0, :])
		specificity = con_matrix1[1, 1] / np.sum(con_matrix1[1, :])
		ppv = con_matrix1[0, 0] / np.sum(con_matrix1[:, 0])
		npv = con_matrix1[1, 1] / np.sum(con_matrix1[:, 1])
		prevalence = np.sum(con_matrix1[0, :]) / np.sum(con_matrix1)
		detection_rate = np.sum(con_matrix1[0, :]) / np.sum(con_matrix1[:, 0])
		
		con_table.iloc[i-1, 0] = acc
		con_table.iloc[i-1, 1] = acc_p_val
		con_table.iloc[i-1, 2] = sensitivity
		con_table.iloc[i-1, 3] = specificity
		con_table.iloc[i-1, 4] = ppv
		con_table.iloc[i-1, 5] = npv
		con_table.iloc[i-1, 6] = prevalence
		con_table.iloc[i-1, 7] = detection_rate

	return con_table

def convert_date(x):
	if pd.isna(x):
		return pd.to_datetime('2018-01-01').date()
	else:
		return pd.to_datetime(x).date()
    
def var_parser(kettle, var_name):
	with open(kettle, 'r') as f:
		for line in f:
			if var_name in line:
				name, value = line.split('=', 1)
				return value.strip()
				
			else:
				pass
    
def confusion_matrix_report(df, actual, predicted, cutoff):
    
	df['prediction_factor'] = 0
	df.loc[(df[predicted] >= cutoff), 'prediction_factor'] = 1

	cm = confusion_matrix(df[actual], df['prediction_factor'], labels=[1,0])

	accuracy = accuracy_score(df[actual], df['prediction_factor'])
	precision = precision_score(df[actual], df['prediction_factor'])
	recall = recall_score(df[actual], df['prediction_factor'])
	f1 = f1_score(df[actual], df['prediction_factor'])

	# using this for 3.9.13
	tp = np.where((df[actual] == 1) & (df['prediction_factor'] == 1), 1, 0).sum()
	tn = np.where((df[actual] == 0) & (df['prediction_factor'] == 0), 1, 0).sum()
	fp = np.where((df[actual] == 0) & (df['prediction_factor'] == 1), 1, 0).sum()
	fn = np.where((df[actual] == 1) & (df['prediction_factor'] == 0), 1, 0).sum()

	sensitivity = tp / (tp + fn)
	specificity = tn / (tn + fp)

	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1,0])

	print("Confusion matrix:")

	disp.plot()
	plt.show()

	print('Cutoff: ', cutoff)
	print('Sensitivity:', sensitivity)
	print('Specificity:', specificity)

	print("\nAccuracy:", accuracy)
	print("Precision:", precision)
	print("Recall:", recall)
	print("F1 score:", f1)

	print("\nClassification report:")
	print(classification_report(df[actual], df['prediction_factor']))

def backward_elimination_model(df, independent_variables):
	X = df[independent_variables]

	formula = 'attrition ~ ' + ' + '.join(X.columns)

	bw_model = sm.formula.glm(formula=formula, family=sm.families.Binomial(), data=df).fit()

	while len(X.columns) > 1:
		p_values = bw_model.pvalues[1:]
		
		max_p_value = p_values.max()
		
		if max_p_value > 0.05:
			
			variable_to_remove = p_values.idxmax()
			
			if '[' in variable_to_remove:
				# Extract variable name without categorical information
				variable_name = variable_to_remove.split('[')[0]
				
				# Remove variable from formula string
				formula = formula.replace(' + ' + variable_to_remove, '')
				
				# Remove variable from X dataframe
				X = X.drop(variable_name, axis=1)
			else:
				X = X.drop(variable_to_remove, axis=1)
			
			formula = 'attrition ~ ' + ' + '.join(X.columns)
			
			bw_model = sm.formula.glm(formula=formula, family=sm.families.Binomial(), data=df).fit()
		
		else:
			break

	print(formula + '\n')

	return bw_model

def model_odds(model_coefficients, df):
	import math

	odds_dict = {}
	odds_dict_numerical = {}
	odds_dict_categorical = {}

	for key, value in model_coefficients.items():
		odds_dict[key] = (math.exp(value) - 1) * 100
		
	for key, value in model_coefficients.items():
		if key == "Intercept":
			continue
		elif ':' in key:
			continue
		elif '[' in key:
			var = key.split('[')[0]
			if str(df[var].dtype).startswith('int') or str(df[var].dtype).startswith('float'):
				odds_dict_numerical[key] = (math.exp(value) - 1) * 100
			else:
				odds_dict_categorical[key] = (math.exp(value) - 1) * 100
		else:
			if str(df[key].dtype).startswith('int') or str(df[key].dtype).startswith('float'):
				odds_dict_numerical[key] = (math.exp(value) - 1) * 100
			else:
				odds_dict_categorical[key] = (math.exp(value) - 1) * 100

	return odds_dict, odds_dict_numerical, odds_dict_categorical

def plot_model_coef(vars, coefs, xlabel):
    
	# Set color map
	colors = ['green' if p < 0 else 'red' for p in coefs]

	# Create bar chart
	fig, ax = plt.subplots()
	bars = ax.barh(vars, coefs, color=colors)

	# Set x-axis limits and format as percentage
	ax.set_xlabel(xlabel)
	ax.xaxis.set_major_formatter('{x:.2f}%')

	# Set y-axis label and hide y-axis line
	ax.spines['left'].set_visible(False)

	# Add data labels
	for bar in bars:
		width = bar.get_width()
		sign = '-' if width < 0 else ''
		label = f"{sign}{width:.2f}%"
		ax.annotate(label, xy=(width, bar.get_y() + bar.get_height() / 2),
					xytext=(20, 0), textcoords='offset points',
					ha='right', va='center')

	# Show plot
	plt.show()

def accuracy_bucket(df, col_metric, col_prediction, col_actual):
    
	df['rate_bucket'] = pd.qcut(df[col_metric], q=5, labels=False)

	# calculate accuracy of predictions for each bucket
	accuracy = df.groupby('rate_bucket').apply(lambda x: (x[col_prediction] == x[col_actual]).mean())

	# create bar chart
	ax = accuracy.plot(kind='bar')
	accuracy.plot(kind='bar')
	plt.xticks(range(5), [f'{int(df[col_metric].quantile((q-1)/5))} - {int(df[col_metric].quantile(q/5))}' for q in range(1,6)])
	plt.xlabel('Rate Buckets')
	plt.ylabel('Average Accuracy')
	plt.title('Accuracy of Churn Prediction by Customer Rate Bucket')
	plt.xticks(rotation=0)

	# add data labels to the bars
	for i, v in enumerate(accuracy):
		ax.annotate(str(round(v, 2)), xy=(i-0.12, v+0.01))

	plt.show()
    
def accuracy_bucket_label(df, col_metric, col_prediction, col_actual):
    
	# calculate accuracy of predictions for each bucket
	accuracy = df.groupby(['rate_bucket', col_actual]).apply(lambda x: (x[col_prediction] == x[col_actual]).mean()).reset_index(name='accuracy')

	# create bar chart for each bucket
	fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
	axs = axs.flatten()  # flatten axes for easier indexing

	num_plots = df['rate_bucket'].max()

	for i, ax in enumerate(axs):
		
		if i <= num_plots:
			pass
		else:
			break
		
		# get data for current bucket
		bucket_data = accuracy[accuracy['rate_bucket'] == i]
		
		# create bar chart for current bucket
		bucket_data.plot(x='attrition', y='accuracy', kind='bar', ax=axs[i], color='C0', width=0.4, edgecolor='black', alpha=0.8)
		
		# set xtick labels
		axs[i].set_xticklabels(['not churn', 'churn'])
		
		# set title and x-axis label for current bucket
		ax.set_title(f'{int(df[col_metric].quantile(i/5))} - {int(df[col_metric].quantile((i+1)/5))}')
		ax.set_xlabel('Churn Actual')
		
		# set ylim to 0-1 for consistent scaling across subplots
		axs[i].set_ylim([0, 1])
		
		axs[i].legend().remove()
			
		# add data labels to the bars
		for i, v in enumerate(bucket_data['accuracy']):
			ax.annotate(str(round(v, 2)), xy=(i-0.12, v+0.01))
		
		# hide y-axis label for all but first subplot
		if i != 0:
			ax.set_ylabel('')
        
        
	# remove extra subplots
	for j in range(5, len(axs)):
		axs[j].remove()
		
	# set y-axis label for entire figure
	axs[0].set_ylabel('Average Accuracy')

	# set title for entire figure
	fig.suptitle('Accuracy of Churn Prediction by Customer Rate Bucket and Churn Actual')

	# adjust layout to prevent overlapping of subplots
	plt.tight_layout()

	for ax in axs:
		ax.tick_params(axis='x', rotation=0)

	# adjust spacing between subplots
	plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.5)

	# display plot
	plt.show()

def get_null_columns(df):
	columns_with_null = df.columns[df.isnull().any()].tolist()
	return columns_with_null

def impute(df_to_impute, imp_columns, col_category):
	import miceforest as mf
	import pandas as pd
	# change data types for miceforest imputation
	# object data type needs to be categorical data type
	df_to_impute[col_category] = df_to_impute[col_category].astype('category')

	kernel = mf.ImputationKernel(
	data=df_to_impute[imp_columns],
	save_all_iterations=True,
	random_state=1991
	)

	# Run the MICE algorithm for 3 iterations on each of the datasets
	kernel.mice(3,verbose=False)

	# Our new dataset
	df_imputed = kernel.impute_new_data(df_to_impute[imp_columns]).complete_data(0)
	df_to_impute.update(df_imputed, overwrite=True)

	return df_to_impute

def create_histogram(df, col, segment = None, bins = 'auto', x_label = 'Value', y_label = 'Frequency', title = 'Histogram', log = None):
    
	if log is not None:
		df[col] = np.log(df[col] + log)

	# Create a histogram
	if segment is None:
		frequencies, bins, patches = plt.hist(df[col], bins=bins)
	else:
		# Segment the data
		segment_data = sorted(df[segment].unique())
		segmented_data = [df[df[segment] == t][col] for t in segment_data]
		# Plot the stacked histogram with different colors for each segment
		frequencies, bins, patches = plt.hist(segmented_data, bins=bins, stacked=True, label=segment_data)

	# Set labels and title
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(title)

	# Set X-axis labels based on the bin ranges
	bin_ranges = [f'{int(bins[i])}' for i in range(len(bins)-1)]
	plt.xticks(bins[:-1], bin_ranges, rotation=45, ha='right')

	# Add data labels to each bar/bin
	# if segment is None:
	#     total = len(df)
	#     for freq, patch in zip(frequencies, patches):
	#         height = patch.get_height()
	#         percentage = (freq / total) * 100
	#         label = f'{percentage:.0f}%'

	#         # Adjust font size for large number of bins
	#         if len(bins) >= 15:
	#             font_size = 'small'
	#         else:
	#             font_size = 'medium'

	#         plt.annotate(label, xy=(patch.get_x() + patch.get_width() / 2, height),
	#                     xytext=(0, 5), textcoords='offset points', ha='center', fontsize=font_size)
	# else:
	#     # Move the legend outside the histogram and adjust its position using 'bbox_to_anchor'
	plt.legend(title='Segments', loc='center left', bbox_to_anchor=(1, 0.5))

	# Display the histogram
	plt.show()
	# print(frequencies)

def create_boxplot(df, col, title = 'Boxplot', log = None):
	import matplotlib.pyplot as plt
	import pandas as pd
	import numpy as np

	if log is not None:
		df[col] = np.log(df[col] + log)
		
	plt.boxplot(df[col])
	# Set labels and title
	plt.xlabel('')
	plt.ylabel(col)
	plt.title(title)
    
def eda_plot(df, x_axis, y_axis):
	
	# using ci argument makes the plot takes way longer
	sns.regplot(data=df, x=x_axis, y=y_axis, scatter=False, logistic=True, ci=None, truncate=True, line_kws={'color': 'red'})
    
def eda_plot_all(df, y_axis, cols_to_plot):
    
	num_plots = len(cols_to_plot)
	num_cols = min(3, num_plots)
	num_rows = (num_plots - 1) // num_cols + 1

	# create figures and subplots
	fig, axs = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))

	# plot each column on its own subplot
	for i, col in enumerate(cols_to_plot):
		row = i // num_cols
		col_num = i % num_cols
		ax = axs[row][col_num] if num_rows > 1 else axs[col_num]
		sns.regplot(data=df, x=col, y=y_axis, scatter=True, logistic=True, ci=None, truncate=True, line_kws={'color': 'red'}, ax=ax)
		ax.set_xlabel(col)
		ax.set_ylabel(y_axis)

	# adjust spacing between subplots
	fig.tight_layout()

	# show plot
	plt.show()

def survival_analysis_process(df, start_date, end_date, month_number, id):
    
	# Convert date columns to datetime type
	df[start_date] = pd.to_datetime(df[start_date])
	df[end_date] = pd.to_datetime(df[end_date])

	# maximum tenure is 13 months
	df.loc[df[month_number]>13, month_number] = 13

	# Create a column for the cohort month
	df['cohort_month'] = df[start_date].dt.to_period('M')

	# cohort_data = df.groupby('cohort_month').apply(lambda x: pd.Series({
	#     'total_customers': len(x),
	#     'churned_customers': sum(x['cancel_date'].notnull()),
	#     'churn_rate': sum(x['cancel_date'].notnull()) / len(x)
	# })).reset_index()

	# Create an empty DataFrame to store the expanded data
	expanded_df = pd.DataFrame()

	# Iterate over each row in the original DataFrame
	for index, row in df.iterrows():
		# Extract the relevant information from the row
		customer_id = row[id]
		cohort = row['cohort_month']
		tenure_month = row[month_number]
		
		# Create a temporary DataFrame for the current customer and cohort
		temp_df = pd.DataFrame({'customer_id': customer_id,
								'cohort': cohort,
								'months_after_acquisition': range(tenure_month + 1)})
		
		# Append the temporary DataFrame to the expanded DataFrame
		expanded_df = pd.concat([expanded_df, temp_df], ignore_index=True)
		
	df_survival_curve = expanded_df.groupby(['cohort', 'months_after_acquisition']).count().reset_index()

	df_survival_curve.rename(columns={'customer_id': 'number_of_accounts'}, inplace=True)

	# df_survival_curve = expanded_df.groupby(['cohort', 'months_after_acquisition']).count().reset_index()
	df_survival_curve = df_survival_curve.sort_values(by=['cohort', 'months_after_acquisition'])

	# shift months_after_acquisition by 1
	df_survival_curve = df_survival_curve[df_survival_curve['months_after_acquisition'] > 0]
	df_survival_curve['months_after_acquisition'] = df_survival_curve['months_after_acquisition'] - 1

	# Create a dictionary to store the total number of accounts for each cohort
	# cohort_totals = {}

	# Initialize the 'percentage_survived' and 'customers_lost' columns with NaN
	df_survival_curve['customers_lost'] = pd.NA

	max_accounts = df_survival_curve.groupby('cohort').max('number_of_accounts').reset_index()
	max_accounts.rename(columns={'number_of_accounts': 'max_accounts'}, inplace=True)

	df_survival_curve = pd.merge(df_survival_curve, max_accounts[['cohort', 'max_accounts']], how='left', on='cohort')

	df_survival_curve['percentage_survived'] = (df_survival_curve['number_of_accounts'] / df_survival_curve['max_accounts']) * 100

	# Iterate over each row in the DataFrame
	for index, row in df_survival_curve.iterrows():
		cohort = row['cohort']
		months_after_acquisition = row['months_after_acquisition']
		number_of_accounts = row['number_of_accounts']
		
		# Handle the case when months_after_acquisition == 1 (no previous month data)
		if months_after_acquisition < 1:
			customers_lost = 0
		else:
			customers_lost = number_of_accounts - df_survival_curve.loc[(df_survival_curve['cohort'] == cohort) & (df_survival_curve['months_after_acquisition'] == months_after_acquisition - 1), 'number_of_accounts'].values[0]
		
		df_survival_curve.loc[index, 'customers_lost'] = customers_lost
		
	df_survival_curve['previous_number_of_accounts'] = df_survival_curve.groupby('cohort')['number_of_accounts'].shift(+1)

	df_survival_curve['churn_rate'] = abs(df_survival_curve['customers_lost'] / df_survival_curve['previous_number_of_accounts']) * 100

	df_survival_curve['retention_rate'] = 100 - df_survival_curve['churn_rate']

	return df_survival_curve

def survival_plot(df,col_y, col_x,  x_label = 'time', y_label = '# of items', col_cohort='cohort', title = 'Survival Curve'):
    
	# Get the unique cohorts for plotting separate lines
	cohorts = df.loc[(df[col_cohort] >= '2021-05') & (df[col_cohort] < '2022-07'), col_cohort].unique()

	# Create a plot
	plt.figure(figsize=(10, 6))

	# Plot each cohort as a separate line
	for cohort in cohorts:
		cohort_data = df[df[col_cohort] == cohort]
		plt.plot(cohort_data[col_x], cohort_data[col_y], label=cohort)

	# Add labels and legend
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(title)

	# Move the legend outside the chart and adjust its position using 'bbox_to_anchor'
	plt.legend(title='Cohort', loc='center left', bbox_to_anchor=(1, 0.5))

	# Show the plot
	plt.grid(True)
	plt.show()
    
def survival_pivot(df, col_name, value_name, row='cohort', aggregation=None):
	df_survival_pivot = df.pivot_table(index=row, columns=col_name, values=value_name, aggfunc='mean')

	if aggregation is None:
		pass
	else:
		df_survival_pivot.loc['Mean'] = df_survival_pivot.mean()

	return df_survival_pivot

def plot_roc(fpr, tpr, thresholds, roc_auc):
    
	# Plot the ROC curve
	fig, ax = plt.subplots(figsize=(8, 6))
	ax.plot(fpr, tpr, color='navy', lw=1.5, label='ROC curve (AUC = %0.2f)' % roc_auc)
	ax.plot([0, 1], [0, 1], color='grey', lw=1.5, linestyle='--', label='Random guess')

	plot_x = []
	plot_y = []
	plot_cutoff = []

	# Add data labels to ROC curve
	for i, threshold in enumerate(np.arange(0, 1.05, 0.05)):
		idx = np.argmin(np.abs(threshold - thresholds))
		# ax.text(fpr[idx]+0.02, tpr[idx]-0.03, f"{threshold:.1f}", fontsize=10)
		plt.plot(fpr[idx], tpr[idx], 'o', markersize=3, label=None, color='red')
		plt.annotate(f"{threshold:.2f}", (fpr[idx], tpr[idx]), textcoords="offset points", xytext=(0,10), ha='center')
		
		plot_x.append(fpr[idx])
		plot_y.append(tpr[idx])
		plot_cutoff.append(threshold)

	# Set plot parameters
	ax.set_xlim([-0.05, 1.05])
	ax.set_ylim([-0.05, 1.05])
	ax.set_xlabel('False Positive Rate (1 - Specificity)')
	ax.set_ylabel('True Positive Rate (Sensitivity)')
	ax.set_title('Receiver Operating Characteristic')
	ax.legend(loc="lower right")

	# Show the plot
	plt.show()

	df = pd.DataFrame({'X': plot_x, 'Y': plot_y, 'cutoff': plot_cutoff})

	return df

def plot_precision_recall(y_true, y_scores):
    
	precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
	average_precision = average_precision_score(y_true, y_scores)

	# Plot the precision-recall curve
	plt.figure(figsize=(8, 6))
	plt.plot(recall, precision, color='navy', lw=1.5, label=f'Precision-Recall curve (AP = {average_precision:.2f})')
	plt.xlabel('Recall (Sensitivity)')
	plt.ylabel('Precision')
	plt.title('Precision-Recall Curve')
	plt.legend(loc='best')
		
	# Annotate specific thresholds
	threshold_values = [round(i * 0.05, 2) for i in range(21)]  # Create a list of thresholds (0.00, 0.05, 0.10, ..., 1.00)

	for threshold_value in threshold_values:
		closest_idx = (np.abs(thresholds - threshold_value)).argmin()
		plt.scatter(recall[closest_idx], precision[closest_idx], marker='o', color='red', s=50)
		plt.annotate(f"{threshold_value:.2f}", (recall[closest_idx], precision[closest_idx]), textcoords="offset points", xytext=(0,10), ha='center')

	# Show the plot
	plt.show()
    
def plot_roc_and_precision_recall(y_true, y_scores):
    
	# Calculate ROC curve and AUC
	fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
	roc_auc = auc(fpr, tpr)

	# Calculate Precision-Recall curve and average precision
	precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)
	average_precision = average_precision_score(y_true, y_scores)

	# Plot ROC curve
	plt.figure(figsize=(12, 6))
	plt.subplot(1, 2, 1)
	plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
	plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random guess')
	plt.xlabel('False Positive Rate (1 - Specificity)')
	plt.ylabel('True Positive Rate (Sensitivity)')
	plt.title('ROC Curve')
	plt.legend(loc="lower right")

		# Annotate specific thresholds on ROC curve
	threshold_values_roc = [round(i * 0.05, 2) for i in range(21)]  # Create a list of thresholds (0.00, 0.05, 0.10, ..., 1.00)
	for threshold_value in threshold_values_roc:
		closest_idx = (np.abs(thresholds_roc - threshold_value)).argmin()
		plt.scatter(fpr[closest_idx], tpr[closest_idx], marker='o', color='red', s=50)
		plt.annotate(f"{threshold_value:.2f}", (fpr[closest_idx], tpr[closest_idx]), textcoords="offset points", xytext=(0,10), ha='center')

	# Plot Precision-Recall curve
	plt.subplot(1, 2, 2)
	plt.plot(recall, precision, color='orange', lw=2, label=f'Precision-Recall curve (AP = {average_precision:.2f})')
	plt.xlabel('Recall (Sensitivity)')
	plt.ylabel('Precision')
	plt.title('Precision-Recall Curve')
	plt.legend(loc='best')

	# Annotate specific thresholds on Precision-Recall curve
	threshold_values_pr = [round(i * 0.05, 2) for i in range(21)]  # Create a list of thresholds (0.00, 0.05, 0.10, ..., 1.00)
	for threshold_value in threshold_values_pr:
		closest_idx = (np.abs(thresholds_pr - threshold_value)).argmin()
		plt.scatter(recall[closest_idx], precision[closest_idx], marker='o', color='red', s=50)
		plt.annotate(f"{threshold_value:.2f}", (recall[closest_idx], precision[closest_idx]), textcoords="offset points", xytext=(0,10), ha='center')

	# Show the plot
	plt.tight_layout()
	plt.show()
