import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import numpy as np

class DataVisualizer:
    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df

        temp_train = self.train_df.copy()
        temp_train['whoami']='train'
        temp_test = self.test_df.copy()
        temp_test['whoami'] = 'test'
        self.concat_df = pd.concat([temp_train,temp_test],axis = 0)
        # return concat_df

    def visualize_numerical_distribution(self):
        numerical_cols = self.train_df.select_dtypes(include=['int', 'float']).columns

        num_numerical_cols = len(numerical_cols)
        ncols = 3
        nrows = num_numerical_cols

        fig, axes = plt.subplots(nrows, ncols, figsize=(30, 5 * nrows))

        for i, col in enumerate(numerical_cols):

            sns.histplot(x=self.train_df[col], ax=axes[i, 0], color='#F8766D', label=col, fill=True)
            sns.histplot(x=self.test_df[col], ax=axes[i, 0], color='#00BFC4', label=col, fill=True)

            sns.kdeplot(x=self.train_df[col], ax=axes[i, 1], color='#F8766D', label=col, fill=True)
            sns.kdeplot(x=self.test_df[col], ax=axes[i, 1], color='#00BFC4', label=col, fill=True)
            sns.boxplot(x = col,y='whoami',data=self.concat_df,ax = axes[i,2],orient='h',hue='whoami')

            axes[i,0].legend()
            axes[i,1].legend()
            axes[i,0].title.set_text("Histogram Plot")
            axes[i,1].title.set_text("Distribution Plot")
            axes[i,2].title.set_text("Box Plot")

        fig.tight_layout()
        plt.show()

    def visualize_categorical_distribution(self):
        categorical_cols = self.train_df.select_dtypes(include=['object']).columns.tolist()
        n_rows = len(categorical_cols)//2 + 1
        plt.figure(figsize = (30,n_rows*10))
        for i in range(len(categorical_cols)):
            plt.subplot(len(categorical_cols),2,i+1)
            sns.countplot(data=self.concat_df,x=categorical_cols[i],hue='whoami',orient='h')
            plt.title(categorical_cols[i])
        plt.legend()
        plt.show()

    def plot_correlation_heatmap(self, title_name='Correlation Heatmap', excluded_columns=[]):
        # Copy the DataFrame to avoid modifying the original
        df_encoded = self.train_df.copy()

        # Encode categorical variables if not already encoded
        categorical_vars = df_encoded.select_dtypes(include=['object']).columns.tolist()
        self.label_encoders = {}
        for column in categorical_vars:
            le = LabelEncoder()
            df_encoded[column] = le.fit_transform(df_encoded[column])
            self.label_encoders[column] = le

        # Remove excluded columns
        columns_without_excluded = [col for col in df_encoded.columns if col not in excluded_columns]

        # Calculate correlation matrix
        corr = df_encoded[columns_without_excluded].corr()
    
        # Create heatmap
        fig, axes = plt.subplots(figsize=(14, 10))
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(corr, mask=mask, linewidths=.5, cmap='YlOrBr_r', annot=True, annot_kws={"size": 6})
        plt.title(title_name)
        plt.show()



