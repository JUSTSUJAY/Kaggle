import pandas as pd
from colorama import Style, Fore
from IPython.display import display

blk = Style.BRIGHT + Fore.BLACK
red = Style.BRIGHT + Fore.RED
blu = Style.BRIGHT + Fore.BLUE
cyan = Style.BRIGHT + Fore.CYAN
green = Style.BRIGHT + Fore.GREEN
res = Style.RESET_ALL

class DataExplorer:
    def __init__(self, train):
        self.train = train

    def nullExploration(self):
        print("Null Exploration")
        if self.train.isnull().sum().sum() > 0:
            print(f"{red}Nulls found{res}")
            null = dict()
            for col in self.train.columns.tolist():
                if self.train[col].isnull().sum():
                    null[col] = self.train[col].isnull().sum()

            display(pd.DataFrame({'Column Name': list(null.keys()), 'Null Values': list(null.values())}))
        else:
            print(f"{green}No Nulls found{res}")

    def columnSaggregation(self):
        total_cols = len(self.train.columns)
        num = [col for col in self.train.columns if self.train[col].dtype in ['int64', 'float64']]
        cat = [col for col in self.train.columns if self.train[col].dtype not in ['int64', 'float64']]
        return total_cols, num, cat

    def unique_in_cat(self, cat):
        d = {col: self.train[col].nunique() for col in cat}
        return d

    def describe_df(self):
        def Dtype(feature):
            return self.train[feature].dtype

        def NULL(feature):
            return self.train[feature].isna().any()

        def unique_percent(feature):
            return round((self.train[feature].nunique() / len(self.train[feature])) * 100, 2)

        def possible_outliers_2_std(feature):
            mean = self.train[feature].mean()
            std = self.train[feature].std()
            maxi = self.train[feature].max()
            return maxi > (mean + 2 * std)

        def possible_outliers_3_std(feature):
            mean = self.train[feature].mean()
            std = self.train[feature].std()
            maxi = self.train[feature].max()
            return maxi > (mean + 3 * std)

        def outlier_percentage_2(feature):
            mean = self.train[feature].mean()
            std = self.train[feature].std()
            outlier_per = len(self.train[self.train[feature] > mean + 2 * std]) / len(self.train[feature]) * 100
            return outlier_per

        def outlier_percentage_3(feature):
            mean = self.train[feature].mean()
            std = self.train[feature].std()
            outlier_per = round((len(self.train[self.train[feature] > mean + 3 * std]) / len(self.train[feature]) * 100), 2)
            return outlier_per
        cat = [col for col in self.train.columns if self.train[col].dtype not in ['int64', 'float64']]
        dtypes = self.train.drop(columns=cat).columns.map(Dtype)
        nulls = self.train.drop(columns=cat).columns.map(NULL)
        uniques = self.train.drop(columns=cat).columns.map(unique_percent)
        std_2_outliers = self.train.drop(columns=cat).columns.map(possible_outliers_2_std)
        std_3_outliers = self.train.drop(columns=cat).columns.map(possible_outliers_3_std)
        outlier_per_2 = self.train.drop(columns=cat).columns.map(outlier_percentage_2)
        outlier_per_3 = self.train.drop(columns=cat).columns.map(outlier_percentage_3)

        describe = pd.DataFrame(self.train.drop(columns=cat).describe(include='all').T)
        describe['Dtype'] = dtypes
        describe['Nulls'] = nulls
        describe['Unique%'] = uniques
        describe['STD2'] = std_2_outliers
        describe['STD3'] = std_3_outliers
        describe['outlier_>2std%'] = outlier_per_2
        describe['outlier_>3std%'] = outlier_per_3

        def unique_style(value):
            return 'color: red' if value < 1 else ''

        def std3_style(value):
            return 'color: red' if value > 1 else ''

        styled_df = describe.style.set_properties(**{
            'background-color': 'black',
            'color': 'lawngreen',
            'border-color': 'white'
        }).applymap(unique_style, subset=['Unique%']).applymap(std3_style, subset=['outlier_>3std%'])

        return styled_df



    def info(self):
        print(f"{res}SHAPE OF THE DATA:{green} {self.train.shape}")

        total_cols, num, cat = self.columnSaggregation()
        print(f"{res}The dataframe has {green} {total_cols} ({len(num)} numeric + {len(cat)} categorical){res} columns")
          
        self.nullExploration()
        
        print(f'{blk}Unique Categorical values')
        uniques = self.unique_in_cat(cat)  
        display(pd.DataFrame({'Categorical Column': list(uniques.keys()), 'Number of Unique Values': list(uniques.values())}))
        print(f'{blk}Data Describe')
        described = self.describe_df()
        display(described)
        print(f'{blk}Data sample')
        display(self.train.sample(5))
