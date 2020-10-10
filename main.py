
import pandas as pd
from univariateAnalysis import test_logit, test_pearson_r, apply_WOE_IV, apply_cramer_v

# Read df and assign index
data_path = "datasets/home-credit-defaut-risk/"
df = pd.read_csv(data_path + "application_train.csv")
df = df.set_index("SK_ID_CURR")

# We arbitrarily create rejected observations
# The objective is to simulate a proper dataset for reject inference
# The arbitrary rule to reject an application is that EXT_SOURCE_2 < 0.39, which represent the lowest quartile
df['rejected'] = 0
df.loc[df['EXT_SOURCE_2'] < 0.39, 'rejected'] = 1

# Gen series with rejected and target (default)
se_target = df['TARGET'].copy()
se_target.name = 'target'
se_rejected = df['rejected'].copy()

# Drop from df
df = df.drop(['TARGET', 'rejected'], axis=1)

# gen df with only numerical and factor variables
df_num = df.select_dtypes(include='number')
df_cat = df.select_dtypes(exclude='number')

#######################################################################################################################
############################################ Univariate analysis
#######################################################################################################################

## Numerical variables
# Pearson correlation on numerical variables
df_pearson = test_pearson_r(df=df_num, se_target=se_rejected)

# Univariate logistic regression with numerical variable
df_uni_logit = test_logit(df=df_num, se_target=se_rejected, factor_variable=False)

# todo test cramer v and WOE



'''
input df; col (col name)
output a df with dummy encoded variable, the reference category is the most frequent
'''

df = df_cat
col = 'NAME_INCOME_TYPE'




df_dummies = gen_hot_encoded(df_cat, col='NAME_INCOME_TYPE', regroup_smallest_class=True)

# Test category of variables

# t-square test

#
# https://www.researchgate.net/post/Can_I_use_Pearsons_correlation_coefficient_to_know_the_relation_between_perception_and_gender_age_income#:~:text=You%20can%20use%20chi%20square,be%20measured%20with%20Spearman%20coefficient.&text=If%20the%20categorical%20variable%20has,equivalent%20to%20the%20Pearson%20correlation


# for col in df_cat.columns:
#     print(col)
#     temp_count = df_cat[col].value_counts()
#     print(temp_count)
#
# df_num.describe()

# make a model to predict loan acceptance
