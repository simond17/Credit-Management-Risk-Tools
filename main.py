import pandas as pd
from univariateAnalysis import test_logit, test_pearson_r, apply_WOE_IV, apply_cramer_v
from dataProcessing import gen_hot_encoded_df
import statsmodels.api as sm

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

################# Numerical variables

# Pearson correlation on numerical variables
df_pearson = test_pearson_r(df=df_num, se_target=se_rejected)

# Univariate logistic regression with numerical variable
df_uni_logit = test_logit(df=df_num, se_target=se_rejected, factor_variable=False)

################# Factor variables

# Test cramer correlation on target variable
df_cramer = apply_cramer_v(df=df_cat, se_target=se_rejected)

# Compute information value on target variable
df_iv = apply_WOE_IV(df=df_cat, se_target=se_rejected)

print(df_iv)

#######################################################################################################################
############################################ Reject inference
#######################################################################################################################

##### Data preparation

independant_cat_vars = ['NAME_EDUCATION_TYPE', 'HOUSETYPE_MODE', 'NAME_INCOME_TYPE']

independant_num_vars = ['REGION_RATING_CLIENT', 'DAYS_LAST_PHONE_CHANGE', 'EXT_SOURCE_1', 'REGION_POPULATION_RELATIVE']

df_num_hot = df_num[independant_num_vars].copy()
df_cat_hot = df_cat[independant_cat_vars].copy()

for col in df_num_hot:
    df_num_hot[col] = pd.cut(df_num[col], bins=4)

df_num_bin = df_num_hot.astype('category')

df_num_hot = gen_hot_encoded_df(df_num_bin)
df_cat_hot = gen_hot_encoded_df(df_cat_hot)

X = pd.concat([df_num_hot, df_cat_hot], axis=1)
X['constant'] = 1

y = se_rejected
# Fit logit model for inference

logit_model = sm.Logit(endog=y, exog=X).fit()

crosstab = pd.crosstab(index=X.index, columns=[y, X['constant']])

crosstab = check_for_perfect_predictor(se_target=y, df=X)