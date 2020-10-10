
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


