import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import scipy.stats as ss

#######################################################################################################################
####################################### Numerical variables
#######################################################################################################################


def test_pearson_r(df, se_target):
    df_target = pd.DataFrame(se_target)

    df_pearson = pd.DataFrame(columns=['r', 'abs-r', 'p-value'], index=df.columns)
    # Test pearson correlation on numeric variables to find potential explanatory variables
    for col in df.columns:
        temp_df = df_target.join(df[col])

        temp_df = temp_df.dropna()

        p = ss.pearsonr(temp_df.iloc[:, 0].values, temp_df.iloc[:, 1].values)

        df_pearson.loc[col, 'r'] = p[0]
        df_pearson.loc[col, 'p-value'] = p[1]

    df_pearson['abs-r'] = np.abs(df_pearson['r'].values)

    df_pearson = df_pearson.sort_values('abs-r', ascending=False)

    return df_pearson

########################################### Test logistic model
#todo implementation factor variable

def test_logit(df, se_target, factor_variable=False):

    df_target = pd.DataFrame(se_target)

    col_names_perfo_logit = ['pseudo-R', 'accuracy', 'AUROC', 'converged']

    df_logit = pd.DataFrame(index=df.columns, columns=col_names_perfo_logit)

    for col in df.columns:
        # Make a df with only target and feature of interest
        temp_df = df_target.join(df[col])
        temp_df = temp_df.dropna()

        temp_target = pd.DataFrame(temp_df.iloc[:, 0])
        temp_df = temp_df.iloc[:, 1:].copy()

        # Fit logit reg
        logit_results = sm.Logit(temp_target.values, temp_df.values).fit()

        # Extract results from reg
        df_res = logit_results.summary2().tables[0]

        df_logit.loc[col, 'converged'] = df_res.iloc[0, 3]
        df_logit.loc[col, 'pseudo-R'] = df_res.iloc[6, 1]

        # Predicted prob
        y_hat_prob = logit_results.fittedvalues
        # Predicted values
        y_hat = (np.round(y_hat_prob))

        df_logit.loc[col, 'accuracy'] = accuracy_score(temp_target.iloc[:, 0].values, y_hat)
        df_logit.loc[col, 'AUROC'] = roc_auc_score(temp_target.iloc[:, 0].values, y_hat_prob)

    df_logit = df_logit.sort_values(['AUROC'], ascending=False)

    return df_logit

#######################################################################################################################
####################################### Factor variables
#######################################################################################################################

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def apply_cramer_v(df, se_target):

    df_target = pd.DataFrame(se_target)

    df_cramer = pd.DataFrame(columns=['cramer V'], index=df.columns)

    for col in df.columns:

        print(col)

        # Create a temp df with only pertinent variables
        temp_df = df_target.join(df[col])

        temp_df = temp_df.dropna()

        # Compute cramer V
        try:
            p = cramers_v(temp_df.iloc[:, 0].values, temp_df.iloc[:, 1].values)

        except:
            print(col)
            print('Ooooops, could not coompute cramer V for ' + col)
            print()

            p = np.nan

        # Assign cramer correlation to appropriate variable
        df_cramer.loc[col, 'cramer V'] = p

    df_cramer = df_cramer.sort_values(['cramer V'], ascending=False)

    return df_cramer


"""
This function input two df, with columns name and returns a table with the computed WOE and IV

source:
    https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
"""

def compute_woe_iv(df, se_target, col_factor, drop_na=False, na_value='NA'):

    # convert serie to df
    df_target = pd.DataFrame(se_target)
    col_target = se_target.name

    # Gen df for the score of this function
    temp_df = df_target.join(df[col_factor])
    temp_df['count'] = 1

    if(drop_na):
        temp_df = temp_df.dropna()

    else:
        temp_df[col_factor] = temp_df[col_factor].astype('str')
        temp_df[col_factor] = temp_df[col_factor].fillna(na_value)
        temp_df[col_factor] = temp_df[col_factor].astype('str')

    # Since the target variable is binary (1 or 0), we can sum to find number of event and sum of observation
    df_woe = temp_df.groupby([col_factor]).sum()

    # Compute non-event per factor
    df_woe['non_event'] = df_woe['count'] - df_woe[col_target]

    # Compute ratio of event and non-event
    df_woe['ratio_event'] = df_woe[col_target] / df_woe['count']
    df_woe['ratio_non_event'] = 1 - df_woe['ratio_event']

    # Drop factors without variation of the target
    df_woe = df_woe.loc[(0 < df_woe['ratio_event']) & (df_woe['ratio_event'] < 1), :]

    # Compute sum of total observation and total event and non-event
    df_woe['count_total'] = df_woe['count'].sum()
    df_woe['event_total'] = df_woe[col_target].sum()
    df_woe['non_event_total'] = df_woe['count_total'] - df_woe['event_total']

    # Compute relative importance of a factor for all the events and the non-events

    df_woe['frac_event'] = df_woe[col_target] / df_woe['event_total']
    df_woe['frac_non_event'] = df_woe['non_event'] / df_woe['non_event_total']

    # Compute WOE and adjusted WOE
    df_woe['WOE'] = np.log(df_woe['frac_non_event'] / df_woe['frac_event'])

    df_woe['adj_WOE_num'] = (df_woe['event_total'] + 0.5) / df_woe['non_event_total']
    df_woe['adj_WOE_den'] = (df_woe[col_target] + 0.5) / df_woe['event_total']

    df_woe['adj_WOE'] = np.log(df_woe['adj_WOE_num'] / df_woe['adj_WOE_den'])

    # Compute individual IV and adj IV

    df_woe['IV_i'] = (df_woe['frac_non_event'] - df_woe['frac_event']) * df_woe['WOE']
    df_woe['adj_IV_i'] = (df_woe['frac_non_event'] - df_woe['frac_event']) * df_woe['adj_WOE']

    # Compute sum IV and adj IV

    df_woe['IV'] = df_woe['IV_i'].sum()
    df_woe['adj_IV'] = df_woe['adj_IV_i'].sum()

    return df_woe


"""
This function apply the computation of WOE and IV to a df of categorical features
"""


def apply_WOE_IV(df, se_target):
    
    df_target = pd.DataFrame(se_target)

    df_iv = pd.DataFrame(columns=['IV', 'adj_IV'], index=df.columns)

    for col_factor in df.columns:
        temp_df_woe = compute_woe_iv(df, se_target, col_factor)

        df_iv.loc[col_factor, 'IV'] = temp_df_woe.loc[temp_df_woe.index[0], 'IV']
        df_iv.loc[col_factor, 'adj_IV'] = temp_df_woe.loc[temp_df_woe.index[0], 'adj_IV']

    df_iv = df_iv.sort_values('IV', ascending=False)

    return df_iv
