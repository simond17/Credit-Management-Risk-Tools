import pandas as pd


def gen_hot_encoded(df, col, regroup_smallest_class=False):
    print('Treatment of col : ' + col)
    print('')

    temp_col = df[col].copy()

    temp_col_value_counts = temp_col.value_counts(normalize=True)

    most_common_class_name = str(temp_col_value_counts.index[0])

    print(temp_col_value_counts)

    if (regroup_smallest_class):
        smallest_1_class_name = str(temp_col_value_counts.index[-1])
        smallest_2_class_name = str(temp_col_value_counts.index[-2])

        print('...')
        print('Regrouping smallest class ' + smallest_1_class_name + ' with ' + smallest_2_class_name)

        new_smallest_class_name = smallest_2_class_name + '_' + smallest_1_class_name

        temp_col.loc[(temp_col == temp_col_value_counts.index[-1]) | (
                temp_col == temp_col_value_counts.index[-2])] = new_smallest_class_name

    # Gen the
    df_col_dummies = pd.get_dummies(temp_col, prefix=col, prefix_sep='_')

    # Drop the most frequent category
    df_col_dummies = df_col_dummies.drop(col + '_' + most_common_class_name, axis=1)

    return df_col_dummies


def gen_hot_encoded_df(df):
    lst_df_dummies = list()

    for col in df.columns:
        df_temp = gen_hot_encoded(df=df, col=col, regroup_smallest_class=False)

        lst_df_dummies.append(df_temp)

    df_dummies = pd.concat(lst_df_dummies, axis=1)

    return df_dummies


def check_for_perfect_predictor_df(se_target, df):
    df_crosstab = pd.DataFrame(index=df.columns, columns=['nb_1_1', 'ratio_1_1', 'perfect_predictor'])

    nb_obs = df.shape[0]

    for col in df.columns:

        temp_crosstab = pd.crosstab(index=df.index, columns=[se_target, df[col]])

        nb_1_1 = temp_crosstab.iloc[:, 1].sum()
        ratio_1_1 = nb_1_1 / nb_obs

        if (ratio_1_1 == 0) | (ratio_1_1 == 1):

            print(col + ' is a perfect predictor')
            perfect_predictor = True

        else:
            perfect_predictor = False

        df_crosstab.loc[col, 'nb_1_1'] = nb_1_1
        df_crosstab.loc[col, 'ratio_1_1'] = ratio_1_1
        df_crosstab.loc[col, 'perfect_predictor'] = perfect_predictor

    return df_crosstab

