

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