import numpy as np
import pandas as pd
import random
import math
import pickle


def check_datatypes(df, name_class):
    non_numeric_columns = df.select_dtypes(exclude=[int, float]).columns.tolist()
    non_numeric_columns += name_class
    # Get the indexes of the non-numeric columns
    non_numeric_indexes = [df.columns.get_loc(col) for col in non_numeric_columns]

    return non_numeric_indexes


def out_of_range(minimum, maximum):
    foo = ["up", "down"]
    f = random.choice(foo)
    dist = maximum - minimum

    if f == "up":
        number = random.uniform(maximum, maximum + dist * 1.2)
    else:
        number = random.uniform(minimum - dist * 1.2, minimum)

    return number


def injection(df_pandas, seed, name, name_class):
    np.random.seed(seed)
    random.seed(seed)

    df_list = []
    minimum = []
    maximum = []

    # percentuale di errori
    perc = [0.5]
    for p in perc:
        prev_row = None
        prev_val = None
        df_dirt = df_pandas.copy()
        cols = df_dirt.columns
        excluded_columns = check_datatypes(df_dirt, name_class)
        rows, mask = create_matrix_mask(df_dirt.shape[0], df_dirt.shape[1], p, excluded_columns)

        for j in range(len(df_dirt.columns)):
            if j not in excluded_columns:
                minimum.append(float(df_dirt[cols[j]].min()))
                maximum.append(float(df_dirt[cols[j]].max()))
            else:
                minimum.append(0)
                maximum.append(0)

        for i in range(mask.shape[0]):
            col = mask[i,1]
            if mask[i, 0] == prev_row:
                if math.isnan(prev_val):
                    mask[i,2] = None
                else:
                    mask[i,2] = out_of_range(minimum[int(col)], maximum[int(col)])
            else:
                c=random.random()
                if c>=0.5:
                    mask[i,2] = out_of_range(minimum[int(col)], maximum[int(col)])
                else:
                    mask[i,2] = None
            prev_row = mask[i, 0]
            prev_val = mask[i, 2]

        for row, col, value in mask:
            df_dirt.iat[int(row), int(col)] = value

        rows = np.nonzero(rows)[0].tolist()

        df_list.append(df_dirt)
        print("saved {}-accuracy {}%".format(name, round((1 - p) * 100)))

        df_list[0].to_csv("../../Datasets/Occupancy_injected_mix.csv", sep=",", index=False, na_rep=' NA')
    return df_list, rows


def create_matrix_mask(rows, cols, p, not_acceptable):
    mask_rows = np.random.choice([True, False], rows, p=[p, 1 - p])
    matrix_mask = np.full((rows, cols), False, dtype=bool)
    for row_idx, is_true in enumerate(mask_rows):
        if is_true:
            bool_array = np.full(cols, False, dtype=bool)
            num_values = np.random.choice([1, 2, 3], size=1, p=[0.7, 0.2, 0.1])
            true_indices = np.random.choice([x for x in range(cols) if x not in not_acceptable], num_values,
                                            replace=False)
            bool_array[true_indices] = True
            matrix_mask[row_idx] = bool_array

    true_indices = np.argwhere(matrix_mask)
    outliers = np.append(true_indices, np.empty((true_indices.shape[0], 1)), axis=1)

    return mask_rows, outliers


if __name__ == '__main__':
    path = "../../Datasets/Occupancy_Estimation_Used.csv"
    df = pd.read_csv(path, sep=",")
    injection(df, seed=1, name='PRSA', name_class=['Room_Occupancy_Count'])
