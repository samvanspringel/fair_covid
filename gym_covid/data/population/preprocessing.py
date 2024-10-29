import pandas as pd
import numpy as np


def preprocess_population_be(df):
    # relevant columns are 'Leeftijd' and 'Bevolking op X'
    # drop last row as it is total of population
    df = df[:-1]
    age = df['Leeftijd'].copy()
    # special case for < 1yo kids
    age[age.str.contains('Minder')] = '0'
    # other rows have 'x jaar', only keep 'x'
    age = age.str.split(' ', n=1, expand=True)[0]
    age = age.astype(int)
    # get actual population
    pop_col = [n for n in df.columns if n.startswith('Bevolking op')]
    assert len(pop_col) == 1, 'did not find population column'
    pop = df[pop_col[0]]
    return np.stack((age, pop), axis=1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Preprocessing of belgian population data')
    parser.add_argument('inp', type=str, help='a population .csv file')
    parser.add_argument('out', type=str, help='the name of the processed .csv file')
    args = parser.parse_args()
    
    df = pd.read_csv(args.inp)
    res = preprocess_population_be(df)
    pd.DataFrame(res, columns=['age', 'population']).to_csv(args.out, index=None)
    