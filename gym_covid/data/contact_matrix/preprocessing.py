import pyreadr
import numpy as np
import pandas as pd


def prepocess_scm_be(df, pop):
    # partial matrices are saved in this order as last dimension of df
    names = ['home', 'work', 'school', 'transport', 'leisure', 'otherplace']
    # get partial matrices from df
    matrices = [df['matrix_all'][:,:,i] for i in range(len(names))]
    matrices = np.array(matrices)
    # partial matrices only go from age=0 to 80+. Make 90+ group by duplicating 80+ data
    matrices = np.concatenate((matrices, matrices[:,:,[-1]]), axis=-1)
    matrices = np.concatenate((matrices, matrices[:,[-1],:]), axis=1)

    # [ 0., 10., 20., 30., 40., 50., 60., 70., 80., inf]
    age_groups = np.concatenate((np.arange(0, 90, 10), (np.inf,)))
    g = pd.cut(pop.age, age_groups, right=False)
    pop['group'] = g
    population = pop.groupby('group').agg('sum').population
    # as with partial matrices, duplicate 80+ data to make 90+ group
    population_groups = np.concatenate((population, population[-1:]))

    # convert matrices to a per-capita rate
    population_groups = population_groups.repeat(len(population_groups)).reshape(len(population_groups), len(population_groups))
    matrices = matrices/population_groups[None]

    return {names[i]: matrices[i] for i in range(len(names))}



if __name__ == '__main__':
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Preprocessing of social contact matrices (SCM)')
    parser.add_argument('inp', type=str, help='a SCM .RData file')
    parser.add_argument('pop', type=str, help='the location of the population .csv file')
    parser.add_argument('out', type=str, help='output dir where the SCM matrices will be saved as .csv files')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pyreadr.read_r(args.inp)
    pop = pd.read_csv(args.pop)
    res = prepocess_scm_be(df, pop)

    for k, v in res.items():
        pd.DataFrame(v).to_csv(out / f'{k}.csv', index=None, header=False)