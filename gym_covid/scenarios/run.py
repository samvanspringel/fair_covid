import sys
import os



sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import datetime

# Now you can import it
from gym_covid import *
from scenario.create_fair_env import *


def plot_states(states, alpha):
    i_hosp_new = states[:,-3].sum(axis=1)
    i_icu_new = states[:,-2].sum(axis=1)
    d_new = states[:,-1].sum(axis=1)

    axs = plt.gcf().axes
    # hospitalizations
    ax = axs[0]
    ax.plot(i_hosp_new, alpha=alpha, label='hosp', color='blue')
    ax.plot(i_icu_new,  alpha=alpha, label='icu', color='green')
    ax.plot(i_hosp_new+i_icu_new, label='hosp+icu',  alpha=alpha, color='orange')

    # deaths
    ax = axs[1]
    ax.plot(d_new, alpha=alpha, label='deaths', color='red')


def plot_simulation(states_per_stoch_run, ode_states, datapoints=None):
    _, axs = plt.subplots(2, 1)

    for states in states_per_stoch_run:
        plot_states(states, 0.2)
    plot_states(ode_states, 1.)

    if datapoints is not None:
        h = datapoints['hospitalizations']
        axs[0].scatter(np.arange(len(h)), h, facecolors='none', edgecolors='black')
        d = datapoints['deaths']
        axs[1].scatter(np.arange(len(d)), d, facecolors='none', edgecolors='black')

    axs[0].set_xlabel('days')
    axs[0].set_ylabel('hospitalizations')

    axs[1].set_xlabel('days')
    axs[1].set_ylabel('deaths')
        
    plt.show()


def simulate_scenario(env, scenario):
    states = []
    s = env.reset()
    d = False
    timestep = 0
    ret = 0
    # at start of simulation, no restrictions are applied
    action = np.ones(3)
    actions = []
    rewards = []
    today = datetime.date(2020, 3, 1)
    days = []
    while not d:
        # at every timestep check if there are new restrictions
        s = scenario[scenario['timestep'] == timestep]
        if len(s):
            print(f'timesteps {timestep}: {s["phase"]}')
            # found new restrictions
            action = np.ones(3)

        s, r, d, info = env.step(action)
        df = env.state_df()[0].to_string()
        #print(df)
        print(s[1].shape)
        exit(1)

        # state is tuple (compartments, events, prev_action), only keep compartments
        states.append(s[1])
        timestep += 1
        ret += r
        actions.append(action)
        rewards.append(r)
        for i in range(7):
            days.append(datetime.date(2020, 3, 1)+datetime.timedelta(days=(timestep-1)*7+i))
    # array of shape [Week DayOfWeek Compartment AgeGroup]
    states = np.stack(states, 0)
    #print(ret)
    # reshape to [Day Compartment AgeGroup]
    states =  np.array(states).reshape(states.shape[0]*states.shape[1], *states.shape[2:])
    print(states)
    #exit(1)
    with open('/tmp/run.csv', 'a') as f:
        f.write('dates,i_hosp_new,i_icu_new,d_new,p_w,p_s,p_l')
        i_hosp_new = states[:,-3].sum(axis=1)
        print(i_hosp_new)
        i_icu_new = states[:,-2].sum(axis=1)
        d_new = states[:,-1].sum(axis=1)
        # actions.append(actions[-1])
        actions = np.array(actions)
        rewards = np.stack(rewards, 0)
        actions = actions.repeat(7, 0)
        rewards = rewards.repeat(7, 0)
        for i in range(len(i_hosp_new)):
            f.write(f'{days[i]},{i_hosp_new[i]},{i_icu_new[i]},{d_new[i]},{actions[i][0]},{actions[i][1]},{actions[i][2]}\n')

    return states


if __name__ == '__main__':
    import gym
    from gym.wrappers import TimeLimit
    import numpy as np
    

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('scenario', type=str, help='Scenario file to be run.')
    parser.add_argument('--runs', default=1, type=int, help='Number of binomial runs. Use 0 for ODE run only. Default : 1')
    parser.add_argument('--seed', default=22122021, type=int, help='RNG seed. Default : 22122021')
    
    args = parser.parse_args()
    print(args)
    runs = args.runs

    np.random.seed(seed=args.seed)

    # load the environments
    env_type = 'ODE'
    budget = 5
    bin_env = gym.make('BECovidBinomialContinuous-v0')
    ode_env = gym.make(f'BECovidWithLockdown{env_type}Budget{budget}Continuous-v0')
    days_per_timestep = bin_env.days_per_timestep

    # simulation timesteps in weeks
    start = datetime.date(2020, 3, 1)
    end = datetime.date(2020, 9, 5)
    timesteps = round((end-start).days/days_per_timestep)
    print("TIMESTEPS:", timesteps)

    # apply timestep limit to environments
    bin_env = TimeLimit(bin_env, timesteps)
    ode_env = TimeLimit(ode_env, timesteps)

    # load scenario and convert phase-dates to timesteps
    scenario = pd.read_csv(args.scenario)
    scenario['date'] = scenario['date'].astype(str)
    to_timestep = lambda d: round((datetime.datetime.strptime(d, '%Y-%m-%d').date()-start).days/days_per_timestep)
    scenario['timestep'] = [to_timestep(d) for d in scenario['date']]
    print(ode_env)

    states_per_run = []
    for run in range(runs):
        states = simulate_scenario(ode_env, scenario)
        states_per_run.append(states)
        
    # plots assume 3 compartments
    ode_states = simulate_scenario(ode_env, scenario)
    plot_simulation(states_per_run, ode_states, ode_env.datapoints)
