from scenario.main_pcn_core import *
import matplotlib.pyplot as plt


def main2():
    env = create_covid_env([])
    obs = env.reset()

    objectives = ["R_ARI", "R_ARH", "R_SB_W", "R_SB_S", "R_SB_L", "SB_SUM"]
    desired_return = np.array([100, 100, 0, -5, 0, 0],
                              dtype=np.float32)

    current_return = desired_return.copy()
    horizon_weeks = 1.0
    current_horizon = 3


    pcn = torch.load('/Users/samvanspringel/Documents/School/VUB/Master 2/Jaar/Thesis/fair_covid/fairRLresults/'
                       'covid/2024-11-26_11-45-30/model_10.pt')
    pcn.eval()

    all_actions = []
    all_hospitalizations = []
    all_sbs = []

    for t in range(int(horizon_weeks)):
        # model expects (obs, desired_return, desired_horizon)
        action = choose_action(pcn, obs, current_return, current_horizon, eval=True)

        obs, reward, done, info = env.step(action)
        #print(f"{t}: {info}")
        state = obs[0]
        # TODO check shape of obs compared to state in run.py
        print(obs[0].shape)
        state_df = env.state_df()

        hospitalizations = state_df["I_sev"]

        all_hospitalizations.append(hospitalizations)

        #current_horizon = max(current_horizon - 1, 1)

        all_actions.append(action)

        #if done:
        #    break

    plot_compartment(all_hospitalizations, "I_sev")

    plot_actions(all_actions)


def plot_compartment(list_compartment, label):
    list_by_age_group = list(zip(*list_compartment))
    age_groups = ["[0, 10[", "[10, 20[", "[20, 30[", "[30, 40[", "[40, 50[", "[50, 60[", "[60, 70[", "[70, 80[",
                  "[80, 90[", "[90, inf["]

    # Plot the evolution for each age group
    plt.figure(figsize=(10, 6))
    for age_group, values in zip(age_groups, list_by_age_group):
        plt.plot(values, label=age_group, marker='o')

    # Add labels and title
    plt.title(f"Evolution of {label} Values for Each Age Group")
    plt.xlabel("Time Steps")
    plt.ylabel(f"{label} Values")
    plt.legend(title="Age Groups")
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_actions(all_actions):
    action_labels = ["Reduction Work", "Reduction School", "Reduction Leisure"]

    # Create a plot
    plt.figure(figsize=(10, 6))

    # Transpose the data to group by actions
    actions_by_dimension = list(zip(*all_actions))

    # Plot each action dimension
    for action_label, action_values in zip(action_labels, actions_by_dimension):
        plt.plot(action_values, marker='o', label=action_label)

    # Customize the plot
    plt.title("Evolution of Continuous Reductions")
    plt.xlabel("Time Steps")
    plt.ylabel("Reductions")
    plt.xticks(range(len(all_actions)), labels=[f"{i + 1}" for i in range(len(all_actions))])
    plt.legend(title="Actions")
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.show()

def simulate_scenario(env):
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
    new_h = []
    while not d:

        # at every timestep check if there are new restrictions
        # action =

        s, r, d, info = env.step(action)
        #s = ([0],) + s
        df = env.state_df()[0].to_string()
        #print(df)
        print(len(s[1]))
        #exit(1)
        # state is tuple (compartments, events, prev_action), only keep compartments
        #curr_state = s[1].T
        #.append(curr_state)
        states.append(s[1])
        timestep += 1
        ret += r
        actions.append(action)
        rewards.append(r)
        for i in range(7):
            days.append(datetime.date(2020, 3, 1)+datetime.timedelta(days=(timestep-1)*7+i))
    # array of shape [Week DayOfWeek Compartment AgeGroup]

    states = np.stack(states, 0)

    print(states)
    # reshape to [Day Compartment AgeGroup]
    states = np.array(states).reshape(states.shape[0]*states.shape[1], *states.shape[2:])
    #exit(1)

    with open('/tmp/test.csv', 'a') as f:
        f.write('dates,i_hosp_new,i_icu_new,d_new,p_w,p_s,p_l')
        df = env.state_df()[0]
        i_hosp_new = states[:,-3].sum(axis=1)
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
    #main2()

    runs = 1

    env = create_covid_env([])
    print(env)

    days_per_timestep = 7

    # simulation timesteps in weeks
    start = datetime.date(2020, 3, 1)
    end = datetime.date(2020, 9, 5)
    timesteps = round((end-start).days/days_per_timestep)

    # apply timestep limit to environments
    env = TimeLimit(env, timesteps)

    states_per_run = []
    for run in range(runs):
        states_run = simulate_scenario(env)
        states_per_run.append(states_run)


