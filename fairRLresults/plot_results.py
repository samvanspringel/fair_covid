from scenario.main_pcn_core import *
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
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
        state_df = env.state_df()

        hospitalizations = state_df["I_sev"]

        all_hospitalizations.append(hospitalizations)

        #current_horizon = max(current_horizon - 1, 1)

        all_actions.append(action)

        #if done:
        #    break

    plot_compartment(all_hospitalizations, "I_sev")

    plot_actions(all_actions)

