import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from scenario.create_fair_env import create_covid_env
from scenario.main_pcn_core import *

# -------------------------------
# Keep your scaling parameters:
scale = np.array([800000, 10000, 50., 20, 50, 100])
ref_point = np.array([-15000000, -200000, -1000., -1000., -1000., -1000.]) / scale
max_return = np.array([0, 0, 0, 0, 0, 0]) / scale

MODEL_PATH = "/Users/samvanspringel/Documents/School/VUB/Master 2/Jaar/Thesis/fair_covid/fairRLresults/cluster/steps_300000/objectives_R_ARH:R_SB_W:R_SB_S:R_SB_L_SBS:ABFTA/distance_metric_none/seed_0/2025-03-17_17-29-13/model_10.pt"  # your single best model
N_RUNS = 10000
OUTPUT_CSV = "pareto_points_1.csv"


def run_policy_with_params(model, desired_return, desired_horizon):
    env = create_covid_env([])
    env.reset()

    done = False
    totalHosp = 0.0
    totalLostContacts = 0.0

    while not done:
        state_df, C_diff_6x10x10 = env.state_df()

        # sum daily new hospital admissions
        daily_hosp = state_df["I_hosp_new"].sum() if "I_hosp_new" in state_df else 0.0
        totalHosp += daily_hosp

        # sum lost contacts from the 6x10x10 matrix
        if C_diff_6x10x10 is not None:
            totalLostContacts += C_diff_6x10x10.sum()

        budget = np.ones(3) * 4
        action_so_far = env.current_action
        events = env.current_events_n
        state = env.current_state_n


        action = choose_action(
            model,
            (budget, state, events, action_so_far),
            desired_return,
            desired_horizon,
            eval=True
        )

        _, _, done, info = env.step(action)

    return totalHosp, totalLostContacts


def main():
    # load your single best model
    model = torch.load(MODEL_PATH)
    model.eval()

    rows = []
    returns = [
        [-5.1892776, -7.0472817, -8.265566, -4.621168, -0.6635878, -4.490676],
        [-4.454619, -5.2865844, -7.8608537, -2.3692327, -2.6395705, -4.7700486],
        [-4.327653, -5.8891516, -9.125591, -4.9921074, -0.8018981, -4.968472],
        [-3.4918618, -4.273028, -11.227012, -5.264081, -0.73639953, -5.8621016],
        [-5.760371, -7.7996206, -7.303715, -4.4191437, -0.51790714, -3.995533],
        [-2.519569, -3.187968, -12.248552, -5.071588, -1.3338823, -6.504613],
        [-7.8118687, -9.822556, -3.8101838, -3.4873881, -4.06822, -3.8639],
        [-3.9419968, -5.010753, -11.78131, -3.9191167, -0.61513436, -5.8183713],
        [-5.2960443, -7.1813564, -7.6987653, -4.5016427, -0.5481895, -4.186505],
        [-4.2376127, -5.3106103, -11.686236, -3.3974411, -0.77348745, -5.7577915],
        [-4.1146092, -5.1522927, -11.693502, -3.4304042, -0.63877374, -5.710182],
        [-5.322415, -6.2332983, -4.5095544, -2.903651, -6.6354833, -5.1277075],
        [-4.124155, -5.1495185, -11.809991, -3.4377213, -0.6377319, -5.759505],
        [-9.888371, -12.994406, -0.67034703, -4.230611, -0.78106695, -1.3098576],
        [-8.065246, -10.227113, -7.5174, -2.7512531, -0.54120183, -3.8162928],
        [-5.5279465, -7.4517517, -7.514515, -4.33261, -0.7403988, -4.161649],
        [-8.903418, -11.779604, -7.1205196, -4.199131, -0.8159727, -4.0067267],
        [-8.751562, -10.9119625, -7.6697426, -2.8089592, -0.6811794, -3.9477112],
        [-2.5783885, -4.0407085, -12.294, -5.473369, -0.5694083, -6.2719812],
        [-2.4185083, -3.7533653, -12.478023, -5.578001, -0.64769626, -6.3987164],
        [-7.7106953, -9.854736, -7.533856, -3.068138, -0.6416374, -3.9178123],
        [-2.2861807, -2.9857996, -12.632467, -5.288892, -0.6242491, -6.405113],
        [-7.5456944, -9.487881, -7.452866, -2.4798799, -0.65796554, -3.7928267],
        [-2.9728642, -4.4737225, -12.108289, -5.517229, -0.5414621, -6.1902685],
        [-5.4398503, -7.3461504, -8.107417, -4.821106, -0.54473877, -4.4085827],
        [-1.6728479, -2.1866155, -13.083144, -5.4766874, -1.3625797, -6.9318333],
        [-4.0943284, -4.209968, -10.983944, -5.6213064, -0.9061021, -5.89107],
        [-5.4418583, -7.381099, -8.148224, -4.8929076, -0.79276896, -4.540898]
    ]

    for i in range(len(returns)):
        desired_return = np.array(returns[i]).astype(np.float32) #np.random.uniform(ref_point, max_return).astype(np.float32)

        # Also vary horizon if you want
        desired_horizon = 15 #np.random.randint(1, 12)  # e.g. random in [1..12]

        hosp, lostC = run_policy_with_params(model, desired_return, desired_horizon)
        rows.append({
            "desired_return": desired_return.tolist(),
            "desired_horizon": float(desired_horizon),
            "totalHosp": hosp,
            "totalLostContacts": lostC,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} points to {OUTPUT_CSV}")

    xvals = df["totalHosp"] / 1e5
    yvals = -df["totalLostContacts"]  # might be negative, so it goes downward
    #xvals, yvals = np.array(xvals, yvals) * np.array([[10000], [100]])
    plt.scatter(xvals, yvals, marker="o", color="blue")

    plt.xlabel("Cumulative number of daily new hospitalizations ×10^5 (negated)")
    plt.ylabel("Cumulative lost contacts")
    plt.title("Approx. Pareto front from model_10 with random desired_returns")
    plt.show()

def plot_pareto_points():
    df = pd.read_csv(OUTPUT_CSV)
    xvals = df["totalHosp"] / 1e5
    yvals = -df["totalLostContacts"]  # might be negative, so it goes downward
    #xvals, yvals = np.array(xvals, yvals) * np.array([[10000], [100]])
    plt.scatter(xvals, yvals, marker="o", color="blue")

    plt.xlabel("Cumulative number of daily new hospitalizations ×10^5 (negated)")
    plt.ylabel("Cumulative lost contacts")
    plt.title("Approx. Pareto front from model_10 with random desired_returns")
    plt.show()
if __name__ == "__main__":
    #plot_pareto_points()
    main()