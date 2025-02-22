import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for Dash
from io import BytesIO
import base64
import pandas as pd
from scenario.main_pcn_core import *
import matplotlib.pyplot as plt

from dash import Dash, html, dcc, Input, Output, State, dash_table, no_update
import dash_bootstrap_components as dbc

# Assume these are defined similarly to your original code
objectives = ["R_ARI", "R_ARH", "R_SB_W", "R_SB_S", "R_SB_L", "SB_SUM"]

# Directory where model files (model_*.pt) are stored:
MODEL_DIR = "/Users/samvanspringel/Documents/School/VUB/Master 2/Jaar/Thesis/fair_covid/fairRLresults/covid/2024-12-12_12-43-27/"

def list_models():
    # List all model_i.pt files from MODEL_DIR
    files = os.listdir(MODEL_DIR)
    model_files = [f for f in files if f.startswith("model_") and f.endswith(".pt")]
    return sorted(model_files)

def plot_compartment(list_compartment, label):
    list_by_age_group = list(zip(*list_compartment))
    age_groups = ["[0, 10[", "[10, 20[", "[20, 30[", "[30, 40[", "[40, 50[", "[50, 60[", "[60, 70[", "[70, 80[",
                  "[80, 90[", "[90, inf["]

    plt.figure(figsize=(10, 6))
    for age_group, values in zip(age_groups, list_by_age_group):
        plt.plot(values, label=age_group, marker='o')

    plt.title(f"Evolution of {label} Values for Each Age Group")
    plt.xlabel("Time Steps")
    plt.ylabel(f"{label} Values")
    plt.legend(title="Age Groups")
    plt.grid(True)
    plt.tight_layout()

    # Convert plot to PNG image and return as base64
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    encoded = base64.b64encode(img.read()).decode("utf-8")
    return "data:image/png;base64,{}".format(encoded)

def plot_actions(all_actions):
    action_labels = ["Reduction Work", "Reduction School", "Reduction Leisure"]

    plt.figure(figsize=(10, 6))
    actions_by_dimension = list(zip(*all_actions))
    for action_label, action_values in zip(action_labels, actions_by_dimension):
        plt.plot(action_values, marker='o', label=action_label)

    plt.title("Evolution of Continuous Reductions")
    plt.xlabel("Time Steps")
    plt.ylabel("Reductions")
    plt.xticks(range(len(all_actions)), labels=[f"{i + 1}" for i in range(len(all_actions))])
    plt.legend(title="Actions")
    plt.grid(True)
    plt.tight_layout()

    # Convert to image
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    encoded = base64.b64encode(img.read()).decode("utf-8")
    return "data:image/png;base64,{}".format(encoded)

def load_parameters_from_file():
    df = pd.read_csv(MODEL_DIR + "pcn_log.csv")
    reference = df.iloc[-1]

    objectives_desired_returns = [reference["return_0_desired"],
                                  reference["return_1_desired"],
                                  reference["return_2_desired"],
                                  reference["return_3_desired"],
                                  reference["return_4_desired"],
                                  reference["return_5_desired"]]
    horizon = reference["desired_horizon"]
    desired_horizon = reference["horizon_distance"]

    return objectives_desired_returns, horizon, desired_horizon

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

model_options = [{"label": m, "value": m} for m in list_models()]

objective_inputs = []
for i, obj_name in enumerate(objectives):
    objective_inputs.append(
        dbc.InputGroup([
            dbc.InputGroupText(obj_name),
            dbc.Input(id=f"obj-input-{i}", type="number", value=0.0)
        ], className="mb-2")
    )

app.layout = dbc.Container([
    # Header Row
    dbc.Row([
        dbc.Col([
            html.H1("PCN Simulation Dashboard"),
            html.P("Select a model and desired returns, then run the simulation.")
        ])
    ]),

    # Inputs and Plots
    dbc.Row([
        # Input Column
        dbc.Col([
            html.H3("Select Model"),
            dcc.Dropdown(
                id="model-dropdown",
                options=model_options,
                value=model_options[0]["value"] if model_options else None,
                clearable=False
            ),
            html.Br(),
            html.H3("Set Desired Returns"),
            html.P("Enter the values for each objective:"),
            *objective_inputs,
            html.Br(),
            html.H3("Set Horizon Weeks"),
            dbc.InputGroup([
                dbc.InputGroupText("Horizon Weeks"),
                dbc.Input(id="horizon-input", type="number", value=12)
            ], className="mb-2"),
            html.Br(),
            html.H3("Set Desired Horizon"),
            dbc.InputGroup([
                dbc.InputGroupText("Desired Horizon Weeks"),
                dbc.Input(id="desired_horizon-input", type="number", value=3)
            ], className="mb-2"),
            html.Br(),
            dbc.Button("Load Parameters", id="load-button", color="secondary", n_clicks=0),
            html.Br(),
            html.Br(),
            dbc.Button("Run Epidemic", id="run-button", color="primary", n_clicks=0),
        ], width=4),

        # Plots Column
        dbc.Col([
            html.H3("Hospitalizations Plot"),
            html.Img(id="hospitalization-plot", style={"width": "50%", "height": "auto"}),
            html.H3("Infections Plot"),
            html.Img(id="infections-plot", style={"width": "50%", "height": "auto"}),
            html.H3("Actions Plot"),
            html.Img(id="actions-plot", style={"width": "50%", "height": "auto"}),
        ], width=8)
    ]),

    # Dataframes at the Bottom
    dbc.Row([
        dbc.Col([
            html.H3("Dataframes"),
            html.H4("Last State Dataframe"),
            dash_table.DataTable(id="last-df"),
            html.Br(),
            html.H4("Other Dataframe"),
            dash_table.DataTable(id="other-df")
        ], width=12)
    ])
], fluid=True)

@app.callback(
    [Output("obj-input-"+str(i), "value") for i in range(len(objectives))] +
    [Output("horizon-input", "value"), Output("desired_horizon-input", "value")],
    [Input("load-button", "n_clicks")]
)
def load_parameters(n_clicks):
    if n_clicks > 0:
        obj_values, horizon, desired_horizon = load_parameters_from_file()
        return obj_values + [horizon, desired_horizon]
    else:
        # No changes if not clicked
        return [no_update]*(len(objectives)+2)

@app.callback(
    [Output("hospitalization-plot", "src"),
     Output("infections-plot", "src"),
     Output("actions-plot", "src"),
     Output("last-df", "data"),
     Output("last-df", "columns"),
     Output("other-df", "data"),
     Output("other-df", "columns")],
    [Input("run-button", "n_clicks")],
    [State("model-dropdown", "value")] + [State(f"obj-input-{i}", "value") for i in range(len(objectives))] +
    [State("horizon-input", "value")] + [State("desired_horizon-input", "value")]
)
def run_simulation(n_clicks, model_file, *args):
    # args contains the objectives values followed by horizon and desired horizon
    if n_clicks == 0:
        # No run yet, return empty
        empty_img = ""
        return empty_img, empty_img, empty_img, [], [], [], []

    vals = args[0:6]
    horizon = args[6]
    desired_horizon = args[7]
    desired_return = np.array(vals, dtype=np.float32)

    # Load model
    model_path = os.path.join(MODEL_DIR, model_file)
    pcn = torch.load(model_path)
    pcn.eval()

    # Create environment
    env = create_covid_env([])
    obs = env.reset()

    all_actions = []
    all_hospitalizations = []
    all_infections = []

    last_state_df = None

    # Run simulation
    for t in range(int(horizon)):
        action = choose_action(pcn, obs, desired_return, desired_horizon, eval=True)
        obs, reward, done, info = env.step(action)
        state_df = env.state_df()
        last_state_df = state_df
        hospitalizations = state_df["I_sev"]
        infections = state_df["I_presym"]
        all_hospitalizations.append(hospitalizations)
        all_infections.append(infections)
        all_actions.append(action)

        print(state_df)

    hospitalizations_plot = plot_compartment(all_hospitalizations, "I_sev")
    infections_plot = plot_compartment(all_infections, "I_presym")
    actions_plot = plot_actions(all_actions)

    # Convert last_state_df to data and columns
    last_df_data = last_state_df.to_dict('records')
    last_df_columns = [{"name": col, "id": col} for col in last_state_df.columns]

    df = pd.read_csv(MODEL_DIR + "pcn_log.csv")
    reference = df.iloc[-1]


    horizon = reference["desired_horizon"]

    other_df = pd.DataFrame({
        "Col_A": [1, 2, 3],
        "Col_B": [4, 5, 6]
    })
    other_df_data = other_df.to_dict('records')
    other_df_columns = [{"name": col, "id": col} for col in other_df.columns]

    return hospitalizations_plot, infections_plot, actions_plot, last_df_data, last_df_columns, other_df_data, other_df_columns

if __name__ == "__main__":
    app.run_server(debug=True)