import os
import numpy as np
import torch
import matplotlib

from scenario.create_fair_env import create_covid_env
from scenario.main_pcn_core import *

matplotlib.use('Agg')
from io import BytesIO
import base64
import pandas as pd
import datetime

from dash import Dash, html, dcc, Input, Output, State, dash_table, no_update
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

import plotly.express as px  # If you prefer Plotly for easy date axes
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Import your environment + model code.
# Example placeholders for demonstration:
# from scenario.main_pcn_core import create_covid_env, choose_action
# ---------------------------------------------------------------------

OBJECTIVES = ["R_ARI", "R_ARH", "R_SB_W", "R_SB_S", "R_SB_L", "SB_SUM"]
MODEL_DIR = "/Users/samvanspringel/Documents/School/VUB/Master 2/Jaar/Thesis/fair_covid/fairRLresults/covid/2024-11-26_11-45-30"
CSV_HOSP_PATH = "/path/to/run_0.csv"
CSV_ICU_PATH  = "/path/to/run_0.csv"

# Global references to environment and model, so each callback can modify them
GLOBAL_ENV = None
GLOBAL_MODEL = None

def list_models():
    """Returns all candidate model files from MODEL_DIR."""
    files = os.listdir(MODEL_DIR)
    model_files = [f for f in files if f.startswith("model_") and f.endswith(".pt")]
    return model_files

def load_model(model_file):
    """Loads the PyTorch model from disk and sets it to eval mode."""
    print("loading", str(model_file))
    model_path = os.path.join(MODEL_DIR, model_file)
    loaded = torch.load(model_path)
    loaded.eval()
    return loaded

def plot_matplotlib(dates, ys_list, labels, title, y_label):
    """
    A helper for plotting multiple lines with matplotlib, using `dates` on the x-axis.
    `ys_list` is a list of lists (one list per line),
    `labels` is a list of line labels,
    `dates` is a list of x-values (datetime or string),
    `title` and `y_label` are strings.
    Returns a base64-encoded PNG string.
    """
    plt.figure(figsize=(6, 4))
    for y, label in zip(ys_list, labels):
        plt.plot(dates, y, label=label, marker='o')

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(y_label)
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()

    img_bytes = BytesIO()
    plt.savefig(img_bytes, format='png', bbox_inches='tight')
    plt.close()
    img_bytes.seek(0)
    encoded = base64.b64encode(img_bytes.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def load_csv_column(csv_path, column_name):
    """
    Example CSV loader for hospital or ICU data.
    Groups by 'dates' and returns a list of lists (one sublist per day).
    Adapt to your real structure if needed.
    """
    df = pd.read_csv(csv_path)
    grouped = df.groupby('dates')
    list_compartment = []
    for _, frame in grouped:
        list_compartment.append(frame[column_name].tolist())
    return list_compartment

# ------------------------------------------------------------------------------
# Dash App
# ------------------------------------------------------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

model_options = [{"label": m, "value": m} for m in list_models()]

# We'll keep the user-specified objectives, horizon, etc. in a Store,
# as well as the simulation progress (dates, data arrays).
# The 'data' field in a dcc.Store must be JSON-serializable, so we only store numbers/lists, not the env object.
app.layout = dbc.Container([
    dcc.Store(id="sim-data", data={
        "all_dates":   [],   # list of string or ISO date for x-axis
        "actions":     [],   # accumulates chosen actions
        "hosp_values": [],   # accumulates total hospital values
        "icu_values":  [],   # accumulates total ICU values
        "horizon":     12,   # default
        "step_count":  0,    # how many steps have been done
    }),

    # Header
    dbc.Row([
        dbc.Col([
            html.H1("PCN Simulation Dashboard"),
            html.P("Press 'Run Epidemic' to advance one time step each press.")
        ])
    ]),

    # Left Column: model & objective input
    dbc.Row([
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
            *[
                dbc.InputGroup([
                    dbc.InputGroupText(obj_name),
                    dbc.Input(id=f"obj-input-{i}", type="number", value=0.0)
                ], className="mb-2")
                for i, obj_name in enumerate(OBJECTIVES)
            ],
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
        ], width=3),

        # Right Column: Action Display + Plots
        dbc.Col([
            html.H3("Current Action Chosen"),
            html.Div(id="current-action-display", style={"fontSize": "1.2em", "fontWeight": "bold"}),
            html.Br(),

            html.H4("Most Recent Data Point"),
            html.Div(id="latest-data-display", style={"fontSize": "1.0em"}),
            html.Br(),

            html.H3("Hospital / ICU / Actions Plots"),
            html.Img(id="hospital-plot", style={"width": "60%", "height": "auto"}),
            html.Br(),
            html.Img(id="icu-plot", style={"width": "60%", "height": "auto"}),
            html.Br(),
            html.Img(id="actions-plot", style={"width": "60%", "height": "auto"}),
            html.Br(),

            # Example CSV comparison plots
            html.H3("Hospital CSV Plot"),
            html.Img(id="hospital-csv-plot", style={"width": "60%", "height": "auto"}),
            html.Br(),

            html.H3("ICU CSV Plot"),
            html.Img(id="icu-csv-plot", style={"width": "60%", "height": "auto"}),
            html.Br(),
        ], width=9),
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

# ------------------------------------------------------------------------------
# Utility to read the final row from your pcn_log.csv
def load_parameters_from_file():
    df = pd.read_csv(os.path.join(MODEL_DIR, "pcn_log.csv"))
    reference = df.iloc[-1]
    # Example mapping
    objectives_desired_returns = [
        reference["return_0_desired"],
        reference["return_1_desired"],
        reference["return_2_desired"],
        reference["return_3_desired"],
        reference["return_4_desired"],
        reference["return_5_desired"]
    ]
    horizon = reference["desired_horizon"]  # or whichever column
    desired_horizon = reference["horizon_distance"]
    return objectives_desired_returns, horizon, desired_horizon


# ------------------------------------------------------------------------------
# CALLBACK: Load parameters from file when "Load Parameters" is pressed
# ------------------------------------------------------------------------------
@app.callback(
    [Output("obj-input-"+str(i), "value") for i in range(len(OBJECTIVES))]
    + [Output("horizon-input", "value"), Output("desired_horizon-input", "value")],
    [Input("load-button", "n_clicks")]
)
def load_parameters(n_clicks):
    if n_clicks > 0:
        obj_values, horizon, desired_horizon = load_parameters_from_file()
        return obj_values + [horizon, desired_horizon]
    else:
        # No changes if not clicked
        return [no_update]*(len(OBJECTIVES)+2)


# ------------------------------------------------------------------------------
# CALLBACK: Run 1 Timestep of the Epidemic
# ------------------------------------------------------------------------------
@app.callback(
    [
        Output("sim-data", "data"),
        Output("hospital-plot", "src"),
        Output("icu-plot", "src"),
        Output("actions-plot", "src"),
        Output("hospital-csv-plot", "src"),
        Output("icu-csv-plot", "src"),
        Output("current-action-display", "children"),
        Output("latest-data-display", "children"),
        Output("last-df", "data"),
        Output("last-df", "columns"),
        Output("other-df", "data"),
        Output("other-df", "columns"),
    ],
    [Input("run-button", "n_clicks")],
    [
        State("model-dropdown", "value"),
        State("sim-data", "data")
    ]
    + [State(f"obj-input-{i}", "value") for i in range(len(OBJECTIVES))]
    + [State("horizon-input", "value")]
    + [State("desired_horizon-input", "value")]
)
def run_simulation(
    n_clicks,
    model_file,
    sim_data,        # the dictionary we stored
    *args
):
    """
    Each time the user presses the "Run Epidemic" button, we advance exactly one step.
    The 'sim_data' dictionary accumulates the results across presses.
    """
    # If no clicks, do nothing
    if n_clicks < 1:
        raise PreventUpdate

    # Unpack objective/horizon from *args
    # We have 6 objectives in the snippet, so that is indexes 0..5
    # Then horizon is index 6, desired_horizon is index 7
    # Adjust if the dimension of OBJECTIVES changes
    num_objs = len(OBJECTIVES)
    desired_return_vals = args[0:num_objs]
    horizon_weeks        = args[num_objs]
    desired_horizon_weeks= args[num_objs+1]

    # Convert to the type you need, e.g. float32
    desired_return = np.array(desired_return_vals, dtype=np.float32)

    global GLOBAL_ENV
    global GLOBAL_MODEL

    # 1) Initialize environment & model if needed
    if GLOBAL_MODEL is None or (model_file is not None and model_file != ""):
        GLOBAL_MODEL = load_model(model_file)

    # If environment doesn't exist, create from scratch
    if GLOBAL_ENV is None:
        # create_covid_env can accept additional parameters as needed
        GLOBAL_ENV = create_covid_env([])
        GLOBAL_ENV.reset()

    # We store "sim_data['horizon']" on the first run in the store or from user input
    # so ensure they are in sync.
    # If user changed horizon on the inputs, we can align it with the stored horizon:
    if n_clicks == 1:
        # On the first button click, set the store's horizon from the input
        sim_data["horizon"] = horizon_weeks

    # 2) If horizon is <= 0, we can skip stepping or forcibly do no_update
    if sim_data["horizon"] <= 0:
        return [sim_data, no_update, no_update, no_update, no_update, no_update,
                "Horizon exhausted", "No more steps possible", [], [], [], []]

    # 3) Advance 1 step
    #    choose_action(...) is your code to pick an action from the environment obs.
    state = GLOBAL_ENV.current_state_n  # or however you get the current observation
    action = GLOBAL_ENV.current_action
    events = GLOBAL_ENV.current_events_n
    budget = np.ones(3)*4
    print("Budget", GLOBAL_MODEL.sb_emb is None)

    action = choose_action(GLOBAL_MODEL, (budget, state, events, action), desired_return, desired_horizon_weeks, eval=True)

    # Step environment
    obs, reward, done, info = GLOBAL_ENV.step(action)
    state_df = GLOBAL_ENV.state_df()[0]

    # 4) Extract data we want from the environment’s state for plotting:
    #    e.g. "I_hosp_new" and "I_icu_new".
    hospitalizations = state_df["I_hosp_new"].sum()
    icu_admissions   = state_df["I_icu_new"].sum()

    # 5) Append new data to the sim_data lists
    sim_data["actions"].append(action.tolist() if isinstance(action, np.ndarray) else list(action))
    sim_data["hosp_values"].append(float(hospitalizations))
    sim_data["icu_values"].append(float(icu_admissions))

    # For the date axis, let's generate the next date from the step_count or from a start date
    step_count = sim_data["step_count"]
    # We can define a start date. For example:
    start_date = datetime.date(2025, 1, 1)
    # Next date is:
    current_date = start_date + datetime.timedelta(weeks=step_count)
    sim_data["all_dates"].append(current_date.isoformat())

    # Increase step count
    sim_data["step_count"] += 1

    # Decrement horizon
    sim_data["horizon"] -= 1

    # 6) Build the updated plots
    #    We use the sim_data arrays to plot everything so it accumulates.
    dates_for_plot = sim_data["all_dates"]

    # Plot hospital
    hosp_plot = plot_matplotlib(
        dates_for_plot,
        [sim_data["hosp_values"]],  # single line
        ["Hospitalizations"],
        "Hospitalizations Over Time",
        "I_hosp_new",
    )

    # Plot ICU
    icu_plot = plot_matplotlib(
        dates_for_plot,
        [sim_data["icu_values"]],  # single line
        ["ICU Admissions"],
        "ICU Admissions Over Time",
        "I_icu_new",
    )

    # Plot actions (we have 3 continuous dimensions)
    # We'll transpose the list of lists
    all_actions = list(zip(*sim_data["actions"]))  # 3 sublists
    actions_plot = plot_matplotlib(
        dates_for_plot,
        all_actions,
        ["Reduction Work", "Reduction School", "Reduction Leisure"],
        "Continuous Reductions Over Time",
        "Action Values",
    )

    # We can reload the CSV data each time (or once globally).
    # Then plot them for comparison. For brevity we do it each time:
    csv_hosp_data = load_csv_column(CSV_HOSP_PATH, "i_hosp_new")
    csv_icu_data  = load_csv_column(CSV_ICU_PATH,  "i_icu_new")

    # Suppose each "csv_hosp_data" is a list-of-lists. If you want to sum up each day:
    # We'll also track a made-up date for them. E.g. same start date:
    # This is just to give them an x-axis. Adjust as needed.
    csv_dates = []
    # If the CSV has N days, do something like:
    for i in range(len(csv_hosp_data)):
        day_date = start_date + datetime.timedelta(weeks=i)
        csv_dates.append(day_date.isoformat())

    # For demonstration, sum each sublist:
    daily_hosp_sums = [sum(x) for x in csv_hosp_data]
    daily_icu_sums  = [sum(x) for x in csv_icu_data]

    hosp_csv_plot = plot_matplotlib(
        csv_dates,
        [daily_hosp_sums],
        ["Hospital CSV Summed"],
        "Hospital (CSV) Over Time",
        "I_hosp_new (CSV)",
    )
    icu_csv_plot = plot_matplotlib(
        csv_dates,
        [daily_icu_sums],
        ["ICU CSV Summed"],
        "ICU (CSV) Over Time",
        "I_icu_new (CSV)",
    )

    # 7) Prepare DataTables
    #    "last_state_df" is from the environment
    last_df_data = state_df.to_dict("records")
    last_df_columns = [{"name": c, "id": c} for c in state_df.columns]

    # Example "other_df"
    df = pd.read_csv(os.path.join(MODEL_DIR, "pcn_log.csv"))
    reference = df.iloc[-1]
    other_df = pd.DataFrame({
        "Col_A": [1, 2, 3],
        "Col_B": [4, 5, 6]
    })
    other_df_data = other_df.to_dict('records')
    other_df_columns = [{"name": col, "id": col} for col in other_df.columns]

    # 8) Build a short text display for the current action + the new data point
    action_str = f"Work={action[0]:.3f}, School={action[1]:.3f}, Leisure={action[2]:.3f}"
    new_data_str = (
        f"Date: {current_date.isoformat()} | "
        f"Hospitalizations: {hospitalizations:.2f} | "
        f"ICU: {icu_admissions:.2f}"
    )
    # Also print to server console, if desired:
    print("[New Step] " + new_data_str)

    return [
        sim_data,
        hosp_plot,
        icu_plot,
        actions_plot,
        hosp_csv_plot,
        icu_csv_plot,
        action_str,
        new_data_str,
        last_df_data,
        last_df_columns,
        other_df_data,
        other_df_columns
    ]


if __name__ == "__main__":
    app.run_server(debug=True)