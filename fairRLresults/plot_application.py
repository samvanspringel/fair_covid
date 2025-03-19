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

from dash import Dash, html, dcc, Input, Output, State, no_update, callback_context
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

import plotly.express as px  # optional
import matplotlib.pyplot as plt

########################################################################
# Paths & Global Vars
########################################################################

# CSV scenario uses the same CSV for demonstration
CSV_HOSP_PATH = "/Users/samvanspringel/Documents/School/VUB/Master 2/Jaar/Thesis/fair_covid/fairRLresults/run_0.csv"
CSV_ICU_PATH  = "/Users/samvanspringel/Documents/School/VUB/Master 2/Jaar/Thesis/fair_covid/fairRLresults/run_0.csv"

# For the middle scenario (model #1)
MODEL_DIR_MID = "/Users/samvanspringel/Documents/School/VUB/Master 2/Jaar/Thesis/fair_covid/fairRLresults/cluster/'covid/steps_300000/objectives_R_ARH:R_SB_W:R_SB_S:R_SB_L_SBS:ABFTA/distance_metric_none/window_100/seed_0/'/2025-03-13_15-18-42"

# For the right scenario (model #2)
MODEL_DIR_RIGHT = "/Users/samvanspringel/Documents/School/VUB/Master 2/Jaar/Thesis/fair_covid/fairRLresults/cluster/'covid/steps_300000/objectives_R_ARI:R_SB_W:R_SB_S:R_SB_L_SBS:ABFTA/distance_metric_none/window_100/seed_0/'/2025-03-13_15-19-10"

OBJECTIVES = ["R_ARI", "R_ARH", "R_SB_W", "R_SB_S", "R_SB_L", "SB_SUM"]

# We'll have separate "global" references for each environment + model
GLOBAL_MODEL_MID = None
GLOBAL_ENV_MID = None

GLOBAL_MODEL_RIGHT = None
GLOBAL_ENV_RIGHT = None

########################################################################
# Utilities
########################################################################

def list_models(dir_path):
    """Return all candidate *.pt model files from a directory, matching 'model_10...'.pt."""
    files = os.listdir(dir_path)
    model_files = [f for f in files if f.startswith("model_10") and f.endswith(".pt")]
    return model_files

def reorder_models(models):
    """Ensure 'model10.pt' is first in the list if it exists."""
    if "model10.pt" in models:
        models.remove("model10.pt")
        models.insert(0, "model10.pt")
    return models

def load_model(model_file, dir_path):
    """Load the PyTorch model from disk and set it to eval mode."""
    print("loading model:", model_file, "from:", dir_path)
    model_path = os.path.join(dir_path, model_file)
    loaded = torch.load(model_path)
    loaded.eval()
    return loaded

def plot_matplotlib(dates, ys_list, labels, title):
    """
    Plot multiple lines on the same figure, storing as base64-encoded PNG.
    `dates`: x-axis list
    `ys_list`: list of lists, one per line
    `labels`: list of line labels
    """
    plt.figure(figsize=(6, 4))
    for y, lbl in zip(ys_list, labels):
        plt.plot(dates, y, label=lbl, marker='o')

    plt.title(title)
    plt.xlabel("Date")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

def load_csv_column(csv_path, column_name):
    """Load CSV, group by 'dates', return list-of-lists for that column."""
    df = pd.read_csv(csv_path)
    grouped = df.groupby('dates')
    result = []
    for _, frame in grouped:
        result.append(frame[column_name].tolist())
    return result

########################################################################
# Dash App + Layout
########################################################################

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Reorder model lists so model10.pt is first if present
models_mid = reorder_models(list_models(MODEL_DIR_MID))
model_options_mid = [{"label": m, "value": m} for m in models_mid]

models_right = reorder_models(list_models(MODEL_DIR_RIGHT))
model_options_right = [{"label": m, "value": m} for m in models_right]

# Column 1 store (CSV scenario):
CSV_STORE_DEFAULT = {
    "index": 0,
    "dates": [],
    "hosp":  [],
    "icu":   [],
}

# Column 2 store (Middle scenario):
MID_STORE_DEFAULT = {
    "all_dates":   [],
    "hosp_values": [],
    "icu_values":  [],
    "actions":     [],
    "horizon":     32,
    "step_count":  0,
}

# Column 3 store (Right scenario):
RIGHT_STORE_DEFAULT = {
    "all_dates":   [],
    "hosp_values": [],
    "icu_values":  [],
    "actions":     [],
    "horizon":     32,
    "step_count":  0,
}

# We'll load objectives from run_0.csv, which has columns "o_0..o_5"
RUN_OBJECTIVES_PATH = CSV_HOSP_PATH

def load_obj_columns(csv_path):
    """Reads run_0.csv, storing each o_i column in a list, returning a dict."""
    df = pd.read_csv(csv_path)
    d = {}
    for i in range(6):
        col = f"o_{i}"
        if col in df.columns:
            d[col] = df[col].tolist()
        else:
            d[col] = []
    return d

def make_objstore_default():
    return {
        "index": 0,
        "o_0": [],
        "o_1": [],
        "o_2": [],
        "o_3": [],
        "o_4": [],
        "o_5": []
    }

# App Layout
app.layout = dbc.Container([
    dbc.Row([
        # Left Column: CSV scenario
        dbc.Col([
            html.H2("CSV Scenario"),
            dcc.Store(id="csv-data", data=CSV_STORE_DEFAULT),
            dbc.Button("Next CSV Step", id="csv-run-button", n_clicks=0),
            dbc.Button("Reset CSV", id="csv-reset-button", n_clicks=0, color="danger", style={"marginLeft": "10px"}),
            html.Br(), html.Br(),

            html.Div("Action:"),
            html.Div(id="csv-action-value"),
            html.Div("Hosp:"),
            html.Div(id="csv-hosp-value"),
            html.Div("ICU:"),
            html.Div(id="csv-icu-value"),

            html.Img(id="csv-main-plot", style={"width": "80%", "display": "block", "margin": "auto"}),
            html.Img(id="csv-small-plot", style={"width": "60%", "display": "block", "margin": "auto"}),
        ], width=4),

        # Middle Column: ARH environment
        dbc.Col([
            html.H2("ARH Model"),
            dcc.Store(id="sim-data-middle", data=MID_STORE_DEFAULT),
            dcc.Store(id="mid-objstore", data=make_objstore_default()),

            html.Div("Select Model (ARH)"),
            dcc.Dropdown(
                id="model-dropdown-middle",
                options=model_options_mid,
                value=model_options_mid[0]["value"] if model_options_mid else None,
                clearable=False
            ),
            html.Br(),

            dbc.Button("Load Next Obj (ARH)", id="mid-load-obj-button", n_clicks=0, color="secondary"),
            html.Br(), html.Br(),

            # Objectives in one row, bigger fields, labeled
            html.Div([
                "Objectives:",
                dbc.Row([
                    dbc.Col([
                        html.Div(OBJECTIVES[i]),
                        dbc.Input(
                            id=f"mid-obj-input-{i}",
                            type="number",
                            value=0.0,
                            style={"width": "100px"}  # bigger width
                        )
                    ], width="auto") for i in range(len(OBJECTIVES))
                ], justify="start", align="center"),
            ]),
            html.Br(),

            html.Div("Desired Horizon:"),
            dbc.Input(id="mid-desired-horizon", type="number", value=2, style={"width": "80px"}),
            html.Br(),

            dbc.Button("Run", id="mid-run-button", color="primary", n_clicks=0),
            dbc.Button("Reset", id="mid-reset-button", color="danger", n_clicks=0, style={"marginLeft": "10px"}),
            html.Br(), html.Br(),

            html.Div("Action (Mid):"),
            html.Div(id="mid-action-display"),
            html.Div("Hosp (Mid):"),
            html.Div(id="mid-hosp-display"),
            html.Div("ICU (Mid):"),
            html.Div(id="mid-icu-display"),
            html.Img(id="mid-main-plot", style={"width": "80%", "display": "block", "margin": "auto"}),
        ], width=4),

        # Right Column: ARI environment
        dbc.Col([
            html.H2("ARI Model"),
            dcc.Store(id="sim-data-right", data=RIGHT_STORE_DEFAULT),
            dcc.Store(id="right-objstore", data=make_objstore_default()),

            html.Div("Select Model (ARI)"),
            dcc.Dropdown(
                id="model-dropdown-right",
                options=model_options_right,
                value=model_options_right[0]["value"] if model_options_right else None,
                clearable=False
            ),
            html.Br(),

            dbc.Button("Load Next Obj (ARI)", id="right-load-obj-button", n_clicks=0, color="secondary"),
            html.Br(), html.Br(),

            # Objectives in one row, bigger fields, labeled
            html.Div([
                "Objectives:",
                dbc.Row([
                    dbc.Col([
                        html.Div(OBJECTIVES[i]),
                        dbc.Input(
                            id=f"right-obj-input-{i}",
                            type="number",
                            value=0.0,
                            style={"width": "100px"}  # bigger width
                        )
                    ], width="auto") for i in range(len(OBJECTIVES))
                ], justify="start", align="center"),
            ]),
            html.Br(),

            html.Div("Desired Horizon:"),
            dbc.Input(id="right-desired-horizon", type="number", value=2, style={"width": "80px"}),
            html.Br(),

            dbc.Button("Run", id="right-run-button", color="primary", n_clicks=0),
            dbc.Button("Reset", id="right-reset-button", color="danger", n_clicks=0, style={"marginLeft": "10px"}),
            html.Br(), html.Br(),

            html.Div("Action (Right):"),
            html.Div(id="right-action-display"),
            html.Div("Hosp (Right):"),
            html.Div(id="right-hosp-display"),
            html.Div("ICU (Right):"),
            html.Div(id="right-icu-display"),
            html.Img(id="right-main-plot", style={"width": "80%", "display": "block", "margin": "auto"}),
        ], width=4),
    ])
], fluid=True)

########################################################################
# Left Column: CSV scenario (Run + Reset)
########################################################################

@app.callback(
    [
        Output("csv-data", "data"),
        Output("csv-action-value", "children"),
        Output("csv-hosp-value", "children"),
        Output("csv-icu-value", "children"),
        Output("csv-main-plot", "src"),
        Output("csv-small-plot", "src"),
    ],
    [
        Input("csv-run-button", "n_clicks"),
        Input("csv-reset-button", "n_clicks")
    ],
    [State("csv-data", "data")]
)
def update_csv_data(run_clicks, reset_clicks, store_data):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if triggered_id == "csv-reset-button":
        return [
            CSV_STORE_DEFAULT,
            "N/A",
            "0",
            "0",
            no_update,
            no_update
        ]

    if run_clicks < 1:
        raise PreventUpdate

    if not store_data["dates"]:
        csv_hosp_data = load_csv_column(CSV_HOSP_PATH, "i_hosp_new")
        csv_icu_data  = load_csv_column(CSV_ICU_PATH,  "i_icu_new")

        daily_hosp_sums = [sum(x) for x in csv_hosp_data]
        daily_icu_sums  = [sum(x) for x in csv_icu_data]

        start_date = datetime.date(2020, 3, 1)
        dates = []
        for i in range(len(daily_hosp_sums)):
            day_date = start_date + datetime.timedelta(weeks=i)
            dates.append(day_date.isoformat())

        store_data["hosp"] = daily_hosp_sums
        store_data["icu"]  = daily_icu_sums
        store_data["dates"] = dates
        store_data["index"] = 0

    idx = store_data["index"]
    all_dates = store_data["dates"]
    hosp_vals = store_data["hosp"]
    icu_vals  = store_data["icu"]

    if idx >= len(all_dates):
        raise PreventUpdate

    current_hosp = hosp_vals[idx]
    current_icu  = icu_vals[idx]
    store_data["index"] += 1

    x_plot = all_dates[: store_data["index"]]
    hosp_plot_vals = hosp_vals[: store_data["index"]]
    icu_plot_vals  = icu_vals[: store_data["index"]]

    main_plot = plot_matplotlib(
        x_plot,
        [hosp_plot_vals, icu_plot_vals],
        ["Hosp", "ICU"],
        "CSV Data (Hosp + ICU)",
    )
    small_plot = plot_matplotlib(
        all_dates,
        [hosp_vals, icu_vals],
        ["Hosp", "ICU"],
        "All CSV Data",
    )

    return [
        store_data,
        "N/A",
        f"{current_hosp:.0f}",
        f"{current_icu:.0f}",
        main_plot,
        small_plot
    ]

########################################################################
# Middle Column: Load Next Obj + Run / Reset
########################################################################

@app.callback(
    [
        Output("mid-objstore", "data"),
        Output("mid-obj-input-0", "value"),
        Output("mid-obj-input-1", "value"),
        Output("mid-obj-input-2", "value"),
        Output("mid-obj-input-3", "value"),
        Output("mid-obj-input-4", "value"),
        Output("mid-obj-input-5", "value"),
    ],
    [Input("mid-load-obj-button", "n_clicks")],
    [State("mid-objstore", "data")]
)
def load_next_obj_mid(n_clicks, store):
    if n_clicks < 1:
        raise PreventUpdate

    if not store["o_0"]:
        all_obj = load_obj_columns(RUN_OBJECTIVES_PATH)
        for i in range(6):
            store[f"o_{i}"] = all_obj[f"o_{i}"]

    idx = store["index"]
    if idx >= len(store["o_0"]):
        raise PreventUpdate

    val0 = store["o_0"][idx]
    val1 = store["o_1"][idx]
    val2 = store["o_2"][idx]
    val3 = store["o_3"][idx]
    val4 = store["o_4"][idx]
    val5 = store["o_5"][idx]

    store["index"] += 1

    return [
        store,
        val0, val1, val2, val3, val4, val5
    ]

@app.callback(
    [
        Output("sim-data-middle", "data"),
        Output("mid-action-display", "children"),
        Output("mid-hosp-display", "children"),
        Output("mid-icu-display", "children"),
        Output("mid-main-plot", "src"),
    ],
    [
        Input("mid-run-button", "n_clicks"),
        Input("mid-reset-button", "n_clicks"),
    ],
    [
        State("model-dropdown-middle", "value"),
        State("sim-data-middle", "data")
    ]
    + [State(f"mid-obj-input-{i}", "value") for i in range(len(OBJECTIVES))]
    + [State("mid-desired-horizon", "value")]
)
def combined_callback_middle(run_clicks, reset_clicks,
                             model_file, sim_data,
                             *args):
    global GLOBAL_MODEL_MID, GLOBAL_ENV_MID

    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if triggered_id == "mid-reset-button":
        GLOBAL_ENV_MID = None
        GLOBAL_MODEL_MID = None
        return (
            {
                "all_dates":   [],
                "hosp_values": [],
                "icu_values":  [],
                "actions":     [],
                "horizon":     32,
                "step_count":  0,
            },
            "Reset",
            "0",
            "0",
            no_update
        )

    if (run_clicks or 0) < 1:
        raise PreventUpdate

    num_objs = len(OBJECTIVES)
    desired_return_vals = args[0:num_objs]
    desired_horizon_weeks = args[num_objs]
    desired_return = np.array(desired_return_vals, dtype=np.float32)

    if GLOBAL_MODEL_MID is None:
        GLOBAL_MODEL_MID = load_model(model_file, MODEL_DIR_MID)

    if GLOBAL_ENV_MID is None:
        GLOBAL_ENV_MID = create_covid_env([])
        GLOBAL_ENV_MID.reset()

    if sim_data["horizon"] <= 0:
        return [
            sim_data,
            "Horizon exhausted",
            "0", "0",
            no_update
        ]

    state = GLOBAL_ENV_MID.current_state_n
    action = GLOBAL_ENV_MID.current_action
    events = GLOBAL_ENV_MID.current_events_n
    budget = np.ones(3)*4

    action = choose_action(
        GLOBAL_MODEL_MID, (budget, state, events, action),
        desired_return, desired_horizon_weeks, eval=True
    )
    obs, reward, done, info = GLOBAL_ENV_MID.step(action)
    state_df = GLOBAL_ENV_MID.state_df()[0]

    hospitalizations = float(state_df["I_hosp_new"].sum())
    icu_admissions   = float(state_df["I_icu_new"].sum())

    sim_data["actions"].append(list(action))
    sim_data["hosp_values"].append(hospitalizations)
    sim_data["icu_values"].append(icu_admissions)

    step_count = sim_data["step_count"]
    start_date = datetime.date(2020, 3, 1)
    current_date = start_date + datetime.timedelta(weeks=step_count)
    sim_data["all_dates"].append(current_date.isoformat())

    sim_data["step_count"] += 1
    sim_data["horizon"] -= 1

    main_plot = plot_matplotlib(
        sim_data["all_dates"],
        [sim_data["hosp_values"], sim_data["icu_values"]],
        ["Hosp", "ICU"],
        "Simulated (Hosp + ICU)",
    )

    action_str = f"{action}"
    hosp_str   = f"{hospitalizations:.0f}"
    icu_str    = f"{icu_admissions:.0f}"

    return [
        sim_data,
        action_str,
        hosp_str,
        icu_str,
        main_plot
    ]

########################################################################
# Right Column: Load Next Obj + Run / Reset
########################################################################

@app.callback(
    [
        Output("right-objstore", "data"),
        Output("right-obj-input-0", "value"),
        Output("right-obj-input-1", "value"),
        Output("right-obj-input-2", "value"),
        Output("right-obj-input-3", "value"),
        Output("right-obj-input-4", "value"),
        Output("right-obj-input-5", "value"),
    ],
    [Input("right-load-obj-button", "n_clicks")],
    [State("right-objstore", "data")]
)
def load_next_obj_right(n_clicks, store):
    if n_clicks < 1:
        raise PreventUpdate

    if not store["o_0"]:
        all_obj = load_obj_columns(RUN_OBJECTIVES_PATH)
        for i in range(6):
            store[f"o_{i}"] = all_obj[f"o_{i}"]

    idx = store["index"]
    if idx >= len(store["o_0"]):
        raise PreventUpdate

    val0 = store["o_0"][idx]
    val1 = store["o_1"][idx]
    val2 = store["o_2"][idx]
    val3 = store["o_3"][idx]
    val4 = store["o_4"][idx]
    val5 = store["o_5"][idx]

    store["index"] += 1

    return [
        store,
        val0, val1, val2, val3, val4, val5
    ]

@app.callback(
    [
        Output("sim-data-right", "data"),
        Output("right-action-display", "children"),
        Output("right-hosp-display", "children"),
        Output("right-icu-display", "children"),
        Output("right-main-plot", "src"),
    ],
    [
        Input("right-run-button", "n_clicks"),
        Input("right-reset-button", "n_clicks"),
    ],
    [
        State("model-dropdown-right", "value"),
        State("sim-data-right", "data")
    ]
    + [State(f"right-obj-input-{i}", "value") for i in range(len(OBJECTIVES))]
    + [State("right-desired-horizon", "value")]
)
def combined_callback_right(run_clicks, reset_clicks,
                            model_file, sim_data,
                            *args):
    global GLOBAL_MODEL_RIGHT, GLOBAL_ENV_RIGHT

    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if triggered_id == "right-reset-button":
        GLOBAL_ENV_RIGHT = None
        GLOBAL_MODEL_RIGHT = None
        return (
            {
                "all_dates":   [],
                "hosp_values": [],
                "icu_values":  [],
                "actions":     [],
                "horizon":     32,
                "step_count":  0,
            },
            "Reset",
            "0",
            "0",
            no_update
        )

    if (run_clicks or 0) < 1:
        raise PreventUpdate

    num_objs = len(OBJECTIVES)
    desired_return_vals = args[0:num_objs]
    desired_horizon_weeks = args[num_objs]
    desired_return = np.array(desired_return_vals, dtype=np.float32)

    if GLOBAL_MODEL_RIGHT is None:
        GLOBAL_MODEL_RIGHT = load_model(model_file, MODEL_DIR_RIGHT)

    if GLOBAL_ENV_RIGHT is None:
        GLOBAL_ENV_RIGHT = create_covid_env([])
        GLOBAL_ENV_RIGHT.reset()

    if sim_data["horizon"] <= 0:
        return [
            sim_data,
            "Horizon exhausted",
            "0", "0",
            no_update
        ]

    state = GLOBAL_ENV_RIGHT.current_state_n
    action = GLOBAL_ENV_RIGHT.current_action
    events = GLOBAL_ENV_RIGHT.current_events_n
    budget = np.ones(3)*4

    action = choose_action(
        GLOBAL_MODEL_RIGHT, (budget, state, events, action),
        desired_return, desired_horizon_weeks, eval=True
    )
    obs, reward, done, info = GLOBAL_ENV_RIGHT.step(action)
    state_df = GLOBAL_ENV_RIGHT.state_df()[0]

    hospitalizations = float(state_df["I_hosp_new"].sum())
    icu_admissions   = float(state_df["I_icu_new"].sum())

    sim_data["actions"].append(list(action))
    sim_data["hosp_values"].append(hospitalizations)
    sim_data["icu_values"].append(icu_admissions)

    step_count = sim_data["step_count"]
    start_date = datetime.date(2020, 3, 1)
    current_date = start_date + datetime.timedelta(weeks=step_count)
    sim_data["all_dates"].append(current_date.isoformat())

    sim_data["step_count"] += 1
    sim_data["horizon"] -= 1

    main_plot = plot_matplotlib(
        sim_data["all_dates"],
        [sim_data["hosp_values"], sim_data["icu_values"]],
        ["Hosp", "ICU"],
        "Simulated (Hosp + ICU, Right)",
    )

    action_str = f"{action}"
    hosp_str   = f"{hospitalizations:.0f}"
    icu_str    = f"{icu_admissions:.0f}"

    return [
        sim_data,
        action_str,
        hosp_str,
        icu_str,
        main_plot
    ]


if __name__ == "__main__":
    app.run_server(debug=True)
