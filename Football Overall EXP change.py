# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 12:36:41 2025

@author: Sukhdeep.Sangha
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO, StringIO
from matplotlib.backends.backend_pdf import PdfPages
import requests

# --- Load Data ---
@st.cache_data
def load_data():
    url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTFcAq9JNJsfPhiPe3sdaEipD1jofgrCNozbcld55JzDguXCxVpFbM0KwKd5txhLh4FLlBvFy43WqAX/pub?output=csv"
    response = requests.get(url)
    response.raise_for_status()
    data = StringIO(response.text)
    df = pd.read_csv(data)
    df['EVENT_START_TIMESTAMP'] = pd.to_datetime(df['EVENT_START_TIMESTAMP'], errors='coerce', dayfirst=True)
    return df.dropna(subset=['EVENT_START_TIMESTAMP'])

df = load_data()

# --- Sidebar Filters ---
st.sidebar.header("Filters")

exp_types = st.sidebar.multiselect("Select Expectancy Types (up to 6)", [
    'Favourite Goals', 'Underdog Goals', 'Total Goals',
    'Favourite Corners', 'Underdog Corners', 'Total Corners'
], default=['Favourite Goals'])

min_date = df['EVENT_START_TIMESTAMP'].min().date()
max_date = df['EVENT_START_TIMESTAMP'].max().date()
start_date, end_date = st.sidebar.date_input("Select Date Range", [min_date, max_date])
df = df[(df['EVENT_START_TIMESTAMP'].dt.date >= start_date) & (df['EVENT_START_TIMESTAMP'].dt.date <= end_date)]

fav_options = ['Strong Favourite', 'Medium Favourite', 'Slight Favourite']
scoreline_options = ['Favourite Winning', 'Scores Level', 'Underdog Winning']

fav_filter = st.sidebar.multiselect("Goal Favouritism Level", fav_options, default=fav_options)
scoreline_filter = st.sidebar.multiselect("Goal Scoreline Filter", scoreline_options, default=scoreline_options)

st.markdown("*Favourites are determined using Goal Expectancy at minute 0")

# --- Functions ---

def classify_favouritism(row):
    diff = abs(row['GOAL_EXP_HOME'] - row['GOAL_EXP_AWAY'])
    if diff > 1:
        return 'Strong Favourite'
    elif diff > 0.5:
        return 'Medium Favourite'
    else:
        return 'Slight Favourite'

def classify_scoreline_simple(row):
    fav = 'Home' if row['GOAL_EXP_HOME'] > row['GOAL_EXP_AWAY'] else 'Away'
    score_diff = row['GOALS_HOME'] - row['GOALS_AWAY']
    fav_diff = score_diff if fav == 'Home' else -score_diff
    if fav_diff > 0:
        return "Favourite Winning"
    elif fav_diff == 0:
        return "Scores Level"
    else:
        return "Underdog Winning"

def compute_exp_by_role(df, role='Favourite', target='Goals'):
    col_map = {
        'Goals': ('GOAL_EXP_HOME', 'GOAL_EXP_AWAY'),
        'Corners': ('CORNERS_EXP_HOME', 'CORNERS_EXP_AWAY')
    }

    home_col, away_col = col_map[target]
    change_data = []

    for event_id, group in df.groupby('SRC_EVENT_ID'):
        group = group.sort_values('MINUTES')
        base_row = group[group['MINUTES'] == 0].iloc[0]

        home_exp_0 = base_row['GOAL_EXP_HOME']
        away_exp_0 = base_row['GOAL_EXP_AWAY']
        home_is_fav = home_exp_0 > away_exp_0
        if home_exp_0 == away_exp_0:
            continue  # skip events with no clear favourite

        base_val = base_row[home_col if (role == 'Favourite' and home_is_fav) or (role == 'Underdog' and not home_is_fav) else away_col]
        prev_val = base_val

        for _, row in group.iterrows():
            minute = row['MINUTES']
            team_val = row[home_col if (role == 'Favourite' and home_is_fav) or (role == 'Underdog' and not home_is_fav) else away_col]
            if team_val != prev_val:
                overall_change = team_val - base_val
                prev_val = team_val

                change_data.append({
                    'MINUTES': minute,
                    'Change': overall_change,
                    'GOAL_EXP_HOME': row['GOAL_EXP_HOME'],
                    'GOAL_EXP_AWAY': row['GOAL_EXP_AWAY'],
                    'GOALS_HOME': row['GOALS_HOME'],
                    'GOALS_AWAY': row['GOALS_AWAY'],
                    'EVENT_START_TIMESTAMP': row['EVENT_START_TIMESTAMP'],
                    'SRC_EVENT_ID': event_id
                })
    return pd.DataFrame(change_data)

# --- Graph Generation ---
plots = []
cols = st.columns(2) if len(exp_types) > 1 else [st]

for i, exp_type in enumerate(exp_types[:6]):
    role = 'Favourite' if 'Favourite' in exp_type else 'Underdog'
    market = 'Goals' if 'Goals' in exp_type else 'Corners'
    df_changes = compute_exp_by_role(df, role=role, target=market)

    df_changes['Favouritism'] = df_changes.apply(classify_favouritism, axis=1)
    df_changes['Scoreline'] = df_changes.apply(classify_scoreline_simple, axis=1)
    df_changes = df_changes[df_changes['Favouritism'].isin(fav_filter) & df_changes['Scoreline'].isin(scoreline_filter)]

    df_changes['Time Band'] = pd.cut(
        df_changes['MINUTES'],
        bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 1000],
        right=False,
        labels=[f"{i}-{i+5}" for i in range(0, 90, 5)]
    )

    avg_change = df_changes.groupby('Time Band')['Change'].mean()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(avg_change.index, avg_change.values, marker='o', color='black')
    ax.set_title(f"{exp_type} Expectancy Change")
    ax.set_xlabel("Time Band (Minutes)")
    ax.set_ylabel("Avg Change")
    ax.grid(True)
    fig.tight_layout()

    plots.append(fig)
    cols[i % 2].pyplot(fig)

# --- Export to PDF Button ---
def export_all_to_pdf(figures):
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches='tight')
    buffer.seek(0)
    return buffer

if plots:
    st.download_button(
        label="Download All Charts as PDF",
        data=export_all_to_pdf(plots),
        file_name="expectancy_graphs.pdf",
        mime="application/pdf"
    )
