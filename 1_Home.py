import pandas as pd
import numpy as np
import os
import streamlit as st
import dtale
import re
import plotly.express as px
import statsmodels
from geopandas import GeoDataFrame

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_hc0, cov_hc1
from statsmodels.formula.api import ols
import statsmodels.api as sm
from patsy.contrasts import Treatment
import folium
from shapely import wkt
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px
from shapely.geometry import Polygon, MultiPolygon

df = pd.read_csv('data/cleaned_FPS2.csv')
mis_type = pd.read_csv('data/missiles_and_uav.csv')
launched_geom = pd.read_csv('data/launched_geom2.csv')
probit_df = pd.read_csv('data/probit_model2.csv')

# Data cleaning section
df1 = df[df["launch_place"].notna()]

df1['time_start'] = pd.to_datetime(df1['time_start'], format='mixed')
df1['time_end'] = pd.to_datetime(df1['time_end'], format='mixed')
df1['duration'] = df1['time_end'] - df1['time_start']

dist_columns = [col for col in df1.columns if col.startswith("dist_aff_")]
df1['num_non_na_dist'] = df1[dist_columns].notna().sum(axis=1)

# split by if affected_region1 has NAs and those that aren't
cols_to_fill = ['launched','destroyed', 'not_reach_goal','num_hit_location']

# affected_region1 is na portion
#print(df1.columns.tolist())
df_empty_aff = df1[df1['affected_region1'].isna()]
df_empty_aff = df_empty_aff.copy()
df_empty_aff[cols_to_fill] = df_empty_aff[cols_to_fill].fillna(0)
df_empty_aff['hit'] = (
    (df_empty_aff['launched'] - df_empty_aff['destroyed'] - df_empty_aff['not_reach_goal']) / 
    df_empty_aff['launched']
)
df_empty_aff['hit'] = df_empty_aff['hit'].replace([np.inf, -np.inf], np.nan)
df_empty_aff['neutralized'] = (
    (df_empty_aff['destroyed'] + df_empty_aff['not_reach_goal']) / df_empty_aff['launched'])

df_empty_aff['neutralized'] = df_empty_aff['neutralized'].replace([np.inf, -np.inf], np.nan)

# df_affected portion
df_affected = df1[df1['affected_region1'].notna()]
df_affected[cols_to_fill] = df_affected[cols_to_fill].fillna(0)
df_affected[cols_to_fill] = df_affected[cols_to_fill].div(df_affected['num_non_na_dist'], axis=0)

## pivoting
#use group by to find pivot errors?
#df_wide = df_affected.reset_index(drop=True)
#df_wide['row_id'] = df_wide.index
#dtale.show(df_wide).open_browser()

#dist_aff_cols = [col for col in df_wide.columns if col.startswith("dist_aff_")]
#id_vars = [col for col in df_wide.columns if col == "row_id" or col not in dist_aff_cols]
df_wide = df_affected.reset_index(drop=True)
df_wide['row_id'] = df_wide.index

df_affected = df_affected.reset_index(drop=True)
df_affected['row_id'] = df_affected.index + 1  # Optional +1 to match R indexing

# Select all columns starting with 'dist_aff_'
dist_cols = [col for col in df_affected.columns if col.startswith('dist_aff_')]
region_cols = [col for col in df_affected.columns if col.startswith('affected_region')]

# Sort both to keep order aligned

def sort_by_suffix(col_list, prefix):
    return sorted(col_list, key=lambda x: int(re.search(rf"{prefix}(\d+)", x).group(1)))

dist_cols = sort_by_suffix(
    [col for col in df_affected.columns if col.startswith('dist_aff_')],
    'dist_aff_'
)
region_cols = sort_by_suffix(
    [col for col in df_affected.columns if col.startswith('affected_region')],
    'affected_region'
)

# Build a long-format DataFrame manually, grouping by row (launch site)
long_rows = []
for idx, row in df_affected.iterrows():
    for i in range(len(dist_cols)):
        long_rows.append({
            'row_id': row['row_id'],
            'aff_index': i + 1,
            'distance': row[dist_cols[i]],
            'region_name': row[region_cols[i]],
            **{col: row[col] for col in df_affected.columns if col not in dist_cols + region_cols}
        })

df_long = pd.DataFrame(long_rows)
df_long['region_name'] = df_long['region_name'].replace('', np.nan)
df_long = df_long[df_long['region_name'].notna()]


#dtale.show(df_long).open_browser()

########

df_long['neutralized'] = (
    (df_long['destroyed'] + df_long['not_reach_goal']) / df_long['launched']
    )

df_long['neutralized'] = df_long['neutralized'].replace([np.inf, -np.inf], np.nan)

#dtale.show(df_long).open_browser()

df3 = pd.concat([df_long, df_empty_aff], ignore_index=True)
#st.dataframe(df3)

### adding missile data
mis_type_clean = mis_type[["model", "category"]]

def category(x):
    if x == "X-101/X-555 and Kalibr and X-59/X-69":
        return "cruise missile"
    elif x == "X-47 Kinzhal":
        return "ballistic missile"
    else:
        return x

mis_type_clean['category'] = mis_type_clean['category'].apply(category)

df_mis = pd.merge(df3, mis_type_clean, on='model', how='left')

df_mis['distance'] = pd.to_numeric(df_mis['distance'])
df_mis['duration_sec'] = df_mis['duration'].dt.total_seconds()
df_mis['Duration (min)'] = df_mis['duration_sec']/60
df_mis = df_mis[df_mis['neutralized'] <=1]
df_mis['Neutralization Rate (%)'] = df_mis['neutralized']*100

##########################
#neutralized plot
fig = px.scatter(
    df_mis,
    x="Duration (min)",
    y="Neutralization Rate (%)",
    color = "category",
    title="Neutralization Rate by Duration",
    labels={
        "Duration (min)": "Duration (min)",
        "Neutralization Rate (%)": "Neutralization Rate (%)"
    }
)

fig.update_layout(  # Optional: set fixed y range
    xaxis=dict(type='linear')
)

st.plotly_chart(fig, use_container_width=True)