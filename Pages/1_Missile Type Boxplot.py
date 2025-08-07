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

PATH = r'C:\UChicago\2025 Summer Internship - CSIS\Firepower Strike\Data'
path_to_file = os.path.join(PATH, 'cleaned_FPS2.csv')
df = pd.read_csv(path_to_file)

path_to_file = os.path.join(PATH, 'missiles_and_uav.csv')
mis_type = pd.read_csv(path_to_file)

path_to_file = os.path.join(PATH, 'launched_geom2.csv')
launched_geom = pd.read_csv(path_to_file)

path_to_file = os.path.join(PATH, 'probit_model2.csv')
probit_df = pd.read_csv(path_to_file)

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
### graphs

#st.scatter_chart(data=df_mis, x='category', y='neutralized')

###
df_mis2 = df_mis
df_mis2['launched'] = df_mis2['launched'].replace([np.inf, -np.inf], np.nan)
sd_launch = np.nanstd(df_mis2['launched'])

df_mis = df_mis[df_mis['launched'].notna()]
df_mis['launched'] = pd.to_numeric(df_mis['launched']).astype(float)
sd_launch = np.std(df_mis['launched'])

df_unfiltered = df_mis


#display(df_filtered)
#dtale.show(df_mis_try).open_browser()

df_mis['neutralized'] = pd.to_numeric(df_mis['neutralized'], errors='coerce')

df_mis = df_mis[df_mis['neutralized']<= 1]

######
df_mis['date_month'] = pd.to_datetime(df_mis['time_start']).dt.to_period('M').dt.to_timestamp()

monthly_sd = (
    df_mis.groupby('date_month')
    .agg(
        salvo_month=('launched', lambda x: x.std(skipna=True) * 1.5),
        n=('launched', 'count')
    )
    .reset_index()
)

df_mis['date_month'] = pd.to_datetime(df_mis['time_start']).dt.to_period('M').dt.to_timestamp()

df_mis = df_mis.merge(monthly_sd, on='date_month', how='left')
df_mis['is_salvo_month'] = (df_mis['launched'] >= df_mis['salvo_month']).astype(int)
#df_mis[['is_salvo_month', 'date_month', 'salvo_month', 'launched']]

df_mis['distance'] = pd.to_numeric(df_mis['distance'])
df_mis['duration_sec'] = df_mis['duration'].dt.total_seconds()
df_mis['Duration (min)'] = df_mis['duration_sec']/60
df_mis = df_mis[df_mis['neutralized'] <=1]
df_mis['Neutralization Rate (%)'] = df_mis['neutralized']*100
df_mis["Category of Missile"] = df_mis['category']
#############################
# Missile boxplot

fig_box = px.box(df_mis, x='Category of Missile', y='Neutralization Rate (%)', title='Neutralized Counts by Missile Category')
# Compute counts per category
counts = df_mis['Category of Missile'].value_counts()

# Add count annotations above each category
for i, cat in enumerate(counts.index):
    fig_box.add_annotation(
        x=cat,
        y=105,  # slightly above your y-axis max
        text=f"n={counts[cat]}",
        showarrow=False,
        font=dict(size=12),
        yanchor='bottom'
    )

# Optional: Extend y-axis to fit annotations
fig_box.update_layout(yaxis=dict(range=[0, 110]))
st.plotly_chart(fig_box, use_container_width=True)

##############################
