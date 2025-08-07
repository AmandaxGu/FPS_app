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

df_mis_try = df_mis.groupby('launch_place', as_index=False)['launched'].sum()
df_filtered = df_mis_try[df_mis_try['launched'] > 25]
#display(df_filtered)
#dtale.show(df_mis_try).open_browser()

df_mis['neutralized'] = pd.to_numeric(df_mis['neutralized'], errors='coerce')
df_mis = df_mis[df_mis['launch_place'].isin(df_filtered['launch_place'])]
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


##########################################################
###### cloropleth map

launched_geom = launched_geom[launched_geom['launch_place'].apply(lambda x: isinstance(x, str))]

# Convert WKT to geometry and create GeoDataFrame in one step
launched_geom['geometry'] = launched_geom['geometry_wkt'].apply(wkt.loads)
gdf = gpd.GeoDataFrame(launched_geom, geometry='geometry', crs='EPSG:4326')

# Simplify geometries
gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.01, preserve_topology=True)

def remove_holes(geom):
    if geom is None:
        return None
    if isinstance(geom, Polygon):
        return Polygon(geom.exterior)
    elif isinstance(geom, MultiPolygon):
        return MultiPolygon([Polygon(p.exterior) for p in geom.geoms])
    return geom

gdf['geometry'] = gdf['geometry'].apply(remove_holes)

######################
df_summary = df.groupby("launch_place", as_index=False)["launched"].sum()

## create new dataset with launched and neutralized with geoms
df_mis_group = df_unfiltered.groupby("launch_place", as_index=False).agg({
    "launched": "sum",
    "neutralized": "mean",
    "launch_place": "first"   # or "last" - whichever you prefer
})

####
# Step 1: Group by both launch_place and model, summing launched
model_counts = df_unfiltered.groupby(["launch_place", "model"])["launched"].sum().reset_index()

# Step 2: For each launch_place, get top 5 models by launched
top5 = (
    model_counts.sort_values(['launch_place', 'launched'], ascending=[True, False])
    .groupby('launch_place')
    .head(5)
    .copy()
)

# Step 3: Calculate total launched per launch_place for % calc
totals = top5.groupby('launch_place')['launched'].sum().rename("total")
top5 = top5.merge(totals, on='launch_place')
top5['pct'] = top5['launched'] / top5['total'] * 100

# Step 4: Assign rank 1–5 to the top models
top5['rank'] = top5.groupby('launch_place')['launched'].rank(ascending=False, method='first').astype(int)

# Step 5: Pivot to wide format with 5 columns for names and 5 for percentages
model_name_pivot = top5.pivot(index='launch_place', columns='rank', values='model')
model_pct_pivot = top5.pivot(index='launch_place', columns='rank', values='pct')

# Step 6: Rename columns
model_name_pivot.columns = [f'model_{i}' for i in model_name_pivot.columns]
model_pct_pivot.columns = [f'model_{i}_pct' for i in model_pct_pivot.columns]

# Step 7: Combine them
top5_wide = pd.concat([model_name_pivot, model_pct_pivot], axis=1).reset_index()

# Step 8: Merge with main summary df
df_mis_group = df_unfiltered.groupby("launch_place", as_index=False).agg({
    "launched": "sum",
    "neutralized": "mean"
})
df_final = df_mis_group.merge(top5_wide, on='launch_place', how='left')

#####

#df_mis_group['launch_place'].nunique()
df_mis_geom = df_final.merge(gdf, on='launch_place', how='left')
df_mis_geom = df_mis_geom.drop(columns='geometry_wkt')

#####
# Make sure 'launch_place' is unique in gdf
choropleth_gdf = df_mis_geom.copy()
choropleth_gdf = choropleth_gdf.round(3)

choropleth_gdf = choropleth_gdf.set_geometry('geometry')
choropleth_gdf = choropleth_gdf.set_crs('EPSG:4326')

# Create choropleth map
hover_cols = ['neutralized', 'launched']

# Step 2: Convert percentage values and track valid hover columns
for i in range(1, 6):
    pct_col = f'model_{i}_pct'
    if pct_col in choropleth_gdf.columns:
        # Convert to % string, round to 1 decimal, leave NaNs as is
        choropleth_gdf[pct_col] = choropleth_gdf[pct_col].round(1)
        choropleth_gdf[pct_col] = choropleth_gdf[pct_col].astype(str) + '%'
        choropleth_gdf[pct_col] = choropleth_gdf[pct_col].where(choropleth_gdf[f'model_{i}'].notna())

choropleth_gdf['neutralized'] = choropleth_gdf['neutralized'] * 100
# Step 3: Build clean custom hover column manually
def build_hover(row):
    lines = [
        f"<b>{row['launch_place']}</b>",
        f"Launched: {row['launched']:.3f}",
        f"Neutralized: {row['neutralized']:.3f}"
    ]
    for i in range(1, 6):
        model = row.get(f'model_{i}')
        pct = row.get(f'model_{i}_pct')
        if pd.notna(model) and pd.notna(pct):
            lines.append(f"{model}: {pct}")
    return "<br>".join(lines)

choropleth_gdf['hover_text'] = choropleth_gdf.apply(build_hover, axis=1)


#choropleth_gdf.to_csv('choropleth.csv', index=False)

#### Probit map #############

sig_results = probit_df[probit_df['p_value'] < 0.05]

sig_results['mean_predicted_prob'] = sig_results['mean_predicted_prob'].round(3)
sig_results['p_value'] = sig_results['p_value'].round(3)

# Step 2: Get distinct coordinates for each launch place
df_launch = df[['launch_place', 'lon_launch', 'lat_launch']].drop_duplicates()
df_launch = df_launch.dropna(subset=['lon_launch', 'lat_launch'])


# Step 3: Merge and process
sig_map_df = sig_results.merge(df_launch, on='launch_place', how='left')
sig_map_df = sig_map_df.dropna(subset=['lon_launch', 'lat_launch'])

# Step 4: Add sign and magnitude columns
sig_map_df['magnitude'] = sig_map_df['mean_predicted_prob'].abs()
sig_map_df['label'] = sig_map_df['launch_place']

sig_map_df['Mean Predicted Probability (%)'] = (
    sig_map_df['mean_predicted_prob'] * 100
).round(1)
sig_map_df['P. Value'] = sig_map_df['p_value']
##########################

#probit map
st.set_page_config(layout="wide")
fig_probit = px.scatter_mapbox(
    sig_map_df,
    lat='lat_launch',
    lon='lon_launch',
    color='Mean Predicted Probability (%)',
    size='magnitude',
    hover_name='label',
    hover_data={
        'Mean Predicted Probability (%)': True,
        'P. Value': True,
        'lat_launch': False,
        'lon_launch': False,
        'magnitude' : False
    },
    color_continuous_scale='Viridis', 
    zoom=4,
    title="Significant Launch Place Effects"
)

fig_probit.update_layout(mapbox_style='carto-positron', height=600)
fig_probit.update_layout(margin={"r":0,"t":30,"l":0,"b":0})

st.plotly_chart(fig_probit, use_container_width=True)