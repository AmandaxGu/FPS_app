import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, GeometryCollection
import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np
import os
import streamlit as st
import re
from statsmodels.stats.sandwich_covariance import cov_hc0, cov_hc1
from statsmodels.formula.api import ols
from shapely.geometry import GeometryCollection

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


# ------------------------------
# Function: Initialize Geometries
# ------------------------------
def initialize_geometries_vectorized(df):
    df = df[df['affected_region1'].notna()].copy()

    # Launch geometries
    mask_launch = df['lon_launch'].notna() & df['lat_launch'].notna()
    df.loc[mask_launch, 'geom_launch'] = gpd.points_from_xy(df.loc[mask_launch, 'lon_launch'], df.loc[mask_launch, 'lat_launch'])
    df.loc[~mask_launch, 'geom_launch'] = GeometryCollection()

    # Affected geometries
    for i in range(1, 11):
        lon_col = f'lon_aff_{i}'
        lat_col = f'lat_aff_{i}'
        geom_col = f'geom_aff_{i}'

        mask = df[lon_col].notna() & df[lat_col].notna()
        df.loc[mask, geom_col] = gpd.points_from_xy(df.loc[mask, lon_col], df.loc[mask, lat_col])
        df.loc[~mask, geom_col] = GeometryCollection()

    return df

# ------------------------------
# Load and process your data
# ------------------------------

# Replace with your own loaded data
# df_affected = pd.read_csv("your_data.csv")
# df_empty_aff = pd.read_csv("your_other_data.csv")

# For demonstration, we assume df_affected and df_empty_aff are already available
combined_df = pd.concat([df_affected, df_empty_aff], ignore_index=True)
new_aff = initialize_geometries_vectorized(combined_df)

# ------------------------------
# Build launch and affected GeoDataFrames
# ------------------------------

affected_sfs = []
for i in range(1, 10):
    geom_col = f'geom_aff_{i}'
    region_col = f'affected_region{i}'
    if geom_col in new_aff.columns and region_col in new_aff.columns:
        aff = new_aff.loc[~new_aff[geom_col].isna()]
        aff = aff.loc[aff[geom_col].map(lambda g: not g.is_empty)]
        aff_gdf = gpd.GeoDataFrame(
            aff[[region_col, geom_col]].rename(columns={region_col: 'region'}),
            geometry=geom_col,
            crs="EPSG:4326"
        )
        affected_sfs.append(aff_gdf)

launch_sf = gpd.GeoDataFrame(
    new_aff.loc[~new_aff['geom_launch'].isna() & new_aff['geom_launch'].map(lambda g: not g.is_empty)].copy(),
    geometry='geom_launch',
    crs="EPSG:4326"
)

# ------------------------------
# Create Line GeoDataFrame
# ------------------------------

# Combine affected points
affected_sfs_renamed = [gdf.rename_geometry('geometry') for gdf in affected_sfs]
all_aff_points = gpd.GeoDataFrame(pd.concat(affected_sfs_renamed, ignore_index=True), crs="EPSG:4326")

# Prepare columns
launch_points = launch_sf[['launch_place', 'geom_launch', 'launched']].rename(columns={'geom_launch': 'geometry_launch'})
all_aff_points = all_aff_points.rename(columns={'geometry': 'geometry_aff'})

# Cross join
cross = launch_points.merge(all_aff_points, how='cross')
cross['geometry'] = [LineString([launch, aff]) for launch, aff in zip(cross['geometry_launch'], cross['geometry_aff'])]

lines_gdf = gpd.GeoDataFrame(
    cross[['launch_place', 'launched', 'region', 'geometry']],
    geometry='geometry',
    crs="EPSG:4326"
).rename(columns={'region': 'region_name'})

# ------------------------------
# Convert for PyDeck
# ------------------------------

lines_df = pd.DataFrame({
    'start_lon': lines_gdf.geometry.apply(lambda g: g.coords[0][0]),
    'start_lat': lines_gdf.geometry.apply(lambda g: g.coords[0][1]),
    'end_lon': lines_gdf.geometry.apply(lambda g: g.coords[1][0]),
    'end_lat': lines_gdf.geometry.apply(lambda g: g.coords[1][1]),
    'launch_place': lines_gdf['launch_place'],
    'region_name': lines_gdf['region_name'],
    'launched': lines_gdf['launched'].fillna(1)
})

# ------------------------------
# Streamlit + PyDeck Display
# ------------------------------

st.title("Missile Launch → Impact Dyads")

line_layer = pdk.Layer(
    "LineLayer",
    data=lines_df,
    get_source_position=["start_lon", "start_lat"],
    get_target_position=["end_lon", "end_lat"],
    get_width="launched",
    get_color=[255, 0, 0, 120],
    pickable=True,
    auto_highlight=True
)

view_state = pdk.ViewState(
    longitude=lines_df['start_lon'].mean(),
    latitude=lines_df['start_lat'].mean(),
    zoom=5.5,
    pitch=0
)

st.pydeck_chart(pdk.Deck(
    layers=[line_layer],
    initial_view_state=view_state,
    tooltip={"text": "Launch: {launch_place}\nImpact: {region_name}\nLaunched: {launched}"}
))
