# -*- coding: utf-8 -*-
"""
Created on 2025-10-25 17:00:17

@author: Maurice Roots

Description:
     - Functions to perform GeoSlicing
"""

from shapely.ops import nearest_points
import geopandas as gpd 
from shapely.geometry import Point


def _add_distance_from_point(gdf: gpd.GeoDataFrame, location: Point):
    gdf_temp = gdf.copy().to_crs(3857)
    location = gpd.GeoSeries([location], crs="EPSG:4326").to_crs(3857).iloc[0]
    gdf_temp["distance_km"] = gdf_temp.geometry.distance(location) / 1000
    return gdf_temp.to_crs(gdf.crs)

def nearest_obs(gdf1, gdf2, radius_km):
    original_gdf1 = gdf1.copy()
    original_gdf2 = gdf2.copy()

    gdf1 = gdf1[
        [original_gdf1.geometry.name]
        ]
    
    gdf2 = gdf2[
        [original_gdf2.geometry.name]
        ].drop_duplicates(
            original_gdf2.geometry.name
            )

    gdf1 = gdf1.to_crs(epsg=3857)
    gdf2 = gdf2.to_crs(epsg=3857)

    joined_left = gpd.sjoin_nearest(
        gdf1,
        gdf2,
        how="left",
        max_distance=radius_km*1000,
        distance_col="distance"
        ).dropna(subset=["distance"]).to_crs(original_gdf1.crs)

    joined_right = gpd.sjoin_nearest(
        gdf1,
        gdf2,
        how="right",
        max_distance=radius_km*1000,
        distance_col="distance"
        ).dropna(subset=["distance"]).to_crs(original_gdf2.crs)
    
    return joined_left, joined_right


