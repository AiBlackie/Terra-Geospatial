# app.py
# -*- coding: utf-8 -*-
"""Terra Caribbean Property Intelligence:Geospatial View - Streamlit Version"""

import streamlit as st
import pandas as pd
import geopandas as gpd
import osmnx as ox
import folium
from shapely.geometry import Point
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import tempfile
import os
import traceback
import unicodedata
import re
import datetime
import math
import openai # Ensure openai is imported
import requests # Added for weather API calls

# OSMnx settings
ox.settings.log_console = False
ox.settings.use_cache = True
ox.settings.timeout = 600

# --- Key Setup ---
# This pattern works on Cloud Run (using environment variables) and locally (using secrets.toml)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
else:
    # Add a message in the app if the key is missing, but don't block execution
    pass


# --- Helper functions ---

@st.cache_data(ttl=600, show_spinner=False) # Cache weather data for 10 minutes
def get_weather_data(api_key, city="Bridgetown,BB"):
    """Fetches weather data from OpenWeatherMap API."""
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    complete_url = f"{base_url}?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(complete_url, timeout=10)
        response.raise_for_status() 
        data = response.json()
        
        main = data.get("main")
        weather_list = data.get("weather")
        wind = data.get("wind") 
        
        if main and weather_list:
            weather_info = weather_list[0]
            temp = main.get("temp")
            feels_like = main.get("feels_like")
            humidity = main.get("humidity")
            description = weather_info.get("description")
            icon_code = weather_info.get("icon")
            icon_url = f"http://openweathermap.org/img/wn/{icon_code}@2x.png" if icon_code else ""
            wind_speed = wind.get("speed") if wind else None
            
            return {
                "temp": temp,
                "feels_like": feels_like,
                "humidity": humidity,
                "wind_speed": wind_speed,
                "description": description.capitalize() if description else "N/A",
                "icon_url": icon_url,
                "city_name": data.get("name", city.split(',')[0])
            }
        else:
            return None
    except requests.exceptions.RequestException:
        return None
    except Exception: 
        return None

def parse_size_to_sqft_static(size_str):
    if pd.isna(size_str) or not isinstance(size_str, str): return np.nan
    size_str_l = size_str.lower(); num_match = re.search(r'([\d,]+\.?\d*)\s*(.*)', size_str_l)
    if not num_match: return np.nan
    try: num_val = float(num_match.group(1).replace(',', '')); unit_str = num_match.group(2).strip()
    except ValueError: return np.nan
    if 'acre' in unit_str: return num_val * 43560
    if ('sq' in unit_str and ('ft' in unit_str or 'feet' in unit_str)) or 'sf' in unit_str: return num_val
    if unit_str == "" and num_val > 200: return num_val 
    return np.nan

def standardize_property_type_static(pt_series_input):
    if not isinstance(pt_series_input, pd.Series):
        if isinstance(pt_series_input, str):
            pt_series_input = pd.Series([pt_series_input])
        else:
            return pd.Series(pt_series_input, dtype=object)

    pt_series_clean = pt_series_input.fillna("").astype(str).str.lower().str.strip()
    standardized_values = pd.Series(['Other'] * len(pt_series_clean), index=pt_series_input.index, dtype=object)

    standardized_values.loc[pt_series_clean == 'residential'] = 'Residential'
    standardized_values.loc[pt_series_clean == 'commercial'] = 'Commercial'
    standardized_values.loc[pt_series_clean == 'land'] = 'Land'
    standardized_values.loc[pt_series_clean == 'lot'] = 'Land'

    res_keywords = [
        'house', 'home', 'villa', 'apartment', 'condo', 'townhouse',
        'dwelling', 'bungalow', 'chalet', 'duplex', 'resi unit',
        'flat', 'cottage', 'terrace', 'maisonette', 'penthouse', 'studio'
    ]
    com_keywords = [
        'office', 'industrial', 'retail', 'shop', 'warehouse', 'business',
        'hotel', 'restaurant', 'bar', 'showroom', 'plaza', 'mall',
        'factory', 'plant', 'comm bldg', 'comm. bldg',
        'store', 'center', 'centre', 'complex'
    ]
    land_keywords_fallback = ['plot', 'acre', 'field', 'farm', 'ground']

    res_pattern = r'\b(' + '|'.join(res_keywords) + r')\b'
    com_pattern = r'\b(' + '|'.join(com_keywords) + r')\b'
    land_pattern_fallback = r'\b(' + '|'.join(land_keywords_fallback) + r')\b'

    mask_res_fallback = pt_series_clean.str.contains(res_pattern, na=False, regex=True) & (standardized_values == 'Other')
    standardized_values.loc[mask_res_fallback] = 'Residential'

    mask_com_fallback = pt_series_clean.str.contains(com_pattern, na=False, regex=True) & (standardized_values == 'Other')
    standardized_values.loc[mask_com_fallback] = 'Commercial'

    mask_land_fallback = pt_series_clean.str.contains(land_pattern_fallback, na=False, regex=True) & (standardized_values == 'Other')
    standardized_values.loc[mask_land_fallback] = 'Land'

    other_strings = ['', 'unknown', 'nan', 'n/a', '-', 'other', 'none']
    standardized_values.loc[pt_series_clean.isin(other_strings)] = 'Other'

    return standardized_values


def clean_parish_name_generic_static(name_val):
    if pd.isna(name_val): return None
    s = str(name_val).strip()
    if not s or s.lower() in ['nan', 'none']: return None
    s_normalized = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'); s_titled = s_normalized.title()
    replacements = {'St.': 'Saint', 'St ': 'Saint ', 'St. ': 'Saint ', 'St': 'Saint','Christchurch': 'Christ Church','Parish Of Saint Michael': 'Saint Michael', 'Parish Of Christ Church': 'Christ Church','Parish Of Saint James': 'Saint James', 'Parish Of Saint Lucy': 'Saint Lucy','Parish Of Saint Peter': 'Saint Peter', 'Parish Of Saint Andrew': 'Saint Andrew','Parish Of Saint Joseph': 'Saint Joseph', 'Parish Of Saint Thomas': 'Saint Thomas','Parish Of Saint John': 'Saint John', 'Parish Of Saint George': 'Saint George','Parish Of Saint Philip': 'Saint Philip',}
    cleaned_s = s_titled
    for old, new in replacements.items(): cleaned_s = cleaned_s.replace(old, new)
    for saint_parish_base in ['Lucy', 'Peter', 'Andrew', 'Joseph', 'John', 'George', 'Thomas', 'Philip', 'Michael', 'James']:
        if (f' {saint_parish_base}' in cleaned_s or cleaned_s.endswith(saint_parish_base) or cleaned_s == saint_parish_base) and \
           f'Saint {saint_parish_base}' not in cleaned_s and 'Christ Church' not in cleaned_s :
            cleaned_s = cleaned_s.replace(saint_parish_base, f'Saint {saint_parish_base}')
    return cleaned_s.strip().replace('  ', ' ')

def decimal_to_dms(deg, is_lat):
    if pd.isna(deg): return "N/A"
    try:
        d = int(deg); m_float = abs(deg - d) * 60; m = int(m_float); s = (m_float - m) * 60
        direction = ('N' if deg >= 0 else 'S') if is_lat else ('E' if deg >= 0 else 'W')
        return f"{abs(d)}° {m}' {s:.3f}\" {direction}"
    except Exception: return "Error converting"

def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    if any(pd.isna([lat1, lon1, lat2, lon2])): return np.nan
    R = 6371
    try:
        lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(math.radians, [lat1, lon1, lat2, lon2])
        dlon, dlat = lon2_rad - lon1_rad, lat2_rad - lat1_rad
        a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
    except Exception: return np.nan

@st.cache_data(show_spinner=False, persist="disk")
def load_cached_property_data(file_content_bytes, filename_for_log, log_capture_list_ref):
    log_capture_list_ref.append(f"Executing load_cached_property_data for {filename_for_log}...\n")
    df = None
    file_like_object = io.BytesIO(file_content_bytes)
    try:
        if filename_for_log.lower().endswith('.xlsx'): df = pd.read_excel(file_like_object, engine='openpyxl')
        elif filename_for_log.lower().endswith('.csv'): df = pd.read_csv(file_like_object)
        else: log_capture_list_ref.append(f"Unsupported file type: {filename_for_log}\n"); return pd.DataFrame()
        log_capture_list_ref.append(f"Successfully parsed: {filename_for_log}\n")
        df['original_index'] = df.index
        column_map = {'Parish': ['Parish', 'parish', 'PARISH', 'Location'],'Price': ['Price', 'price', 'PRICE', 'List Price'],'Description': ['Description', 'description', 'DESCRIPTION', 'Remarks'],'Property Type': ['Property Type', 'Type', 'TYPE', 'Property_Type', 'property type'],'Category': ['Category', 'category', 'CATEGORY', 'Listing Type'],'Name': ['Name', 'name', 'NAME', 'Property Name', 'Listing Name'],'Size': ['Size', 'size', 'Lot Size', 'Land Area', 'Building Size', 'Floor Area']}
        for std_name, alt_names in column_map.items():
            found_col = False
            for alt_name in alt_names:
                if alt_name in df.columns:
                    if std_name != alt_name: df.rename(columns={alt_name: std_name}, inplace=True)
                    found_col = True; break
            if not found_col and std_name not in df.columns: df[std_name] = None
        if 'Parish' not in df.columns or df['Parish'].isnull().all(): log_capture_list_ref.append("CRIT: Parish col missing/all null.\n"); return pd.DataFrame()
        parish_na_mask = df['Parish'].isna() | (df['Parish'].astype(str).str.strip() == ""); df = df[~parish_na_mask].copy()
        if df.empty: log_capture_list_ref.append("Empty after Parish clean.\n"); return pd.DataFrame()
        def clean_excel_parish_name(name_val):
            s = str(name_val).strip(); s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn').title()
            replacements = {'St.':'Saint','St ':'Saint ','St. ':'Saint ','St':'Saint','Christchurch':'Christ Church'}
            for old, new in replacements.items(): s = s.replace(old, new)
            for p_name in ['Lucy','Peter','Andrew','Joseph','John','George','Thomas','Philip','Michael','James']:
                if p_name in s and f'Saint {p_name}' not in s and 'Christ Church' not in s: s = s.replace(p_name, f'Saint {p_name}')
            return s.strip().replace('  ',' ')
        df['Parish'] = df['Parish'].apply(clean_excel_parish_name)
        if 'Price' not in df.columns or df['Price'].isnull().all(): log_capture_list_ref.append("CRIT: Price col missing/all null.\n"); return pd.DataFrame()
        df['Price_numeric'] = pd.to_numeric(df['Price'].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
        df = df[df['Price_numeric'].notna()].copy(); df['Price'] = df['Price_numeric']
        df.drop(columns=['Price_numeric'], errors='ignore', inplace=True)
        if df.empty: log_capture_list_ref.append("Empty after Price clean.\n"); return pd.DataFrame()
        if 'Size' in df.columns: df['Size_sqft'] = df['Size'].apply(parse_size_to_sqft_static)
        else: df['Size_sqft'] = np.nan
        if 'Description' in df.columns: df['Bedrooms'] = df['Description'].astype(str).str.extract(r'(\d+)\s*Bed', expand=False).astype(float)
        else: df['Bedrooms'] = np.nan
        for col_check in ['Category', 'Property Type']:
            if col_check not in df.columns: df[col_check] = 'Other'
        df['Property Type Standardized'] = standardize_property_type_static(df['Property Type'])
        df['Category'] = df['Category'].fillna('Unknown').astype(str).str.strip()
        final_columns = [c for c in ['Name','Parish','Property Type','Property Type Standardized','Price','Bedrooms','Category','Size_sqft','original_index'] if c in df.columns]
        return df[final_columns].copy()
    except Exception as e: log_capture_list_ref.append(f"Error loading property data: {e}\n{traceback.format_exc()}\n"); return pd.DataFrame()


@st.cache_data(show_spinner=False, persist="disk")
def get_cached_geodata(log_capture_list_ref):
    log_capture_list_ref.append("Executing get_cached_geodata (OSM fetch)...\n")
    results_dict = {k: gpd.GeoDataFrame() for k in ["parishes_gdf", "beaches_gdf", "tourism_gdf", "feature_polygons_gdf", "raw_parishes_from_osm_snapshot", "schools_gdf", "supermarkets_gdf", "roads_gdf"]}
    results_dict["total_island_area_sqkm"] = np.nan
    gdf_keys_order = ["parishes_gdf", "beaches_gdf", "tourism_gdf", "feature_polygons_gdf", "raw_parishes_from_osm_snapshot", "schools_gdf", "supermarkets_gdf", "roads_gdf", "total_island_area_sqkm"]

    try:
        log_capture_list_ref.append("Downloading Barbados country boundary...\n")
        barbados_boundary = ox.geocode_to_gdf("Barbados")
        if barbados_boundary.empty:
            log_capture_list_ref.append("Failed to download Barbados boundary.\n")
            return tuple(results_dict[k] for k in gdf_keys_order)

        poly = barbados_boundary.geometry.iloc[0]
        try:
            barbados_boundary_proj = barbados_boundary.to_crs("EPSG:32620")
            results_dict["total_island_area_sqkm"] = barbados_boundary_proj.area.iloc[0] / 1_000_000
            log_capture_list_ref.append(f"Total island area: {results_dict['total_island_area_sqkm']:.2f} sq km.\n")
        except Exception as e_area:
            log_capture_list_ref.append(f"Could not calculate total island area: {e_area}\n")


        tags_list = [{"boundary":"administrative","admin_level":"6"},{"place":"parish"}]
        log_capture_list_ref.append("Attempting to download parish boundaries from OSM...\n")
        temp_parishes_data = gpd.GeoDataFrame()
        found_parish_data_source = False
        for tags in tags_list:
            try:
                current_fetch = ox.features_from_polygon(poly, tags)
                if not current_fetch.empty and any(name_tag in current_fetch.columns for name_tag in ['name', 'official_name', 'name:en']):
                    current_fetch = current_fetch[current_fetch.geometry.type.isin(['Polygon','MultiPolygon'])]
                    if not current_fetch.empty:
                        temp_parishes_data = current_fetch
                        log_capture_list_ref.append(f"Successfully downloaded features using tags: {tags}\n")
                        found_parish_data_source = True; break
            except Exception as e: log_capture_list_ref.append(f"Parish download attempt with tags {tags} failed: {e}\n")

        if not found_parish_data_source or temp_parishes_data.empty:
            log_capture_list_ref.append("OSM parish download failed or no usable data.\n")
        else:
            parishes_gdf_temp = temp_parishes_data
            results_dict["raw_parishes_from_osm_snapshot"] = parishes_gdf_temp.copy()
            name_col_candidates = ['name', 'official_name', 'name:en', 'alt_name', 'loc_name']
            primary_name_col_found = next((col for col in name_col_candidates if col in parishes_gdf_temp.columns), None)
            if primary_name_col_found:
                   parishes_gdf_temp['OSM_Parish_Name'] = parishes_gdf_temp[primary_name_col_found].fillna('Unnamed Parish')
            else: parishes_gdf_temp['OSM_Parish_Name'] = parishes_gdf_temp.index.astype(str).fillna('Unnamed Parish')
            parishes_gdf_temp['name'] = parishes_gdf_temp['OSM_Parish_Name'].apply(clean_parish_name_generic_static)
            parishes_gdf_temp = parishes_gdf_temp[['name', 'OSM_Parish_Name', 'geometry']].copy()
            parishes_gdf_temp.set_crs(barbados_boundary.crs, inplace=True)
            parishes_gdf_temp = parishes_gdf_temp[parishes_gdf_temp.geometry.is_valid & parishes_gdf_temp.geometry.notna()]
            results_dict["parishes_gdf"] = parishes_gdf_temp
            if not parishes_gdf_temp.empty: log_capture_list_ref.append(f"Found and processed {len(parishes_gdf_temp)} valid OSM parish geometries.\n")

        layer_configs = {
            "beaches_gdf": {"tags": {"natural": "beach"}, "name": "beach"},
            "tourism_gdf": {"tags": {"tourism": True}, "name": "tourism points"},
            "feature_polygons_gdf": {"tags": {"leisure": ["park", "golf_course", "nature_reserve"], "amenity": ["university", "college"], "landuse": ["cemetery", "religious"]}, "name": "feature polygons", "geom_types": ['Polygon', 'MultiPolygon'], "name_notna": True},
            "schools_gdf": {"tags": {"amenity": "school"}, "name": "school"},
            "supermarkets_gdf": {"tags": {"shop": "supermarket"}, "name": "supermarket"},
        }
        for gdf_name, config in layer_configs.items():
            log_capture_list_ref.append(f"Downloading {config['name']} data from OSM...\n")
            try:
                current_gdf = ox.features_from_polygon(poly, config["tags"])
                if not current_gdf.empty:
                    current_gdf = current_gdf[current_gdf.geometry.notna()]
                    if "geom_types" in config: current_gdf = current_gdf[current_gdf.geometry.type.isin(config["geom_types"])]
                    if 'name' in current_gdf.columns and config.get("name_notna", False): current_gdf = current_gdf[current_gdf['name'].notna()]
                    if not current_gdf.empty: current_gdf = current_gdf[current_gdf.geometry.is_valid]
                    results_dict[gdf_name] = current_gdf
                    log_capture_list_ref.append(f"Found {len(current_gdf)} {config['name']}.\n")
                else: log_capture_list_ref.append(f"No {config['name']} found in OSM.\n")
            except Exception as e_feat: log_capture_list_ref.append(f"Error fetching {config['name']}: {e_feat}\n")

        try:
            log_capture_list_ref.append("Downloading road network for Barbados from OSM...\n")
            graph_roads = ox.graph_from_polygon(poly, network_type="drive", retain_all=False, truncate_by_edge=True)
            _, edges_gdf = ox.graph_to_gdfs(graph_roads)
            roads_gdf_temp = edges_gdf[['geometry', 'name', 'highway', 'length']].copy() 
            if not roads_gdf_temp.empty:
                roads_gdf_temp = roads_gdf_temp[roads_gdf_temp.geometry.notna() & roads_gdf_temp.geometry.is_valid]
                results_dict["roads_gdf"] = roads_gdf_temp
                log_capture_list_ref.append(f"Found {len(roads_gdf_temp)} road segments.\n")
            else: log_capture_list_ref.append("No road segments found in OSM.\n")
        except Exception as e_road: log_capture_list_ref.append(f"Error fetching road network: {e_road}\n{traceback.format_exc()}")

        return tuple(results_dict[k] for k in gdf_keys_order)

    except Exception as e:
        log_capture_list_ref.append(f"General error in get_geodata: {e}\n{traceback.format_exc()}\n")
        return tuple(results_dict[k] for k in gdf_keys_order)

@st.cache_data(show_spinner=False, persist="disk")
def get_cached_alternative_parish_data(log_capture_list_ref):
    log_capture_list_ref.append("Executing get_cached_alternative_parish_data...\n")
    parishes_alt = gpd.GeoDataFrame()
    try:
        log_capture_list_ref.append("Attempting to load parish data from local file 'barbados_parishes.geojson'...\n")
        script_dir = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
        possible_paths = ["barbados_parishes.geojson", os.path.join(script_dir, "barbados_parishes.geojson")]
        found_path = next((p for p in possible_paths if os.path.exists(p)), None)
        if not found_path: log_capture_list_ref.append("Local parish file 'barbados_parishes.geojson' not found.\n"); return parishes_alt
        parishes_alt = gpd.read_file(found_path); log_capture_list_ref.append(f"Loaded local parishes: {found_path}\n")
        primary_name_col = None
        if 'name' in parishes_alt.columns: primary_name_col = 'name'
        elif 'Name' in parishes_alt.columns: primary_name_col = 'Name'
        elif 'parish' in parishes_alt.columns: primary_name_col = 'parish'
        if primary_name_col and primary_name_col != 'name': parishes_alt.rename(columns={primary_name_col: 'name'}, inplace=True)
        if 'name' not in parishes_alt.columns: log_capture_list_ref.append("Error: No suitable name column in local GeoJSON.\n"); return gpd.GeoDataFrame()
        problematic_strings_to_none = ['nan', 'NaN', 'Nan', 'NONE', 'None', 'none', '', 'null', 'NULL', '<NA>']
        parishes_alt['name'] = parishes_alt['name'].replace(problematic_strings_to_none, None, regex=False)
        parishes_alt['OSM_Parish_Name'] = parishes_alt['name'].copy()
        parishes_alt['name'] = parishes_alt['name'].apply(clean_parish_name_generic_static)
        parishes_alt = parishes_alt[parishes_alt.geometry.is_valid & parishes_alt.geometry.notna()]
        if parishes_alt.empty: log_capture_list_ref.append("No valid GeoJSON geometries after validation.\n"); return gpd.GeoDataFrame()
        if parishes_alt.crs is None: parishes_alt.set_crs("EPSG:4326", inplace=True)
        final_alt_cols = ['name', 'OSM_Parish_Name', 'geometry']; existing_final_cols = [col for col in final_alt_cols if col in parishes_alt.columns]
        for col_ensure in ['name', 'OSM_Parish_Name']:
            if col_ensure not in parishes_alt.columns:
                parishes_alt[col_ensure] = pd.Series([None] * len(parishes_alt), index=parishes_alt.index)
                if col_ensure not in existing_final_cols: existing_final_cols.append(col_ensure)
        return parishes_alt[existing_final_cols]
    except Exception as e: log_capture_list_ref.append(f"Error alt parish data: {e}\n{traceback.format_exc()}\n"); return parishes_alt

class TerraDashboardLogic:
    def __init__(self, uploaded_file_object=None):
        self.analyzed_properties = gpd.GeoDataFrame()
        self.parishes = gpd.GeoDataFrame()
        self.beaches = gpd.GeoDataFrame()
        self.tourism_points = gpd.GeoDataFrame()
        self.feature_polygons = gpd.GeoDataFrame()
        self.schools = gpd.GeoDataFrame()
        self.supermarkets = gpd.GeoDataFrame()
        self.roads = gpd.GeoDataFrame()
        self.total_island_area_sqkm = np.nan
        self.map_html_content = None
        self.chart_path = ""
        self.stats_data_for_streamlit = []
        self.log_capture = []
        self.raw_parishes_from_osm = gpd.GeoDataFrame()
        self.uploaded_file_object = uploaded_file_object
        self.parish_summary_df = pd.DataFrame()
        self.ai_parish_road_assessment_text = None
        self.ai_parish_property_assessment_text = None 

    def _capture_print(self, message): self.log_capture.append(str(message) + "\n")

    def _load_property_data(self):
        if self.uploaded_file_object:
            return load_cached_property_data(self.uploaded_file_object.getvalue(), self.uploaded_file_object.name, self.log_capture)
        return pd.DataFrame()


    def _get_geodata(self):
        parishes, beaches, tourism, features, raw_osm, schools, supermarkets, roads, island_area = get_cached_geodata(self.log_capture)
        self.raw_parishes_from_osm = raw_osm
        self.schools = schools
        self.supermarkets = supermarkets
        self.roads = roads
        self.total_island_area_sqkm = island_area
        return parishes, beaches, tourism, features

    def run_analysis_streamlit(self):
        self.log_capture = []
        self._capture_print("=== Starting Terra Caribbean Analysis (Streamlit) ===")
        try:
            self._capture_print("\nLoading property data...")
            properties = self._load_property_data()
            if properties.empty: self._capture_print("No properties loaded. Aborting."); return False
            self._capture_print(f"\nLoaded and initially processed {len(properties)} properties")

            self._capture_print("\nDownloading/Loading geospatial data...")
            primary_parishes, self.beaches, self.tourism_points, self.feature_polygons = self._get_geodata()

            self.parishes = primary_parishes if not primary_parishes.empty else self._get_alternative_parish_data()
            if self.parishes.empty: self._capture_print("CRITICAL: All parish geospatial data sources failed. Aborting."); return False

            if not self.parishes.empty and 'geometry' in self.parishes.columns:
                self._capture_print(f"\nProcessing {len(self.parishes)} parishes...")
                if 'OSM_Parish_Name' not in self.parishes.columns and 'name' in self.parishes.columns:
                    self.parishes['OSM_Parish_Name'] = self.parishes['name']
                elif 'OSM_Parish_Name' not in self.parishes.columns:
                        self.parishes['OSM_Parish_Name'] = 'Unknown Parish'
                if 'name' not in self.parishes.columns:
                    self.parishes['name'] = self.parishes['OSM_Parish_Name'].apply(clean_parish_name_generic_static)
                self.parishes['OSM_Parish_Name'] = self.parishes['OSM_Parish_Name'].fillna('Unknown Parish').astype(str)
                self.parishes['name'] = self.parishes['name'].fillna(self.parishes['OSM_Parish_Name']).astype(str)
                try:
                    if self.parishes.crs is None: self.parishes.set_crs("EPSG:4326", inplace=True)
                    parishes_projected = self.parishes.to_crs("EPSG:32620")
                    self.parishes['area_sqkm'] = parishes_projected.geometry.area / 1_000_000
                    self._capture_print("Calculated parish areas (sq km).")
                except Exception as e:
                    self._capture_print(f"Error projecting parishes for area calculation: {e}"); self.parishes['area_sqkm'] = np.nan
            else: self.parishes['area_sqkm'] = np.nan


            self._capture_print("\nGeocoding properties...");
            geo_properties = self.geocode_properties_internal(properties, self.parishes.copy())
            if geo_properties.empty: self._capture_print("No properties geocoded. Aborting."); self.display_stats_internal(); return False
            self._capture_print(f"\nSuccessfully geocoded {len(geo_properties)} properties.")

            self._capture_print("\nPerforming spatial analysis...");
            self.analyzed_properties = self.analyze_properties_internal(geo_properties, self.beaches, self.tourism_points)
            if self.analyzed_properties.empty: self._capture_print("Spatial analysis resulted in no properties. Aborting."); return False

            self._capture_print("\nGenerating visualizations...")
            self.create_visualizations_internal(self.analyzed_properties, self.parishes.copy(), self.feature_polygons.copy(), self.tourism_points.copy())
            self.display_stats_internal()
            self._capture_print("\n=== Analysis complete! ===")
            return True
        except Exception as e: self._capture_print(f"\nERROR during analysis: {str(e)}\n{traceback.format_exc()}"); return False

    def geocode_properties_internal(self, df_props, parishes_gdf_in):
        try:
            if df_props.empty or parishes_gdf_in.empty or 'geometry' not in parishes_gdf_in.columns:
                self._capture_print("Cannot geocode: Property data or Parish GDF (with geometry) is empty."); return gpd.GeoDataFrame()
            if 'name' not in parishes_gdf_in.columns and 'OSM_Parish_Name' in parishes_gdf_in.columns:
                parishes_gdf_in['name'] = parishes_gdf_in['OSM_Parish_Name']
            elif 'name' not in parishes_gdf_in.columns:
                   self._capture_print("CRIT ERROR: No suitable 'name' or 'OSM_Parish_Name' in parishes_gdf for join."); return gpd.GeoDataFrame()
            def create_join_key(s_series): return s_series.astype(str).str.lower().str.replace(r'[^a-z0-9\s]','',regex=True).str.replace(r'\s+',' ',regex=True).str.strip()
            df_props['Parish_join_key'] = create_join_key(df_props['Parish'])
            parishes_gdf_in['name_join_key'] = create_join_key(parishes_gdf_in['name'])
            parishes_gdf_in.drop_duplicates(subset=['name_join_key'], keep='first', inplace=True)
            cols_to_merge = ['name_join_key','OSM_Parish_Name','geometry']
            if 'area_sqkm' in parishes_gdf_in.columns: cols_to_merge.append('area_sqkm')
            gdf_merged = pd.merge(df_props, parishes_gdf_in[cols_to_merge], left_on='Parish_join_key', right_on='name_join_key', how='left')
            gdf_matched = gdf_merged[gdf_merged['geometry'].notna()].copy()
            if gdf_matched.empty: self._capture_print("CRIT: No properties could be matched to parish geometries."); return gpd.GeoDataFrame()
            gdf_matched = gpd.GeoDataFrame(gdf_matched, geometry='geometry', crs=parishes_gdf_in.crs or "EPSG:4326")
            if gdf_matched.crs is None: gdf_matched.set_crs("EPSG:4326", inplace=True)
            projected_crs = "EPSG:32620"
            try:
                gdf_matched_projected = gdf_matched.to_crs(projected_crs)
                gdf_matched['geometry'] = gdf_matched_projected.geometry.centroid.to_crs(gdf_matched.crs)
            except Exception as e:
                self._capture_print(f"WARN: Error projecting for centroid, using geographic CRS: {e}")
                gdf_matched['geometry'] = gdf_matched.geometry.centroid
            return gdf_matched.drop(columns=['name_join_key', 'Parish_join_key'], errors='ignore')
        except Exception as e: self._capture_print(f"Error during geocoding: {str(e)}\n{traceback.format_exc()}"); return gpd.GeoDataFrame()

    def analyze_properties_internal(self, prop_gdf_in, beaches_in, tourism_points_in):
        prop_gdf = prop_gdf_in.copy();
        beaches_gdf = beaches_in.copy() if beaches_in is not None and not beaches_in.empty else gpd.GeoDataFrame();
        tourism_gdf = tourism_points_in.copy() if tourism_points_in is not None and not tourism_points_in.empty else gpd.GeoDataFrame()
        try:
            if prop_gdf.empty: return gpd.GeoDataFrame()
            if prop_gdf.crs is None: prop_gdf.set_crs("EPSG:4326",inplace=True)
            analysis_crs="EPSG:32620";
            prop_proj = prop_gdf.to_crs(analysis_crs)
            prop_proj['beach_dist_m']=np.nan; prop_proj['tourism_count_2km']=0
            if not beaches_gdf.empty:
                if beaches_gdf.crs is None: beaches_gdf.set_crs("EPSG:4326",inplace=True)
                beaches_proj = beaches_gdf.to_crs(analysis_crs)[beaches_gdf.geometry.notna()]
                if not beaches_proj.empty:
                    coords=[];
                    for geom in beaches_proj.geometry.explode(index_parts=False):
                        if geom.geom_type=='Point': coords.append((geom.x,geom.y))
                        elif geom.geom_type=='LineString': coords.extend(list(geom.coords))
                        elif geom.geom_type=='Polygon': coords.extend(list(geom.exterior.coords)); [coords.extend(list(i.coords)) for i in geom.interiors]
                    if coords:
                        prop_points_for_tree = prop_proj.geometry.apply(lambda g: (g.x, g.y))
                        distances, _ = cKDTree(list(set(coords))).query(list(prop_points_for_tree),k=1)
                        prop_proj['beach_dist_m'] = distances
            if not tourism_gdf.empty:
                if tourism_gdf.crs is None: tourism_gdf.set_crs("EPSG:4326",inplace=True)
                tourism_proj = tourism_gdf.to_crs(analysis_crs)[tourism_gdf.geometry.notna()]
                if not tourism_proj.empty:
                    tourism_pts_for_sjoin = tourism_proj.copy()
                    if not tourism_pts_for_sjoin.geometry.geom_type.eq('Point').all(): tourism_pts_for_sjoin['geometry'] = tourism_pts_for_sjoin.geometry.representative_point()
                    if not tourism_pts_for_sjoin.empty and not prop_proj.empty:
                        if prop_proj.crs != tourism_pts_for_sjoin.crs: tourism_pts_for_sjoin = tourism_pts_for_sjoin.to_crs(prop_proj.crs)
                        buffers=gpd.GeoDataFrame(geometry=prop_proj.geometry.buffer(2000),crs=prop_proj.crs,index=prop_proj.index)
                        joined=gpd.sjoin(buffers,tourism_pts_for_sjoin,how="left",predicate="intersects")
                        prop_proj['tourism_count_2km']=joined.groupby(joined.index).size().reindex(prop_proj.index).fillna(0).astype(int)
            return prop_proj.to_crs("EPSG:4326")
        except Exception as e: self._capture_print(f"Error spatial analysis: {e}\n{traceback.format_exc()}"); return prop_gdf_in.to_crs("EPSG:4326") if prop_gdf_in.crs != "EPSG:4326" else prop_gdf_in

    def get_parish_for_coords(self, lat, lon):
        if not hasattr(self, 'parishes') or self.parishes.empty or pd.isna(lat) or pd.isna(lon):
            return "Parish data not available or coordinates invalid."
        try:
            point_geom = Point(lon, lat)
            point_gdf = gpd.GeoDataFrame([1], geometry=[point_geom], crs="EPSG:4326")
            parishes_4326 = self.parishes.to_crs("EPSG:4326")
            parishes_to_join = parishes_4326.copy()
            mask_name = parishes_to_join['name'].fillna('').astype(str).str.lower() != 'barbados' if 'name' in parishes_to_join.columns else pd.Series([True]*len(parishes_to_join), index=parishes_to_join.index)
            parishes_to_join = parishes_to_join[mask_name]
            if parishes_to_join.empty and not parishes_4326.empty:
                self._capture_print("WARN: Filtering 'Barbados' removed all parishes. Using unfiltered."); parishes_to_join = parishes_4326
            elif parishes_to_join.empty: return "No parish data to perform lookup."
            joined_gdf = gpd.sjoin(point_gdf, parishes_to_join, how="inner", predicate="within")
            if not joined_gdf.empty:
                for col_pref in ['name', 'OSM_Parish_Name']:
                    if col_pref in joined_gdf.columns:
                        for parish_candidate in joined_gdf[col_pref].unique():
                            if pd.notna(parish_candidate) and str(parish_candidate).strip().lower() not in ['', 'barbados', 'parish name n/a', 'unknown parish', 'unnamed parish']:
                                return str(parish_candidate)
                return joined_gdf['name'].iloc[0] if 'name' in joined_gdf.columns and pd.notna(joined_gdf['name'].iloc[0]) else "Parish name not identified"
            return "Not within a known parish boundary."
        except Exception as e: self._capture_print(f"Error finding parish: {e}\n{traceback.format_exc()}"); return "Error during parish lookup."

    def create_map_object_streamlit(self, center_lat=None, center_lon=None, zoom=None):
        if (not hasattr(self, 'analyzed_properties') or self.analyzed_properties.empty) and \
           (not hasattr(self, 'parishes') or self.parishes.empty):
            self._capture_print("Cannot create map: Missing analyzed properties or parishes.")
            return None
        analyzed_gdf = self.analyzed_properties.copy(); all_parishes_gdf = self.parishes.copy()
        feature_polygons = self.feature_polygons.copy() if hasattr(self, 'feature_polygons') else gpd.GeoDataFrame()
        tourism_points = self.tourism_points.copy() if hasattr(self, 'tourism_points') else gpd.GeoDataFrame()
        schools = self.schools.copy() if hasattr(self, 'schools') else gpd.GeoDataFrame()
        supermarkets = self.supermarkets.copy() if hasattr(self, 'supermarkets') else gpd.GeoDataFrame()
        roads = self.roads.copy() if hasattr(self, 'roads') else gpd.GeoDataFrame()
        parish_summary = self.parish_summary_df.copy() if hasattr(self, 'parish_summary_df') and not self.parish_summary_df.empty else pd.DataFrame()
        map_lat = center_lat if center_lat is not None else 13.1939
        map_lon = center_lon if center_lon is not None else -59.5432
        map_zoom = zoom if zoom is not None else 10
        m = folium.Map(location=[map_lat, map_lon], zoom_start=map_zoom, tiles="Stamen Terrain", attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.')
        folium.TileLayer("OpenTopoMap", name="Topographic Detail", show=False, attr='Map data: &copy; OpenStreetMap contributors, SRTM | Map style: &copy; OpenTopoMap (CC-BY-SA)').add_to(m)
        folium.TileLayer("CartoDB positron", name="Light Base Map", show=False, attr='&copy; OpenStreetMap contributors &copy; CARTO').add_to(m)
        if not all_parishes_gdf.empty and 'name' in all_parishes_gdf.columns and 'geometry' in all_parishes_gdf.columns:
            parishes_4326 = all_parishes_gdf.to_crs("EPSG:4326")[all_parishes_gdf.geometry.is_valid & all_parishes_gdf.geometry.notna()]
            if not parishes_4326.empty:
                parish_boundary_layer = folium.FeatureGroup(name="Parish Boundaries & Centers", show=True).add_to(m)
                folium.GeoJson(parishes_4326,style_function=lambda x: {'fillColor':'#D3D3D3','color':'#333333','weight':1.5,'fillOpacity':0.3},
                                       highlight_function=lambda x: {'weight':3, 'color':'#555555', 'fillOpacity':0.5},
                                       tooltip=folium.GeoJsonTooltip(fields=['name'], aliases=['<b>Parish:</b>'], style=("background-color:white;color:black;font-family:Arial;font-size:12px;padding:5px;border-radius:3px;box-shadow:3px 3px 5px grey;"),sticky=True)
                                     ).add_to(parish_boundary_layer)
                for _, parish_row in parishes_4326.iterrows():
                    name_display = parish_row['name']
                    try:
                        if parish_row.geometry.is_empty or not parish_row.geometry.is_valid: continue
                        centroid = parish_row.geometry.centroid;
                        if centroid.is_empty: continue
                        folium.CircleMarker(location=[centroid.y, centroid.x], radius=3, color='#4A4A4A', weight=1, fill=True, fill_color='#808080', fill_opacity=0.5, tooltip=f"<b>{name_display}</b> (Center)").add_to(parish_boundary_layer)
                    except Exception: pass
        if not feature_polygons.empty and 'name' in feature_polygons.columns and 'geometry' in feature_polygons.columns:
            feature_polygons_4326 = feature_polygons.to_crs("EPSG:4326")[feature_polygons.geometry.is_valid & feature_polygons.geometry.notna() & feature_polygons['name'].notna()]
            if not feature_polygons_4326.empty:
                def get_feature_style(props):
                    style = {'fillOpacity': 0.4, 'weight': 1.5, 'color': 'grey'}; leisure = props.get('leisure'); amenity = props.get('amenity')
                    if leisure == 'park': style.update({'fillColor': '#86C166', 'color': '#5E8C4A'})
                    return style
                feature_layer = folium.FeatureGroup(name="Key Land Features", show=False).add_to(m)
                for _, row in feature_polygons_4326.iterrows():
                    single_feature_gdf = gpd.GeoDataFrame([row], crs=feature_polygons_4326.crs)
                    folium.GeoJson(single_feature_gdf, style_function=lambda x: get_feature_style(x['properties']),
                                   highlight_function=lambda x: {'weight': 3, 'color': 'black', 'fillOpacity': 0.6},
                                   tooltip=folium.GeoJsonTooltip(fields=['name'], aliases=['<b>Name:</b>'], style=("background-color:white;color:black;font-family:Arial;font-size:12px;padding:5px;border-radius:3px;box-shadow:3px 3px 5px grey;"),sticky=False)).add_to(feature_layer)
        if not tourism_points.empty and 'name' in tourism_points.columns and 'geometry' in tourism_points.columns:
            tourism_4326 = tourism_points.to_crs("EPSG:4326")[tourism_points.geometry.is_valid & tourism_points.geometry.notna()]
            if not tourism_4326.empty:
                landmark_pts_layer = folium.FeatureGroup(name="Points of Interest (Tourism)", show=False).add_to(m)
                for _, pt_row in tourism_4326.iterrows():
                    name = pt_row.get('name','POI'); pt_geom=pt_row.geometry;
                    if pt_geom.geom_type!='Point': pt_geom=pt_geom.representative_point()
                    if pt_geom.is_empty: continue
                    folium.Marker([pt_geom.y,pt_geom.x],tooltip=f"<b>{name}</b><br><small>{pt_row.get('tourism','N/A').replace('_',' ').title()}</small>", icon=folium.Icon(color='purple',icon='star',prefix='fa')).add_to(landmark_pts_layer)
        if not schools.empty and 'geometry' in schools.columns:
            schools_4326 = schools.to_crs("EPSG:4326")[schools.geometry.is_valid & schools.geometry.notna()]
            if not schools_4326.empty:
                schools_layer = folium.FeatureGroup(name="Schools", show=False).add_to(m)
                for _, pt_row in schools_4326.iterrows():
                    name = pt_row.get('name', 'School'); pt_geom = pt_row.geometry
                    if pt_geom.geom_type != 'Point': pt_geom = pt_geom.representative_point()
                    if pt_geom.is_empty: continue
                    folium.Marker([pt_geom.y, pt_geom.x], tooltip=f"<b>{name}</b><br><small>School</small>", icon=folium.Icon(color='blue', icon='graduation-cap', prefix='fa')).add_to(schools_layer)
        if not supermarkets.empty and 'geometry' in supermarkets.columns:
            supermarkets_4326 = supermarkets.to_crs("EPSG:4326")[supermarkets.geometry.is_valid & supermarkets.geometry.notna()]
            if not supermarkets_4326.empty:
                supermarkets_layer = folium.FeatureGroup(name="Supermarkets", show=False).add_to(m)
                for _, pt_row in supermarkets_4326.iterrows():
                    name = pt_row.get('name', 'Supermarket'); pt_geom = pt_row.geometry
                    if pt_geom.geom_type != 'Point': pt_geom = pt_geom.representative_point()
                    if pt_geom.is_empty: continue
                    folium.Marker([pt_geom.y, pt_geom.x], tooltip=f"<b>{name}</b><br><small>Supermarket</small>", icon=folium.Icon(color='green', icon='shopping-cart', prefix='fa')).add_to(supermarkets_layer)
        if not roads.empty and 'geometry' in roads.columns:
            roads_4326 = roads.to_crs("EPSG:4326")[roads.geometry.is_valid & roads.geometry.notna()]
            if not roads_4326.empty:
                road_network_layer = folium.FeatureGroup(name="Road Network", show=False).add_to(m)
                road_style = {'color': '#708090', 'weight': 2, 'opacity': 0.7}
                folium.GeoJson(roads_4326, name="Roads", style_function=lambda x: road_style,
                               tooltip=folium.features.GeoJsonTooltip(fields=['name', 'highway'], aliases=['Road Name:', 'Type:'], sticky=False, style=("background-color: white; color: black; font-family: arial; font-size: 12px; padding: 5px;"))
                             ).add_to(road_network_layer)
        if not parish_summary.empty:
            parish_summary_layer = folium.FeatureGroup(name="Property Summaries by Parish", show=True).add_to(m)
            summary_display_name_col = 'OSM_Parish_Name' if 'OSM_Parish_Name' in parish_summary.columns else 'Parish'
            if summary_display_name_col not in parish_summary.columns and 'name' in parish_summary.columns:
                summary_display_name_col = 'name'
            for _, p_row in parish_summary.iterrows():
                if pd.isna(p_row['latitude']) or pd.isna(p_row['longitude']): continue
                loc=[p_row['latitude'],p_row['longitude']]; fill_color='#FF4500'; radius=max(8,min(10+(p_row['total_properties']/12),35))
                beach_km_str = f"{(p_row['avg_beach_dist_m']/1000):.1f} km" if pd.notna(p_row['avg_beach_dist_m']) else 'N/A'
                avg_tour_str = f"{p_row['avg_tourism_count_2km']:.1f}" if pd.notna(p_row['avg_tourism_count_2km']) else "N/A"
                avg_sz_str = f"{p_row['avg_size_sqft']:,.0f} sq ft" if pd.notna(p_row['avg_size_sqft']) else "N/A"
                area_val = p_row.get('area_sqkm')
                area_str = f"{area_val:.1f} sq km" if pd.notna(area_val) and area_val > 0 else "N/A"
                prop_density_str = f"{p_row.get('property_density',0):.1f} props/km²" if pd.notna(p_row.get('property_density')) else "N/A"
                total_road_len_str = f"{p_row.get('total_road_length_km', 0):.1f} km" if pd.notna(p_row.get('total_road_length_km')) else "N/A"
                road_density_str = f"{p_row.get('road_density_km_sqkm', 0):.2f} km/km²" if pd.notna(p_row.get('road_density_km_sqkm')) else "N/A"
                popup_html = f"""<div style="width:320px; font-family: Arial, sans-serif; font-size: 13px; line-height: 1.4;"><h4 style="margin: 5px 0 10px 0; padding-bottom: 5px; border-bottom: 1px solid #ddd; color: #005A9C;">{p_row.get(summary_display_name_col, 'Parish Summary')} Summary</h4><table style="width: 100%; border-collapse: collapse;"><tr style="background-color: #f9f9f9;"><td style="padding: 4px; font-weight: bold;">Total Properties:</td><td style="padding: 4px; text-align: right;">{p_row['total_properties']}</td></tr><tr><td style="padding: 4px; font-weight: bold;">Parish Area:</td><td style="padding: 4px; text-align: right;">{area_str}</td></tr><tr style="background-color: #f9f9f9;"><td style="padding: 4px; font-weight: bold;">Property Density:</td><td style="padding: 4px; text-align: right;">{prop_density_str}</td></tr><tr><td style="padding: 4px; font-weight: bold;">For Sale:</td><td style="padding: 4px; text-align: right;">{p_row['total_for_sale']}</td></tr><tr style="background-color: #f9f9f9;"><td style="padding: 4px; font-weight: bold; padding-left: 15px;">↳ Residential:</td><td style="padding: 4px; text-align: right;">{p_row.get('for_sale_res_count',0)}</td></tr><tr><td style="padding: 4px; font-weight: bold; padding-left: 15px;">↳ Commercial:</td><td style="padding: 4px; text-align: right;">{p_row.get('for_sale_com_count',0)}</td></tr><tr style="background-color: #f9f9f9;"><td style="padding: 4px; font-weight: bold; padding-left: 15px;">↳ Land:</td><td style="padding: 4px; text-align: right;">{p_row.get('for_sale_land_count',0)}</td></tr><tr><td style="padding: 4px; font-weight: bold; padding-left: 15px;">↳ Other:</td><td style="padding: 4px; text-align: right;">{p_row.get('for_sale_oth_std_count',0)}</td></tr><tr><td style="padding: 4px; font-weight: bold;">For Rent:</td><td style="padding: 4px; text-align: right;">{p_row['total_for_rent']}</td></tr><tr style="background-color: #f9f9f9;"><td style="padding: 4px; font-weight: bold; padding-left: 15px;">↳ Residential:</td><td style="padding: 4px; text-align: right;">{p_row.get('for_rent_res_count',0)}</td></tr><tr><td style="padding: 4px; font-weight: bold; padding-left: 15px;">↳ Commercial:</td><td style="padding: 4px; text-align: right;">{p_row.get('for_rent_com_count',0)}</td></tr><tr style="background-color: #f9f9f9;"><td style="padding: 4px; font-weight: bold; padding-left: 15px;">↳ Land:</td><td style="padding: 4px; text-align: right;">{p_row.get('for_rent_land_count',0)}</td></tr><tr><td style="padding: 4px; font-weight: bold; padding-left: 15px;">↳ Other:</td><td style="padding: 4px; text-align: right;">{p_row.get('for_rent_oth_std_count',0)}</td></tr><tr style="background-color: #f9f9f9;"><td style="padding: 4px; font-weight: bold;">Avg. Beach Dist:</td><td style="padding: 4px; text-align: right;">{beach_km_str}</td></tr><tr><td style="padding: 4px; font-weight: bold;">Avg. Attractions (2km):</td><td style="padding: 4px; text-align: right;">{avg_tour_str}</td></tr><tr style="background-color: #f9f9f9;"><td style="padding: 4px; font-weight: bold;">Avg. Size:</td><td style="padding: 4px; text-align: right;">{avg_sz_str}</td></tr><tr><td style="padding: 4px; font-weight: bold;">Total Road Length:</td><td style="padding: 4px; text-align: right;">{total_road_len_str}</td></tr><tr style="background-color: #f9f9f9;"><td style="padding: 4px; font-weight: bold;">Road Density:</td><td style="padding: 4px; text-align: right;">{road_density_str}</td></tr></table></div>"""
                folium.CircleMarker(loc, radius=radius, color='#000000', weight=2, fill=True, fill_color=fill_color, fill_opacity=0.7, popup=folium.Popup(popup_html, max_width=350), tooltip=f"<b>{p_row.get(summary_display_name_col, 'Parish')}</b><br>Total Data Properties: {p_row['total_properties']}").add_to(parish_summary_layer)
        if center_lat is not None and center_lon is not None and st.session_state.get('location_explicitly_set', False):
            folium.Marker(location=[center_lat, center_lon],popup=f"Located: {center_lat:.4f}, {center_lon:.4f}",tooltip="Your Location",icon=folium.Icon(color="red", icon="star")).add_to(m)
        folium.LayerControl().add_to(m)
        return m

    def create_visualizations_internal(self, analyzed_gdf_in, all_parishes_gdf_in, feature_polygons_in, tourism_points_in):
        analyzed_gdf = analyzed_gdf_in.copy() if analyzed_gdf_in is not None and not analyzed_gdf_in.empty else gpd.GeoDataFrame()
        try:
            if analyzed_gdf.empty:
                self._capture_print("No analyzed data to create visualizations.")
                self.map_html_content, self.parish_summary_df = None, pd.DataFrame()
                return

            if 'Property Type Standardized' not in analyzed_gdf.columns:
                analyzed_gdf['Property Type Standardized'] = standardize_property_type_static(analyzed_gdf.get('Property Type', pd.Series(dtype=str)))
            analyzed_gdf['Category'] = analyzed_gdf.get('Category', pd.Series(dtype=str)).fillna('Unknown').astype(str).str.strip()
            analyzed_gdf['Property Type Standardized'] = analyzed_gdf.get('Property Type Standardized', pd.Series(dtype=str)).fillna('Other').astype(str).str.strip()
            
            group_by_col_summary = 'OSM_Parish_Name'
            if group_by_col_summary not in analyzed_gdf.columns or analyzed_gdf[group_by_col_summary].isnull().all():
                   group_by_col_summary = 'Parish' 
            
            if group_by_col_summary not in analyzed_gdf.columns or analyzed_gdf[group_by_col_summary].isnull().all(): 
                   self._capture_print(f"CRITICAL: Cannot find a suitable parish grouping column for summary ({group_by_col_summary}). Analyzed GDF columns: {analyzed_gdf.columns}"); 
                   self.parish_summary_df = pd.DataFrame();
            else:
                analyzed_gdf[group_by_col_summary] = analyzed_gdf[group_by_col_summary].astype(str)
                def summarize_parish(group):
                    data = {'total_properties': int(len(group))}
                    is_sale = group['Category'].str.upper() != 'FOR RENT'; is_rent = group['Category'].str.upper() == 'FOR RENT'
                    prop_type = group['Property Type Standardized']
                    for p_type, p_label in [('Residential', 'res'), ('Commercial', 'com'), ('Land', 'land'), ('Other', 'oth_std')]:
                        is_curr_type = prop_type == p_type
                        data[f'for_sale_{p_label.lower()}_count'] = int((is_sale & is_curr_type).sum())
                        data[f'for_rent_{p_label.lower()}_count'] = int((is_rent & is_curr_type).sum())
                    data.update({'total_for_sale': int(is_sale.sum()), 'total_for_rent': int(is_rent.sum()),
                                 'avg_beach_dist_m': group['beach_dist_m'].mean(), 
                                 'avg_tourism_count_2km': group['tourism_count_2km'].mean(),
                                 'avg_size_sqft': group['Size_sqft'].mean(),
                                 'avg_bedrooms': group['Bedrooms'].mean() if 'Bedrooms' in group and group['Bedrooms'].notna().any() else np.nan, 
                                 'avg_price': group['Price'].mean() if 'Price' in group and group['Price'].notna().any() else np.nan, 
                                 'min_price': group['Price'].min() if 'Price' in group and group['Price'].notna().any() else np.nan, 
                                 'max_price': group['Price'].max() if 'Price' in group and group['Price'].notna().any() else np.nan, 
                                 'area_sqkm': group['area_sqkm'].iloc[0] if 'area_sqkm' in group.columns and not group['area_sqkm'].empty and pd.notna(group['area_sqkm'].iloc[0]) else np.nan})
                    if not group.empty and 'geometry' in group.columns and group['geometry'].iloc[0] is not None:
                        first_geom = group['geometry'].iloc[0]; data['latitude'], data['longitude'] = (first_geom.y, first_geom.x) if hasattr(first_geom, 'x') else (np.nan, np.nan)
                    else: data['latitude'], data['longitude'] = np.nan, np.nan
                    return pd.Series(data)
                self.parish_summary_df = analyzed_gdf.groupby(group_by_col_summary, dropna=False).apply(summarize_parish).reset_index()

                if not self.parish_summary_df.empty:
                    if 'area_sqkm' in self.parish_summary_df.columns and 'total_properties' in self.parish_summary_df.columns:
                        self.parish_summary_df['property_density'] = self.parish_summary_df.apply(
                            lambda row: (row['total_properties'] / row['area_sqkm']) 
                                        if pd.notna(row['area_sqkm']) and row['area_sqkm'] > 0 and pd.notna(row['total_properties']) and row['total_properties'] > 0
                                        else np.nan,
                            axis=1
                        )
                        self._capture_print("Calculated property_density for parish summary.")
                    else:
                        self.parish_summary_df['property_density'] = np.nan
                        self._capture_print("WARN: Could not calculate property_density due to missing 'area_sqkm' or 'total_properties' in parish_summary_df.")

            parish_road_stats_dict = {}
            if hasattr(self, 'roads') and not self.roads.empty and hasattr(self, 'parishes') and not self.parishes.empty and 'geometry' in self.roads.columns and 'geometry' in self.parishes.columns:
                self._capture_print("Calculating road statistics per parish...")
                try:
                    parishes_proj_for_roads = self.parishes.to_crs("EPSG:32620")
                    roads_proj_for_calc = self.roads.to_crs("EPSG:32620")
                    
                    if 'length' in roads_proj_for_calc.columns and 'length_m' not in roads_proj_for_calc.columns:
                         roads_proj_for_calc.rename(columns={'length': 'length_m'}, inplace=True) 
                    elif 'geometry' in roads_proj_for_calc and 'length_m' not in roads_proj_for_calc.columns:
                         roads_proj_for_calc['length_m'] = roads_proj_for_calc.geometry.length 
                    elif 'length_m' not in roads_proj_for_calc.columns: 
                         raise ValueError("Roads GDF missing geometry or a way to calculate length.")

                    for _, parish_row in parishes_proj_for_roads.iterrows():
                        parish_name_key = parish_row.get('name') 
                        if pd.isna(parish_name_key) or parish_name_key == 'Unknown Parish': continue
                        
                        parish_geom = parish_row.geometry
                        parish_summary_entry = self.parish_summary_df[self.parish_summary_df[group_by_col_summary] == parish_name_key]
                        parish_area_sqkm = parish_summary_entry['area_sqkm'].iloc[0] if not parish_summary_entry.empty and 'area_sqkm' in parish_summary_entry and pd.notna(parish_summary_entry['area_sqkm'].iloc[0]) else np.nan

                        intersecting_roads = roads_proj_for_calc[roads_proj_for_calc.geometry.intersects(parish_geom)]
                        total_length_m = 0.0
                        if not intersecting_roads.empty:
                            clipped_roads = gpd.clip(intersecting_roads, parish_geom, keep_geom_type=False)
                            if not clipped_roads.empty and 'length_m' in clipped_roads.columns : 
                                total_length_m = clipped_roads['length_m'].sum()
                            elif not clipped_roads.empty: 
                                total_length_m = clipped_roads.geometry.length.sum()

                        total_length_km = total_length_m / 1000.0
                        road_density_km_sqkm = (total_length_km / parish_area_sqkm) if pd.notna(parish_area_sqkm) and parish_area_sqkm > 0 and total_length_km > 0 else 0.0
                        parish_road_stats_dict[parish_name_key] = {'total_road_length_km': total_length_km, 'road_density_km_sqkm': road_density_km_sqkm}
                    self._capture_print(f"Calculated road stats for {len(parish_road_stats_dict)} unique parish names.")
                except Exception as e_road_stats: self._capture_print(f"Error calculating parish road stats: {e_road_stats}\n{traceback.format_exc()}")

            if parish_road_stats_dict and not self.parish_summary_df.empty:
                road_stats_df = pd.DataFrame.from_dict(parish_road_stats_dict, orient='index').reset_index()
                road_stats_df.rename(columns={'index': group_by_col_summary}, inplace=True)
                try:
                    self.parish_summary_df[group_by_col_summary] = self.parish_summary_df[group_by_col_summary].astype(str)
                    road_stats_df[group_by_col_summary] = road_stats_df[group_by_col_summary].astype(str)
                except Exception as e_type: self._capture_print(f"Type conversion error for merge: {e_type}")
                self.parish_summary_df = pd.merge(self.parish_summary_df, road_stats_df, on=group_by_col_summary, how='left')
                self.parish_summary_df[['total_road_length_km', 'road_density_km_sqkm']] = self.parish_summary_df[['total_road_length_km', 'road_density_km_sqkm']].fillna(0)
                self._capture_print("Merged road stats into parish summary.")
            elif not self.parish_summary_df.empty: 
                   self.parish_summary_df['total_road_length_km'] = self.parish_summary_df.get('total_road_length_km', 0.0)
                   self.parish_summary_df['road_density_km_sqkm'] = self.parish_summary_df.get('road_density_km_sqkm', 0.0)

            m = self.create_map_object_streamlit()
            if m: self.map_html_content = m.get_root().render(); self._capture_print("\nInteractive map data generated.")
            else: self.map_html_content = None; self._capture_print("\nCould not generate map - data missing?")

            fig_chart, ax_chart = plt.subplots(figsize=(10,6))
            chart_group_col = group_by_col_summary 
            unique_parishes_for_chart = sorted(analyzed_gdf[chart_group_col].unique()) if chart_group_col and not analyzed_gdf.empty and chart_group_col in analyzed_gdf.columns else []
            if unique_parishes_for_chart:
                cmap_name = 'viridis' 
                try:
                    cmap = plt.cm.get_cmap(cmap_name, len(unique_parishes_for_chart))
                except ValueError: 
                    cmap = plt.cm.get_cmap(cmap_name)

                colors = [cmap(i) for i in range(len(unique_parishes_for_chart))]
                for i, p_name_chart in enumerate(unique_parishes_for_chart):
                    p_data = analyzed_gdf[analyzed_gdf[chart_group_col]==p_name_chart].dropna(subset=['beach_dist_m','Price'])
                    if not p_data.empty: ax_chart.scatter(p_data['beach_dist_m']/1000, p_data['Price'], label=p_name_chart, alpha=0.65, s=45, color=colors[i])
            ax_chart.set_title("Property Prices vs. Beach Distance by Parish",fontsize=14); ax_chart.set_xlabel("Distance to Nearest Beach (km)",fontsize=12); ax_chart.set_ylabel("Price (USD)",fontsize=12)
            if unique_parishes_for_chart: ax_chart.legend(title="Parish", fontsize='small', title_fontsize='medium')
            ax_chart.grid(True, linestyle=':', alpha=0.7); ax_chart.ticklabel_format(style='plain',axis='y'); fig_chart.tight_layout()
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile_chart: self.chart_path = tmpfile_chart.name
            fig_chart.savefig(self.chart_path,dpi=100,bbox_inches='tight'); plt.close(fig_chart); self._capture_print(f"Price vs. Beach Distance chart generated: {self.chart_path}");

        except Exception as e: self._capture_print(f"Error visualizations: {e}\n{traceback.format_exc()}");


    def display_stats_internal(self):
        self.stats_data_for_streamlit = []
        stats_gdf = self.analyzed_properties
        try:
            if stats_gdf.empty: self.stats_data_for_streamlit=["No data available for statistics."]
            else:
                group_col = 'OSM_Parish_Name' if 'OSM_Parish_Name' in stats_gdf.columns else 'Parish'
                num_analyzed = len(stats_gdf)
                num_parishes = stats_gdf[group_col].nunique() if group_col in stats_gdf.columns and stats_gdf[group_col].notna().any() else 0
                avg_beach_dist_val = stats_gdf['beach_dist_m'].mean() if 'beach_dist_m' in stats_gdf and stats_gdf['beach_dist_m'].notna().any() else np.nan
                avg_beach_dist_str = f"{(avg_beach_dist_val/1000):.1f} km" if pd.notna(avg_beach_dist_val) else 'N/A'
                h_pr = f"${stats_gdf['Price'].max():,.0f}" if 'Price' in stats_gdf and stats_gdf['Price'].notna().any() else 'N/A'; l_pr = f"${stats_gdf['Price'].min():,.0f}" if 'Price' in stats_gdf and stats_gdf['Price'].notna().any() else 'N/A'
                avg_tr_val = stats_gdf['tourism_count_2km'].mean() if 'tourism_count_2km' in stats_gdf and stats_gdf['tourism_count_2km'].notna().any() else np.nan
                avg_tr_str = f"{avg_tr_val:.1f}" if pd.notna(avg_tr_val) else 'N/A'
                s_cnt = (stats_gdf['Category'].astype(str).str.strip().str.upper()!='FOR RENT').sum() if 'Category' in stats_gdf else 0; r_cnt = (stats_gdf['Category'].astype(str).str.strip().str.upper()=='FOR RENT').sum() if 'Category' in stats_gdf else 0
                avg_sz_val = stats_gdf['Size_sqft'].mean() if 'Size_sqft' in stats_gdf and stats_gdf['Size_sqft'].notna().any() else np.nan
                avg_sz_str = f"{avg_sz_val:,.0f} sq ft" if pd.notna(avg_sz_val) else "N/A"
                
                self.stats_data_for_streamlit = [
                    f"Total Properties Analyzed: {num_analyzed}",
                    f"Unique Parishes in Data: {num_parishes}",
                    f"Avg Beach Dist: {avg_beach_dist_str}",
                    f"Highest Property Price: {h_pr}",
                    f"Lowest Property Price: {l_pr}",
                    f"Avg Attractions (2km): {avg_tr_str}",
                    f"Properties For Sale: {s_cnt}",
                    f"Properties For Rent: {r_cnt}",
                    f"Average Property Size: {avg_sz_str}"
                ]

                if hasattr(self, 'roads') and not self.roads.empty:
                    road_length_col = None
                    if 'length_m' in self.roads.columns:
                        road_length_col = 'length_m'
                    elif 'length' in self.roads.columns:
                        road_length_col = 'length'
                    
                    if road_length_col and self.roads[road_length_col].notna().any():
                        total_island_road_length_m = self.roads[road_length_col].sum()
                        total_island_road_length_km = total_island_road_length_m / 1000.0
                        self.stats_data_for_streamlit.append(f"Total Road Network Length (Island): {total_island_road_length_km:,.1f} km")
                        if hasattr(self, 'total_island_area_sqkm') and pd.notna(self.total_island_area_sqkm) and self.total_island_area_sqkm > 0 and total_island_road_length_km > 0:
                            overall_road_density = total_island_road_length_km / self.total_island_area_sqkm
                            self.stats_data_for_streamlit.append(f"Overall Road Density (Island): {overall_road_density:.2f} km/km²")
                        else:
                            self.stats_data_for_streamlit.append("Overall Road Density (Island): N/A (Island area or road length unknown/zero)")
                    else:
                        self.stats_data_for_streamlit.append("Total Road Network Length (Island): N/A (Road length data missing or invalid in roads GDF)")
                        self.stats_data_for_streamlit.append("Overall Road Density (Island): N/A")
                else:
                    self.stats_data_for_streamlit.append("Total Road Network Length (Island): N/A (Road data not loaded)")
                    self.stats_data_for_streamlit.append("Overall Road Density (Island): N/A")


        except Exception as e: self._capture_print(f"Error display_stats: {e}\n{traceback.format_exc()}"); self.stats_data_for_streamlit.append("Error loading stats.")

    def get_export_dataframe(self):
        if hasattr(self,'analyzed_properties') and not self.analyzed_properties.empty:
            df_to_export = self.analyzed_properties.copy()
            if 'Property Type' not in df_to_export.columns and 'Property Type Standardized' in df_to_export.columns:
                if 'Property Type' in self.analyzed_properties.columns:
                    df_to_export['Property Type_Original'] = self.analyzed_properties['Property Type']
            if 'geometry' in df_to_export.columns and not df_to_export.geometry.empty:
                if df_to_export.crs and df_to_export.crs != "EPSG:4326":
                        df_to_export = df_to_export.to_crs("EPSG:4326")
                df_to_export['latitude'] = df_to_export.geometry.y
                df_to_export['longitude'] = df_to_export.geometry.x
            cols_to_drop = ['geometry', 'original_index', 'Parish_join_key']
            actual_cols_to_drop = [col for col in cols_to_drop if col in df_to_export.columns]
            if actual_cols_to_drop:
                df_to_export = df_to_export.drop(columns=actual_cols_to_drop)
            return df_to_export
        return pd.DataFrame()

    def generate_ai_parish_road_assessment(self):
        if not OPENAI_API_KEY:
            self.ai_parish_road_assessment_text = "AI features disabled: OpenAI API key not configured."
            return self.ai_parish_road_assessment_text
            
        if not hasattr(self, 'parish_summary_df') or self.parish_summary_df.empty:
            self.ai_parish_road_assessment_text = "Parish summary data not available for AI road assessment. Please run analysis first."
            return self.ai_parish_road_assessment_text
            
        required_road_cols = ['total_road_length_km', 'road_density_km_sqkm', 'area_sqkm', 'total_properties']
        name_col_options = ['OSM_Parish_Name', 'Parish', 'name'] 
        name_col = next((col for col in name_col_options if col in self.parish_summary_df.columns), None)

        if not name_col:
            self.ai_parish_road_assessment_text = "Suitable parish name column not found in summary for road assessment."
            return self.ai_parish_road_assessment_text

        missing_cols_messages = []
        for col in required_road_cols:
            if col not in self.parish_summary_df.columns:
                missing_cols_messages.append(f"'{col}' column missing")
            elif self.parish_summary_df[col].isnull().all():
                 missing_cols_messages.append(f"'{col}' column has no data")
        
        if missing_cols_messages:
            self.ai_parish_road_assessment_text = f"Road network statistics ({', '.join(missing_cols_messages)}) are missing or empty in parish summary. Cannot perform AI road assessment."
            return self.ai_parish_road_assessment_text

        parish_data_for_ai = self.parish_summary_df[[name_col] + required_road_cols].copy()
        parish_data_for_ai.rename(columns={name_col: 'Parish Name'}, inplace=True)

        for col in required_road_cols: 
            if pd.api.types.is_numeric_dtype(parish_data_for_ai[col]):
                if col in ['total_road_length_km', 'area_sqkm']:
                    parish_data_for_ai[col] = parish_data_for_ai[col].apply(lambda x: f"{x:,.1f}" if pd.notna(x) else "N/A")
                elif col == 'road_density_km_sqkm': 
                    parish_data_for_ai[col] = parish_data_for_ai[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A")
                elif col == 'total_properties':
                     parish_data_for_ai[col] = parish_data_for_ai[col].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "N/A")


        data_summary_str = parish_data_for_ai.to_string(index=False, na_rep='N/A')
        max_prompt_table_len = 10000 
        if len(data_summary_str) > max_prompt_table_len:
            self._capture_print(f"WARN: Road assessment summary string for AI is too long ({len(data_summary_str)} chars), truncating.")
            data_summary_str = data_summary_str[:max_prompt_table_len] + "\n... (table truncated due to length)"

        prompt = f"""
You are a Senior Real Estate Strategist at Terra Caribbean, specializing in the impact of infrastructure on property markets in Barbados.
You have been provided with data summarizing road network characteristics for various parishes: Parish Name, Total Road Length (km), Road Density (km/km²), Parish Area (km²), and Total Properties (from an analyzed dataset in that parish). 
'N/A' means data is not available or zero.

Road Network Data by Parish:
{data_summary_str}

Your task is to analyze this road data and provide strategic insights specifically for Terra Caribbean's decision-makers. Focus on how the road network characteristics of each parish might affect property values, development potential, marketability, and overall real estate strategy for Terra Caribbean.

Please structure your analysis as follows:

1.  **Executive Summary (Key Road Network Implications for Terra Caribbean):**
    * Summarize 2-3 critical insights from the road data that directly impact Terra Caribbean's business strategies (e.g., areas with strong vs. weak accessibility impacting property appeal, development opportunities linked to good road infrastructure, markets potentially limited by current road networks).

2.  **Parish Road Network Profiles & Real Estate Impact on Terra Caribbean's Portfolio/Strategy:**
    * For each parish, or for groups of parishes with similar road characteristics:
        * Briefly describe its road network profile (e.g., "St. Michael shows high road density (X km/km²), indicating extensive urban connectivity").
        * **Crucially, analyze the direct implications for Terra Caribbean's business:**
            * **Property Values & Marketability:** How might the road network (density, total length relative to area) influence property prices and the ease of selling/renting properties Terra Caribbean might list or manage in this parish? (e.g., "High road density in Parish A likely supports higher property values and quicker sales due to excellent accessibility, benefiting Terra's listings there.")
            * **Development Suitability & Advisory:** From Terra Caribbean's perspective advising clients or considering its own development interests, does the road infrastructure suggest suitability for new residential or commercial development? Are there areas that might be underserved by roads, potentially limiting development appeal or requiring clients to factor in infrastructure costs? (e.g., "Parish B's lower road density, despite a large area, suggests that advising clients on development there should include considerations for accessibility challenges or potential for advocating/awaiting infrastructure upgrades.")
            * **Investment Attractiveness for Clients:** How does the road network contribute to or detract from the investment attractiveness of properties in this parish for different types of investors Terra Caribbean serves (e.g., residential, commercial, land speculation)?

3.  **Strategic Opportunities for Terra Caribbean Related to Road Infrastructure:**
    * Based *only* on the road data, identify any parishes where the road network might present specific opportunities for Terra Caribbean. Examples:
        * Highlighting well-connected areas in marketing materials for listings.
        * Advising developer clients on areas with robust existing infrastructure, potentially reducing their upfront costs.
        * Identifying areas where anticipated (though not in data) or lobbied-for road improvements could unlock significant property value, guiding client acquisition strategies.

4.  **Potential Challenges or Considerations for Terra Caribbean:**
    * Identify any parishes where road network characteristics might pose challenges for Terra Caribbean's operations or client advisory (e.g., difficulty in accessing properties for viewings/valuations, lower client interest in areas with poor road density, challenges in marketing development land in poorly connected zones).

**Important Guidelines:**
* **Terra Caribbean Focus:** All insights must be framed from the perspective of their relevance and impact on Terra Caribbean's real estate business operations, client advisory, and strategic interests.
* **Data-Driven:** Strictly base your analysis on the provided road network data. Do not use external knowledge about specific road projects or conditions.
* **Actionable Tone:** Insights should highlight areas for strategic focus, further investigation, or specific advisory points for Terra Caribbean.
* **Clarity and Conciseness:** Present information clearly and directly for a business audience.
"""
        try:
            self._capture_print(f"\nSending prompt for AI Parish Road (Strategic) Assessment. Data string length: {len(data_summary_str)}")
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125", 
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4, 
                max_tokens=1500 
            )
            self.ai_parish_road_assessment_text = response.choices[0].message.content
            return self.ai_parish_road_assessment_text
        except Exception as e:
            self._capture_print(f"AI Parish Road (Strategic) Assessment failed: {e}\n{traceback.format_exc()}")
            self.ai_parish_road_assessment_text = f"AI Parish Road (Strategic) Assessment failed: {str(e)}"
            return self.ai_parish_road_assessment_text

    def generate_ai_parish_property_analysis(self): 
        if not OPENAI_API_KEY:
            self.ai_parish_property_assessment_text = "AI features disabled: OpenAI API key not configured."
            return self.ai_parish_property_assessment_text
        
        if not hasattr(self, 'parish_summary_df') or self.parish_summary_df.empty:
            self.ai_parish_property_assessment_text = "Parish summary data not available. Please run analysis first."
            return self.ai_parish_property_assessment_text

        summary_df_for_ai = self.parish_summary_df.copy()

        name_col_options = ['OSM_Parish_Name', 'Parish', 'name']
        name_col_to_use = next((col for col in name_col_options if col in summary_df_for_ai.columns), None)
        if not name_col_to_use:
             self.ai_parish_property_assessment_text = "Parish name column missing from summary data."
             return self.ai_parish_property_assessment_text
        summary_df_for_ai.rename(columns={name_col_to_use: 'Parish Name'}, inplace=True)

        if 'avg_beach_dist_m' in summary_df_for_ai.columns:
            summary_df_for_ai['avg_beach_dist_km'] = (summary_df_for_ai['avg_beach_dist_m'] / 1000)
        else:
            summary_df_for_ai['avg_beach_dist_km'] = np.nan


        cols_for_ai_prompt = [
            'Parish Name', 'total_properties', 'area_sqkm', 'property_density',
            'total_for_sale', 'total_for_rent',
            'for_sale_res_count', 'for_sale_com_count', 'for_sale_land_count', 'for_sale_oth_std_count',
            'for_rent_res_count', 'for_rent_com_count', 'for_rent_land_count', 'for_rent_oth_std_count',
            'avg_price', 'min_price', 'max_price',
            'avg_size_sqft', 'avg_bedrooms',
            'avg_beach_dist_km', 'avg_tourism_count_2km',
            'total_road_length_km', 'road_density_km_sqkm'
        ]
        
        existing_cols_for_prompt = [col for col in cols_for_ai_prompt if col in summary_df_for_ai.columns]
        if 'Parish Name' in existing_cols_for_prompt:
            existing_cols_for_prompt.insert(0, existing_cols_for_prompt.pop(existing_cols_for_prompt.index('Parish Name')))
        else:
            if 'Parish Name' in summary_df_for_ai.columns:
                 existing_cols_for_prompt.insert(0, 'Parish Name')

        parish_data_for_ai = summary_df_for_ai[existing_cols_for_prompt].copy()

        for col in parish_data_for_ai.columns: 
            if col == 'Parish Name': continue
            if pd.api.types.is_numeric_dtype(parish_data_for_ai[col]):
                if 'price' in col.lower():
                    parish_data_for_ai[col] = parish_data_for_ai[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
                elif col in ['area_sqkm', 'avg_beach_dist_km', 'avg_bedrooms', 'total_road_length_km', 'avg_tourism_count_2km', 'avg_size_sqft']:
                     parish_data_for_ai[col] = parish_data_for_ai[col].apply(lambda x: f"{x:,.1f}" if pd.notna(x) else "N/A")
                else: 
                     parish_data_for_ai[col] = parish_data_for_ai[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A")
            elif pd.api.types.is_integer_dtype(parish_data_for_ai[col]): 
                 parish_data_for_ai[col] = parish_data_for_ai[col].apply(lambda x: f"{x:,}" if pd.notna(x) else "N/A")
            else: 
                 parish_data_for_ai[col] = parish_data_for_ai[col].apply(lambda x: str(x) if pd.notna(x) else "N/A")


        data_summary_str = parish_data_for_ai.to_string(index=False, na_rep='N/A')
        max_prompt_table_len = 15000 
        if len(data_summary_str) > max_prompt_table_len:
            self._capture_print(f"WARN: Parish summary string for AI is too long ({len(data_summary_str)} chars), truncating.")
            data_summary_str = data_summary_str[:max_prompt_table_len] + "\n... (table truncated due to length)"

        prompt = f"""
You are a Chief Market Strategist for Terra Caribbean, Barbados. Your primary audience is the company's leadership team.
Your goal is to analyze the provided summary of property listing data and related geospatial metrics, aggregated by parish, to deliver a forward-looking strategic assessment. This assessment should clearly outline current business impacts for Terra Caribbean and identify future opportunities, risks, and actionable recommendations. Your analysis must be based *solely* on the data provided.

The data includes metrics such as: {', '.join(existing_cols_for_prompt)}.
- 'avg_beach_dist_km': average distance to the nearest beach in kilometers.
- 'property_density': properties per square kilometer.
- Prices are in USD. Sizes are in square feet. 'N/A' indicates data is not available or effectively zero for counts.

Parish Property Data Summary:
{data_summary_str}

Please structure your strategic report for Terra Caribbean's leadership as follows:

1.  **Executive Summary (Top 3-5 Strategic Implications for Terra Caribbean):**
    * Concisely highlight the most critical data-driven findings. What are the immediate top opportunities Terra Caribbean should pursue, and what are the most significant potential challenges or market shifts to prepare for, based on this parish-level data?

2.  **Parish-by-Parish Strategic Deep Dive:**
    * For key parishes or logical groupings of parishes, provide an analysis covering:
        * **Current Market Position & Terra's Operational Impact:**
            * Describe the current market character (e.g., dominant property types for sale/rent, price tiers, listing volumes).
            * How do these current conditions directly impact Terra Caribbean's existing sales strategies, rental portfolio management, agent resource allocation, and marketing focus in that parish?
        * **Future Outlook & Strategic Imperatives for Terra Caribbean:**
            * Based on metrics like land availability (`for_sale_land_count`), `property_density`, `avg_price` (relative values), infrastructure (`road_density_km_sqkm`), and amenity proximity (`avg_beach_dist_km`, `avg_tourism_count_2km`), what is the future potential of this parish?
            * What specific future opportunities (e.g., targeting emerging luxury segments, advising on affordable housing development, focusing on commercial expansion) or risks (e.g., market saturation, declining appeal) should Terra Caribbean anticipate and plan for in this parish?
            * Suggest 1-2 strategic imperatives or areas of focus for Terra Caribbean for this parish over the next 1-3 years.

3.  **Island-Wide Strategic Themes & Recommendations for Terra Caribbean:**
    * Synthesize overarching themes from the parish analyses (e.g., are there island-wide trends in demand for specific property types? Are certain regions clearly emerging as growth corridors versus established markets?).
    * Provide 2-3 high-level, actionable recommendations for Terra Caribbean's leadership to optimize its overall business strategy, capitalize on identified opportunities, and mitigate risks across Barbados in the coming years. These should be justified by the aggregated data patterns.

**Important Guidelines for Your Report:**
* **Terra Caribbean Centric:** All interpretations, opportunities, risks, and recommendations must be framed in terms of their direct relevance and impact on Terra Caribbean's business goals and operations.
* **Data-Driven & Evidence-Based:** Every assertion must be explicitly tied back to the provided data table. Avoid speculation or external knowledge.
* **Forward-Looking & Actionable:** Emphasize future implications and provide suggestions that can inform concrete business decisions.
* **Clarity, Brevity & Professionalism:** Use language appropriate for an executive audience. Be direct and to the point.
* **Acknowledge Data Limitations:** If 'N/A' values or the snapshot nature of the data limit the depth of certain future projections, clearly state this.
"""
        try:
            self._capture_print(f"\nSending prompt for AI Parish Property (Strategic Business) Analysis. Data string length: {len(data_summary_str)}")
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125", 
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3, 
                max_tokens=2000 
            )
            self.ai_parish_property_assessment_text = response.choices[0].message.content
            return self.ai_parish_property_assessment_text
        except Exception as e:
            self._capture_print(f"AI Parish Property (Strategic Business) Analysis failed: {e}\n{traceback.format_exc()}")
            self.ai_parish_property_assessment_text = f"AI Parish Property (Strategic Business) Analysis failed: {str(e)}"
            return self.ai_parish_property_assessment_text


# --- Streamlit App UI and Main Logic ---
def main():
    st.set_page_config(page_title="Terra Caribbean: Terrain View", layout="wide", initial_sidebar_state="expanded")
    if 'analysis_triggered' not in st.session_state: st.session_state.analysis_triggered = False
    if 'dashboard_logic_instance' not in st.session_state: st.session_state.dashboard_logic_instance = None
    if 'error_during_analysis' not in st.session_state: st.session_state.error_during_analysis = False
    if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0
    if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
    if 'location_explicitly_set' not in st.session_state: st.session_state.location_explicitly_set = False
    if 'needs_map_update' not in st.session_state: st.session_state.needs_map_update = False
    if 'location_info' not in st.session_state: st.session_state.location_info = None
    if 'distance_result' not in st.session_state: st.session_state.distance_result = None
    if 'ai_parish_road_assessment_result' not in st.session_state: st.session_state.ai_parish_road_assessment_result = None
    if 'ai_parish_property_assessment_result' not in st.session_state: st.session_state.ai_parish_property_assessment_result = None


    LOGO_URL = "https://s3.us-east-2.amazonaws.com/terracaribbean.com/wp-content/uploads/2025/04/08080016/site-logo.png"
    dashboard_instance = st.session_state.get('dashboard_logic_instance')

    with st.sidebar:
        st.image(LOGO_URL, width=200) # Correct usage of LOGO_URL

        WEATHER_API_KEY = os.environ.get("OPENWEATHERMAP_API_KEY") or st.secrets.get("OPENWEATHERMAP_API_KEY") 
        
        if WEATHER_API_KEY:
            weather_data = get_weather_data(WEATHER_API_KEY, city="Bridgetown,BB") 
            if weather_data:
                st.subheader(f"Weather in {weather_data.get('city_name', 'Bridgetown')}")
                cols = st.columns([1, 3]) 
                with cols[0]: 
                    if weather_data.get("icon_url"):
                        st.image(weather_data["icon_url"], width=60) 
                with cols[1]: 
                    st.markdown("<div style='font-size: 12px; color: #707070; margin-bottom: -2px;'>Temp</div>", unsafe_allow_html=True)
                    if weather_data.get("temp") is not None:
                        st.markdown(f"<div style='font-size: 36px; font-weight: bold; line-height: 1.1;'>{weather_data['temp']:.0f}°C</div>", unsafe_allow_html=True)
                    details_for_line = []
                    if weather_data.get("description"): details_for_line.append(weather_data['description'])
                    if weather_data.get("feels_like") is not None: details_for_line.append(f"Feels like: {weather_data['feels_like']:.0f}°C")
                    if weather_data.get("humidity") is not None: details_for_line.append(f"Humidity: {weather_data['humidity']}%")
                    main_details_string = " ".join(details_for_line)
                    wind_string_part = ""
                    if weather_data.get("wind_speed") is not None: wind_string_part = f"Wind: {weather_data['wind_speed']:.1f} m/s"
                    final_display_string = main_details_string
                    if wind_string_part:
                        if final_display_string: final_display_string += f" | {wind_string_part}"
                        else: final_display_string = wind_string_part
                    if final_display_string: st.markdown(f"<div style='font-size: 13px; line-height: 1.4; margin-top: 2px;'>{final_display_string}</div>", unsafe_allow_html=True)
            else: st.warning("🌦️ Could not fetch weather data.")
        else: st.info("🌦️ Weather display disabled. Add `OPENWEATHERMAP_API_KEY` to Streamlit secrets.")

        st.markdown("---") 
        st.header("Controls")
        uploaded_file = st.file_uploader("Upload Property Data (Excel .xlsx or CSV .csv)", type=["xlsx", "csv"], key=f"file_uploader_{st.session_state.uploader_key}")
        status_placeholder = st.empty()
        run_button_clicked = st.button("🚀 Run Analysis", use_container_width=True)

        if st.session_state.get('analysis_done') and dashboard_instance:
            st.markdown("---"); st.header("Locate on Map")
            lat_val, lon_val, zoom_val = st.session_state.get('map_center_lat_input', 13.1731), st.session_state.get('map_center_lon_input', -59.6369), st.session_state.get('map_zoom_input', 14)
            lat_input, lon_input, zoom_input = st.number_input("Latitude:", value=lat_val, format="%.6f", key="lat_input_widget"), st.number_input("Longitude:", value=lon_val, format="%.6f", key="lon_input_widget"), st.slider("Zoom Level:", 10, 18, zoom_val, key="zoom_input_widget")
            if st.button("📍 Go to Location", use_container_width=True):
                st.session_state.update({'map_center_lat': lat_input, 'map_center_lon': lon_input, 'map_zoom': zoom_input, 'location_explicitly_set': True, 'map_center_lat_input': lat_input, 'map_center_lon_input': lon_input, 'map_zoom_input': zoom_input, 'needs_map_update': True})
                identified_parish_name = dashboard_instance.get_parish_for_coords(lat_input, lon_input)
                parish_info_strs = {"parish_area": "N/A", "parish_avg_beach_dist": "N/A", "parish_avg_tourism_count": "N/A", "parish_total_road_length": "N/A", "parish_road_density": "N/A"}
                valid_parish_identified = identified_parish_name and identified_parish_name not in ["Not within a known parish boundary.", "Error during parish lookup.", "Parish data not available or coordinates invalid.", "Parish name not identified", "Unknown Parish", "Unnamed Parish"]
                if valid_parish_identified and hasattr(dashboard_instance, 'parish_summary_df') and not dashboard_instance.parish_summary_df.empty:
                    summary_key_col = next((col for col in [dashboard_instance.parish_summary_df.columns[0], 'name', 'OSM_Parish_Name', 'Parish'] if col in dashboard_instance.parish_summary_df.columns), None)
                    if summary_key_col:
                        parish_stats = dashboard_instance.parish_summary_df[dashboard_instance.parish_summary_df[summary_key_col].astype(str) == str(identified_parish_name)]
                        if not parish_stats.empty:
                            stats_row = parish_stats.iloc[0]
                            area_val = stats_row.get('area_sqkm'); parish_info_strs["parish_area"] = f"{area_val:.1f} sq km" if pd.notna(area_val) and area_val > 0 else "N/A"
                            beach_dist_val = stats_row.get('avg_beach_dist_m'); parish_info_strs["parish_avg_beach_dist"] = f"{(beach_dist_val / 1000):.1f} km" if pd.notna(beach_dist_val) else "N/A"
                            tourism_count_val = stats_row.get('avg_tourism_count_2km'); parish_info_strs["parish_avg_tourism_count"] = f"{tourism_count_val:.1f}" if pd.notna(tourism_count_val) else "N/A"
                            road_len_val = stats_row.get('total_road_length_km'); parish_info_strs["parish_total_road_length"] = f"{road_len_val:.1f} km" if pd.notna(road_len_val) else "N/A"
                            road_dens_val = stats_row.get('road_density_km_sqkm'); parish_info_strs["parish_road_density"] = f"{road_dens_val:.2f} km/km²" if pd.notna(road_dens_val) else "N/A"
                st.session_state.location_info = {"lat_dec": lat_input, "lon_dec": lon_input, "lat_dms": decimal_to_dms(lat_input, True), "lon_dms": decimal_to_dms(lon_input, False), "parish": identified_parish_name, **parish_info_strs}
                st.rerun()

            if st.session_state.location_info and st.session_state.location_explicitly_set:
                info = st.session_state.location_info
                st.markdown("---"); st.subheader("Point Location Details:")
                st.markdown(f"**Coordinates (Entered Point):**\n&nbsp;&nbsp;&nbsp;Lat: {info['lat_dec']:.6f} / {info['lat_dms']}\n&nbsp;&nbsp;&nbsp;Lon: {info['lon_dec']:.6f} / {info['lon_dms']}")
                st.markdown(f"**Identified Parish:**\n&nbsp;&nbsp;&nbsp;Name: {info['parish']}")
                if info["parish"] and info["parish"] not in ["Not within a known parish boundary.", "Error during parish lookup.", "Parish data not available or coordinates invalid.", "Parish name not identified", "Unknown Parish", "Unnamed Parish"]:
                    st.markdown(f"**Parish Level Summary ({info['parish']}):**")
                    st.markdown(f"&nbsp;&nbsp;&nbsp;Area: {info.get('parish_area', 'N/A')}")
                    st.markdown(f"&nbsp;&nbsp;&nbsp;Avg. Beach Distance: {info.get('parish_avg_beach_dist', 'N/A')}")
                    st.markdown(f"&nbsp;&nbsp;&nbsp;Avg. Attractions (2km): {info.get('parish_avg_tourism_count', 'N/A')}")
                    st.markdown(f"&nbsp;&nbsp;&nbsp;Total Road Length: {info.get('parish_total_road_length', 'N/A')}")
                    st.markdown(f"&nbsp;&nbsp;&nbsp;Road Density: {info.get('parish_road_density', 'N/A')}")
        
        st.markdown("---") 
        st.header("📏 Calculate Distance")
        st.write("Enter the coordinates for two points to calculate the 'as-the-crow-flies' distance between them.")
        col1, col2 = st.columns(2)
        with col1: st.subheader("Point 1"); lat1_input = st.number_input("Start Latitude:", value=13.1731, format="%.6f", key="dist_lat1"); lon1_input = st.number_input("Start Longitude:", value=-59.6369, format="%.6f", key="dist_lon1")
        with col2: st.subheader("Point 2"); lat2_input = st.number_input("End Latitude:", value=13.0801, format="%.6f", key="dist_lat2"); lon2_input = st.number_input("End Longitude:", value=-59.4876, format="%.6f", key="dist_lon2")
        if st.button("Calculate Distance Between Points", use_container_width=True):
            distance_km = calculate_haversine_distance(lat1_input, lon1_input, lat2_input, lon2_input)
            st.session_state.distance_result = f"**Distance:** {distance_km:.2f} km ({(distance_km * 0.621371):.2f} miles)" if pd.notna(distance_km) else "**Error:** Please ensure all coordinates are valid numbers."
        if st.session_state.distance_result: st.success(st.session_state.distance_result)


    st.title("🏞️ Terra Caribbean: Terrain & Property View")
    st.markdown("Interactive terrain-focused map to analyze property listings with geospatial data for Barbados.")
    st.markdown("---")
    st.info("""
    **Understanding the Terrain Map:**
    This view emphasizes the island's topography. Key features include:
    * **Terrain Base Map:** See hills, valleys, and coastal features. Optional topographic layers are available in the map's layer control.
    * **Parish Boundaries & Centers:** Clearly marked parishes.
    * **Property Summaries:** Click on parish markers for aggregated property data, including Residential/Commercial splits and road network statistics.
    * **Locate on Map (Sidebar):** Enter Latitude and Longitude, click "Go to Location" to center the map, add a marker, and see location details (including parish road stats) in the sidebar.
    * **Calculate Distance (Sidebar):** Enter two sets of coordinates to find the direct distance between them.
    * ***Other layers (Schools, Supermarkets, Road Network, Land Features, POIs) are hidden by default but can be enabled via the layer control (top-right of the map).***
    """)

    if run_button_clicked:
        if uploaded_file is not None:
            st.session_state.update({
                'analysis_triggered': True, 'error_during_analysis': False, 'dashboard_logic_instance': None,
                'location_explicitly_set': False, 'needs_map_update': False, 'location_info': None, 
                'distance_result': None, 'ai_parish_road_assessment_result': None,
                'ai_parish_property_assessment_result': None 
            }) 
            for key in ['map_center_lat', 'map_center_lon', 'map_zoom', 'map_center_lat_input', 'map_center_lon_input', 'map_zoom_input']:
                if key in st.session_state: del st.session_state[key]
            with status_placeholder.container(), st.spinner("Analysis in progress... This may take a few minutes for the first run or new data."):
                dashboard = TerraDashboardLogic(uploaded_file_object=uploaded_file)
                analysis_successful = dashboard.run_analysis_streamlit()
            if analysis_successful:
                st.session_state.update({'dashboard_logic_instance': dashboard, 'analysis_done': True, 'needs_map_update': True})
                status_placeholder.success("Analysis Complete!")
                st.rerun()
            else:
                st.session_state.update({'dashboard_logic_instance': dashboard, 'analysis_done': False, 'error_during_analysis': True})
                status_placeholder.error("Analysis failed. Check Console Log tab.")
            st.session_state.uploader_key += 1
        else:
            st.sidebar.warning("⚠️ Please upload a property data file first!"); st.session_state.analysis_done = False

    dashboard_instance = st.session_state.get('dashboard_logic_instance')
    if dashboard_instance:
        tab_titles = ["🗺️ Terrain Map", "📊 Chart", "📈 Key Statistics", "📊 Parish Summary", 
                      "💡 AI Road Insights", "🏡 AI Property Insights", 
                      "📥 Export Data", "📋 Console Log", "ℹ️ Calculations & Notes"]
        tabs = st.tabs(tab_titles)

        with tabs[0]: 
            st.subheader("Interactive Terrain Map with Property Summaries")
            if st.session_state.get('analysis_done') and dashboard_instance:
                if st.session_state.get('needs_map_update', False) or not dashboard_instance.map_html_content:
                    with st.spinner("Generating / Updating map..."):
                        map_lat, map_lon, map_zoom = st.session_state.get('map_center_lat'), st.session_state.get('map_center_lon'), st.session_state.get('map_zoom')
                        map_obj = dashboard_instance.create_map_object_streamlit(center_lat=map_lat, center_lon=map_lon, zoom=map_zoom)
                        dashboard_instance.map_html_content = map_obj.get_root().render() if map_obj else "<p>Error generating map. Check data.</p>"
                    st.session_state.needs_map_update = False
                if dashboard_instance.map_html_content: st.components.v1.html(dashboard_instance.map_html_content, height=700, scrolling=True)
                else: st.info("Map could not be generated.")
            elif st.session_state.get('analysis_triggered'): st.info("Map is being generated...")
            else: st.info("Map will be displayed here after analysis is run.")


        with tabs[1]: 
            st.subheader("Price vs. Beach Distance Chart")
            if st.session_state.get('analysis_done') and dashboard_instance.chart_path and os.path.exists(dashboard_instance.chart_path):
                try: st.image(Image.open(dashboard_instance.chart_path), caption="Property Price vs. Distance to Nearest Beach", use_container_width=True)
                except Exception as e: st.error(f"Could not load chart: {e}")
            elif st.session_state.get('analysis_triggered'): st.info("Chart is being generated...")
            else: st.info("Chart will be displayed here after analysis is run.")

        with tabs[2]: 
            st.subheader("Key Statistics")
            if st.session_state.get('analysis_done') and hasattr(dashboard_instance, 'stats_data_for_streamlit') and dashboard_instance.stats_data_for_streamlit:
                for item in dashboard_instance.stats_data_for_streamlit: st.markdown(f"- {item}")
            elif st.session_state.get('analysis_triggered'): st.info("Statistics are being generated.")
            else: st.info("Key statistics will be displayed here after analysis is run.")

        with tabs[3]: 
            st.subheader("Consolidated Parish Summary")
            if st.session_state.get('analysis_done') and hasattr(dashboard_instance, 'parish_summary_df') and not dashboard_instance.parish_summary_df.empty:
                summary_df_display = dashboard_instance.parish_summary_df.copy()
                
                parish_name_col_options = ['OSM_Parish_Name', 'Parish', 'name']
                actual_parish_name_col = next((col for col in parish_name_col_options if col in summary_df_display.columns), summary_df_display.columns[0] if not summary_df_display.empty else 'Parish')

                display_cols_ordered = [
                    actual_parish_name_col, 'total_properties', 'area_sqkm', 'property_density',
                    'avg_price', 'min_price', 'max_price', 'avg_size_sqft', 'avg_bedrooms',
                    'total_for_sale', 'for_sale_res_count', 'for_sale_com_count', 'for_sale_land_count', 'for_sale_oth_std_count',
                    'total_for_rent', 'for_rent_res_count', 'for_rent_com_count', 'for_rent_land_count', 'for_rent_oth_std_count',
                    'avg_beach_dist_m', 'avg_tourism_count_2km', 'total_road_length_km', 'road_density_km_sqkm'
                ]
                
                final_display_cols = [col for col in display_cols_ordered if col in summary_df_display.columns]
                from collections import OrderedDict 
                final_display_cols = list(OrderedDict.fromkeys(final_display_cols))

                summary_df_to_show = summary_df_display[final_display_cols].copy()

                numeric_format_map = {
                    'avg_price': "${:,.0f}", 'min_price': "${:,.0f}", 'max_price': "${:,.0f}",
                    'avg_size_sqft': "{:,.0f} sq ft",
                    'avg_bedrooms': "{:.1f}",
                    'area_sqkm': "{:.1f} km²",
                    'property_density': "{:.2f} props/km²",
                    'avg_beach_dist_m': (lambda x: f"{(x/1000):.1f} km" if pd.notna(x) else "N/A"),
                    'avg_tourism_count_2km': "{:.1f}",
                    'total_road_length_km': "{:.1f} km",
                    'road_density_km_sqkm': "{:.2f} km/km²"
                }

                for col, fmt_str_or_func in numeric_format_map.items():
                    if col in summary_df_to_show.columns:
                        summary_df_to_show[col] = pd.to_numeric(summary_df_to_show[col], errors='coerce')
                        if callable(fmt_str_or_func):
                             summary_df_to_show[col] = summary_df_to_show[col].apply(fmt_str_or_func)
                        else:
                             summary_df_to_show[col] = summary_df_to_show[col].apply(lambda x: fmt_str_or_func.format(x) if pd.notna(x) else "N/A")
                
                int_cols = [
                    'total_properties', 'total_for_sale', 'for_sale_res_count', 'for_sale_com_count', 
                    'for_sale_land_count', 'for_sale_oth_std_count', 'total_for_rent', 
                    'for_rent_res_count', 'for_rent_com_count', 'for_rent_land_count', 'for_rent_oth_std_count'
                ]
                for col in int_cols:
                    if col in summary_df_to_show.columns:
                        summary_df_to_show[col] = pd.to_numeric(summary_df_to_show[col], errors='coerce').apply(lambda x: f"{int(x):,}" if pd.notna(x) else "0")


                st.dataframe(summary_df_to_show.drop(columns=['latitude', 'longitude'], errors='ignore'))
                
                @st.cache_data
                def convert_summary_df_to_csv_export(df_to_convert): return df_to_convert.to_csv(index=False).encode('utf-8')
                st.download_button(label="Download Parish Summary as CSV (Raw Data)", data=convert_summary_df_to_csv_export(summary_df_display.drop(columns=['latitude', 'longitude'], errors='ignore')), file_name="terra_parish_summary_detailed_raw.csv", mime="text/csv", key="download_detailed_summary_csv_button")
            elif st.session_state.get('analysis_triggered'): st.info("Parish summary is being generated.")
            else: st.info("Parish summary data will be displayed here after analysis is run.")

        with tabs[4]: # AI Road Insights
            with st.expander("💡 AI-Powered Parish Road Network Assessment", expanded=True): # Default to expanded
                if st.session_state.get('analysis_done') and dashboard_instance:
                    if not OPENAI_API_KEY:
                        st.warning("OpenAI API key not configured. This AI feature is disabled.")
                    elif not hasattr(dashboard_instance, 'parish_summary_df') or dashboard_instance.parish_summary_df.empty:
                        st.info("Parish summary data is not yet available. Please run a full analysis first.")
                    else:
                        required_road_cols_check = ['total_road_length_km', 'road_density_km_sqkm', 'area_sqkm', 'total_properties']
                        name_col_check = next((col for col in ['OSM_Parish_Name', 'Parish', 'name'] if col in dashboard_instance.parish_summary_df.columns), None)
                        
                        all_present_and_has_data = name_col_check is not None
                        missing_cols_display = []
                        if all_present_and_has_data:
                            for col in required_road_cols_check:
                                if col not in dashboard_instance.parish_summary_df.columns or dashboard_instance.parish_summary_df[col].isnull().all():
                                    all_present_and_has_data = False
                                    missing_cols_display.append(col)
                        else: 
                            all_present_and_has_data = False
                            missing_cols_display.append("Parish Name identifier")

                        if not all_present_and_has_data:
                            st.info(f"Parish summary data is missing key road statistics (e.g., {', '.join(missing_cols_display)}) or these columns lack data. Ensure the analysis has run successfully and these metrics are generated.")
                        else:
                            if st.button("🤖 Generate AI Road Network Assessment", key="ai_road_assess_button"):
                                with st.spinner("AI is analyzing parish road networks..."):
                                    assessment_text = dashboard_instance.generate_ai_parish_road_assessment()
                                    st.session_state.ai_parish_road_assessment_result = assessment_text
                            
                            st.caption("""
                                ℹ️ **AI Perspective:** The insights below are generated from the viewpoint of a Senior Real Estate Strategist at Terra Caribbean. 
                                The analysis focuses on how road networks impact local property markets and is based *solely* on the processed data presented to the AI. 
                                Different data or a modified analytical focus could lead to different perspectives or conclusions.
                                """)

                            if st.session_state.get('ai_parish_road_assessment_result'):
                                st.markdown("---")
                                st.markdown(st.session_state.ai_parish_road_assessment_result)
                            elif not st.session_state.get('ai_parish_road_assessment_result') and not missing_cols_display : 
                                st.caption("Click the 'Generate' button above to get an AI assessment of parish road networks, focusing on implications for Terra Caribbean.")
                elif st.session_state.get('analysis_triggered'):
                   st.info("Analysis is running. AI assessment will be available once complete.")
                else:
                   st.info("Run an analysis first to enable AI Parish Road Network Assessment.")

        with tabs[5]: # AI Property Insights
            with st.expander("🏡 AI-Powered Strategic Property Market Insights", expanded=True): # Default to expanded
                if st.session_state.get('analysis_done') and dashboard_instance:
                    if not OPENAI_API_KEY:
                        st.warning("OpenAI API key not configured. This AI feature is disabled.")
                    elif not hasattr(dashboard_instance, 'parish_summary_df') or dashboard_instance.parish_summary_df.empty:
                        st.info("Parish summary data is not yet available. Please run a full analysis first.")
                    else:
                        expected_cols_for_prop_ai = ['avg_price', 'total_properties', 'property_density'] 
                        missing_cols_data = [col for col in expected_cols_for_prop_ai if col not in dashboard_instance.parish_summary_df.columns or dashboard_instance.parish_summary_df[col].isnull().all()]
                        
                        if missing_cols_data:
                            st.info(f"Parish summary data is missing key columns for property market analysis (e.g., {', '.join(missing_cols_data)}) or these columns lack data. Ensure the analysis has run successfully and these metrics are generated.")
                        else:
                            if st.button("📈 Generate Strategic AI Property Market Insights", key="ai_property_assess_button"):
                                with st.spinner("AI is generating strategic property market insights for Terra Caribbean leadership... This may take a moment."):
                                    assessment_text = dashboard_instance.generate_ai_parish_property_analysis()
                                    st.session_state.ai_parish_property_assessment_result = assessment_text
                            
                            st.caption("""
                                ℹ️ **AI Perspective:** This analysis is generated from the viewpoint of a Chief Market Strategist for Terra Caribbean. 
                                It provides a forward-looking strategic assessment based *only* on the processed parish summary data. 
                                The insights aim to inform business decisions. Different data inputs or analytical objectives could change the focus.
                                """)
                                                        
                            if st.session_state.get('ai_parish_property_assessment_result'):
                                st.markdown("---")
                                st.markdown(st.session_state.ai_parish_property_assessment_result)
                            elif not st.session_state.get('ai_parish_property_assessment_result') and not missing_cols_data:
                                st.caption("Click the 'Generate' button above to get an AI-powered strategic analysis of the property market based on the parish summary data.")
                elif st.session_state.get('analysis_triggered'):
                    st.info("Analysis is running. AI property market assessment will be available once complete.")
                else:
                    st.info("Run an analysis first to enable AI Property Market Analysis.")


        with tabs[6]: 
            st.subheader("Export Analyzed Data")
            if st.session_state.get('analysis_done'):
                export_df = dashboard_instance.get_export_dataframe()
                if not export_df.empty:
                    @st.cache_data
                    def convert_df_to_csv(df_to_convert): return df_to_convert.to_csv(index=False).encode('utf-8')
                    st.download_button(label="Download Analyzed Property Data as CSV", data=convert_df_to_csv(export_df), file_name="terra_analysis_results.csv", mime="text/csv", key="download_csv_button")
                elif st.session_state.get('analysis_triggered') : st.info("No analyzed data available for export.")
            else: st.info("Analyzed data will be available for export here after analysis is run.")

        with tabs[7]: 
            st.subheader("Analysis Log")
            if hasattr(dashboard_instance, 'log_capture') and dashboard_instance.log_capture:
                st.text_area("Log Output:", value="".join(dashboard_instance.log_capture), height=500, key="console_log_area_display", disabled=True)
            elif st.session_state.get('analysis_triggered'): st.info("Attempting to capture logs...")
            else: st.info("Console logs from the analysis will appear here.")

        with tabs[8]: 
            st.subheader("Understanding the Calculations, Data & Conversions")
            st.markdown("""
            This section explains how data is processed by the application and provides some useful unit conversions.

            #### 1. Geographic Data & Calculations 🗺️
            * **Primary Data Source:** OpenStreetMap (OSM) is the main source for geographic features.
            * **Boundaries & Features:** Parish boundaries, the road network, beaches, and various Points of Interest (POIs) like schools and supermarkets are fetched from OSM for Barbados.
            * **Projections for Accuracy:** For calculations involving area or length (e.g., parish area, road length), the script uses the `EPSG:32620` projection (UTM Zone 20N). This projection is suitable for Barbados and provides measurements in meters, which are then often converted to kilometers. For display on maps, `EPSG:4326` (WGS84, standard latitude/longitude) is typically used.
            * **Parish Area:** The area of each parish (shown in square kilometers, km²) is calculated based on its OSM polygon geometry after being projected.
            * **Road Network Statistics (Parish Level - shown in map popups & sidebar):**
            * **Total Road Length (km) per Parish:** This is the sum of the lengths of all road segments (obtained from OSM) that fall within the boundaries of a specific parish. To ensure accuracy, roads that cross parish boundaries are geometrically "clipped" to only include the portion within that parish before their lengths are summed.
            * **Road Density (km/km²) per Parish:** This metric is calculated by dividing the `Total Road Length (km)` within a parish by that `Parish Area (km²)`. It indicates how densely a parish is covered by roads (e.g., a higher value means more kilometers of road for each square kilometer of land in that parish).
            * **Road Network Statistics (Island Level - shown in "Key Statistics" tab):**
            * **Total Road Network Length (Island):** The sum of lengths of all drivable road segments fetched for the entire island of Barbados from OSM. Lengths are originally in meters from OSM and converted to kilometers.
            * **Overall Road Density (Island):** This is the `Total Road Network Length (Island)` in kilometers divided by the total land area of Barbados in square kilometers. It gives an average measure of road coverage for the entire island.
            * **Property Geocoding:** Properties from your uploaded data file are assigned coordinates. These coordinates are based on the **centroid (geographic center point) of their listed parish**. A slight random "jitter" (small offset) is added to these coordinates if multiple properties fall in the same parish to prevent them from overlapping perfectly on the map. *Therefore, property markers on the map indicate the general parish area, not the exact street address of the property.*
            * **Distances to Features:**
            * **Beach Distance:** Calculated using a KDTree algorithm. This finds the shortest straight-line ("as the crow flies") distance from a property's assigned location (parish centroid) to the nearest point on any beach geometry fetched from OSM.
            * **Tourism Count:** Represents the number of tourism-related POIs (from OSM) found within a 2-kilometer radius (a circular buffer) around a property's assigned location.
            * **Haversine Distance Calculator (Sidebar Tool):** This tool calculates the great-circle distance (the shortest path on the surface of a sphere) between two latitude/longitude points that you enter. It's an "as-the-crow-flies" distance and **does not represent actual road/travel distance or take into account terrain or obstacles.**

            #### 2. Property Data Processing 🏡
            * **Input Data:** The core analysis relies on the data you upload via an Excel (.xlsx) or CSV (.csv) file.
            * **Size Conversion:** The 'Size' field from your input is parsed to extract building or land area. The script primarily aims to convert this to square footage (`Size_sqft`). It also attempts to convert "acres" to square feet if "acre" units are specified in the size description.
            * **Bedrooms/Bathrooms:** These are typically extracted from the 'Description' field of your property data using pattern matching (e.g., looking for "3 Bed" or "2 Bath").
            * **Data Cleaning & Standardization:**
            * 'Property Type' values from your file are standardized into broader categories like Residential, Commercial, Land, or Other.
            * Parish names are cleaned and standardized to match a known list of official parish names for Barbados to ensure accurate grouping and geocoding.

            #### 3. Important Unit Conversions 📏
            * `1 kilometer (km) = 1,000 meters (m)`
            * `1 kilometer (km) ≈ 0.621371 miles`
            * `1 square kilometer (km²) = 100 hectares (ha)`
            * `1 square kilometer (km²) ≈ 0.386102 square miles (sq mi)`
            * `1 square kilometer (km²) ≈ 247.105 acres`
            * `1 acre = 43,560 square feet (sq ft)`

            #### 4. Limitations & Considerations ⚠️
            * **OSM Data Quality:** The accuracy and completeness of geographic features like roads, Points of Interest, and administrative boundaries depend on the data available in OpenStreetMap. OSM is a community-edited project, so data quality can vary by region and time.
            * **Uploaded Data Accuracy:** The insights and analyses generated by this tool are heavily influenced by the accuracy, completeness, and formatting of the property data you provide in your uploaded file.
            * **Geocoding Approximation:** Because properties are mapped to parish centroids (not precise addresses), any analysis based on exact location (like distance to a specific small POI or a particular road) is an approximation based on that parish-level geocoding.
            * **Calculations as Estimates:** All derived statistics (distances, areas, densities, etc.) should be considered estimates based on the available data and the methodologies described above. They provide valuable insights but may not always reflect official or surveyed figures perfectly.
            """)
            st.markdown("---"); st.write("Raw OSM Parish Data (if loaded):")
            if hasattr(dashboard_instance, 'raw_parishes_from_osm') and not dashboard_instance.raw_parishes_from_osm.empty:
                st.dataframe(dashboard_instance.raw_parishes_from_osm.drop(columns='geometry', errors='ignore'))
            else: st.info("Raw OSM parish data was not loaded or is empty.")

        if st.session_state.get('error_during_analysis'): st.error("An error occurred during the last analysis. Review the 'Console Log' tab.")
    elif st.session_state.get('analysis_triggered', False) and st.session_state.get('error_during_analysis', False):
        st.error("Analysis failed. Check the console log if available.")
        if st.session_state.get('dashboard_logic_instance') and hasattr(st.session_state.dashboard_logic_instance, 'log_capture') and st.session_state.dashboard_logic_instance.log_capture:
            st.subheader("Partial Analysis Log"); st.text_area("Log Output:", value="".join(st.session_state.dashboard_logic_instance.log_capture), height=500, key="error_console_log_area_display_alt", disabled=True)
    else: st.info("👋 Welcome! Please upload a property data file and click '🚀 Run Analysis' in the sidebar to begin.")

    st.markdown("---")
    property_count = len(dashboard_instance.analyzed_properties) if st.session_state.get('analysis_done') and dashboard_instance and hasattr(dashboard_instance, 'analyzed_properties') and not dashboard_instance.analyzed_properties.empty else 0
    credits_line1 = f"Data Sources: Terra Caribbean, OpenStreetMap{' • Displaying ' + str(property_count) + ' properties.' if property_count > 0 else ''}"
    credits_line2 = f"© {datetime.datetime.now().year} Terra Caribbean Geospatial Analytics Platform • All Prices in USD"
    st.caption(f"{credits_line1}\n{credits_line2}\nApp Created by Matthew Blackman. Assisted by AI.")

if __name__ == "__main__":
    main()