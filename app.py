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

# OSMnx settings
ox.settings.log_console = False
ox.settings.use_cache = True
ox.settings.timeout = 600

# --- Helper functions ---

def parse_size_to_sqft_static(size_str):
    if pd.isna(size_str) or not isinstance(size_str, str): return np.nan
    size_str_l = size_str.lower(); num_match = re.search(r'([\d,]+\.?\d*)\s*(.*)', size_str_l)
    if not num_match: return np.nan
    try: num_val = float(num_match.group(1).replace(',', '')); unit_str = num_match.group(2).strip()
    except ValueError: return np.nan
    if 'acre' in unit_str: return num_val * 43560
    if ('sq' in unit_str and ('ft' in unit_str or 'feet' in unit_str)) or 'sf' in unit_str: return num_val
    if unit_str == "" and num_val > 200: return num_val # Assume sqft if no unit and large number
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
    """Converts decimal degrees to Degrees, Minutes, Seconds."""
    if pd.isna(deg):
        return "N/A"
    try:
        d = int(deg)
        m_float = abs(deg - d) * 60
        m = int(m_float)
        s = (m_float - m) * 60

        if is_lat:
            direction = 'N' if deg >= 0 else 'S'
        else:
            direction = 'E' if deg >= 0 else 'W'

        return f"{abs(d)}° {m}' {s:.3f}\" {direction}"
    except Exception:
        return "Error converting"

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
    parishes_gdf = gpd.GeoDataFrame(); beaches_gdf = gpd.GeoDataFrame(); tourism_gdf = gpd.GeoDataFrame(); feature_polygons_gdf = gpd.GeoDataFrame();
    raw_parishes_from_osm_snapshot = gpd.GeoDataFrame()
    try:
        log_capture_list_ref.append("Downloading Barbados country boundary...\n"); barbados_boundary = ox.geocode_to_gdf("Barbados")
        if barbados_boundary.empty: log_capture_list_ref.append("Failed to download Barbados boundary.\n"); return parishes_gdf, beaches_gdf, tourism_gdf, feature_polygons_gdf, raw_parishes_from_osm_snapshot
        poly = barbados_boundary.geometry.iloc[0]
        tags_list = [{"boundary":"administrative","admin_level":"6"},{"place":"parish"}]; log_capture_list_ref.append("Attempting to download parish boundaries from OSM...\n")
        temp_parishes_data = gpd.GeoDataFrame(); found_parish_data_source = False
        for tags in tags_list:
            try:
                current_fetch = ox.features_from_polygon(poly, tags)
                if not current_fetch.empty:
                    current_fetch = current_fetch[current_fetch.geometry.type.isin(['Polygon','MultiPolygon'])]
                    if not current_fetch.empty:
                        if any(name_tag in current_fetch.columns for name_tag in ['name', 'official_name', 'name:en']):
                            temp_parishes_data = current_fetch; log_capture_list_ref.append(f"Successfully downloaded features using tags: {tags}\n"); found_parish_data_source = True; break
            except Exception as e: log_capture_list_ref.append(f"Parish download attempt with tags {tags} failed: {e}\n")
        if not found_parish_data_source or temp_parishes_data.empty: log_capture_list_ref.append("OSM parish download failed or no usable data.\n"); return parishes_gdf, beaches_gdf, tourism_gdf, feature_polygons_gdf, raw_parishes_from_osm_snapshot
        parishes_gdf = temp_parishes_data
        if not parishes_gdf.empty: raw_parishes_from_osm_snapshot = parishes_gdf.copy();
        if not parishes_gdf.empty:
            if 'name' not in parishes_gdf.columns: parishes_gdf['name'] = parishes_gdf.index.astype(str)
            else: parishes_gdf['name'] = parishes_gdf['name'].replace(['nan', 'NaN', 'Nan', 'none', 'None', '', 'null', 'NULL'], None, regex=False)
            name_preference = ['name', 'official_name', 'name:en', 'alt_name', 'loc_name']
            parishes_gdf['OSM_Parish_Name'] = pd.Series([None] * len(parishes_gdf), index=parishes_gdf.index, dtype=object)
            for col_name_pref in name_preference:
                if col_name_pref in parishes_gdf.columns:
                    cleaned_pref_col = parishes_gdf[col_name_pref].replace(['nan', 'NaN', 'Nan', 'none', 'None', '', 'null', 'NULL'], None, regex=False)
                    parishes_gdf['OSM_Parish_Name'] = parishes_gdf['OSM_Parish_Name'].fillna(cleaned_pref_col)
            parishes_gdf['name'] = parishes_gdf['name'].apply(clean_parish_name_generic_static)
            final_parish_cols = ['name', 'OSM_Parish_Name', 'geometry']; existing_final_cols = [col for col in final_parish_cols if col in parishes_gdf.columns]
            if not all(col in existing_final_cols for col in ['name', 'OSM_Parish_Name']):
                if 'name' not in parishes_gdf.columns: parishes_gdf['name'] = pd.Series([None] * len(parishes_gdf), index=parishes_gdf.index)
                if 'OSM_Parish_Name' not in parishes_gdf.columns: parishes_gdf['OSM_Parish_Name'] = pd.Series([None] * len(parishes_gdf), index=parishes_gdf.index)
                existing_final_cols = [col for col in final_parish_cols if col in parishes_gdf.columns]
            parishes_gdf = parishes_gdf[existing_final_cols].copy()
            if 'geometry' in parishes_gdf.columns:
                parishes_gdf.set_crs(barbados_boundary.crs,inplace=True); parishes_gdf = parishes_gdf[parishes_gdf.geometry.is_valid & parishes_gdf.geometry.notna()]
                if not parishes_gdf.empty: log_capture_list_ref.append(f"Found and processed {len(parishes_gdf)} valid OSM parish geometries.\n")
            else: parishes_gdf = gpd.GeoDataFrame()
        log_capture_list_ref.append("Downloading beach data from OSM...\n"); beaches_gdf = ox.features_from_polygon(poly,tags={"natural":"beach"});
        if not beaches_gdf.empty: beaches_gdf = beaches_gdf[beaches_gdf.geometry.notna()]
        log_capture_list_ref.append("Downloading tourism points from OSM...\n"); tourism_gdf = ox.features_from_polygon(poly, tags={"tourism": True});
        if not tourism_gdf.empty: tourism_gdf = tourism_gdf[tourism_gdf.geometry.notna()]
        feature_polygon_tags = {"leisure": ["park", "golf_course", "nature_reserve", "recreation_ground"], "amenity": ["university", "college", "school"], "landuse": ["cemetery", "religious"]}
        try:
            feature_polygons_gdf = ox.features_from_polygon(poly, tags=feature_polygon_tags)
            if not feature_polygons_gdf.empty:
                feature_polygons_gdf = feature_polygons_gdf[feature_polygons_gdf.geometry.type.isin(['Polygon', 'MultiPolygon']) & feature_polygons_gdf['name'].notna() & feature_polygons_gdf.geometry.notna()]
                if not feature_polygons_gdf.empty: feature_polygons_gdf = feature_polygons_gdf[feature_polygons_gdf.geometry.is_valid]
        except Exception as e_fp: log_capture_list_ref.append(f"Could not fetch feature polygons from OSM: {e_fp}\n")
        return parishes_gdf, beaches_gdf, tourism_gdf, feature_polygons_gdf, raw_parishes_from_osm_snapshot
    except Exception as e: log_capture_list_ref.append(f"General error in get_geodata: {e}\n{traceback.format_exc()}\n"); return gpd.GeoDataFrame(), gpd.GeoDataFrame(), gpd.GeoDataFrame(), gpd.GeoDataFrame(), gpd.GeoDataFrame()

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
            if col_ensure not in parishes_alt.columns: parishes_alt[col_ensure] = pd.Series([None] * len(parishes_alt), index=parishes_alt.index); existing_final_cols.append(col_ensure)
        return parishes_alt[existing_final_cols]
    except Exception as e: log_capture_list_ref.append(f"Error alt parish data: {e}\n{traceback.format_exc()}\n"); return parishes_alt

class TerraDashboardLogic:
    def __init__(self, uploaded_file_object=None):
        self.analyzed_properties = gpd.GeoDataFrame()
        self.parishes = gpd.GeoDataFrame()
        self.beaches = gpd.GeoDataFrame()
        self.tourism_points = gpd.GeoDataFrame()
        self.feature_polygons = gpd.GeoDataFrame()
        self.map_html_content = None
        self.chart_path = ""
        self.stats_data_for_streamlit = []
        self.log_capture = []
        self.raw_parishes_from_osm = gpd.GeoDataFrame()
        self.uploaded_file_object = uploaded_file_object
        self.parish_summary_df = pd.DataFrame()

    def _capture_print(self, message):
        self.log_capture.append(str(message) + "\n")

    def _load_property_data(self):
        if self.uploaded_file_object:
            content_bytes = self.uploaded_file_object.getvalue()
            filename = self.uploaded_file_object.name
            return load_cached_property_data(content_bytes, filename, self.log_capture)
        return pd.DataFrame()

    def _get_geodata(self):
        parishes, beaches, tourism, features, raw_osm = get_cached_geodata(self.log_capture)
        self.raw_parishes_from_osm = raw_osm
        return parishes, beaches, tourism, features

    def _get_alternative_parish_data(self):
        return get_cached_alternative_parish_data(self.log_capture)

    def run_analysis_streamlit(self):
        self.log_capture = []
        self._capture_print("=== Starting Terra Caribbean Analysis (Streamlit) ===")
        try:
            self._capture_print("\nLoading property data...")
            properties = self._load_property_data()

            if properties.empty: self._capture_print("No properties loaded. Aborting."); return False
            self._capture_print(f"\nLoaded and initially processed {len(properties)} properties")

            if 'Property Type Standardized' in properties.columns:
                self._capture_print("\nCounts of 'Property Type Standardized' after loading:")
                self._capture_print(str(properties['Property Type Standardized'].value_counts(dropna=False).to_string()))
            if 'Category' in properties.columns:
                self._capture_print("\nCounts of 'Category' after loading:")
                self._capture_print(str(properties['Category'].value_counts(dropna=False).to_string()))

            self._capture_print("\nDownloading/Loading geospatial data...")
            primary_parishes, self.beaches, self.tourism_points, self.feature_polygons = self._get_geodata()

            if primary_parishes.empty:
                self._capture_print("Primary geodata for parishes failed. Trying alternative parish data source...")
                self.parishes = self._get_alternative_parish_data()
                if self.parishes.empty:
                    self._capture_print("CRITICAL: All parish geospatial data sources failed. Aborting."); return False
            else:
                self.parishes = primary_parishes

            if not self.parishes.empty:
                self._capture_print(f"\nProcessing {len(self.parishes)} parishes for map display and matching.")
                problematic_strings_to_none = ['nan', 'NaN', 'Nan', 'NONE', 'None', 'none', '', 'null', 'Null', 'NULL', '<NA>']
                if 'name' in self.parishes.columns: self.parishes['name'] = self.parishes['name'].replace(problematic_strings_to_none, None, regex=False)
                if 'OSM_Parish_Name' in self.parishes.columns: self.parishes['OSM_Parish_Name'] = self.parishes['OSM_Parish_Name'].replace(problematic_strings_to_none, None, regex=False)
                else:
                    if 'name' in self.parishes.columns:
                        self.parishes['OSM_Parish_Name'] = self.parishes['name'].copy(); self.parishes['OSM_Parish_Name'] = self.parishes['OSM_Parish_Name'].replace(problematic_strings_to_none, None, regex=False)
                    else: self.parishes['OSM_Parish_Name'] = pd.Series([None] * len(self.parishes), index=self.parishes.index)
                if 'name' in self.parishes.columns: self.parishes['OSM_Parish_Name'] = self.parishes['OSM_Parish_Name'].fillna(self.parishes['name'])
                self.parishes['OSM_Parish_Name'] = self.parishes['OSM_Parish_Name'].fillna('Parish Name N/A').astype(str)

                if 'geometry' in self.parishes.columns:
                    if not isinstance(self.parishes, gpd.GeoDataFrame):
                        self.parishes = gpd.GeoDataFrame(self.parishes, geometry='geometry', crs="EPSG:4326")
                    try:
                        if self.parishes.crs is None: self.parishes.set_crs("EPSG:4326", inplace=True)
                        parishes_projected = self.parishes.to_crs("EPSG:32620") # UTM Zone 20N for Barbados
                        self.parishes['area_sqkm'] = parishes_projected.geometry.area / 1_000_000
                        self._capture_print("Calculated parish areas (sq km).")
                    except Exception as e:
                        self._capture_print(f"Error projecting parishes for area calculation: {e}")
                        self.parishes['area_sqkm'] = np.nan
                else:
                    self.parishes['area_sqkm'] = np.nan
            else:
                self._capture_print("No parish data was loaded; skipping parish name processing and area calculation.")
                if not self.parishes.empty: # Ensure column exists
                    self.parishes['area_sqkm'] = np.nan


            if not self.feature_polygons.empty: self._capture_print(f"Fetched {len(self.feature_polygons)} feature polygons.")
            self._capture_print("\nGeocoding properties...");
            geo_properties = self.geocode_properties_internal(properties, self.parishes.copy())
            if geo_properties.empty: self._capture_print("No properties geocoded. Aborting."); self.display_stats_internal(gpd.GeoDataFrame()); return False
            self._capture_print(f"\nSuccessfully geocoded {len(geo_properties)} properties.")
            self._capture_print("\nPerforming spatial analysis...");
            self.analyzed_properties = self.analyze_properties_internal(geo_properties, self.beaches, self.tourism_points)
            if self.analyzed_properties.empty: self._capture_print("Spatial analysis resulted in no properties. Aborting."); return False
            self._capture_print(f"\nSpatial analysis complete for {len(self.analyzed_properties)} properties.")
            self._capture_print("\nGenerating visualizations...")
            self.create_visualizations_internal(self.analyzed_properties, self.parishes.copy(), self.feature_polygons.copy(), self.tourism_points.copy())
            self.display_stats_internal(self.analyzed_properties)
            self._capture_print("\n=== Analysis complete! ===")
            self._capture_print(f"Final # of properties in analysis: {len(self.analyzed_properties)}")
            return True

        except Exception as e:
            self._capture_print(f"\nERROR during analysis: {str(e)}");
            self._capture_print(traceback.format_exc())
            return False

    def parse_size_to_sqft(self, size_str):
        return parse_size_to_sqft_static(size_str)

    def _clean_parish_name_generic(self, name_val):
        return clean_parish_name_generic_static(name_val)

    def geocode_properties_internal(self, df_props, parishes_gdf_in):
        try:
            if df_props.empty or parishes_gdf_in.empty: self._capture_print("Cannot geocode: Property data or Parish GDF is empty."); return gpd.GeoDataFrame()
            parishes_gdf = parishes_gdf_in.copy();
            if 'geometry' not in parishes_gdf.columns or parishes_gdf['geometry'].isna().all(): self._capture_print("CRIT: Parish GDF for geocoding has no valid geometries."); return gpd.GeoDataFrame()
            if parishes_gdf['geometry'].isna().any(): parishes_gdf.dropna(subset=['geometry'], inplace=True)
            if parishes_gdf.empty: self._capture_print("CRIT: All parish geometries were null after dropna."); return gpd.GeoDataFrame()

            def create_join_key(s): return s.astype(str).str.lower().str.replace(r'[^a-z0-9\s]','',regex=True).str.replace(r'\s+',' ',regex=True).str.strip()
            df_props['Parish_join_key'] = create_join_key(df_props['Parish'])

            if 'name' not in parishes_gdf.columns:
                self._capture_print("WARN: 'name' column (cleaned for join) missing in parishes_gdf. Attempting join with OSM_Parish_Name if available.")
                if 'OSM_Parish_Name' in parishes_gdf.columns:
                    parishes_gdf['name_join_key'] = create_join_key(parishes_gdf['OSM_Parish_Name'])
                else:
                    self._capture_print("CRIT ERROR: No suitable join key ('name' or 'OSM_Parish_Name') in parishes_gdf."); return gpd.GeoDataFrame()
            else:
                parishes_gdf['name_join_key'] = create_join_key(parishes_gdf['name'])

            parishes_gdf.drop_duplicates(subset=['name_join_key'], keep='first', inplace=True)

            cols_to_merge = ['name_join_key','OSM_Parish_Name','geometry']
            if 'area_sqkm' in parishes_gdf.columns:
                cols_to_merge.append('area_sqkm')

            gdf_merged = pd.merge(df_props, parishes_gdf[cols_to_merge], left_on='Parish_join_key', right_on='name_join_key', how='left', indicator=True)
            gdf_matched = gdf_merged[gdf_merged['geometry'].notna()].copy()

            if gdf_matched.empty and not df_props.empty: self._capture_print("CRIT: No properties could be matched to parish geometries after merge."); return gpd.GeoDataFrame()
            gdf_matched = gpd.GeoDataFrame(gdf_matched, geometry='geometry', crs=parishes_gdf.crs)

            if gdf_matched.crs is None:
                gdf_matched.set_crs("EPSG:4326", inplace=True)

            projected_crs = "EPSG:32620"
            try:
                gdf_matched_projected = gdf_matched.to_crs(projected_crs)
                gdf_matched['geometry'] = gdf_matched_projected.geometry.centroid.to_crs(gdf_matched.crs)
            except Exception as e:
                self._capture_print(f"WARN: Error projecting for centroid calculation, using geographic CRS: {e}")
                gdf_matched['geometry'] = gdf_matched.geometry.centroid

            gdf_matched.drop(columns=['_merge', 'name_join_key'], inplace=True, errors='ignore')

            return gdf_matched
        except Exception as e: self._capture_print(f"Error during geocoding: {str(e)}\n{traceback.format_exc()}"); return gpd.GeoDataFrame()

    def analyze_properties_internal(self, prop_gdf_in, beaches_in, tourism_points_in):
        prop_gdf = prop_gdf_in.copy();
        beaches_gdf = beaches_in.copy() if beaches_in is not None and not beaches_in.empty else gpd.GeoDataFrame();
        tourism_gdf = tourism_points_in.copy() if tourism_points_in is not None and not tourism_points_in.empty else gpd.GeoDataFrame()
        try:
            if prop_gdf.empty: return gpd.GeoDataFrame()
            if prop_gdf.crs is None: prop_gdf.set_crs("EPSG:4326",inplace=True)
            analysis_crs="EPSG:32620"; # Using UTM Zone 20N for Barbados distance calculations
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
        """Finds the parish name for given coordinates."""
        if not hasattr(self, 'parishes') or self.parishes.empty or pd.isna(lat) or pd.isna(lon):
            return "Parish data not available or coordinates invalid."

        try:
            point_geom = Point(lon, lat)
            point_gdf = gpd.GeoDataFrame([1], geometry=[point_geom], crs="EPSG:4326")

            parishes_4326 = self.parishes.to_crs("EPSG:4326")
            
            # Create a copy for filtering
            parishes_to_join = parishes_4326.copy()

            # Define a mask to filter out 'Barbados' by checking both relevant name columns
            mask_osm_name = pd.Series([True] * len(parishes_to_join), index=parishes_to_join.index)
            if 'OSM_Parish_Name' in parishes_to_join.columns:
                mask_osm_name = parishes_to_join['OSM_Parish_Name'].fillna('').astype(str).str.lower() != 'barbados'
            
            mask_name = pd.Series([True] * len(parishes_to_join), index=parishes_to_join.index)
            if 'name' in parishes_to_join.columns:
                mask_name = parishes_to_join['name'].fillna('').astype(str).str.lower() != 'barbados'
            
            # Combine masks: a row is kept if NEITHER of its name fields are 'Barbados'
            # (or if a name field doesn't exist, its mask remains all True, effectively not filtering on it)
            parishes_to_join = parishes_to_join[mask_osm_name & mask_name]

            if parishes_to_join.empty and not parishes_4326.empty:
                self._capture_print("WARN: Filtering out 'Barbados' from name columns removed all parish entries. Consider checking parish data source. Using unfiltered for now.")
                # Fallback if filtering removed everything, but this might re-introduce the problem
                # A better approach if this happens often is to ensure the source self.parishes is clean.
                parishes_to_join = parishes_4326 
            elif parishes_to_join.empty and parishes_4326.empty:
                return "No parish data to perform lookup."

            joined_gdf = gpd.sjoin(point_gdf, parishes_to_join, how="inner", predicate="within")

            if not joined_gdf.empty:
                # Prioritize OSM_Parish_Name, then name, ensuring it's not 'Barbados'
                # Iterate through matches if the first one is 'Barbados' or not specific enough
                found_parish = "Parish name not identified" 
                for idx, row in joined_gdf.iterrows():
                    osm_name = row.get('OSM_Parish_Name')
                    name_col = row.get('name')
                    
                    if osm_name and pd.notna(osm_name) and osm_name.strip().lower() not in ['', 'barbados', 'parish name n/a']:
                        found_parish = osm_name
                        break 
                    if name_col and pd.notna(name_col) and name_col.strip().lower() not in ['', 'barbados', 'parish name n/a']:
                        found_parish = name_col
                        break
                
                # If after iterating, we still have the default or a generic non-parish name from the first .iloc[0] approach:
                if found_parish == "Parish name not identified" or found_parish.lower() == 'barbados':
                     # Check the first result again with less strict conditions if loop failed.
                     first_match_osm = joined_gdf['OSM_Parish_Name'].iloc[0] if 'OSM_Parish_Name' in joined_gdf.columns else None
                     first_match_name = joined_gdf['name'].iloc[0] if 'name' in joined_gdf.columns else None
                     if first_match_osm and pd.notna(first_match_osm) and first_match_osm.lower() != 'barbados':
                         found_parish = first_match_osm
                     elif first_match_name and pd.notna(first_match_name) and first_match_name.lower() != 'barbados':
                         found_parish = first_match_name
                     # If it's still 'Barbados', it means that was the only non-generic match.
                     # This scenario should ideally be prevented by cleaner source data or more robust filtering above.

                return found_parish
            else:
                return "Not within a known parish boundary."
        except Exception as e:
            self._capture_print(f"Error finding parish: {e}\n{traceback.format_exc()}")
            return "Error during parish lookup."

    def create_map_object_streamlit(self, center_lat=None, center_lon=None, zoom=None):
        """Creates or updates the Folium map object."""
        if (not hasattr(self, 'analyzed_properties') or self.analyzed_properties.empty) and \
           (not hasattr(self, 'parishes') or self.parishes.empty):
            self._capture_print("Cannot create map: Missing analyzed properties or parishes.")
            return None

        analyzed_gdf = self.analyzed_properties.copy()
        all_parishes_gdf = self.parishes.copy()
        feature_polygons = self.feature_polygons.copy() if hasattr(self, 'feature_polygons') else gpd.GeoDataFrame()
        tourism_points = self.tourism_points.copy() if hasattr(self, 'tourism_points') else gpd.GeoDataFrame()
        parish_summary = self.parish_summary_df.copy() if hasattr(self, 'parish_summary_df') else pd.DataFrame()

        stamen_terrain_attribution = 'Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://www.openstreetmap.org/copyright">ODbL</a>.'
        open_topo_map_attribution = 'Map data: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, <a href="http://viewfinderpanoramas.org">SRTM</a> | Map style: &copy; <a href="https://opentopomap.org">OpenTopoMap</a> (<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)'
        carto_positron_attribution = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'

        map_lat = center_lat if center_lat is not None else 13.1939
        map_lon = center_lon if center_lon is not None else -59.5432
        map_zoom = zoom if zoom is not None else 10

        m = folium.Map(location=[map_lat, map_lon], zoom_start=map_zoom, tiles="Stamen Terrain", attr=stamen_terrain_attribution)
        folium.TileLayer("OpenTopoMap", name="Topographic Detail", show=False, attr=open_topo_map_attribution).add_to(m)
        folium.TileLayer("CartoDB positron", name="Light Base Map", show=False, attr=carto_positron_attribution).add_to(m)

        if not all_parishes_gdf.empty and 'OSM_Parish_Name' in all_parishes_gdf.columns and 'geometry' in all_parishes_gdf.columns:
            parishes_4326 = all_parishes_gdf.to_crs("EPSG:4326")[all_parishes_gdf.geometry.is_valid & all_parishes_gdf.geometry.notna()]
            if not parishes_4326.empty:
                parish_boundary_layer = folium.FeatureGroup(name="Parish Boundaries & Centers", show=True).add_to(m)
                folium.GeoJson(parishes_4326,style_function=lambda x: {'fillColor':'#D3D3D3','color':'#333333','weight':1.5,'fillOpacity':0.3},
                                 highlight_function=lambda x: {'weight':3, 'color':'#555555', 'fillOpacity':0.5},
                                 tooltip=folium.GeoJsonTooltip(fields=['OSM_Parish_Name'], aliases=['<b>Parish:</b>'], style=("background-color:white;color:black;font-family:Arial;font-size:12px;padding:5px;border-radius:3px;box-shadow:3px 3px 5px grey;"),sticky=True)
                                ).add_to(parish_boundary_layer)
                for idx, parish_row in parishes_4326.iterrows():
                    name = parish_row['OSM_Parish_Name']
                    try:
                        if parish_row.geometry.is_empty or not parish_row.geometry.is_valid: continue
                        centroid = parish_row.geometry.centroid;
                        if centroid.is_empty: continue
                        folium.CircleMarker(location=[centroid.y, centroid.x], radius=3, color='#4A4A4A', weight=1, fill=True, fill_color='#808080', fill_opacity=0.5, tooltip=f"<b>{name}</b> (Center)").add_to(parish_boundary_layer)
                    except Exception: pass

        if not feature_polygons.empty and 'name' in feature_polygons.columns and 'geometry' in feature_polygons.columns:
            feature_polygons_4326 = feature_polygons.to_crs("EPSG:4326")[feature_polygons.geometry.is_valid & feature_polygons.geometry.notna()]
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

        if not parish_summary.empty:
            parish_summary_layer = folium.FeatureGroup(name="Property Summaries by Parish", show=True).add_to(m)
            group_by_col = 'OSM_Parish_Name' if 'OSM_Parish_Name' in parish_summary.columns else 'Parish'
            for _, p_row in parish_summary.iterrows():
                if pd.isna(p_row['latitude']) or pd.isna(p_row['longitude']): continue
                loc=[p_row['latitude'],p_row['longitude']]; fill_color='#FF4500'; radius=max(8,min(10+(p_row['total_properties']/12),35))
                beach_km_str = f"{(p_row['avg_beach_dist_m']/1000):.1f} km" if pd.notna(p_row['avg_beach_dist_m']) else 'N/A'; avg_tour_str = f"{p_row['avg_tourism_count_2km']:.1f}" if pd.notna(p_row['avg_tourism_count_2km']) else "N/A"; avg_sz_str = f"{p_row['avg_size_sqft']:,.0f} sq ft" if pd.notna(p_row['avg_size_sqft']) else "N/A"
                area_val = p_row.get('area_sqkm')
                area_str = f"{area_val:.1f} sq km" if pd.notna(area_val) and area_val > 0 else "N/A"
                prop_density_str = "N/A"
                if pd.notna(area_val) and area_val > 0 and pd.notna(p_row['total_properties']) and p_row['total_properties'] > 0 :
                    density = p_row['total_properties'] / area_val
                    prop_density_str = f"{density:.1f} props/sq km"
                popup_html = f"""
                <div style="width:320px; font-family: Arial, sans-serif; font-size: 13px; line-height: 1.4;">
                    <h4 style="margin: 5px 0 10px 0; padding-bottom: 5px; border-bottom: 1px solid #ddd; color: #005A9C;">{p_row[group_by_col]} Summary</h4>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr style="background-color: #f9f9f9;"><td style="padding: 4px; font-weight: bold;">Total Properties:</td><td style="padding: 4px; text-align: right;">{p_row['total_properties']}</td></tr>
                        <tr><td style="padding: 4px; font-weight: bold;">Parish Area:</td><td style="padding: 4px; text-align: right;">{area_str}</td></tr>
                        <tr style="background-color: #f9f9f9;"><td style="padding: 4px; font-weight: bold;">Property Density:</td><td style="padding: 4px; text-align: right;">{prop_density_str}</td></tr>
                        <tr><td style="padding: 4px; font-weight: bold;">For Sale:</td><td style="padding: 4px; text-align: right;">{p_row['total_for_sale']}</td></tr>
                        <tr style="background-color: #f9f9f9;"><td style="padding: 4px; font-weight: bold; padding-left: 15px;">↳ Residential:</td><td style="padding: 4px; text-align: right;">{p_row['for_sale_residential_count']}</td></tr>
                        <tr><td style="padding: 4px; font-weight: bold; padding-left: 15px;">↳ Commercial:</td><td style="padding: 4px; text-align: right;">{p_row['for_sale_commercial_count']}</td></tr>
                        <tr style="background-color: #f9f9f9;"><td style="padding: 4px; font-weight: bold; padding-left: 15px;">↳ Land:</td><td style="padding: 4px; text-align: right;">{p_row.get('for_sale_land_count',0)}</td></tr>
                        <tr><td style="padding: 4px; font-weight: bold; padding-left: 15px;">↳ Other:</td><td style="padding: 4px; text-align: right;">{p_row.get('for_sale_other_std_count',0)}</td></tr>
                        <tr><td style="padding: 4px; font-weight: bold;">For Rent:</td><td style="padding: 4px; text-align: right;">{p_row['total_for_rent']}</td></tr>
                        <tr style="background-color: #f9f9f9;"><td style="padding: 4px; font-weight: bold; padding-left: 15px;">↳ Residential:</td><td style="padding: 4px; text-align: right;">{p_row['for_rent_residential_count']}</td></tr>
                        <tr><td style="padding: 4px; font-weight: bold; padding-left: 15px;">↳ Commercial:</td><td style="padding: 4px; text-align: right;">{p_row['for_rent_commercial_count']}</td></tr>
                        <tr style="background-color: #f9f9f9;"><td style="padding: 4px; font-weight: bold; padding-left: 15px;">↳ Land:</td><td style="padding: 4px; text-align: right;">{p_row.get('for_rent_land_count',0)}</td></tr>
                        <tr><td style="padding: 4px; font-weight: bold; padding-left: 15px;">↳ Other:</td><td style="padding: 4px; text-align: right;">{p_row.get('for_rent_other_std_count',0)}</td></tr>
                        <tr style="background-color: #f9f9f9;"><td style="padding: 4px; font-weight: bold;">Avg. Beach Dist:</td><td style="padding: 4px; text-align: right;">{beach_km_str}</td></tr>
                        <tr><td style="padding: 4px; font-weight: bold;">Avg. Attractions (2km):</td><td style="padding: 4px; text-align: right;">{avg_tour_str}</td></tr>
                        <tr style="background-color: #f9f9f9;"><td style="padding: 4px; font-weight: bold;">Avg. Size:</td><td style="padding: 4px; text-align: right;">{avg_sz_str}</td></tr>
                    </table>
                </div>
                """
                folium.CircleMarker(loc, radius=radius, color='#000000', weight=2, fill=True, fill_color=fill_color, fill_opacity=0.7, popup=folium.Popup(popup_html, max_width=350), tooltip=f"<b>{p_row[group_by_col]}</b><br>Total Data Properties: {p_row['total_properties']}").add_to(parish_summary_layer)

        if not tourism_points.empty and 'name' in tourism_points.columns and 'geometry' in tourism_points.columns:
            tourism_4326 = tourism_points.to_crs("EPSG:4326")[tourism_points.geometry.is_valid & tourism_points.geometry.notna()]
            if not tourism_4326.empty:
                landmark_pts_layer = folium.FeatureGroup(name="Points of Interest (Tourism)", show=False).add_to(m)
                for idx, pt_row in tourism_4326.iterrows():
                    name = pt_row.get('name','POI'); pt_geom=pt_row.geometry;
                    if pt_geom.geom_type!='Point': pt_geom=pt_geom.representative_point()
                    if pt_geom.is_empty: continue
                    folium.Marker([pt_geom.y,pt_geom.x],tooltip=f"<b>{name}</b><br><small>{pt_row.get('tourism','N/A').replace('_',' ').title()}</small>", icon=folium.Icon(color='blue',icon='info-sign',prefix='fa')).add_to(landmark_pts_layer)

        # Add marker for the located point if it was set by the user
        if center_lat is not None and center_lon is not None and st.session_state.get('location_explicitly_set', False):
            folium.Marker(
                location=[center_lat, center_lon],
                popup=f"Located: {center_lat:.4f}, {center_lon:.4f}",
                tooltip="Your Location",
                icon=folium.Icon(color="red", icon="star"),
            ).add_to(m)

        folium.LayerControl().add_to(m)
        return m


    def create_visualizations_internal(self, analyzed_gdf_in, all_parishes_gdf_in, feature_polygons_in, tourism_points_in):
        analyzed_gdf = analyzed_gdf_in.copy() if analyzed_gdf_in is not None and not analyzed_gdf_in.empty else gpd.GeoDataFrame()
        try:
            if analyzed_gdf.empty:
                self._capture_print("No analyzed data to create visualizations.")
                self.map_html_content = None
                self.parish_summary_df = pd.DataFrame()
                return

            # --- Calculate and Store Parish Summary ---
            if 'Property Type Standardized' not in analyzed_gdf.columns:
                analyzed_gdf['Property Type Standardized'] = standardize_property_type_static(analyzed_gdf.get('Property Type', pd.Series(dtype=str)))
            analyzed_gdf['Category'] = analyzed_gdf.get('Category', pd.Series(dtype=str)).fillna('Unknown').astype(str).str.strip()
            analyzed_gdf['Property Type Standardized'] = analyzed_gdf.get('Property Type Standardized', pd.Series(dtype=str)).fillna('Other').astype(str).str.strip()
            group_by_col = 'OSM_Parish_Name' if 'OSM_Parish_Name' in analyzed_gdf.columns and analyzed_gdf['OSM_Parish_Name'].notna().any() else 'Parish'

            if group_by_col in analyzed_gdf.columns and analyzed_gdf[group_by_col].notna().any():
                def summarize_parish(group):
                    data = {}
                    category = group['Category']
                    prop_type = group['Property Type Standardized']
                    data['total_properties'] = int(len(group))
                    is_sale = category.str.upper() != 'FOR RENT'
                    is_rent = category.str.upper() == 'FOR RENT'
                    is_res = prop_type == 'Residential'
                    is_com = prop_type == 'Commercial'
                    is_land = prop_type == 'Land'
                    is_oth = prop_type == 'Other'
                    data['for_sale_residential_count'] = int((is_sale & is_res).sum())
                    data['for_rent_residential_count'] = int((is_rent & is_res).sum())
                    data['for_sale_commercial_count'] = int((is_sale & is_com).sum())
                    data['for_rent_commercial_count'] = int((is_rent & is_com).sum())
                    data['for_sale_land_count'] = int((is_sale & is_land).sum())
                    data['for_rent_land_count'] = int((is_rent & is_land).sum())
                    data['for_sale_other_std_count'] = int((is_sale & is_oth).sum())
                    data['for_rent_other_std_count'] = int((is_rent & is_oth).sum())
                    data['total_for_sale'] = int(is_sale.sum())
                    data['total_for_rent'] = int(is_rent.sum())
                    data['avg_beach_dist_m'] = group['beach_dist_m'].mean()
                    data['avg_tourism_count_2km'] = group['tourism_count_2km'].mean()
                    data['avg_size_sqft'] = group['Size_sqft'].mean()
                    data['area_sqkm'] = group['area_sqkm'].iloc[0] if 'area_sqkm' in group.columns and not group['area_sqkm'].empty and pd.notna(group['area_sqkm'].iloc[0]) else np.nan
                    if not group.empty and 'geometry' in group.columns and group['geometry'].iloc[0] is not None:
                        first_geom = group['geometry'].iloc[0]
                        data['latitude'] = first_geom.y if hasattr(first_geom, 'y') else np.nan
                        data['longitude'] = first_geom.x if hasattr(first_geom, 'x') else np.nan
                    else: data['latitude'], data['longitude'] = np.nan, np.nan
                    return pd.Series(data)
                self.parish_summary_df = analyzed_gdf.groupby(group_by_col, dropna=False).apply(summarize_parish).reset_index()
            else:
                self.parish_summary_df = pd.DataFrame()
            # --- End Parish Summary Calculation ---


            # --- Generate Initial Map ---
            m = self.create_map_object_streamlit() # Call with defaults
            if m:
                self.map_html_content = m.get_root().render()
                self._capture_print("\nInteractive map data generated.")
            else:
                self.map_html_content = None
                self._capture_print("\nCould not generate map - data missing?")
            # --- End Initial Map ---


            # --- Generate Chart ---
            fig_chart, ax_chart = plt.subplots(figsize=(10,6))
            group_by_col_chart = 'OSM_Parish_Name' if 'OSM_Parish_Name' in analyzed_gdf.columns else 'Parish'
            unique_parishes_for_chart = sorted(analyzed_gdf[group_by_col_chart].unique()) if group_by_col_chart in analyzed_gdf and not analyzed_gdf.empty else []
            if unique_parishes_for_chart:
                cmap = plt.cm.get_cmap('viridis', len(unique_parishes_for_chart))
                colors = [cmap(i) for i in range(len(unique_parishes_for_chart))]

                for i, p_name_chart in enumerate(unique_parishes_for_chart):
                    p_data = analyzed_gdf[analyzed_gdf[group_by_col_chart]==p_name_chart].dropna(subset=['beach_dist_m','Price'])
                    if not p_data.empty: ax_chart.scatter(p_data['beach_dist_m']/1000, p_data['Price'], label=p_name_chart, alpha=0.65, s=45, color=colors[i])
            ax_chart.set_title("Property Prices vs. Beach Distance by Parish",fontsize=14)
            ax_chart.set_xlabel("Distance to Nearest Beach (km)",fontsize=12)
            ax_chart.set_ylabel("Price (USD)",fontsize=12)
            if unique_parishes_for_chart: ax_chart.legend(title="Parish", fontsize='small', title_fontsize='medium')
            ax_chart.grid(True, linestyle=':', alpha=0.7); ax_chart.ticklabel_format(style='plain',axis='y'); fig_chart.tight_layout()
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile_chart:
                self.chart_path = tmpfile_chart.name
            fig_chart.savefig(self.chart_path,dpi=100,bbox_inches='tight'); plt.close(fig_chart);
            self._capture_print(f"Price vs. Beach Distance chart generated: {self.chart_path}");
            # --- End Chart ---

        except Exception as e: self._capture_print(f"Error visualizations: {e}\n{traceback.format_exc()}");


    def display_stats_internal(self, stats_gdf_in):
        self.stats_data_for_streamlit = []
        stats_gdf = stats_gdf_in.copy() if stats_gdf_in is not None else gpd.GeoDataFrame()
        try:
            if stats_gdf.empty: self.stats_data_for_streamlit=["No data available for statistics."]
            else:
                num_analyzed = len(stats_gdf); num_parishes = stats_gdf[('OSM_Parish_Name' if 'OSM_Parish_Name' in stats_gdf.columns else 'Parish')].nunique() if ('OSM_Parish_Name' in stats_gdf.columns or 'Parish' in stats_gdf.columns) else 0
                avg_beach_dist_val = stats_gdf['beach_dist_m'].mean() if 'beach_dist_m' in stats_gdf and stats_gdf['beach_dist_m'].notna().any() else np.nan
                avg_beach_dist_str = f"{(avg_beach_dist_val/1000):.1f} km" if pd.notna(avg_beach_dist_val) else 'N/A'
                h_pr = f"${stats_gdf['Price'].max():,.0f}" if 'Price' in stats_gdf and stats_gdf['Price'].notna().any() else 'N/A'; l_pr = f"${stats_gdf['Price'].min():,.0f}" if 'Price' in stats_gdf and stats_gdf['Price'].notna().any() else 'N/A'
                avg_tr_val = stats_gdf['tourism_count_2km'].mean() if 'tourism_count_2km' in stats_gdf and stats_gdf['tourism_count_2km'].notna().any() else np.nan
                avg_tr_str = f"{avg_tr_val:.1f}" if pd.notna(avg_tr_val) else 'N/A'
                s_cnt = (stats_gdf['Category'].astype(str).str.strip().str.upper()!='FOR RENT').sum() if 'Category' in stats_gdf else 0; r_cnt = (stats_gdf['Category'].astype(str).str.strip().str.upper()=='FOR RENT').sum() if 'Category' in stats_gdf else 0
                avg_sz_val = stats_gdf['Size_sqft'].mean() if 'Size_sqft' in stats_gdf and stats_gdf['Size_sqft'].notna().any() else np.nan
                avg_sz_str = f"{avg_sz_val:,.0f} sq ft" if pd.notna(avg_sz_val) else "N/A"
                self.stats_data_for_streamlit = [f"Total Properties Analyzed: {num_analyzed}",f"Unique Parishes in Data: {num_parishes}", f"Avg Beach Dist: {avg_beach_dist_str}",
                                                 f"Highest Property Price: {h_pr}",f"Lowest Property Price: {l_pr}", f"Avg Attractions (2km): {avg_tr_str}",
                                                 f"Properties For Sale: {s_cnt}",f"Properties For Rent: {r_cnt}",f"Average Property Size: {avg_sz_str}"]
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
            if 'area_sqkm' in df_to_export.columns: cols_to_drop.append('area_sqkm')
            actual_cols_to_drop = [col for col in cols_to_drop if col in df_to_export.columns]
            if actual_cols_to_drop:
                df_to_export = df_to_export.drop(columns=actual_cols_to_drop)
            return df_to_export
        return pd.DataFrame()

# --- Streamlit App UI and Main Logic ---
def main():
    st.set_page_config(page_title="Terra Caribbean: Terrain View", layout="wide", initial_sidebar_state="expanded")

    # Initialize session state variables if they don't exist
    if 'analysis_triggered' not in st.session_state: st.session_state.analysis_triggered = False
    if 'dashboard_logic_instance' not in st.session_state: st.session_state.dashboard_logic_instance = None
    if 'error_during_analysis' not in st.session_state: st.session_state.error_during_analysis = False
    if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0
    if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
    if 'location_explicitly_set' not in st.session_state: st.session_state.location_explicitly_set = False
    if 'needs_map_update' not in st.session_state: st.session_state.needs_map_update = False
    if 'location_info' not in st.session_state: st.session_state.location_info = None


    LOGO_URL = "https://s3.us-east-2.amazonaws.com/terracaribbean.com/wp-content/uploads/2025/04/08080016/site-logo.png"

    dashboard_instance = st.session_state.get('dashboard_logic_instance')

    with st.sidebar:
        st.image(LOGO_URL, width=200)
        st.header("Controls")
        uploaded_file = st.file_uploader("Upload Property Data (Excel .xlsx or CSV .csv)",
                                         type=["xlsx", "csv"],
                                         key=f"file_uploader_{st.session_state.uploader_key}")
        status_placeholder = st.empty()
        run_button_clicked = st.button("🚀 Run Analysis", use_container_width=True)

        # --- Location Finder ---
        if st.session_state.get('analysis_done') and dashboard_instance: # Check instance exists
            st.markdown("---")
            st.header("Locate on Map")
            # Use session state to keep the values after input
            lat_val = st.session_state.get('map_center_lat_input', 13.1939)
            lon_val = st.session_state.get('map_center_lon_input', -59.5432)
            zoom_val = st.session_state.get('map_zoom_input', 14)

            lat_input = st.number_input("Latitude:", value=lat_val, format="%.6f", key="lat_input_widget")
            lon_input = st.number_input("Longitude:", value=lon_val, format="%.6f", key="lon_input_widget")
            zoom_input = st.slider("Zoom Level:", min_value=10, max_value=18, value=zoom_val, key="zoom_input_widget")
            go_button_clicked = st.button("📍 Go to Location", use_container_width=True)

            if go_button_clicked:
                st.session_state.map_center_lat = lat_input
                st.session_state.map_center_lon = lon_input
                st.session_state.map_zoom = zoom_input
                st.session_state.location_explicitly_set = True
                st.session_state.map_center_lat_input = lat_input
                st.session_state.map_center_lon_input = lon_input
                st.session_state.map_zoom_input = zoom_input
                st.session_state.needs_map_update = True

                # Calculate and store info
                lat_dms = decimal_to_dms(lat_input, True)
                lon_dms = decimal_to_dms(lon_input, False)
                identified_parish_name = dashboard_instance.get_parish_for_coords(lat_input, lon_input)

                parish_area_str = "N/A"
                parish_avg_beach_dist_str = "N/A"
                parish_avg_tourism_count_str = "N/A"

                valid_parish_identified = identified_parish_name and \
                                           identified_parish_name not in ["Not within a known parish boundary.",
                                                                          "Error during parish lookup.",
                                                                          "Parish data not available or coordinates invalid.",
                                                                          "Parish name not identified",
                                                                          "Parish Name Found (but empty)"]

                if valid_parish_identified:
                    if hasattr(dashboard_instance, 'parish_summary_df') and not dashboard_instance.parish_summary_df.empty:
                        summary_parish_col = None
                        if 'OSM_Parish_Name' in dashboard_instance.parish_summary_df.columns:
                            summary_parish_col = 'OSM_Parish_Name'
                        elif 'Parish' in dashboard_instance.parish_summary_df.columns:
                            summary_parish_col = 'Parish'
                        
                        if summary_parish_col:
                            # Ensure identified_parish_name is a string for comparison
                            parish_stats = dashboard_instance.parish_summary_df[
                                dashboard_instance.parish_summary_df[summary_parish_col].astype(str) == str(identified_parish_name)
                            ]

                            if not parish_stats.empty:
                                stats_row = parish_stats.iloc[0]
                                
                                area_val = stats_row.get('area_sqkm')
                                parish_area_str = f"{area_val:.1f} sq km" if pd.notna(area_val) and area_val > 0 else "N/A"
                                
                                beach_dist_val = stats_row.get('avg_beach_dist_m')
                                parish_avg_beach_dist_str = f"{(beach_dist_val / 1000):.1f} km" if pd.notna(beach_dist_val) else "N/A"
                                
                                tourism_count_val = stats_row.get('avg_tourism_count_2km')
                                parish_avg_tourism_count_str = f"{tourism_count_val:.1f}" if pd.notna(tourism_count_val) else "N/A"

                st.session_state.location_info = {
                    "lat_dec": lat_input,
                    "lon_dec": lon_input,
                    "lat_dms": lat_dms,
                    "lon_dms": lon_dms,
                    "parish": identified_parish_name,
                    "parish_area": parish_area_str,
                    "parish_avg_beach_dist": parish_avg_beach_dist_str,
                    "parish_avg_tourism_count": parish_avg_tourism_count_str
                }
                st.rerun()

            if st.session_state.location_info and st.session_state.location_explicitly_set:
                info = st.session_state.location_info
                st.markdown("---")
                st.subheader("Point Location Details:")
                st.markdown(f"**Coordinates (Entered Point):**")
                st.markdown(f"&nbsp;&nbsp;&nbsp;Lat: {info['lat_dec']:.6f} / {info['lat_dms']}")
                st.markdown(f"&nbsp;&nbsp;&nbsp;Lon: {info['lon_dec']:.6f} / {info['lon_dms']}")
                
                st.markdown(f"**Identified Parish:**")
                st.markdown(f"&nbsp;&nbsp;&nbsp;Name: {info['parish']}")

                valid_parish_for_stats = info["parish"] and \
                                          info["parish"] not in ["Not within a known parish boundary.",
                                                                  "Error during parish lookup.",
                                                                  "Parish data not available or coordinates invalid.",
                                                                  "Parish name not identified",
                                                                  "Parish Name Found (but empty)"]
                if valid_parish_for_stats:
                    st.markdown(f"**Parish Level Summary ({info['parish']}):**")
                    st.markdown(f"&nbsp;&nbsp;&nbsp;Area: {info.get('parish_area', 'N/A')}")
                    st.markdown(f"&nbsp;&nbsp;&nbsp;Avg. Beach Distance: {info.get('parish_avg_beach_dist', 'N/A')}")
                    st.markdown(f"&nbsp;&nbsp;&nbsp;Avg. Attractions (2km): {info.get('parish_avg_tourism_count', 'N/A')}")
        # --- End Location Finder ---


    st.title("🏞️ Terra Caribbean: Terrain & Property View")
    st.markdown("Interactive terrain-focused map to analyze property listings with geospatial data for Barbados.")
    st.markdown("---")
    st.info("""
    **Understanding the Terrain Map:**
    This view emphasizes the island's topography. Key features include:
    * **Terrain Base Map:** See hills, valleys, and coastal features. Optional topographic layers are available in the map's layer control.
    * **Parish Boundaries & Centers:** Clearly marked parishes.
    * **Property Summaries:** Click on parish markers for aggregated property data, including Residential/Commercial splits.
    * **Locate on Map (Sidebar):** Enter Latitude and Longitude, click "Go to Location" to center the map, add a marker, and see location details in the sidebar.
    * *Other layers (Key Land Features, Tourism POIs) are hidden by default but can be enabled via the layer control (usually top-right of the map).*
    """)

    if run_button_clicked:
        if uploaded_file is not None:
            # Reset location state on new analysis
            st.session_state.location_explicitly_set = False
            st.session_state.needs_map_update = False
            st.session_state.location_info = None # Clear info
            if 'map_center_lat' in st.session_state: del st.session_state.map_center_lat
            if 'map_center_lon' in st.session_state: del st.session_state.map_center_lon
            if 'map_zoom' in st.session_state: del st.session_state.map_zoom
            if 'map_center_lat_input' in st.session_state: del st.session_state.map_center_lat_input
            if 'map_center_lon_input' in st.session_state: del st.session_state.map_center_lon_input
            if 'map_zoom_input' in st.session_state: del st.session_state.map_zoom_input

            st.session_state.analysis_triggered = True
            st.session_state.error_during_analysis = False
            st.session_state.dashboard_logic_instance = None
            with status_placeholder.container():
                with st.spinner("Analysis in progress... This may take a few minutes for the first run or new data."):
                    dashboard = TerraDashboardLogic(uploaded_file_object=uploaded_file)
                    analysis_successful = dashboard.run_analysis_streamlit()
                if analysis_successful:
                    st.session_state.dashboard_logic_instance = dashboard
                    st.session_state.analysis_done = True
                    st.session_state.needs_map_update = True # Trigger initial map generation
                    status_placeholder.success("Analysis Complete!")
                    st.rerun() # Force rerun to process map generation and sidebar update.
                else:
                    st.session_state.dashboard_logic_instance = dashboard
                    st.session_state.analysis_done = False
                    st.session_state.error_during_analysis = True
                    status_placeholder.error("Analysis failed. Check Console Log tab.")
            st.session_state.uploader_key += 1 # Rerender uploader


        else:
            st.sidebar.warning("⚠️ Please upload a property data file first!")
            st.session_state.analysis_done = False

    dashboard_instance = st.session_state.get('dashboard_logic_instance') # Re-get instance after potential rerun
    if dashboard_instance:
        tab_map, tab_chart, tab_stats, tab_parish_summary, tab_export, tab_console, tab_info = st.tabs([
            "🗺️ Terrain Map", "📊 Chart", "📈 Key Statistics", "📊 Parish Summary",
            "📥 Export Data", "📋 Console Log", "ℹ️ Calculations & Notes"
        ])

        with tab_map:
            st.subheader("Interactive Terrain Map with Property Summaries")
            if st.session_state.get('analysis_done') and dashboard_instance:
                # Decide if we need to regenerate the map
                regenerate_map = st.session_state.get('needs_map_update', False)

                if regenerate_map or not dashboard_instance.map_html_content:
                    with st.spinner("Generating / Updating map..."):
                        map_lat = st.session_state.get('map_center_lat') # Use session state or None
                        map_lon = st.session_state.get('map_center_lon') # Use session state or None
                        map_zoom = st.session_state.get('map_zoom') # Use session state or None

                        map_obj = dashboard_instance.create_map_object_streamlit(
                           center_lat=map_lat,
                           center_lon=map_lon,
                           zoom=map_zoom
                        )
                        if map_obj:
                            dashboard_instance.map_html_content = map_obj.get_root().render()
                        else:
                             dashboard_instance.map_html_content = "<p>Error generating map. Check if data was loaded correctly.</p>"
                    st.session_state.needs_map_update = False # Reset flag after update

                if dashboard_instance.map_html_content:
                    st.components.v1.html(dashboard_instance.map_html_content, height=700, scrolling=True)
                else:
                    st.info("Map could not be generated.")

            elif st.session_state.get('analysis_triggered'): st.info("Map is being generated or was not created. If analysis failed, check logs.")
            else: st.info("Map will be displayed here after analysis is run.")


        with tab_chart:
            st.subheader("Price vs. Beach Distance Chart")
            if st.session_state.get('analysis_done') and dashboard_instance.chart_path and os.path.exists(dashboard_instance.chart_path):
                try:
                    image = Image.open(dashboard_instance.chart_path)
                    st.image(image, caption="Property Price vs. Distance to Nearest Beach", use_container_width=True)
                except Exception as e: st.error(f"Could not load chart: {e}")
            elif st.session_state.get('analysis_triggered'): st.info("Chart is being generated or was not created. If analysis failed, check logs.")
            else: st.info("Chart will be displayed here after analysis is run.")

        with tab_stats:
            st.subheader("Key Statistics")
            if st.session_state.get('analysis_done') and hasattr(dashboard_instance, 'stats_data_for_streamlit') and dashboard_instance.stats_data_for_streamlit:
                for item in dashboard_instance.stats_data_for_streamlit: st.markdown(f"- {item}")
            elif st.session_state.get('analysis_triggered'): st.info("Statistics are being generated.")
            else: st.info("Key statistics will be displayed here after analysis is run.")

        with tab_parish_summary:
            st.subheader("Consolidated Parish Summary")
            if st.session_state.get('analysis_done') and hasattr(dashboard_instance, 'parish_summary_df') and not dashboard_instance.parish_summary_df.empty:
                summary_df = dashboard_instance.parish_summary_df.copy()
                display_df = summary_df.drop(columns=['latitude', 'longitude'], errors='ignore')
                st.dataframe(display_df)

                @st.cache_data
                def convert_summary_df_to_csv(df_to_convert):
                    return df_to_convert.to_csv(index=False).encode('utf-8')

                csv_summary_data = convert_summary_df_to_csv(display_df)
                st.download_button(
                    label="Download Parish Summary as CSV",
                    data=csv_summary_data,
                    file_name="terra_parish_summary.csv",
                    mime="text/csv",
                    key="download_summary_csv_button"
                )
            elif st.session_state.get('analysis_triggered'): st.info("Parish summary is being generated.")
            else: st.info("Parish summary data will be displayed here after analysis is run.")

        with tab_export:
            st.subheader("Export Analyzed Data")
            if st.session_state.get('analysis_done'):
                export_df = dashboard_instance.get_export_dataframe()
                if not export_df.empty:
                    @st.cache_data
                    def convert_df_to_csv(df_to_convert): return df_to_convert.to_csv(index=False).encode('utf-8')
                    csv_data = convert_df_to_csv(export_df)
                    st.download_button(label="Download Analyzed Data as CSV", data=csv_data, file_name="terra_analysis_results.csv", mime="text/csv", key="download_csv_button")
                elif st.session_state.get('analysis_triggered') : st.info("No analyzed data available for export (result might be empty).")
            else: st.info("Analyzed data will be available for export here after analysis is run.")

        with tab_console:
            st.subheader("Analysis Log")
            if hasattr(dashboard_instance, 'log_capture') and dashboard_instance.log_capture:
                log_content = "".join(dashboard_instance.log_capture)
                st.text_area("Log Output:", value=log_content, height=500, key="console_log_area_display", disabled=True)
            elif st.session_state.get('analysis_triggered'): st.info("Attempting to capture logs...")
            else: st.info("Console logs from the analysis will appear here.")

        with tab_info:
            st.subheader("Understanding the Calculations, Data & Conversions")
            st.markdown("""
            This section explains how the parish area figures are calculated and provides useful unit conversions.

            **1. How Geographic Areas are Calculated:**

            The areas of geographical units like Barbados's parishes are usually calculated using **Geographic Information Systems (GIS)**. Here's a simplified overview:

            * **Data Source:** The process starts with a digital map. This map data can come from various sources:
                * *Official Government Surveys:* National land and survey departments often provide the most authoritative boundary data.
                * *OpenStreetMap (OSM):* As used in this app, this is a collaborative, open-source mapping project. Its data is constantly updated but can sometimes vary from official sources.
                * *Satellite Imagery:* Used for tracing boundaries.
            * **Defining Boundaries:** Parishes are represented as **polygons** (shapes) on the map, defined by coordinates.
            * **Map Projections:** The Earth is 3D, maps are 2D. A **map projection** translates coordinates. This app uses `EPSG:4326` (WGS 84 - standard latitude/longitude) for display and `EPSG:32620` (UTM Zone 20N) for calculations (like area and distance) because it's better suited for measuring in meters in Barbados. *This choice influences the final area numbers.*
            * **Area Calculation:** GIS software calculates the area within each polygon using the chosen projection. The app calculates `area_sqkm` by projecting to `EPSG:32620` and calculating the geometric area.
            * **Why Differences Occur:** Figures can differ due to:
                * Variations in mapped boundaries (OSM vs. Official).
                * Different map projections used.
                * Inclusion/exclusion of small features.
                * Data update cycles.

            **2. Key Data Processing Steps in this App:**

            * **Parish Loading:** Attempts to load Parish boundaries primarily from OpenStreetMap (OSM) or a local GeoJSON file as a fallback.
            * **Parish Cleaning:** Standardizes Parish names from your uploaded file and the map data (e.g., 'St. James' becomes 'Saint James') for better matching.
            * **Geocoding:** Matches properties to parishes based on the cleaned names and assigns a *central point (centroid)* within the matched parish as the property's location. **Note:** This is a *parish-level* geocoding, not a precise address-point geocoding.
            * **Area Calculation:** Calculates parish area in square kilometers (`area_sqkm`) using the `EPSG:32620` projection.
            * **Distance Calculation:** Calculates distance to the nearest beach (`beach_dist_m`) using `EPSG:32620`.
            * **Tourism Count:** Counts tourism points within a 2km radius, also using `EPSG:32620`.
            * **Size Parsing:** Converts the 'Size' column from your file into square feet (`Size_sqft`), handling 'acres' and 'sq ft' units.

            **3. Important Unit Conversions:**

            * **Square Kilometers (km²) to Square Miles (sq mi):**
                * `1 km² ≈ 0.386102 sq mi`
            * **Square Kilometers (km²) to Hectares (ha):**
                * `1 km² = 100 ha`
            * **Square Kilometers (km²) to Acres:**
                * `1 km² ≈ 247.105 acres`
            * **Acres to Square Feet (sq ft):**
                * `1 acre = 43,560 sq ft`
            * **Meters (m) to Kilometers (km):**
                * `1 km = 1,000 m`
            * **Meters (m) to Miles (mi):**
                * `1 mi ≈ 1609.34 m`

            **4. Data Source & Limitations:**

            * **Primary Geo Source:** OpenStreetMap (OSM). Data quality can vary.
            * **Geocoding Accuracy:** Property locations are based on *parish centroids*, not specific addresses.
            * **Calculations:** Based on available OSM data and specific projections; may differ from official land survey figures.
            * **Uploaded Data:** The accuracy of the analysis heavily depends on the quality and format of your uploaded property file.
            """)
            st.markdown("---")
            st.write("Raw OSM Parish Data (if loaded):")
            if hasattr(dashboard_instance, 'raw_parishes_from_osm') and not dashboard_instance.raw_parishes_from_osm.empty:
                st.dataframe(dashboard_instance.raw_parishes_from_osm.drop(columns='geometry', errors='ignore'))
            else:
                st.info("Raw OSM parish data was not loaded or is empty.")

        if st.session_state.get('error_during_analysis'): st.error("An error occurred during the last analysis. Please review the console log in the 'Console Log' tab.")

    elif st.session_state.get('analysis_triggered', False) and st.session_state.get('error_during_analysis', False):
        st.error("Analysis failed. Check the console log if available from a partial run.")
        if st.session_state.get('dashboard_logic_instance') and hasattr(st.session_state.dashboard_logic_instance, 'log_capture') and st.session_state.dashboard_logic_instance.log_capture:
            st.subheader("Partial Analysis Log")
            log_content = "".join(st.session_state.dashboard_logic_instance.log_capture)
            st.text_area("Log Output:", value=log_content, height=500, key="error_console_log_area_display_alt", disabled=True)
    else: st.info("👋 Welcome! Please upload a property data file and click '🚀 Run Analysis' in the sidebar to begin.")

    st.markdown("---")
    property_count = 0
    dashboard_instance = st.session_state.get('dashboard_logic_instance')
    if st.session_state.get('analysis_done') and dashboard_instance and hasattr(dashboard_instance, 'analyzed_properties') and not dashboard_instance.analyzed_properties.empty:
        property_count = len(dashboard_instance.analyzed_properties)

    current_year = datetime.datetime.now().year
    credits_line1 = "Data Sources: Terra Caribbean, OpenStreetMap"
    if property_count > 0: credits_line1 += f" • Displaying {property_count} properties."
    credits_line2 = f"© {current_year} Terra Caribbean Geospatial Analytics Platform • All Prices in USD"
    credits_line3 = "App Created by Matthew Blackman. Assisted by AI."
    st.caption(f"{credits_line1}\n{credits_line2}\n{credits_line3}")


if __name__ == "__main__":
    main()