# Terra Caribbean: Geospatial Property Intelligence - Streamlit Edition

## Overview

This Streamlit application provides an interactive platform for visualizing and analyzing Terra Caribbean property listings within Barbados. It leverages geospatial data to offer insights into property locations, proximity to key features, and market trends across different parishes. The application generates an interactive map, summary statistics, charts, and allows for data export.

## Key Features

* **Data Upload:** Supports uploading property data via Excel (`.xlsx`) or CSV (`.csv`) files.
* **Data Cleaning & Standardization:**
    * Automatically identifies and renames key columns (e.g., `Parish`, `Price`, `Size`).
    * Cleans and standardizes Parish names for accurate matching.
    * Standardizes property types (Residential, Commercial, Land, Other).
    * Parses and converts various property size formats into square feet.
* **Geospatial Data Integration:**
    * Fetches and caches geospatial data from OpenStreetMap (OSM) via `OSMnx`.
    * Includes layers for:
        * Parish Boundaries
        * Beaches
        * Tourism Points of Interest (POIs)
        * Key Land Features (Parks, Golf Courses, etc.)
        * **New:** Schools
        * **New:** Supermarkets
    * Supports an optional local `barbados_parishes.geojson` file as a fallback.
* **Geocoding:** Approximates property locations by assigning them the centroid (center point) of their respective parish.
* **Spatial Analysis:**
    * Calculates the distance from each property to the nearest beach.
    * Counts the number of tourism POIs within a 2km radius of each property.
* **Interactive Visualization (Folium Map):**
    * Displays parish boundaries and centers.
    * Shows aggregated property summary data for each parish (counts, types, averages) on clickable markers.
    * Allows toggling different map layers (Base maps, POIs, Schools, Supermarkets, etc.).
* **Location Finder:** Center the map on specific Latitude/Longitude coordinates and view details about that point, including its parish and parish-level statistics.
* **Distance Calculator:** Calculate the direct "as-the-crow-flies" (Haversine) distance between any two points on the island.
* **Reporting & Export:**
    * Generates a scatter plot showing Property Prices vs. Beach Distance.
    * Displays key summary statistics for the entire dataset.
    * Provides a detailed parish-by-parish summary table.
    * Allows downloading the analyzed and enriched property data as a CSV file.
* **Logging:** Includes a console log tab to monitor the analysis process and troubleshoot issues.

## How it Works

1.  **Upload:** The user uploads a property data file.
2.  **Load & Clean:** The script reads the file, identifies and renames columns, cleans parish names, parses prices and sizes, and standardizes property types.
3.  **Fetch GeoData:** It downloads (or uses cached) boundary and point-of-interest data from OpenStreetMap. If OSM fails for parishes, it tries a local GeoJSON file.
4.  **Geocode:** It merges the property data with the parish geometries, assigning the parish centroid's latitude and longitude to each property within that parish.
5.  **Analyze:** It calculates distances (beaches) and proximity counts (tourism) using spatial joins and KD-Trees.
6.  **Visualize:** It generates an interactive Folium map with multiple layers and popups.
7.  **Summarize:** It calculates overall and parish-level statistics.
8.  **Display:** It presents the map, charts, and data tables within the Streamlit interface across various tabs.

## Dependencies

The application relies on several Python libraries:

* `streamlit`: For building the interactive web application.
* `pandas`: For data manipulation and analysis.
* `geopandas`: For working with geospatial data.
* `osmnx`: For fetching data from OpenStreetMap.
* `folium`: For creating interactive maps.
* `shapely`: For geometric operations.
* `scipy`: For spatial algorithms (KDTree).
* `matplotlib`: For generating charts.
* `numpy`: For numerical operations.
* `Pillow (PIL)`: For handling image files (chart display).
* `openpyxl`: For reading Excel files.
* `unicodedata`, `re`, `datetime`, `math`, `io`, `tempfile`, `os`, `traceback`: Standard Python libraries.

## Setup & Installation

1.  **Clone the Repository (or download the script):**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install Dependencies:** Create a `requirements.txt` file with the libraries listed above (or use `pip freeze > requirements.txt` if you install them manually) and run:
    ```bash
    pip install streamlit pandas geopandas osmnx folium shapely scipy matplotlib numpy Pillow openpyxl
    ```
    *Note: Installing `geopandas` and its dependencies (like `fiona`, `pyproj`, `gdal`) can sometimes be complex depending on your OS. Refer to the [GeoPandas installation guide](https://geopandas.org/en/stable/getting_started/install.html) for detailed instructions.*
4.  **(Optional) Add GeoJSON:** If you have a `barbados_parishes.geojson` file, place it in the same directory as `app.py`.

## Usage

1.  **Navigate to the script directory:**
    ```bash
    cd <your-repo-directory>
    ```
2.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
3.  **Interact:**
    * Open your web browser to the local address provided by Streamlit.
    * Use the sidebar to **upload** your property data file (Excel or CSV).
    * Click the **"🚀 Run Analysis"** button.
    * Wait for the analysis to complete (it might take a minute on the first run).
    * Explore the results in the different tabs: Map, Chart, Stats, etc.
    * Use the **"Locate on Map"** and **"Calculate Distance"** tools in the sidebar.

## Input Data Format

The application attempts to find columns with common names for:

* **Parish:** (e.g., `Parish`, `location`) - **Crucial for geocoding.**
* **Price:** (e.g., `Price`, `List Price`) - **Crucial for analysis.**
* **Description:** (e.g., `Description`, `Remarks`) - Used to extract bedroom counts.
* **Property Type:** (e.g., `Property Type`, `Type`) - Used for standardization.
* **Category:** (e.g., `Category`, `Listing Type`) - Used to distinguish Sale vs. Rent.
* **Name:** (e.g., `Name`, `Property Name`)
* **Size:** (e.g., `Size`, `Lot Size`, `Building Size`) - Can handle formats like "10,000 sq ft", "2.5 acres".

Ensure your input file contains, at a minimum, `Parish` and `Price` columns for the analysis to work effectively.

## Limitations

* **Geocoding:** Properties are currently geocoded to their *parish centroid*, not their exact street address. This provides a regional view but not pinpoint accuracy.
* **OSM Data:** The accuracy and completeness of data (parish boundaries, POIs) depend on OpenStreetMap, which is community-edited.
* **Distance Calculation:** The `Calculate Distance` tool uses the Haversine formula, which gives the shortest distance over the Earth's surface (great-circle) and does *not* account for road networks or terrain.
* **Caching:** Data fetching and loading are cached. While this speeds up subsequent runs, you might need to clear Streamlit's cache if underlying data sources change significantly.

## Potential Future Work

* Implement address-level geocoding (requires an external geocoding service/API).
* Add road network analysis (e.g., driving distance/time).
* Incorporate more data layers (e.g., flood zones, zoning).
* Enhance filtering and selection capabilities on the map.
* Develop more sophisticated statistical models.

## Credits

* **App Development:** Matthew Blackman (with AI assistance).
* **Data Sources:** Terra Caribbean, OpenStreetMap Contributors.
* **Core Libraries:** The amazing teams behind Streamlit, Pandas, GeoPandas, OSMnx, Folium, and other open-source Python projects.
