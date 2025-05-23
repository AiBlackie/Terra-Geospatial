# Terra Caribbean Property Intelligence: Geospatial View

## 🏝️ Overview

This project is an interactive Streamlit dashboard designed to provide geospatial insights into Terra Caribbean property listings for Barbados. It allows users to upload their property data (in Excel or CSV format), which is then processed, geocoded, and analyzed against various geospatial features fetched from OpenStreetMap (OSM). The results are presented through an interactive map, a price-distance chart, key statistics, and an exportable dataset.

## ✨ Features

* **Data Upload:** Easily upload property data via Excel (.xlsx) or CSV (.csv) files.
* **Data Cleaning:** Automatically cleans and standardizes key data points like Parish names, property sizes (converting to sqft), and property types.
* **Geospatial Integration:** Fetches and caches geospatial data for Barbados from OpenStreetMap, including:
    * Parish boundaries.
    * Beaches.
    * Tourism-related points of interest.
    * Key land features (parks, golf courses, schools, etc.).
* **Fallback Data:** Includes an option to load parish boundaries from a local `barbados_parishes.geojson` file if OSM data fails.
* **Geocoding:** Approximates property locations by matching them to their respective parish centroids.
* **Spatial Analysis:** Calculates:
    * Distance from each property to the nearest beach.
    * Count of tourism points within a 2km radius of each property.
* **Interactive Map (Folium):** Visualizes:
    * Parish boundaries with tooltips.
    * Key land features (parks, golf courses).
    * Aggregated property summaries (count, types, avg. metrics) per parish via clickable markers.
    * Toggleable layer for points of interest.
* **Data Visualization (Matplotlib):** Generates a scatter plot showing the relationship between property prices and their distance to the nearest beach.
* **Key Statistics:** Displays summary statistics across all analyzed properties.
* **Data Export:** Allows users to download the enriched and analyzed property data as a CSV file.
* **Console Logging:** Provides a log tab to monitor the analysis process and diagnose potential issues.
* **Caching:** Uses Streamlit's caching mechanisms to speed up data loading and OSM fetches on subsequent runs.

## ⚙️ How it Works

1.  **Upload:** The user uploads a property data file via the Streamlit sidebar.
2.  **Load & Clean:** The script reads the data using `pandas`, identifies key columns (Parish, Price, Type, Size, etc.), cleans parish names, converts sizes to square feet, standardizes property types, and extracts bedroom counts.
3.  **Fetch GeoData:** `osmnx` is used to download boundaries, beaches, tourism points, and other features for Barbados from OpenStreetMap. If OSM parish data fails, it attempts to load from `barbados_parishes.geojson`.
4.  **Geocode:** Properties are matched to their parish's cleaned name and assigned the parish centroid as their approximate location using `geopandas`.
5.  **Analyze:** `geopandas` and `scipy.spatial.cKDTree` are used to calculate distances to beaches and proximity to tourism points.
6.  **Visualize:**
    * `folium` creates an interactive map, layering parishes, features, and property summary markers.
    * `matplotlib` generates a scatter plot, saved as a temporary image for display.
7.  **Display:** `Streamlit` presents the UI, including the sidebar, tabs for the map, chart, statistics, export, and console log.

## 🚀 Installation & Setup

1.  **Prerequisites:**
    * Python 3.8 or higher.
    * It's recommended to use a virtual environment.

2.  **Clone the Repository (or save the script):**
    ```bash
    # If using git
    git clone <repository_url>
    cd <repository_directory>
    # Or save the provided script as streamlit_app.py
    ```

3.  **Install Dependencies:**
    ```bash
    pip install streamlit pandas geopandas osmnx folium shapely scipy matplotlib numpy Pillow openpyxl
    ```
    *Note: Installing `geopandas` can sometimes be complex due to its underlying C libraries (GEOS, GDAL, PROJ). Using Conda or pre-compiled wheels might be easier depending on your OS.*

4.  **(Optional) Add Local Parish Data:**
    * If you have a `barbados_parishes.geojson` file, place it in the same directory as `streamlit_app.py`. This serves as a backup if OpenStreetMap parish data cannot be fetched.

5.  **Run the App:**
    ```bash
    streamlit run streamlit_app.py
    ```
    This will start the Streamlit server and open the application in your default web browser.

## 📋 Input Data Format

The application accepts Excel (`.xlsx`) or CSV (`.csv`) files. For best results, your file should include the following columns (the script attempts to map common variations):

* **Parish** (Required): The parish where the property is located (e.g., 'Saint Michael', 'Christ Church').
* **Price** (Required): The listing price (numeric values or strings with currency symbols/commas).
* **Property Type:** (e.g., 'House', 'Land', 'Office', 'Apartment'). This is standardized into 'Residential', 'Commercial', 'Land', or 'Other'.
* **Size:** The size of the property (e.g., '5,000 sq ft', '2 acres'). This is converted to square feet.
* **Description:** A text description, used to extract the number of bedrooms.
* **Category:** Typically 'For Sale' or 'For Rent'.
* **Name:** The name or title of the listing.

*The script is robust but works best with clearly named columns and consistent data.*

## 📊 Output

The application provides the following outputs through different tabs:

* **Map:** An interactive map of Barbados showing property insights.
* **Chart:** A scatter plot visualizing price vs. beach distance.
* **Key Statistics:** A list of important summary metrics.
* **Export Data:** A download button for the analyzed data in CSV format, including calculated metrics and coordinates.
* **Console Log:** A text area showing detailed logs from the analysis run.

## 📦 Dependencies

* `streamlit`
* `pandas`
* `geopandas`
* `osmnx`
* `folium`
* `shapely`
* `scipy`
* `matplotlib`
* `numpy`
* `Pillow`
* `openpyxl`
* `unicodedata`
* `re`

## 🌍 Data Sources

* Property Data: Provided by the user (Terra Caribbean).
* Geospatial Data: [OpenStreetMap (OSM)](https://www.openstreetmap.org/) via `osmnx`.

## 🙏 Credits

* **Author:** Matthew Blackman
* **AI Assistance:** Google AI (for code assistance and documentation).
* **Data Providers:** Terra Caribbean, OpenStreetMap Contributors.
