# Terra Caribbean: Terrain & Property View (Barbados)

**Version:** 1.0
**Date:** May 24, 2025
**Author:** Matthew Blackman (Assisted by AI)

## 📍 Overview

This project is an interactive Streamlit dashboard designed for visualizing and analyzing Terra Caribbean property listings within Barbados. It leverages geospatial data to provide insights into property distribution, pricing, and proximity to key features like beaches and tourist attractions, all presented against an interactive terrain map of the island.

The application allows users to upload their property data (in Excel or CSV format) and automatically processes it, geocodes it to the parish level, performs basic spatial analysis, and displays the results through various interactive visualizations and data tables.

## ✨ Key Features

* **Interactive Terrain Map:** Utilizes `Folium` to display an interactive map centered on Barbados, featuring:
    * Stamen Terrain base layer with optional Topographic and Light base maps.
    * Overlay of Barbados Parish boundaries fetched from OpenStreetMap (OSM) or a local GeoJSON file.
    * Markers for each parish center, displaying aggregated property summaries (counts, types, averages) on click.
    * Optional layers for Key Land Features (parks, golf courses) and Tourism Points of Interest.
* **Data Upload:** Supports `.xlsx` and `.csv` file uploads for property listings.
* **Automated Data Cleaning:** Standardizes parish names and property types, and parses property sizes (acres/sqft).
* **Parish-Level Geocoding:** Assigns properties to parishes and places them at the parish centroid for visualization.
* **Spatial Analysis:**
    * Calculates the approximate distance from each parish center to the nearest beach (using OSM data).
    * Counts the number of tourism points within a 2km radius of each parish center.
* **Data Visualization:**
    * **Chart:** Scatter plot showing the relationship between property prices and distance to the beach, color-coded by parish.
    * **Key Statistics:** A summary view of key metrics from the analyzed data.
    * **Parish Summary Table:** A detailed table showing aggregated statistics for each parish.
* **Data Export:** Allows users to download the analyzed and geocoded property data as a CSV file.
* **Informational Tab:** Explains the data sources, calculation methods (including map projections and area calculations), unit conversions, and known limitations.
* **Console Log:** Provides detailed logs of the analysis process for debugging and transparency.

## ⚙️ Technology Stack

* **Language:** Python 3.8+
* **Web Framework:** Streamlit
* **Data Handling:** Pandas
* **Geospatial:** GeoPandas, OSMnx, Folium, Shapely, SciPy (cKDTree)
* **Plotting:** Matplotlib
* **Utilities:** NumPy, OpenPyXL (for Excel reading)

## 📂 Project Structure
. ├── app.py # Main Streamlit application script ├── barbados_parishes.geojson # (Optional) Fallback GeoJSON ├── requirements.txt # List of Python dependencies └── README.md # This file


## 💾 Data Requirements

The application requires an input file (Excel or CSV) with property listings. For best results, the file should contain the following columns (case-insensitive, common variations handled):

* **`Parish` (Required):** The parish where the property is located (e.g., "Saint Michael", "St. James", "Christ Church").
* **`Price` (Required):** The listing price (numeric values, currency symbols/commas are handled).
* **`Property Type`:** The type of property (e.g., "Residential", "House", "Commercial", "Land"). This is used for standardization.
* **`Category`:** Listing type (e.g., "For Sale", "For Rent"). Used to differentiate sale/rent counts.
* **`Size`:** The size of the property or lot. Should include units (e.g., "1.5 acres", "5000 sq ft"). If no unit is present, large numbers are assumed to be sq ft.
* **`Description`:** Property description; used to attempt extracting the number of bedrooms.
* **`Name`:** A name or title for the listing.

## 🚀 Setup & Installation

1.  **Clone the Repository (or download files):**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```
    Or, simply save `app.py` (and optionally `barbados_parishes.geojson`) to a local folder.

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    * **Using `conda` (Highly Recommended for Geopandas):**
        ```bash
        conda create -n terra_env python=3.9
        conda activate terra_env
        conda install -c conda-forge geopandas osmnx folium streamlit pandas matplotlib openpyxl scipy
        ```
    * **Using `pip`:** Create a `requirements.txt` file:
        ```txt
        streamlit
        pandas
        geopandas
        osmnx
        folium
        matplotlib
        numpy
        openpyxl
        scipy
        ```
        Then run:
        ```bash
        pip install -r requirements.txt
        ```
        *If you encounter issues with `geopandas` via pip, consult the [official Geopandas installation guide](https://geopandas.org/en/stable/getting_started/install.html) for platform-specific instructions.*

4.  **(Optional) Add GeoJSON:** If you want a local fallback for parish boundaries, place `barbados_parishes.geojson` in the same directory as `app.py`.

## 📈 Usage

1.  **Navigate to your project directory** in your terminal.
2.  **Activate your virtual environment** (if you created one).
3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
4.  The application will open in your web browser.
5.  Use the **sidebar** to **upload** your property data file (`.xlsx` or `.csv`).
6.  Click the "**🚀 Run Analysis**" button.
7.  Wait for the analysis to complete.
8.  Explore the results using the **tabs**.

## ⚠️ Notes & Limitations

* **Geocoding:** Property locations are based on the **centroid (center point) of their respective parish**. This is an approximation.
* **Data Source:** Geospatial data is primarily sourced from **OpenStreetMap (OSM)**; accuracy depends on OSM contributors.
* **Calculations:** Area/distance calculations use the `EPSG:32620` (UTM Zone 20N) projection and are approximate.
* **Performance:** The initial run might be slower due to data downloading and caching.

## 🔮 Future Enhancements

* Implement address-level geocoding.
* Add more interactive charts and filtering.
* Incorporate historical property data analysis.
* Integrate official data layers (zoning, etc.).
* Improve input data validation.

## 🌍 Data Sources

* Property Data: Provided by the user (Terra Caribbean).
* Geospatial Data: [OpenStreetMap (OSM)](https://www.openstreetmap.org/) via `osmnx`.

## 🙏 Credits

* **Author:** Matthew Blackman
* **AI Assistance:** Google AI
* **Data Providers:** Terra Caribbean, OpenStreetMap Contributors.
