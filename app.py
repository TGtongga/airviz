import streamlit as st
from streamlit_bokeh import streamlit_bokeh
import polars as pl
import pandas as pd
import hvplot.pandas
import holoviews as hv
from holoviews import opts, dim
from holoviews.operation.downsample import downsample1d
from holoviews.operation.datashader import rasterize, dynspread, datashade
import panel as pn
from bokeh.models import HoverTool
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

TILE_PROVIDERS = {
    'OpenStreetMap': hv.element.tiles.OSM(),
    'CartoDB Positron (Light)': hv.element.tiles.CartoLight(),
    'CartoDB Dark Matter': hv.element.tiles.CartoDark(),
    'Stamen Terrain': hv.element.tiles.StamenTerrain(),
    'Stamen Toner (B&W)': hv.element.tiles.StamenToner(),
    'Stamen Toner Background': hv.element.tiles.StamenTonerBackground(),
    'Stamen Watercolor': hv.element.tiles.StamenWatercolor(),
    'ESRI Imagery (Satellite)': hv.element.tiles.EsriImagery(),
}

# Recommended colors for verified tiles
TILE_ROUTE_COLORS = {
    'OpenStreetMap': '#4a90e2',
    'CartoDB Positron (Light)': '#2563eb',
    'CartoDB Dark Matter': '#60a5fa',
    'Stamen Terrain': '#ef4444',
    'Stamen Toner (B&W)': '#f97316',
    'Stamen Toner Background': '#3b82f6',
    'Stamen Watercolor': '#8b5cf6',
    'ESRI Imagery (Satellite)': '#fbbf24',
}

TILE_AIRPORT_COLORS = {
    'OpenStreetMap': '#ff6b6b',
    'CartoDB Positron (Light)': '#ef4444',
    'CartoDB Dark Matter': '#fbbf24',
    'Stamen Terrain': '#10b981',
    'Stamen Toner (B&W)': '#22c55e',
    'Stamen Toner Background': '#ef4444',
    'Stamen Watercolor': '#ec4899',
    'ESRI Imagery (Satellite)': '#f43f5e',
}

# --- helper: lon/lat (deg) -> Web Mercator (meters) ---
def to_web_mercator(lon_deg, lat_deg):
    k = 6378137.0
    lon_rad = np.radians(lon_deg)
    lat_rad = np.radians(lat_deg)
    x = k * lon_rad
    y = k * np.log(np.tan(np.pi/4.0 + lat_rad/2.0))
    return x, y

# Configure HoloViews and Panel
hv.extension('bokeh')
pn.extension('bokeh')

# Set page config
st.set_page_config(
    page_title="Global Airline Routes Explorer",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for advanced but minimalist design
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    h1 {
        color: #1e3a8a;
        font-weight: 300;
        letter-spacing: -1px;
    }
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
    .stMetric label {
        color: white !important;
        font-weight: 300;
    }
    .stMetric div[data-testid="metric-container"] div:nth-child(2) {
        color: white !important;
        font-size: 2rem;
        font-weight: 300;
    }
    </style>
""", unsafe_allow_html=True)

# Cache data loading functions
@st.cache_data
def load_data():
    """Load all datasets using Polars for efficiency"""
    try:
        # Load airports data
        airports_df = pl.read_csv(
            'dataset/airports.dat',
            has_header=False,
            null_values=['\\N'],
            new_columns=[
                'airport_id', 'name', 'city', 'country', 'iata', 'icao',
                'latitude', 'longitude', 'altitude', 'timezone', 'dst',
                'tz_database', 'type', 'source'
            ]
        )
        
        # Load airlines data
        airlines_df = pl.read_csv(
            'dataset/airlines.dat',
            has_header=False,
            null_values=['\\N'],
            new_columns=[
                'airline_id', 'name', 'alias', 'iata', 'icao',
                'callsign', 'country', 'active'
            ]
        )
        
        # Load routes data
        routes_df = pl.read_csv(
            'dataset/routes.dat',
            has_header=False,
            null_values=['\\N'],
            new_columns=[
                'airline', 'airline_id', 'source_airport', 'source_airport_id',
                'dest_airport', 'dest_airport_id', 'codeshare', 'stops', 'equipment'
            ]
        )
        
        # Load countries data
        countries_df = pl.read_csv(
            'dataset/countries.dat',
            has_header=False,
            null_values=['\\N'],
            new_columns=['country_name', 'iso_code', 'dafif_code']
        )
        
        # Load planes data
        planes_df = pl.read_csv(
            'dataset/planes.dat',
            has_header=False,
            null_values=['\\N'],
            new_columns=['name', 'iata_code', 'icao_code']
        )
        
        return airports_df, airlines_df, routes_df, countries_df, planes_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None, None

@st.cache_data
def prepare_route_data(airports_df, airlines_df, routes_df):
    """Prepare route data with airport and airline information"""
    
    # Convert to pandas for merging
    airports_pd = airports_df.to_pandas()
    airlines_pd = airlines_df.to_pandas()
    routes_pd = routes_df.to_pandas()
    
    # Convert ID columns to numeric, handling errors
    routes_pd['source_airport_id'] = pd.to_numeric(routes_pd['source_airport_id'], errors='coerce')
    routes_pd['dest_airport_id'] = pd.to_numeric(routes_pd['dest_airport_id'], errors='coerce')
    routes_pd['airline_id'] = pd.to_numeric(routes_pd['airline_id'], errors='coerce')
    airports_pd['airport_id'] = pd.to_numeric(airports_pd['airport_id'], errors='coerce')
    airlines_pd['airline_id'] = pd.to_numeric(airlines_pd['airline_id'], errors='coerce')
    
    # Merge source airport info
    routes_with_source = routes_pd.merge(
        airports_pd[['airport_id', 'latitude', 'longitude', 'name', 'city', 'country', 'iata']],
        left_on='source_airport_id',
        right_on='airport_id',
        how='left',
        suffixes=('', '_source')
    )
    routes_with_source.rename(columns={
        'latitude': 'source_lat',
        'longitude': 'source_lon',
        'name': 'source_name',
        'city': 'source_city',
        'country': 'source_country',
        'iata': 'source_iata'
    }, inplace=True)
    
    # Merge destination airport info
    routes_with_dest = routes_with_source.merge(
        airports_pd[['airport_id', 'latitude', 'longitude', 'name', 'city', 'country', 'iata']],
        left_on='dest_airport_id',
        right_on='airport_id',
        how='left',
        suffixes=('_source', '_dest')
    )
    routes_with_dest.rename(columns={
        'latitude': 'dest_lat',
        'longitude': 'dest_lon',
        'name': 'dest_name',
        'city': 'dest_city',
        'country': 'dest_country',
        'iata': 'dest_iata'
    }, inplace=True)
    
    # Merge airline info
    routes_complete = routes_with_dest.merge(
        airlines_pd[['airline_id', 'name', 'country', 'active']],
        on='airline_id',
        how='left',
        suffixes=('', '_airline')
    )
    routes_complete.rename(columns={
        'name': 'airline_name',
        'country': 'airline_country'
    }, inplace=True)
    
    # Clean up and filter valid routes
    routes_complete = routes_complete.dropna(subset=['source_lat', 'source_lon', 'dest_lat', 'dest_lon'])
    
    return routes_complete

# MODIFICATION 1: NEW FUNCTION - Create visualization controls in expandable sidebar section
def create_visualization_controls():
    """Create visualization controls for routes and airports in an expandable sidebar section"""
    with st.sidebar.expander("🎨 Visualization Settings", expanded=False):
        st.markdown("#### Route Styling")
        route_color_override = st.color_picker(
            "Route Color",
            value="#4a90e2",
            key="route_color_picker"
        )
        route_alpha = st.slider(
            "Route Transparency",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            key="route_alpha_slider"
        )
        route_size = st.slider(
            "Route Width",
            min_value=0.1,
            max_value=2.0,
            value=0.1,
            step=0.1,
            key="route_size_slider"
        )
        route_hover = st.checkbox(
            "Enable Route Hover Highlighting",
            value=False,
            key="route_hover_check"
        )
        
        st.markdown("#### Airport Styling")
        airport_color_override = st.color_picker(
            "Airport Color",
            value="#ff6b6b",
            key="airport_color_picker"
        )
        airport_size = st.slider(
            "Airport Size",
            min_value=0.1,
            max_value=5.0,
            value=0.5,
            step=0.1,
            key="airport_size_slider"
        )
        airport_alpha = st.slider(
            "Airport Transparency",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            key="airport_alpha_slider"
        )
        airport_hover = st.checkbox(
            "Enable Airport Hover Highlighting",
            value=True,
            key="airport_hover_check"
        )
    
    return {
        'route_color': route_color_override,
        'route_alpha': route_alpha,
        'route_size': route_size,
        'route_hover': route_hover,
        'airport_color': airport_color_override,
        'airport_size': airport_size,
        'airport_alpha': airport_alpha,
        'airport_hover': airport_hover
    }

# MODIFICATION 2: UPDATED FUNCTION - create_world_map with visualization controls parameter
def create_world_map(airports_pd, routes_complete, selected_airline=None, selected_country=None, 
                     tile_style='CartoDB Dark Matter', viz_controls=None):
    """Create an interactive world map with airports and routes
    
    Parameters:
    -----------
    viz_controls : dict, optional
        Dictionary containing visualization control settings:
        - route_color, route_alpha, route_size, route_hover
        - airport_color, airport_size, airport_alpha, airport_hover
    """
    
    # Set default controls if not provided
    if viz_controls is None:
        viz_controls = {
            'route_color': None,
            'route_alpha': 0.2,
            'route_size': 0.1,
            'route_hover': False,
            'airport_color': None,
            'airport_size': 0.5,
            'airport_alpha': 0.2,
            'airport_hover': True
        }
    
    # Filter data based on selections
    if selected_airline and selected_airline != "All Airlines":
        routes_filtered = routes_complete[routes_complete['airline_name'] == selected_airline]
    elif selected_country and selected_country != "All Countries":
        routes_filtered = routes_complete[
            (routes_complete['source_country'] == selected_country) |
            (routes_complete['dest_country'] == selected_country)
        ]
    else:
        # Use stratified sampling for "All" view - better distribution
        routes_filtered = routes_complete.sample(min(500000, len(routes_complete)))
    
    # Get airports that have routes
    airport_ids = set(routes_filtered['source_airport_id'].dropna()) | set(routes_filtered['dest_airport_id'].dropna())
    airports_filtered = airports_pd[airports_pd['airport_id'].isin(airport_ids)]
    
    # Convert airport coordinates to Web Mercator
    ax, ay = to_web_mercator(airports_filtered['longitude'].values,
                             airports_filtered['latitude'].values)
    airports_web = airports_filtered.copy()
    airports_web['x'] = ax
    airports_web['y'] = ay

    # Use selected tile provider
    tiles = TILE_PROVIDERS.get(tile_style, hv.element.tiles.CartoDark())

    # Get colors based on tile style, with override from viz_controls
    route_color = viz_controls['route_color'] if viz_controls['route_color'] else TILE_ROUTE_COLORS.get(tile_style, '#4a90e2')
    airport_color = viz_controls['airport_color'] if viz_controls['airport_color'] else TILE_AIRPORT_COLORS.get(tile_style, '#ff6b6b')
    
    # Get visualization parameters from controls
    route_alpha = viz_controls['route_alpha']
    route_size = viz_controls['route_size']
    route_hover = viz_controls['route_hover']
    airport_alpha = viz_controls['airport_alpha']
    airport_size = viz_controls['airport_size']
    airport_hover = viz_controls['airport_hover']

    if len(routes_filtered) > 0:
        
        # Strategy: Use straight Segments for large datasets, curved Paths for small datasets
        use_curved_routes = len(routes_filtered) <= 100
        
        if use_curved_routes:
            # Optimization: Vectorize ALL operations - no Python loops
            routes_array = routes_filtered[['source_lat', 'source_lon', 'dest_lat', 'dest_lon']].values
            
            # Adaptive points based on distance (fewer points = faster)
            lons = routes_array[:, [1, 3]]
            lats = routes_array[:, [0, 2]]
            distances = np.sqrt(np.sum((lons[:, 1] - lons[:, 0])**2 + (lats[:, 1] - lats[:, 0])**2, axis=0))
            # Use 10-30 points instead of 50 (50 is overkill)
            num_points_per_route = np.clip((distances * 15).astype(int), 10, 30)
            
            # Use maximum points for interpolation, mask unused
            max_points = 30
            t_base = np.linspace(0, 1, max_points)
            
            # Vectorized great-circle computation for ALL routes at once
            lat1 = np.radians(routes_array[:, 0])[:, np.newaxis]
            lon1 = np.radians(routes_array[:, 1])[:, np.newaxis]
            lat2 = np.radians(routes_array[:, 2])[:, np.newaxis]
            lon2 = np.radians(routes_array[:, 3])[:, np.newaxis]
            
            t = t_base[np.newaxis, :]  # Shape: (1, max_points)
            
            # Vectorized interpolation for ALL routes simultaneously
            lat = np.arcsin(np.sin(lat1) * (1-t) + np.sin(lat2) * t)
            lon = lon1 + np.arctan2(
                np.sin(lon2 - lon1) * np.cos(lat2),
                np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
            ) * t
            
            lat_deg = np.degrees(lat)  # Shape: (n_routes, max_points)
            lon_deg = np.degrees(lon)
            
            # Vectorized Web Mercator conversion
            k = 6378137.0
            lon_rad = np.radians(lon_deg)
            lat_rad = np.radians(lat_deg)
            x_all = k * lon_rad
            y_all = k * np.log(np.tan(np.pi/4.0 + lat_rad/2.0))
            
            # Build DataFrame in one shot (not concatenating in loop)
            n_routes = len(routes_filtered)
            route_ids = np.repeat(np.arange(n_routes), max_points)
            
            # Flatten arrays
            x_flat = x_all.ravel()
            y_flat = y_all.ravel()
            
            # Repeat metadata efficiently using numpy
            airlines = routes_filtered['airline_name'].values
            source_cities = routes_filtered['source_city'].values
            source_iatas = routes_filtered['source_iata'].values
            dest_cities = routes_filtered['dest_city'].values
            dest_iatas = routes_filtered['dest_iata'].values
            
            airline_repeated = np.repeat(airlines, max_points)
            route_labels = np.array([
                f"{sc} ({si}) → {dc} ({di})" 
                for sc, si, dc, di in zip(source_cities, source_iatas, dest_cities, dest_iatas)
            ])
            route_repeated = np.repeat(route_labels, max_points)
            
            # Single DataFrame creation (not concat)
            combined_paths = pd.DataFrame({
                'x': x_flat,
                'y': y_flat,
                'route_id': route_ids,
                'airline': airline_repeated,
                'route': route_repeated
            })
            
            # MODIFICATION 3: Apply custom visualization controls for curved routes
            route_opts = {
                'color': route_color,
                'alpha': route_alpha,
                'line_width': route_size,
            }
            
            # Add hover highlighting if enabled
            if route_hover:
                route_opts.update({
                    'tools': ['hover'],
                    'hover_color': '#ff6b6b',
                    'hover_alpha': 0.9,
                    'hover_line_width': route_size * 2
                })
            
            routes_layer = hv.Path(
                combined_paths,
                kdims=['x', 'y'],
                vdims=['route_id', 'airline', 'route']
            ).opts(**route_opts)
        
        else:
            # Straight segments (for large datasets - much faster)
            segments_data = []
            
            for idx, route in routes_filtered.iterrows():
                sx, sy = to_web_mercator(route['source_lon'], route['source_lat'])
                dx, dy = to_web_mercator(route['dest_lon'], route['dest_lat'])
                
                segments_data.append({
                    'x0': sx, 'y0': sy,
                    'x1': dx, 'y1': dy,
                    'airline': route['airline_name'],
                    'route': f"{route['source_city']} ({route['source_iata']}) → {route['dest_city']} ({route['dest_iata']})"
                })
            
            segments_df = pd.DataFrame(segments_data)
            
            # MODIFICATION 4: Apply custom visualization controls for straight segments
            route_opts = {
                'color': route_color,
                'alpha': route_alpha,
                'line_width': route_size,
            }
            
            # Add hover highlighting if enabled
            if route_hover:
                route_opts.update({
                    'tools': ['hover'],
                    'hover_color': '#ff6b6b',
                    'hover_alpha': 0.9,
                    'hover_line_width': route_size * 2
                })
            
            routes_layer = hv.Segments(
                segments_df,
                kdims=['x0', 'y0', 'x1', 'y1'],
                vdims=['airline', 'route']
            ).opts(**route_opts)

    else:
        routes_layer = hv.Overlay([])

    # MODIFICATION 5: Apply custom visualization controls for airports
    airport_opts = {
        'size': airport_size,
        'color': airport_color,
        'marker': 'circle',
        'line_color': 'white',
        'line_width': 0.5,
        'fill_alpha': airport_alpha,
    }
    
    # Add hover highlighting if enabled
    if airport_hover:
        airport_opts.update({
            'tools': ['hover'],
            'hover_fill_color': '#10b981',
            'hover_alpha': 1.0,
            'hover_line_color': 'white',
            'hover_line_width': 1.5
        })
    else:
        airport_opts['tools'] = []
    
    airport_points = hv.Points(
        airports_web,
        kdims=['x', 'y'],
        vdims=['name', 'city', 'country', 'iata']
    ).opts(**airport_opts)

    # MODIFICATION 6: Reduced map height from 600 to 400
    map_plot = tiles * routes_layer * airport_points
    map_plot.opts(
        title="",
        xlabel="Longitude", ylabel="Latitude",
        width=1200, height=800,  # MODIFICATION 6: Reduced from 600 to 400 for better space usage
        tools=['pan', 'wheel_zoom', 'box_zoom', 'reset', 'hover'],
        active_tools=['wheel_zoom', 'pan'],
        toolbar='above',
        backend_opts={'plot.sizing_mode': 'stretch_width'}
    )
    return map_plot

def optimize_for_viewport(plot_obj, max_routes=10000):
    """Apply viewport-based downsampling for better performance"""
    try:
        # Apply viewport algorithm to only send visible data to frontend
        optimized = downsample1d(plot_obj, algorithm='viewport')
        return optimized
    except:
        return plot_obj

# MODIFICATION 7: UPDATED FUNCTION - create_statistics with filter support
def create_statistics(airports_df, airlines_df, routes_df, routes_complete=None, 
                     selected_airline=None, selected_country=None):
    """Create statistics for the sidebar - updated based on filters
    
    Parameters:
    -----------
    selected_airline : str, optional
        If provided, filter statistics to this airline
    selected_country : str, optional
        If provided, filter statistics to this country
    routes_complete : pd.DataFrame, optional
        The complete routes dataframe (converted to pandas). Used for filtering stats.
    """
    
    # If no filters applied, return global statistics
    if (selected_airline == "All Airlines" or selected_airline is None) and \
       (selected_country == "All Countries" or selected_country is None):
        stats = {
            'total_airports': len(airports_df),
            'total_airlines': len(airlines_df.filter(pl.col('active') == 'Y')),
            'total_routes': len(routes_df),
            'total_countries': airports_df['country'].n_unique()
        }
    else:
        # Filter statistics based on user selections
        if routes_complete is not None:
            if selected_airline and selected_airline != "All Airlines":
                filtered_routes = routes_complete[routes_complete['airline_name'] == selected_airline]
            elif selected_country and selected_country != "All Countries":
                filtered_routes = routes_complete[
                    (routes_complete['source_country'] == selected_country) |
                    (routes_complete['dest_country'] == selected_country)
                ]
            else:
                filtered_routes = routes_complete
            
            # Get unique airports from filtered routes
            airport_ids = set(filtered_routes['source_airport_id'].dropna()) | \
                         set(filtered_routes['dest_airport_id'].dropna())
            
            # Convert airport_id to numeric for filtering
            airports_df_pd = airports_df.to_pandas()
            airports_df_pd['airport_id'] = pd.to_numeric(airports_df_pd['airport_id'], errors='coerce')
            filtered_airports = airports_df_pd[airports_df_pd['airport_id'].isin(airport_ids)]
            
            # Get unique airlines from filtered routes
            filtered_airlines = set(filtered_routes['airline_name'].dropna().unique())
            
            stats = {
                'total_airports': len(filtered_airports),
                'total_airlines': len(filtered_airlines),
                'total_routes': len(filtered_routes),
                'total_countries': filtered_airports['country'].nunique()
            }
        else:
            stats = {
                'total_airports': len(airports_df),
                'total_airlines': len(airlines_df.filter(pl.col('active') == 'Y')),
                'total_routes': len(routes_df),
                'total_countries': airports_df['country'].n_unique()
            }
    
    return stats

# Main app
def main():
    # Header
    st.markdown("""
        <h1 style='text-align: center; margin-bottom: 0;'>
            ✈️ Global Airline Routes Explorer
        </h1>
        <p style='text-align: center; color: #6b7280; margin-top: 0; font-size: 1.1rem;'>
            Interactive visualization of worldwide airline routes and airports
        </p>
    """, unsafe_allow_html=True)
    
    # Load data
    airports_df, airlines_df, routes_df, countries_df, planes_df = load_data()
    
    if airports_df is None:
        st.error("Failed to load data. Please check if the dataset files exist.")
        return
    
    # Prepare route data
    routes_complete = prepare_route_data(airports_df, airlines_df, routes_df)
    airports_pd = airports_df.to_pandas()
    airlines_pd = airlines_df.to_pandas()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎨 Map Style")
        
        # Only verified tile options
        tile_options = list(TILE_PROVIDERS.keys())
        
        # Add emoji indicators for better UX
        tile_display_names = {
            'OpenStreetMap': '🗺️ OpenStreetMap (Classic)',
            'CartoDB Positron (Light)': '☀️ Light Mode',
            'CartoDB Dark Matter': '🌙 Dark Mode (Recommended)',
            'Stamen Terrain': '🏔️ Terrain',
            'Stamen Toner (B&W)': '⚫ Black & White',
            'Stamen Toner Background': '⚪ Minimal Light',
            'Stamen Watercolor': '🎨 Watercolor Art',
            'ESRI Imagery (Satellite)': '🛰️ Satellite View',
        }
        
        selected_tile_display = st.selectbox(
            "Choose Map Background",
            [tile_display_names[t] for t in tile_options],
            index=2  # Default to Dark Matter
        )
        
        # Reverse lookup to get actual tile name
        selected_tile = [k for k, v in tile_display_names.items() if v == selected_tile_display][0]
        
        # Show quick preview description
        tile_descriptions = {
            'OpenStreetMap': 'Standard street map with detailed labels',
            'CartoDB Positron (Light)': 'Clean, minimal light background',
            'CartoDB Dark Matter': 'Sleek dark theme, perfect for data viz',
            'Stamen Terrain': 'Topographic map showing elevation',
            'Stamen Toner (B&W)': 'High contrast black and white design',
            'Stamen Toner Background': 'Subtle, minimal background',
            'Stamen Watercolor': 'Artistic watercolor-style map',
            'ESRI Imagery (Satellite)': 'Real satellite imagery from space',
        }
        
        st.caption(tile_descriptions[selected_tile])

        viz_controls = create_visualization_controls()
        
        st.markdown("---")  # Divider
        
        st.markdown("### 🎯 Filter Options")
        
        # Filter by airline
        active_airlines = airlines_pd[airlines_pd['active'] == 'Y']['name'].dropna().unique()
        selected_airline = st.selectbox(
            "Select Airline",
            ["All Airlines"] + sorted(active_airlines.tolist()),
            index=0
        )
        
        # Filter by country
        countries = sorted(airports_pd['country'].dropna().unique().tolist())
        selected_country = st.selectbox(
            "Select Country",
            ["All Countries"] + countries,
            index=0
        )
        
        # MODIFICATION 8: Statistics - Updated to reflect user filters dynamically
        st.markdown("### 📊 Statistics")
        stats = create_statistics(
            airports_df, 
            airlines_df, 
            routes_df,
            routes_complete=routes_complete,
            selected_airline=selected_airline,
            selected_country=selected_country
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Airports", f"{stats['total_airports']:,}")
            st.metric("Routes", f"{stats['total_routes']:,}")
        with col2:
            st.metric("Airlines", f"{stats['total_airlines']:,}")
            st.metric("Countries", f"{stats['total_countries']:,}")
        
        # MODIFICATION 1: Add visualization controls in expandable area
        
        
        # Info section
        st.markdown("### ℹ️ About")
        st.info(
            "This application visualizes global airline routes using data from OpenFlights up to June 2014. "
            "Use the filters above to explore specific airlines or countries. "
            "Hover over airports and routes for detailed information."
        )
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🗺️ Map View", "📈 Analytics", "🔍 Data Explorer"])
    
    with tab1:
        st.markdown("### Interactive Route Map")
        
        # Create and display map
        with st.spinner("Loading map visualization..."):
            # MODIFICATION 9: Pass visualization controls to map function
            map_plot = create_world_map(
                airports_pd, 
                routes_complete, 
                selected_airline, 
                selected_country,
                tile_style=selected_tile,
                viz_controls=viz_controls
            )
            
            # Convert HoloViews plot to Bokeh and display
            # NOTE: Viewport optimization is already built into the Path/datashader rendering
            bokeh_plot = hv.render(map_plot, backend='bokeh')
            streamlit_bokeh(bokeh_plot, use_container_width=True, key="route_map")
        
        # Display info about current selection
        if selected_airline != "All Airlines":
            airline_routes = routes_complete[routes_complete['airline_name'] == selected_airline]
            st.markdown(f"**Showing {len(airline_routes)} routes for {selected_airline}**")
        elif selected_country != "All Countries":
            country_routes = routes_complete[
                (routes_complete['source_country'] == selected_country) |
                (routes_complete['dest_country'] == selected_country)
            ]
            st.markdown(f"**Showing {len(country_routes)} routes for {selected_country}**")
    
    with tab2:
        st.markdown("### Route Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top airlines by routes
            st.markdown("#### Top 10 Airlines by Number of Routes")
            top_airlines = routes_complete['airline_name'].value_counts().head(10)
            
            fig_airlines = top_airlines.to_frame('routes').reset_index().hvplot.barh(
                x='airline_name', 
                y='routes',
                color='#4a90e2',
                height=400,
                title=""
            ).opts(
                xlabel="Number of Routes",
                ylabel="Airline",
                invert_axes=False
            )
            streamlit_bokeh(hv.render(fig_airlines, backend='bokeh'), use_container_width=True)
        
        with col2:
            # Top airports by connections
            st.markdown("#### Top 10 Airports by Connections")
            source_counts = routes_complete.groupby('source_city').size()
            dest_counts = routes_complete.groupby('dest_city').size()
            total_counts = (source_counts + dest_counts).sort_values(ascending=False).head(10)
            
            fig_airports = total_counts.to_frame('connections').reset_index().hvplot.barh(
                x='index',
                y='connections',
                color='#ff6b6b',
                height=400,
                title=""
            ).opts(
                xlabel="Number of Connections",
                ylabel="City",
                invert_axes=False
            )
            streamlit_bokeh(hv.render(fig_airports, backend='bokeh'), use_container_width=True)
        
        # Aircraft usage
        st.markdown("#### Most Common Aircraft Types")
        equipment_list = []
        for equip in routes_complete['equipment'].dropna():
            equipment_list.extend(equip.split())
        equipment_counts = pd.Series(equipment_list).value_counts().head(15)
        
        fig_equipment = equipment_counts.to_frame('count').reset_index().hvplot.bar(
            x='index',
            y='count',
            color='#10b981',
            height=300,
            rot=45,
            title=""
        ).opts(
            xlabel="Aircraft Type",
            ylabel="Number of Routes"
        )
        streamlit_bokeh(hv.render(fig_equipment, backend='bokeh'), use_container_width=True)
    
    with tab3:
        st.markdown("### Data Explorer")
        
        data_type = st.selectbox(
            "Select Dataset",
            ["Airports", "Airlines", "Routes"]
        )
        
        if data_type == "Airports":
            st.dataframe(
                airports_pd[['name', 'city', 'country', 'iata', 'icao', 'latitude', 'longitude']].head(100),
                use_container_width=True,
                hide_index=True
            )
        elif data_type == "Airlines":
            st.dataframe(
                airlines_pd[airlines_pd['active'] == 'Y'][['name', 'iata', 'icao', 'country']].head(100),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.dataframe(
                routes_complete[['airline_name', 'source_city', 'dest_city', 'stops', 'equipment']].head(100),
                use_container_width=True,
                hide_index=True
            )

if __name__ == "__main__":
    main()