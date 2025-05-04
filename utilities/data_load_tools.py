# Import necessary libraries
import pandas as pd
import geopandas as gpd
from pathlib import Path
import os

def load_filtered_csv(path, filter_func=None, dtype=None, chunksize=100_000):
    """
    Load and filter a CSV file in chunks
    
    Parameters:
    path : Path to CSV file
    filter_func : Function to filter rows (optional)
    dtype : Dictionary of column data types
    chunksize : Number of rows to process at once
    """
    chunks = []
    for chunk in pd.read_csv(path, dtype=dtype, chunksize=chunksize, low_memory=False):
        if filter_func is not None:
            chunk = chunk[filter_func(chunk)]
        chunks.append(chunk)
        
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

def filter_south_yorkshire(df):
    return df['police_force'] == 14

def clean_and_organize_data(merged_data):
    """
    Clean up and organize columns in the merged dataset
    
    Parameters:
    merged_data : DataFrame containing the merged data
    
    Returns:
    DataFrame with cleaned and reorganized columns
    """
    print("  Cleaning and organizing columns...")
    
    # List of columns to drop as identified earlier
    columns_to_drop = [
        'accident_year_x', 'accident_year_y',
        'accident_reference_x', 'accident_reference_y'
    ]
    
    # Drop redundant columns
    cleaned_data = merged_data.drop(columns=columns_to_drop, errors='ignore')
    
    # Rename 'vehicle_reference_x' if it exists
    if 'vehicle_reference_x' in cleaned_data.columns:
        cleaned_data = cleaned_data.rename(columns={'vehicle_reference_x': 'vehicle_reference'})
    
    # Reorder columns to bring reference columns to the front
    reference_columns = [col for col in [
        'accident_index', 'accident_year', 'accident_reference', 
        'vehicle_reference', 'casualty_reference'
    ] if col in cleaned_data.columns]
    
    # Identify remaining columns that aren't in reference_columns
    remaining_columns = [col for col in cleaned_data.columns if col not in reference_columns]
    
    # Combine the lists to reorder DataFrame columns
    ordered_columns = reference_columns + remaining_columns
    
    return cleaned_data[ordered_columns]

def process_in_chunks(casualty_path, collision_path, vehicle_path, output_path, filter_func=None, chunksize=50_000, dtype_dict=None):
    """
    Process large datasets while preserving relationships between records
    
    Parameters:
    casualty_path : Path to casualty CSV
    collision_path : Path to collision CSV
    vehicle_path : Path to vehicle CSV
    output_path : Where to save the final filtered data
    filter_func : Function to filter the dataset (optional)
    chunksize : Number of rows to process at once
    dtype_dict : Dictionary of column data types
    """
    if dtype_dict is None:
        dtype_dict = {}

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Step 1: First filter collision data to get relevant accident indices
    print("Step 1: Filtering collision data to get relevant accident indices...")
    relevant_accident_indices = set()
    
    for chunk in pd.read_csv(collision_path, dtype=dtype_dict, chunksize=chunksize, low_memory=False):
        if filter_func is not None:
            filtered_chunk = chunk[filter_func(chunk)]
            if not filtered_chunk.empty:
                relevant_accident_indices.update(filtered_chunk['accident_index'])
    
    print(f"Found {len(relevant_accident_indices)} relevant accidents")
    
    if not relevant_accident_indices:
        print("No relevant data found after filtering. Process complete.")
        return
    
    # Step 2: Process each dataset in chunks, but filter by the complete set of accident indices
    print("\nStep 2: Processing casualty data...")
    processed_casualties = []
    
    for chunk in pd.read_csv(casualty_path, dtype=dtype_dict, chunksize=chunksize, low_memory=False):
        filtered_chunk = chunk[chunk['accident_index'].isin(relevant_accident_indices)]
        if not filtered_chunk.empty:
            processed_casualties.append(filtered_chunk)
    
    if not processed_casualties:
        print("No casualty data found for the filtered accidents. Process complete.")
        return
    
    casualty_data = pd.concat(processed_casualties, ignore_index=True)
    print(f"Processed {len(casualty_data)} casualty records")
    
    # Step 3: Process vehicle data
    print("\nStep 3: Processing vehicle data...")
    processed_vehicles = []
    
    for chunk in pd.read_csv(vehicle_path, dtype=dtype_dict, chunksize=chunksize, low_memory=False):
        filtered_chunk = chunk[chunk['accident_index'].isin(relevant_accident_indices)]
        if not filtered_chunk.empty:
            processed_vehicles.append(filtered_chunk)
    
    if not processed_vehicles:
        print("No vehicle data found for the filtered accidents. Process complete.")
        return
    
    vehicle_data = pd.concat(processed_vehicles, ignore_index=True)
    print(f"Processed {len(vehicle_data)} vehicle records")
    
    # Step 4: Load the filtered collision data
    print("\nStep 4: Loading filtered collision data...")
    processed_collisions = []
    
    for chunk in pd.read_csv(collision_path, dtype=dtype_dict, chunksize=chunksize, low_memory=False):
        filtered_chunk = chunk[chunk['accident_index'].isin(relevant_accident_indices)]
        if not filtered_chunk.empty:
            processed_collisions.append(filtered_chunk)
    
    collision_data = pd.concat(processed_collisions, ignore_index=True)
    print(f"Loaded {len(collision_data)} collision records")
    
    # Step 5: Merge datasets
    print("\nStep 5: Merging datasets...")
    print("  Merging casualty data with collision data...")
    merged_casualty_collision = casualty_data.merge(
        collision_data, on="accident_index", how="inner")
    
    print("  Merging with vehicle data...")
    final_data = merged_casualty_collision.merge(
        vehicle_data, on=["accident_index", "vehicle_reference"], how="inner")
    
    # Step 6: Clean and organize the merged data
    print("\nStep 6: Cleaning and organizing data...")
    final_data = clean_and_organize_data(final_data)
    
    # Step 7: Write the complete dataset
    print("\nStep 7: Writing processed data to file...")
    final_data.to_csv(output_path, index=False)
    
    print(f"\nProcessing complete. {len(final_data)} records saved to {output_path}")
    
    # Clean up memory
    del casualty_data, collision_data, vehicle_data, final_data, merged_casualty_collision



# -------------------------------------------------------
# ROUTES TOOLS
# -------------------------------------------------------

def build_school_name_map(folder_names, formatter=None):
    """
    Builds a dictionary mapping folder names to formatted school names.
    
    Parameters
    ----------
    folder_names : list of raw folder names (strings)
    formatter : optional function to format folder names; defaults to title case + space replacing underscore
    
    Returns
    -------
    dict : mapping from raw folder name (lowercase) to formatted name
    """
    if formatter is None:
        # Default: replace underscores with spaces, capitalize each word
        formatter = lambda name: name.replace("_", " ").title()
    
    return {name.lower(): formatter(name) for name in folder_names}

def read_routes_data(base_path, output_dir, file_name="all_routes_collisions", name_map=None, save_format="csv"):
    """
    Reads all .gpkg route files from nested school folders under each borough/level/merged_data path.

    Expected structure:
      base_path/
        ├── borough/
            ├── primary school analysis/
                └── merged data/
                    └── school folders
            ├── secondary school analysis/
                └── merged data/
                    └── school folders

    Parameters
    ----------
    base_path : Path or str
        Root directory containing borough folders
    output_dir : Path or str
        Where to save combined files
    name_map : dict, optional
        Mapping raw folder name → formatted school name

    Saves
    -----
    GeoPackage and CSV
    """
    base_path = Path(base_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if name_map is None:
        school_folders = []
        for borough in base_path.iterdir():
            if borough.is_dir():
                for level in ["primary school analysis", "secondary school analysis"]:
                    merged_data = borough / level / "merged data"
                    if merged_data.is_dir():
                        school_folders.extend([f for f in merged_data.iterdir() if f.is_dir()])
        folder_names = [f.name for f in school_folders]
        name_map = build_school_name_map(folder_names)

    all_data = []
    print(f"🔍 Searching route files under {base_path}...\n")

    for borough in base_path.iterdir():
        if borough.is_dir():
            for level in ["primary school analysis", "secondary school analysis"]:
                merged_data = borough / level / "merged data"
                if merged_data.is_dir():
                    for school_folder in merged_data.iterdir():
                        if school_folder.is_dir():
                            raw_name = school_folder.name.lower()
                            school_name = name_map.get(raw_name, raw_name)
                            print(f"📂 {borough.name}/{level}/merged data/{school_folder.name} → '{school_name}'")

                            for gpkg_file in school_folder.glob("*.gpkg"):
                                try:
                                    gdf = gpd.read_file(gpkg_file)[["accident_index", "geometry", "length_km"]].copy()
                                    file_stem = gpkg_file.stem
                                    parts = file_stem.split("_")
                                    route_number = parts[-1] if parts else "0"

                                    route_id = school_name.replace(" ", "_").lower() + f"_{route_number}"
                                    gdf["school_name"] = school_name
                                    gdf["borough"] = borough.name.title()
                                    gdf["school_level"] = "Primary" if "primary" in level else "Secondary"
                                    gdf["route_id"] = route_id

                                    all_data.append(gdf)
                                    print(f"✅ {gpkg_file.name} | route_id: {route_id} | rows: {len(gdf)}")
                                except Exception as e:
                                    print(f"⚠️ Failed to read {gpkg_file}: {e}")

    if all_data:
        final_gdf = gpd.GeoDataFrame(pd.concat(all_data, ignore_index=True))
        print(f"\n📊 Combined {len(all_data)} files → {len(final_gdf)} rows total.")

        filename = output_dir / file_name
        if save_format.lower() == "gpkg":
            final_gdf.to_file(f"{filename}.gpkg", driver="GPKG")
            print(f"💾 Saved GeoPackage: {filename}.gpkg")
        elif save_format.lower() == "csv":
            final_gdf.drop(columns="geometry").to_csv(f"{filename}.csv", index=False)
            print(f"💾 Saved CSV (geometry dropped): {filename}.csv")
        else:
            print(f"❌ Unknown save_format '{save_format}'. No file saved.")
    else:
        print("⚠️ No data found.")




def list_school_folders(base_path, output_txt=None):
    """
    Traverse nested folder structure and collect school folder names under 'merged data'.

    Parameters
    ----------
    base_path : str or Path
        Path to main folder
    output_txt : str or Path, optional
        If provided, saves the school names to a text file

    Returns
    -------
    List of school folder names (unformatted)
    """
    base_path = Path(base_path)
    school_folders = []

    for borough in base_path.iterdir():
        if borough.is_dir():
            for level in ["primary school analysis", "secondary school analysis"]:
                merged_data = borough / level / "merged data"
                if merged_data.is_dir():
                    for school_folder in merged_data.iterdir():
                        if school_folder.is_dir():
                            school_folders.append(school_folder.name)
    
    print(f"Found {len(school_folders)} school folders across all boroughs.")

    if output_txt:
        output_txt = Path(output_txt)
        output_txt.parent.mkdir(parents=True, exist_ok=True)
        with open(output_txt, 'w', encoding='utf-8') as f:
            for name in sorted(school_folders):
                f.write(name + '\n')
        print(f"Saved raw folder names to {output_txt}")

    return school_folders

