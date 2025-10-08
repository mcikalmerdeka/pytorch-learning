import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

def generate_indonesian_house_dataset(n_rows=50000, output_file='indonesian_house_prices.csv'):
    """
    Generate a synthetic dataset for Indonesian house price prediction.
    
    Parameters:
    - n_rows: Number of rows to generate (default: 50000)
    - output_file: Name of the output CSV file
    
    Returns:
    - DataFrame with the generated data
    """
    
    print(f"Generating {n_rows} rows of Indonesian house price data...")
    
    # Define Indonesian cities with different price ranges
    cities = ['Jakarta', 'Surabaya', 'Bandung', 'Medan', 'Semarang', 
              'Makassar', 'Palembang', 'Tangerang', 'Depok', 'Bekasi',
              'Bali (Denpasar)', 'Yogyakarta', 'Bogor', 'Malang', 'Batam']
    
    # City tier impacts price multiplier
    city_tier = {
        'Jakarta': 1.5, 'Bali (Denpasar)': 1.4, 'Tangerang': 1.3,
        'Surabaya': 1.2, 'Bandung': 1.1, 'Bekasi': 1.1,
        'Medan': 1.0, 'Semarang': 0.9, 'Depok': 1.0,
        'Makassar': 0.85, 'Palembang': 0.8, 'Yogyakarta': 0.95,
        'Bogor': 0.9, 'Malang': 0.85, 'Batam': 1.0
    }
    
    # Property types
    property_types = ['House', 'Villa', 'Townhouse', 'Apartment']
    property_type_multiplier = {
        'House': 1.0,
        'Villa': 1.6,
        'Townhouse': 0.85,
        'Apartment': 0.7
    }
    
    # Districts (simplified)
    districts = ['Central', 'North', 'South', 'East', 'West', 'Suburban']
    district_multiplier = {
        'Central': 1.4,
        'North': 1.1,
        'South': 1.15,
        'East': 1.0,
        'West': 1.05,
        'Suburban': 0.75
    }
    
    # Certificate types (land ownership)
    certificates = ['SHM (Freehold)', 'HGB (Building Rights)', 'Girik', 'AJB']
    certificate_multiplier = {
        'SHM (Freehold)': 1.2,
        'HGB (Building Rights)': 1.0,
        'Girik': 0.85,
        'AJB': 0.9
    }
    
    # Furnishing status
    furnishing = ['Unfurnished', 'Semi-Furnished', 'Fully Furnished']
    furnishing_addition = {
        'Unfurnished': 0,
        'Semi-Furnished': 50_000_000,
        'Fully Furnished': 150_000_000
    }
    
    # Generate base features
    data = {
        'city': np.random.choice(cities, n_rows),
        'district': np.random.choice(districts, n_rows),
        'property_type': np.random.choice(property_types, n_rows, 
                                         p=[0.45, 0.15, 0.25, 0.15]),
        'land_size_sqm': np.random.gamma(shape=3, scale=50, size=n_rows).astype(int) + 50,
        'building_size_sqm': None,  # Will be calculated
        'bedrooms': np.random.choice([1, 2, 3, 4, 5, 6], n_rows, 
                                    p=[0.05, 0.25, 0.35, 0.25, 0.08, 0.02]),
        'bathrooms': None,  # Will be calculated based on bedrooms
        'floors': np.random.choice([1, 2, 3], n_rows, p=[0.5, 0.45, 0.05]),
        'building_age_years': np.random.gamma(shape=2, scale=5, size=n_rows).astype(int),
        'carports': np.random.choice([0, 1, 2, 3], n_rows, p=[0.1, 0.5, 0.3, 0.1]),
        'certificate_type': np.random.choice(certificates, n_rows,
                                            p=[0.5, 0.35, 0.1, 0.05]),
        'furnishing': np.random.choice(furnishing, n_rows,
                                      p=[0.4, 0.35, 0.25]),
        'has_swimming_pool': np.random.choice([0, 1], n_rows, p=[0.85, 0.15]),
        'has_garden': np.random.choice([0, 1], n_rows, p=[0.4, 0.6]),
        'has_security': np.random.choice([0, 1], n_rows, p=[0.3, 0.7]),
        'distance_to_city_center_km': None,  # Will be calculated
        'nearby_facilities_count': np.random.randint(0, 15, n_rows),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate building size based on land size and property type
    df['building_size_sqm'] = (df['land_size_sqm'] * 
                                np.random.uniform(0.4, 0.8, n_rows)).astype(int)
    
    # Apartments have different logic (high-rise)
    apartment_mask = df['property_type'] == 'Apartment'
    df.loc[apartment_mask, 'building_size_sqm'] = np.random.randint(30, 150, apartment_mask.sum())
    df.loc[apartment_mask, 'land_size_sqm'] = df.loc[apartment_mask, 'building_size_sqm']
    
    # Calculate bathrooms (usually bedrooms - 1 or equal to bedrooms)
    df['bathrooms'] = df['bedrooms'] - np.random.choice([0, 1], n_rows, p=[0.4, 0.6])
    df['bathrooms'] = df['bathrooms'].clip(lower=1)
    
    # Calculate distance to city center based on district
    district_distance = {
        'Central': (1, 5),
        'North': (5, 15),
        'South': (5, 15),
        'East': (8, 20),
        'West': (8, 20),
        'Suburban': (15, 40)
    }
    
    df['distance_to_city_center_km'] = df['district'].apply(
        lambda x: np.random.uniform(*district_distance[x])
    ).round(1)
    
    # Calculate price in IDR (Indonesian Rupiah)
    # Base price calculation
    base_price_per_sqm = 15_000_000  # IDR per sqm (approx $1000 USD)
    
    df['price_idr'] = 0
    
    for idx, row in df.iterrows():
        # Start with building size
        price = row['building_size_sqm'] * base_price_per_sqm
        
        # Apply multipliers
        price *= city_tier[row['city']]
        price *= district_multiplier[row['district']]
        price *= property_type_multiplier[row['property_type']]
        price *= certificate_multiplier[row['certificate_type']]
        
        # Add land value (more significant for houses and villas)
        if row['property_type'] in ['House', 'Villa']:
            land_value = row['land_size_sqm'] * 5_000_000 * city_tier[row['city']]
            price += land_value
        
        # Adjust for number of bedrooms
        price *= (1 + (row['bedrooms'] - 2) * 0.1)
        
        # Depreciation for building age
        depreciation = 1 - (row['building_age_years'] * 0.02)
        depreciation = max(depreciation, 0.5)  # Minimum 50% of original value
        price *= depreciation
        
        # Add premium features
        if row['has_swimming_pool']:
            price += 200_000_000
        
        if row['has_garden']:
            price += 50_000_000
        
        if row['has_security']:
            price += 30_000_000
        
        # Multiple floors add value
        if row['floors'] > 1:
            price *= (1 + (row['floors'] - 1) * 0.15)
        
        # Carports
        price += row['carports'] * 20_000_000
        
        # Furnishing
        price += furnishing_addition[row['furnishing']]
        
        # Distance penalty
        distance_factor = max(0.7, 1 - (row['distance_to_city_center_km'] * 0.01))
        price *= distance_factor
        
        # Nearby facilities boost
        price += row['nearby_facilities_count'] * 10_000_000
        
        # Add some random noise (+/- 15%)
        noise = np.random.uniform(0.85, 1.15)
        price *= noise
        
        df.at[idx, 'price_idr'] = int(price)
    
    # Ensure minimum price
    df['price_idr'] = df['price_idr'].clip(lower=300_000_000)
    
    # Convert to USD for reference (approximate rate: 1 USD = 15,700 IDR)
    df['price_usd'] = (df['price_idr'] / 15700).round(0).astype(int)
    
    # Add listing date (last 2 years)
    start_date = datetime.now() - timedelta(days=730)
    df['listing_date'] = [start_date + timedelta(days=np.random.randint(0, 730)) 
                          for _ in range(n_rows)]
    df['listing_date'] = df['listing_date'].dt.strftime('%Y-%m-%d')
    
    # Reorder columns for better readability
    column_order = [
        'listing_date', 'city', 'district', 'property_type', 
        'price_idr', 'price_usd', 
        'land_size_sqm', 'building_size_sqm', 
        'bedrooms', 'bathrooms', 'floors',
        'building_age_years', 'carports',
        'certificate_type', 'furnishing',
        'has_swimming_pool', 'has_garden', 'has_security',
        'distance_to_city_center_km', 'nearby_facilities_count'
    ]
    
    df = df[column_order]
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"\nDataset saved to '{output_file}'")
    print(f"\nDataset Statistics:")
    print(f"Total rows: {len(df)}")
    print(f"\nPrice Statistics (IDR):")
    print(df['price_idr'].describe())
    print(f"\nPrice Statistics (USD):")
    print(df['price_usd'].describe())
    print(f"\nPrice range: IDR {df['price_idr'].min():,.0f} - IDR {df['price_idr'].max():,.0f}")
    print(f"Price range: USD ${df['price_usd'].min():,.0f} - USD ${df['price_usd'].max():,.0f}")
    
    return df

# Generate the dataset
if __name__ == "__main__":
    # You can adjust the number of rows here
    df = generate_indonesian_house_dataset(n_rows=50000, 
                                          output_file='indonesian_house_prices.csv')
    
    # Display first few rows
    print("\nFirst 10 rows of the dataset:")
    print(df.head(10))
    
    print("\nDataset info:")
    print(df.info())