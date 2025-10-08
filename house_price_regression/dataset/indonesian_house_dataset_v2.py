
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

def generate_indonesian_house_dataset_v2(n_rows=50000, output_file='indonesian_house_prices_v2.csv'):
    """
    Generate a synthetic but learnable dataset for Indonesian house price prediction.
    Relationships are structured and contain lower random noise to allow ML models to learn effectively.
    """
    
    print(f"Generating {n_rows} rows of Indonesian house price data (v2)...")
    
    # Define cities with realistic tier multipliers
    cities = ['Jakarta', 'Surabaya', 'Bandung', 'Medan', 'Semarang', 
              'Makassar', 'Palembang', 'Tangerang', 'Depok', 'Bekasi',
              'Bali (Denpasar)', 'Yogyakarta', 'Bogor', 'Malang', 'Batam']
    
    city_tier = {
        'Jakarta': 1.6, 'Bali (Denpasar)': 1.5, 'Tangerang': 1.3,
        'Surabaya': 1.2, 'Bandung': 1.1, 'Bekasi': 1.05,
        'Medan': 1.0, 'Semarang': 0.95, 'Depok': 1.0,
        'Makassar': 0.9, 'Palembang': 0.85, 'Yogyakarta': 0.9,
        'Bogor': 0.9, 'Malang': 0.85, 'Batam': 1.0
    }
    
    property_types = ['House', 'Villa', 'Townhouse', 'Apartment']
    property_type_multiplier = {
        'House': 1.0,
        'Villa': 1.4,
        'Townhouse': 0.9,
        'Apartment': 0.8
    }
    
    districts = ['Central', 'North', 'South', 'East', 'West', 'Suburban']
    district_multiplier = {
        'Central': 1.4,
        'North': 1.1,
        'South': 1.15,
        'East': 1.0,
        'West': 1.0,
        'Suburban': 0.75
    }
    
    certificates = ['SHM (Freehold)', 'HGB (Building Rights)', 'Girik', 'AJB']
    certificate_multiplier = {
        'SHM (Freehold)': 1.2,
        'HGB (Building Rights)': 1.0,
        'Girik': 0.85,
        'AJB': 0.9
    }
    
    furnishing = ['Unfurnished', 'Semi-Furnished', 'Fully Furnished']
    furnishing_addition = {
        'Unfurnished': 0,
        'Semi-Furnished': 30_000_000,
        'Fully Furnished': 100_000_000
    }
    
    # Generate base features
    df = pd.DataFrame({
        'city': np.random.choice(cities, n_rows),
        'district': np.random.choice(districts, n_rows),
        'property_type': np.random.choice(property_types, n_rows, p=[0.5, 0.15, 0.25, 0.1]),
        'land_size_sqm': np.random.gamma(shape=3, scale=40, size=n_rows).astype(int) + 40,
        'building_age_years': np.random.gamma(shape=2, scale=5, size=n_rows).astype(int),
        'floors': np.random.choice([1, 2, 3], n_rows, p=[0.6, 0.35, 0.05]),
        'carports': np.random.choice([0, 1, 2, 3], n_rows, p=[0.1, 0.5, 0.3, 0.1]),
        'certificate_type': np.random.choice(certificates, n_rows, p=[0.5, 0.35, 0.1, 0.05]),
        'furnishing': np.random.choice(furnishing, n_rows, p=[0.4, 0.35, 0.25]),
        'has_swimming_pool': np.random.choice([0, 1], n_rows, p=[0.85, 0.15]),
        'has_garden': np.random.choice([0, 1], n_rows, p=[0.4, 0.6]),
        'has_security': np.random.choice([0, 1], n_rows, p=[0.3, 0.7]),
        'nearby_facilities_count': np.random.randint(0, 15, n_rows),
    })
    
    # Correlated features
    df['building_size_sqm'] = (df['land_size_sqm'] * np.random.uniform(0.6, 0.85, n_rows)).astype(int)
    df['bedrooms'] = np.round(df['building_size_sqm'] / 40).clip(1, 6).astype(int)
    df['bathrooms'] = np.clip(df['bedrooms'] - 1, 1, 5).astype(int)
    
    district_distance = {
        'Central': (1, 5),
        'North': (5, 15),
        'South': (5, 15),
        'East': (8, 20),
        'West': (8, 20),
        'Suburban': (15, 40)
    }
    
    df['distance_to_city_center_km'] = df['district'].apply(lambda x: np.random.uniform(*district_distance[x])).round(1)
    
    # Price calculation with more structured relationships
    base_price_per_sqm = 10_000_000
    df['price_idr'] = (
        df['building_size_sqm'] * base_price_per_sqm
        * df['city'].map(city_tier)
        * df['district'].map(district_multiplier)
        * df['property_type'].map(property_type_multiplier)
        * df['certificate_type'].map(certificate_multiplier)
    )
    
    # Add land value and modifiers
    df['price_idr'] += df['land_size_sqm'] * 4_000_000 * df['city'].map(city_tier)
    df['price_idr'] *= (1 + (df['bedrooms'] - 2) * 0.08)
    df['price_idr'] *= (1 - df['building_age_years'] * 0.015).clip(lower=0.6)
    df['price_idr'] *= (1 + (df['floors'] - 1) * 0.12)
    df['price_idr'] += df['carports'] * 15_000_000
    df['price_idr'] += df['furnishing'].map(furnishing_addition)
    df['price_idr'] += df['has_swimming_pool'] * 150_000_000
    df['price_idr'] += df['has_garden'] * 50_000_000
    df['price_idr'] += df['has_security'] * 25_000_000
    df['price_idr'] *= np.maximum(0.7, 1 - (df['distance_to_city_center_km'] * 0.008))
    df['price_idr'] += df['nearby_facilities_count'] * 8_000_000
    
    # Add smaller random noise (3%)
    df['price_idr'] *= np.random.uniform(0.97, 1.03, n_rows)
    df['price_idr'] = df['price_idr'].clip(lower=300_000_000).astype(int)
    
    df['price_usd'] = (df['price_idr'] / 15700).round(0).astype(int)
    
    # Add listing date
    start_date = datetime.now() - timedelta(days=730)
    df['listing_date'] = [start_date + timedelta(days=np.random.randint(0, 730)) for _ in range(n_rows)]
    df['listing_date'] = df['listing_date'].dt.strftime('%Y-%m-%d')
    
    # Reorder columns
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
    print(f"Dataset saved to '{output_file}' with {n_rows} rows.")
    print(df.head())
    return df

if __name__ == "__main__":
    df = generate_indonesian_house_dataset_v2()
