from pathlib import Path
import pandas as pd
from src.config import PROCESSED_DIR, CITIES

CSV = PROCESSED_DIR / "patch_training_data_full.csv"

if not CSV.exists():
    print(f"ERROR: {CSV} not found.")
    raise SystemExit(1)


def map_city_to_region(city_code):
    label = CITIES.get(city_code, '')
    s = label.lower()
    if any(k in s for k in ['usa', 'united states', 'canada']):
        return 'North America'
    if any(k in s for k in ['argentina', 'colombia', 'venezuela', 'ecuador', 'brazil']):
        return 'Latin America'
    if any(k in s for k in ['france', 'spain', 'italy', 'germany', 'uk', 'united kingdom', 'belgium', 'poland', 'romania', 'slovakia', 'hungary', 'netherlands', 'czech', 'sweden', 'finland', 'ireland', 'norway', 'austria', 'switzerland', 'greece', 'estonia', 'lithuania', 'latvia', 'bosnia', 'macedonia', 'ukraine', 'belarus']):
        return 'Europe'
    if any(k in s for k in ['india', 'japan', 'uzbekistan', 'vietnam', 'thailand']):
        return 'Asia'
    if any(k in s for k in ['morocco', 'egypt', 'iran', 'turkey', 'israel']):
        return 'MENA'
    if any(k in s for k in ['australia']):
        return 'Oceania'
    return 'Other'


df = pd.read_csv(CSV, usecols=lambda c: c in ['patch_id', 'city_code'])
df['region'] = df['city_code'].apply(map_city_to_region)
others = df[df['region'] == 'Other']

print(f"Total patches: {len(df):,}")
print(f"Patches labeled 'Other': {len(others):,}\n")

print("City counts for 'Other':")
print(others['city_code'].value_counts().to_string())

print('\nSample rows (up to 50):')
print(others.head(50).to_string(index=False))
