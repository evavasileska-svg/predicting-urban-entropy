from pathlib import Path
import sys

# ensure project root is on sys.path so `src` imports work
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import CITIES


def map_city_to_region_label(label):
    s = label.lower()
    if any(k in s for k in ['usa', 'united states', 'canada']):
        return 'North America'
    if any(k in s for k in ['argentina', 'colombia', 'venezuela', 'ecuador', 'brazil']):
        return 'Latin America'
    if any(k in s for k in ['france', 'spain', 'italy', 'germany', 'uk', 'united kingdom', 'belgium', 'poland', 'romania', 'slovakia', 'hungary', 'netherlands', 'czech', 'sweden', 'finland', 'ireland', 'norway', 'austria', 'switzerland', 'greece', 'estonia', 'lithuania', 'latvia', 'serbia', 'bosnia', 'macedonia', 'ukraine', 'belarus']):
        return 'Europe'
    if any(k in s for k in ['india', 'japan', 'uzbekistan', 'vietnam', 'thailand']):
        return 'Asia'
    if any(k in s for k in ['morocco', 'egypt', 'iran', 'turkey', 'israel']):
        return 'MENA'
    if any(k in s for k in ['australia']):
        return 'Oceania'
    return 'Other'


others = []
for code, label in CITIES.items():
    region = map_city_to_region_label(label)
    if region == 'Other':
        others.append((code, label))

print(f"Total cities in 'Other': {len(others)}\n")
for code, label in sorted(others):
    print(f"{code}: {label}")
