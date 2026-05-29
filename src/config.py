"""
Configuration for the patch-level entropy prediction project.

This config defines:
- Project paths
- Patch generation settings
- Stratified sampling parameters
- The full city list (67 cities with downloaded street graphs)
"""

from pathlib import Path

# ── project paths ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DIR      = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR  = PROJECT_ROOT / "results"
FIGURES_DIR  = RESULTS_DIR / "figures"

# ── patch generation settings ──────────────────────────────────────
PATCH_SIZE_M  = 800   # patch dimensions (square, 800m x 800m)
GRID_STEP_M   = 800   # step between patches (no overlap when = PATCH_SIZE)
MIN_SEGMENTS  = 15    # minimum street segments per valid patch
MIN_COVERAGE  = 0.50  # minimum street coverage to keep patch

# ── entropy computation ────────────────────────────────────────────
N_BEARING_BINS    = 36
N_CIRCUITY_PAIRS  = 150

# ── stratified sampling ────────────────────────────────────────────
TARGET_PATCH_COUNT = 2500   # final dataset size after stratified sampling
N_ENTROPY_BINS     = 20     # number of bins for stratified sampling

# ── city list (67 cities with downloaded street graphs) ────────────
CITIES = {
    # strong grids (15)
    'chicago':       'Chicago, USA',
    'manhattan':     'Manhattan, New York, USA',
    'vancouver':     'Vancouver, Canada',
    'portland':      'Portland, Oregon, USA',
    'denver':        'Denver, Colorado, USA',
    'miami':         'Miami, Florida, USA',
    'phoenix':       'Phoenix, Arizona, USA',
    'saltlakecity':  'Salt Lake City, USA',
    'buenosaires':   'Buenos Aires, Argentina',
    'laplata':       'La Plata, Argentina',
    'bogota':        'Bogotá, Colombia',
    'barcelona':     'Barcelona, Spain',
    'bologna':       'Bologna, Italy',
    'turin':         'Turin, Italy',
    'verona':        'Verona, Italy',

    # strong organic (15)
    'sarajevo':      'Sarajevo, Bosnia and Herzegovina',
    'marrakesh':     'Marrakesh, Morocco',
    'riga':          'Riga, Latvia',
    'ghent':         'Ghent, Belgium',
    'lisbon':        'Lisbon, Portugal',
    'bruges':        'Bruges, Belgium',
    'fez':           'Fez, Morocco',
    'casablanca':    'Casablanca, Morocco',
    'jerusalem':     'Jerusalem',
    'istanbul':      'Istanbul, Turkey',
    'cairo':         'Cairo, Egypt',
    'hanoi':         'Hanoi, Vietnam',
    'bangkok':       'Bangkok, Thailand',
    'bengaluru':     'Bengaluru, India',
    'chennai':       'Chennai, India',

    # extreme terrain (9)
    'caracas':       'Caracas, Venezuela',
    'quito':         'Quito, Ecuador',
    'medellin':      'Medellín, Colombia',
    'tbilisi':       'Tbilisi, Georgia',
    'genoa':         'Genoa, Italy',
    'bergen':        'Bergen, Norway',
    'tehran':        'Tehran, Iran',
    'lyon':          'Lyon, France',
    'macau':         'Macau',

    # planned modernist (9 — canberra missing)
    'brasilia':      'Brasília, Brazil',
    'chandigarh':    'Chandigarh, India',
    'telaviv':       'Tel Aviv, Israel',
    'warsaw':        'Warsaw, Poland',
    'rotterdam':     'Rotterdam, Netherlands',
    'skopje':        'Skopje, North Macedonia',
    'minsk':         'Minsk, Belarus',
    'kyiv':          'Kyiv, Ukraine',
    'lehavre':       'Le Havre, France',

    # mixed transitional global (8)
    'kyoto':         'Kyoto, Japan',
    'adelaide':      'Adelaide, Australia',
    'vienna':        'Vienna, Austria',
    'munich':        'Munich, Germany',
    'prague':        'Prague, Czech Republic',
    'amsterdam':     'Amsterdam, Netherlands',
    'stockholm':     'Stockholm, Sweden',
    'helsinki':      'Helsinki, Finland',

    # additional European (12 — from original 31-city European analysis)
    'birmingham':    'Birmingham, UK',
    'bratislava':    'Bratislava, Slovakia',
    'brussels':      'Brussels, Belgium',
    'bucharest':     'Bucharest, Romania',
    'budapest':      'Budapest, Hungary',
    'cologne':       'Cologne, Germany',
    'dublin':        'Dublin, Ireland',
    'leeds':         'Leeds, UK',
    'sofia':         'Sofia, Bulgaria',
    'tallinn':       'Tallinn, Estonia',
    'thessaloniki':  'Thessaloniki, Greece',
    'vilnius':       'Vilnius, Lithuania',

    
    # Additional ordered cities (added to balance entropy distribution)
    'mendoza':  'Departamento Capital, Mendoza, Argentina',
    'sapporo':  'Sapporo, Hokkaido, Japan',
    'tashkent': 'Tashkent, Uzbekistan',
}

# sanity check on import
if __name__ == "__main__":
    print(f"Project root:      {PROJECT_ROOT}")
    print(f"Total cities:      {len(CITIES)}")
    print(f"Patch size:        {PATCH_SIZE_M}m x {PATCH_SIZE_M}m")
    print(f"Grid step:         {GRID_STEP_M}m")
    print(f"Min segments:      {MIN_SEGMENTS}")
    print(f"Target patches:    {TARGET_PATCH_COUNT}")
    print(f"Entropy bins:      {N_ENTROPY_BINS}")