from datetime import date, datetime, timedelta, timezone
import time
import pandas as pd
import yfinance as yf
import json

# ----------------------------
# 1) Initial DJIA membership
# ----------------------------
# Source: Wikipedia "Historical components of the Dow Jones Industrial Average" (https://en.wikipedia.org/wiki/Historical_components_of_the_Dow_Jones_Industrial_Average)
# Snapshot: May 6, 1991
# Inclusive of changes made on this date
# Names are exactly as in table in webpage
INITIAL_DATE = date(1991, 5, 6)
INITIAL_CONSTITUENTS_1991_05_06 = [
    "Allied-Signal Incorporated",
    "Aluminum Company of America",
    "American Express Company",
    "AT&T Corporation", # NOTE: formerly American Telephone and Telegraph Company
    "Bethlehem Steel Corporation",
    "The Boeing Company",
    "Caterpillar Inc.",
    "Chevron Corporation",
    "The Coca-Cola Company",
    "E.I. du Pont de Nemours & Company",
    "Eastman Kodak Company",
    "Exxon Corporation",
    "General Electric Company",
    "General Motors Corporation",
    "Goodyear Tire and Rubber Company",
    "International Business Machines Corporation",
    "International Paper Company",
    "J.P. Morgan & Company",
    "McDonald's Corporation",
    "Merck & Co., Inc.",
    "Minnesota Mining & Manufacturing Company",
    "The Procter & Gamble Company",
    "Philip Morris Companies Inc.",
    "Sears Roebuck & Company",
    "Texaco Incorporated",
    "Union Carbide Corporation",
    "United Technologies Corporation",
    "The Walt Disney Company",
    "Westinghouse Electric Corporation",
    "F. W. Woolworth Company",
]

# NOTE: things on the left are "removed", and assigning them to the right says what they were before, in case they aren't in "added" or INITIAL_CONSTITUENTS_1991_05_06. Note, not all of these are simple renames/abbreviations--in the case of Citigroup, AT&T and ExxonMobil, for example, things like divergences happened in between the right and the left. However, for simplicity, I will also treat these as renames--they are few, and the source doesn't treat them as separate entities anyway given there are no associated additions/removals to accompany the aforementioned transitions. When it comes to explicit removals/additions, e.g. DuPont -> DowDuPont, I will treat these as separate entities and query data separately for them, using their individual Yahoo Finance symbols.
# Texaco = Texaco Incorporated
# Bethlehem Steel = Bethlehem Steel Corporation
# Goodyear Tire = Goodyear Tire and Rubber Company
# Sears Roebuck = Sears Roebuck & Company
# Union Carbide = Union Carbide Corporation
# Kodak = Eastman Kodak Company
# International Paper = International Paper Company
# Altria Group = Philip Morris Companies Inc.
# Honeywell = Allied-Signal Incorporated
# Motors Liquidation Company = General Motors Corporation
# Citigroup = Travelers Inc.
# Alcoa = Aluminum Company of America
# AT&T = SBC Communications
# DuPont = E.I. du Pont de Nemours & Company
# General Electric = General Electric Company
# United Technologies = United Technologies Corporation
# ExxonMobil = Exxon Corporation
# NOTE: the below is a dictionary mapping the above right to the above left
name_to_canonical = {
    "Texaco Incorporated": "Texaco",
    "Bethlehem Steel Corporation": "Bethlehem Steel",
    "Goodyear Tire and Rubber Company": "Goodyear Tire",
    "Sears Roebuck & Company": "Sears Roebuck",
    "Union Carbide Corporation": "Union Carbide",
    "Eastman Kodak Company": "Kodak",
    "International Paper Company": "International Paper",
    "Philip Morris Companies Inc.": "Altria Group",
    "Allied-Signal Incorporated": "Honeywell",
    "General Motors Corporation": "Motors Liquidation Company", # TODO: this one may be tricky
    "Travelers Inc.": "Citigroup",  # TODO: this one may be tricky
    "Aluminum Company of America": "Alcoa",
    "SBC Communications": "AT&T",   # TODO: this one may be tricky
    "E.I. du Pont de Nemours & Company": "DuPont",
    "General Electric Company": "General Electric",
    "United Technologies Corporation": "United Technologies",
    "Exxon Corporation": "ExxonMobil",  # TODO: this one may be tricky
}

# ----------------------------
# 2) DJIA change events
# ----------------------------
# Source: Wikipedia "Dow Jones Industrial Average" (https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average)
# "Changes to the index since May 6th 1991 (exclusive) are as follows"
#
# Convention here:
# - additions become active on event_date
# - removals stop being active on event_date - 1 day    # TODO: significance?
CHANGE_EVENTS = [
    {
        "event_date": date(1997, 3, 17),
        "added": [
            "Travelers Inc.",
            "Hewlett-Packard",
            "Johnson & Johnson",
            "Walmart",
        ],
        "removed": [
            "Westinghouse Electric",
            "Texaco",
            "Bethlehem Steel",
            "F. W. Woolworth Company",
        ],
    },
    {
        "event_date": date(1999, 11, 1),
        "added": [
            "Microsoft",
            "Intel",
            "SBC Communications",
            "Home Depot",
        ],
        "removed": [
            "Goodyear Tire",
            "Sears Roebuck",
            "Union Carbide",
            "Chevron Corporation",
        ],
    },
    {
        "event_date": date(2004, 4, 8),
        "added": [
            "American International Group",
            "Pfizer",
            "Verizon Communications",
        ],
        "removed": [
            "AT&T Corporation",
            "Kodak",
            "International Paper",
        ],
    },
    {
        "event_date": date(2008, 2, 19),
        "added": [
            "Chevron Corporation",
            "Bank of America",
        ],
        "removed": [
            "Altria Group",
            "Honeywell",
        ],
    },
    {
        "event_date": date(2008, 9, 22),
        "added": [
            "Kraft Foods Inc.",
        ],
        "removed": [
            "American International Group",
        ],
    },
    {
        "event_date": date(2009, 6, 8),
        "added": [
            "The Travelers Companies",
            "Cisco Systems",
        ],
        "removed": [
            "Motors Liquidation Company",   # NOTE: formerly General Motors
            "Citigroup",
        ],
    },
    {
        "event_date": date(2012, 9, 24),
        "added": [
            "UnitedHealth Group",
        ],
        "removed": [
            "Kraft Foods Inc.", # NOTE: Kraft split into Mondelez International and Kraft Foods, initiating this change event
        ],
    },
    {
        "event_date": date(2013, 9, 23),
        "added": [
            "Goldman Sachs",
            "Nike, Inc.",
            "Visa Inc.",
        ],
        "removed": [
            "Alcoa",
            "Bank of America",
            "Hewlett-Packard",
        ],
    },
    {
        "event_date": date(2015, 3, 19),
        "added": [
            "Apple Inc.",
        ],
        "removed": [
            "AT&T",
        ],
    },
    {
        "event_date": date(2017, 9, 1),
        "added": [
            "DowDuPont",
        ],
        "removed": [
            "DuPont",   # NOTE: this merged with Dow Chemical Company to make DowDuPont, initiating change event
        ],
    },
    {
        "event_date": date(2018, 6, 26),
        "added": [
            "Walgreens Boots Alliance",
        ],
        "removed": [
            "General Electric",
        ],
    },
    {
        "event_date": date(2019, 4, 2),
        "added": [
            "Dow Inc.", # NOTE: a spin-off of DowDuPont
        ],
        "removed": [
            "DowDuPont",
        ],
    },
    {
        "event_date": date(2020, 4, 6),
        "added": [
            "Raytheon Technologies",
        ],
        "removed": [
            "United Technologies",  # NOTE: merged on 3rd April 2020 with Raytheon Company to make Raytheon Technologies, in a newly combined conglomerate not including previous subsidiaries Carrier Global or Otis Worldwide
        ],
    },
    {
        "event_date": date(2020, 8, 31),
        "added": [
            "Amgen",
            "Honeywell",
            "Salesforce.com",
        ],
        "removed": [
            "ExxonMobil",
            "Pfizer",
            "Raytheon Technologies",
        ],
    },
    {
        "event_date": date(2024, 2, 26),
        "added": [
            "Amazon",
        ],
        "removed": [
            "Walgreens Boots Alliance",
        ],
    },
    {
        "event_date": date(2024, 11, 8),
        "added": [
            "Nvidia",
            "Sherwin-Williams",
        ],
        "removed": [
            "Intel",
            "Dow Inc.",
        ],
    },
]

names = set(INITIAL_CONSTITUENTS_1991_05_06)
for event in CHANGE_EVENTS:
    names.update(event["added"])
canonical_names = {name_to_canonical.get(name, name) for name in names}

canonical_name_to_symbol = {
    'Motors Liquidation Company': None, # Delisted
    'Bank of America': 'BAC',   # https://uk.finance.yahoo.com/quote/BAC/
    'Union Carbide': None, # Delisted
    'Chevron Corporation': 'CVX',   # https://uk.finance.yahoo.com/quote/CVX/
    'Bethlehem Steel': None,   # Delisted
    'Pfizer': 'PFE',    # https://uk.finance.yahoo.com/quote/PFE/
    'General Electric': 'GE',  # No longer same entity as when in the DJIA, but correct historical data should be returned for GE (https://uk.finance.yahoo.com/quote/GE/)
    'Nike, Inc.': 'NKE',    # https://uk.finance.yahoo.com/quote/NKE/
    'Sears Roebuck': None, # Delisted
    'Texaco': None,    # Delisted
    'DowDuPont': None, # Delisted
    'United Technologies': None,   # Delisted
    'ExxonMobil': 'XOM',  # No longer same entity as when in the DJIA, but correct historical data should be returned for XOM (https://uk.finance.yahoo.com/quote/XOM/)
    'Home Depot': 'HD',    # https://uk.finance.yahoo.com/quote/HD/
    'The Procter & Gamble Company': 'PG',   # https://uk.finance.yahoo.com/quote/PG/
    'Alcoa': 'AA',  # No longer same entity as when in the DJIA, but correct historical data should be returned for AA (https://uk.finance.yahoo.com/quote/AA/)
    'American Express Company': 'AXP',
    'DuPont': 'DD',    # https://uk.finance.yahoo.com/quote/DD/
    "McDonald's Corporation": 'MCD',    # https://uk.finance.yahoo.com/quote/MCD/
    'AT&T Corporation': None,  # Delisted
    'Cisco Systems': 'CSCO',    # https://uk.finance.yahoo.com/quote/CSCO/
    'Goldman Sachs': 'GS', # https://uk.finance.yahoo.com/quote/GS/
    'Westinghouse Electric Corporation': None, # Delisted
    'Visa Inc.': 'V',   # https://uk.finance.yahoo.com/quote/V/
    'J.P. Morgan & Company': 'JPM', # https://uk.finance.yahoo.com/quote/JPM/
    'Kodak': None, # No longer same entity as when in the DJIA, and correct historical data shouldn't be returned for KODK (https://uk.finance.yahoo.com/quote/KODK/)
    'Minnesota Mining & Manufacturing Company': 'MMM',  # https://uk.finance.yahoo.com/quote/MMM/
    'Johnson & Johnson': 'JNJ', # https://uk.finance.yahoo.com/quote/JNJ/
    'Amazon': 'AMZN',   # https://uk.finance.yahoo.com/quote/AMZN/
    'The Boeing Company': 'BA', # https://uk.finance.yahoo.com/quote/BA/
    'F. W. Woolworth Company': None, # Delisted
    'The Travelers Companies': 'TRV',   # https://uk.finance.yahoo.com/quote/TRV/
    'AT&T': 'T',  # https://uk.finance.yahoo.com/quote/T/
    'Nvidia': 'NVDA',   # https://uk.finance.yahoo.com/quote/NVDA/
    'The Coca-Cola Company': 'KO',  # https://uk.finance.yahoo.com/quote/KO/
    'Microsoft': 'MSFT',    # https://uk.finance.yahoo.com/quote/MSFT/
    'Intel': 'INTC', # https://uk.finance.yahoo.com/quote/INTC/
    'Verizon Communications': 'VZ',    # https://uk.finance.yahoo.com/quote/VZ/
    'Citigroup': 'C', # https://uk.finance.yahoo.com/quote/C/
    'Apple Inc.': 'AAPL',   # https://uk.finance.yahoo.com/quote/AAPL/
    'Caterpillar Inc.': 'CAT',  # https://uk.finance.yahoo.com/quote/CAT/
    'Merck & Co., Inc.': 'MRK', # https://uk.finance.yahoo.com/quote/MRK/
    'International Paper': 'IP',   # https://uk.finance.yahoo.com/quote/IP/
    'Altria Group': 'MO',   # https://uk.finance.yahoo.com/quote/MO/
    'Honeywell': 'HON', # https://uk.finance.yahoo.com/quote/HON/
    'Sherwin-Williams': 'SHW',  # https://uk.finance.yahoo.com/quote/SHW/
    'Walmart': 'WMT',   # https://uk.finance.yahoo.com/quote/WMT/
    'American International Group': 'AIG',  # https://uk.finance.yahoo.com/quote/AIG/
    'Salesforce.com': 'CRM',    # https://uk.finance.yahoo.com/quote/CRM/
    'Hewlett-Packard': 'HPQ',   # No longer same entity as when in the DJIA, but correct historical data should be returned for HPQ (https://uk.finance.yahoo.com/quote/HPQ/)
    'Walgreens Boots Alliance': None,  # Delisted
    'Amgen': 'AMGN',    # https://uk.finance.yahoo.com/quote/AMGN/
    'Goodyear Tire': 'GT',  # https://uk.finance.yahoo.com/quote/GT/
    'International Business Machines Corporation': 'IBM',   # https://uk.finance.yahoo.com/quote/IBM/
    'UnitedHealth Group': 'UNH',    # https://uk.finance.yahoo.com/quote/UNH/
    'Raytheon Technologies': 'RTX', # https://uk.finance.yahoo.com/quote/RTX/
    'The Walt Disney Company': 'DIS',   # https://uk.finance.yahoo.com/quote/DIS/
    'Dow Inc.': 'DOW',  # https://uk.finance.yahoo.com/quote/DOW/
    'Kraft Foods Inc.': None   # Delisted
}

assert canonical_names == set(canonical_name_to_symbol.keys())

START_DATE = "1990-01-01"
END_DATE_INCLUSIVE = "2025-08-01"
END_DATE_EXCLUSIVE = (datetime(2025, 8, 1) + timedelta(days=1)).strftime("%Y-%m-%d")

tickers = [t for t in canonical_name_to_symbol.values() if t is not None]

series_map = {}
failed = []

for ticker in tickers:
    ok = False
    for attempt in range(3):
        try:
            df = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE_EXCLUSIVE,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if not df.empty and "Adj Close" in df.columns:
                series_map[ticker] = df["Adj Close"].iloc[:, 0].rename(ticker)
            else:
                failed.append(ticker)
            ok = True
            break
        except Exception as e:
            print(f"{ticker} failed on attempt {attempt + 1}: {e}")
            time.sleep(5)
    if not ok:
        failed.append(ticker)

ACQUIRED_AS_OF_DATETIME_UTC = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
ACQUIRED_AS_OF_DATE = ACQUIRED_AS_OF_DATETIME_UTC[:10]

adj_close = pd.concat(series_map.values(), axis=1) if series_map else pd.DataFrame()
adj_close.to_csv(f"djia_adj_close_yfinance_{ACQUIRED_AS_OF_DATETIME_UTC}.csv")

print("Failed tickers:", sorted(set(failed)))

metadata = {
    "data_source": "Yahoo Finance via yfinance",
    "price_field": "Adj Close",
    "start_date": START_DATE,
    "end_date_inclusive": END_DATE_INCLUSIVE,
    "end_date_exclusive_used_for_query": END_DATE_EXCLUSIVE,
    "acquired_as_of_date": ACQUIRED_AS_OF_DATE,
    "acquired_as_of_datetime_utc": ACQUIRED_AS_OF_DATETIME_UTC,
    "yfinance_version": yf.__version__,
    "canonical_name_to_symbol": canonical_name_to_symbol,
}

with open(f"djia_adj_close_yfinance_{ACQUIRED_AS_OF_DATETIME_UTC}.json", "w") as f:
    json.dump(metadata, f, indent=4)