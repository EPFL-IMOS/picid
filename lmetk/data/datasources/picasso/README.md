# Energy Market Data Collection

This repository contains a comprehensive collection of energy market datasets covering various aspects of the German electricity grid and European energy markets from 2022 to 2024.

## Dataset Overview

| Type                                                       | Period | Resolution | Source |
|------------------------------------------------------------|--------|------------|--------|
| aFRR Energy Prices PICASSO*                                | 2022.05 – 2024.12 | 4s | [transnetbw.de/picasso](https://www.transnetbw.de/en/energy-market/ancillary-services/picasso) |
| aFRR Energy Exchange PICASSO                               | 2022.05 – 2024.12 | 4s | [transnetbw.de/picasso](https://www.transnetbw.de/en/energy-market/ancillary-services/picasso) |
| Intraday Price Index (IP Index)*                           | 2022.01 – 2024.12 | 15 min | [netztransparenz](https://www.netztransparenz.de/en/Balancing-Capacity/Imbalance-price/IP-Index) |
| aFRR Capacity Prices*                                      | 2022.01 – 2024.12 | 4h | [regelleistung](https://www.regelleistung.net/apps/datacenter/tendering-files/) |
| Forecast Load - Transnet                                   | 2022.01 – 2024.12 | 15 min | [smard.de](https://www.smard.de/en/downloadcenter/download-market-data/) |
| Actual Load - Transnet                                     | 2022.01 – 2024.12 | 15 min | [smard.de](https://www.smard.de/en/downloadcenter/download-market-data/) |
| Day-ahead Prices - Transnet*                               | 2022.01 – 2024.12 | Hour | [smard.de](https://www.smard.de/en/downloadcenter/download-market-data/) |
| Forecast Generations Day-ahead - Transnet*                 | 2022.01 – 2024.12 | Hour/15min | [smard.de](https://www.smard.de/en/downloadcenter/download-market-data/) |
| Forecast Generations Intraday (PV&Wind) - Transnet         | 2022.01 – 2024.12 | 15min | [smard.de](https://www.smard.de/en/downloadcenter/download-market-data/) |
| Actual Generations - Transnet, 50 Hertz, Amprion*          | 2022.01 – 2024.12 | 15 min | [smard.de](https://www.smard.de/en/downloadcenter/download-market-data/) |
| Heating & Cooling Demands                                  | 2022.01 – 2024.12 | Hour | [renewables.ninja](https://www.renewables.ninja/) |
| Weather                                                    | 2022.01 – 2024.12 | Hour | [renewables.ninja](https://www.renewables.ninja/) |
| List of bids                                               | 2022.01 – 2024.12 | Hour | [regelleistung](https://www.regelleistung.net/apps/datacenter/tendering-files/) |
| Merit Order List of balancing energy                       | 2022.01 – 2024.12 | Hour | [tennet.eu](https://www.tennet.eu/nl-en/node/3479) |
| Price of Activated Balancing Energy                        | 2022.01 – 2024.12 | Hour | [entsoe.eu](https://iop-transparency.entsoe.eu/balancing/r3/pricesOfActivatedBalancingEnergy/show) |
| mFRR activated volume                                      | 2022.01 – 2024.12 | 15 min | [netztransparenz](https://www.netztransparenz.de/en/Balancing-Capacity/Imbalance-price/IP-Index) |
| GCC balance traffic light                                  | 2022.01 – 2024.12 | 1 min | [netztransparenz](https://www.netztransparenz.de/en/Balancing-Capacity/Imbalance-price/IP-Index) |
| AEPModule                                                  | 2022.01 – 2024.12 | 15 min | [netztransparenz](https://www.netztransparenz.de/en/Balancing-Capacity/Imbalance-price/Imbalance-price-modules-IP) |
| VoAA                                                       | 2022.01 – 2024.12 | 15 min | [netztransparenz](https://www.netztransparenz.de/en/Balancing-Capacity/Imbalance-price/IP-Index) |
| reBAP                                                      | 2022.01 – 2024.12 | 15 min | [netztransparenz](https://www.netztransparenz.de/en/Balancing-Capacity/Imbalance-price/Uniform-imbalance-price-reBAP) |
| AepSchaetzer                                               | 2022.01 – 2024.12 | 15 min | [netztransparenz](https://www.netztransparenz.de/en/Balancing-Capacity/Imbalance-price/IP-estimator) |
| Aggregated Balancing Energy Bids*                          | 2022.01 – 2024.12 | 15 min | [entsoe.eu](https://newtransparency.entsoe.eu/balancing/energy/aggregatedBids/typeOfProductAggregation) |
| Generation Forecast for Wind & Solar - Intraday and Actual | 2022.01 – 2024.12 | 15 min | [entsoe.eu](https://newtransparency.entsoe.eu/balancing/energy/aggregatedBids/typeOfProductAggregation) |
| Total Load - Day-ahead and Actual*                         | 2022.01 – 2024.12 | 15 min | [entsoe.eu](https://newtransparency.entsoe.eu/balancing/energy/aggregatedBids/typeOfProductAggregation) |

\* The data is selected to be used as input for the models

## Data Sources

### Primary APIs and Downloads

1. **PICASSO data**: [TransnetBW PICASSO Platform](https://www.transnetbw.de/en/energy-market/ancillary-services/picasso)
2. **German grid data**: [Netztransparenz.de](https://www.netztransparenz.de/)
3. **The other data**: Downloaded from their websites

### Key Data Categories

- **Balancing Services**: aFRR (automatic Frequency Restoration Reserve) prices and exchanges
- **Market Prices**: Day-ahead and intraday pricing data
- **Load Data**: Forecast and actual electricity consumption
- **Generation Data**: Renewable energy forecasts and actual generation
- **Weather Data**: Supporting meteorological information
- **Balancing Energy**: Bids, merit orders, and activation data

## Time Coverage

All datasets span from **January 2022** to **December 2024**, with PICASSO data beginning in **May 2022**.

## Download PICASSO data from transnetbw
Download daily aFRR energy price files of PICASSO platform
```aiignore
cd path/to/this/repository
python download_picasso.py --start-date 2023-01-01 --end-date 2025-01-01 --type price
```
- start-date: the first date of downloading, in format `YYYY-MM-DD`
- end-date: the last day of downloading, in format `YYYY-MM-DD`
- type: `price` or `volume`
- lang: language, default = `de`
- save-dir: directory path to save the downloaded date, by default `Path.cwd() / "data" / "picasso"`


## Process PICASSO data from transnetbw
Downsample PICASSO data from transnetbw in 4s resolution to 15 min resolution.
```aiignore
cd path/to/this/repository

python
>>> from preprocessor import sample_picasso
# Customize start/end date
>>> start = datetime.strptime("2023-01-01", "%Y-%m-%d") 
>>> end = datetime.strptime("2023-01-03", "%Y-%m-%d")
>>> sample_picasso('price', 'AMP', start, end, '15min', 'min')
# The output csv will be save to: Path.cwd() / "data" / "picasso
```

## Netztransparenz API Data Fetcher

[//]: # (1. Go to [netztransparenz API]&#40;https://api-portal.netztransparenz.de/&#41;, and create a user account with your name and email.)

[//]: # (2. After registration, add client&#40;s&#41; in [my-clients tab]&#40;https://api-portal.netztransparenz.de/my-clients&#41;, and take note of the Client-ID and Client-secret.)

[//]: # (3. Add the Client-ID and Client-secret to your environment variables. &#40;Create an .env file in your project, add `IPNT_CLIENT_ID` and `IPNT_CLIENT_SECRET`&#41;)

[//]: # (4. Run python file `ip_index_api.py` to download the data. Remember to specify start date and end date&#40;not included&#41;. e.g. )

[//]: # (```angular2html)

[//]: # ( python ip_index_api.py --date-from 2022-02-02 --date-to 2022-02-03)

[//]: # (```)

[//]: # (# )

A Python script to fetch energy market data from the Netztransparenz API, including balancing energy prices (ID index), traffic light status, and other German grid data.

### Prerequisites

#### Required Python Packages
```bash
pip install requests python-dotenv
```

#### API Credentials
You need to obtain API credentials from Netztransparenz:
1. Visit [Netztransparenz.de](https://www.netztransparenz.de/)
2. Register for API access
3. Obtain your `CLIENT_ID` and `CLIENT_SECRET`

### Setup

#### 1. Environment Variables (Recommended)
Create a `.env` file in the same directory as your script:
```env
IPNT_CLIENT_ID=your_client_id_here
IPNT_CLIENT_SECRET=your_client_secret_here
```

#### 2. Directory Structure
The script will automatically create the following directory structure:
```
your_project/
├── netz-api.py
├── .env
└── data/
    └── netz/          # Downloaded CSV files will be saved here
```

### Usage

#### Basic Usage
```bash
python netz_api.py
```
This will download all available datasets from 2022-01-01 to 2025-01-01.

### Command Line Arguments

#### Authentication
```bash
# Override environment variables
python netz_api.py --client-id YOUR_CLIENT_ID --client-secret YOUR_CLIENT_SECRET
```

#### Date Range
```bash
# Specify custom date range
python netz_api.py --date-from 2023-01-01 --date-to 2024-01-01
```

#### Specific Endpoints
```bash
# Download specific data types
python netz_api.py --endpoints rebap
python netz_api.py --endpoints aep-module
python netz_api.py --endpoints traffic_light
```

### Available Data Types

| Endpoint | Description | Data Type |
|----------|-------------|-----------|
| `rebap` | Uniform imbalance price (reBAP) | Qualitaetsgesichert |
| `aep-module` | AEP Module data | Qualitaetsgesichert |
| `finan-aep` | Financial impact of AEP Module | Qualitaetsgesichert |
| `aep-estimator` | AEP Estimator | Betrieblich |
| `voaa` | VoAA (Volume of Avoided Activation) | Qualitaetsgesichert |
| `traffic_light` | GCC balance traffic light | Raw data |

### Examples

#### Download All Data for 2023
```bash
python netz_api.py --date-from 2023-01-01 --date-to 2024-01-01
```

#### Download Only Traffic Light Data
```bash
python netz_api.py --endpoints traffic_light --date-from 2024-01-01 --date-to 2024-12-31
```

#### Use Command Line Credentials
```bash
python netz_api.py \
    --client-id YOUR_CLIENT_ID \
    --client-secret YOUR_CLIENT_SECRET \
    --date-from 2024-06-01 \
    --date-to 2024-06-30 \
    --endpoints rebap
```

### Output

#### File Naming Convention
Downloaded files follow this pattern:
```
{data_type}_{product}_{start_date}_{end_date}.csv
```

Examples:
- `nrvsaldo_reBAP_Qualitaetsgesichert_2022-01-01_2025-01-01.csv`
- `TrafficLight_None_2022-01-01_2025-01-01.csv`

### Output Location
All files are saved to: `./data/netz/`
