from dotenv import load_dotenv
import os
import requests
import sys
import argparse

# Load environment variables from .env file
load_dotenv()
dir_path = os.path.dirname(__file__)
save_path = os.path.join(dir_path, 'data', 'netz')

endpoint_dict = {'rebap': {'data':'nrvsaldo/reBAP', 'product': 'Qualitaetsgesichert'},
                    'aep-module': {'data':'nrvsaldo/AEPModule', 'product': 'Qualitaetsgesichert'},
                    'finan-aep':{'data':'nrvsaldo/FinanzielleWirkungAEPModule', 'product': 'Qualitaetsgesichert'},
                    'aep-estimator':{'data':'nrvsaldo/AepSchaetzer', 'product': 'Betrieblich'},
                    'voaa':{'data':'nrvsaldo/VoAA', 'product': 'Qualitaetsgesichert'},
                    'traffic_light': {'data': 'TrafficLight', 'product': None},

                }

all_endpoint = endpoint_dict.keys()

parser = argparse.ArgumentParser(description='Fetch data from Netztransparenz API')

# Allow override of environment variables
parser.add_argument('--client-id',
                    default=os.environ.get('IPNT_CLIENT_ID'),
                    help='IPNT Client ID (or set IPNT_CLIENT_ID env var)')
parser.add_argument('--client-secret',
                    default=os.environ.get('IPNT_CLIENT_SECRET'),
                    help='IPNT Client Secret (or set IPNT_CLIENT_SECRET env var)')
parser.add_argument('--date-from', default='2022-01-01', help='Start date')
parser.add_argument('--date-to', default='2025-01-01', help='End date')
parser.add_argument('--endpoints', default=all_endpoint, help='End point list')

args = parser.parse_args()

# Check if credentials are provided
if not args.client_id or not args.client_secret:
    print("Error: Client ID and Secret must be provided via command line or environment variables")
    sys.exit(1)

# Rest of your code...
print(f"Using Client ID: {args.client_id[:10]}...")  # Only show first 10 chars for security

ACCESS_TOKEN_URL = "https://identity.netztransparenz.de/users/connect/token"

# Get token
response = requests.post(ACCESS_TOKEN_URL,
                         data={
                             'grant_type': 'client_credentials',
                             'client_id': args.client_id,
                             'client_secret': args.client_secret,
                         })

if response.ok:
    TOKEN = response.json()['access_token']
else:
    print(f'Error retrieving token\n{response.status_code}:{response.reason}',
          file=sys.stderr)
    exit(-1)

# Now get the actual data
endpoints =  args.endpoints if isinstance(args.endpoints, list) else [args.endpoints]
for name in endpoints:
    data = endpoint_dict[name]['data']
    product = endpoint_dict[name]['product']
    if product is None:
        data_url = f"https://ds.netztransparenz.de/api/v1/data/{data}/{args.date_from}/{args.date_to}"
    else:
        data_url = f"https://ds.netztransparenz.de/api/v1/data/{data}/{product}/{args.date_from}/{args.date_to}"
    response = requests.get(data_url, headers={'Authorization': 'Bearer {}'.format(TOKEN)})
    if response.ok:
        data_text = response.text

        # Save to file
        os.makedirs(save_path, exist_ok=True)
        data = '_'.join(data.split('/')) if len(data.split('/')) > 1 else data
        file_name = f'{data}_{product}_{args.date_from}_{args.date_to}.csv'
        with open(os.path.join(save_path, file_name), 'w') as f:
            f.write(data_text)
            f.close()

        print(f"Data retrieved successfully! Saved to {os.path.join(save_path, file_name)}")

    else:
        print(f'Error retrieving data\n{response.status_code}:{response.reason}')
        print(response.text)

