import requests
from datetime import datetime, timedelta
import argparse
import os
from datetime import datetime
from pathlib import Path

# url example for exchange volumes: https://api.transnetbw.de/picasso-interchange/csv?date={date_str}&lang={lang}
# url example for Cross-Border Marginal Prices: https://api.transnetbw.de/picasso-cbmp/csv?date={date_str}&lang={lang}

def download_picasso_csv(start_date, end_date, save_dir, data_type='price', lang='de'):
    type_dict = {
        'volume': 'interchange',
        'price': 'cbmp',
    }
    type_in_transnet = type_dict[data_type]

    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        url = f'https://api.transnetbw.de/picasso-{type_in_transnet}/csv?date={date_str}&lang={lang}'
        filename = f'{date_str} picasso_{type_in_transnet}.csv'
        save_path = os.path.join(save_dir, filename)

        print(f'Downloading: {url}')
        try:
            response = requests.get(url)
            response.raise_for_status()

            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f'Saved to {save_path}')
        except requests.HTTPError as err:
            print(f'Failed to download {date_str}: {err}')

        current_date += timedelta(days=1)


def get_default_save_dir():
    """Get a robust default save directory"""
    # Option 1: Use user's home directory
    home_dir = Path.home()
    default_dir = home_dir / "Documents" / "master_thesis" / "data" / "picasso"

    # Option 2: Use current working directory if home approach fails
    if not default_dir.exists():
        default_dir = Path.cwd() / "data" / "picasso"

    return str(default_dir)


def parse_date(date_string):
    """Parse date string in YYYY-MM-DD format"""
    try:
        return datetime.strptime(date_string, '%Y-%m-%d')
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_string}. Use YYYY-MM-DD")


def main():
    parser = argparse.ArgumentParser(
        description='Download Picasso CSV data with specified parameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--start-date',
        type=parse_date,
        default=datetime(2023, 1, 1),
        help='Start date in YYYY-MM-DD format (default: 2023-01-01)'
    )

    parser.add_argument(
        '--end-date',
        type=parse_date,
        default=datetime(2025, 1, 1),
        help='End date in YYYY-MM-DD format (default: 2025-01-01)'
    )

    parser.add_argument(
        '--save-dir',
        type=str,
        default=get_default_save_dir(),
        help='Directory to save the CSV files'
    )

    parser.add_argument(
        '--type',
        choices=['price', 'volume'],
        default='price',
        help='Type of data to download, can be price or volume (default: price)'
    )

    parser.add_argument(
        '--create-dirs',
        action='store_true',
        help='Create save directory if it doesn\'t exist'
    )

    args = parser.parse_args()

    # Convert save_dir to Path object for easier manipulation
    save_path = Path(args.save_dir)

    # Create directory if it doesn't exist and --create-dirs flag is set
    if args.create_dirs:
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {save_path}")
    elif not save_path.exists():
        print(f"Warning: Directory {save_path} does not exist.")
        print("Use --create-dirs flag to create it automatically.")
        return

    # Validate date range
    if args.start_date >= args.end_date:
        print("Error: Start date must be before end date.")
        return

    # Call the download function
    download_picasso_csv(
        start_date=args.start_date,
        end_date=args.end_date,
        save_dir=str(save_path),
        data_type=args.type,
        lang='de'
    )


if __name__ == '__main__':
    main()