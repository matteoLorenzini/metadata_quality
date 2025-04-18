import requests
import xml.etree.ElementTree as ET
import csv
import sys
from tqdm import tqdm

def fetch_records(endpoint, set_name, metadata_prefix="pico"):
    records = []
    params = {
        "verb": "ListRecords",
        "metadataPrefix": metadata_prefix,
        "set": set_name
    }
    
    total_records = 0
    try:
        with tqdm(desc="Fetching records", unit="record", bar_format="{l_bar}{bar}| {n_fmt} records [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            while True:
                response = requests.get(endpoint, params=params)
                if response.status_code != 200:
                    print(f"Error: Unable to fetch records. HTTP Status Code: {response.status_code}")
                    sys.exit(1)
                
                root = ET.fromstring(response.content)
                ns = {
                    'oai': 'http://www.openarchives.org/OAI/2.0/',
                    'dc': 'http://purl.org/dc/elements/1.1/',
                    'pico': 'http://purl.org/pico/1.0/',
                    'dcterms': 'http://purl.org/dc/terms/'
                }
                
                found_records = False
                for record in root.findall('.//pico:record', ns):  # Adjusted to find <pico:record>
                    title = record.find('dc:title', ns)
                    description = record.find('dc:description', ns)
                    type_ = record.find('dc:type', ns)
                    subject = record.find('dc:subject', ns)
                    
                    records.append({
                        "title": title.text if title is not None else "",
                        "description": description.text if description is not None else "",
                        "type": type_.text if type_ is not None else "",
                        "subject": subject.text if subject is not None else ""
                    })
                    total_records += 1
                    pbar.update(1)
                    found_records = True
                    '''                
                    # Stop fetching if the limit of 30 records is reached
                    if total_records >= 30:
                        print("Reached the limit of 30 records.")
                        return records
                    ''' 
                if not found_records:
                    break
                
                # Check for resumptionToken for pagination
                resumption_token = root.find('.//oai:resumptionToken', ns)
                if resumption_token is None or resumption_token.text is None:
                    break
                params = {
                    "verb": "ListRecords",
                    "resumptionToken": resumption_token.text
                }
    except Exception as e:
        print(f"Error during fetching records: {e}")
        sys.exit(1)
    
    return records

def save_to_csv(records, output_file):
    try:
        with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["title", "description", "type", "subject"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for record in records:
                writer.writerow(record)
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) != 3:
        print("Usage: python list_records_download.py <dataset_name> <output_file>")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    output_file = sys.argv[2]
    endpoint = "https://www.culturaitalia.it/oaiProviderCI/OAIHandler"
    
    print(f"Fetching records for dataset: {dataset_name}")
    records = fetch_records(endpoint, dataset_name)
    print(f"Fetched {len(records)} records.")
    
    print(f"Saving records to {output_file}")
    save_to_csv(records, output_file)
    print("Done.")

if __name__ == "__main__":
    main()
