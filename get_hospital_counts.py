import requests
import pandas as pd
import itertools

root = "https://data.chhs.ca.gov"
data_source = "/api/3/action/datastore_search?resource_id=0997fa8e-ef7c-43f2-8b9a-94672935fa60&limit=1000"

all_records = []

while data_source is not None:
    url = "{}{}".format(root, data_source)
    print("Getting {}".format(url))
    r = requests.get(url=url)
    data = r.json()

    if not data['success']:
        raise IOError

    records = data['result']['records']
    if not records:
        break
    all_records.append(records)

    if 'next' in data['result']['_links']:
        data_source = data['result']['_links']['next']
    else:
        data_source = None

df = pd.DataFrame(itertools.chain.from_iterable(all_records)).drop('_id', axis=1)

hospital_totals = df.groupby(['COUNTY_NAME', 'BED_CAPACITY_TYPE']).sum()
pd.set_option('display.max_rows', hospital_totals.shape[0] + 1)
print(hospital_totals)
hospital_totals.to_pickle("hospital_capacity.pkl")
