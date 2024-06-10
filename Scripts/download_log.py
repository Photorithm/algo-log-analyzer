
# =========== downnload manually
from google.cloud import storage
storage_client = storage.Client()
list_files = storage_client.list_blobs('production-nanobebe-iot-logs', prefix=f'analysis/{mac_id}')
# download the file modified in 24-11

blobs = []
for idx, file in enumerate(list_files):
    if pd.to_datetime(file.time_created).tz_localize(None).day!=_date.tz_localize(None).day or pd.to_datetime(file.time_created).tz_localize(None).month!=_date.tz_localize(None).month:
        continue
    blobs.append(file)
    pass

blob = blobs[-1]
blob_uri = f"gs://{blob.bucket.name}/{blob.name}"
blob.download_to_filename(f'./{mac_id}.tgz')
print('saved to ./'+mac_id+'.tgz')
