from google.cloud import storage
import os
import pandas as pd
import cv2
import datetime
from datetime import timezone, date
import pytz

# generate list of daily videos
class ListRetriever:

    def __init__(self):
        self.storage_client = storage.Client(project='nanobebe-production')
        self.bucket = 'production-nanobebe-video'

    def get_list(self, _date = date.today(), date_range = None, filters = []):
        #build the prefix
        year = _date.year
        week = _date.isocalendar()[1]
        day = _date.isocalendar()[2] + 1
        if day==8:
            day = 1
        prefix = str(year) + '/' + str(week) + '/' + str(day) + '/'
        blobs = list(self.storage_client.list_blobs(self.bucket,prefix = prefix))
        tz = pytz.timezone('UTC')

        # make _Date a datetime object
        _date = datetime.datetime.combine(_date, datetime.time(0, 0, 0, tzinfo=timezone.utc))
        # Filter blobs by the specific date
        if date_range is None:
            filtered_blobs = [
                blob for blob in blobs
                if blob.time_created.date() == _date.date() and any(
                    filter_string in blob.name for filter_string in filters)
            ]
        else:
            filtered_blobs = [
                blob for blob in blobs
                if
                blob.time_created >= date_range[0].tz_localize(tz) and blob.time_created <= date_range[1].tz_localize(
                    tz) and any(
                    filter_string in blob.name for filter_string in filters)
            ]

        #urls = [self.generate_signed_url(b) for b in filtered_blobs]
        return filtered_blobs

    def generate_signed_url(self, blob):



        url = blob.generate_signed_url(
            expiration=datetime.timedelta(minutes=10),
            method='GET',
            version='v4')

        return url

    def __call__(self, path, process):
        self.download(path)
        process(self.filePath)
        self.delete()
