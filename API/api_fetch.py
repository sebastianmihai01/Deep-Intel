import datetime
import sys
import requests
import json


def format_print(jsonObj):
    val = json.dumps(jsonObj, sort_keys=True, indent=3)
    print(val)


def has_timestamp(response, timelist):
    times = []
    for ts in timelist:
        time = datetime.fromtimestamp(ts)
        times.append(time)
        print(time)
    time_list = response.json()['response']
    format_print(times)


def fetch(url, params):
    try:
        passedUrl = 'https://www.worldcoinindex.com/apiservice/ticker?key=umeifelkMQozfPe2dxpAXI5p6kAZlLbyfX9&label=ethbtc-ltcbtc&fiat=btc'
        if not url:
            passedUrl = url

        if not params:
            for elems in params:
                if not elems:
                    passedUrl += elems

        response = requests.get(passedUrl)
        print("Response code: ", response.status_code)
        format_print(response.json())

    except Exception as exc:
        try:
            exc_info = sys.exc_info()
            try:
                raise
            except:
                pass
        finally:
            del exc_info


try:
    val = input("Please enter your URL (press 'Enter' to access the predefined API) : ")
    if val == "":
        val = " ";
    fetch(val, list)
except ValueError as e:
    print("Error when introducing parameters! (input number should be of type int)")
