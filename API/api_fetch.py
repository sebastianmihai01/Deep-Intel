import datetime
import json
import sys

import requests


def format_print(jsonObj):
    # Takes JSON object => formats to string
    val = json.dumps(jsonObj, sort_keys=True, indent=3)
    print(val)
    # json.dumps() — Takes in a Python object, and converts (dumps) it to a string.
    # json.loads() — Takes a JSON string, and converts (loads) it to a Python object.


# can be used if the JSON element has a timestamp within
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
        # if input is empty => basic url is the one above
        if not url: #check if string is empty
            passedUrl = url

        if not params:
            for elems in params:
                if not elems:
                    passedUrl += elems

        response = requests.get(passedUrl)
        print("Response code: ", response.status_code)
        #print("Data: ", response.json())

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
    # Put your url here:
    val = input("Please enter your URL (press 'Enter' to access the predefined API) : ")

    #easier than string.split()
    if val == "":
        val = " ";

    # list = []
    # n = int(input("Enter the number of params: "))
    #
    # for i in range(0, n):
    #     if not inp:
    #         inp = input("Param: ")
    #         list.append(inp)
    fetch(val, list)
except ValueError as e:
    print("Error when introducing parameters! (input number should be of type int)")