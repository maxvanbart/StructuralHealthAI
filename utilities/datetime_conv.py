import datetime


def convert_to_datetime(text):
    """Function to convert a datetime string to a timestamp"""
    date, t = text.split(' ')
    year, month, day = date.split('-')
    year, month, day = int(year), int(month), int(day)

    hour, minute, seconds = t.split(':')
    hour, minute, seconds = int(hour), int(minute), int(seconds)
    second = int(seconds // 1)
    milisecond = int(round((seconds % 1)*10**6))
    date_obj = datetime.datetime(year, month, day, hour, minute, second, milisecond)
    return datetime.datetime.timestamp(date_obj)
