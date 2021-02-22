# Article: [*5 Levels of Handling Date and Time in Python*](https://medium.com/techtofreedom/5-levels-of-handling-date-and-time-in-python-46b601e47f65)


# 1. Switch Between Datetimes and Strings Properly
- `datetime.strptime()`: convert a string to a datetime object
- `datetime.strftime()`: convert a datetime object to a string

```python
from datetime import datetime

# convert a string to a datetime
string_time = '2020-12-25 20:20:20'
t = datetime.strptime(string_time, '%Y-%m-%d %H:%M:%S')
print(t)
# 2020-12-25 20:20:20
print(type(t))
# <class 'datetime.datetime'>

# convert a datetime to a string
now = datetime.now()
string_now = now.strftime('%a,  %d/%m/%Y %H:%M:%S')
print(string_now)
# Wed,  02/12/2020 23:27:05
print(type(string_now))
# <class 'str'>
```


# 2. Handle Time Zones Skilfully

```python
from datetime import datetime
import pytz

local = datetime.now()
print(local.strftime("%d/%m/%Y, %H:%M:%S"))
# 02/12/2020, 23:56:01

NY = pytz.timezone('America/New_York')
datetime_NY = datetime.now(NY)
print(datetime_NY.strftime("%d/%m/%Y, %H:%M:%S"))
# 02/12/2020, 18:56:01

Tokyo = pytz.timezone('Asia/Tokyo')
datetime_Tokyo = datetime.now(Tokyo)
print(datetime_Tokyo.strftime("%d/%m/%Y, %H:%M:%S"))
# 03/12/2020, 08:56:01
```
We can print the checklist to check all names of time zones:
```python
print(pytz.all_timezones)
```
