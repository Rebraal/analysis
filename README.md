analysis
Analysis of memento data
For epoch 2019-09-01 04:00 to 2021-07-01 03:59

Restarting analysis completely and reconstructing the Analysis and Data sources folders inside here.
This means there will be duplicate data around, although this will contain the complete epoch data for all constructed stuff
Really ought to dump the whole lot onto a usb for transportablity.


Use Create numpy files as a basis for stuff. 
To do:
  Run create numpy files on TAS 2021-06, then Dump, selecting appropriate entries.
  Probably the simplest way to do this is to use Create numpy files directly and to alter the database
  funtions to access all three databases and return all the info at once. Just before I do this, have a check and see what supplements
  were actually cleaned.
  Going to have to do something with the useful_indices problem. Put it inside the list_content?
  Complete create numpy files, then run on all databases to complete epoch data.
  Then clean everything...

lib = 'Current status'
contains partial daily entries from 2021-10-01 to 2021-11-01
counts.min() = 1, counts.max() = 10

lib = 'Dump'
contains entries from 2021-06-01 to 2021-07-05 as old epoch. Staffs preps started here.
contains entries from 2021-07-05 to 2021-09-30 as partial daily entries. 

lib = 'Current status Copy'
contains nothing of use

lib = 'TAS 2021-06'
contains entries from 2021-01-01 to 2021-05-31 


2021-12-27 17:28
Create location and time lists sensibly from google

2021-12-29 14:12
Weather information from Visualcrossing up to date. 
Now intending to look at Ambee pollen data. While this is not pursuing initial main goal, this can be running in the background
as only a certain number of resources can be obtained per day.

2012-12-29 15:51
Can create arrays of minutes elapsed and from that date strings that respect daylight savings. Pollen data functions need an array
of locations with a location for each entry in the date string array.

2012-12-30 01:35
Can create such an expanded location array, and from it Pollen input data.

2021-12-30 11:15
Epoch pollen data collated.
