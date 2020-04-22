
**The data in the files are not existent, and are just to familiarise with the format of the required file.**

### Close-1990.csv  

A dummy/fake data which has no correspondence with actual stock data. 
This file is just to familiarise with the format of the required file. 
The file should be named like: "type-yyyy.csv", where type = {'Close', 'Open'} and yyyy is the start year of training period.

### SPXconst.csv

A dummy constituent file of S&P500. 
The file has monthly constituent list of SP500.
Each column contains list of constituent stocks in SP500 for every month in the study period. 

**NOTE**: Please note that the names of the stock in SPXconst.csv should match the names of stocks in files with adjusted close and adjusted open prices.  

In the applications of the [paper](https://arxiv.org/abs/2004.10178), we use Bloomberg to retrieve adjusted stock prices.
