## Overview
This project is about building deployable dashboards based on the requirements of BlueCrest Capital Coding Project. In this project, I choose python module $${\color{red}Streamlit}$$ for the front end design. 

## The whole submission includes the following components
1. README, for detailed introduction
2. requirements.txt, for listing the all the required python packages.
3. streamlit_app, the folder including all the python scripts about data ETL, calculating, and front-end dashboard design (with the help of Python module streamlit and plotly)
4. $${\color{blue}Screenshot Example}$$, the folder including the screenshot of the dashboard with default parameters. It enables user view the application page & layout without running it.
5. $${\color{blue}Screencast Example}$$, the folder including the screencast of the dashboard with default parameter. It enables user dynamically view the application page & layout without running it.
6. The database backend, by Mongodb. To build the database locally, the user should create database with name ‘bluecrest_data’, create the collection with name‘stock_data_prod1’. Then, user shoudl store the data by running /streamlit_app/ data_load/data_loader.py

## Steps of running App:
1. Create Python virtual environment:
   ```
   python -m venv env_name
   ```
2. Activate environment:
   ```
   source env_name/bin/activate
   ```
3. Installing all the required modules:
   ```
   python -m pip install -r /path/to/requirements.txt
   ```
4. Caching relevant data into the database:
   ```
   python path/to/streamlit_app/data_load/data_loader.py
   ```
5. Navigate to the path /path/to/streamlit_app/, type
   ```
   streamlit run main_page.py
   ```
6. Email wangzihao_fcb@outlook.com for any questions 

## Note on Dashboard
1. It contains a main page as greeting page.
2. Page1 (Equities Pair Trading) is for Project 1. Page2 (Index Regression) is For Project 2. The Cached data, by default, ranges from 2020.1 - 2024.8
3. Page1 has 4 parts (see corresponding 4 screenshots). 
   - Part1 is about the plotting the time series of close price and conducting correlation analysis given the stocks list and the date range. 
   - Part2 is about identifying the most highly correlated pairs based on the selected metrics and showing the selected pair’s correlation strength historical chart.
   - Part3 is about designing a simple trading strategy based on mean-reverting behavior of the price difference of the selected pair, with backtesting metrics.
   - Part4 is about designing a simple trading strategy based on mean-reverting behavior of the price ratio of the selected pair, with backtesting metrics. 
   - For simplicity, part3 and part4 do not take transaction cost into account. It assume we can get the signals by the end of the previous day and enter / exit the position using next day’s open price.
   - There are some designed buttons for user to type-in and try different parameters, and the plot and strategy evaluation will be updated accordingly.
4. Page2 (Index Regression) has 2 parts (see corresponding 2 screenshots). 
   - Part1 is about displaying the regression where the selected index’s daily return will be explained by the provided names’ return.
   - Part2 is about the tool that computes and identifies the set of Top K stocks best explaining the selected index within the date range. 
   - The regression details are showed as tables and plot in both part1 and part2. By clicking the checkbox of regressors, the user can see how the original series is v.s. the fitted one.
   - Part2 uses a step-by-step forward regressor selection method to find the top K names. Please note that, given the huge size of the stock pools, the process can be super slow for large K.
