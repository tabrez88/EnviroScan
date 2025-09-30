🌳 EnviroScan Dashboard: Advanced Environmental Monitoring and Source Attribution
I. Project Overview
The EnviroScan Dashboard is a robust, Streamlit-based application designed for the advanced analysis, visualization, and attribution of air quality data. This system integrates multiple data sources—OpenAQ, OpenWeather, and OpenStreetMap (via OSMnx)—to provide a comprehensive environmental intelligence platform. The primary objective is to move beyond simple monitoring to predict the likely source of pollution and enable proactive alerting.

II. Key Features
The platform's capabilities are focused on data fusion, machine learning, and actionable reporting:

Multi-Source Data Integration: A unified pipeline for merging heterogeneous datasets: ambient pollutant concentrations (OpenAQ), meteorological conditions (OpenWeather), and critical geospatial features (e.g., road density, industrial proximity) extracted using OSMnx.

Source Attribution Modeling: Utilizes sophisticated Machine Learning Classifiers (including Logistic Regression, Random Forest, and MLP) to classify pollution events into distinct source categories (e.g., Traffic, Industrial, Agricultural).

Automated Alerting Protocol: Implements an automated email dispatch system to notify stakeholders immediately when pollutant levels surpass established regulatory or health thresholds.

Geospatial Visualization: Features interactive Folium Heatmaps and marker clusters to visualize contamination hotspots and spatial distribution, aiding in quick situational assessment.

Comprehensive Reporting: Provides functionality for generating and exporting analysis outputs, including raw data summaries in CSV format and professional, formatted reports in PDF format.

III. Installation and Setup
3.1. Dependency Management
The application requires Python 3.8 or higher. All necessary libraries can be installed using the following command:

Bash

!pip install streamlit pandas numpy osmnx requests scikit-learn matplotlib seaborn folium streamlit-folium reportlab imblearn joblib
!pip install --upgrade scikit-learn
3.2. Configuration Requirements
Successful operation requires external service authentication:

API Key Integration: An active OpenWeatherMap API key must be configured within the application code to retrieve essential meteorological data.

Email Service Configuration: The alert system requires specific sender and recipient email addresses, along with an application-specific password for the sender's account, to ensure successful SMTP communication.

IV. Operational Guide
4.1. Application Launch
The dashboard is launched via the Streamlit Command-Line Interface (CLI):

streamlit run app.py

4.2. Analytical Workflow
Data Ingestion: Upload the target OpenAQ CSV file via the sidebar interface and define the necessary analysis parameters (time range, coordinates).

Data Processing: Initiate the ETL process by clicking the Process Data button. This step executes data cleaning, feature engineering, and external data fusion.

Analysis Execution: Select Train Models to execute the machine learning pipeline, providing metrics on model performance and source predictions.

Reporting: Utilize the dedicated section to generate and download all relevant summary and detailed reports.

V. Licensing
This project is released under the MIT License.
