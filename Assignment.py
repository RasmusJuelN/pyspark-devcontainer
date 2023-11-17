from datetime import datetime
import time
import requests
import pyspark.pandas as ps
import matplotlib.pyplot as plt
from pyspark.sql.functions import sum, col
from pyspark.sql.types import StructType, StructField, StringType, FloatType, DateType
from pyspark.sql import SparkSession


# Projektopgave for Faget BIG DATA - ETL 2. 

spark = SparkSession.builder.getOrCreate()

def extract(url):
    """
    Extracts data from the specified URL.

    Parameters:
    url (str): The URL to extract data from.

    Returns:
    dict: The extracted JSON data.
    """
    response = requests.get(url)
    result = response.json()
    return result

def transform_to_pyspark_dataframe(response):
    """
    Transforms the extracted JSON data into a PySpark DataFrame.

    Parameters:
    response (dict): The JSON data to transform.

    Returns:
    pyspark.pandas.DataFrame: Transformed PySpark DataFrame.
    """
    transformed_data = []
    for record in response['records']:
        transformed_data.append([record['Minutes5DK'], record['PriceArea'], record['OffshoreWindPower'], record['OnshoreWindPower'], record['SolarPower']])

    schema = StructType([
        StructField("Minutes5DK", StringType(), True),
        StructField("PriceArea", StringType(), True),
        StructField("OffshoreWindPower", FloatType(), True),
        StructField("OnshoreWindPower", FloatType(), True),
        StructField("SolarPower", FloatType(), True)
    ])

    # creates pyspark dataframe with the transformed data and schema
    psdf = spark.createDataFrame(data=transformed_data, schema=schema)
    psdf.show()

    return psdf

def get_summary_pr_5_minutes(psdf):
    """
    Summarizes the values for each 5-minute timestamp to get the total production for the whole of DK.

    Parameters:
    - psdf - PySpark DataFrame.

    Returns:
    pyspark DataFrame with the new data.
    """
    production_timestamp_summed = psdf.groupBy('Minutes5DK').sum()
    production_timestamp_summed.show()
    return production_timestamp_summed

def get_summary_for_priceArea(psdf):
    """
    Summarizes the values of the columns by PriceArea (for all columns where PriceArea is DK1 and where they are DK2).

    Parameters:
    psdf - PySpark DataFrame.

    Returns:
    pyspark DataFrame with the new data.
    """
    production_PriceArea_summed = psdf.groupBy('PriceArea').sum()
    production_PriceArea_summed.show()
    return production_PriceArea_summed

def calculate_percentage_contribution(psdf):
    """
    Calculates the percentage contribution of each energy type to the total sum.

    Parameters:
    psdf - PySpark DataFrame.

    Returns:
    DataFrame with percentage contribution columns.
    """
    # Total (renewable) energy production to calculate percentage
    psdf_totals = psdf.withColumn('TotalPower', col('OffshoreWindPower') + col('OnshoreWindPower') + col('SolarPower'))

    # how big a percentage does offshore wind, onshore wind and solar power contribute to the total production based on only these observations.
    psdf_percentage = psdf_totals.withColumn(
        'OffshoreWindPowerPercentage', (col('OffshoreWindPower') / col('TotalPower')) * 100
    ).withColumn(
        'OnshoreWindPowerPercentage', (col('OnshoreWindPower') / col('TotalPower')) * 100
    ).withColumn(
        'SolarPowerPercentage', (col('SolarPower') / col('TotalPower')) * 100
    )

    psdf_percentage.show()
    return psdf_percentage

def create_graph(psdf, price_areas=['DK1', 'DK2']):
    """
    Creates a line graph using matplotlib.pyplot. showing the SolarPower values for the specified PriceAreas (DK1 and DK2).

    Parameters:
    psdf - PySpark DataFrame.
    price_areas (list): List of PriceAreas to include in the graph.

    saves the graph to a .png file 'SolarPowerProduction_DK1_DK2.png'
    """
    plt.figure(figsize=(10, 6))

    for price_area in price_areas:
        filtered_psdf = psdf.filter(psdf['PriceArea'] == price_area)

        y_ans_val = [val.SolarPower for val in filtered_psdf.select('SolarPower').collect()]
        x_ts = [val.Minutes5DK for val in filtered_psdf.select('Minutes5DK').collect()]

        plt.plot(x_ts, y_ans_val, label=f'{price_area}')

    plt.ylabel('SolarPower')
    plt.xlabel('Minutes5DK')
    plt.title('ASN values for time')
    plt.legend(loc='upper left')

    # inverts the x axis to show the development from left to right instead.
    plt.gca().invert_xaxis()
    plt.savefig('SolarPowerProduction_DK1_DK2.png', dpi=1200)

def load_data_to_parquet(data):
    """
    Writes the DataFrame to a Parquet file.

    Parameters:
    - data (pyspark.pandas.DataFrame): Input PySpark DataFrame.
    """
    data.write.mode('overwrite').parquet('file.parquet')

def main():
    """
    Main function to execute the entire data processing pipeline.
    runs every 5 minutes in a loop, always getting the newest data.
    """

    # Api - getting electricity production every 5 minutes in real time. i put a limit 
    # on 10 entries (5 for each area DK1 and DK2) so i only have the newest data to work with all the time.
    url = 'https://api.energidataservice.dk/dataset/ElectricityProdex5MinRealtime?limit=10'

    while True:
        data = extract(url)
        transformed_data = transform_to_pyspark_dataframe(data)
        get_summary_for_priceArea(transformed_data) 
        get_summary_pr_5_minutes(transformed_data) 
        calculate_percentage_contribution(transformed_data)
        create_graph(transformed_data)
        load_data_to_parquet(transformed_data)
        time.sleep(300)

if __name__ == "__main__":
    main()
