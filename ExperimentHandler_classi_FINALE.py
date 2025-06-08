from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when, lower, regexp_replace, isnan
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import Imputer, VectorAssembler, MinMaxScaler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import when
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col, lower, regexp_replace
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
os.makedirs("grafici", exist_ok=True)

spark = SparkSession.builder \
    .appName("FoodHealthClassifier") \
    .master("local[*]") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "8g") \
    .config("spark.storage.memoryFraction", "0.3") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .config("spark.network.timeout", "300s") \
    .config("spark.driver.maxResultSize", "2g") \
    .getOrCreate()

classifier = ExperimentHandler(spark)


class ExperimentHandler:
    """
    Gestisce l'intera pipeline Spark per l'analisi di classificazione.
    Inizializza una sessione Spark e definisce gli slot condivisi per i dati.
    """

    def __init__(self, spark):
        self.spark = spark  # La SparkSession viene passata dall’esterno!
        self.df = None
        self.numeric_features = []
        self.categorical_features = []