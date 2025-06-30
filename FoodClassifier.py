import os
import time
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

from pyspark import SparkConf
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when, regexp_replace, lower
from pyspark.sql.types import DoubleType

from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array
from pyspark.ml.feature import Imputer, VectorAssembler, MinMaxScaler

from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
class DataLoader:
    """
    Classe responsabile del caricamento del dataset TSV.
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.df: DataFrame = None
        self.schema_info = {}
        self.n_rows = None
        self.n_cols = None

    def load(self, path: str) -> None:
        """
        Carica il dataset e aggiorna lo stato interno.
        """
        self.df = self.spark.read.csv(
            path,
            header=True,
            sep="\t",
            inferSchema=True,
            multiLine=True
        )
        self._update_metadata()

    def _update_metadata(self) -> None:
        """
        Aggiorna le informazioni di stato dopo il caricamento.
        """
        self.n_rows = self.df.count()
        self.n_cols = len(self.df.columns)
        self.schema_info = {field.name: field.dataType.simpleString() for field in self.df.schema.fields}

    def summary(self) -> dict:
        """
        Restituisce un dizionario riassuntivo con schema e dimensioni.
        """
        return {
            "rows": self.n_rows,
            "columns": self.n_cols,
            "schema": self.schema_info
        }

    def print_summary(self) -> None:
        """
        Stampa le informazioni sul dataset in formato leggibile.
        """
        print("\n Schema del dataset:")
        self.df.printSchema()
        print(f" Righe: {self.n_rows} | Colonne: {self.n_cols}")

    def get_df(self) -> DataFrame:
        """
        Restituisce il DataFrame caricato.
        """
        return self.df
class Cleaner:
    """
    Classe generica per la pulizia dei dati, adattabile a qualunque dataset.
    """

    def __init__(self, df: DataFrame, numeric_filter: callable = None, target_col: str = None):
        self.df = df
        self.numeric_filter = numeric_filter
        self.target_col = target_col
        self.numeric_features = []
        self.categorical_features = []
        self.cleaned = False
        self.cleaning_summary = {}

    def run(self) -> None:
        """
        Esegue tutte le fasi di pulizia in sequenza.
        """
        self.select_numeric_features()
        self.filter_nulls_and_cast()
        self.clean_categorical_columns()
        self._update_summary()
        self.cleaned = True

    def select_numeric_features(self) -> None:
        """
        Seleziona le feature numeriche secondo una logica esterna (funzione passata).
        """
        if self.numeric_filter:
            self.numeric_features = self.numeric_filter(self.df.columns)
        else:
            self.numeric_features = []

    def filter_nulls_and_cast(self) -> None:
        for c in self.numeric_features:
            self.df = self.df.withColumn(c, col(c).cast(DoubleType()))

        total_rows = self.df.count()
        null_stats = []
        for c in self.numeric_features:
            null_count = self.df.filter(col(c).isNull()).count()
            null_pct = null_count / total_rows
            if null_pct < 0.999:
                null_stats.append((c, null_pct))

        self.numeric_features = [c for c, _ in sorted(null_stats, key=lambda x: x[1])]

        selected_cols = self.numeric_features.copy()
        if self.target_col and self.target_col in self.df.columns:
            selected_cols.append(self.target_col)

        self.df = self.df.select(*selected_cols)

    def clean_categorical_columns(self) -> None:
        """
        Pulisce alcune colonne categoriche comuni, se presenti.
        """
        categorical = ["packaging", "labels", "categories", "ingredients_text"]
        cleaned_cols = []
        for c in categorical:
            if c in self.df.columns:
                self.df = self.df.withColumn(
                    c, lower(regexp_replace(col(c), '[^a-zA-Z0-9 ]', ''))
                )
                cleaned_cols.append(c)
        self.categorical_features = cleaned_cols

    def _update_summary(self) -> None:
        self.cleaning_summary = {
            "final_row_count": self.df.count(),
            "numeric_features_count": len(self.numeric_features),
            "categorical_features_count": len(self.categorical_features)
        }

    def summary(self) -> dict:
        return self.cleaning_summary if self.cleaned else {"warning": "Dataset not cleaned yet."}

    def print_summary(self) -> None:
        if not self.cleaned:
            print(" Il dataset non è stato ancora pulito.")
            return
        print("\n Riepilogo della pulizia:")
        print(f" - Righe finali: {self.cleaning_summary['final_row_count']}")
        print(f" - Feature numeriche selezionate: {self.cleaning_summary['numeric_features_count']}")
        print(f" - Feature categoriche pulite: {self.cleaning_summary['categorical_features_count']}")

    def print_numeric_features(self) -> None:
        if not self.numeric_features:
            print(" Nessuna feature numerica selezionata.")
            return
        print("\n Feature numeriche utilizzate:")
        for feature in self.numeric_features:
            print(f" - {feature}")

    def get_df(self) -> DataFrame:
        return self.df

    def get_numeric_features(self) -> list:
        return self.numeric_features
class TargetEngineer:
    """
    Si occupa della definizione della variabile target (label) per il task di classificazione.
    Il comportamento è personalizzabile tramite una funzione di mapping esterna.
    """

    def __init__(
        self,
        df: DataFrame,
        label_column: str,
        mapping_fn: Callable[[Column], Column]
    ):
        self.df = df
        self.label_column = label_column
        self.mapping_fn = mapping_fn
        self.label_stats = {}
        self._rows_before_labeling = None
        self._rows_after_labeling = None

    def run(self) -> None:
        """
        Esegue l'assegnazione della colonna 'label' e aggiorna le statistiche.
        """
        self._assign_label()
        self._update_label_stats()

    def _assign_label(self) -> None:
        """
        Aggiunge la colonna 'label' al DataFrame utilizzando la funzione di mapping specificata.
        """
        self._rows_before_labeling = self.df.count()
        self.df = self.df.withColumn("label", self.mapping_fn(col(self.label_column)))
        self.df = self.df.dropna(subset=["label"])
        self._rows_after_labeling = self.df.count()

    def _update_label_stats(self) -> None:
        """
        Calcola statistiche sulla distribuzione della variabile target.
        """
        count_0 = self.df.filter(col("label") == 0).count()
        count_1 = self.df.filter(col("label") == 1).count()
        discarded = self._rows_before_labeling - self._rows_after_labeling

        self.label_stats = {
            "sano": count_0,
            "non_sano": count_1,
            "discarded": discarded
        }

    def get_df(self) -> DataFrame:
        """
        Restituisce il DataFrame aggiornato con la colonna 'label'.
        """
        return self.df

    def get_label_stats(self) -> dict:
        """
        Restituisce le statistiche sulla distribuzione della variabile target.
        """
        return self.label_stats

    def print_label_stats(self) -> None:
        """
        Stampa a console le statistiche sulla distribuzione della variabile target.
        """
        if not self.label_stats:
            print(" Le statistiche del target non sono ancora disponibili.")
            return

        print("\n Statistiche sulla variabile target:")
        print(f" - Sano (0): {self.label_stats['sano']}")
        print(f" - Non sano (1): {self.label_stats['non_sano']}")
        print(f" - Righe scartate: {self.label_stats['discarded']}")

    def plot_label_distribution(self, output_path: str = "grafici/distribuzione_label.png") -> None:
        """
        Genera e salva un grafico a barre della distribuzione della variabile target.
        """
        pandas_df = self.df.select("label").toPandas()
        self._plot_label_bar_chart(pandas_df, output_path)

    def _plot_label_bar_chart(self, pandas_df: pd.DataFrame, output_path: str) -> None:
        counts = pandas_df["label"].value_counts().sort_index()

        plt.figure(figsize=(5, 4))
        counts.plot(kind="bar", color=["green", "red"])
        plt.title("Distribuzione delle classi")
        plt.xticks([0, 1], ["Sano (0)", "Non sano (1)"])
        plt.ylabel("Numero di campioni")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
class Preprocessor:
    """
    Gestisce il preprocessing dei dati: imputazione, scalatura e assemblaggio del vettore delle feature.
    """

    def __init__(self, df: DataFrame, numeric_features: list):
        self.df = df
        self.numeric_features = numeric_features
        self.valid_features = []
        self.status = {
            "elapsed_time": None,
            "num_features": None
        }

    def run(self) -> None:
        """
        Esegue tutte le fasi di preprocessing in sequenza.
        """
        start = time.time()
        self.impute_missing_values()
        self.scale_and_assemble_features()
        self.explode_features_for_debug()
        elapsed = time.time() - start
        self.status["elapsed_time"] = round(elapsed, 2)
        self.status["num_features"] = len(self.numeric_features)

    def impute_missing_values(self) -> None:
        imputed_cols = [f"{c}_imputed" for c in self.numeric_features]
        imputer = Imputer(inputCols=self.numeric_features, outputCols=imputed_cols).setStrategy("median")
        self.df = imputer.fit(self.df).transform(self.df)
        self.numeric_features = imputed_cols

    def scale_and_assemble_features(self) -> None:
        assembler = VectorAssembler(inputCols=self.numeric_features, outputCol="unscaled_features")
        scaler = MinMaxScaler(inputCol="unscaled_features", outputCol="features")
        pipeline = Pipeline(stages=[assembler, scaler])
        self.df = pipeline.fit(self.df).transform(self.df)

    def explode_features_for_debug(self) -> None:
        self.df = self.df.withColumn("features_array", vector_to_array("features"))
        for i, name in enumerate(self.numeric_features):
            self.df = self.df.withColumn(name, self.df["features_array"][i])
        selected_cols = self.numeric_features + ["features", "label"]
        self.df = self.df.select(*selected_cols)
        self.valid_features = self.numeric_features

    def get_df(self) -> DataFrame:
        return self.df

    def get_valid_features(self) -> list:
        return self.valid_features

    def get_status(self) -> dict:
        return self.status

    def print_status(self) -> None:
        """
        Visualizza le informazioni sul preprocessing eseguito.
        """
        if self.status["elapsed_time"] is not None:
            print("\n Preprocessing completato:")
            print(f" - Tempo impiegato: {self.status['elapsed_time']} secondi")
            print(f" - Numero di feature utilizzate: {self.status['num_features']}")
        else:
            print(" Preprocessing non ancora eseguito.")
class DatasetExporter:
    """
    Classe responsabile dell'esportazione e dell'anteprima del dataset finale.
    """

    def __init__(self, df: DataFrame, exclude_cols: set = None):
        """
        Inizializza l'oggetto DatasetExporter.

        :param df: DataFrame da esportare o visualizzare.
        :param exclude_cols: Colonne da escludere nelle esportazioni e anteprime.
        """
        self.df = df
        self.exclude_cols = exclude_cols if exclude_cols else {"features", "unscaled_features"}
        self.last_export_path = None
        self.last_preview = None

    def export(self, path: str = "final_dataset.csv") -> None:
        """
        Esporta il dataset in formato CSV, escludendo le colonne tecniche.
        """
        pd.options.display.float_format = '{:.5f}'.format
        export_df = self._get_readable_df()
        export_df.write.csv(path, header=True, mode="overwrite")
        self.last_export_path = path

    def preview(self, limit: int = 1, transpose: bool = True) -> pd.DataFrame:
        """
        Restituisce un'anteprima del dataset finale.
        """
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_rows', None)

        pdf = self._get_readable_df().limit(limit).toPandas()
        self.last_preview = pdf.T if transpose else pdf
        return self.last_preview

    def print_preview(self) -> None:
        """
        Stampa l'ultima anteprima generata o la genera al volo.
        """
        if self.last_preview is None:
            self.preview()
        print("\n Anteprima del dataset finale:\n")
        print(self.last_preview.to_string(header=False, index=True))

    def _get_readable_df(self) -> DataFrame:
        """
        Restituisce il DataFrame filtrato escludendo le colonne tecniche.
        """
        readable_cols = [c for c in self.df.columns if c not in self.exclude_cols]
        return self.df.select(readable_cols)

    def get_df(self) -> DataFrame:
        return self.df

    def get_last_preview(self) -> pd.DataFrame:
        return self.last_preview
class ModelTrainer:
    """
    Classe responsabile dell'addestramento di un classificatore.
    Di default utilizza Random Forest, ma è compatibile con altri modelli Spark ML.
    """

    def __init__(self, df: DataFrame, classifier=None):
        self.df = df
        self.model = None
        self.train_data = None
        self.test_data = None
        self.training_time = None

        # Classificatore di default: Random Forest
        self.classifier = classifier or RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            numTrees=100,
            maxDepth=5,
            seed=42
        )

    def split_data(self, train_ratio: float = 0.8, seed: int = 42) -> None:
        """
        Divide il dataset in training e test set.
        """
        self.train_data, self.test_data = self.df.randomSplit([train_ratio, 1 - train_ratio], seed=seed)

    def train(self) -> None:
        """
        Addestra il modello specificato nel costruttore.
        """
        if self.train_data is None or self.test_data is None:
            self.split_data()

        start_time = time.time()
        self.model = self.classifier.fit(self.train_data)
        self.training_time = time.time() - start_time

    def print_training_info(self) -> None:
        """
        Visualizza le informazioni sull'addestramento eseguito.
        """
        if self.model is None:
            print(" Nessun modello è stato addestrato.")
            return

        print("\n Modello addestrato:")
        print(f" - Tipo: {type(self.classifier).__name__}")
        print(f" - Righe nel training set: {self.train_data.count()}")
        print(f" - Tempo di addestramento: {self.training_time:.2f} secondi")

    def get_model(self):
        return self.model

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def get_training_time(self):
        return self.training_time
class ModelVisualizer:
    """
    Classe responsabile della visualizzazione e salvataggio dei risultati del modello.
    """

    def __init__(self, numeric_features: list):
        self.numeric_features = numeric_features
        self.last_feature_path = "grafici/feature_importance.png"
        self.last_cm_path = None
        self.last_roc_path = None
        self.top_features = []

    def plot_feature_importance(self, model, top_n: int = 15) -> None:
        importances = model.featureImportances.toArray()
        sorted_idx = np.argsort(importances)[::-1]
        top_idx = sorted_idx[:top_n]
        self.top_features = [(self.numeric_features[i], importances[i]) for i in top_idx]

        self._plot_feature_bar(importances, top_idx)
        self._export_feature_rank(importances)

    def _plot_feature_bar(self, importances, top_idx) -> None:
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(top_idx)), importances[top_idx])
        plt.xticks(range(len(top_idx)), [self.numeric_features[i] for i in top_idx], rotation=90)
        plt.tight_layout()
        os.makedirs("grafici", exist_ok=True)
        plt.savefig(self.last_feature_path)
        plt.close()

    def _export_feature_rank(self, importances) -> None:
        all_sorted = sorted(zip(self.numeric_features, importances), key=lambda x: -x[1])
        self.top_features = all_sorted[:10]
        os.makedirs("grafici", exist_ok=True)
        with open("grafici/feature_importance.txt", "w") as f:
            for name, score in all_sorted:
                f.write(f"{name}: {score:.4f}\n")

    def print_feature_importance_info(self) -> None:
        print(f"\n Feature Importance salvata in: {self.last_feature_path}")
        print("Top 10 Feature Importance:")
        for name, score in self.top_features:
            print(f" - {name}: {score:.4f}")

    def plot_confusion_matrix(self, predictions, title: str = "Confusion Matrix") -> None:
        y_true, y_pred = self._extract_labels_from_predictions(predictions)
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Sano (0)", "Non sano (1)"])

        plt.figure(figsize=(6, 4))
        disp.plot(cmap="Blues", values_format='d')
        plt.title(title)
        plt.tight_layout()

        os.makedirs("grafici", exist_ok=True)
        self.last_cm_path = f"grafici/{title.replace(' ', '_').lower()}.png"
        plt.savefig(self.last_cm_path)
        plt.close()

    def plot_roc_curves(self, models: dict, test_data, output_path="grafici/roc_curve.png") -> None:
        os.makedirs("grafici", exist_ok=True)
        plt.figure(figsize=(6, 5))

        for name, model in models.items():
            fpr, tpr, roc_auc = self._get_roc_data(model, test_data)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        self.last_roc_path = output_path

    def _extract_labels_from_predictions(self, predictions):
        y_true = [int(row["label"]) for row in predictions.select("label").collect()]
        y_pred = [int(row["prediction"]) for row in predictions.select("prediction").collect()]
        return y_true, y_pred

    def _get_roc_data(self, model, test_data):
        predictions = model.transform(test_data)
        pred_df = predictions.select("label", "probability") \
            .withColumn("prob", vector_to_array("probability")[1]) \
            .toPandas()

        fpr, tpr, _ = roc_curve(pred_df["label"], pred_df["prob"])
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc

    def get_top_features(self) -> list:
        return self.top_features

    def get_paths(self) -> dict:
        return {
            "feature_importance": self.last_feature_path,
            "confusion_matrix": self.last_cm_path,
            "roc_curve": self.last_roc_path
        }
class ModelEvaluator:
    """
    Classe responsabile della valutazione del modello e della gestione delle metriche.
    """

    def __init__(self, df: DataFrame, numeric_features: list):
        self.df = df
        self.numeric_features = numeric_features
        self.last_metrics = {}
        self.last_path = None
        self.predictions = None

    def evaluate(self, model, test_data) -> None:
        """
        Valuta il modello sul test set e aggiorna le metriche interne.
        """
        start_time = time.time()
        self.predictions = model.transform(test_data)
        self.last_metrics = self._compute_metrics(self.predictions)
        self.last_metrics["tempo"] = time.time() - start_time

    def _compute_metrics(self, predictions) -> dict:
        total = predictions.count()
        correct = predictions.filter(col("label") == col("prediction")).count()
        accuracy = correct / total if total > 0 else 0

        tp = predictions.filter((col("label") == 1) & (col("prediction") == 1)).count()
        tn = predictions.filter((col("label") == 0) & (col("prediction") == 0)).count()
        fp = predictions.filter((col("label") == 0) & (col("prediction") == 1)).count()
        fn = predictions.filter((col("label") == 1) & (col("prediction") == 0)).count()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        evaluator = BinaryClassificationEvaluator(
            labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
        )
        auc = evaluator.evaluate(predictions)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc
        }

    def export_metrics(self, ottimizzato: bool = False) -> None:
        """
        Esporta le metriche su file di testo.
        """
        if not self.last_metrics:
            print(" Nessuna metrica da salvare.")
            return

        os.makedirs("grafici", exist_ok=True)
        nome_modello = "ottimizzato" if ottimizzato else "base"
        path = f"grafici/valutazione_modello_{nome_modello}.txt"
        self.last_path = path

        with open(path, "w") as f:
            f.write(f" Valutazione del modello {nome_modello}\n")
            f.write(f"Accuracy:  {self.last_metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {self.last_metrics['precision']:.4f}\n")
            f.write(f"Recall:    {self.last_metrics['recall']:.4f}\n")
            f.write(f"F1 Score:  {self.last_metrics['f1']:.4f}\n")
            f.write(f"AUC:       {self.last_metrics['auc']:.4f}\n")
            f.write(f"Tempo:     {self.last_metrics['tempo']:.2f} sec\n")

    def print_metrics(self) -> None:
        """
        Stampa a schermo le metriche dell'ultima valutazione.
        """
        m = self.last_metrics
        if not m:
            print(" Nessuna metrica disponibile.")
            return

        print("\n Valutazione del modello:")
        print(f" - Accuracy:  {m['accuracy']:.4f}")
        print(f" - Precision: {m['precision']:.4f}")
        print(f" - Recall:    {m['recall']:.4f}")
        print(f" - F1 Score:  {m['f1']:.4f}")
        print(f" - AUC:       {m['auc']:.4f}")
        print(f" - Tempo:     {m['tempo']:.2f} sec")

    def print_wrong_predictions(self, limit: int = 5) -> None:
        """
        Stampa un sottoinsieme di predizioni errate.
        """
        if self.predictions is None:
            print(" Nessuna predizione disponibile.")
            return

        renamed_df = self.df
        for col_name in self.numeric_features:
            renamed_df = renamed_df.withColumnRenamed(col_name, f"{col_name}_orig")
        renamed_df = renamed_df.withColumnRenamed("label", "label_orig")

        joined = self.predictions.join(renamed_df, on="features", how="inner")
        wrong = joined.filter(col("label") != col("prediction"))

        top_features = [
            "saturated-fat_100g_imputed",
            "fat_100g_imputed",
            "sugars_100g_imputed",
            "energy_100g_imputed",
            "salt_100g_imputed"
        ]
        columns_to_show = ["label", "prediction"] + [f"{f}_orig" for f in top_features if f in self.numeric_features]

        print(f"\n Esempi di predizioni errate (max {limit}):")
        wrong.select(*columns_to_show).show(n=limit, truncate=False)

    def get_metrics(self) -> dict:
        return self.last_metrics

    def get_predictions(self) -> DataFrame:
        return self.predictions

    def get_path(self) -> str:
        return self.last_path
class ModelOptimizer:
    """
    Classe responsabile della ricerca dei migliori iperparametri tramite cross-validation.
    """

    def __init__(self, df: DataFrame):
        self.df = df
        self.cv_model = None
        self.train_data = None
        self.test_data = None
        self.best_params = {}
        self.cv_duration = None
        self.param_grid = None
        self.evaluator = None

    def run(self, num_folds: int = 3, parallelism: int = 2) -> None:
        """
        Esegue la cross-validation completa.
        """
        self._split_data()
        self._build_param_grid()
        self._build_evaluator()

        rf = RandomForestClassifier(featuresCol="features", labelCol="label", seed=42)

        cv = CrossValidator(
            estimator=rf,
            estimatorParamMaps=self.param_grid,
            evaluator=self.evaluator,
            numFolds=num_folds,
            parallelism=parallelism
        )

        start = time.time()
        fitted_cv = cv.fit(self.train_data)
        self.cv_duration = time.time() - start

        self.cv_model = fitted_cv.bestModel
        self.best_params = {
            "numTrees": self.cv_model.getNumTrees,
            "maxDepth": self.cv_model.getOrDefault("maxDepth")
        }

    def _split_data(self, ratio: float = 0.8, seed: int = 42) -> None:
        self.train_data, self.test_data = self.df.randomSplit([ratio, 1 - ratio], seed=seed)

    def _build_param_grid(self) -> None:
        rf = RandomForestClassifier()
        self.param_grid = ParamGridBuilder() \
            .addGrid(rf.numTrees, [50, 100]) \
            .addGrid(rf.maxDepth, [5, 10]) \
            .build()

    def _build_evaluator(self) -> None:
        self.evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

    def print_summary(self) -> None:
        """
        Stampa i risultati della cross-validation.
        """
        if not self.cv_model:
            print(" Cross-validation non ancora eseguita.")
            return

        print("\n Cross-validation completata.")
        print(f" Durata: {self.cv_duration:.2f} secondi")
        print(" Parametri ottimali trovati:")
        print(f" - NumTrees: {self.best_params['numTrees']}")
        print(f" - MaxDepth: {self.best_params['maxDepth']}")

    def get_best_model(self):
        return self.cv_model

    def get_test_data(self):
        return self.test_data

    def get_train_data(self):
        return self.train_data

    def get_params(self):
        return self.best_params

    def get_duration(self):
        return self.cv_duration

    def get_evaluator(self):
        return self.evaluator

    def get_param_grid(self):
        return self.param_grid
class ModelComparer:
    """
    Classe responsabile del confronto tra modelli classici
    e tra versioni base/ottimizzata della Random Forest.
    """

    def __init__(self, df: DataFrame):
        self.df = df
        self.results = []
        self.rf_comparison_text = ""
        self.last_classic_path = "grafici/confronto_modelli_classici.txt"
        self.last_rf_path = "grafici/confronto_modelli.txt"

    def run_comparison(self) -> None:
        """
        Esegue il confronto tra Random Forest, Logistic Regression e GBT.
        """
        train_data, test_data = self.df.randomSplit([0.8, 0.2], seed=42)

        models = {
            "Random Forest": RandomForestClassifier(labelCol="label", featuresCol="features", seed=42),
            "Logistic Regression": LogisticRegression(labelCol="label", featuresCol="features", maxIter=10),
            "GBT": GBTClassifier(labelCol="label", featuresCol="features", maxIter=10)
        }

        evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
        results = []

        for name, algo in models.items():
            acc, auc = self._evaluate_model(algo, train_data, test_data, evaluator)
            results.append((name, acc, auc))

        self.results = results
        self._save_classic_comparison()

    def _evaluate_model(self, model_algo, train_data, test_data, evaluator) -> tuple:
        """
        Allena e valuta un classificatore restituendo accuracy e AUC.
        """
        model = model_algo.fit(train_data)
        predictions = model.transform(test_data)

        auc = evaluator.evaluate(predictions)
        correct = predictions.filter(col("label") == col("prediction")).count()
        total = predictions.count()
        accuracy = correct / total if total > 0 else 0

        return accuracy, auc

    def _save_classic_comparison(self) -> None:
        """
        Salva il confronto tra i modelli classici su file.
        """
        os.makedirs("grafici", exist_ok=True)
        with open(self.last_classic_path, "w") as f:
            f.write(" Confronto tra modelli (Random Forest, Logistic Regression, GBT)\n")
            f.write("-" * 60 + "\n")
            for name, acc, auc in self.results:
                f.write(f"{name:>20}: Accuracy = {acc:.4f} | AUC = {auc:.4f}\n")

    def compare_rf_versions(self, base_cm: tuple, optimized_cm: tuple) -> None:
        """
        Confronta due confusion matrix (base vs ottimizzato) e salva su file.
        """
        self.rf_comparison_text = self._format_rf_comparison(base_cm, optimized_cm)
        self._save_rf_comparison()

    def _format_rf_comparison(self, base_cm, optimized_cm) -> str:
        """
        Ritorna una stringa formattata con il confronto tra due confusion matrix.
        """
        tn_b, fp_b, fn_b, tp_b = base_cm
        tn_o, fp_o, fn_o, tp_o = optimized_cm
        err_b = fp_b + fn_b
        err_o = fp_o + fn_o

        lines = [
            " Confronto tra Modello Base e Ottimizzato",
            "------------------------------------------------",
            f"{'':<15} | {'Base':^10} | {'Ottimizzato':^12}",
            "-" * 43,
            f"{'TP':<15} | {tp_b:^10} | {tp_o:^12}",
            f"{'TN':<15} | {tn_b:^10} | {tn_o:^12}",
            f"{'FP':<15} | {fp_b:^10} | {fp_o:^12}",
            f"{'FN':<15} | {fn_b:^10} | {fn_o:^12}",
            f"{'Errori totali':<15} | {err_b:^10} | {err_o:^12}",
            "------------------------------------------------"
        ]
        return "\n".join(lines)

    def _save_rf_comparison(self) -> None:
        os.makedirs("grafici", exist_ok=True)
        with open(self.last_rf_path, "w") as f:
            f.write(self.rf_comparison_text + "\n")

    def print_classic_comparison(self) -> None:
        """
        Stampa i risultati del confronto tra i modelli classici.
        """
        if not self.results:
            print(" Nessun confronto disponibile.")
            return

        print("\n Confronto tra modelli classici:")
        for name, acc, auc in self.results:
            print(f" {name:>20}: Accuracy = {acc:.4f} | AUC = {auc:.4f}")

    def print_rf_comparison(self) -> None:
        """
        Stampa il confronto testuale tra RF base e ottimizzato.
        """
        if self.rf_comparison_text:
            print("\n Confronto Modello Base vs Ottimizzato:")
            print(self.rf_comparison_text)
        else:
            print(" Nessun confronto disponibile.")

    def get_classic_results(self) -> list:
        return self.results

    def get_rf_comparison_text(self) -> str:
        return self.rf_comparison_text

    def get_paths(self) -> dict:
        return {
            "classic": self.last_classic_path,
            "rf_comparison": self.last_rf_path
        }
def run_experiment(spark, path: str):
    """
    Esegue l'intero esperimento end-to-end:
    - Caricamento e cleaning dei dati
    - Definizione del target
    - Preprocessing
    - Training base e ottimizzato
    - Valutazione e confronto tra modelli
    """
    # === 1. Caricamento Dataset ===
    loader = DataLoader(spark)
    loader.load(path)
    df = loader.df


    # === 2. Cleaning Dataset ===
    def select_features(cols):
        """
        Funzione specifica per il dataset: seleziona solo le feature numeriche che finiscono con '_100g',
        escludendo alcune variabili note non utili.
        """
        exclude = {"nutrition_grade_fr", "nutrition_grade_uk", "nutrition-score-fr_100g", "nutrition-score-uk_100g"}
        return [c for c in cols if c.endswith("_100g") and c not in exclude]

    cleaner = Cleaner(df, numeric_filter=select_features, target_col="nutrition_grade_fr")
    cleaner.run()
    cleaner.print_summary()
    cleaner.print_numeric_features()
    df = cleaner.get_df()
    numeric_features = cleaner.get_numeric_features()

    # === 3. Definizione Target ===
    def food_label_mapping(col_):
        return when(col_.isin("a", "b"), 0).when(col_.isin("d", "e"), 1)

    target_engineer = TargetEngineer(
        df,
        label_column="nutrition_grade_fr",
        mapping_fn=food_label_mapping
    )
    target_engineer.run()
    target_engineer.print_label_stats()
    df = target_engineer.get_df()
    target_engineer.plot_label_distribution()
    print("\n Grafico distribuzione label salvato in: grafici/distribuzione_label.png")

    # === 4. Preprocessing ===
    preprocessor = Preprocessor(df, numeric_features)
    preprocessor.run()
    df = preprocessor.df

    # === 5. Esportazione Dataset Finale ===
    exporter = DatasetExporter(df)
    exporter.print_preview()
    exporter.export()

    # === 6. Training del Modello Base ===
    trainer = ModelTrainer(df)
    trainer.train()
    model = trainer.get_model()
    test_data = trainer.get_test_data()

    # === 7. Valutazione del Modello Base ===
    evaluator = ModelEvaluator(df, numeric_features)
    evaluator.evaluate(model, test_data)
    evaluator.export_metrics(ottimizzato=False)
    evaluator.print_metrics()
    base_predictions = model.transform(test_data)
    visualizer = ModelVisualizer(numeric_features)

    visualizer.plot_confusion_matrix(base_predictions, title="Confusion Matrix - Base")

    # === 8. Visualizzazione Confusion Matrix Base ===
    visualizer = ModelVisualizer(numeric_features)
    visualizer.plot_confusion_matrix(base_predictions, title="Confusion Matrix - Base")

    # === 9. Ottimizzazione con Cross-Validation ===
    optimizer = ModelOptimizer(df)
    optimizer.run()
    best_model = optimizer.get_best_model()
    test_data_opt = optimizer.get_test_data()
    df = optimizer.df
    optimizer.print_summary()

    # === 10. Valutazione Modello Ottimizzato ===
    evaluator.evaluate(best_model, test_data_opt)
    evaluator.export_metrics(ottimizzato=True)
    evaluator.print_metrics()

    opt_predictions = best_model.transform(test_data_opt)
    visualizer.plot_confusion_matrix(opt_predictions, title="Confusion Matrix - Ottimizzato")

    # === 11. Visualizzazione Feature Importance ===
    visualizer.plot_feature_importance(best_model)
    visualizer.print_feature_importance_info()

    # === 12. Confronto Confusion Matrix Base vs Ottimizzato ===
    cm_base = visualizer._extract_labels_from_predictions(base_predictions)
    cm_opt = visualizer._extract_labels_from_predictions(opt_predictions)
    cm_base_values = confusion_matrix(*cm_base).ravel()
    cm_opt_values = confusion_matrix(*cm_opt).ravel()

    comparer = ModelComparer(df)
    comparer.compare_rf_versions(tuple(cm_base_values), tuple(cm_opt_values))

    # === 13. Visualizzazione Predizioni Errate ===
    evaluator.print_wrong_predictions(limit=5)

    # === 14. Confronto tra Modelli Classici ===
    comparer.run_comparison()
    comparer.print_classic_comparison()
    comparer.print_rf_comparison()

    # === 15. ROC Curve dei Modelli ===
    train_roc, _ = df.randomSplit([0.8, 0.2], seed=42)
    models = {
        "Random Forest": model,
        "Logistic Regression": LogisticRegression(labelCol="label", featuresCol="features", maxIter=10).fit(train_roc),
        "GBT": GBTClassifier(labelCol="label", featuresCol="features", maxIter=10).fit(train_roc)
    }
    visualizer.plot_roc_curves(models, test_data)

if __name__ == "__main__":
    # Imposta JAVA_HOME in modo dinamico
    os.environ["JAVA_HOME"] = subprocess.check_output(
        ["/usr/libexec/java_home"]
    ).decode().strip()

    # Configurazione Spark
    conf = SparkConf()
    conf.set("spark.driver.extraJavaOptions", "-Duser.timezone=UTC")

    spark = SparkSession.builder \
        .config(conf=conf) \
        .appName("FoodClassifier") \
        .master("local[*]") \
        .getOrCreate()

    print("\nEsecuzione pipeline...")

    # Avvia la pipeline con funzione globale
    run_experiment(spark, "/Users/Cristian/Desktop/DatasetFoodfacts.tsv")

    print("\nProcesso completato con successo!")

    spark.stop()

