import os

import time
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from pyspark import SparkConf
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when, regexp_replace, lower
from pyspark.sql.types import DoubleType

from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array
from pyspark.ml.feature import Imputer, VectorAssembler, MinMaxScaler
from pyspark.ml.classification import (
    RandomForestClassifier,
    GBTClassifier,
    LogisticRegression,

)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,

)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

class DataLoader:
    """
    Classe responsabile del caricamento del dataset.
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.df: DataFrame = None

    def load(self, path: str) -> None:
        """
        Carica il dataset da file TSV e aggiorna lo slot self.df.
        """
        self.df = self.spark.read.csv(
            path,
            header=True,
            sep="\t",
            inferSchema=True,
            multiLine=True
        )

    def show_summary(self) -> None:
        """
        Mostra informazioni sullo schema del dataset.
        """
        print("Schema del dataset:")
        self.df.printSchema()
        print(f"Righe: {self.df.count()} - Colonne: {len(self.df.columns)}")

class Cleaner:
    """
    Classe responsabile della pulizia dei dati e dell'analisi dei valori mancanti.
    """

    def __init__(self, df: DataFrame):
        self.df = df
        self.numeric_features = []
        self.categorical_features = []
        self.cleaned = False
        self.cleaning_summary = {}

    def run(self) -> None:
        """
        Avvia la procedura di pulizia del dataset.
        """
        self._drop_missing_grades()
        self._select_numeric_features()
        self._filter_nulls_and_cast()
        self._clean_categorical_columns()

        self.cleaned = True
        self.cleaning_summary = {
            "final_row_count": self.df.count(),
            "numeric_features_count": len(self.numeric_features),
            "categorical_features_count": len(self.categorical_features)
        }

    def _drop_missing_grades(self) -> None:
        self.df = self.df.dropna(subset=["nutrition_grade_fr"])

    def _select_numeric_features(self) -> None:
        all_numeric = [c for c in self.df.columns if c.endswith("_100g")]
        features_to_exclude = [
            "nutrition_grade_fr", "nutrition_grade_uk",
            "nutrition-score-fr_100g", "nutrition-score-uk_100g"
        ]
        self.numeric_features = [c for c in all_numeric if c not in features_to_exclude]

    def _filter_nulls_and_cast(self) -> None:
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
        self.df = self.df.select(*self.numeric_features, "nutrition_grade_fr")

    def _clean_categorical_columns(self) -> None:
        categorical = ["packaging", "labels", "categories", "ingredients_text"]
        cleaned_cols = []
        for c in categorical:
            if c in self.df.columns:
                self.df = self.df.withColumn(
                    c, lower(regexp_replace(col(c), '[^a-zA-Z0-9 ]', ''))
                )
                cleaned_cols.append(c)

        self.categorical_features = cleaned_cols

    def show_summary(self) -> None:
        """
        Mostra un riepilogo della struttura del dataset pulito.
        """
        if not self.cleaned:
            print(" Il dataset non è stato ancora pulito.")
            return

        print("\n Riepilogo della pulizia:")
        print(f" - Righe finali: {self.cleaning_summary['final_row_count']}")
        print(f" - Feature numeriche selezionate: {self.cleaning_summary['numeric_features_count']}")
        print(f" - Feature categoriche pulite: {self.cleaning_summary['categorical_features_count']}")

    def show_numeric_features(self) -> None:
        """
        Mostra l'elenco delle feature numeriche selezionate per il modello.
        """
        if not self.numeric_features:
            print(" Nessuna feature numerica selezionata.")
            return

        print("\n Feature numeriche utilizzate:")
        for feature in self.numeric_features:
            print(f" - {feature}")

class TargetEngineer:
    """
    Si occupa della definizione della variabile target (label) per il task di classificazione.
    """

    def __init__(self, df: DataFrame):
        self.df = df
        self.label_stats = {}
        self._rows_before_labeling = None
        self._rows_after_labeling = None

    def run(self) -> None:
        """
        Avvia la creazione della colonna 'label' e aggiorna le statistiche.
        """
        self._assign_label_column()
        self._update_label_stats()

    def _assign_label_column(self) -> None:
        self._rows_before_labeling = self.df.count()
        self.df = self.df.withColumn(
            "label",
            when(col("nutrition_grade_fr").isin("a", "b"), 0)
            .when(col("nutrition_grade_fr").isin("d", "e"), 1)
        ).dropna(subset=["label"])
        self._rows_after_labeling = self.df.count()

    def _update_label_stats(self) -> None:
        count_0 = self.df.filter(col("label") == 0).count()
        count_1 = self.df.filter(col("label") == 1).count()
        discarded = self._rows_before_labeling - self._rows_after_labeling

        self.label_stats = {
            "sano": count_0,
            "non_sano": count_1,
            "discarded": discarded
        }

    def show_label_stats(self) -> None:
        """
        Mostra le statistiche sulla colonna 'label'.
        """
        if not self.label_stats:
            print(" Le statistiche del target non sono ancora disponibili.")
            return

        print("\n Statistiche sulla variabile target:")
        print(f" - Sano (0): {self.label_stats['sano']}")
        print(f" - Non sano (1): {self.label_stats['non_sano']}")
        print(f" - Righe scartate: {self.label_stats['discarded']}")

    def plot_label_distribution(self) -> None:
        """
        Genera un grafico a barre che mostra la distribuzione delle classi (label)
        nel dataset corrente (self.df) e lo salva su 'grafici/distribuzione_label.png'.
        """
        pandas_df = self.df.select("label").toPandas()
        self._plot_label_bar_chart(pandas_df)

    def _plot_label_bar_chart(self, pandas_df: pd.DataFrame) -> None:
        """
        Crea e salva il grafico della distribuzione delle classi.
        """
        counts = pandas_df["label"].value_counts().sort_index()

        plt.figure(figsize=(5, 4))
        counts.plot(kind="bar", color=["green", "red"])
        plt.title("Distribuzione delle classi")
        plt.xticks([0, 1], ["Sano (0)", "Non sano (1)"])
        plt.ylabel("Numero di campioni")
        plt.tight_layout()
        plt.savefig("grafici/distribuzione_label.png")
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
        Esegue tutte le fasi di preprocessing.
        """
        start = time.time()

        self._impute_missing_values()
        self._scale_and_assemble_features()
        self._explode_features_for_debug()

        elapsed = time.time() - start
        self.status["elapsed_time"] = round(elapsed, 2)
        self.status["num_features"] = len(self.numeric_features)

    def _impute_missing_values(self) -> None:
        imputed_cols = [f"{c}_imputed" for c in self.numeric_features]
        imputer = Imputer(inputCols=self.numeric_features, outputCols=imputed_cols).setStrategy("median")
        self.df = imputer.fit(self.df).transform(self.df)
        self.numeric_features = imputed_cols

    def _scale_and_assemble_features(self) -> None:
        assembler = VectorAssembler(inputCols=self.numeric_features, outputCol="unscaled_features")
        scaler = MinMaxScaler(inputCol="unscaled_features", outputCol="features")
        pipeline = Pipeline(stages=[assembler, scaler])
        self.df = pipeline.fit(self.df).transform(self.df)

    def _explode_features_for_debug(self) -> None:
        self.df = self.df.withColumn("features_array", vector_to_array("features"))
        for i, name in enumerate(self.numeric_features):
            self.df = self.df.withColumn(name, self.df["features_array"][i])

        selected_cols = self.numeric_features + ["features", "label"]
        self.df = self.df.select(*selected_cols)
        self.valid_features = self.numeric_features

    def show_status(self) -> None:
        """
        Visualizza le informazioni sul preprocessing eseguito.
        """
        if self.status["elapsed_time"] is not None:
            print(f"\n Preprocessing completato in {self.status['elapsed_time']} secondi.")
            print(f" Numero di feature utilizzate: {self.status['num_features']}")
        else:
            print(" Preprocessing non ancora eseguito.")

class DatasetExporter:
    """
    Classe responsabile dell'esportazione e dell'anteprima del dataset finale.
    """

    def __init__(self, df: DataFrame):
        self.df = df
        self.last_export_path = None
        self.last_preview = None

    def export_final_dataset(self, path: str = "final_dataset.csv") -> None:
        """
        Esporta il dataset in formato CSV, escludendo colonne tecniche.
        """
        exclude_cols = {"features", "unscaled_features"}
        readable_cols = [c for c in self.df.columns if c not in exclude_cols]

        pd.options.display.float_format = '{:.5f}'.format
        self.df.select(readable_cols).toPandas().to_csv(path, index=False)

        self.last_export_path = path

    def show_export_info(self) -> None:
        """
        Visualizza informazioni sull'ultima esportazione effettuata.
        """
        if self.last_export_path:
            print(f"\n Dataset esportato in: {self.last_export_path}")
        else:
            print(" Nessun dataset è stato ancora esportato.")

    def preview_dataset(self, limit: int = 1, transpose: bool = True) -> None:
        """
        Mostra un'anteprima leggibile del dataset finale.
        """
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_rows', None)

        exclude_cols = {"features", "unscaled_features"}
        readable_cols = [c for c in self.df.columns if c not in exclude_cols]

        pdf = self.df.select(readable_cols).limit(limit).toPandas()
        self.last_preview = pdf.T if transpose else pdf

        print("\n Anteprima del dataset finale:\n")
        print(self.last_preview.to_string(header=False if transpose else True, index=transpose))

class ModelTrainer:
    """
    Classe responsabile dell'addestramento del modello Random Forest.
    """

    def __init__(self, df: DataFrame):
        self.df = df
        self.model = None
        self.train_data = None
        self.test_data = None
        self.training_time = None

    def train_model(self) -> tuple:
        """
        Addestra un modello Random Forest e salva gli oggetti interni.
        """
        self.train_data, self.test_data = self._split_data()

        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            numTrees=100,
            maxDepth=5,
            seed=42
        )
        start_time = time.time()
        self.model = rf.fit(self.train_data)
        self.training_time = time.time() - start_time

        return self.model, self.test_data

    def _split_data(self) -> tuple:
        """
        Divide il dataset in training e test (80/20).
        """
        return self.df.randomSplit([0.8, 0.2], seed=42)

    def show_training_info(self) -> None:
        """
        Visualizza le informazioni sull'addestramento eseguito.
        """
        if self.model is None:
            print(" Nessun modello addestrato.")
            return

        print("\n Modello Random Forest addestrato.")
        print(f" - NumTrees: {self.model.getNumTrees}")
        print(f" - MaxDepth: {self.model.getMaxDepth()}")
        print(f" - Training set: {self.train_data.count()} righe")
        print(f" - Tempo di addestramento: {self.training_time:.2f} secondi")

class ModelVisualizer:
    def __init__(self, numeric_features):
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

        with open("grafici/feature_importance.txt", "w") as f:
            for name, score in all_sorted:
                f.write(f"{name}: {score:.4f}\n")

    def show_feature_importance(self) -> None:
        print(f"\n Grafico salvato in: {self.last_feature_path}")
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

    def show_confusion_matrix_info(self) -> None:
        if self.last_cm_path:
            print(f"\n Confusion matrix salvata in: {self.last_cm_path}")
        else:
            print("\n Confusion matrix non ancora generata.")

    def _extract_labels_from_predictions(self, predictions):
        y_true = [int(row["label"]) for row in predictions.select("label").collect()]
        y_pred = [int(row["prediction"]) for row in predictions.select("prediction").collect()]
        return y_true, y_pred

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

    def show_roc_info(self) -> None:
        if self.last_roc_path:
            print(f"\n ROC Curve salvata in: {self.last_roc_path}")
        else:
            print("\n ROC Curve non ancora generata.")

    def _get_roc_data(self, model, test_data):
        predictions = model.transform(test_data)
        pred_df = predictions.select("label", "probability") \
            .withColumn("prob", vector_to_array("probability")[1]) \
            .toPandas()

        fpr, tpr, _ = roc_curve(pred_df["label"], pred_df["prob"])
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc

class ModelEvaluator:
    def __init__(self, df, numeric_features):
        self.df = df
        self.numeric_features = numeric_features
        self.last_metrics = {}
        self.last_path = None
        self.predictions = None

    def evaluate_model(self, model, test_data, modello_ottimizzato: bool = False) -> None:
        """
        Valuta un modello sul test set. Salva le metriche nello stato della classe.
        """
        start_time = time.time()
        self.predictions = model.transform(test_data)
        self.last_metrics = self._compute_classification_metrics(self.predictions)
        self.last_metrics["tempo"] = time.time() - start_time

        self._save_metrics_to_file(self.last_metrics, modello_ottimizzato)

    def _compute_classification_metrics(self, predictions):
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

    def _save_metrics_to_file(self, metrics: dict, ottimizzato: bool) -> None:
        os.makedirs("grafici", exist_ok=True)
        nome_modello = "ottimizzato" if ottimizzato else "base"
        path = f"grafici/valutazione_modello_{nome_modello}.txt"
        self.last_path = path

        with open(path, "w") as f:
            f.write(f" Valutazione del modello {nome_modello}\n")
            f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall:    {metrics['recall']:.4f}\n")
            f.write(f"F1 Score:  {metrics['f1']:.4f}\n")
            f.write(f"AUC:       {metrics['auc']:.4f}\n")
            f.write(f"Tempo:     {metrics['tempo']:.2f} sec\n")

    def show_metrics(self) -> None:
        """Stampa le metriche salvate dallo step di valutazione."""
        m = self.last_metrics
        if not m:
            print("\n Nessuna metrica disponibile.")
            return

        print("\n Valutazione del modello:")
        print(f" - Accuracy:  {m['accuracy']:.4f}")
        print(f" - Precision: {m['precision']:.4f}")
        print(f" - Recall:    {m['recall']:.4f}")
        print(f" - F1 Score:  {m['f1']:.4f}")
        print(f" - AUC:       {m['auc']:.4f}")
        print(f" - Tempo:     {m['tempo']:.2f} sec")

    def get_metrics(self) -> dict:
        return self.last_metrics

    def get_path(self) -> str:
        return self.last_path

    def show_wrong_predictions(self, limit: int = 5) -> None:
        """Mostra esempi di predizioni errate dopo evaluate_model()."""
        if self.predictions is None:
            print(" Nessuna predizione trovata.")
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

class ModelOptimizer:
    def __init__(self, df):
        self.df = df
        self.cv_model = None
        self.test_data = None
        self.best_params = {}
        self.cv_duration = None

    def cross_validate_model(self):
        """
        Esegue la cross-validation e salva i risultati nello stato della classe.
        """
        train_data, test_data = self.df.randomSplit([0.8, 0.2], seed=42)
        param_grid = self._build_param_grid()

        evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
        rf = RandomForestClassifier(featuresCol="features", labelCol="label", seed=42)

        cv = CrossValidator(
            estimator=rf,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=3,
            parallelism=2
        )

        start = time.time()
        cv_model = cv.fit(train_data)
        self.cv_duration = time.time() - start

        self.cv_model = cv_model.bestModel
        self.test_data = test_data
        self.best_params = {
            "numTrees": self.cv_model.getNumTrees,
            "maxDepth": self.cv_model.getOrDefault("maxDepth")
        }

        return self.cv_model, self.test_data

    def _build_param_grid(self):
        """Costruisce la griglia di parametri per la Random Forest."""
        rf = RandomForestClassifier()
        return ParamGridBuilder() \
            .addGrid(rf.numTrees, [50, 100]) \
            .addGrid(rf.maxDepth, [5, 10]) \
            .build()

    def summarize_results(self):
        """Mostra i parametri migliori e la durata della cross-validation."""
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

    def get_params(self):
        return self.best_params

    def get_duration(self):
        return self.cv_duration

class ModelComparer:
    def __init__(self, df):
        self.df = df
        self.results = []
        self.rf_comparison_text = ""

    def compare_models(self) -> None:
        """
        Confronta Random Forest, Logistic Regression e GBT
        sul dataset corrente, salvando i risultati su file e stato interno.
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
            acc, auc = self._train_and_evaluate_model(algo, train_data, test_data, evaluator)
            results.append((name, acc, auc))

        self.results = results
        self._save_model_comparison(results)

    def _train_and_evaluate_model(self, model_algo, train_data, test_data, evaluator):
        """Allena e valuta un classificatore restituendo accuracy e AUC."""
        model = model_algo.fit(train_data)
        predictions = model.transform(test_data)

        auc = evaluator.evaluate(predictions)
        correct = predictions.filter(col("label") == col("prediction")).count()
        total = predictions.count()
        accuracy = correct / total if total > 0 else 0

        return accuracy, auc

    def _save_model_comparison(self, results):
        """Salva il confronto dei modelli in un file txt."""
        os.makedirs("grafici", exist_ok=True)
        path = "grafici/confronto_modelli_classici.txt"

        with open(path, "w") as f:
            f.write(" Confronto tra modelli (Random Forest, Logistic Regression, GBT)\n")
            f.write("-" * 60 + "\n")
            for name, acc, auc in results:
                f.write(f"{name:>20}: Accuracy = {acc:.4f} | AUC = {auc:.4f}\n")

    def compare_rf_version(self, base_cm: tuple, ottimizzato_cm: tuple):
        """
        Confronta due confusion matrix (base vs ottimizzato) e salva il confronto su file.
        """
        self.rf_comparison_text = self._format_confusion_comparison(base_cm, ottimizzato_cm)
        self._save_comparison_to_file(self.rf_comparison_text, "grafici/confronto_modelli.txt")

    def _format_confusion_comparison(self, base_cm, ottimizzato_cm) -> str:
        """Ritorna una stringa formattata con il confronto delle due confusion matrix."""
        tn_b, fp_b, fn_b, tp_b = base_cm
        tn_o, fp_o, fn_o, tp_o = ottimizzato_cm

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

    def _save_comparison_to_file(self, text: str, path: str) -> None:
        """Salva un blocco di testo in un file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text + "\n")

    def summarize_comparison(self):
        """Stampa il confronto classico dei modelli (solo se disponibile)."""
        if not self.results:
            print(" Nessun risultato disponibile.")
            return

        print("\n Confronto tra modelli classici:")
        for name, acc, auc in self.results:
            print(f" {name:>20}: Accuracy = {acc:.4f} | AUC = {auc:.4f}")

    def show_rf_comparison(self):
        """Mostra il confronto testuale tra Modello Base e Ottimizzato."""
        if self.rf_comparison_text:
            print("\n Confronto Modello Base vs Ottimizzato:")
            print(self.rf_comparison_text)
        else:
            print(" Nessun confronto disponibile.")

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
    cleaner = Cleaner(df)
    cleaner.run()
    df = cleaner.df
    numeric_features = cleaner.numeric_features

    # === 3. Definizione Target ===
    target_engineer = TargetEngineer(df)
    target_engineer.run()
    df = target_engineer.df

    # === 4. Preprocessing ===
    preprocessor = Preprocessor(df, numeric_features)
    preprocessor.run()
    df = preprocessor.df

    # === 5. Esportazione Dataset Finale ===
    exporter = DatasetExporter(df)
    exporter.preview_dataset()
    exporter.export_final_dataset()

    # === 6. Training del Modello Base ===
    trainer = ModelTrainer(df)
    model, test_data = trainer.train_model()

    # === 7. Valutazione del Modello Base ===
    evaluator = ModelEvaluator(df, numeric_features)
    evaluator.evaluate_model(model, test_data, modello_ottimizzato=False)
    base_predictions = model.transform(test_data)

    # === 8. Visualizzazione Confusion Matrix Base ===
    visualizer = ModelVisualizer(numeric_features)
    visualizer.plot_confusion_matrix(base_predictions, title="Confusion Matrix - Base")

    # === 9. Ottimizzazione con Cross-Validation ===
    optimizer = ModelOptimizer(df)
    best_model, test_data_opt = optimizer.cross_validate_model()
    df = optimizer.df  # In caso venga aggiornato internamente
    optimizer.summarize_results()

    # === 10. Valutazione Modello Ottimizzato ===
    evaluator.evaluate_model(best_model, test_data_opt, modello_ottimizzato=True)
    evaluator.show_metrics()
    opt_predictions = best_model.transform(test_data_opt)
    visualizer.plot_confusion_matrix(opt_predictions, title="Confusion Matrix - Ottimizzato")

    # === 11. Visualizzazione Feature Importance ===
    visualizer.plot_feature_importance(best_model)
    visualizer.show_feature_importance()

    # === 12. Confronto Confusion Matrix Base vs Ottimizzato ===
    cm_base = visualizer._extract_labels_from_predictions(base_predictions)
    cm_opt = visualizer._extract_labels_from_predictions(opt_predictions)
    cm_base_values = confusion_matrix(*cm_base).ravel()
    cm_opt_values = confusion_matrix(*cm_opt).ravel()

    comparer = ModelComparer(df)
    comparer.compare_rf_version(tuple(cm_base_values), tuple(cm_opt_values))

    # === 13. Visualizzazione Predizioni Errate ===
    evaluator.show_wrong_predictions(limit=5)

    # === 14. Confronto tra Modelli Classici ===
    comparer.compare_models()
    comparer.summarize_comparison()
    comparer.show_rf_comparison()

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

    print("\nEsecuzione pipeline completa...")

    # Avvia la pipeline con funzione globale
    run_experiment(spark, "/Users/Cristian/Desktop/DatasetFoodfacts.tsv")

    print("\nProcesso completato con successo!")

    spark.stop()
