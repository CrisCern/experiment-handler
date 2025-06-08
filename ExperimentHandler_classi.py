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

class ExperimentHandler:
    """
    Gestisce l'intera pipeline Spark per l'analisi di classificazione.
    Inizializza una sessione Spark e definisce gli slot condivisi per i dati.
    """
    def __init__(self, app_name="FoodHealthClassifier"):
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .master("local[*]") \
            .config("spark.executor.memory", "8g") \
            .config("spark.driver.memory", "8g") \
            .config("spark.storage.memoryFraction", "0.3") \
            .config("spark.executor.heartbeatInterval", "60s") \
            .config("spark.network.timeout", "300s") \
            .config("spark.driver.maxResultSize", "2g") \
            .getOrCreate()

        self.df = None
        self.numeric_features = []      # Verranno definite in clean_data()
        self.categorical_features = []  # (se usate)

    def load_data(self, path: str) -> None:
        """
        Carica il dataset da file TSV e aggiorna lo slot self.df.

        Parametri:
        - path (str): percorso del file da caricare.
        """
        self._read_dataset(path)
        self._print_schema_info()

    def _read_dataset(self, path: str) -> None:
        """Legge il file TSV e assegna il DataFrame a self.df."""
        self.df = self.spark.read.csv(
            path,
            header=True,
            sep="\t",
            inferSchema=True,
            multiLine=True
        )

    def _print_schema_info(self) -> None:
        """Stampa il numero di righe, colonne e lo schema del dataset."""
        print(f"\n Dataset caricato. Righe: {self.df.count()}, Colonne: {len(self.df.columns)}")
        self.df.printSchema()

    def plot_missing_heatmap(self) -> None:
        """
        Converte il DataFrame Spark in Pandas e genera una heatmap dei valori nulli.
        Salva il grafico in 'grafici/heatmap_nulli.png'.
        """
        pandas_df = self.df.toPandas()
        self._plot_null_heatmap(pandas_df)

    def _plot_null_heatmap(self, pandas_df: pd.DataFrame) -> None:
        """Genera e salva la heatmap dei valori nulli a partire da un DataFrame Pandas."""
        plt.figure(figsize=(12, 6))
        sns.heatmap(pandas_df.isnull(), cbar=False)
        plt.title("Heatmap dei valori nulli")
        plt.tight_layout()
        plt.savefig("grafici/heatmap_nulli.png")
        plt.close()

    def clean_data(self) -> None:
        """
        Esegue la pulizia del dataset:
        - Rimuove righe con nutrition_grade_fr mancante
        - Seleziona feature numeriche rilevanti
        - Rimuove colonne con troppi nulli
        - Pulisce colonne testuali
        """
        self._drop_missing_grades()
        self._select_numeric_features()
        self._filter_nulls_and_cast()
        self._clean_categorical_columns()

        print(
            f"\n Pulizia completata. Righe finali: {self.df.count()}, "
            f"Feature numeriche selezionate: {len(self.numeric_features)}"
        )

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

    def summary(self):
        """
        Stampa un riepilogo della struttura del dataset pulito.
        """
        print("\n Riepilogo del dataset:")
        print(f"  Righe totali: {self.df.count()}")
        print(f"  Feature numeriche usate: {len(self.numeric_features)}")
        print(f"  Feature categoriche usate: {len(self.categorical_features)}")

    def print_numeric_features(self) -> None:
        """
        Stampa l'elenco delle feature numeriche selezionate per l'addestramento del modello.
        """
        print("\n Feature numeriche utilizzate per il modello:")
        for feature in self.numeric_features:
            print(f" - {feature}")

    def label_distribution(self) -> None:
        """
        Mostra la distribuzione della variabile target 'label' nel dataset corrente (self.df).
        """
        print("\n Distribuzione della variabile target:")
        distribuzione = self.df.groupBy("label").count().collect()
        for row in distribuzione:
            classe = " Sano (0)" if row["label"] == 0 else " Non sano (1)"
            print(f"{classe}: {row['count']}")

    def define_target(self) -> None:
        """
        Crea una colonna binaria 'label' nel dataset corrente (self.df),
        basata sulla colonna 'nutrition_grade_fr'.
        Etichetta 0 = sano ('a' o 'b'), 1 = non sano ('d' o 'e').
        Le righe prive di label vengono rimosse.
        """
        self._assign_label_column()
        self._log_label_stats()

    def _assign_label_column(self) -> None:
        self._rows_before_labeling = self.df.count()

        self.df = self.df.withColumn(
            "label",
            when(col("nutrition_grade_fr").isin("a", "b"), 0)
            .when(col("nutrition_grade_fr").isin("d", "e"), 1)
        ).dropna(subset=["label"])

    def _log_label_stats(self) -> None:
        after = self.df.count()
        count_0 = self.df.filter(col("label") == 0).count()
        count_1 = self.df.filter(col("label") == 1).count()

        print(f"\n Target definito:")
        print(f"   -  Sano (0): {count_0}")
        print(f"   -  Non sano (1): {count_1}")
        print(f"   -  Righe scartate: {self._rows_before_labeling - after}")

    def plot_label_distribution(self) -> None:
        """
        Genera un grafico a barre che mostra la distribuzione delle classi (label)
        nel dataset corrente (self.df) e lo salva su 'grafici/distribuzione_label.png'.
        """
        pandas_df = self.df.select("label").toPandas()
        self._plot_label_bar_chart(pandas_df)

    def _plot_label_bar_chart(self, pandas_df: pd.DataFrame) -> None:
        """Crea e salva il grafico della distribuzione delle classi."""
        counts = pandas_df["label"].value_counts().sort_index()

        plt.figure(figsize=(5, 4))
        counts.plot(kind="bar", color=["green", "red"])
        plt.title("Distribuzione delle classi")
        plt.xticks([0, 1], ["Sano (0)", "Non sano (1)"])
        plt.ylabel("Numero di campioni")
        plt.tight_layout()
        plt.savefig("grafici/distribuzione_label.png")
        plt.close()

    def preprocess_data(self) -> None:
        """
        Applica il preprocessing al dataset:
        - Imputazione dei valori nulli (mediana)
        - Scalatura MinMax e creazione della colonna 'features'
        - Esplosione del vettore in colonne leggibili per esportazione/debug
        Il risultato viene salvato in self.df.
        """
        print("\n Preprocessing intelligente...")
        start = time.time()

        self._impute_missing_values()
        self._scale_and_assemble_features()
        self._explode_features_for_debug()

        elapsed = time.time() - start
        print(f" Preprocessing completato in {elapsed:.2f} secondi.")
        print(f" Numero di feature usate nel modello: {len(self.numeric_features)}")

    def _impute_missing_values(self) -> None:
        imputed_cols = [f"{c}_imputed" for c in self.numeric_features]
        imputer = Imputer(inputCols=self.numeric_features, outputCols=imputed_cols).setStrategy("median")
        self.df = imputer.fit(self.df).transform(self.df)
        self.numeric_features = imputed_cols  # aggiorno direttamente per step successivi

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

    def export_final_dataset(self, path: str = "final_dataset.csv") -> None:
        """
        Esporta il dataset contenuto in self.df in formato CSV.
        Esclude colonne tecniche ('features', 'unscaled_features') per renderlo leggibile.

        Parametri:
        - path (str): percorso di salvataggio del file CSV.
        """
        exclude_cols = {"features", "unscaled_features"}
        readable_cols = [c for c in self.df.columns if c not in exclude_cols]

        pd.options.display.float_format = '{:.5f}'.format
        self.df.select(readable_cols).toPandas().to_csv(path, index=False)

        print(f"\n Dataset esportato con successo in: {path}")

    def preview_dataset(self, limit: int = 1, transpose: bool = True) -> None:
        """
        Mostra un'anteprima leggibile del dataset finale contenuto in self.df.
        Se transpose=True, mostra le feature in verticale (utile con molte colonne).

        Parametri:
        - limit (int): numero di righe da mostrare.
        - transpose (bool): se True stampa le feature in verticale.
        """
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_rows', None)

        exclude_cols = {"features", "unscaled_features"}
        readable_cols = [c for c in self.df.columns if c not in exclude_cols]

        pdf = self.df.select(readable_cols).limit(limit).toPandas()

        print("\n Anteprima del dataset finale:\n")
        if transpose:
            print(pdf.T.to_string(header=False))  # stampa verticale
        else:
            print(pdf.to_string(index=False))  # stampa orizzontale

    def train_model(self):
        """
        Addestra un modello Random Forest usando il wrapper esterno.
        Salva il modello in self.model e restituisce anche il dataset di test.

        Returns:
        - model: modello addestrato
        - test_data: sottoinsieme per valutazione
        """
        print("\n Addestramento modello Random Forest...")
        start_time = time.time()

        train_data, test_data = self._split_data()
        rf = RandomForestWrapper()
        model = rf.train(train_data)

        self.model = model
        duration = time.time() - start_time

        print(f" Modello addestrato con {model.getNumTrees} alberi, maxDepth = {model.getMaxDepth()}")
        print(f" Training set: {train_data.count()} righe")
        print(f" Tempo di addestramento: {duration:.2f} secondi")

        return model, test_data

    def _split_data(self):
        """Divide il dataset corrente in train/test (80/20)."""
        return self.df.randomSplit([0.8, 0.2], seed=42)

    def plot_feature_importance(self, model, top_n: int = 15) -> None:
        """
        Genera un grafico dell'importanza delle feature (top N) e salva anche la lista completa su file.

        Parametri:
        - model: modello Spark MLlib già addestrato con featureImportances
        - top_n (int): numero di feature da visualizzare nel grafico
        """
        importances = model.featureImportances.toArray()
        features = self.numeric_features

        sorted_idx = np.argsort(importances)[::-1]
        top_idx = sorted_idx[:top_n]

        self._plot_feature_bar(importances, top_idx, features)
        self._export_feature_rank(importances, features)

    def _plot_feature_bar(self, importances, top_idx, features) -> None:
        """Crea e salva il grafico delle feature più importanti."""
        plt.figure(figsize=(10, 5))
        plt.title("Top Feature Importance")
        plt.bar(range(len(top_idx)), importances[top_idx])
        plt.xticks(range(len(top_idx)), [features[i] for i in top_idx], rotation=90)
        plt.tight_layout()
        os.makedirs("grafici", exist_ok=True)
        plt.savefig("grafici/feature_importance.png")
        plt.close()
        print("\n Grafico salvato in 'grafici/feature_importance.png'")


    def _export_feature_rank(self, importances, features) -> None:
        """Esporta la classifica delle feature in un file TXT e stampa le top 10."""
        all_features_sorted = sorted(zip(features, importances), key=lambda x: -x[1])
        with open("grafici/feature_importance.txt", "w") as f:
            for name, score in all_features_sorted:
                f.write(f"{name}: {score:.4f}\n")

        print("\n Top 10 Feature Importance (Random Forest):")
        for name, score in all_features_sorted[:10]:
            print(f" - {name}: {score:.4f}")

    def evaluate_model(self, model, test_data, modello_ottimizzato: bool = False) -> None:
        """
        Valuta le prestazioni di un modello classificatore sul test set specificato.
        Stampa e salva metriche: Accuracy, Precision, Recall, F1, AUC.

        Parametri:
        - model: modello già addestrato
        - test_data: DataFrame con dati di test
        - modello_ottimizzato (bool): se True, salva le metriche con etichetta 'ottimizzato'
        """
        start_time = time.time()

        predictions = model.transform(test_data)
        metrics = self._compute_classification_metrics(predictions)
        metrics["tempo"] = time.time() - start_time

        self._print_metrics(metrics)
        self._save_metrics_to_file(metrics, modello_ottimizzato)

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

        evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction",
                                                  metricName="areaUnderROC")
        auc = evaluator.evaluate(predictions)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc
        }

    def _print_metrics(self, metrics: dict) -> None:
        print("\n Valutazione del modello:")
        print(f" - Accuracy:  {metrics['accuracy']:.4f}")
        print(f" - Precision: {metrics['precision']:.4f}")
        print(f" - Recall:    {metrics['recall']:.4f}")
        print(f" - F1 Score:  {metrics['f1']:.4f}")
        print(f" - AUC:       {metrics['auc']:.4f}")
        print(f" - Tempo:     {metrics['tempo']:.2f} sec")

    def _save_metrics_to_file(self, metrics: dict, ottimizzato: bool) -> None:
        os.makedirs("grafici", exist_ok=True)

        nome_modello = "ottimizzato" if ottimizzato else "base"
        path = f"grafici/valutazione_modello_{nome_modello}.txt"

        with open(path, "w") as f:
            f.write(f" Valutazione del modello {nome_modello}\n")
            f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall:    {metrics['recall']:.4f}\n")
            f.write(f"F1 Score:  {metrics['f1']:.4f}\n")
            f.write(f"AUC:       {metrics['auc']:.4f}\n")
            f.write(f"Tempo:     {metrics['tempo']:.2f} sec\n")

        print(f"\n Risultati salvati in: {path}")


    def plot_confusion_matrix(self, predictions, title: str = "Confusion Matrix", show: bool = False) -> None:
        """
        Calcola e salva la confusion matrix a partire dalle predizioni fornite.

        Parametri:
        - predictions: DataFrame Spark contenente 'label' e 'prediction'
        - title (str): titolo per il grafico e nome del file
        - show (bool): se True mostra a schermo, altrimenti solo salva su file
        """
        y_true, y_pred = self._extract_labels_from_predictions(predictions)

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Sano (0)", "Non sano (1)"])

        plt.figure(figsize=(6, 4))
        disp.plot(cmap="Blues", values_format='d')
        plt.title(title)
        plt.tight_layout()

        path = f"grafici/{title.replace(' ', '_').lower()}.png"
        plt.savefig(path)
        print(f"\n Confusion matrix salvata in: {path}")

        if show:
            plt.show()
        else:
            plt.close()

    def _extract_labels_from_predictions(self, predictions):
        """Estrae le colonne 'label' e 'prediction' da un DataFrame Spark e le restituisce come liste."""
        y_true = [int(row["label"]) for row in predictions.select("label").collect()]
        y_pred = [int(row["prediction"]) for row in predictions.select("prediction").collect()]
        return y_true, y_pred


    def cross_validate_model(self):
        """
        Esegue la cross-validation sul dataset corrente (self.df) usando Random Forest.
        Restituisce il miglior modello e il dataset di test.
        """
        print("\n Ottimizzazione con Cross-Validation...")

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

        best_model = self._run_cross_validation(cv, train_data)


        self.cv_model = best_model
        self.test_data = test_data

        print(f" Miglior numero di alberi: {best_model.getNumTrees}")
        print(f" Migliore profondità massima: {best_model.getOrDefault('maxDepth')}")

        return best_model, test_data


    def _build_param_grid(self):
        """Costruisce la griglia di parametri per la Random Forest."""
        rf = RandomForestClassifier()
        return ParamGridBuilder() \
            .addGrid(rf.numTrees, [50, 100]) \
            .addGrid(rf.maxDepth, [5, 10]) \
            .build()

    def _run_cross_validation(self, cv, train_data):
        """Esegue la cross-validation e ritorna il miglior modello trovato."""
        start = time.time()
        cv_model = cv.fit(train_data)
        duration = time.time() - start
        print(f" Cross-validation completata in {duration:.2f} secondi.")
        return cv_model.bestModel

    def show_wrong_predictions(self, predictions: DataFrame, limit: int = 5) -> None:
        """
        Mostra esempi di predizioni errate, evidenziando le feature più rilevanti.

        Parametri:
        - predictions: DataFrame con colonne 'label', 'prediction' e 'features'
        - limit (int): numero massimo di righe da mostrare
        """
        # Rinomina le colonne dell'originale per evitare conflitti
        renamed_df = self.df
        for col_name in self.numeric_features:
            renamed_df = renamed_df.withColumnRenamed(col_name, f"{col_name}_orig")
        renamed_df = renamed_df.withColumnRenamed("label", "label_orig")

        # Join sui vettori delle features
        joined = predictions.join(renamed_df, on="features", how="inner")

        # Filtro degli errori
        wrong = joined.filter(col("label") != col("prediction"))

        # Feature da evidenziare negli errori
        top_features = [
            "saturated-fat_100g_imputed",
            "fat_100g_imputed",
            "sugars_100g_imputed",
            "energy_100g_imputed",
            "salt_100g_imputed"
        ]
        columns_to_show = ["label", "prediction"] + [f"{f}_orig" for f in top_features if f in self.numeric_features]

        print(f"\n Esempi di predizioni errate (max {limit}):")
        wrong.select(*columns_to_show).show(limit, truncate=False)

    def compare_models(self) -> None:
        """
        Confronta le prestazioni di Random Forest, Logistic Regression e GBT
        sul dataset corrente (self.df), salvando i risultati su file.
        """
        print("\n Confronto tra modelli (Random Forest, Logistic Regression, GBT):")

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
            print(f" {name:>20}: Accuracy = {acc:.4f} | AUC = {auc:.4f}")

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

        print(f"\n Confronto salvato in: {path}")

    def plot_roc_curves(self, models: dict, test_data, output_path="grafici/roc_curve.png") -> None:
        """
        Genera e salva la ROC curve per più modelli classificatori.

        Parametri:
        - models (dict): dizionario {nome_modello: oggetto_modello}
        - test_data: dataset di test su cui calcolare le curve
        - output_path (str): percorso di salvataggio del grafico ROC
        """
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

        print(f"\n ROC Curve salvata in: {output_path}")

    def _get_roc_data(self, model, test_data):
        """
        Calcola FPR, TPR e AUC per un modello usando sklearn.
        """
        predictions = model.transform(test_data)
        pred_df = predictions.select("label", "probability") \
            .withColumn("prob", vector_to_array("probability")[1]) \
            .toPandas()

        fpr, tpr, _ = roc_curve(pred_df["label"], pred_df["prob"])
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc

    def print_feature_importance(self, model, top_n: int = 20) -> None:
        """
        Stampa l'importanza delle feature per un modello Random Forest.

        Parametri:
        - model: modello addestrato con attributo 'featureImportances'
        - top_n (int): numero massimo di feature da visualizzare
        """
        importances = model.featureImportances
        features = self.numeric_features

        sorted_features = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)

        print(f"\n Top {top_n} Feature Importance (Random Forest):")
        for feature, importance in sorted_features[:top_n]:
            print(f" - {feature}: {importance:.4f}")

    def compare_rf_version(self,
                           base_cm: tuple,  # (TN, FP, FN, TP)
                           ottimizzato_cm: tuple):  # (TN, FP, FN, TP)
        """
        Confronta due confusion matrix (base vs ottimizzato) e salva il confronto in un file.

        Parametri:
        - base_cm: tuple (TN, FP, FN, TP) del modello base
        - ottimizzato_cm: tuple (TN, FP, FN, TP) del modello ottimizzato
        """
        output = self._format_confusion_comparison(base_cm, ottimizzato_cm)
        print("\n Confronto tra Modello Base e Ottimizzato:\n")
        print(output)

        self._save_comparison_to_file(output, "grafici/confronto_modelli.txt")

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
        print(f" Confronto salvato in: {path}")



class RandomForestWrapper:
    """
    Wrapper dedicato per la configurazione e addestramento del modello Random Forest.
    """
    def __init__(self, featuresCol="features", labelCol="label", numTrees=100, maxDepth=5, seed=42):
        self.classifier = RandomForestClassifier(
            featuresCol=featuresCol,
            labelCol=labelCol,
            numTrees=numTrees,
            maxDepth=maxDepth,
            seed=seed
        )

    def train(self, train_data: DataFrame):
        """
        Allena il classificatore sul training set e restituisce il modello addestrato.
        """
        return self.classifier.fit(train_data)

if __name__ == "__main__":
    from pyspark.sql.functions import col
    from pyspark.ml.classification import LogisticRegression, GBTClassifier
    classifier = ExperimentHandler()

    # 1. Caricamento e anteprima iniziale
    print("\n Caricamento dati...")
    classifier.load_data("/Users/Cristian/Desktop/DatasetFoodfacts.tsv")
    classifier.df.select(classifier.df.columns[:10]).show(2, truncate=False)
    classifier.plot_missing_heatmap()
    # 2. Pulizia dati
    print("\n Pulizia dati...")
    classifier.clean_data()
    classifier.summary()
    classifier.print_numeric_features()
    # 3. Definizione della label
    print("\n Definizione del target...")
    classifier.define_target()
    classifier.df.select("label", *classifier.df.columns[:5]).show(5, truncate=False)
    classifier.label_distribution()
    classifier.plot_label_distribution()

    # 4. Preprocessing
    print("\n Preprocessing...")
    classifier.preprocess_data()

    # 5. Esportazione e preview
    classifier.export_final_dataset(path="final_dataset.csv")
    classifier.preview_dataset(limit=1, transpose=True)

    # 6. Addestramento modello base
    print("\n Addestramento modello base...")
    model, test_data = classifier.train_model()
    classifier.plot_feature_importance(model, top_n=15)
    classifier.print_feature_importance(model)

    print("\n Valutazione modello base...")
    classifier.evaluate_model(model, test_data)

    print("\n Confusion Matrix (modello base):")
    predictions = model.transform(test_data)
    classifier.plot_confusion_matrix(predictions, title="Confusion Matrix - Modello Base")
    classifier.show_wrong_predictions(predictions, limit=5)

    # 7. Cross-validation e ottimizzazione
    print("\n Ottimizzazione con Cross-Validation...")
    best_model, test_data = classifier.cross_validate_model()
    classifier.evaluate_model(best_model, test_data, modello_ottimizzato=True)

    print("\n Confusion Matrix (modello ottimizzato):")
    predictions_best = best_model.transform(test_data)
    classifier.plot_confusion_matrix(predictions_best, title="Confusion Matrix - Modello Ottimizzato")

    classifier.compare_rf_version(
        base_cm=(15043, 804, 1069, 23665),
        ottimizzato_cm=(15615, 232, 552, 24182)
    )

    # 8. Confronto modelli classici
    print("\n Confronto tra modelli (Random Forest, Logistic Regression, GBT):")
    classifier.compare_models()

    # 9. ROC Curve dei modelli
    print("\n ROC Curve dei modelli:")
    train_roc, _ = classifier.df.randomSplit([0.8, 0.2], seed=42)
    models = {
        "Random Forest": model,
        "Logistic Regression": LogisticRegression(labelCol="label", featuresCol="features", maxIter=10).fit(
            train_roc),
        "GBT": GBTClassifier(labelCol="label", featuresCol="features", maxIter=10).fit(train_roc)
    }
    classifier.plot_roc_curves(models, test_data)

    # 10. Salvataggio modello ottimizzato
    print("\n Salvataggio del modello ottimizzato...")
    best_model.write().overwrite().save("modello_ottimizzato_rf")

    print("\n Processo completato con successo!")


