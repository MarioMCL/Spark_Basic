import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

object TitanicML_LogisticRegresion {
  def main(args: Array[String]): Unit = {
    // Crear la sesión de Spark
    val spark = SparkSession.builder
      .appName("Titanic ML")
      .master("local[*]")
      .getOrCreate()

    // Leer el archivo CSV
    val df = spark.read.option("header", "true").csv("data/Titanic-Dataset.csv")

    // Preprocesar los datos
    val dfClean = df.withColumn("Age", col("Age").cast("double"))
      .withColumn("Survived", col("Survived").cast("int"))
      .withColumn("Pclass", col("Pclass").cast("int"))
      .withColumn("SibSp", col("SibSp").cast("int"))
      .withColumn("Parch", col("Parch").cast("int"))
      .withColumn("Fare", col("Fare").cast("double"))

    val dfNoNulls = dfClean.na.fill(Map(
      "Age" -> dfClean.agg(avg("Age")).first().getDouble(0),
      "Embarked" -> "S"
    ))

    // Convertir las columnas categóricas en índices
    val indexerSex = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex")
    val indexerEmbarked = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkedIndex")

    val dfIndexed = indexerSex.fit(dfNoNulls).transform(dfNoNulls)
    val dfFinal = indexerEmbarked.fit(dfIndexed).transform(dfIndexed)

    // Seleccionar las características para el modelo
    val assembler = new VectorAssembler()
      .setInputCols(Array("Pclass", "SexIndex", "Age", "SibSp", "Parch", "Fare", "EmbarkedIndex"))
      .setOutputCol("features")

    val dfFeatures = assembler.transform(dfFinal)

    // Balancear el dataset
    val fractions = dfFeatures.stat.sampleBy("Survived", Map(0 -> 1.0, 1 -> 1.0), 42)

    // Dividir el dataset en entrenamiento y prueba
    val Array(trainingData, testData) = fractions.randomSplit(Array(0.7, 0.3))

    // Crear el modelo de regresión logística
    val lr = new LogisticRegression()
      .setLabelCol("Survived")
      .setFeaturesCol("features")

    // Entrenar el modelo
    val model = lr.fit(trainingData)

    // Hacer predicciones
    val predictions = model.transform(testData)

    // Evaluar el modelo
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("Survived")
      .setMetricName("areaUnderROC")

    val accuracy = evaluator.evaluate(predictions)
    println(s"Área bajo la curva ROC: $accuracy")

    // Mostrar algunas predicciones
    predictions.select("Survived", "prediction", "probability").show(10)

    // Detener la sesión de Spark
    spark.stop()
  }
}
