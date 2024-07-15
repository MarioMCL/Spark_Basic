import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

object TitanicML_RandomForest {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.appName("Titanic Random Forest").master("local[*]").getOrCreate()

    val df = spark.read.option("header", "true").csv("data/Titanic-Dataset.csv")

    val dfClean = df.withColumn("Age", col("Age").cast("double"))
      .withColumn("Survived", col("Survived").cast("int"))
      .withColumn("Pclass", col("Pclass").cast("int"))
      .withColumn("SibSp", col("SibSp").cast("int"))
      .withColumn("Parch", col("Parch").cast("int"))
      .withColumn("Fare", col("Fare").cast("double"))

    val dfNoNulls = dfClean.na.fill(Map("Age" -> dfClean.agg(avg("Age")).first().getDouble(0), "Embarked" -> "S"))

    val indexerSex = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex")
    val indexerEmbarked = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkedIndex")

    val dfIndexed = indexerSex.fit(dfNoNulls).transform(dfNoNulls)
    val dfFinal = indexerEmbarked.fit(dfIndexed).transform(dfIndexed)

    val assembler = new VectorAssembler().setInputCols(Array("Pclass", "SexIndex", "Age", "SibSp", "Parch", "Fare", "EmbarkedIndex")).setOutputCol("features")

    val dfFeatures = assembler.transform(dfFinal)

    val Array(trainingData, testData) = dfFeatures.randomSplit(Array(0.7, 0.3))

    val rf = new RandomForestClassifier().setLabelCol("Survived").setFeaturesCol("features").setNumTrees(100)

    val rfModel = rf.fit(trainingData)

    val rfPredictions = rfModel.transform(testData)

    val evaluator = new BinaryClassificationEvaluator().setLabelCol("Survived").setMetricName("areaUnderROC")

    val rfAccuracy = evaluator.evaluate(rfPredictions)
    println(s"Área bajo la curva ROC para Random Forest: $rfAccuracy")

    rfPredictions.select("Survived", "prediction", "probability").show(10)

    spark.stop()
  }
}
