import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object TitanicAnalysis {
  def main(args: Array[String]): Unit = {
    // Crear la sesión de Spark
    val spark = SparkSession.builder
      .appName("Titanic Analysis")
      .master("local[*]")
      .getOrCreate()

    // Leer el archivo CSV
    val df = spark.read.option("header", "true").csv("data/Titanic-Dataset.csv")

    // Mostrar el esquema y las primeras filas
    df.printSchema()
    df.show()

    // Convertir las columnas necesarias a tipos apropiados
    val dfClean = df.withColumn("Age", col("Age").cast("double"))
      .withColumn("Survived", col("Survived").cast("int"))
      .withColumn("Pclass", col("Pclass").cast("int"))
      .withColumn("SibSp", col("SibSp").cast("int"))
      .withColumn("Parch", col("Parch").cast("int"))

    // Eliminar filas con valores nulos en las columnas seleccionadas
    val dfNoNulls = dfClean.na.drop(Seq("Age", "Survived", "Pclass", "SibSp", "Parch"))

    // Tasa de supervivencia global
    val survivalRate = dfNoNulls.groupBy().agg(avg("Survived")).first().getDouble(0)
    println(s"Tasa de supervivencia global: ${survivalRate * 100}%")

    // Tasa de supervivencia por clase de pasajero
    val survivalByClass = dfNoNulls.groupBy("Pclass").agg(avg("Survived"))
    survivalByClass.show()

    // Tasa de supervivencia por género
    val survivalByGender = dfNoNulls.groupBy("Sex").agg(avg("Survived"))
    survivalByGender.show()

    // Guardar los resultados (opcional)
    // survivalByClass.write.option("header", "true").csv("path/to/output/survival_by_class.csv")
    // survivalByGender.write.option("header", "true").csv("path/to/output/survival_by_gender.csv")

    // Detener la sesión de Spark
    spark.stop()
  }
}
