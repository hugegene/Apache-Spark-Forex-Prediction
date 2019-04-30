package example
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{FloatType, DoubleType, StructType, StructField}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.ml.feature.{CountVectorizer, RegexTokenizer, StopWordsRemover}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.Pipeline
import java.util.Calendar
import java.text.SimpleDateFormat
import org.bson.Document
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.{Row, Dataset}
import scala.util.parsing.json.JSON
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import com.mongodb.spark._


object Price {

  val schema = StructType( StructField("k", FloatType, true) :: StructField("v", FloatType, false) :: Nil)
  
  import org.apache.spark.sql.SparkSession
  val spark = SparkSession.builder()
    .master("local")
    .appName("Time Series Modeling")
    .config("spark.mongodb.input.uri", "mongodb://localhost/Sentiments.sentiments")
    .config("spark.mongodb.output.uri", "mongodb://localhost/Sentiments.sentiments")
    .getOrCreate()
    
  val pricedf = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("C:\\Users\\eugene\\Desktop\\fx_intraday_1min_EUR_USD.csv")
  pricedf.printSchema()

  val w = Window.partitionBy().orderBy("timestamp")
  
//  Access MongoDB retrieve latest sentiment store it as  sentimentindex
  val df = MongoSpark.load(spark)
  val sentimentindex = df.orderBy(desc("date")).first.getDouble(2)
  
  val addt = pricedf.withColumn("t-1", lag("close", 1).over(w)).na.drop()
    .withColumn("t-2", lag("close", 2).over(w)).na.drop()
    .withColumn("sentiment", lit(sentimentindex))
    .withColumn("label", when(lag("close", -1, 0).over(w) > pricedf("close"), 1).otherwise(0)).na.drop()
  
  addt.show(addt.count.toInt, false)
  addt.printSchema()
  
  val assembler = new VectorAssembler()
  .setInputCols(Array("close", "t-1", "t-2", "sentiment"))
  .setOutputCol("features")
  
  val vectorised = assembler.transform(addt)
  
  val lr = new LogisticRegression()
  .setMaxIter(10)
  .setRegParam(0.2)
  .setElasticNetParam(0.0)
  
  val lrModel = lr.fit(vectorised)
  
  lrModel.write.overwrite().save("myModelPath")
  val sameModel = LogisticRegressionModel.load("myModelPath")

  val results = sameModel.transform(vectorised)
  
  val addresults = results.withColumn("changes", (col("close")-col("t-1"))/col("t-1"))
    .withColumn("signal", when(col("prediction") === 1, 1).otherwise(-1))
    .withColumn("returns", lag("signal", 1, 0).over(w)*col("changes")+1)
    .withColumn("cumulativeReturns", exp(sum(log(col("returns"))).over(w)))

  addresults.show(20, false)
  addresults.printSchema()
  
}