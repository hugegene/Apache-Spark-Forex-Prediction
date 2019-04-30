package example

import com.mongodb.spark._

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{FloatType, DoubleType, StringType}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.ml.feature.{CountVectorizer, RegexTokenizer, StopWordsRemover}
import org.apache.spark.sql._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import java.util.Calendar
import java.text.SimpleDateFormat
import org.bson.Document
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.{Row, Dataset}
import scala.util.parsing.json.JSON


object Sentiment {
  
  import org.apache.spark.sql.SparkSession
  val spark = SparkSession.builder()
    .master("local")
    .appName("MongoSparkConnectorIntro")
    .config("spark.mongodb.input.uri", "mongodb://localhost/BloombergNews.news")
    .config("spark.mongodb.output.uri", "mongodb://localhost/BloombergNews.news")
    .getOrCreate()
  import spark.implicits._
  
  val df = MongoSpark.load(spark) 
  df.printSchema() 
  df.show()
  
  df.registerTempTable("data")
  val dataExplode = spark.sql("select to_date(date) as date, explode(news) as news from data")
//  dataExplode.show(dataExplode.count.toInt, false)
//  dataExplode.printSchema()

  val pricedf = spark.read.format("csv").option("header", "true").load("fx_daily_EUR_USD.csv")
  pricedf.printSchema()
  pricedf.show(pricedf.count.toInt, false)
 
  pricedf.registerTempTable("pricedata")
  val selectpricedf = spark.sql("select to_date(timestamp) as date, close as close from pricedata")
  selectpricedf.withColumn("close", selectpricedf("close") cast FloatType)
  
  val w = Window.partitionBy().orderBy("date")
  val pricetable = selectpricedf.withColumn("signal", when(lag("close", -1, 0).over(w) > selectpricedf("close"), 1).otherwise(0))
  pricetable.show(pricetable.count.toInt, false)
  
  pricetable.registerTempTable("pricetable")
  dataExplode.registerTempTable("newstable")
  
  val merged = spark.sql("""SELECT pricetable.date as date, newstable.news as news, pricetable.signal as label FROM  pricetable JOIN  newstable ON pricetable.date == newstable.date""")
  merged.printSchema()
  merged.show(merged.count.toInt, false)
  
  val mergedcolumn = merged.withColumn("rank", percent_rank().over(Window.partitionBy().orderBy("date")))
   
  val train = mergedcolumn.where("rank <= .8").drop("rank")
  train.show()
  
  val test = mergedcolumn.where("rank > .8").drop("rank")
  test.show(test.count.toInt)
  
  val stopwords: Array[String] = spark.sparkContext.textFile("stopwords.txt").flatMap(_.stripMargin.split("\\s+")).collect ++ Array("rt")
 
//  stopwords.foreach(println)
  
  val tokenizer = new RegexTokenizer()
  .setGaps(false)
  .setPattern("\\p{L}+")
  .setInputCol("news")
  .setOutputCol("words")
  
  val filterer = new StopWordsRemover()
  .setStopWords(stopwords)
  .setCaseSensitive(false)
  .setInputCol("words")
  .setOutputCol("filtered")
  
  val countVectorizer = new CountVectorizer()
  .setInputCol("filtered")
  .setOutputCol("features")
  
  val hashingTF = new HashingTF()
  .setInputCol("filtered").setOutputCol("rawFeatures").setNumFeatures(20)
  
  val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
  
  val lr = new LogisticRegression()
  .setMaxIter(10)
  .setRegParam(0.2)
  .setElasticNetParam(0.0)
  
  val pipeline = new Pipeline().setStages(Array(tokenizer, filterer, hashingTF, idf, lr))
  
  val lrModel = pipeline.fit(train)
  val results = lrModel.transform(test)
  results.show(20, false)
  results.printSchema()
  
  val toArr: Any => Array[Double] = _.asInstanceOf[DenseVector].toArray
  val toArrUdf = udf(toArr)
  val arraydata = results.withColumn("prob", toArrUdf(col("probability")))
  arraydata.printSchema()
  
  val newg = arraydata.select(col("date"), col("prob").getItem(1).as("pos"))
  
  val newf = newg.withColumn("math", abs(col("pos") -0.5))
  newf.show(false)
  
  val abc = newf.groupBy("date").max("math").withColumnRenamed("max(math)", "math") 
  abc.show(false)
  
  val join =  newf.join(abc, Seq("date", "math"), "inner").drop("math").groupBy("date").max("pos").withColumnRenamed("max(pos)", "sent")
  join.show(false)
  
  join.write.format("com.mongodb.spark.sql.DefaultSource").mode("overwrite").option("database","Sentiments").option("collection", "sentiments").save()

}