import sbt._

object Dependencies {
  lazy val scalaTest = "org.scalatest" %% "scalatest" % "3.0.5"
  lazy val sparkCore =  "org.apache.spark" %% "spark-core" % "2.2.0"
  lazy val sparkSQL = "org.apache.spark" %% "spark-sql" % "2.2.0"
  lazy val sparkMLlib ="org.apache.spark" %% "spark-mllib" % "2.2.0"
  lazy val mongoSparkConnector = "org.mongodb.spark" %% "mongo-spark-connector" % "2.2.6"
 

}
