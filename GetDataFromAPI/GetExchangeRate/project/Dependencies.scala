import sbt._

object Dependencies {
  lazy val httpComponents = "org.apache.httpcomponents" % "fluent-hc" % "4.5.5"
  lazy val hadoopClient = "org.apache.hadoop" % "hadoop-client" % "2.7.0"
  lazy val json = "org.json" % "json" % "20180813"
}
