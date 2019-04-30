# GetExchangeRate
1. Clone this project
2. Use "sbt", then type "eclipse", ensure all the packages are imported and runnable in Eclipse (can use "refresh" by right click in Eclipse to refresh the project after rebuild)
3. Use "sbt clean assembly" under this project, make sure the jar file "./target/scala-2.11/GetExchangeRate-assembly-0.1.0-SNAPSHOT.jar" is generated 
4. Download scala from "https://downloads.lightbend.com/scala/2.11.12/scala-2.11.12.rpm", into "Download" directory, then go to "Download" directory and install scala by "sudo yum install scala-2.11.12.rpm"
5. Run the Xrate Streaming Project (See README in that project)
6. Run the script under this project, use "sh run\_getxrate\_scala.sh" 
7. Run the flume under "flume" directory, use "sh run\_flume.sh" 
