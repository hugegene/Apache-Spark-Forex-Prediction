package example

import org.apache.log4j.Logger;
import org.apache.log4j.Level;
import org.apache.log4j.BasicConfigurator;
import java.net._
import java.io._
import scala.io._
import org.json.JSONObject;
import java.util.{ Calendar, Date }
import java.text.SimpleDateFormat
import scala.util.Random

object Hello extends Greeting with App {
  //  println(greeting)

  val fromCurrency = "EUR"
  val toCurrency = "USD"

  BasicConfigurator.configure()
  val logger = Logger.getRootLogger()
  logger.setLevel(Level.WARN)

  val api = new Api()

  while (true) {
    val dateFormatter = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss")
    val key = dateFormatter.format(new Date())

    // mock up data
//    val xrate = Random.nextFloat() / 1000
//    val convertedResult = f"""{\"datetime\":\"$key\",\"xrate\":${1.12460640 + xrate}%.8f}"""
//    println(convertedResult)
//    writeToSocket(convertedResult)

    // Fetch real-time data from api
      val result = api.getData(fromCurrency, toCurrency)
      if (!result.isEmpty()) {
        val json = (new JSONObject(result)).getJSONObject("Realtime Currency Exchange Rate");
        val convertedResult = f"""{\"datetime\":\"${json.get("6. Last Refreshed")}\",\"xrate\":${json.get("5. Exchange Rate")}}"""
        println(convertedResult)
        writeToSocket(convertedResult)
      }
    Thread.sleep(60000)
  }

  private def writeToSocket(result: String) {
    try {
      val s = new Socket(InetAddress.getByName("localhost"), 9999)
      lazy val in = new BufferedSource(s.getInputStream()).getLines()
      val out = new PrintStream(s.getOutputStream())

      out.println(result)
      out.flush()
      //      println("Received: " + in.next())
      s.close()
    } catch {
      case _: Throwable => println("Cannot send to localhost:9999")
    }
  }
}

trait Greeting {
  lazy val greeting: String = "Start to get exchange rate from API"
}
