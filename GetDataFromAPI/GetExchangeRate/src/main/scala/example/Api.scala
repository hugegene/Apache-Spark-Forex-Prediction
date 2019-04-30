package example

import org.apache.http.client.fluent.Request
import scala.util.Try
import scala.util.Success
import scala.util.Failure

class Api {
  private val baseUri = "https://www.alphavantage.co/query"
//  private val key = "YOGEYZ72NCX11UPI"
  private val key = "SW4MQ605LBYKUT4O"
  private val function = "CURRENCY_EXCHANGE_RATE"
  
  private def httpRequest(fromCurrency: String, toCurrency: String): Try[String] = Try(
    Request
      .Get(s"$baseUri?function=$function&from_currency=$fromCurrency&to_currency=$toCurrency&apikey=$key")
      .execute()
      .returnContent()
      .asString())
      
  private def request(fromCurrency: String, toCurrency: String): String = {
    val result = httpRequest(fromCurrency, toCurrency)
    result match {
      case Success(v) => v
      case Failure(f) => 
        println(s"Error attempting to call the API. [${f.getLocalizedMessage}]")
        ""
    }
  }
  
  def getData(fromCurrency: String = "EUR", toCurrency: String = "USD"): String = {
    val rawJson = request(fromCurrency, toCurrency)
    rawJson
  }      
  
}