package example

object Hello extends Greeting with App {
  println(greeting)
  val sent = Sentiment
//  val pr = Price
}

trait Greeting {
  lazy val greeting: String = "hello"
}
