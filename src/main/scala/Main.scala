import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql.functions.regexp_replace
import org.apache.spark.ml.feature.NGram
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression

object Main {


  def main(args: Array[String]) {

    // Set this to where Hadoop is.
    System.setProperty("hadoop.home.dir", "H:\\old downloads\\")

    // Apparently SparkSession is the way to go. It encapsulates
    // SparkConf, SparkContext and SQLContext, so this covers everything.
    val spark = org.apache.spark.sql.SparkSession.builder
      .master("local")
      .appName("Toxic Comment Challenge")
      .getOrCreate

    // "id","comment_text","toxic","severe_toxic","obscene","threat","insult","identity_hate"
    // this is the training data.
    val toxic_comment_data = spark.read
      .format("csv")
      .option("delimiter", "\t")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("H:\\old downloads\\train.csv\\train2.txt")

    // This is the test data.
    val toxic_test_data = spark.read
      .format("csv")
      .option("delimiter", "\t")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("H:\\old downloads\\test.csv\\test2.txt")

    // We use the na.drop() methods to ensure that there are no empty lines in the data.
    val toxic_comment_data_drop = toxic_comment_data.na.drop()
    val toxic_test_data_drop = toxic_test_data.na.drop()

    // And then, we remove all punctuation from the files.
    val punctRemoved = toxic_comment_data_drop.
      withColumn("no_punct",
        regexp_replace(toxic_comment_data_drop("comment_text"),"[\\p{Punct}]","" ))

    val punctRemovedTest = toxic_test_data_drop.
      withColumn("no_punct",
        regexp_replace(toxic_test_data_drop("comment_text"),"[\\p{Punct}]","" ))

    // We make a var to be added with the test results of each test.
    var test_results = punctRemovedTest.select("id")

    // Make a list of each classification
    val classifications = List("toxic","severe_toxic","obscene","threat","insult","identity_hate")
    // And then perform classification based on each label.
    classifications.foreach { label =>
      // We're going to tokenize our file, which will separate a string into words.
      val tokenizer = new Tokenizer()
        .setInputCol("no_punct")
        .setOutputCol("raw_tokens")

      // Since stop words add no value to most text, we're going to remove them.
      val remover = new StopWordsRemover()
        .setInputCol("raw_tokens")
        .setOutputCol("no_stops")

      // N-Grams are a way to obtain the context around words. The tokens {"how","are","you"} would have the 2-Grams of
      //   {"how are","are you"}. Now we can get a better understanding of how the tokens are used in a sentence.
      val ngram = new NGram()
        .setN(2)
        .setInputCol("no_stops")
        .setOutputCol("ngrams")

      // Now, we're going to calculate the Term Frequency of every comment. This will give us a way to compare
      // each of the comments across the training and test sets.
      val hashingTF = new HashingTF()
        .setInputCol("ngrams")
        .setOutputCol("rawFeatures")

      // And then, we're going to use Inverse Document Frequency to get a better weight of each term
      // with respect to the entire corpus.
      val idf = new IDF()
        .setInputCol("rawFeatures")
        .setOutputCol("features")

      // We will be using a Logistic Regression model, since we are solving a binary classification problem
      // at the core. We will train 6 of these models, one for each comment classification, and then use the model
      // to predict the probability of each classification.
      val lr = new LogisticRegression()
        .setMaxIter(10000)
        .setRegParam(0.001)
        .setElasticNetParam(0.999)
        .setLabelCol(label)
        .setPredictionCol(label + "_pred")
        .setProbabilityCol(label + "_prob")

      // A pipeline allows us to define a pre-processing set of commands.
      // Our comments will be tokenized, removed of stop words, composed into 2-Grams, and then weighted with
      // TF and IDF, and finally put into the linear regression model.
      val pipeline = new Pipeline()
        .setStages(Array(tokenizer, remover, ngram, hashingTF, idf, lr))

      // Now we build our model with our training set.
      val model = pipeline.fit(punctRemoved)

      // And make predictions with the test set. We select two columns, id and the probability.
      val results = model.transform(punctRemovedTest)
        .select("id", label+ "_prob")

      // We go ahead rename the probability to the label name as per submission guidelines.
      val transform_results = results.withColumn(label, results(label + "_prob")).drop(results(label + "_prob"))

      transform_results.show(10)
      test_results = test_results.join(transform_results, Seq("id"))
      test_results.show(10)


    }
    /*
    This is an example of the RDD that is output.
    This contains the id and the probability that a comment belongs to a particular label.

  +----------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
  |              id|               toxic|        severe_toxic|             obscene|              threat|              insult|       identity_hate|
  +----------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+
  |000968ce11f5ee34|[0.97186210891170...|[0.99318172846923...|[0.98264537023420...|[0.99784826993255...|[0.98291695945392...|[0.99325625227037...|
  |00491682330fdd1d|[0.99990938085939...|[0.99334173557477...|[0.99932729676847...|[0.99784826993255...|[0.99938144666155...|[0.99393633083135...|
  |008eb47c4684d190|[0.54827169318710...|[0.97263280899933...|[0.95613737012007...|[0.99548257768974...|[0.96752071339646...|[0.99325625227037...|
  |00d251f47486b6d2|[0.91721829166062...|[0.99318172846923...|[0.95921366659689...|[0.99784826993255...|[0.96058887441408...|[0.99325625227037...|
  |0114ae82c53101a9|[0.91225499783238...|[0.99318172846923...|[0.95921366659689...|[0.99784826993255...|[0.96058887441408...|[0.99325625227037...|
  |012c7429c5a34290|[0.95298606267204...|[0.99318172846923...|[0.97528858049012...|[0.99784826993255...|[0.97709525335499...|[0.99325625227037...|
  |015017ec394a264e|[0.91225499783238...|[0.99318172846923...|[0.95921366659689...|[0.99784826993255...|[0.96058887441408...|[0.99325625227037...|
  |01d94c94a86a4327|[0.98279856167350...|[0.99318172846923...|[0.99206329688031...|[0.99784826993255...|[0.99557215154021...|[0.99325625227037...|
  |020eb3a1af28453f|[0.93618813466413...|[0.99318172846923...|[0.97825161299510...|[0.99784826993255...|[0.97287324461995...|[0.99325625227037...|
  |0216909e11cfeac0|[0.99388881469940...|[0.99353107802657...|[0.99625881642836...|[0.99784826993255...|[0.99683321779663...|[0.99518231508045...|
  +----------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+

  */
    test_results.rdd.map(element => element.mkString(",")).saveAsTextFile("submission.csv")
  }

}

