package com.umkc.sparkML

import com.umkc.sparkML.ModelEvaluation._
import com.umkc.sparkML.NLPUtils._
import com.umkc.sparkML.Utils._
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by Mayanka on 14-Jul-15.
 */
object SparkW2V {

  def main(args: Array[String]) {
    System.setProperty("hadoop.home.dir","F:\\winutils")
    val sparkConf =new SparkConf().setMaster("local[*]").setAppName("SparkNaiveBayes").set("spark.driver.memory", "6g")
    val sc = new SparkContext(sparkConf)
    //Stopwords are broadcast to all RDD's across the cluster
    val stopWords = sc.broadcast(loadStopWords("/stopwords.txt")).value
    // map containing labels to numeric values for labeled Naive Bayes. "alt.atheism" -> 4
    val labelToNumeric = createLabelMap("data/training/")

    // tokenize, stem,
    val training = sc.wholeTextFiles("data/training/*")
      .map(rawText => createLabeledDocument(rawText, labelToNumeric, stopWords))
    val test = sc.wholeTextFiles("data/test/*")
      .map(rawText => createLabeledDocument(rawText,labelToNumeric, stopWords))

    //create features
    val X_train = tfidfTransformer(training)
    val X_test = tfidfTransformer(test)






    //Train / Predict
    val model = NaiveBayes.train(X_train,lambda = 1.0)
    val predictionAndLabel = X_test.map(x => (model.predict(x.features), x.label))
    val accuracy = 1.0 *  predictionAndLabel.filter(x => x._1 == x._2).count() / X_test.count()

    println("*************Accuracy Report:***********************")
    println(accuracy)
    evaluateModel(predictionAndLabel,"Naive Bayes Results")

  }

}
