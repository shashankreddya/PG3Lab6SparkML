package com.umkc.sparkML

import com.umkc.sparkML.ModelEvaluation._
import com.umkc.sparkML.NLPUtils._
import com.umkc.sparkML.Utils._
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.{SparkConf, SparkContext}
/**
 * Created by Mayanka on 14-Jul-15.
 */
object SparkDecisionTree {
  def main(args: Array[String]) {
    System.setProperty("hadoop.home.dir","F:\\winutils")
    val sparkConf =new SparkConf().setMaster("local[*]").setAppName("SparkDecisionTree").set("spark.driver.memory", "4g")
    val sc = new SparkContext(sparkConf)
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
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxDepth = 5
    val maxBins = 32
    val model = DecisionTree.trainClassifier(X_train, numClasses, categoricalFeaturesInfo,
      impurity, maxDepth, maxBins)
    val predictionAndLabel = X_test.map(x => (model.predict(x.features), x.label))
    val accuracy = 1.0 *  predictionAndLabel.filter(x => x._1 == x._2).count() / X_test.count()

    println("*************Accuracy Report:***********************")
    println(accuracy)
    evaluateModel(predictionAndLabel,"Decision Tree Results")

  }

}
