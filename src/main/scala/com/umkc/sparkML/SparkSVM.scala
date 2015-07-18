package com.umkc.sparkML

import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by Mayanka on 14-Jul-15.
 */
object SparkSVM {
  def main(args: Array[String]) {
    System.setProperty("hadoop.home.dir","C:\\Program Files (x86)\\winutils\\bin")
    val sparkConf =new SparkConf().setMaster("local[*]").setAppName("SparkSVM").set("spark.driver.memory", "4g")
    val sc = new SparkContext(sparkConf)
    // Load training data in LIBSVM format.
    val data = MLUtils.loadLibSVMFile(sc, "sample_libsvm_data.txt")

    // Split data into training (60%) and test (40%).
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    // Run training algorithm to build the model
    val numIterations = 100
    val model = SVMWithSGD.train(training, numIterations)

    // Clear the default threshold.
    model.clearThreshold()

    // Compute raw scores on the test set.
    val scoreAndLabels = test.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

    // Get evaluation metrics.
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()

    println("Area under ROC = " + auROC)

  }

}
