package com.umkc.sparkML

import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.rdd.RDD

object Vectors {
  def evaluateModel(predictionAndLabels: RDD[(Double, Double)], msg: String) = {
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val cfMatrix = metrics.confusionMatrix
    println(msg)
    println(" |=================== Confusion matrix ==========================")
    println(cfMatrix)


  }
}
