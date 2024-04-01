package wc

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.log4j.LogManager
import org.apache.log4j.Level

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.feature.{StringIndexer, IndexToString, VectorIndexer,VectorAssembler}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

import java.io.PrintWriter

object DTree {
  
  def main(args: Array[String]) {
    val logger: org.apache.log4j.Logger = LogManager.getRootLogger
    if (args.length != 2) {
      logger.error("Usage:\nwc.DTree <input dir> <output dir>")
      System.exit(1)
    }
    val conf = new SparkConf().setAppName("DTree")
    val sc = new SparkContext(conf)

    // Change this when running on AWS Cluster
    val spark = SparkSession.builder().appName("DTree").getOrCreate()

		val df = spark.read
      .format("csv")
      .option("inferSchema","true")
      .option("header", "true")
      .option("NaN", "None")
      .load(args(0))



    val featureCols = df.columns.filter(_ != "target")

    // Vector Assembler to combine feature columns into a single vector column
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")
      .setHandleInvalid("skip")

    // Decision Tree Classifier
    val dt = new DecisionTreeClassifier()
      .setLabelCol("target")
      .setFeaturesCol("features")

    // Pipeline
    val pipeline = new Pipeline()
      .setStages(Array(assembler, dt))

    // Split data into training and test sets
    val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3))

    // Fit the pipeline to training documents
    val model = pipeline.fit(trainingData)

    // Make predictions on test documents
    val predictions = model.transform(testData)

    // Evaluate the model
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("target")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")
      
    val accuracy = evaluator.evaluate(predictions)
    println(s"Accuracy: $accuracy")

    val treeModel = model.stages(1).asInstanceOf[DecisionTreeClassificationModel]
    println("Learned classification tree model:\n" + treeModel.toDebugString)
    val treeModelString = "Learned classification tree model:\n" + treeModel.toDebugString

    // Concatenate accuracy and tree model string
    val outputString = s"Accuracy: $accuracy\n$treeModelString"

    // Write output to text file
    val outputFile = new PrintWriter(args(1)) // Use the output file path from args(1)
    outputFile.println(outputString)
    outputFile.close()

  }
}