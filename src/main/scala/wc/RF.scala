package wc

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.log4j.LogManager
import org.apache.log4j.Level

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassifier, RandomForestClassificationModel}
import org.apache.spark.ml.feature.{StringIndexer, IndexToString, VectorIndexer, VectorAssembler}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

import java.io.PrintWriter

object RF {
  
  def main(args: Array[String]) {
    val logger: org.apache.log4j.Logger = LogManager.getRootLogger
    if (args.length != 2) {
      logger.error("Usage:\nwc.RF <input dir> <output dir>")
      System.exit(1)
    }
    val conf = new SparkConf().setAppName("RF")
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

    // Random Forest Classifier
    val rf = new RandomForestClassifier()
      .setLabelCol("target")
      .setFeaturesCol("features")

    // Pipeline
    val pipeline = new Pipeline()
      .setStages(Array(assembler, rf))

    // Split data into training and test sets
    val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3))

    // Fit the pipeline to training documents
    val model = pipeline.fit(trainingData)

    val featureImportances = model.stages(1).asInstanceOf[RandomForestClassificationModel].featureImportances
    val featureImportancePairs = featureCols.zip(featureImportances.toArray)


    // Make predictions on test documents
    val predictions = model.transform(testData)

    // Evaluate the model
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("target")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")
      
    val accuracy = evaluator.evaluate(predictions)
    println(s"Accuracy: $accuracy")
    println(featureImportances)

    // Construct the output string
    val outputStringBuilder = new StringBuilder
    outputStringBuilder.append(s"Accuracy: $accuracy\nFeature Importance:\n")
    featureImportancePairs.foreach { case (featureName, importance) =>
        outputStringBuilder.append(s"$featureName: $importance\n")
    }

    // Write output to text file
    val outputFile = new PrintWriter(args(1)) // Use the output file path from args(1)
    outputFile.println(outputStringBuilder.toString())
    outputFile.close()
  
  }
}
