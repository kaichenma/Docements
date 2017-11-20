package com.sparkProject


import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}



object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

   /** Load the dataset **/

   val df: DataFrame = spark
     .read
     .parquet("/Users/kaichenma/Documents/copies/TP_Spark/data")


   /** The process of taking text and breaking it into individual terms **/

    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol( "text" )
      .setOutputCol( "tokens" )

    // val tokenized = tokenizer.transform(df)
    // tokenized.show()


    /** Drop all the stop words from the input sequences **/

    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")

    // val removered = remover.transform(tokenized)
    // removered.show()


    /** Convert a collection of text documents to vectors of token counts **/

    val cvModel = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("vecterized")
      .setMinDF(10)

    // val cvModeled = cvModel.fit(removered).transform(removered)
    // cvModeled.show()


    /** fit on dataset and produce an IDFModel **/

    val idf = new IDF()
      .setInputCol("vecterized")
      .setOutputCol("tfidf")

    // var idfed = idf.fit(cvModeled).transform(cvModeled)
    // idfed.show()


    /** encode string column "country_2" and "currency_2" into a column of label indices **/

    val indexer_country = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      .setHandleInvalid("skip")

    // val indexed_country = indexer_country.fit(idfed).transform(idfed)
    // indexed_country.show()


    val indexer_currency = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")
      .setHandleInvalid("skip")

    // val indexed_currency = indexer_currency.fit(indexed_country).transform(indexed_country)
    // indexed_currency.show()


    /** Combine given list of columns into a single vector column **/

    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa","goal","country_indexed","currency_indexed"))
      .setOutputCol("features")

    // val featured = assembler.transform(indexed_currency)
    // featured.show()

    /** Create a logistic regression model **/

    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)

    // val lred = lr.fit(featured).transform(featured)
    // lred.show()

    /** Pipeline: run all algorithms above **/

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, cvModel, idf, indexer_country, indexer_currency, assembler, lr))

    // val model = pipeline.fit(df)
    // val data = model.transform(df)
    // data.show()


    /** Divide the dataset into "training" and "test" **/

    val Array(training, test) = df.randomSplit(Array(0.9, 0.1))

    //training.show()
    //test.show()


    /** Build a param grid used in grid search-based model selection **/

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(10E-8, 10E-6, 10E-4, 10E-2))
      .addGrid(cvModel.minDF, Array(55.0, 75.0, 95.0))
      .build()


    /** Build an evaluator by multiclass classification **/

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")


    /** Fits the Estimator by TrainValidationSplit **/

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)


    /** Apply constructed estimator model to training set **/

    val model_valided = trainValidationSplit.fit(training)

    val df_WithPredictions = model_valided.transform(test)

    val accuracy = evaluator.evaluate(df_WithPredictions)




    /** Display the result **/

    println("The Score of this estimator is: " + accuracy)

    df_WithPredictions.groupBy( "final_status", "predictions").count.show()



    /** Save the valided model and the forecasted result **/

    model_valided.write.overwrite.save("/Users/kaichenma/Documents/copies/TP_Spark/model_pred")

    df_WithPredictions.write.mode(SaveMode.Overwrite).parquet("/Users/kaichenma/Documents/copies/TP_Spark/result_pred")



  }
}
