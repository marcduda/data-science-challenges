package uga.preg.im
import org.apache.spark.sql._
import org.apache.spark.rdd.RDD
import java.io._
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions
import java.util.Arrays
import scala.util.Try

object Scoremovie {
  
  def getScoreType(str: String)={
    if (str.contains("Mystery|Thriller")){
      1.0
    }
    else if (str.contains("Mystery") || str.contains("Thriller")){
      0.5
    }
    else{0.0}
  }

  def main(args: Array[String]): Unit = {
    var spark: SparkSession = null
    var mainFolder: String = ""

    try {
      mainFolder = args(0)
    } catch {
      case e: Exception => println("Main Folder not specified using default " + mainFolder)
    }

    try {
      
      spark = SparkSession.builder().appName("Find best recommendation after seeing movie Inferno").config("spark.master", "local[*]").getOrCreate()
      val sqlContext = new org.apache.spark.sql.SQLContext(spark.sparkContext)
      import sqlContext.implicits._
      
      //get the genre and id of the movie Inferno (visual inspection)
      val movies = spark.sparkContext.textFile(mainFolder + "ml-latest/moviesNew.csv").filter(l=>l!="movieId,title,genres").filter(l=> l.contains("Inferno")).collect().foreach(x=>println(x))
      
      //calculate the metric associated with the genre 
      val moviesScoreType = spark.sparkContext.textFile(mainFolder + "ml-latest/moviesNew.csv").filter(l=>l!="movieId,title,genres").map(l=>l.split(";")).filter(l=> !(l(0).contains("165101")))
      .map(l=>(l(0),l(1),getScoreType(l(2)),l(1).replace(")","").replace("\"","").split("\\(").last )).toDF("MovieId","MovieTitle","genreScore","year")//.take(10).foreach(x=>println(x))
      
      //get tags associated with the movie Inferno and the frequency of those tags
      val tagsInferno = spark.sparkContext.textFile(mainFolder + "ml-latest/tags.csv").filter(l=>l!="userId,movieId,tag,timestamp").map(l=>l.split(",")).filter(l=>l(1).contains("165101"))//.take(50).foreach(x=>println(x.toList))
      val infernoTagsFreq = tagsInferno.map(l=>l(2)).collect().toList.groupBy(identity).map(l=>(l._1.toLowerCase(),l._2.length))
      println("tags of Inferno are :"+infernoTagsFreq)

      //calculate the metric associated with the tags
      val moviesSameTags = spark.sparkContext.textFile(mainFolder + "ml-latest/tags.csv").filter(l=>l!="userId,movieId,tag,timestamp").map(l=>l.split(",")).filter(l=> !(l(1).contains("165101")))
      .map(l=>(l(1),l(2))).groupByKey()//
      .mapValues(l=>(l.toList,l.size)).mapValues(l=>l._1.groupBy(identity).map(x=>(x._1.toLowerCase(),x._2.length.toDouble/l._2.toDouble)))
      .mapValues(l=>l.map(x=>(x._1,if (infernoTagsFreq.contains(x._1)){x._2*infernoTagsFreq(x._1).toDouble/infernoTagsFreq.values.sum}else{0.0})) )
      .mapValues(l=>l.values.sum)
      
      //get the average rating of each movies
      val ratings = spark.sparkContext.textFile(mainFolder + "ml-latest/ratings.csv").filter(l=>l!="userId,movieId,rating,timestamp").map(l=>l.split(","))//.take(10).foreach(x=>println(x.toList))
      val averageMovieRatings = ratings.map(l=>(l(1),l(2).toDouble) ).groupByKey().mapValues(l=>l.toList.sum/l.size).toDF("MovieId","ratingScore")
      println("the average rating for the movie Inferno (2016) is "+ratings.filter(l=> l(1).contains("165101")).map(l=>l(2).toDouble).mean())
      
      //normalize the metric associated with the tags
      val maxTag = moviesSameTags.map(l=>l._2).max()
      val minTag = moviesSameTags.map(l=>l._2).min()
      val moviesSameTagsNorm = moviesSameTags.mapValues(l=>(l-minTag)/(maxTag-minTag)).toDF("MovieId","tagScore")
      
      //compute the the total score ie sum of the two metrics and remove films that don't have a screening date and have a lower average rating than Inferno
      val movieTotalScore = moviesScoreType.join(moviesSameTagsNorm,"MovieId").join(averageMovieRatings,"MovieId")
      movieTotalScore.show(10, false)
      
      // get the users that rated Inferno with 5 stars, and get the movies that those users also rated with 5 starts, filter the raking with this
      val usersGoodRatings = ratings.filter(l=>l(2).toDouble==5.0 & l(1).contains("165101")).map(l=>l(0)).distinct().collect()
      val goodMovies = ratings.filter(l=>usersGoodRatings.contains(l(0)) & l(2).toDouble==5.0 ).map(l=>l(1)).distinct.collect()
      val rddMovieTotalScore = movieTotalScore.rdd.map{row=>(row.getString(0),row.getString(1),row.getDouble(2),row.getString(3),row.getDouble(4),row.getDouble(5))}
      .filter(l=>goodMovies.contains(l._1) ).map(l=>(l._1,l._2,l._3+l._5,l._6)).filter(l=>l._4>3.02).sortBy(l=>l._3,false).take(30).foreach(x=>println(x))//,MoviesZipRE.findAllMatchIn(l._2),l._5) //+(l._4.toDouble-minYear)/(maxYear-minYear)
      
    
    
    
    } catch {
      case e: Exception => throw e
    } finally {
      spark.stop()
    }
    println("done")
}
 
}