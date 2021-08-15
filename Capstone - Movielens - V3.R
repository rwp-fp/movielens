#Initial code is from EDX and creates the datasets 
#Project code to generate estimates starts from line 70

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                           title = as.character(title),
#                                           genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)











#####################################
# Project code start from this point
#####################################

# Make a copy of the edx and validation datasets (quicker to restore if required - code above takes long time to run)
edx_copy <- edx
validation_copy <- validation

###################################################################
# Initial code tests different approaches to generating predictors
###################################################################

# Make Training and Test sets from the edx dataset
# Test set is 20% of the edx dataset

set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
ind <- createDataPartition(edx$rating,0.2, times=1)
edx_train <- edx[-ind$Resample1,]
edx_test <- edx[ind$Resample1,]

#Filter edx_test dataset to remove any movies not included in edx_train dataset

edx_test <- edx_test %>%
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")

# Define function to measure RMSE
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

rm(ind)

#########################################
# Predict based on mean rating only. 
#########################################

rating_mean <- mean(edx_train$rating)
edx_train <- edx_train %>% mutate(pred_mean = rating_mean)

#add predictors to the edx_test set so impact of each predictor
#can be checked as they are generated.

edx_test <- edx_test %>% mutate(pred_mean = rating_mean)

#Calculate RMSE and put in data.frame to track results
Results_Tracker <- data.frame(Method = "Mean Only", RMSE = RMSE(edx_test$rating,edx_test$pred_mean))

rm(rating_mean)

##################################################
# Predict based on average rating for each movie. 
##################################################

movie_effect <- edx_train %>%
  group_by(movieId) %>% summarize(movie_predictor = mean(rating - pred_mean))

edx_train <-edx_train  %>% 
  left_join(movie_effect, by='movieId') 

edx_test <- edx_test %>% 
  left_join(movie_effect, by='movieId') 

#Calculate RMSE and put in data.frame to track results
Results_Tracker <- Results_Tracker %>% rbind(c("Mean + Movie", RMSE(edx_test$rating,(edx_test$pred_mean+edx_test$movie_predictor))))

rm(movie_effect)

##################################################################
# Check the impact of using a regularised movie average predictor. 
##################################################################

#Tune Lambda to find optimum value that minimises RMSE
lambda <- seq(1,3,0.1)

lambda_results <- sapply(lambda,function(l){ 
  print(l)
  movie_effect_reg <- edx_train %>%
    group_by(movieId) %>% summarize(movie_reg_predictor = sum(rating - pred_mean)/(n()+l))
  
  regularised_pred <- edx_test %>% 
    left_join(movie_effect_reg, by='movieId')
  
  RMSE(edx_test$rating,regularised_pred$pred_mean + regularised_pred$movie_reg_predictor)
})

#Plot results to find optimum value of Lambda
qplot(lambda,lambda_results)
lambda[which.min(lambda_results)]
#above gives 2.4 as optimum value of Lambda.
#use this value of Lambda to calculate the regularised movie average predictor

lambda <- lambda[which.min(lambda_results)]

movie_effect_reg <- edx_train %>%
  group_by(movieId) %>% summarize(movie_reg_predictor = sum(rating - pred_mean)/(n()+lambda))

edx_train <- edx_train %>% left_join(movie_effect_reg, by = 'movieId')
edx_test <- edx_test %>% left_join(movie_effect_reg, by = 'movieId')

#Calculate RMSE and put in data.frame to track results
Results_Tracker <- Results_Tracker %>% rbind(c("Mean + Movie (Regularised)", 
                                               RMSE(edx_test$rating,(edx_test$pred_mean+edx_test$movie_reg_predictor))))

rm(movie_effect_reg)

###############################################
# Predict using average rating from each user 
###############################################

user_effect <- edx_train %>% group_by(userId) %>% summarize(user_predictor = mean(rating - pred_mean - movie_predictor))

edx_train <-edx_train  %>% 
  left_join(user_effect, by='userId') 

edx_test <- edx_test %>% left_join(user_effect, by = "userId")

#Calculate RMSE and put in data.frame to track results
Results_Tracker <- Results_Tracker %>% rbind(c("Mean + Movie + User", RMSE(edx_test$rating,(edx_test$pred_mean+edx_test$movie_predictor+edx_test$user_predictor))))

rm(user_effect)

##################################################################
# Check the impact of using a regularised user average predictor. 
##################################################################

#Tune Lambda
lambda <- seq(4,6,0.1)

lambda_results <- sapply(lambda,function(l){ 
  print(l)
  user_effect_reg <- edx_train %>%
    group_by(userId) %>% summarize(user_reg_predictor = sum(rating - pred_mean - movie_reg_predictor)/(n()+l))
  
  regularised_pred <- edx_test %>% 
    left_join(user_effect_reg, by='userId')
  
  RMSE(edx_test$rating,regularised_pred$user_reg_predictor+ regularised_pred$movie_reg_predictor + regularised_pred$pred_mean)
})

#Plot results to find optimum value of Lambda
qplot(lambda,lambda_results)
lambda[which.min(lambda_results)]
#above gives 4.9 as optimum value of Lambda use this in the following code 

lambda <- lambda[which.min(lambda_results)]

user_effect_reg <- edx_train %>%
  group_by(userId) %>% summarize(user_reg_predictor = sum(rating - pred_mean - movie_reg_predictor)/(n()+lambda))

edx_train <- edx_train %>% left_join(user_effect_reg, by = 'userId')

edx_test <- edx_test %>% 
  left_join(user_effect_reg, by='userId') 

#Calculate RMSE and put in data.frame to track results
Results_Tracker <- Results_Tracker %>% rbind(c("Mean + Movie + User (Regularised)", 
                                               RMSE(edx_test$rating,(edx_test$pred_mean+edx_test$movie_reg_predictor + edx_test$user_reg_predictor))))

rm(user_effect_reg)

#########################################################
# Use the Years Since a Movie was Released as a Predictor
#########################################################

#Extract the year of the movie's release from its title (in brackets)
#Search for brackets in titles containing 4 digits.
edx_train <- edx_train %>% mutate(year = str_extract(edx_train$title,"\\(\\d{4}\\)"))

#Remove brackets from the extracted year and convert into a numeric
edx_train$year <- edx_train$year %>% str_replace_all("\\(","") %>% str_replace_all("\\)","")
edx_train$year <- as.numeric(edx_train$year)

#Calculate the years between movie release and date of the rating.
#timestamp in dataset must be converted to date and year extracted.
edx_train <- edx_train %>% mutate(years_since_release = year(as.POSIXct(timestamp, origin = "1970-1-1")) - year)

#Complete the same process on the test data set to obtain year of release
#and years since release rating was given
edx_test <- edx_test %>% mutate(year = str_extract(edx_test$title,"\\(\\d{4}\\)"))
edx_test$year <- edx_test$year %>% str_replace_all("\\(","") %>% str_replace_all("\\)","")
edx_test$year <- as.numeric(edx_test$year)
edx_test <- edx_test %>% mutate(years_since_release = year(as.POSIXct(timestamp, origin = "1970-1-1")) - year)

#Use the new years since release data as a predictor
years_since_effect <- edx_train %>% group_by(years_since_release) %>% summarise(years_since_predictor = mean(rating - pred_mean - movie_predictor - user_predictor))

edx_train <- edx_train %>% left_join(years_since_effect, by="years_since_release")

edx_test <- edx_test %>% left_join(years_since_effect, by ="years_since_release")

#Calculate RMSE and put in data.frame to track results
Results_Tracker <- Results_Tracker %>% rbind(c("Mean + Movie + Years Since Release", RMSE(edx_test$rating,(edx_test$pred_mean+edx_test$movie_predictor+edx_test$user_predictor+ edx_test$years_since_predictor))))

rm(years_since_effect)

########################################################################
# Check the impact of using a regularised years since release predictor. 
########################################################################

#Tune Lambda
lambda <- seq(20,30,0.5)

lambda_results <- sapply(lambda,function(l){ 
  print(l)
  years_since_effect_reg <- edx_train %>%
    group_by(years_since_release) %>% summarize(years_since_reg_predictor = sum(rating - pred_mean - movie_reg_predictor - user_reg_predictor)/(n()+l))
  
  regularised_pred <- edx_test %>% 
    left_join(years_since_effect_reg, by='years_since_release')
  
  RMSE(edx_test$rating, regularised_pred$years_since_reg_predictor +regularised_pred$user_reg_predictor+ regularised_pred$movie_reg_predictor + regularised_pred$pred_mean)
})

#Plot results to find optimum value of Lambda
qplot(lambda,lambda_results)
lambda[which.min(lambda_results)]
#above gives NO optimum value of Lambda 

lambda <- lambda[which.min(lambda_results)]

years_since_effect_reg <- edx_train %>%
  group_by(years_since_release) %>% summarize(years_since_reg_predictor = sum(rating - pred_mean - movie_reg_predictor - user_reg_predictor)/(n()+lambda))

edx_train <- edx_train %>% left_join(years_since_effect_reg, by = 'years_since_release')

edx_test <- edx_test %>% 
  left_join(years_since_effect_reg, by='years_since_release') 

#Calculate RMSE and put in data.frame to track results
Results_Tracker <- Results_Tracker %>% rbind(c("Mean + Movie + User + Years Since Release (Regularised)", 
                                               RMSE(edx_test$rating,(edx_test$pred_mean+edx_test$movie_reg_predictor + edx_test$user_reg_predictor + edx_test$years_since_reg_predictor))))

rm(years_since_effect_reg)

###############################
# Use Genre as a Predictor
###############################

genre_effect <- edx_train %>% group_by(genres) %>% summarise(genre_predictor = mean(rating - pred_mean - movie_predictor - user_predictor - years_since_predictor))

edx_train <- edx_train %>% left_join(genre_effect, by="genres")

edx_test <- edx_test %>% left_join(genre_effect, by ="genres")

#Calculate RMSE and put in data.frame to track results
Results_Tracker <- Results_Tracker %>% rbind(c("Mean + Movie + User + Years Since Release + Genre", RMSE(edx_test$rating,(edx_test$pred_mean+edx_test$movie_predictor+edx_test$user_predictor+edx_test$years_since_predictor + edx_test$genre_predictor))))

rm(genre_effect)

#############################################################
# Check the impact of using a regularised genre predictor. 
#############################################################

#Tune Lambda
lambda <- seq(2,3,0.1)

lambda_results <- sapply(lambda,function(l){ 
  print(l)
  genre_effect_reg <- edx_train %>%
    group_by(genres) %>% summarize(genre_reg_predictor = sum(rating - pred_mean - movie_reg_predictor - user_reg_predictor - years_since_reg_predictor)/(n()+l))
  
  regularised_pred <- edx_test %>% 
    left_join(genre_effect_reg, by='genres')
  
  RMSE(edx_test$rating,regularised_pred$genre_reg_predictor +regularised_pred$user_reg_predictor+ regularised_pred$movie_reg_predictor + regularised_pred$years_since_reg_predictor + regularised_pred$pred_mean)
})

#Plot results to find optimum value of Lambda
qplot(lambda,lambda_results)
lambda[which.min(lambda_results)]
#above gives 2.2 as the optimum value of Lambda 

lambda <- lambda[which.min(lambda_results)]

genre_effect_reg <- edx_train %>%
  group_by(genres) %>% summarize(genre_reg_predictor = sum(rating - pred_mean - movie_reg_predictor - user_reg_predictor - years_since_reg_predictor)/(n()+lambda))

edx_train <- edx_train %>% left_join(genre_effect_reg, by = 'genres')

edx_test <- edx_test %>% 
  left_join(genre_effect_reg, by='genres') 

#Calculate RMSE and put in data.frame to track results
Results_Tracker <- Results_Tracker %>% rbind(c("Mean + Movie +User + Years Since Release + Genres (Regularised)", 
                                               RMSE(edx_test$rating,(edx_test$pred_mean+edx_test$movie_reg_predictor + edx_test$user_reg_predictor + edx_test$years_since_reg_predictor + edx_test$genre_reg_predictor))))

rm(genre_effect_reg)

############################################################
# Day Predictor
# Convert timestamp into day of week and use as a predictor
############################################################

# Convert timestamp to day of week - add to training and test datasets as 'day' field
library(lubridate)
edx_train <- edx_train %>% mutate(day = wday(as.POSIXct(timestamp, origin = "1970-1-1")))
edx_test <-edx_test %>% mutate(day = wday(as.POSIXct(timestamp, origin = "1970-1-1")))

# Use day as a predictor
day_effect <- edx_train %>% group_by(day) %>% summarise(day_predictor = mean(rating - pred_mean - movie_predictor - user_predictor - genre_predictor - years_since_predictor))

edx_train <- edx_train %>% left_join(day_effect, by="day")

edx_test <- edx_test %>% left_join(day_effect, by ="day")

#Calculate RMSE and put in data.frame to track results
Results_Tracker <- Results_Tracker %>% rbind(c("Mean + Movie + User + Years Since Release +Genre + Day", RMSE(edx_test$rating,(edx_test$pred_mean+edx_test$movie_predictor+edx_test$user_predictor+edx_test$years_since_predictor + edx_test$day_predictor + edx_test$genre_predictor))))

rm(day_effect)

#############################################################
# Check the impact of using a regularised day predictor. 
#############################################################

#Tune Lambda initially
lambda <- seq(1,20,1)

lambda_results <- sapply(lambda,function(l){ 
  print(l)
  day_effect_reg <- edx_train %>%
    group_by(day) %>% summarize(day_reg_predictor = sum(rating - pred_mean - movie_reg_predictor - user_reg_predictor - years_since_reg_predictor - genre_reg_predictor)/(n()+l))
  
  regularised_pred <- edx_test %>% 
    left_join(day_effect_reg, by='day')
  
  RMSE(edx_test$rating,regularised_pred$day_reg_predictor + regularised_pred$genre_reg_predictor +regularised_pred$user_reg_predictor+ regularised_pred$years_since_reg_predictor + regularised_pred$movie_reg_predictor + regularised_pred$pred_mean)
})

#Plot results to find optimum value of Lambda
qplot(lambda,lambda_results)
lambda[which.min(lambda_results)]
#above gives NO optimum value of Lambda use this in the following code 

lambda <- lambda[which.min(lambda_results)]

day_effect_reg <- edx_train %>%
  group_by(day) %>% summarize(day_reg_predictor = sum(rating - pred_mean - movie_reg_predictor - user_reg_predictor - years_since_reg_predictor - genre_reg_predictor)/(n()+lambda))

edx_train <- edx_train %>% left_join(day_effect_reg, by = 'day')

edx_test <- edx_test %>% 
  left_join(day_effect_reg, by='day') 

#Calculate RMSE and put in data.frame to track results
Results_Tracker <- Results_Tracker %>% rbind(c("Mean + Movie + User + Years Since Release + Genres + Day (Regularised)", 
                                               RMSE(edx_test$rating,(edx_test$pred_mean+edx_test$movie_reg_predictor + edx_test$user_reg_predictor + edx_test$years_since_reg_predictor + edx_test$genre_reg_predictor + edx_test$day_reg_predictor))))

rm(day_effect_reg)

###########################################################
#Create final estimated ratings based on predictors above
###########################################################

final_estimate <- edx_test$pred_mean + edx_test$movie_reg_predictor + edx_test$user_reg_predictor + edx_test$genre_reg_predictor + edx_test$day_reg_predictor + edx_test$years_since_reg_predictor

#Check the final estimated ratings for any >5 or below <0.5
#Reduce any ratings above 5 to 5, or below 0.5 to 0.5

final_estimate <- ifelse(final_estimate >5, 5, final_estimate)
final_estimate <- ifelse(final_estimate <0.5, 0.5, final_estimate)

Results_Tracker <- Results_Tracker %>% rbind(c("Final Estimate - Top & Tailed", RMSE(edx_test$rating,final_estimate)))

#Actual ratings are at 0.5 intervals
#Check whether rounding estimated ratings to nearest 0.5 reduces RMSE
final_estimate <- round(final_estimate/0.5)*0.5
Results_Tracker <- Results_Tracker %>% rbind(c("Final Estimate - Rounded to nearest 0.5", RMSE(edx_test$rating,final_estimate)))

rm(final_estimate)

#Plot all the estimated ratings to visualise the impact of the different predictors
Results_Tracker %>% ggplot(aes(reorder(Method, -as.numeric(RMSE)),RMSE)) + geom_col(fill = "dark orange") + theme(axis.text.x = element_text(angle = 55, vjust = 1, hjust=1))

#Write Results_Tracker to file for use in R Markdown
write.csv(Results_Tracker, "Results_Tracker.csv")





##############################################################
##############################################################
#Use full EDX set to train and run against Validation dataset
##############################################################
##############################################################

#########################
# Calculate mean rating. 
#########################

rating_mean <- mean(edx$rating)
edx <- edx %>% mutate(pred_mean = rating_mean)

#Add all predictors to the Validation dataset as generated for use in 
#calculating final estimated rating
validation <- validation %>% mutate(pred_mean = rating_mean)

#########################################
# Regularised movie average predictor. 
#########################################

# Use previous 2.4 as optimum value of Lambda 

lambda <- 2.4

movie_effect_reg <- edx %>%
  group_by(movieId) %>% summarize(movie_reg_predictor = sum(rating - pred_mean)/(n()+lambda))

edx <- edx %>% left_join(movie_effect_reg, by = 'movieId')
validation <- validation %>% left_join(movie_effect_reg, by = 'movieId')

rm(movie_effect_reg)

#########################################
# Regularised user average predictor. 
#########################################

#use previous 4.9 as optimum value of Lambda 

lambda <- 4.9

user_effect_reg <- edx %>%
  group_by(userId) %>% summarize(user_reg_predictor = sum(rating - pred_mean - movie_reg_predictor)/(n()+lambda))

edx <- edx %>% left_join(user_effect_reg, by = 'userId')

validation <- validation %>% 
  left_join(user_effect_reg, by='userId') 

rm(user_effect_reg)

###############################
# Years Since Release Predictor
###############################

#Extract year of release from movie title
#Search title for 4 digits in brackets.
edx <- edx %>% mutate(year = str_extract(edx$title,"\\(\\d{4}\\)"))

#Remove brackets from extracted text and convert to a numeric.
edx$year <- edx$year %>% str_replace_all("\\(","") %>% str_replace_all("\\)","")
edx$year <- as.numeric(edx$year)

#Calculate the years between the date of release and the date of the rating
#Convert the timestamp of the rating to a date and extract year to do this.
edx <- edx %>% mutate(years_since_release = year(as.POSIXct(timestamp, origin = "1970-1-1")) - year)

#Complete the same process on the validation dataset
#Extract year and calculate years since release for the rating 
validation <- validation %>% mutate(year = str_extract(validation$title,"\\(\\d{4}\\)"))
validation$year <- validation$year %>% str_replace_all("\\(","") %>% str_replace_all("\\)","")
validation$year <- as.numeric(validation$year)
validation <- validation %>% mutate(years_since_release = year(as.POSIXct(timestamp, origin = "1970-1-1")) - year)

#Generate predictor based on years since release
years_since_effect <- edx %>% group_by(years_since_release) %>% summarise(years_since_predictor = mean(rating - pred_mean - movie_reg_predictor - user_reg_predictor))

edx <- edx %>% left_join(years_since_effect, by="years_since_release")

validation <- validation %>% left_join(years_since_effect, by ="years_since_release")

rm(years_since_effect)

###############################
# Regularised Genre Predictor
###############################

#use previous 2.2 as optimum value of Lambda 

lambda <- 2.2

genre_effect_reg <- edx %>% 
  group_by(genres) %>% summarise(genre_reg_predictor = sum(rating - pred_mean - movie_reg_predictor - user_reg_predictor - years_since_predictor)/(n()+lambda))

edx <- edx %>% left_join(genre_effect_reg, by="genres")

validation <- validation %>% left_join(genre_effect_reg, by ="genres")

rm(genre_effect_reg)

###########################################################
# Day Predictor
# convert timestamp into day of week and use as a predictor
###########################################################

# Convert timestamp to day of week - add to EDX and Validation datasets as 'day' field
library(lubridate)
edx <- edx %>% mutate(day = wday(as.POSIXct(timestamp, origin = "1970-1-1")))
validation <-validation %>% mutate(day = wday(as.POSIXct(timestamp, origin = "1970-1-1")))

# Use day as a predictor
day_effect <- edx %>% group_by(day) %>% summarise(day_predictor = mean(rating - pred_mean - movie_reg_predictor - user_reg_predictor - genre_reg_predictor - years_since_predictor))

edx <- edx %>% left_join(day_effect, by="day")

validation <- validation %>% left_join(day_effect, by ="day")

rm(day_effect)

##################################
#Calculate final predicted rating
##################################

validation <- validation %>% mutate(Final_Estimated_Rating = pred_mean + movie_reg_predictor + user_reg_predictor + genre_reg_predictor + day_predictor + years_since_predictor)

#Check predicted scores to reduce estimates >5 and increase estimates <0.5

validation$Final_Estimated_Rating <- ifelse(validation$Final_Estimated_Rating >5, 5, validation$Final_Estimated_Rating)
validation$Final_Estimated_Rating <- ifelse(validation$Final_Estimated_Rating <0.5, 0.5, validation$Final_Estimated_Rating)

Results_Tracker_Validation <- data.frame(Method = "Final Estimated Rating", RMSE = RMSE(validation$rating,validation$Final_Estimated_Rating))

#save EDX and Validation datasets, and Result_Tracker files for use in the R Markdown.
write.csv(edx, file = "edx.csv")
write.csv(validation, file = "validation.csv")
write.csv(Results_Tracker_Validation, file = "Results_Tracker_Validation.csv")
