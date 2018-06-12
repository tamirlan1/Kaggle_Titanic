library(glmnet)
library(MASS)
library(FSelector)
library(tree)
library(rpart)
library(splines)
library(ggplot2)

data = read.csv('C:\\Users\\Cameron\\Documents\\MS&E 235\\Project\\OnlineNewsPopularity\\OnlineNewsPopularity.csv', header = TRUE)#Load the data
factors = c(14:19,32:39) #Convert these columns to factors
for (i in factors){
  data[,i] = as.factor(data[,i])
}
data = data[,-1] #Remove URL Identifier
data = data[,-1] #Remove time delta

set.seed(1)
train.ind = sample(nrow(data), .8*nrow(data)) #Generate training indices
train = data[train.ind,] #Create training data
test = data[-train.ind,] #Create test data

################
lm.fit = lm(shares~., data = train) #Fit lm model using all predictors
summary(lm.fit) 
plot(lm.fit$residuals) #Plot residuals. Residuals indicate that model does not fit data
lm.pred = predict(lm.fit, test[,1:58]) #Make predictions on test set with model
lm.err = mean((lm.pred - test[,59])^2);lm.err #Calculate mean squared error

################
#Fit lm model using all predictors and all factor interaction terms
lm.fit2 = lm(shares~. +data_channel_is_lifestyle:weekday_is_monday + data_channel_is_lifestyle:weekday_is_tuesday
                              + data_channel_is_lifestyle:weekday_is_wednesday + data_channel_is_lifestyle:weekday_is_thursday + data_channel_is_lifestyle:weekday_is_friday
                              +data_channel_is_lifestyle:weekday_is_saturday + data_channel_is_lifestyle:weekday_is_sunday + data_channel_is_lifestyle:is_weekend
                              +data_channel_is_entertainment:weekday_is_monday + data_channel_is_entertainment:weekday_is_tuesday
                              + data_channel_is_entertainment:weekday_is_wednesday + data_channel_is_entertainment:weekday_is_thursday + data_channel_is_entertainment:weekday_is_friday
                              +data_channel_is_entertainment:weekday_is_saturday + data_channel_is_entertainment:weekday_is_sunday + data_channel_is_entertainment:is_weekend
                              +data_channel_is_bus:weekday_is_monday + data_channel_is_bus:weekday_is_tuesday
                              + data_channel_is_bus:weekday_is_wednesday + data_channel_is_bus:weekday_is_thursday + data_channel_is_bus:weekday_is_friday
                              +data_channel_is_bus:weekday_is_saturday + data_channel_is_bus:weekday_is_sunday + data_channel_is_bus:is_weekend
                              +data_channel_is_socmed:weekday_is_monday + data_channel_is_socmed:weekday_is_tuesday
                              + data_channel_is_socmed:weekday_is_wednesday + data_channel_is_socmed:weekday_is_thursday + data_channel_is_socmed:weekday_is_friday
                              +data_channel_is_socmed:weekday_is_saturday + data_channel_is_socmed:weekday_is_sunday + data_channel_is_socmed:is_weekend
                              +data_channel_is_tech:weekday_is_monday + data_channel_is_tech:weekday_is_tuesday
                              + data_channel_is_tech:weekday_is_wednesday + data_channel_is_tech:weekday_is_thursday + data_channel_is_tech:weekday_is_friday
                              +data_channel_is_tech:weekday_is_saturday + data_channel_is_tech:weekday_is_sunday + data_channel_is_tech:is_weekend
                              +data_channel_is_world:weekday_is_monday + data_channel_is_world:weekday_is_tuesday
                              + data_channel_is_world:weekday_is_wednesday + data_channel_is_world:weekday_is_thursday + data_channel_is_world:weekday_is_friday
                              +data_channel_is_world:weekday_is_saturday + data_channel_is_world:weekday_is_sunday + data_channel_is_world:is_weekend, data = train)
summary(lm.fit2) 
lm.pred2 = predict(lm.fit2, test[,1:58]) #Make predictions on test set
lm.err2 = mean((lm.pred2 - test[,59])^2);lm.err2 #Calculate mean squared error

##############
lm.fit3 = lm(log(shares)~., data = train) #Fit lm model on log-transformed target variable using all predictors
summary(lm.fit3)
plot(lm.fit3$residuals) #Plot residuals. Residuals indicate the fit is better than before 
lm.pred3 = predict(lm.fit3, test[,1:58]) #Make predictions on test set
lm.err3 = mean((lm.pred3 - log(test[,59]))^2);lm.err3 #Calculate mean squared error

##############
#Fit lm model on log-transformed target variable using all predictors and factor interaction terms
lm.fit4 = lm(log(shares)~. +data_channel_is_lifestyle:weekday_is_monday + data_channel_is_lifestyle:weekday_is_tuesday
             + data_channel_is_lifestyle:weekday_is_wednesday + data_channel_is_lifestyle:weekday_is_thursday + data_channel_is_lifestyle:weekday_is_friday
             +data_channel_is_lifestyle:weekday_is_saturday + data_channel_is_lifestyle:weekday_is_sunday + data_channel_is_lifestyle:is_weekend
             +data_channel_is_entertainment:weekday_is_monday + data_channel_is_entertainment:weekday_is_tuesday
             + data_channel_is_entertainment:weekday_is_wednesday + data_channel_is_entertainment:weekday_is_thursday + data_channel_is_entertainment:weekday_is_friday
             +data_channel_is_entertainment:weekday_is_saturday + data_channel_is_entertainment:weekday_is_sunday + data_channel_is_entertainment:is_weekend
             +data_channel_is_bus:weekday_is_monday + data_channel_is_bus:weekday_is_tuesday
             + data_channel_is_bus:weekday_is_wednesday + data_channel_is_bus:weekday_is_thursday + data_channel_is_bus:weekday_is_friday
             +data_channel_is_bus:weekday_is_saturday + data_channel_is_bus:weekday_is_sunday + data_channel_is_bus:is_weekend
             +data_channel_is_socmed:weekday_is_monday + data_channel_is_socmed:weekday_is_tuesday
             + data_channel_is_socmed:weekday_is_wednesday + data_channel_is_socmed:weekday_is_thursday + data_channel_is_socmed:weekday_is_friday
             +data_channel_is_socmed:weekday_is_saturday + data_channel_is_socmed:weekday_is_sunday + data_channel_is_socmed:is_weekend
             +data_channel_is_tech:weekday_is_monday + data_channel_is_tech:weekday_is_tuesday
             + data_channel_is_tech:weekday_is_wednesday + data_channel_is_tech:weekday_is_thursday + data_channel_is_tech:weekday_is_friday
             +data_channel_is_tech:weekday_is_saturday + data_channel_is_tech:weekday_is_sunday + data_channel_is_tech:is_weekend
             +data_channel_is_world:weekday_is_monday + data_channel_is_world:weekday_is_tuesday
             + data_channel_is_world:weekday_is_wednesday + data_channel_is_world:weekday_is_thursday + data_channel_is_world:weekday_is_friday
             +data_channel_is_world:weekday_is_saturday + data_channel_is_world:weekday_is_sunday + data_channel_is_world:is_weekend, data = train)
summary(lm.fit4)
lm.pred4 = predict(lm.fit4, test[,1:58]) #Make predictions on test set
lm.err4 = mean((lm.pred4 - log(test[,59]))^2);lm.err4 #Calculate mean squared error

#################
cv.lasso.fit = cv.glmnet(data.matrix(train[,1:58]), log(train[,59])) #Fit lasso model using cross validation
coef(cv.lasso.fit) #Coefficients of fitted model
summary(cv.lasso.fit)
lasso.pred = predict(cv.lasso.fit, newx = data.matrix(test[,1:58])) #Make predictions on test set
lasso.error = mean((log(test[,59]) - lasso.pred)^2);lasso.error #Calculate mean squared error

#################
fit.ridge = glmnet(data.matrix(train[,1:58]),log(train[,59]), alpha = 0, family = 'gaussian', nlambda = 100) #Fit Ridge Regression model
cv.ridge <- cv.glmnet(data.matrix(train[,1:58]),log(train[,59]),alpha=0,nfolds=10, family = 'gaussian') #Cross validation with 10 folds
lmin2 = cv.ridge$lambda.min # Choose lambda value corresponsing to min MSE
pred.ridge = predict.glmnet(fit.ridge, newx = data.matrix(test[,1:58]), type='response', s=lmin2) # Predict values using test set
ridge.error = mean((pred.ridge - log(test[,59]))^2); ridge.error # Find MSE

#################
null.model = lm(log(shares)~1, data = train) #Fit null model for stepwise
full.model = lm(log(shares)~., data = train) #Fit all predictors model for stepwise
#Fit all predictors and factor interaction terms for stepwise
factor.interaction.model = lm(log(shares)~. +data_channel_is_lifestyle:weekday_is_monday + data_channel_is_lifestyle:weekday_is_tuesday
                              + data_channel_is_lifestyle:weekday_is_wednesday + data_channel_is_lifestyle:weekday_is_thursday + data_channel_is_lifestyle:weekday_is_friday
                              +data_channel_is_lifestyle:weekday_is_saturday + data_channel_is_lifestyle:weekday_is_sunday + data_channel_is_lifestyle:is_weekend
                              +data_channel_is_entertainment:weekday_is_monday + data_channel_is_entertainment:weekday_is_tuesday
                              + data_channel_is_entertainment:weekday_is_wednesday + data_channel_is_entertainment:weekday_is_thursday + data_channel_is_entertainment:weekday_is_friday
                              +data_channel_is_entertainment:weekday_is_saturday + data_channel_is_entertainment:weekday_is_sunday + data_channel_is_entertainment:is_weekend
                              +data_channel_is_bus:weekday_is_monday + data_channel_is_bus:weekday_is_tuesday
                              + data_channel_is_bus:weekday_is_wednesday + data_channel_is_bus:weekday_is_thursday + data_channel_is_bus:weekday_is_friday
                              +data_channel_is_bus:weekday_is_saturday + data_channel_is_bus:weekday_is_sunday + data_channel_is_bus:is_weekend
                              +data_channel_is_socmed:weekday_is_monday + data_channel_is_socmed:weekday_is_tuesday
                              + data_channel_is_socmed:weekday_is_wednesday + data_channel_is_socmed:weekday_is_thursday + data_channel_is_socmed:weekday_is_friday
                              +data_channel_is_socmed:weekday_is_saturday + data_channel_is_socmed:weekday_is_sunday + data_channel_is_socmed:is_weekend
                              +data_channel_is_tech:weekday_is_monday + data_channel_is_tech:weekday_is_tuesday
                              + data_channel_is_tech:weekday_is_wednesday + data_channel_is_tech:weekday_is_thursday + data_channel_is_tech:weekday_is_friday
                              +data_channel_is_tech:weekday_is_saturday + data_channel_is_tech:weekday_is_sunday + data_channel_is_tech:is_weekend
                              +data_channel_is_world:weekday_is_monday + data_channel_is_world:weekday_is_tuesday
                              + data_channel_is_world:weekday_is_wednesday + data_channel_is_world:weekday_is_thursday + data_channel_is_world:weekday_is_friday
                              +data_channel_is_world:weekday_is_saturday + data_channel_is_world:weekday_is_sunday + data_channel_is_world:is_weekend, data = train)

aic.model = stepAIC(factor.interaction.model, scope = null.model,direction = 'both',  k =2) #Fit lm model using stepwise AIC starting with factor interaction model
pred.aic = predict(aic.model, test[,1:58]) #Make predictions on test set
aic.error = mean((pred.aic - log(test[,59]))^2); aic.error #Calculate mean squared error
summary(aic.model)

##################
aic.model2 = stepAIC(full.model, scope = null.model,direction = 'both',  k =2) #Fit lm model using stepwise AIC starting with all predictors model
pred.aic2 = predict(aic.model2, test[,1:58]) #Make predictions on test set
aic.error2 = mean((pred.aic2 - log(test[,59]))^2); aic.error2 #Calculate mean squared error
summary(aic.model2)

##################
bic.model = stepAIC(factor.interaction.model, scope = null.model,direction = 'both',  k =log(nrow(train))) #Fit lm model using stepwise BIC starting with factor interaction model
pred.bic = predict(bic.model, test[,1:58]) #Make predictions on the test set
bic.error = mean((pred.bic - log(test[,59]))^2); bic.error #Calculate mean squared error
summary(bic.model)

bic.model2 = stepAIC(full.model, scope = null.model,direction = 'both',  k =log(nrow(train))) #Fit lm model using stepwise AIC starting with all predictors model
pred.bic2 = predict(bic.model2, test[,1:58]) #Make prediction on test set
bic.error2 = mean((pred.bic2 - log(test[,59]))^2); bic.error2 #Calculate mean squared error
summary(bic.model2)

#######################
n = nrow(train) #Number of rows in training set
k = 5 #Number of k-folds for cross validation
size = n%/%k #Size of each block
set.seed(5)
rand = runif(n) #Generate n samples from standard normal distribution
rank = rank(rand) #Gives the rank of each of samples
block = (rank-1)%/%size + 1 #Divide the samples into k blocks
block = as.factor(block)

best.bucket = NA #Best minbucket value from random search
best.split = NA #Best minsplit value from random search
best.cp = NA #Best cp value from random search
best.error = 10000000000 #Best mean squared error using the above parameters

for (i in 1:200){
  all.err = numeric(0) #Vector of cv errors
  bucket = sample(1:50, 1) #Sample a value for minbucket
  split = sample(1:50,1) #Sample a value for minsplit
  cp = runif(1) #Sample a value for cp
  for(j in 1:k){
    fit = rpart(log(shares)~., data = train[block!=j,], method = 'anova', minbucket = bucket, minsplit = split, cp = cp) #Fit a tree on a cross validation set
    pred = predict(fit, newdata = train[block==j,1:58]) #Make prediction on the left out fold
    err = mean((pred - log(train[block==j,59]))^2) #Calculate cross validation error
    all.err = rbind(all.err, err) #Add cross validation error to vector
  }
  err = mean(all.err) #Take mean of all cross validation errs
  #Update hyperparameters if error is lower than previous best
  if(err < best.error){
    best.error = err
    best.bucket = bucket
    best.split = split
    best.cp = cp
  }
}
tree.fit = rpart(log(shares)~., data = train, method = 'anova', minbucket = best.bucket, minsplit = best.split, cp = best.cp) #Use best hyperparameters to fit tree on training data
pred.tree = predict(tree.fit, newdata = test[,1:58]) #Make prediction on test set
tree.error = mean((pred.tree - log(test[,59]))^2); tree.error #Calculate mean squared error
summary(tree.fit)
#bucket = 42
#cp = 0.005806056
#split = 43
#best.error = 0.7940875

################
#Fit natural splines model using predictors determined by AIC
splines.fit = lm(log(shares)~data_channel_is_lifestyle+data_channel_is_entertainment+data_channel_is_bus+data_channel_is_socmed+data_channel_is_tech
                                  +weekday_is_monday+weekday_is_tuesday+weekday_is_wednesday+weekday_is_thursday+weekday_is_friday + ns(n_tokens_title+n_tokens_content+n_unique_tokens+n_non_stop_unique_tokens+
                                  num_hrefs+num_self_hrefs+num_imgs+average_token_length+num_keywords+kw_min_min+kw_max_min+kw_avg_min+
                                  kw_min_max+kw_avg_max+kw_min_avg+kw_max_avg+kw_avg_avg+self_reference_min_shares+
                                  self_reference_avg_sharess+LDA_00+LDA_01+LDA_03+LDA_04+global_subjectivity+global_rate_positive_words+global_rate_negative_words+rate_positive_words+min_positive_polarity+
                                  avg_negative_polarity+max_negative_polarity+title_subjectivity+title_sentiment_polarity+abs_title_subjectivity, df = 4),data = train)
summary(splines.fit)
pred.spline = predict(splines.fit, test[,1:58]) #Make prediction on test set
spline.error = mean((pred.spline - log(test[,59]))^2); spline.error #Calculate mean squared error

err = data.frame(Model = c('OLS - No Interaction', 'OLS-Interaction', 'Lasso', 'Ridge', 'AIC-No Interaction', 'AIC - Interaction', 'BIC-No Interaction', 'BIC-Interaction', 'Tree', 'Splines'),
                 Error = c(lm.err3, lm.err4, lasso.error, ridge.error, aic.error2, aic.error, bic.error2, bic.error, tree.error, spline.error))
plot =ggplot(data = err, aes(x = Model, y = Error, fill = Model))+geom_bar(stat = 'identity') 
plot
