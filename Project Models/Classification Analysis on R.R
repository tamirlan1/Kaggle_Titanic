# Importing libraries
library(e1071)
library(MASS)
library(boot)
library(splines)
library(tree)
library(pls)

################################################################################################################################################################################################################################################################################################
# Reading data
dt = read.csv("OnlineNewsPopularity.csv")
dim(dt)
dt$y[dt$shares>=5000] =  1 #create binary response variable
dt$y[dt$shares<5000] =  0
dt = dt[complete.cases(dt),]
dt = dt[,-c(1,2)] # Delete url and timedelta columns
dt = dt[,-59] # Delete shares (continuous variable)

# Divide set into training and test set
set.seed(1)
ind = sample(nrow(dt), 0.8*nrow(dt))
train = dt[ind,]
test = dt[-ind,]

###################################################################################
###################################################################################

# Naive Bayes
nb.fit = naiveBayes(y~., data=train) # Fit model
summary(nb.fit)
nb.fitted = predict(nb.fit, newdata = train, type = 'raw') # fitted values
nb.pred0 = nb.fitted[,2] # fitted values
z.nb = function(tr){ # function to tune threshold
  nb.pred = rep(0, nrow(train))
  nb.pred[nb.pred0>tr] = 1
  conf = table(PRED = nb.pred, REAL = train$y)
  nb.util = 100*conf[2,2] + 15*conf[1,1] - 15*conf[2,1] - 30*conf[1,2]
}
tr = seq(0.01,0.99,0.01) # threshold values
result.nb = sapply(tr, z.nb) # Go through all threshold values
nb.util = max(result.nb); nb.util # Find maximum training utility
max.result.nb = which.max(result.nb) # Find index where maximum utility occurs
max.threshold.nb = tr[which.max(result.nb)]; max.threshold.nb # Find threshold value that leads to maximum utility
nb.pred = predict(nb.fit, newdata = test, type = 'raw') # PRedict values using test set
nb.pred.test = nb.pred[,2]
nb.pred.final = rep(0, nrow(test))
nb.pred.final[nb.pred.test>max.threshold.nb] = 1 # Use max threshold
nb.conf = table(PRED = nb.pred.final, REAL = test$y); nb.conf #confusion matrix
nb.accur = sum(diag(nb.conf))/sum(nb.conf); nb.accur # Accuracy
nb.util = 100*nb.conf[2,2] + 15*nb.conf[1,1] - 15*nb.conf[2,1] - 30*nb.conf[1,2]; nb.util # Utility
# 72970, 0.8354143

###################################################################################
###################################################################################

# Logistic Regression
log.fit = glm(y~., data=train, family = 'binomial') # fit model
summary(log.fit)
log.pred0 = fitted(log.fit) #fitted values
z.log = function(tr){ #fucntion to tune threshold
  log.pred = rep(0, nrow(train))
  log.pred[log.pred0>tr] = 1
  conf = table(PRED = log.pred, REAL = train$y); conf
  log.util = 100*conf[2,2] + 15*conf[1,1] - 15*conf[2,1] - 30*conf[1,2]; log.util
}
tr = seq(0.01,0.59,0.01) #0.59 is the maximum value in an array log.pred0
result.log = sapply(tr, z.log)
log.util = max(result.log); log.util # Find maximum train utility
max.result.log = which.max(result.log) # Find index where maximum utility occurs
max.threshold.log = tr[which.max(result.log)]; max.threshold.log # Find threshold value that leads to maximum utility
log.pr = predict(log.fit, newdata = test, type = 'response') #predictions on test set
log.pred = rep(0, nrow(test))
log.pred[log.pr>max.threshold.log] = 1
log.conf = table(PRED = log.pred, REAL = test$y); log.conf #conf. matrix
log.accur = sum(diag(log.conf))/sum(log.conf); log.accur #accuracy
log.util = 100*log.conf[2,2] + 15*log.conf[1,1] - 15*log.conf[2,1] - 30*log.conf[1,2]; log.util #utility
# 92580, 0.7593644

###################################################################################
###################################################################################

# Linear Discriminant Analysis
lda.fit = lda(y~., data=train) #fit the model
summary(lda.fit)
lda.test.pred = predict(lda.fit, test) #predictions on a test set
lda.conf = table(PRED = lda.test.pred$class, REAL = test$y); lda.conf #conf matrix
lda.accur = sum(diag(lda.conf))/sum(lda.conf); lda.accur # accuracy
lda.util = 100*lda.conf[2,2] + 15*lda.conf[1,1] - 15*lda.conf[2,1] - 30*lda.conf[1,2]; lda.util #utility
# 72920, 0.8612688

###################################################################################
###################################################################################

# Stepwise regression
null.fit = glm(y~1, data=train, family = 'binomial') # empty model
full.fit = glm(y~., data=train, family = 'binomial') # full model
fwd.fit.AIC = step(null.fit, scope=list(lower=formula(null.fit),upper=formula(full.fit)), direction = "forward", family = 'binomial') # AIC stepwise fit
summary(fwd.fit.AIC)
fwd.fit.BIC = step(null.fit, scope=list(lower=formula(null.fit),upper=formula(full.fit)), direction = "forward", family = 'binomial', k = log(length(ind))) # BIC stepwise fit
summary(fwd.fit.BIC)

# AIC covariates
# y ~ kw_avg_avg + kw_max_avg + kw_avg_max + LDA_02 + kw_min_avg + 
#   num_hrefs + data_channel_is_entertainment + data_channel_is_bus + 
#   is_weekend + self_reference_avg_sharess + abs_title_sentiment_polarity + 
#   n_tokens_content + num_self_hrefs + weekday_is_monday + average_token_length + 
#   global_subjectivity + LDA_00 + n_unique_tokens + num_imgs + 
#   data_channel_is_tech + kw_min_min + num_keywords + min_negative_polarity + 
#   data_channel_is_socmed + abs_title_subjectivity + title_sentiment_polarity + 
#   weekday_is_saturday + num_videos + avg_positive_polarity + 
#   kw_avg_min + n_tokens_title + kw_max_min

# BIC covariates
# y ~ kw_avg_avg + kw_max_avg + kw_avg_max + LDA_02 + kw_min_avg + 
#   num_hrefs + data_channel_is_entertainment + data_channel_is_bus + 
#   is_weekend + self_reference_avg_sharess + abs_title_sentiment_polarity + 
#   n_tokens_content + num_self_hrefs + weekday_is_monday

###################################################################################
###################################################################################

# Logistic Regression with covaiates from BIC Fwd Regression
glm.BIC = glm(y ~ kw_avg_avg + kw_max_avg + kw_avg_max + LDA_02 + kw_min_avg + # fit BIC model
                num_hrefs + data_channel_is_entertainment + data_channel_is_bus + 
                is_weekend + self_reference_avg_sharess + abs_title_sentiment_polarity + 
                n_tokens_content + num_self_hrefs + weekday_is_monday, data=train, family="binomial")
summary(glm.BIC)
BIC.pred0 = fitted(glm.BIC) #fitted values
z.BIC = function(tr){ #function to tune threshold
  BIC.pred = rep(0, nrow(train))
  BIC.pred[BIC.pred0>tr] = 1
  conf = table(PRED = BIC.pred, REAL = train$y); conf
  log.util = 100*conf[2,2] + 15*conf[1,1] - 15*conf[2,1] - 30*conf[1,2]; log.util
}
tr = seq(0.01,0.99,0.01) #0.99 is the maximum value in an array log.pred0
result.BIC = sapply(tr, z.BIC)
BIC.util = max(result.BIC); BIC.util # Find maximum train utility
max.result.BIC = which.max(result.BIC) # Find index where maximum utility occurs
max.threshold.BIC = tr[which.max(result.BIC)]; max.threshold.BIC # Find threshold value that leads to maximum utility
BIC.pr = predict(glm.BIC, newdata = test, type = 'response')
BIC.pred = rep(0, nrow(test))
BIC.pred[BIC.pr>max.threshold.BIC] = 1
BIC.conf = table(PRED = BIC.pred, REAL = test$y); BIC.conf #confusion matrix
BIC.accur = sum(diag(BIC.conf))/sum(BIC.conf); BIC.accur #accuracy
BIC.util = 100*BIC.conf[2,2] + 15*BIC.conf[1,1] - 15*BIC.conf[2,1] - 30*BIC.conf[1,2]; BIC.util #utility
# 92010, 0.7468785

###################################################################################
###################################################################################

# Logistic Regression with covaiates from AIC Fwd Regression
glm.AIC = glm(y ~ kw_avg_avg + kw_max_avg + kw_avg_max + LDA_02 + kw_min_avg + # FIT AIC model
                num_hrefs + data_channel_is_entertainment + data_channel_is_bus + 
                is_weekend + self_reference_avg_sharess + abs_title_sentiment_polarity + 
                n_tokens_content + num_self_hrefs + weekday_is_monday + average_token_length + 
                global_subjectivity + LDA_00 + n_unique_tokens + num_imgs + 
                data_channel_is_tech + kw_min_min + num_keywords + min_negative_polarity + 
                data_channel_is_socmed + abs_title_subjectivity + title_sentiment_polarity + 
                weekday_is_saturday + num_videos + avg_positive_polarity + 
                kw_avg_min + n_tokens_title + kw_max_min, data=train, family="binomial")
summary(glm.AIC)
AIC.pred0 = fitted(glm.AIC)
z.AIC = function(tr){ #function to vary threshold
  AIC.pred = rep(0, nrow(train))
  AIC.pred[AIC.pred0>tr] = 1
  conf = table(PRED = AIC.pred, REAL = train$y); conf
  AIC.util = 100*conf[2,2] + 15*conf[1,1] - 15*conf[2,1] - 30*conf[1,2]; AIC.util
}
tr = seq(0.01,0.99,0.01) #0.99 is the maximum value in an array log.pred0
result.AIC = sapply(tr, z.AIC)
AIC.util = max(result.AIC); AIC.util # Find maximum train utility
max.result.AIC = which.max(result.AIC) # Find index where maximum utility occurs
max.threshold.AIC = tr[which.max(result.AIC)]; max.threshold.AIC # Find threshold value that leads to maximum utility
AIC.pr = predict(glm.AIC, newdata = test, type = 'response')
AIC.pred = rep(0, nrow(test))
AIC.pred[AIC.pr>max.threshold.AIC] = 1
AIC.conf = table(PRED = AIC.pred, REAL = test$y); AIC.conf # conf matrix
AIC.accur = sum(diag(AIC.conf))/sum(AIC.conf); AIC.accur # accuracy
AIC.util = 100*AIC.conf[2,2] + 15*AIC.conf[1,1] - 15*AIC.conf[2,1] - 30*AIC.conf[1,2]; AIC.util #utility
# 92820, 0.7427166

###################################################################################
###################################################################################

# Natural splines
glm.ns4 = glm(y ~ ns(kw_avg_avg + kw_max_avg + kw_avg_max + LDA_02 + kw_min_avg + # fit the model
                       num_hrefs + data_channel_is_entertainment + data_channel_is_bus + 
                       is_weekend + self_reference_avg_sharess + abs_title_sentiment_polarity + 
                       n_tokens_content + num_self_hrefs + weekday_is_monday, df=4), data=train, family="binomial")
summary(glm.ns4)
ns4.pred0 = glm.ns4$fitted.values #fitted values
z.ns4 = function(tr){ #function to vary threshold
  ns4.pred = rep(0, nrow(train))
  ns4.pred[ns4.pred0>tr] = 1
  conf = table(PRED = ns4.pred, REAL = train$y); conf
  log.util = 100*conf[2,2] + 15*conf[1,1] - 15*conf[2,1] - 30*conf[1,2]; log.util
}
tr = seq(0.08,0.2,0.01) # vary threshold. max value is only 0.2 for probabilities
result.ns4 = sapply(tr, z.ns4)
ns4.util = max(result.ns4); ns4.util # Find maximum train utility
max.result.ns4 = which.max(result.ns4) # Find index where maximum utility occurs
max.threshold.ns4 = tr[which.max(result.ns4)]; max.threshold.ns4 # Find threshold value that leads to maximum utility
ns4.pr = predict(glm.ns4, newdata = test, type = 'response')
ns4.pred = rep(0, nrow(test))
ns4.pred[ns4.pr>max.threshold.ns4] = 1
ns4.conf = table(PRED = ns4.pred, REAL = test$y); ns4.conf # confusion matrix
ns4.accur = sum(diag(ns4.conf))/sum(ns4.conf); ns4.accur #accuracy
ns4.util = 100*ns4.conf[2,2] + 15*ns4.conf[1,1] - 15*ns4.conf[2,1] - 30*ns4.conf[1,2]; ns4.util #utility
# 70410, 0.8603859

###################################################################################
###################################################################################

# trees
set.seed(1)
fit.tree = tree(y~., data = train) # fit the model
summary(fit.tree)
plot(fit.tree)
text(fit.tree, pretty = 0)
tree.pred = predict(fit.tree, test)#, type = "class") #fitted values
tree.pred[tree.pred>0.1] = 1 #arbitrary threshold
tree.pred[tree.pred<=0.1] = 0
tree.conf = table(PRED = tree.pred, REAL = test$y); tree.conf #confusion
tree.accur = sum(diag(tree.conf))/sum(tree.conf); tree.accur # accuracy
tree.util = 100*tree.conf[2,2] + 15*tree.conf[1,1] - 15*tree.conf[2,1] - 30*tree.conf[1,2]; tree.util #utility
# 87170, 0.7080338

library(rpart)
rpart.fit <- rpart(y ~ ., data = train, cp=0) #fit the moedl
plot(rpart.fit, uniform=TRUE, main="Classification Tree for shares")
text(rpart.fit, use.n=TRUE, all=TRUE, cex=.8)
printcp(rpart.fit) # display the results 
plotcp(rpart.fit) # visualize cross-validation results 
summary(rpart.fit) # detailed summary of splits
rpart.pred = predict(rpart.fit, test) #predict values 
rpart.pred[rpart.pred>0.1] = 1# arbitrary threshold
rpart.pred[rpart.pred<=0.1] = 0
rpart.conf = table(PRED = rpart.pred, REAL = test$y); rpart.conf #conf matrix
rpart.accur = sum(diag(rpart.conf))/sum(rpart.conf); rpart.accur # accuracy
rpart.util = 100*rpart.conf[2,2] + 15*rpart.conf[1,1] - 15*rpart.conf[2,1] - 30*rpart.conf[1,2]; rpart.util # utility
# 87170, 0.7080338

###################################################################################
###################################################################################

# Principal component regression (PLS) with CV
set.seed(2)
pcr.fit = pcr(y~., data=train , scale =TRUE, validation = "CV") # principal component regression with CV
gen0 = function(c){ #function to vary the number of used components
  pcr.pred0 = predict(pcr.fit, newdata = train, type = 'response', ncomp = c)
  z.pcr = function(tr){ #function to vary threshold
    pcr.pred = rep(0, nrow(train))
    pcr.pred[pcr.pred0>tr] = 1
    conf = table(PRED = pcr.pred, REAL = train$y)
    pcr.util = 100*conf[2,2] + 15*conf[1,1] - 15*conf[2,1] - 30*conf[1,2]
  }
  tr = seq(0.01,0.3,0.01)
  result.pcr = sapply(tr, z.pcr)
  pcr.util = max(result.pcr); pcr.util
}
c0 = seq(5,58,1)
result.pcr.c = sapply(c0, gen0) # Best utility
plot(c0,result.pcr.c, ylab = "Training ROI", xlab = "Number of components")
c = 51 # choose optimal number of components
pcr.pred0 = predict(pcr.fit, newdata = train, type = 'response', ncomp = c) #predict on test set
z.pcr = function(tr){ #function to vary threshold
  pcr.pred = rep(0, nrow(train))
  pcr.pred[pcr.pred0>tr] = 1
  conf = table(PRED = pcr.pred, REAL = train$y)
  pcr.util = 100*conf[2,2] + 15*conf[1,1] - 15*conf[2,1] - 30*conf[1,2]
}
tr = seq(0.01,0.3,0.01) # max possible threshold is 0.3
result.pcr = sapply(tr, z.pcr)
pcr.util = max(result.pcr); pcr.util # Find maximum training utility
max.result.pcr = which.max(result.pcr) # Find index where maximum utility occurs
max.threshold.pcr = tr[which.max(result.pcr)]; max.threshold.pcr # Find threshold value that leads to maximum utility
pcr.pred.test = predict(pcr.fit, newdata = test, type = 'response', ncomp = c)
pcr.pred.final = rep(0, nrow(test))
pcr.pred.final[pcr.pred.test>max.threshold.pcr] = 1
pcr.conf = table(PRED = pcr.pred.final, REAL = test$y); pcr.conf # conf matrix
pcr.accur = sum(diag(pcr.conf))/sum(pcr.conf); pcr.accur # accuracy
pcr.util = 100*pcr.conf[2,2] + 15*pcr.conf[1,1] - 15*pcr.conf[2,1] - 30*pcr.conf[1,2]; pcr.util # utility
# 93830, 0.7738681

###################################################################################
###################################################################################

#Partial least squares regression (PLSR) with CV
set.seed(1)
plsr.fit = plsr(y~., data=train , scale =TRUE, validation = "CV")  # fit the model
gen = function(c){ #function to vary the number of components
  plsr.pred0 = predict(plsr.fit, newdata = train, type = 'response', ncomp = c)
  z.plsr = function(tr){ #function to vary threshold
    plsr.pred = rep(0, nrow(train))
    plsr.pred[plsr.pred0>tr] = 1
    conf = table(PRED = plsr.pred, REAL = train$y)
    plsr.util = 100*conf[2,2] + 15*conf[1,1] - 15*conf[2,1] - 30*conf[1,2]
  }
  tr = seq(0.01,0.3,0.01)
  result.plsr = sapply(tr, z.plsr)
  plsr.util = max(result.plsr); plsr.util
}
c = seq(5,58,1)
result.plsr.c = sapply(c, gen)
plot(c,result.plsr.c, ylab = "Training ROI", xlab = "Number of components")

c = 14 # the best number of components
plsr.pred0 = predict(plsr.fit, newdata = train, type = 'response', ncomp = c)
z.plsr = function(tr){ #function to vary threshold
  plsr.pred = rep(0, nrow(train))
  plsr.pred[plsr.pred0>tr] = 1
  conf = table(PRED = plsr.pred, REAL = train$y)
  plsr.util = 100*conf[2,2] + 15*conf[1,1] - 15*conf[2,1] - 30*conf[1,2]
}
tr = seq(0.01,0.3,0.01) # max prob is 0.3. No need to run threshold above that value
result.plsr = sapply(tr, z.plsr)
plsr.util = max(result.plsr); plsr.util # Find maximum training utility
max.result.plsr = which.max(result.plsr) # Find index where maximum utility occurs
max.threshold.plsr = tr[which.max(result.plsr)]; max.threshold.plsr # Find threshold value that leads to maximum utility
plsr.pred.test = predict(plsr.fit, newdata = test, type = 'response', ncomp = c)
plsr.pred.final = rep(0, nrow(test))
plsr.pred.final[plsr.pred.test>max.threshold.plsr] = 1
plsr.conf = table(PRED = plsr.pred.final, REAL = test$y); plsr.conf # conf matrix
plsr.accur = sum(diag(plsr.conf))/sum(plsr.conf); plsr.accur #accuracy
plsr.util = 100*plsr.conf[2,2] + 15*plsr.conf[1,1] - 15*plsr.conf[2,1] - 30*plsr.conf[1,2]; plsr.util # utility
# 93550, 0.7731114
