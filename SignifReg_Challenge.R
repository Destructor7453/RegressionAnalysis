library(stats)
library(MASS)
library(SignifReg)

traindata <- read.table("E:/Spring 2021/Classes/540/Project/Project540/TrainingData.txt", header = FALSE)
testdata <- read.csv("E:/Spring 2021/Classes/540/Project/Project540/TestData.txt", header = FALSE)

for(i in 1:ncol(traindata)){
  traindata[is.na(traindata[,i]), i] <- mean(traindata[,i], na.rm = TRUE)
}

colnames(traindata) <- c('index','precipitation', 'jantemp', 'jultemp', 'pop65', 'housemem', 
                         'schooling', 'kitchen', 'popsqmi', 'nonwhitepop', 'office', 'less3000', 'HCpol', 'NOpol', 'SO2pol', 'atmosmoist', 'deathrate')
traindata

fulltrainmodel<- lm(traindata$deathrate~., data = traindata)
nulltrainmodel <- lm(traindata$deathrate~1, data = traindata)

scope = list(lower = formula(nulltrainmodel), upper= formula(fulltrainmodel))

fit2 = lm(traindata$deathrate~., traindata)
select.fit2 = SignifReg(fit = fit2, scope = scope,alpha = 1, direction = "backward", criterion = "AIC",adjust.method = "fdr",trace=TRUE)
select.fit2$steps.info