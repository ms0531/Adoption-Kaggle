### Kaggle Adoption Competition
### 0.71520 on public LB

library(lubridate)
library(stringr)
library(xgboost)

# read test and train
train <- read.csv("train") 
test <- read.csv("test")

# check for train and test column difference, remove excess
setdiff(names(train), names(test))
train$AnimalID <- NULL
train$OutcomeSubtype <- NULL


# save target and test ID
target <- train$OutcomeType
train$OutcomeType <- NULL
test.id <- test$ID
test$ID <- NULL

# combine data 
alldata <- rbind(train, test)

### Begin extracting/selecting features

# get common breeds and char length of names
frequentBreeds <- names(summary(alldata$Breed,maxsum=20))
NameSummary <- summary(alldata$Name,maxsum=Inf)

# dummy var for mix breed
alldata$Mix <- ifelse(str_detect(alldata$Breed, "Mix")==T,1,0)

# dummy var for Pit in name
alldata$Pit <- ifelse(str_detect(alldata$Breed, "Pit")==T,1,0)

# dummy var for breeds with '/' in name
alldata$UnsureBreed <- ifelse(str_detect(alldata$Breed, "/")==T,1,0)

# extract hour, month,weekday, change time to posixct
dateExtractions <- list('hour','wday','month','year') 

for(i in dateExtractions){
    alldata[i] <- eval(parse(text=paste(i,"(alldata$DateTime)",sep="")))
}

alldata$DateTime <- as.numeric(as.POSIXct(alldata$DateTime)) 

# name length, popular names, drop name
alldata$nameLength <- nchar(as.character(alldata$Name))
alldata$nameObscurity <- NameSummary[match(alldata$Name,names(NameSummary))]
alldata$Name <- NULL

# Change age
multi_sub <- function(pattern, replacement, x, ...){
    result <- x
    for(i in 1:length(pattern)) {
        result <- gsub(pattern[i], replacement[i], result, ...)
    }
    result
}

alldata$AgeuponOutcome <- multi_sub(c(" years?", " months?", " weeks?", " days?"), 
                                    c("0000","00","0",""), 
                                    alldata$AgeuponOutcome)

alldata$AgeuponOutcome <- as.numeric(paste0("0",alldata$AgeuponOutcome))

# dummy var of popular colors
popularColors <- c("Black","White","Brown","Orange","Blue","Tortie","Calico","Chocolate","Gold","Red","Tan","Yellow")

for(i in popularColors){
    alldata[[paste0("col.",i)]] <- grepl(i,alldata$Color)
}

alldata$Color <- NULL

# dummy var for popular breeds
for(i in frequentBreeds){
    alldata[[paste0("breed.",make.names(i))]] <- alldata$Breed == i
}

# change logical to numerical
for(i in names(alldata)) if(is.logical(alldata[[i]])){
    alldata[[i]] <- as.numeric(alldata[[i]])
}

alldata$Breed <- NULL

# change factors to numeric
numFac <- function(x) {
    seq_along(levels(x))[x]
}

alldata$AnimalType <- numFac(alldata$AnimalType)
alldata$SexuponOutcome <- numFac(alldata$SexuponOutcome)

# put train and test back
train.x <- alldata[1:nrow(train),]
test.x <- alldata[26730:nrow(alldata),]

## Setup XGBoost parameters ##
target <- seq_along(levels(target))[target] - 1

train.matrix <- xgb.DMatrix(data=data.matrix(train.x), label=target)

param <- list(objective = "multi:softprob",
              eval_metric = "mlogloss",
              booster = "gbtree",
              eta = 0.02,
              max_depth = 12,
              subsample = 0.7,
              colsample_bytree = 0.7
)

watchlist <- list(train=train.matrix)
numClasses <- max(target) + 1


# cross validation

doCV <- function(params, nrounds) {
    cross.val <- xgb.cv(params = params,
                    nrounds = nrounds,
                    nfold = 3,
                    data = train.matrix,
                    early.stop.round = 50,
                    num_class = numClasses,
                    watchlist = watchlist
    )
    gc()
    best.cv <- min(cross.val$test.mlogloss.mean)
    bestIter <- which(cross.val$test.mlogloss.mean==best.cv)
    
    cat("\n",best.cv,bestIter-1,"\n")
    print(cross.val[bestIter-1])   
}


set.seed(123)
doCV(param, 1500)

# train model
set.seed(123)
nrounds <- 400
xgb.fit <- xgb.train(data = train.matrix,
                     params = param,
                     nrounds = nrounds,
                     num_class = numClasses,
                     verbose = 2
)

predictions <- predict(xgb.fit, data.matrix(test.x))
reshapePred <- t(matrix(predictions, nrow=5, ncol=length(predictions)/5))

sub <- data.frame(cbind(test.id, reshapePred))
names(sub) <- c("ID","Adoption","Died","Euthanasia","Return_to_owner","Transfer")

inputs <- c("nrounds" = nrounds,
            "eta" = param$eta,
            "max_depth" = param$max_depth,
            "subsample" = param$subsample,
            "colsample_bytree" = param$colsample_bytree)

write.csv(sub,
          paste("XGB_",
                paste(as.character(inputs), collapse="_"),
                "_",
                ".csv",sep=""),
          row.names=F)
