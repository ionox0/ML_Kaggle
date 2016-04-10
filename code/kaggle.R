library(dplyr)
library(reshape2)

setwd('~/Desktop/Columbia/ML/kaggle_1')

train = read.csv('data.csv')
train = data.frame(train)
test = read.csv('quiz.csv')

# Remove null values
train = na.omit(train)
# Check for non integers
all(is.numeric(train$label))
# Check for null values
row.has.na <- apply(train, 1, function(x){any(is.na(x))})

numeric_cols = data.frame(train$X59, train$X60, train$label)
d_cor <- as.matrix(cor(numeric_cols))
d_cor_melt <- arrange(melt(d_cor), -abs(value))


dplyr::filter(d_cor_melt, value > .5) # .5 is arbitrary
