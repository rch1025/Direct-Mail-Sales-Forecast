
#########################<프로젝트 보고서>############################
library(caret)
library(reshape)
library(FNN)
library(leaps)
library(forecast)
library(gains)
library(rpart)
library(rpart.plot)
library(neuralnet)
library(lm.beta)
RNGkind(sample.kind = "Rounding")

################## <EDA : 데이터 탐색 및 전처리>####################
setwd("C:/Users/chane/Documents")
retail_df <- read.csv('retail-marketing.csv')
head(retail_df)
colnames(retail_df)
options(scipen=999)
  # 예측변수 : Age, Gender, OwnHome, Married, Location, Salary, Children, History, Catalogs
  # 결과변수 : AmountSpent

  # 변수의 데이터 타입 확인
str(retail_df)
  #(참고) 순위가 있는 범주형 변수 값의 경우 데이터마이닝에서는 이를 연속형 변수로 가정한다.



# <결측치 변환 및 제거>
# history 변수에만 결측치 확인됨.
colSums(is.na(retail_df))  
  # retail-marketing 데이터의 결측치는 history 변수에만 있다. 
  # history 변수는 해당 고객이 이전에 얼마나 구매했었는지를 나타낸다.
  # history 변수의 결측치는 총 303개로 이는 전체 데이터의 3/10을 차지한다. 303개의 결측치를 전부 제거할시 데이터의 손실이
  # 너무 크다고 판단된다. 또한 해당 결측치는 랜덤한 결측치가 아닌 의도적(추론 가능)인 결측치로 보인다.
  # 따라서 결측치들을 k-최근접이웃 알고리즘을 통해 예측하여 대체하였다.

  # 유클리드 거리 계산을 위해 범주형 변수의 이진 가변수 변환
str(retail_df)
  # 회원 번호 변수(Cust_Id)제거
history_omit_df <- retail_df[,-1]
dummy <- as.data.frame(model.matrix(~0+Age, history_omit_df))
dummy1 <- as.data.frame(model.matrix(~0+Gender, history_omit_df))
dummy2 <- as.data.frame(model.matrix(~0+OwnHome, history_omit_df))
dummy3 <- as.data.frame(model.matrix(~0+Married, history_omit_df))
dummy4 <- as.data.frame(model.matrix(~0+Location, history_omit_df))
  # 가변수 생성에서 m-1개가 아닌 m개를 생성하는 이유는 모델에 대한 다양한 범주의 기여도에 불균형을 방지하기 위함이다.
dum_df <- as.data.frame(cbind(dummy,dummy1,dummy2,dummy3,dummy4))
str(history_omit_df)
history_omit_df <- history_omit_df[,c(-1,-2,-3,-4,-5)]
history_omit_df <- cbind(history_omit_df,dum_df)

  # 예측을 위해 결측치가 있는 행들을 따로 두기
omit_df <- history_omit_df[is.na(history_omit_df$History),]
not_omit_df <- history_omit_df[is.na(history_omit_df$History)!=TRUE,]


  # 데이터의 분할
set.seed(3216)
train_index <- createDataPartition(c(1:dim(not_omit_df)[1]), p=0.6, list=FALSE)
training <- not_omit_df[train_index,]
validing <- not_omit_df[-train_index,]

  # 데이터 표준화     
not_omit_df.norm <- not_omit_df
omit_df.norm <- omit_df

norm.values <- preProcess(training[,-3], method=c('center','scale'))
omit_df.norm[,-3] <- predict(norm.values, omit_df.norm[,-3])
training[,-3] <- predict(norm.values, training[,-3])
validing[,-3] <- predict(norm.values, validing[,-3])
not_omit_df.norm[,-3] <- predict(norm.values, not_omit_df[,-3])
str(not_omit_df.norm)


  # 최적의 k 찾기
accuracy.data <- data.frame(k = seq(1, 20, 1), accuracy = rep(0, 20))
knn_pred <- c()
for(i in 1:20){
  knn_pred <- knn(training[,-3],validing[,-3], training[,3], k = i)
  accuracy.data[i, 2] <- confusionMatrix(knn_pred, validing[,3])$overall[1]
}
max(accuracy.data[,2])
accuracy.data
plot(c(1:20), accuracy.data[,2], type = 'b', col = "darkblue",
     main = c("1부터 20까지 K에 대한 정확도"), xlab = "K", ylab = "정확도")
abline(v = which.max(accuracy.data[,2]), lty = 4, col = "darkred", lwd = 2)
  # k=6

# k=6을 사용하여 결측치 예측
class_knn <- knn(training[,-3], omit_df.norm[,-3], training[,3], k=3)
summary(class_knn)
retail_df[is.na(retail_df$History),9] <- class_knn
colSums(is.na(retail_df))




####################### <결과 변수 전처리>######################
  # AmountSpent변수는 고객이 1년동안 해당 업체에서 제품을 구매하는 데 사용한 총금액이다.
  # 종속변수를 보면 데이터의 편차가 매우 크다. 
  # 또한 데이터의 분포가 왼쪽으로 치우쳐져 있음을 알 수 있다.
  # 데이터의 재표현을 통해 결과변수 AmountSpent를 정규분포와 같이 만들어 분산의 이질성을 줄여주었다.

summary(retail_df$AmountSpent)
ggplot(data = retail_df, aes(AmountSpent))+
  geom_density(fill='darkgrey',col='purple')+
  scale_y_continuous(breaks=NULL)

retail_df$AmountSpent <- log(retail_df$AmountSpent)
summary(retail_df$AmountSpent)
ggplot(data = retail_df, aes(AmountSpent))+
  geom_density(fill='darkgrey',col='purple')+
  scale_y_continuous(breaks=NULL)



########################## <예측 변수 전처리>#########################
str(retail_df)
  ### [Cust_Id 변수] ###
# 결과변수 AmountSpent를 예측하는데에 의미가 없는 변수로 제거한다.
unique(retail_df$Cust_Id)
retail_df <- retail_df[,c(-1)]



  ### [Age 변수] ### 
# 빈도 확인
table(retail_df$Age)
ggplot(data = retail_df, aes(Age))+
  geom_bar(aes(fill=Age), width = 0.7)
# 결과변수와 Age 시각화
ggplot(data = retail_df, aes(x = as.factor(Age), y = AmountSpent,fill=as.factor(Age)))+
  geom_boxplot(fill='white',width = 0.5)+
  geom_jitter(aes(col=as.factor(Age), alpha=I(0.3)))

# 모델에서의 보다 좋은 성능을 위해 문자형 변수를 숫자형 변수로 전환
retail_df$Age <- ifelse(retail_df$Age=='Young',0,
                 ifelse(retail_df$Age=='Middle',1,2))

# Age 변수는 고객의 나이를 3개의 구간으로 나눈 변수로 [Young, Middle, Old]를 가진 범주형 데이터이다.
# 고객은 Middle층이 가장 많고 다음은 Young 그리고 Old 순이다.
# 모든 나이대에서 이상치가 존재하는 것으로 보인다.
# 전체적으로 Young한 사람들보다 나이가 많은 사람들이 구매금액이 더 높은 것을 알 수 있다.


  ### [Gender 변수] ###
# 빈도 확인 : 고객의 성별이 여성과 남성, 506명과 494명으로 비슷하다는 것을 알 수 있다.
table(retail_df$Gender)
ggplot(data = retail_df, aes(Gender))+
  geom_bar(fill=c('slategrey','royalblue'), width=0.6)


# 종속변수와 Gender 변수의 Boxplot
ggplot(data = retail_df, aes(x = Gender, y = AmountSpent))+
  geom_boxplot(width = 0.5)+
  geom_jitter(aes(col=Gender, alpha=I(0.4)))
# boxplot으로 볼때 남성의 IQR이 여성의 IQR 보다 높고 중앙값이 더 높다는 점, 그리고 이상치가 더욱 많다는 점으로 고려할때 
# 남성이 여성보다 더 많은 금액을 사용했다는 것을 파악할 수 있다.


    ### OwnHome 변수 ###
# OwnHome 변수는 고객의 집이 본인 소유인지 아니면 빌린 것인지를 나타내는 변수이다.
table(retail_df$OwnHome)
ggplot(data=retail_df, aes(OwnHome))+
  geom_bar(aes(fill=OwnHome), width=0.7)

# 종속변수와 OwnHome 변수의 Boxplot
ggplot(data = retail_df, aes(x = OwnHome, y = AmountSpent))+
  geom_boxplot( width = 0.6)+
  geom_jitter(aes(col=OwnHome, alpha=I(0.4)))
# 중앙값의 위치, 사분위수 범위 등을 고려할 때 집을 소유하고 있는 사람들이 집을 소유하지 않은(Rent한) 사람들 보다 구매를 더 많이 한다는 것을 알 수 있다.


    ### Married 변수 ###
table(retail_df$Married)
ggplot(data=retail_df, aes(Married))+
  geom_bar(aes(fill=Married), width=0.7)

# 종속변수와 Married 변수의 Boxplot
ggplot(data = retail_df, aes(x = Married, y = AmountSpent))+
  geom_boxplot( width = 0.6)+
  geom_jitter(aes(col=Married, alpha=I(0.4)))
# 결혼한 고객이 싱글인 고객 보다 구매금액이 더 높다.


    ### Location 변수 ###
table(retail_df$Location)
ggplot(data=retail_df, aes(Location))+
  geom_bar(aes(fill=Location), width = 0.6)

# 종속변수와 Location 변수의 Boxplot
ggplot(data = retail_df, aes(x = Location, y = AmountSpent))+
  geom_boxplot( fill = 'slategrey', width = 0.6)+
  geom_jitter(aes(col=Location, alpha=I(0.4)))
# Location 변수는 해당 고객의 집으로 부터 비슷한 제품을 파는 매장의 거리를 의미한다.
# 비슷한 제품을 파는 매장과 거리가 가까운 고객의 빈도수가 먼 고객보다 두 배 이상 많다.
# 전체적으로 비슷한 제품을 파는 매장과 거리가 먼 고객이 더 적지만 구매금액은 더 높다는 것을 알 수 있다.


    ### Salary 변수 ###
summary(retail_df$Salary)
ggplot(data = retail_df, aes(Salary))+
  geom_density(fill='darkslateblue')+
  theme(axis.text.y=element_blank())

# 종속변수와 Salary 변수의 관계 시각화
ggplot(retail_df,aes(x=Salary, y=AmountSpent))+
  geom_point(col='darkslateblue')

# 밀도 곡선을 보면 연봉으로 0~70000달러를 받는 고객이 매우 많은 것으로 보인다.
# 산점도를 통해 두 변수의 관계를 보면 대개 연봉이 높아질수록 구매금액도 증가하는 것을 알 수 있다.



    ### Children 변수 ###
table(retail_df$Children)
class(retail_df$Children)
Children_num <- as.factor(retail_df$Children)
ggplot(retail_df, aes(as.factor(Children)))+
  geom_bar(aes(fill=Children_num))
# 결과변수와 Children 변수
ggplot(data = retail_df, aes(as.factor(Children),AmountSpent))+
  geom_boxplot()+
  geom_jitter(aes(col=Children_num, alpha=I(0.4)))
# 자녀가 없는 고객들이 주로 구매를 하는 것으로 보인다.
# 또한 종속변수와의 그림을 보면 자녀 수가 많을 수록 구매금액이 적어진다.
# 즉, 자녀가 적은 고객의 구매금액도 많은 것을 알 수있다.


    ### History 변수 ###
  # History 변수는 과거에 고객이 해당 업체에 얼마나 소비했는가를 의미한다.
table(retail_df$History)
class(retail_df$History)
ggplot(data=retail_df, aes(History))+
  geom_bar(aes(fill=History), width=0.7)
# 종속변수와 History 변수 
ggplot(data = retail_df, aes(History,AmountSpent))+
  geom_boxplot()+
  geom_jitter(aes(col=History, alpha=I(0.3)))
  
# 과거에 소비를 많이한 고객이나 적게한 고개이나 인원수가 거의 비슷하다.
# 과거 구매이력이 많던 사람들이 적던 사람들 보다 구매금액이 더욱 많은 것이 보인다.

# 과거 구매이력은 High, Medium, Low로 위아래가 있다고 판단하여 이를 순서형 변수로 전환.
retail_df$History <- ifelse(retail_df$History=='Low',0,
                     ifelse(retail_df$History=='Medium',1,2))


    ### Catalogs 변수 ###
# 카탈로그는 잡지처럼 생긴 광고물이다.
# 우편 판매의 경우 광고를 카탈로그로 하고 있다. 변수 Catalogs 는 고객에게 보낸 카탈로그 수를 말한다.
table(retail_df$Catalogs)
class(retail_df$Catalogs)
Catalogs_num <- as.factor(retail_df$Catalogs)
ggplot(data=retail_df, aes(as.factor(Catalogs)))+
  geom_bar(aes(fill=Catalogs_num), width = 0.7)
# 카탈로그를 12개 받은 고객이 가장 많다.

# 결과변수와의 시각화
ggplot(data = retail_df, aes(as.factor(Catalogs),AmountSpent))+
  geom_boxplot()+
  geom_jitter(aes(col=Catalogs_num, alpha=I(0.3)))

# 보낸 카탈로그의 수가 많을 수록 구매량이 높다.




# <상관관계>
# 연속형 변수들과 결과변수 간의 상관관계 확인
# 히트맵을 통해 상관관계 표현, 
cor.matrix <- round(cor(retail_df[,c(1,6,7,8,9,10)]),3)
cor.matrix
  # 피벗 테이블 생성
melted.cor.mat <- melt(cor.matrix)
ggplot(melted.cor.mat, aes(x=X1, y=X2,fill=value))+
  geom_tile() +
  geom_text(aes(x=X1, y=X2, label=value))
  # 예측변수에서는  Salary와 History 변수의 상관관계가 0.726,
  # 결과변수에서는 History변수와 결과변수가 0.757, Salary변수와 결과변수가 0.7이다.




# 2. 모델링
####################### a. 다중선형회귀모형###########################
regression_retail_df <- retail_df

# <이상치 확인>
  # 회귀모형의 경우 이상치에 영향을 많이 받기에 종속변수의 이상치를 확인하여 제거한다.
  
  # 이상치 기준을 두 가지로 생각해 보았다.
    # 1. 1사분위수와 3사분위수에서 1.5*IQR만큼 떨어져 있는 수를 이상치로 판단
    # 이상치 확인을 위해 log를 씌웠던 결과변수를 원래대로 복구
regression_retail_df$AmountSpent <- exp(regression_retail_df$AmountSpent)
  # 1사분위수와 3사분위수 구하기
regression_qnt <- quantile(regression_retail_df$AmountSpent, probs=c(0.25, 0.75))
  # 1.5*IQR
regression_iqt <- 1.5*IQR(regression_retail_df$AmountSpent)
regression_retail_df$AmountSpent[regression_retail_df$AmountSpent<(regression_qnt[1]-regression_iqt)]=NA
regression_retail_df$AmountSpent[regression_retail_df$AmountSpent>(regression_qnt[2]+regression_iqt)]=NA
colSums(is.na(regression_retail_df))
regression_retail_df <- na.omit(regression_retail_df)
dim(regression_retail_df)
  #총 5개의 데이터를 이상치로 판단하여 제거하였다.

    # 2. 주관적인 판단으로 자신의 연봉의 절반 이상을 제품을 구매하는 데에 사용하지 않는다고 판단하여 이를 이상치로 설정
summary(regression_retail_df$AmountSpent)
IS_outlier <- c()
for (i in 1:dim(regression_retail_df)[1]){
  IS_outlier <- c(IS_outlier,ifelse(regression_retail_df[i,c(6)]*(1/2) < regression_retail_df[i,10], TRUE, FALSE))
}
table(IS_outlier)
  # 위 조건에 해당되는 고객은 없는 것을 알 수 있다.
regression_retail_df$AmountSpent <- log(regression_retail_df$AmountSpent)
  # 결과변수 log 다시 취하기


# <가변수화>
Gender <- as.data.frame(model.matrix(~0+Gender, data = regression_retail_df))
OwnHome <- as.data.frame(model.matrix(~0+OwnHome, data = regression_retail_df))
Married <- as.data.frame(model.matrix(~0+Married, data = regression_retail_df))
Location <- as.data.frame(model.matrix(~0+Location, data = regression_retail_df))
regression_retail_df <- cbind(regression_retail_df[,-c(2,3,4,5)], Gender[,-2], OwnHome[,-2], Married[,-2], Location[,-2])
names(regression_retail_df)[7:10] <- c('Gender_Female','OwnHome_Rent','Married_Single','Location_Far')
str(regression_retail_df)

# <데이터 분할>
set.seed(2020)
re_train_index <- createDataPartition(c(1:dim(regression_retail_df)[1]), p=0.6, list=FALSE)
regression_re_train_data <- regression_retail_df[re_train_index,]
regression_re_valid_data <- regression_retail_df[-re_train_index,]


  # 변수의 개수는 모델의 성능에 영향을 미치기에 평가지표(adjR^2)를 사용하여 적절한 변수 찾기

retail_exhaustive <- regsubsets(AmountSpent~., data = regression_re_train_data, nbest=1, method='exhaustive')
retail_sum <- summary(retail_exhaustive)
result <- with(retail_sum, round(cbind(which,adjr2,cp,bic),5))
result
max_R <- which.max(retail_sum$adjr2)
max_R
  
  # 변수 확인
names(coef(retail_exhaustive, max_R))
colnames(regression_retail_df)

  # 변수 선택
select_var <- c(7,8)
regression_select_train <- regression_re_train_data[,-select_var]
regression_select_valid <- regression_re_valid_data[,-select_var]
retail_lm <- lm(AmountSpent~., data = regression_select_train)
retail_lm_train_pred <- predict(retail_lm, regression_select_train)
retail_lm_pred <- predict(retail_lm, regression_select_valid)

# 예측값과 실제값 시각화
plot(1:nrow(regression_select_valid),exp(regression_select_valid$AmountSpent),type='l')
lines(1:nrow(regression_select_valid),exp(retail_lm_pred),col='red')


# 로그를 씌웠던 결과변수에 exp()를 취해서 원래대로 만든 후 정확도를 확인하였다.
# 정확도 확인 : RMSE기준
accuracy(exp(retail_lm_pred), exp(regression_select_valid$AmountSpent))
accuracy(exp(retail_lm_train_pred), exp(regression_select_train$AmountSpent))

# 향상차트를 통한 정확도 확인
regression_gain <- gains(exp(regression_select_valid$AmountSpent), exp(retail_lm_pred))
# 향상차트
plot(c(0,regression_gain$cume.pct.of.total*sum(exp(regression_select_valid$AmountSpent)))~c(0,regression_gain$cume.obs),
     xlab="<cases>", ylab="Cumulative AmountSpent", main="model_lift", type="l")
lines(c(0,sum(exp(regression_select_valid$AmountSpent)))~c(0,dim(regression_select_valid)[1]), col="red", lty=2, lwd = 2)




####################### b. 의사결정나무 모델링###########################
  # 회귀트리의 단점
    #단일 분류 회귀 나무의 경우 분산이 높아서 예측이 불안정한 경우가 있다.
    #또한 분산이 높아 예측 정확도가 좋지 않다.
str(retail_df)
rpart_df <- retail_df
# 데이터 분할
set.seed(2020)
re_train_index <- createDataPartition(c(1:dim(rpart_df)[1]), p=0.6, list=FALSE)
rpart_training <- rpart_df[re_train_index,]
rpart_validing <- rpart_df[-re_train_index,]


# 가지치기 
set.seed(2000)
tree_cv_retail <- rpart(AmountSpent~., data = rpart_training, method = 'anova', cp=0.00001, minsplit=30, xval = 5)
  # 학습데이터 개수의 5%인 30개를 minsplit으로 사용하였고, 교차검증 횟수를 5로 지정하였다.
printcp(tree_cv_retail)
0.22042+0.014833

  # 0.235253보다 낮은 xerror를 갖는 것 중에서 가장 작은 분할 수를 갖는 것은 13번째 cp값이다.
set.seed(32161384)
pruned_retail <- prune(tree_cv_retail, cp= 0.00396572 )
prp(pruned_retail, extra = 1, type=2, under=T, varlen = -12)


# 성능평가
  # 회귀 나무 모델은 type='vector'로 지정해 주어야 한다.
rpart_train_pred <- predict(pruned_retail, rpart_training[,-10], type='vector')
rpart_valid_pred <- predict(pruned_retail, rpart_validing[,-10], type='vector')
unique(exp(rpart_train_pred))
unique(exp(rpart_valid_pred))

rpart_train_acc <- accuracy(exp(rpart_train_pred), exp(rpart_training$AmountSpent))
rpart_valid_acc <- accuracy(exp(rpart_valid_pred), exp(rpart_validing$AmountSpent))
rpart_train_acc[2]; rpart_valid_acc[2]
# RMSE=589.3424로 다중회귀모형 보다 더 떨어지는 성능을 보여준다.

  # 검증세트에서 예측값과 실제값 간의 상관관계
cor(exp(rpart_valid_pred), exp(rpart_validing$AmountSpent))
plot(exp(rpart_valid_pred), exp(rpart_validing$AmountSpent))

# 향상차트를 통한 정확도 확인
rpart_gain <- gains(exp(rpart_validing$AmountSpent), exp(rpart_valid_pred))
# 첫번째 모델 향상차트
plot(c(0,rpart_gain$cume.pct.of.total*sum(exp(rpart_validing$AmountSpent)))~c(0,rpart_gain$cume.obs),
     xlab="<cases>", ylab="Cumulative AmountSpent", main="model_lift", type="l")
lines(c(0,sum(exp(rpart_validing$AmountSpent)))~c(0,dim(rpart_validing)[1]), col="red", lty=2, lwd = 2)



########################## c. knn ##############################
  # 분류와 예측 모두에 쓰이는 k-최근접이웃 알고리즘을 적용.
str(retail_df)
knn_retail_df <- retail_df
dummy1 <- as.data.frame(model.matrix(~0+Gender, knn_retail_df))
dummy2 <- as.data.frame(model.matrix(~0+OwnHome, knn_retail_df))
dummy3 <- as.data.frame(model.matrix(~0+Married, knn_retail_df))
dummy4 <- as.data.frame(model.matrix(~0+Location, knn_retail_df))
# 가변수 생성에서 m-1개가 아닌 m개를 생성하는 이유는 모델에 대한 다양한 범주의 기여도에 불균형을 방지하기 위함이다.
dumdum_df <- as.data.frame(cbind(dummy1,dummy2,dummy3,dummy4))
str(dumdum_df)
knn_retail_df <- knn_retail_df[,c(-2,-3,-4,-5)]
knn_retail_df <- cbind(knn_retail_df,dumdum_df)
str(knn_retail_df)


# 데이터의 분할
set.seed(2020)
train_index <- createDataPartition(c(1:dim(knn_retail_df)[1]), p=0.5, list=FALSE)
knn_training <- knn_retail_df[train_index,]
set.seed(2020)
valid_index <- sample(setdiff(rownames(knn_retail_df), train_index), dim(knn_retail_df)[1]*0.3)
knn_validing <- knn_retail_df[valid_index,]
test_index <- setdiff(rownames(knn_retail_df),union(train_index,valid_index))
knn_testing <- knn_retail_df[test_index,]


# 데이터 표준화     
knn_training.norm <- knn_training
knn_validing.norm <- knn_validing
knn_testing.norm <- knn_testing
str(knn_training.norm)
knn_norm.values <- preProcess(knn_training[,-6], method=c('center','scale'))
knn_training.norm[,-6] <- predict(knn_norm.values, knn_training[,-6])
knn_validing.norm[,-6] <- predict(knn_norm.values, knn_validing[,-6])
knn_testing.norm[,-6] <- predict(knn_norm.values, knn_testing[,-6])



# 최적의 k 찾기
# library(FNN)
knn_accuracy.data <- data.frame(k = seq(1, 40, 1), accuracy = rep(0, 40))
knn_pred <- c()
for(i in 1:40){
  knn_pred <- knn(knn_training.norm[,-6],knn_validing.norm[,-6], knn_training.norm[,6], k = i)
  knn_pred <- as.vector(knn_pred)
  knn_pred <- as.numeric(knn_pred)
  knn_accuracy.data[i, 2] <- accuracy(knn_pred[1:length(knn_pred)], knn_validing.norm$AmountSpent)[2]
}
knn_accuracy.data

  # 시각화
plot(c(1:40), knn_accuracy.data[,2], type = 'b', col = "darkblue",
     main = c("K에 대한 RMSE"), xlab = "K", ylab = "RMSE", pch=19)
abline(v = which.min(knn_accuracy.data[,2]), lty = 3, col = "darkred", lwd = 2)
  # k=2일 때 RMSE가 가장 낮다.



# 성능평가
    # 과적합 확인
knn_retail_train <- knn(knn_training.norm[,-6], knn_training.norm[,-6], knn_training.norm[,6], k=2)
accuracy(exp(as.numeric(as.vector(knn_retail_train))), exp(knn_training.norm$AmountSpent))
    # 검증 세트를 최적의 k를 찾는 데 사용 했으니 평가 세트를 사용하여 성능을 평가한다.
knn_retail_final <- knn(knn_training.norm[,-6], knn_testing.norm[,-6], knn_training.norm[,6], k=2)
accuracy(exp(as.numeric(as.vector(knn_retail_final))), exp(knn_testing.norm$AmountSpent))

# 향상차트를 통한 정확도 확인
knn_gain <- gains(exp(knn_testing.norm$AmountSpent), exp(as.numeric(as.vector(knn_retail_final))))
# 향상차트
plot(c(0,knn_gain$cume.pct.of.total*sum(exp(knn_testing.norm$AmountSpent)))~c(0,knn_gain$cume.obs),
     xlab="<cases>", ylab="Cumulative AmountSpent", main="model_lift", type="l")
lines(c(0,sum(exp(knn_testing.norm$AmountSpent)))~c(0,dim(knn_testing.norm)[1]), col="red", lty=2, lwd = 2)

610.0521-481.2988

######################## d. 신경망################################
nnet_retail <- retail_df
str(nnet_retail)

# <가변수화>
Gender <- as.data.frame(model.matrix(~0+Gender, data = nnet_retail))
OwnHome <- as.data.frame(model.matrix(~0+OwnHome, data = nnet_retail))
Married <- as.data.frame(model.matrix(~0+Married, data = nnet_retail))
Location <- as.data.frame(model.matrix(~0+Location, data = nnet_retail))
nnet_retail <- cbind(nnet_retail[,-c(2,3,4,5)], Gender[-2], OwnHome[-2], Married[-2], Location[-2])
str(nnet_retail)


# <데이터 표준화>
retail_nnet_norm <- nnet_retail
normal_fun <- function(data){(data - min(data)) / (max(data) - min(data))}
retail_nnet_norm[,c(1:6)] <- as.data.frame(lapply(nnet_retail[,c(1:6)], normal_fun))
summary(retail_nnet_norm)


# <데이터의 분할>
set.seed(2020)
train_index <- createDataPartition(c(1:dim(nnet_retail)[1]), p=0.5, list=FALSE)
nnet_training <- retail_nnet_norm[train_index,]
set.seed(2020)
valid_index <- sample(setdiff(rownames(nnet_retail), train_index), dim(nnet_retail)[1]*0.3)
nnet_validing <- retail_nnet_norm[valid_index,]
test_index <- setdiff(rownames(nnet_retail),union(train_index,valid_index))
nnet_testing <- retail_nnet_norm[test_index,]


  # 예측값에 대한 비정규화 함수
unnormal_fun <- function(x){
  x*(max(nnet_retail$AmountSpent)-min(nnet_retail$AmountSpent))+min(nnet_retail$AmountSpent)
}

# <은닉 노드의 개수를 찾기>
real_valid_AmountSpent <- as.numeric(lapply(nnet_validing$AmountSpent,unnormal_fun))
RMSE <- c()
set.seed(32161384)
for (x in c(1:10)){
  neuralnet_retail <- neuralnet(AmountSpent ~ Age + Salary + Children + History + Catalogs
                                + GenderFemale + OwnHomeOwn + 
                                  MarriedMarried+LocationClose, data = nnet_training, hidden=x, err.fct = 'sse',act.fct = 'logistic',stepmax = 1e7, linear.output = T)
  nnet_validing.prediction <- compute(neuralnet_retail, nnet_validing[,-6])
  nnet_valid_pred <- as.numeric(lapply(nnet_validing.prediction$net,unnormal_fun))
  # 은닉 노드 개수의 변화에 따른 RMSE
  RMSE <- c(RMSE,accuracy(exp(nnet_valid_pred), exp(real_valid_AmountSpent))[2])
}
cat(RMSE,sep = " // ")

# 최종모델(은닉노드의 개수 = 7)
set.seed(32161384)
neuralnet_retail_7 <- neuralnet(AmountSpent ~ Age + Salary + Children + History + Catalogs
                              + GenderFemale + OwnHomeOwn + 
                                MarriedMarried+LocationClose, data = nnet_training, hidden=7, err.fct = 'sse',act.fct = 'logistic',stepmax = 1e7, linear.output = T)
plot(neuralnet_retail_7)

# <학습 데이터 성능평가 : 과적합 여부 확인>
nnet_training.prediction <- compute(neuralnet_retail_7, nnet_training[,-6])
nnet_train_pred <- as.numeric(lapply(nnet_training.prediction$net,unnormal_fun))
nnet_train_AmountSpent <- as.numeric(lapply(nnet_training$AmountSpent,unnormal_fun))
accuracy(exp(nnet_train_pred),exp(nnet_train_AmountSpent))

# <평가 데이터에 대한 성능평가>
  # 최적의 은닉노드의 수를 찾는 데에 검증세트가 사용되었으므로 아예 사용되지 않은 평가 세트를 사용하여 성능을 평가한다.
nnet_testing.prediction <- compute(neuralnet_retail_7, nnet_testing[,-6])
nnet_test_pred <- as.numeric(lapply(nnet_testing.prediction$net,unnormal_fun))
real_test_AmountSpent <- as.numeric(lapply(nnet_testing$AmountSpent,unnormal_fun))
accuracy(exp(nnet_test_pred),exp(real_test_AmountSpent))

# 향상차트
nnet_gain <- gains(exp(real_test_AmountSpent), exp(nnet_test_pred))
# 향상차트
plot(c(0,nnet_gain$cume.pct.of.total*sum(exp(real_test_AmountSpent)))~c(0,nnet_gain$cume.obs),
     xlab="<cases>", ylab="Cumulative AmountSpent", main="model_lift", type="l")
lines(c(0,sum(exp(real_test_AmountSpent)))~c(0,length(real_test_AmountSpent)), col="red", lty=2, lwd = 2)



#######################<각 모델들의 성능 비교>###################
# 1. 다중회귀 알고리즘
regression_accuracy <- accuracy(exp(retail_lm_pred), exp(regression_select_valid$AmountSpent))[2]
# 2. 의사결정나무 알고리즘
rpart_accuracy <- accuracy(exp(rpart_valid_pred), exp(rpart_validing$AmountSpent))[2]
# 3. knn 알고리즘
knn_accuracy <- accuracy(exp(as.numeric(as.vector(knn_retail_final))), exp(knn_testing.norm$AmountSpent))[2]
# 4. 신경망 알고리즘
neuralnet_accuracy <- accuracy(exp(nnet_test_pred),exp(real_test_AmountSpent))[2]
ac_df <- as.data.frame(cbind(regression_accuracy, rpart_accuracy, knn_accuracy, neuralnet_accuracy))
ac_df


    # 향상차트 비교 #
par(mfrow=c(1,1))
# 1. 다중회귀 알고리즘 향상차트
plot(c(0,regression_gain$cume.pct.of.total*sum(exp(regression_select_valid$AmountSpent)))~c(0,regression_gain$cume.obs),
     xlab="<cases>", ylab="Cumulative AmountSpent", main="<regression_model_lift>", type="l")
lines(c(0,sum(exp(regression_select_valid$AmountSpent)))~c(0,dim(regression_select_valid)[1]), col="red", lty=2, lwd = 2)


# 2. 의사결정나무 알고리즘 향상차트
plot(c(0,rpart_gain$cume.pct.of.total*sum(exp(rpart_validing$AmountSpent)))~c(0,rpart_gain$cume.obs),
     xlab="<cases>", ylab="Cumulative AmountSpent", main="<tree_model_lift>", type="l")
lines(c(0,sum(exp(rpart_validing$AmountSpent)))~c(0,dim(rpart_validing)[1]), col="red", lty=2, lwd = 2)

# 3. knn 알고리즘 향상차트
plot(c(0,knn_gain$cume.pct.of.total*sum(exp(knn_testing.norm$AmountSpent)))~c(0,knn_gain$cume.obs),
     xlab="<cases>", ylab="Cumulative AmountSpent", main="<knn_model_lift>", type="l")
lines(c(0,sum(exp(knn_testing.norm$AmountSpent)))~c(0,dim(knn_testing.norm)[1]), col="red", lty=2, lwd = 2)

# 4. 신경망 알고리즘 향상차트
plot(c(0,nnet_gain$cume.pct.of.total*sum(exp(real_test_AmountSpent)))~c(0,nnet_gain$cume.obs),
     xlab="<cases>", ylab="Cumulative AmountSpent", main="<nnet_model_lift>", type="l")
lines(c(0,sum(exp(real_test_AmountSpent)))~c(0,length(real_test_AmountSpent)), col="red", lty=2, lwd = 2)


# 다중회귀모델에서 가장 영향력 있는 변수 확인
lm.beta(retail_lm)

# 유의점
plot(1:nrow(regression_select_valid),exp(regression_select_valid$AmountSpent),type='l')
lines(1:nrow(regression_select_valid),exp(retail_lm_pred),col='red')

# 차이가 나는 부분 특징 보기
  # 1. 실제값이 2500을 넘는 경우
real_2500 <- exp(regression_select_valid$AmountSpent)[exp(regression_select_valid$AmountSpent)>2500]
real_2500 <- as.vector(real_2500)

exp_regression_select_valid <- round(exp(regression_select_valid$AmountSpent))
for (i in c(1:length(real_2500))){
  print(regression_select_valid[exp_regression_select_valid==round(real_2500[i],0),])
}

  # 2. 220~230번째 데이터
real <- exp(regression_select_valid$AmountSpent[220:230])
pred <- exp(retail_lm_pred[220:230])
cbind(real,pred)
real_gap <- c(2784,2491,1177,1487)

for (i in c(1:length(real_gap))){
  print(regression_select_valid[exp_regression_select_valid==real_gap[i],])
}

# 잔차에 대한 히스토그램
all_residuals <- exp_regression_select_valid-exp(retail_lm_pred)
p_residuals<-qplot(x =all_residuals, fill=..count.., geom="histogram",bins=50,col=I('black')) 
p_residuals+scale_fill_gradient(low='white', high='#1E3269')
