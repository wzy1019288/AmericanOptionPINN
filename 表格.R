setwd("C:\\Users\\twq\\Desktop\\小组作业\\2d\\steps_sol_10")

dt_02_train <- read.table("seed_2\\log_test.txt",sep="\t",header = TRUE)
dt_04_train <- read.table("seed_4\\log_test.txt",sep="\t",header = TRUE)
dt_06_train <- read.table("seed_6\\log_test.txt",sep="\t",header = TRUE)
dt_08_train <- read.table("seed_8\\log_test.txt",sep="\t",header = TRUE)
dt_10_train <- read.table("seed_10\\log_test.txt",sep="\t",header = TRUE)
dt_12_train <- read.table("seed_12\\log_test.txt",sep="\t",header = TRUE)
dt_14_train <- read.table("seed_14\\log_test.txt",sep="\t",header = TRUE)
dt_16_train <- read.table("seed_16\\log_test.txt",sep="\t",header = TRUE)
dt_18_train <- read.table("seed_18\\log_test.txt",sep="\t",header = TRUE)
dt_20_train <- read.table("seed_20\\log_test.txt",sep="\t",header = TRUE)

tmp <- data.frame(seed=2,dt_02_train[which(dt_02_train$Epoch==max(dt_02_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- tmp
tmp <- data.frame(seed=4,dt_04_train[which(dt_04_train$Epoch==max(dt_04_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)
tmp <- data.frame(seed=6,dt_06_train[which(dt_06_train$Epoch==max(dt_06_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)
tmp <- data.frame(seed=8,dt_08_train[which(dt_08_train$Epoch==max(dt_08_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)
tmp <- data.frame(seed=10,dt_10_train[which(dt_10_train$Epoch==max(dt_10_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)
tmp <- data.frame(seed=12,dt_12_train[which(dt_12_train$Epoch==max(dt_12_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)
tmp <- data.frame(seed=14,dt_14_train[which(dt_14_train$Epoch==max(dt_14_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)
tmp <- data.frame(seed=16,dt_16_train[which(dt_16_train$Epoch==max(dt_16_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)
tmp <- data.frame(seed=18,dt_18_train[which(dt_18_train$Epoch==max(dt_18_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)
tmp <- data.frame(seed=20,dt_20_train[which(dt_20_train$Epoch==max(dt_20_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)


write.csv(cmb_train,"cmb_train2.csv")

#########################
setwd("C:\\Users\\twq\\Desktop\\小组作业\\1d\\steps_sol_20")

dt_02_train <- read.table("seed_2\\log_test.txt",sep="\t",header = TRUE)
dt_04_train <- read.table("seed_4\\log_test.txt",sep="\t",header = TRUE)
dt_06_train <- read.table("seed_6\\log_test.txt",sep="\t",header = TRUE)
dt_08_train <- read.table("seed_8\\log_test.txt",sep="\t",header = TRUE)
dt_10_train <- read.table("seed_10\\log_test.txt",sep="\t",header = TRUE)
dt_12_train <- read.table("seed_12\\log_test.txt",sep="\t",header = TRUE)
dt_14_train <- read.table("seed_14\\log_test.txt",sep="\t",header = TRUE)
dt_16_train <- read.table("seed_16\\log_test.txt",sep="\t",header = TRUE)
dt_18_train <- read.table("seed_18\\log_test.txt",sep="\t",header = TRUE)
dt_20_train <- read.table("seed_20\\log_test.txt",sep="\t",header = TRUE)

tmp <- data.frame(seed=2,dt_02_train[which(dt_02_train$Epoch==max(dt_02_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- tmp
tmp <- data.frame(seed=4,dt_04_train[which(dt_04_train$Epoch==max(dt_04_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)
tmp <- data.frame(seed=6,dt_06_train[which(dt_06_train$Epoch==max(dt_06_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)
tmp <- data.frame(seed=8,dt_08_train[which(dt_08_train$Epoch==max(dt_08_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)
tmp <- data.frame(seed=10,dt_10_train[which(dt_10_train$Epoch==max(dt_10_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)
tmp <- data.frame(seed=12,dt_12_train[which(dt_12_train$Epoch==max(dt_12_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)
tmp <- data.frame(seed=14,dt_14_train[which(dt_14_train$Epoch==max(dt_14_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)
tmp <- data.frame(seed=16,dt_16_train[which(dt_16_train$Epoch==max(dt_16_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)
tmp <- data.frame(seed=18,dt_18_train[which(dt_18_train$Epoch==max(dt_18_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)
tmp <- data.frame(seed=20,dt_20_train[which(dt_20_train$Epoch==max(dt_20_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)


write.csv(cmb_train,"cmb_train1d.csv")

#########

setwd("C:\\Users\\twq\\Desktop\\小组作业\\3d\\steps_sol_10")

dt_02_train <- read.table("seed_2\\log_test.txt",sep="\t",header = TRUE)
dt_04_train <- read.table("seed_4\\log_test.txt",sep="\t",header = TRUE)
dt_06_train <- read.table("seed_6\\log_test.txt",sep="\t",header = TRUE)
dt_08_train <- read.table("seed_8\\log_test.txt",sep="\t",header = TRUE)
dt_10_train <- read.table("seed_10\\log_test.txt",sep="\t",header = TRUE)
dt_12_train <- read.table("seed_12\\log_test.txt",sep="\t",header = TRUE)
dt_14_train <- read.table("seed_14\\log_test.txt",sep="\t",header = TRUE)
dt_16_train <- read.table("seed_16\\log_test.txt",sep="\t",header = TRUE)
dt_18_train <- read.table("seed_18\\log_test.txt",sep="\t",header = TRUE)
dt_20_train <- read.table("seed_20\\log_test.txt",sep="\t",header = TRUE)

tmp <- data.frame(seed=2,dt_02_train[which(dt_02_train$Epoch==max(dt_02_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- tmp
tmp <- data.frame(seed=4,dt_04_train[which(dt_04_train$Epoch==max(dt_04_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)
tmp <- data.frame(seed=6,dt_06_train[which(dt_06_train$Epoch==max(dt_06_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)
tmp <- data.frame(seed=8,dt_08_train[which(dt_08_train$Epoch==max(dt_08_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)
tmp <- data.frame(seed=10,dt_10_train[which(dt_10_train$Epoch==max(dt_10_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)
tmp <- data.frame(seed=12,dt_12_train[which(dt_12_train$Epoch==max(dt_12_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)
tmp <- data.frame(seed=14,dt_14_train[which(dt_14_train$Epoch==max(dt_14_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)
tmp <- data.frame(seed=16,dt_16_train[which(dt_16_train$Epoch==max(dt_16_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)
tmp <- data.frame(seed=18,dt_18_train[which(dt_18_train$Epoch==max(dt_18_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)
tmp <- data.frame(seed=20,dt_20_train[which(dt_20_train$Epoch==max(dt_20_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)


write.csv(cmb_train,"cmb_train3d.csv")


#########

setwd("C:\\Users\\twq\\Desktop\\小组作业\\4d\\steps_sol_10")

dt_02_train <- read.table("seed_2\\log_test.txt",sep="\t",header = TRUE)
dt_04_train <- read.table("seed_4\\log_test.txt",sep="\t",header = TRUE)
dt_06_train <- read.table("seed_6\\log_test.txt",sep="\t",header = TRUE)
dt_08_train <- read.table("seed_8\\log_test.txt",sep="\t",header = TRUE)
dt_10_train <- read.table("seed_10\\log_test.txt",sep="\t",header = TRUE)
dt_12_train <- read.table("seed_12\\log_test.txt",sep="\t",header = TRUE)
dt_14_train <- read.table("seed_14\\log_test.txt",sep="\t",header = TRUE)
dt_16_train <- read.table("seed_16\\log_test.txt",sep="\t",header = TRUE)
dt_18_train <- read.table("seed_18\\log_test.txt",sep="\t",header = TRUE)
dt_20_train <- read.table("seed_20\\log_test.txt",sep="\t",header = TRUE)

tmp <- data.frame(seed=2,dt_02_train[which(dt_02_train$Epoch==max(dt_02_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- tmp
tmp <- data.frame(seed=4,dt_04_train[which(dt_04_train$Epoch==max(dt_04_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)
tmp <- data.frame(seed=6,dt_06_train[which(dt_06_train$Epoch==max(dt_06_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)
tmp <- data.frame(seed=8,dt_08_train[which(dt_08_train$Epoch==max(dt_08_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)
tmp <- data.frame(seed=10,dt_10_train[which(dt_10_train$Epoch==max(dt_10_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)
tmp <- data.frame(seed=12,dt_12_train[which(dt_12_train$Epoch==max(dt_12_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)
tmp <- data.frame(seed=14,dt_14_train[which(dt_14_train$Epoch==max(dt_14_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)
tmp <- data.frame(seed=16,dt_16_train[which(dt_16_train$Epoch==max(dt_16_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)
tmp <- data.frame(seed=18,dt_18_train[which(dt_18_train$Epoch==max(dt_18_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)
tmp <- data.frame(seed=20,dt_20_train[which(dt_20_train$Epoch==max(dt_20_train$Epoch)),c(3,7,8, 10,11,12,5,6)]);cmb_train <- rbind(cmb_train,tmp)


write.csv(cmb_train,"cmb_train4d.csv")
