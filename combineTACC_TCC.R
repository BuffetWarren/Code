# #Get command arguments - species prefix
# #library(protr)
# 
 args<-commandArgs(TRUE)
#
 file1 <- args[1]
#
 file2 <- args[2]

 l1<- read.csv(file1)
 l2<- read.csv(file2)


 x1 = cbind(l1,l2)

 write.table(x1,file=" EssentialFeature_collection_TACC_TCC.csv",quote=TRUE,sep=",")
