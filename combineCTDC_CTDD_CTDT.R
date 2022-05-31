# #Get command arguments - species prefix
# #library(protr)
 args<-commandArgs(TRUE)
#
 file1 <- args[1]
#
 file2 <- args[2]

 file3 <- args[3]

 l1<- read.csv(file1)
 l2<- read.csv(file2)
 l3<- read.csv(file3)

 x1 = cbind(l1,l2)
 x1 = cbind(x1,l3)

 write.table(x1,file=" EssentialFeature_collection_CTDC_CTDT_CTDD.csv",quote=TRUE,sep=",")
