library(dplyr)
 args<-commandArgs(TRUE)

 file1 <- args[1]

 l1<- read.csv(file1)

  data = select(l1, C3s:Aromo)

write.table(data,file="EssentialFeature_collection_Codonw.csv",quote=TRUE,sep=",")