#setwd("/home/billdjomkam/Bureau/Testcombincsvfeature")
# #Get command arguments - species prefix
# #library(protr)
# 
 args<-commandArgs(TRUE)
#
 file1 <- args[1]
#
 file2 <- args[2]

 file3 <- args[3]
# 
# #peptide_list_essential=read(file1)
# #coding_list_essential=readFASTA(file2)
#  # l1<- read.csv("EssentialFeature_collection_AAC.csv")
#  # l2<- read.csv("EssentialFeature_collection_APAAC.csv")

 l1<- read.csv(file1)
 l2<- read.csv(file2)
 l3<- read.csv(file3)
#print(l1)

 x1 = cbind(l1,l2)
 x1 = cbind(x1,l3)

 #featureFileSave=paste(featureFile,"Feature_collection_AAC_PAAC.csv",sep="")
 write.table(x1,file=" EssentialFeature_collection_CTDC_CTDT_CTDD.csv",quote=TRUE,sep=",")
# library(dplyr)
#  args<-commandArgs(TRUE)
# 
#  file1 <- args[1]
#  file2 <- args[2]
#  nombre<- args[3]
#  
#  l1<- read.csv(file1)
#  l2<- read.csv(file2)
#  
#   data1=select(l2, l1[1,1])
#   data2=select(l2, l1[2,1])
#   
#   result = cbind(data1,data2)
#  # 
#  # print(head(data))
#  for (line in 2:nombre+1){
#    data=select(l2, l1[line,1])
#    result<-cbind(result,data)
#  }
# #print(head(result))
# write.table(result,file="Top50.csv",quote=TRUE,sep=",")