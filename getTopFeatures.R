library(dplyr)
 args<-commandArgs(TRUE)

 file1 <- args[1]
 file2 <- args[2]
 nombre<- args[3]
 best<- args[4]

 l1<- read.csv(file1)
 l2<- read.csv(file2)

  data1=select(l2, l1[1,1])
  data2=select(l2, l1[2,1])

  result = cbind(data1,data2)
 #
 # print(head(data))
 for (line in 3:nombre){
   data=select(l2, l1[line,1])
   result<-cbind(result,data)
 }
#print(head(result))
  x = paste("Top",nombre,sep = "_")
  name = paste(x,best,sep = "_")
write.table(result,file=paste(name,"csv",sep ="."),quote=TRUE,sep=",")