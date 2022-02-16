library(Seurat)
library(SeuratData)
library(patchwork)
library(dplyr)
library(Matrix)

train<-read.table('MLR_Project_train.csv', sep = ',', header=TRUE)
test<-read.table('MLR_Project_test.csv', sep = ',', header=TRUE)
train <- train[-c(1)]
test <- test[-c(1)]
train=train[,-67]
test= test[,-67]

train_test = rbind(train, test)
train_test=t(train_test)

meta = c(rep('train', nrow(train)), rep('test', nrow(test)))
meta = as.data.frame(meta)

colnames(meta)='TARGET'
colnames(train_test)= paste('Sample', 1:ncol(train_test), sep='')
rownames(meta)=colnames(train_test)
my_data <- CreateSeuratObject(counts = train_test, meta.data=meta,  min.cells = 3, min.genes = 200, project = 'mydata_optiver')
my_data <- SplitObject(my_data, split.by = 'TARGET')


my_data <- lapply(X = my_data, FUN = function(x) {
  x <- NormalizeData(x)
  x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = 60)
})

features <- SelectIntegrationFeatures(object.list = my_data)

anchors <- FindIntegrationAnchors(object.list = my_data, anchor.features = features)

combined <- IntegrateData(anchorset = anchors)

DefaultAssay(combined) <- "integrated"

combined <- ScaleData(combined, verbose = FALSE)
combined <- RunPCA(combined, npcs = 30, verbose = FALSE)
combined <- RunUMAP(combined, reduction = "pca", dims = 1:30)
combined <- FindNeighbors(combined, reduction = "pca", dims = 1:30)
combined <- FindClusters(combined, resolution = 0.5)

p1 <- DimPlot(combined, reduction = "umap", group.by = "TARGET")
p2 <- DimPlot(combined, reduction = "umap", label = TRUE, repel = TRUE)
p1 + p2
DimPlot(combined, reduction = "umap", split.by = "TARGET")

write.table(combined@assays$integrated@scale.data, file="integrate.csv", sep = ',')
