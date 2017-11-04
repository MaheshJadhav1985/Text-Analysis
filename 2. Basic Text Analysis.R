# rm(list=ls())                   # Clear workspace

#--------------------------------------------------------#
# Step 0 - Assign Library & define functions             #
#--------------------------------------------------------#

require(text2vec) || install.packages("text2vec")
require(data.table) || install.packages("data.table")
require(stringr) || install.packages("stringr")
require(tm) || install.packages("tm")
require(RWeka) || install.packages("RWeka")
require(tokenizers) || install.packages("tokenizers")
require(slam) || install.packages("slam")
require(wordcloud) || install.packages("wordcloud")
require(ggplot2) || install.packages("ggplot2")

library(text2vec)
library(data.table)
library(stringr)
library(tm)
library(RWeka)
library(tokenizers)
library(slam)
library(wordcloud)
library(ggplot2)

text.clean = function(x)                    # text data
{ require("tm")
  x  =  gsub("<.*?>", " ", x)               # regex for removing HTML tags
  x  =  iconv(x, "latin1", "ASCII", sub="") # Keep only ASCII characters
  x  =  gsub("[^[:alnum:]]", " ", x)        # keep only alpha numeric 
  x  =  tolower(x)                          # convert to lower case characters
  x  =  removeNumbers(x)                    # removing numbers
  x  =  stripWhitespace(x)                  # removing white space
  x  =  gsub("^\\s+|\\s+$", "", x)          # remove leading and trailing white space
  return(x)
}

#--------------------------------------------------------#
# Step 1 - Reading text data                             #
#--------------------------------------------------------#

# temp.text = readLines('C:\\Users\\30773\\Desktop\\Data Science\\cba batch 7\\text classification\\testdata.txt')
temp.text = readLines(file.choose())  # Q25.txt for ice-cream data, india strikes back twitter.csv
head(temp.text, 5)

data = data.frame(id = 1:length(temp.text),  # creating doc IDs if name is not given
                  text = temp.text, 
                  stringsAsFactors = F)
dim(data)

# Read Stopwords list
stpw1 = readLines(file.choose())      # read-in stopwords.txt
# stpw1 = readLines('https://raw.githubusercontent.com/sudhir-voleti/basic-text-analysis-shinyapp/master/data/stopwords.txt')# stopwords list from git
stpw2 = tm::stopwords('english')      # tm package stop word list; tokenizer package has the same name function, hence 'tm::'
stpw3=readLines(file.choose())
comn  = unique(c(stpw1, stpw2,stpw3))         # Union of two list
stopwords = unique(gsub("'"," ",comn))  # final stop word lsit after removing punctuation

x  = text.clean(data$text)                # applying func defined above to pre-process text corpus
x  =  removeWords(x,stopwords)            # removing stopwords created above
x  =  stripWhitespace(x)                  # removing white space
# x  =  stemDocument(x)                   # can stem doc if needed.

# #--------------------------------------------------------#
# #       Creating DTM with tm package                     #
# #--------------------------------------------------------#
# 
# 
# t1 = Sys.time()
# x1 = Corpus(VectorSource(x))          # Create the corpus
# 
# ngram <- function(x1) NGramTokenizer(x1, Weka_control(min = 2, max = 2))  #From Rweka
# 
# tdm0 <- DocumentTermMatrix(x1, control = list(tokenize = ngram,
#                                               # tolower = TRUE, 
#                                               # removePunctuation = TRUE,
#                                               # removeNumbers = TRUE,
#                                               # stopwords = TRUE,
#                                               # stemDocument = TRUE,
#                                               bounds = list(global = c(10, Inf))))     # patience. Takes a minute.
# 
# # Sort bi-gram with decreasing order of freq
# tsum = as.matrix(t(rollup(tdm0, 1, na.rm=TRUE, FUN = sum))) # find sum of freq for each term
# tsum = tsum[order(tsum, decreasing = T),]       #terms in decreasing order of freq
# head(tsum)
# tail(tsum)
# 
# # # select Top 1000 bigrams to unigram
# # if (length(tsum) > 1000) {n = 1000} else {n = length(tsum)}
# # tsum = tsum[1:n]
# 
# #-------------------------------------------------------
# # Code bi-grams as unigram in clean text corpus
# 
# text2 = x
# text2 = paste("",text2,"")
# 
# pb <- txtProgressBar(min = 1, max = (length(tsum)), style = 3) ; i = 0
# 
# for (term in names(tsum)){
#   i = i + 1
#   focal.term = term        
#   replacement.term = gsub(" ", "-", focal.term)
#   text2 = gsub(paste("",focal.term,""),paste("",replacement.term,""), text2)
#   setTxtProgressBar(pb, i)
# }
# 
# dtm = DocumentTermMatrix(Corpus(VectorSource(text2)))
# dim(dtm)
# 
# print(difftime(Sys.time(), t1, units = 'sec'))


#--------------------------------------------------------#
## Step 2: Create DTM using text2vec package             #
#--------------------------------------------------------#

t1 = Sys.time()

tok_fun = word_tokenizer  # using word & not space tokenizers

it_0 = itoken( x,
                  #preprocessor = text.clean,
                  tokenizer = tok_fun,
                  ids = data$id,
                  progressbar = T)

vocab = create_vocabulary(it_0,    #  func collects unique terms & corresponding statistics
                          ngram = c(2L, 2L) #,
                          #stopwords = stopwords
                          )
# length(vocab); str(vocab)     # view what vocab obj is like

pruned_vocab = prune_vocabulary(vocab,  # filters input vocab & throws out v frequent & v infrequent terms
                                term_count_min = 10)
                                # doc_proportion_max = 0.5,
                                # doc_proportion_min = 0.001)

# length(pruned_vocab);  str(pruned_vocab)

vectorizer = vocab_vectorizer(pruned_vocab) #  creates a text vectorizer func used in constructing a dtm/tcm/corpus

dtm_0  = create_dtm(it_0, vectorizer) # high-level function for creating a document-term matrix

# Sort bi-gram with decreasing order of freq
tsum = as.matrix(t(rollup(dtm_0, 1, na.rm=TRUE, FUN = sum))) # find sum of freq for each term
tsum = tsum[order(tsum, decreasing = T),]       # terms in decreasing order of freq
head(tsum)
tail(tsum)

# # select Top 1000 bigrams to unigram
# if (length(tsum) > 1000) {n = 1000} else {n = length(tsum)}
# tsum = tsum[1:n]

#-------------------------------------------------------
# Code bi-grams as unigram in clean text corpus

text2 = x
text2 = paste("",text2,"")

pb <- txtProgressBar(min = 1, max = (length(tsum)), style = 3) ; i = 0

for (term in names(tsum)){
  i = i + 1
  focal.term = gsub("_", " ",term)        # in case dot was word-separator
  replacement.term = term
  text2 = gsub(paste("",focal.term,""),paste("",replacement.term,""), text2)
  setTxtProgressBar(pb, i)
}


it_m = itoken(text2,     # function creates iterators over input objects to vocabularies, corpora, DTM & TCM matrices
              # preprocessor = text.clean,
              tokenizer = tok_fun,
              ids = data$id,
              progressbar = T)

vocab = create_vocabulary(it_m     # vocab func collects unique terms and corresponding statistics
                          # ngram = c(2L, 2L),
                          #stopwords = stopwords
                          )
# length(vocab); str(vocab)     # view what vocab obj is like

pruned_vocab = prune_vocabulary(vocab,
                                term_count_min = 1)
# doc_proportion_max = 0.5,
# doc_proportion_min = 0.001)

vectorizer = vocab_vectorizer(pruned_vocab)

dtm_m  = create_dtm(it_m, vectorizer)
dim(dtm_m)

dtm = as.DocumentTermMatrix(dtm_m, weighting = weightTf)
  a0 = (apply(dtm, 1, sum) > 0)   # build vector to identify non-empty docs
  dtm = dtm[a0,]                  # drop empty docs
 
print(difftime(Sys.time(), t1, units = 'sec'))

# view a sample of the DTM, sorted from most to least frequent tokens 
dtm = dtm[,order(apply(dtm, 2, sum), decreasing = T)]     # sorting dtm's columns in decreasing order of column sums
inspect(dtm[1:5, 1:5])     # inspect() func used to view parts of a DTM object           

#--------------------------------------------------------#
## Step 2a:     # Build word cloud                       #
#--------------------------------------------------------#

#   1- Using Term frequency(tf)             

tst = round(ncol(dtm)/100)  # divide DTM's cols into 100 manageble parts
a = rep(tst,99)
b = cumsum(a);rm(a)
b = c(0,b,ncol(dtm))

ss.col = c(NULL)
for (i in 1:(length(b)-1)) {
  tempdtm = dtm[,(b[i]+1):(b[i+1])]
  s = colSums(as.matrix(tempdtm))
  ss.col = c(ss.col,s)
  print(i)
}

tsum = ss.col
tsum = tsum[order(tsum, decreasing = T)]       #terms in decreasing order of freq
head(tsum)
tail(tsum)

windows()  # New plot window
wordcloud(names(tsum), tsum,     # words, their freqs 
          scale = c(4, 0.5),     # range of word sizes
          10,                     # min.freq of words to consider
          max.words = 200,       # max #words
          colors = brewer.pal(8, "Dark2"))    # Plot results in a word cloud 
title(sub = "Term Frequency - Wordcloud")     # title for the wordcloud display

# plot barchart for top tokens
test = as.data.frame(round(tsum[1:15],0))

windows()  # New plot window
ggplot(test, aes(x = rownames(test), y = test)) + 
       geom_bar(stat = "identity", fill = "Blue") +
       geom_text(aes(label = test), vjust= -0.20) + 
       theme(axis.text.x = element_text(angle = 90, hjust = 1))
       
dev.off() # [graphical] device off / close it down
# -------------------------------------------------------------- #
# step 2b - Using Term frequency inverse document frequency (tfidf)             
# -------------------------------------------------------------- #

require(textir) || install.packages("textir")
library(textir)
dtm.tfidf = tfidf(dtm, normalize=TRUE)

tst = round(ncol(dtm.tfidf)/100)
a = rep(tst, 99)
b = cumsum(a);rm(a)
b = c(0,b,ncol(dtm.tfidf))

ss.col = c(NULL)
for (i in 1:(length(b)-1)) {
  tempdtm = dtm.tfidf[,(b[i]+1):(b[i+1])]
  s = colSums(as.matrix(tempdtm))
  ss.col = c(ss.col,s)
  print(i)
}

tsum = ss.col

tsum = tsum[order(tsum, decreasing = T)]       #terms in decreasing order of freq
head(tsum)
tail(tsum)

windows()  # New plot window
wordcloud(names(tsum), tsum, scale=c(4,0.5),10, max.words=200,colors=brewer.pal(8, "Dark2")) # Plot results in a word cloud 
title(sub = "Term Frequency Inverse Document Frequency - Wordcloud")

as.matrix(tsum[1:20])     #  to see the top few tokens & their IDF scores
(dtm.tfidf)[1:10, 1:10]   # view first 10x10 cells in the DTM under TF IDF.

# plot barchart for top tokens
test = as.data.frame(round(tsum[1:15],0))
windows()  # New plot window
ggplot(test, aes(x = rownames(test), y = test)) + 
  geom_bar(stat = "identity", fill = "red") +
  geom_text(aes(label = test), vjust= -0.20) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
dev.off()

#------------------------------------------------------#
# step 2c - Term Co-occurance Matrix (TCM)             #
#------------------------------------------------------#

vectorizer = vocab_vectorizer(pruned_vocab, 
                              grow_dtm = FALSE, 
                              skip_grams_window = 5L)

tcm = create_tcm(it_m, vectorizer) # func to build a TCM

tcm.mat = as.matrix(tcm)         # use tcm.mat[1:5, 1:5] to view
adj.mat = tcm.mat + t(tcm.mat)   # since adjacency matrices are symmetric

z = order(colSums(adj.mat), decreasing = T)
adj.mat = adj.mat[z,z]

# Plot Simple Term Co-occurance graph
adj = adj.mat[1:30,1:30]

library(igraph)
cog = graph.adjacency(adj, mode = 'undirected')
cog =  simplify(cog)  

cog = delete.vertices(cog, V(cog)[ degree(cog) == 0 ])

windows()
plot(cog)

#-----------------------------------------------------------#
# Step 2d - a cleaned up or 'distilled' COG PLot            #
#-----------------------------------------------------------#

distill.cog = function(mat1, # input TCM ADJ MAT
                       title, # title for the graph
                       s,    # no. of central nodes
                       k1){  # max no. of connections  
  library(igraph)
  a = colSums(mat1) # collect colsums into a vector obj a
  b = order(-a)     # nice syntax for ordering vector in decr order  
  
  mat2 = mat1[b, b]     # order both rows and columns along vector b
  
  diag(mat2) =  0
  
  ## +++ go row by row and find top k adjacencies +++ ##

  wc = NULL
  
  for (i1 in 1:s){ 
    thresh1 = mat2[i1,][order(-mat2[i1, ])[k1]]
    mat2[i1, mat2[i1,] < thresh1] = 0   # neat. didn't need 2 use () in the subset here.
    mat2[i1, mat2[i1,] > 0 ] = 1
    word = names(mat2[i1, mat2[i1,] > 0])
    mat2[(i1+1):nrow(mat2), match(word,colnames(mat2))] = 0
    wc = c(wc,word)
  } # i1 loop ends
  
  
  mat3 = mat2[match(wc, colnames(mat2)), match(wc, colnames(mat2))]
  ord = colnames(mat2)[which(!is.na(match(colnames(mat2), colnames(mat3))))]  # removed any NAs from the list
  mat4 = mat3[match(ord, colnames(mat3)), match(ord, colnames(mat3))]
  graph <- graph.adjacency(mat4, mode = "undirected", weighted=T)    # Create Network object
  graph = simplify(graph) 
  V(graph)$color[1:s] = "green"
  V(graph)$color[(s+1):length(V(graph))] = "pink"

  graph = delete.vertices(graph, V(graph)[ degree(graph) == 0 ]) # delete singletons?
  
  plot(graph, 
       layout = layout.kamada.kawai, 
       main = title)

  } # func ends

windows()
distill.cog(tcm.mat, 'Distilled COG',  10,  5)

## adj.mat and distilled cog for tfidf DTMs ##

adj.mat = t(dtm.tfidf) %*% dtm.tfidf
diag(adj.mat) = 0
a0 = order(apply(adj.mat, 2, sum), decreasing = T)
adj.mat = as.matrix(adj.mat[a0[1:50], a0[1:50]])

windows()
distill.cog(adj.mat, 'Distilled COG',  10,  10)
