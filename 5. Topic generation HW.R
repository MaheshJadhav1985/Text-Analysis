rm(list=ls())                   # Clear workspace

#--------------------------------------------------------#
# Step 0 - Assign Library & define functions             #
#--------------------------------------------------------#

library(text2vec)
library(data.table)
library(stringr)
library(tm)
library(RWeka)
library(tokenizers)
library(slam)
library(wordcloud)
library(igraph)
library(maptpx)

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

distill.cog = function(mat1, # input TCM ADJ MAT
                       title, # title for the graph
                       s,    # no. of central nodes
                       k1){  # max no. of connections  
  
  a = colSums(mat1) # collect colsums into a vector obj a
  b = order(-a)     # nice syntax for ordering vector in decr order  
  
  mat2 = mat1[b,b]  #
  
  diag(mat2) =  0
  
  ## +++ go row by row and find top k adjacencies +++ ##
  
  wc = NULL
  
  for (i1 in 1:s){ 
    thresh1 = mat2[i1,][order(-mat2[i1, ])[k1]]
    mat2[i1, mat2[i1,] < thresh1] = 0   # wow. didn't need 2 use () in the subset here.
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
  
  graph = delete.vertices(graph, V(graph)[ degree(graph) == 0 ])
  
  plot(graph, 
       layout = layout.kamada.kawai, 
      main = title)

  } # func ends

#--------------------------------------------------------#
# Step 1 - Reading text data                             #
#--------------------------------------------------------#

search_terms = c('Politics','Physics','Entertainment')

file.cr = read.csv(paste0('F:\\politics_google_search.csv'))

file.mi = read.csv(paste0("F:\\","physics_google_search.csv"))

file.lin = read.csv(paste0("F:\\" ,"Entertainment google search.csv"))

file.cr = file.cr[!is.na(file.cr$text)|file.cr$text != '',]

file.mi = file.mi[!is.na(file.mi$text)|file.mi$text != '',]

file.lin = file.lin[!is.na(file.lin$text)|file.lin$text != '',]

n = min(nrow(file.cr),nrow(file.mi),nrow(file.lin))

data = data.frame(id = 1:n, text1 = file.cr$text[1:n],
                      text2 = file.mi$text[1:n],
                      text3 = file.lin$text[1:n],
                      stringsAsFactors = F)
data$text = paste(data$text1,data$text2,data$text3)

dim(data)

# Read Stopwords list
stpw1 = readLines('https://raw.githubusercontent.com/sudhir-voleti/basic-text-analysis-shinyapp/master/data/stopwords.txt')# stopwords list from git
stpw2 = tm::stopwords('english')               # tm package stop word list; tokenizer package has the same name function

comn  = unique(c(stpw1, stpw2))                 # Union of two list #'solid state chemistry','microeconomics','linguistic'
stopwords = unique(gsub("'"," ",comn))  # final stop word lsit after removing punctuation

x  = text.clean(data$text)             # pre-process text corpus
x  =  removeWords(x,stopwords)            # removing stopwords created above
x  =  stripWhitespace(x)                  # removing white space
# x  =  stemDocument(x)

#--------------------------------------------------------#
####### Create DTM using text2vec package                #
#--------------------------------------------------------#

t1 = Sys.time()

tok_fun = word_tokenizer

it_0 = itoken( x,
               #preprocessor = text.clean,
               tokenizer = tok_fun,
               ids = data$id,
               progressbar = T)

vocab = create_vocabulary(it_0,
                          ngram = c(2L, 2L)
                          #stopwords = stopwords
)

pruned_vocab = prune_vocabulary(vocab,
                                term_count_min = 10)
# doc_proportion_max = 0.5,
# doc_proportion_min = 0.001)

vectorizer = vocab_vectorizer(pruned_vocab)

dtm_0  = create_dtm(it_0, vectorizer)

# Sort bi-gram with decreasing order of freq
tsum = as.matrix(t(rollup(dtm_0, 1, na.rm=TRUE, FUN = sum))) # find sum of freq for each term
tsum = tsum[order(tsum, decreasing = T),]       #terms in decreasing order of freq
head(tsum)
tail(tsum)

# select Top 1000 bigrams to unigram
if (length(tsum) > 1000) {n = 1000} else {n = length(tsum)}
tsum = tsum[1:n]

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


it_m = itoken(text2,
              # preprocessor = text.clean,
              tokenizer = tok_fun,
              ids = data$id,
              progressbar = T)

vocab = create_vocabulary(it_m
                          # ngram = c(2L, 2L),
                          #stopwords = stopwords
)

pruned_vocab = prune_vocabulary(vocab,
                                term_count_min = 1)
# doc_proportion_max = 0.5,
# doc_proportion_min = 0.001)

vectorizer = vocab_vectorizer(pruned_vocab)

dtm_m  = create_dtm(it_m, vectorizer)
dim(dtm_m)

dtm = as.DocumentTermMatrix(dtm_m, weighting = weightTf)

print(difftime(Sys.time(), t1, units = 'sec'))

# some basic clean-up ops
dim(dtm)

a0 = apply(dtm, 1, sum)   # apply sum operation to dtm's rows. i.e. get rowSum
  dtm = dtm[(a0 > 5),]    # retain only those rows with token rowSum >5, i.e. delete empty rows
  dim(dtm); rm(a0)        # delete a0 object

a0 = apply(dtm, 2, sum)   # use apply() to find colSUms this time
  dtm = dtm[, (a0 > 4)]     # retain only those terms that occurred > 4 times in the corpus
  dim(dtm); rm(a0)

# view summary wordlcoud
a0 = apply(dtm, 2, sum)     # colSum vector of dtm
  a0[1:5]                   # view what a0 obj is like
  a1 = order(as.vector(a0), decreasing = TRUE)     # vector of token locations
  a0 = a0[a1]     # a0 ordered asper token locations
  a0[1:5]         # view a0 now

windows() # opens new image window
wordcloud(names(a0), a0,     # invoke wordcloud() func. Use ?wordcloud for more info
          scale=c(4,1), 
          3, # min.freq 
          max.words = 100,
          colors = brewer.pal(8, "Dark2"))
title(sub = "Quick Summary Wordcloud")

#------------------------------------------------------#
# Step 1a - Term Co-occurance Matrix                             #
#------------------------------------------------------#

pruned_vocab = prune_vocabulary(vocab,
                                term_count_min = 5)

vectorizer = vocab_vectorizer(pruned_vocab, grow_dtm = FALSE, skip_grams_window = 3L)
tcm = create_tcm(it_m, vectorizer)

tcm.mat = as.matrix(tcm)
adj.mat = tcm.mat + t(tcm.mat)

# how about a quick view of the distilled COG as well, now that we're here?
diag(adj.mat) = 0     # set diagonals of the adj matrix to zero --> node isn't its own neighor
a0 = order(apply(adj.mat, 2, sum), decreasing = T)
adj.mat = as.matrix(adj.mat[a0[1:50], a0[1:50]])

windows()
distill.cog(adj.mat, 'Distilled COG for full corpus',  10,  10)


#################################################
## --- Step 2: model based text analytics ------ ###
#################################################
# -- select optimal num of topics

# K = 6

## Bayes Factor model selection (should choose K or nearby)

# summary(simselect <- topics(dtm, K=K+c(-4:4)), nwrd=0)
# 
# K = simselect$K; K  # Change simselect$K to any the number of topics you want to fit in model

K = 3     # overriding model fit criterion

# -- run topic model for selected K -- #
summary( simfit <- topics(dtm,  K=K, verb=2), nwrd = 12 )
rownames1 = gsub(" ", ".", rownames(simfit$theta));  rownames(simfit$theta) = rownames1;  


## what are the factor components of the factorized DTM?

dim(dtm)     # size of the orig input matrix

str(simfit)     # structure of the output obj

dim(simfit$theta)   # analogous to factor loadings
dim(simfit$omega)   # analogous to factor scores 

simfit$theta[1:5,]
simfit$omega[1:5,]

## why doc numbers? why not firm names?
# firm.names = readLines("clipboard")  # read in first column of vision statements.csv 
# rownames(simfit$omega) = firm.names[as.numeric(rownames(simfit$omega))]
# simfit$omega[1:5,]

# ----------------------------------------------------------#
### Step 2a - compute LIFT for all terms across all topics ###
# ----------------------------------------------------------#

tst = round(ncol(dtm)/100)
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

theta = simfit$theta
lift = theta*0;       # lift will have same dimn as the theta matrix

sum1 = sum(dtm)
pterms = ss.col/sum1     # each column's marginal occurrence probability

for (i in 1:nrow(theta)){  
  for (j in 1:ncol(theta)){
    ptermtopic = 0; pterm = 0;
    ptermtopic = theta[i, j]
    pterm = pterms[i]
    lift[i, j] = ptermtopic/pterm     # divide each cell by the column's marg. occurr. proby.
  }
}   

dim(lift); head(lift, 15)
lift[25:35,]
# Generate A censored Lift matrix
censored.lift = lift
for (i in 1:nrow(lift)){
  censored.lift[i,][censored.lift[i,] < max(censored.lift[i,])] = 0   # hard assigning tokens to topics
} 
head(censored.lift, 10); 


#----------------------------------------------------------------#
# Step 2b - Calculate ETA - each document's score on each topic  #
#----------------------------------------------------------------#

t = Sys.time()

if(nrow(dtm) < 100) {k1 = 10} else {k1= 100}   # to avoid machine choking up in v small datasets

tst = ceiling(nrow(dtm)/k1)  # now using 1% of the rows at a time
a = rep(tst, (k1 - 1))
b = cumsum(a);rm(a)    # cumsum() is cumulative sum.
b = c(0, b, nrow(dtm))  # broke the supermassive dtm into chunks of 1% ncol each
  a0 = which(b > nrow(dtm));    # sometimes, rounding errors cause out of bound errors
  if (length(a0) > 0) {b = b[-a0]}

eta.new = NULL
for (i1 in 1:K){
  
  a2 = c(NULL)
  for (i in 1:(length(b)-1)) {
    tempdtm = dtm[(b[i]+1):(b[i+1]),]
    a = matrix(rep(lift[, i1], nrow(tempdtm)), nrow(tempdtm), ncol(tempdtm), byrow = TRUE)
    a1 = rowSums(as.matrix(tempdtm * a))
    a2 = c(a2, a1); rm(a, a1, tempdtm)
      } # i ends
  
  eta.new = cbind(eta.new, a2); rm(a2)
  
  } # i1 ends

Sys.time() - t  # will take longer than lift building coz ncol is v v high now

rownames(eta.new) = rownames(simfit$omega)
colnames(eta.new) = colnames(simfit$theta)

# so what does eta.new look like? what does it mean?
dim(eta.new)
head(eta.new)

# eta.new = simfit$theta     # if error is happening, worst case

eta.propn = eta.new / rowSums(eta.new)   # calc topic proportions for each document
  eta.propn [1:5,]

# ----------------------------------------#
# Step 3 : Plot Wordcloud for each topic  #
# ----------------------------------------#

df.top.terms = data.frame(NULL)    # can't fit ALL terms in plot, so choose top ones with max loading

for (i in 1:K){       # For each topic 
  a0 = which(censored.lift[,i] > 1) # terms with lift greator than 1 for topic i
  freq = theta[a0, i] # Theta for terms greator than 1
  freq = sort(freq, decreasing = T) # Terms with higher probilities for topic i
  
  # Auto Correction -  Sometime terms in topic with lift above 1 are less than 100. So auto correction
  n = ifelse(length(freq) >= 100, 100, length(freq))
  top_word = as.matrix(freq[1:n])
  
  top.terms = row.names(top_word)
  df.top.terms.t = data.frame(topic = i, top.terms =top.terms, stringsAsFactors = F )
  df.top.terms = rbind(df.top.terms, df.top.terms.t  )
  
} # i loop ends

 
 # pdf(file = paste0(K,' Topic Model results.pdf')) # use pdf() func to save plot directly as PDFs in your getwd()

for (i in 1:K){       # For each topic 
  
  a0 = which(censored.lift[,i] > 1) # terms with lift greator than 1 for topic i
  freq = theta[a0,i] # Theta for terms greator than 1
  freq = sort(freq, decreasing = T) # Terms with higher probilities for topic i
  
  # Auto Correction -  Sometime terms in topic with lift above 1 are less than 100. So auto correction
  n = ifelse(length(freq) >= 100, 100, length(freq))
  top_word = as.matrix(freq[1:n])
  
  # SUB TCM
  sub.tcm = adj.mat[colnames(adj.mat) %in% names(a0),colnames(adj.mat) %in% names(a0)]
  
  #   Plot wordcloud
  windows()
  wordcloud(rownames(top_word), top_word,  scale=c(4,.2), 1,
            random.order=FALSE, random.color=FALSE, 
            colors=brewer.pal(8, "Dark2"))
  mtext(paste("Latent Topic",i), side = 3, line = 2, cex=2)

  # PLot TCM
  windows()
  distill.cog(sub.tcm, '',  5,  5)
  mtext(paste("Term co-occurrence - Topic",i), side = 3, line = 2, cex=2)
    
  } # i loop ends

#dev.off() # closes the graphical devices

### which rows load most on which topics? ###

show.top.loading.rows = function(eta.obj, number.of.units){
  
  K = ncol(eta.obj)    # no. of topic factors
  n = number.of.units
  top.loaders = NULL
  for (i in 1:K){
    a0 = order(eta.obj[,i], decreasing = TRUE)
    a1 = rownames(eta.obj[a0[1:n],])
    top.loaders = cbind(a1, top.loaders)
  } # i loop ends

  a2 = matrix()
  return(top.loaders)
  
} # func ends

show.top.loading.rows(eta.propn, 20)
show.top.loading.rows(eta.new, 10)
