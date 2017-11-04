#install.packages('RSelenium')
rm(list=ls())
library("rvest")

######################################

counts = c(0,10,20,30,40,50)
reviews = NULL
for (j in counts){
  url1 = paste0("http://www.imdb.com/title/tt0944947/reviews?filter=love;filter=love;start=",j)
  url2 = paste0("http://www.imdb.com/title/tt0944947/reviews?filter=hate;filter=hate;start=",j)
  
  page1 = read_html(url1)
  page2 = read_html(url2)
  reviews1 = html_text(html_nodes(page1,'#tn15content p'))
  reviews2 = html_text(html_nodes(page2,'#tn15content p'))
  
  reviews.positive = setdiff(reviews1, c("*** This review may contain spoilers ***","Add another review"))
  reviews.negative = setdiff(reviews2, c("*** This review may contain spoilers ***","Add another review"))
  
  reviews = c(reviews,reviews.positive,reviews.negative)
  
}

reviews = gsub("\n",' ',reviews)
writeLines(reviews,'Game of Thrones IMDB reviews.txt')
