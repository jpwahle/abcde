cat reddit-posts-urls.txt | xargs -n 1 -P 4 wget > download.log 2>&1 &
disown