for year in {2010..2022}; do
    for month in 01 02 03 04 05 06 07 08 09 10 11 12; do
        echo "https://archive.org/download/pushshift_reddit_200506_to_202212/reddit/submissions/RS_${year}-${month}.zst"
    done
done > reddit-posts-urls.txt

for year in {2010..2022}; do
    for month in 01 02 03 04 05 06 07 08 09 10 11 12; do
        echo "https://archive.org/download/pushshift_reddit_200506_to_202212/reddit/comments/RC_${year}-${month}.zst"
    done
done > reddit-comment-urls.txt