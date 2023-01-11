#!/usr/bin/env bash
# Download the Rose Youtube Face Liveness Dataset

# Requires login to download
# Total size 5.4 GB

# This did not work, manual download of individual files was done instead.
# Note: Dataset contains files numbered 1-23, but 2, 8, and 19 are missing.

link="https://rose1.ntu.edu.sg/dataset/faceLivenessDetection/download/"
curl_args="-H 'User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:107.0) Gecko/20100101 Firefox/107.0' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8' -H 'Accept-Language: en-US,en;q=0.5' -H 'Accept-Encoding: gzip, deflate, br' -H 'Referer: https://rose1.ntu.edu.sg/dataset/faceLivenessDetection/download' -H 'Connection: keep-alive' -H 'Cookie: sessionid=3bk6ki3fs1ts06r6g8a7kzumuhpf3dam' -H 'Upgrade-Insecure-Requests: 1' -H 'Sec-Fetch-Dest: document' -H 'Sec-Fetch-Mode: navigate' -H 'Sec-Fetch-Site: same-origin' -H 'Sec-Fetch-User: ?1'"

# file names
files=( "10.zip"  "11.zip"  "12.zip"  "13.zip"  "14.zip"  "15.zip"  "16.zip"  "17.zip"  "18.zip"  "2.zip"  "20.zip"  "21.zip"  "22.zip"  "23.zip"  "3.zip"  "4.zip"  "5.zip"  "6.zip"  "7.zip"  "9.zip" )

# length of files array
len=${#files[@]}
echo "Total files to download: $len"

# file links is a sequence from 177 to 196
#links=mapfile -t array < <()
links=($(seq 177 196))
echo "Total links to download: ${#links[@]}"
#echo "Links: ${links[@]}"

for i in $(seq 4 $((len-1)))
do
#  echo "$i", "${files[$i]}", "${links[$i]}"
  curr_link="'$link${links[$i]}'"
  comm="curl $curr_link $curl_args --output ${files[$i]}"
  echo -e "$comm"
  eval "$comm"
  exit
done


# Example curl commands:
#curl 'https://rose1.ntu.edu.sg/dataset/faceLivenessDetection/download/177' -H 'User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:107.0) Gecko/20100101 Firefox/107.0' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8' -H 'Accept-Language: en-US,en;q=0.5' -H 'Accept-Encoding: gzip, deflate, br' -H 'Referer: https://rose1.ntu.edu.sg/dataset/faceLivenessDetection/download' -H 'Connection: keep-alive' -H 'Cookie: sessionid=3bk6ki3fs1ts06r6g8a7kzumuhpf3dam' -H 'Upgrade-Insecure-Requests: 1' -H 'Sec-Fetch-Dest: document' -H 'Sec-Fetch-Mode: navigate' -H 'Sec-Fetch-Site: same-origin' -H 'Sec-Fetch-User: ?1' --output 10.zip
#curl 'https://rose1.ntu.edu.sg/dataset/faceLivenessDetection/download/178' -H 'User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:107.0) Gecko/20100101 Firefox/107.0' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8' -H 'Accept-Language: en-US,en;q=0.5' -H 'Accept-Encoding: gzip, deflate, br' -H 'Referer: https://rose1.ntu.edu.sg/dataset/faceLivenessDetection/download' -H 'Connection: keep-alive' -H 'Cookie: sessionid=3bk6ki3fs1ts06r6g8a7kzumuhpf3dam' -H 'Upgrade-Insecure-Requests: 1' -H 'Sec-Fetch-Dest: document' -H 'Sec-Fetch-Mode: navigate' -H 'Sec-Fetch-Site: same-origin' -H 'Sec-Fetch-User: ?1' --output 11.zip
# skipped 12
# skipped 13
#curl 'https://rose1.ntu.edu.sg/dataset/faceLivenessDetection/download/181' -H 'User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:107.0) Gecko/20100101 Firefox/107.0' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8' -H 'Accept-Language: en-US,en;q=0.5' -H 'Accept-Encoding: gzip, deflate, br' -H 'Referer: https://rose1.ntu.edu.sg/dataset/faceLivenessDetection/download' -H 'Connection: keep-alive' -H 'Cookie: sessionid=3bk6ki3fs1ts06r6g8a7kzumuhpf3dam' -H 'Upgrade-Insecure-Requests: 1' -H 'Sec-Fetch-Dest: document' -H 'Sec-Fetch-Mode: navigate' -H 'Sec-Fetch-Site: same-origin' -H 'Sec-Fetch-User: ?1' --output 14.zip

