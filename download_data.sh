#!/bin/bash
mkdir -p tmp
cd tmp
echo "Start downloading ..."
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_XpqaXd02Ct4eB375PN6V9jvG0QLYPxu' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_XpqaXd02Ct4eB375PN6V9jvG0QLYPxu" -O data && rm -rf /tmp/cookies.txt
unzip data.zip
echo "Done!"
