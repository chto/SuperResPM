echo "apple banana apple" | awk '{ print gensub(/apple/, "orange", "g", $0) }'

echo "cpu=512,mem=875G,node=10,billing=512" |  awk '{ print int(gensub(/.*cpu=([0-9]+).*/, "\\1", "g", $0)) }' 
