mkdir -p temp/bracken

readLen=150
prop=0.2
tax=S

for i in `tail -n+2 result/metadata.txt | cut -f1`;do
        bracken -d ~/db/kraken2 \
          -i temp/kraken2/${i}.report \
          -r ${readLen} -l ${tax} -t 0 \
          -o temp/bracken/${i}.brk \
          -w temp/bracken/${i}.report; done

wc -l temp/bracken/*.report

tail -n+2 result/metadata.txt | cut -f1 | while read id 
do
      tail -n+2 temp/bracken/${id}.brk | LC_ALL=C sort | cut -f6 | sed "1 s/^/${id}\n/" \
      > temp/bracken/${id}.count
done

h=`tail -n1 result/metadata.txt|cut -f1`

tail -n+2 temp/bracken/${h}.brk | LC_ALL=C sort | cut -f1 | \
      sed "1 s/^/Taxonomy\n/" > temp/bracken/0header.count

ls temp/bracken/*count | wc

paste temp/bracken/*count > result/kraken2/bracken.${tax}.txt