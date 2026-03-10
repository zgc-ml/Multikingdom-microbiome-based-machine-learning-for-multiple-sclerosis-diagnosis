mkdir -p temp/kraken

for i in `tail -n+2 result/metadata.txt | cut -f1`;do
      time kraken2 --db ~/db/kraken2 \
      --paired temp/hr/${i}_paired_1.fastq temp/hr/${i}_paired_2.fastq \
      --threads 24 --use-names --report-zero-counts \
      --confidence 0.2 \
      --report temp/kraken2/${i}.report \
      --output temp/kraken2/${i}.output; done

for i in `tail -n+2result/metadata.txt|cut -f1`;do
   kreport2mpa.py -r temp/kraken2/${i}.report \
   --display-header \
   -o temp/kraken2/${i}.mpa 
done

tail -n+2 result/metadata.txt | cut -f1 | while read id
do
    tail -n+2 temp/kraken2/${id}.mpa | LC_ALL=C sort | cut -f 2 | sed "1 s/^/${id}\n/" > temp/kraken2/${id}_count
done

header=`tail -n 1 result/metadata.txt | cut -f 1`
echo $header

tail -n+2 temp/kraken2/${header}.mpa | LC_ALL=C sort | cut -f 1 | \
      sed "1 s/^/Taxonomy\n/" > temp/kraken2/0header_count
head -n3 temp/kraken2/0header_count

mkdir -p result/kraken2
paste temp/kraken2/*count > result/kraken2/tax_count.txt
head -n 5 result/kraken2/tax_count.txt
