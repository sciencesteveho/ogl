#!/bin/bash

'''
Scripts to filter shared bedfiles before generating graphs. 
'''

# convert gencode v26 GTF to bed 
gtf2bed <  gencode.v26.GRCh38.genes.gtf | awk '$8 == "gene"' > gencode_v26_genes_only_with_GTEx_targets.bed

# get gene symbol lookup table for gencode IDs 
cut -f4,10 gencode_v26_genes_only_with_GTEx_targets.bed | sed 's/;/\t/g' | cut -f1,5 | sed -e 's/ gene_name //g' -e 's/\"//g' > gencode_to_genesymbol_lookup_table.txt

# liftover enhancers and keep an index
awk -v FS='\t' -v OFS='\t' '{print $2, $3, $4, "enhancer_"$1}' Enhancer_regions.txt | tail -n +2 > enhancerregions.txt
awk -v FS='\t' -v OFS='\t' '{print $2":"$3"-"$4, "enhancer_"$1}' Enhancer_regions.txt | tail -n +2 > enhancer_indexes_unlifted.txt

./../liftOver \
    enhancerregions.txt \
    ../hg19ToHg38.over.chain.gz \
    enhancer_regions_lifted.txt \
    enhancers_unlifted.txt

awk -v FS='\t' -v OFS='\t' '{print $1":"$2"-"$3, $4}' enhancer_regions_lifted.txt > enhancer_indexes.txt

# poly-(A) target sites for hg38 are downloaded from PolyASite, homo sapiens v2.0, release 21/04/2020
# filter poly-(A) targets for clusters supported by at least 1% of samples
awk -v FS='\t' -v OFS='\t' '$7 >= 0.01' atlas.clusters.2.0.GRCh38.96.bed \
    | cut -f 1,2,3,4,10 \
    | awk -v FS='\t' -v OFS='\t' '{print "chr"$1,$2,$3,"polya_"$4_$5}' \
    > polyasites_filtered_hg38.bed


# transcription start sites for hg38 are downloaded from refTSS v3.3, release 18/08/2021
awk -v FS='\t' -v OFS='\t' '{print $1,$2,$3,"tss_"$4}' refTSS_v3.3_human_coordinate.hg38.bed \
    > tss_parsed_hg38.bed


# micro RNA (miRNA) target sites for hg19 are downloaded from TargetScanHuman 7.2, release 03/2018
# target sites are lifted over to hg38
shared_dir='/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data'
./preprocess/shared_data/liftOver \
    Predicted_Target_Locations.default_predictions.hg19.bed \
    ${shared_dir}/hg19ToHg38.over.chain.gz \
    miRNAtargets_lifted_hg38.bed \
    mirRNA_unlifted

awk -v FS='\t' -v OFS='\t' '{print $1,$2,$3,"miRNAtarget_"$4}' miRNAtargets_lifted_hg38.bed \
    sed 's/::/__/g' \
    > miRNAtargets_parsed_hg38.bed


# RNA Binding protein sites were downloaded from POSTAR 3
# We first merged sites within 50 base pairs to create RBP binding site clusters
# We keep sites present in at least 2 samples and covering 5% of the represented RNA binding proteins (11)
bedtools merge \
    -d 50 \
    -i human.txt \
    -c 6,6,8 \
    -o distinct,count_distinct,count_distinct \
    > rnab_merged_50.bed
    
awk '($5 >= 11)&&($6 >= 2)' rnab_merged_50.bed \
    | cut -f1,2,3,4 \
    | sed 's/,/_/g' \
    | awk -v FS='\t' -v OFS='\t' '{print $1, $2, $3, "rnab_"$4}' \
    > rbpbindingsites_50_parsed_hg38.bed


# ENCODE SCREEN candidate promoters from SCREEN registry of cCREs V3
awk -v FS='\t' -v OFS='\t' '{print $1,$2,$3,"promoter"}' GRCh38-PLS.bed \
     > promoters_parsed_hg38.bed


### For analysis 
# SVs from HGSVC2, Ebert et al., Science, 2021. Split into ins and dels
tail -n +2 variants_freeze4_sv_insdel.tsv \
    | cut -f2,3,4,5 \
    | grep 'INS' \
    > svins_ebert_hg38.bed

tail -n +2 variants_freeze4_sv_insdel.tsv \
    | cut -f2,3,4,5 \
    | grep 'DEL' \
    > svdel_ebert_hg38.bed


# SNPs from dbSNP release 155 mapped to hg38. Snps are merged if adjacent using bedtools
# first, chrname file, adapted from rrbuterliii on BioStars https://www.biostars.org/p/410789/
wget 'https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.39_GRCh38.p13/GCF_000001405.39_GRCh38.p13_assembly_report.txt'

for k in *assembly_report.txt
  do
    out=$(echo $k | sed 's/.txt/.chrnames/')
    grep -e '^[^#]' $k | awk '{ print $7, "chr"$1 }' > $out
done

grep -v "^#" GCF_000001405.39 \
    | LC_ALL=C grep -w '^#\|chr[1-9]\|chr[1-2][0-9]' \
    | cut -f1,2,3 \
    | awk -v FS='\t' -v OFS='\t' '{print $1, $2-1, $2}' \
    > snps_dbsnp_autosomes.bed


# Convert miRNA coordinates from GFF to bed 
 gff2bed < hsa.gff3 | awk '$8 == "miRNA"' | sed 's/;/\t/g' | cut -f1,2,3,12 | sed 's/Name=//g'

# filter ensembl versions
cut -f1 gencode_to_genesymbol_lookup_table.txt | sed 's/\./ /'g | cut -f1 -d' ' | sort -u > ensemble_genes_cleaned.txt

awk -F'\t' 'NR==FNR{c[$1]++;next;c[$2]' ensemble_genes_cleaned.txt entrez_to_genesymbol.txt > test.txt

# for tissue in hippocampus left_ventricle liver lung mammary pancreas skeletal_muscle;
# do
# rm ${tissue}/local/svins_ebert_hg38.bed ${tissue}/local/svdel_ebert_hg38.bed
# done

# for tissue in hippocampus left_ventricle liver lung mammary pancreas skeletal_muscle;
# do
# ln -s /ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/local/svins_ebert_hg38.bed ${tissue}/local/svins_ebert_hg38.bed
# done

# for tissue in hippocampus left_ventricle liver lung mammary pancreas skeletal_muscle;
# do
# ln -s /ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/local/svdel_ebert_hg38.bed ${tissue}/local/svdel_ebert_hg38.bed
# done
