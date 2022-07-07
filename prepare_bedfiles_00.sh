#! /usr/bin/env bash
# -*- coding: utf-8 -*-

###############################################################################
'''
Code to pre-process bedfiles before regions are extracted
location for tissue-specific regulatory builds
http://ftp.ensembl.org/pub/current_regulation/homo_sapiens/RegulatoryFeatureActivity/
'''
###############################################################################
tissue='hippocampus'
cpg='CpgMethly_hippocampus_GSE46644_lifted.bed' ### downloaded from GEO GSE46644
dnase='DNaseHS_ammonshorn_ENCFF102QMS.bed.gz'
H3K27ac='H3K27ac_hippocampus_ENCFF551LSF.bed.gz'
H3K27me3='H3K27me3_hippocampus_ENCFF810LJW.bed.gz'
H3K36me3='H3K36me3_hippocampus_ENCFF442BXP.bed.gz'
H3K4me1='H3K4me1_hippocampus_ENCFF328YWW.bed.gz'
H3K4me3='H3K4me3_hippocampus_ENCFF451IGE.bed.gz'
H3K9me3='H3K9me3_hippocampus_ENCFF265CFH.bed.gz'
chromHMM='E071_18_core_K27ac_hg38lift_mnemonics.bed'
chromatinloops='Schmitt_2016.Hippocampus.hg38.peakachu-merged.loops'
regulatorybuild='homo_sapiens.GRCh38.brain_1.Regulatory_Build.regulatory_activity.20210107.gff'
enhancers='Astrocyte_EP.txt'
tads='Hippocampus_Schmitt2016-raw_TADs.txt'
H3K9ac='H3K9ac_astrocyte_ENCFF649IVK.bed'
ctcf='CTCF_astrocyte_ENCFF450EVR.bed'
polr2a=''

### 
tissue='left_ventricle'
cpg='cpg_methyl_ENCFF318IKY.bed'
dnase='DNaseHS_lv_NCFF591LZL.bed'
H3K27ac='H3K27ac_lv_ENCFF635VBV.bed'
H3K27me3='H3K27me3_lv_ENCFF242JDO.bed'
H3K36me3='H3K36me3_lv_ENCFF767VOC.bed'
H3K4me1='H3K4me1_lv_ENCFF194KSV.bed'
H3K4me3='H3K4me3_lv_ENCFF175CMC.bed'
H3K9me3='H3K9me3_lv_ENCFF242JDO.bed'
chromHMM='E095_18_core_K27ac_hg38lift_mnemonics.bed'
chromatinloops='Leung_2015.VentricleLeft.hg38.peakachu-merged.loops'
ctcf='ctcf_lv_ENCFF165RCX.bed'
polr2a='polr2a_lv_ENCFF371CYY.bed'
regulatorybuild='homo_sapiens.GRCh38.left_ventricle.Regulatory_Build.regulatory_activity.20210107.gff'
enhancers='Left_ventricle_EP.txt'
tads='VentricleLeft_STL003_Leung_2015-raw_TADs.txt'
H3K9ac=''

nochange=()
for var in $dnase $H3K27ac $H3K27ac $H3K36me3 $H3K4me1 $H3K4me3 $H3K9ac $H3K9me3 $ctcf $polr2a; do
    [ ! -z ${file} ] || continue
    nochange+=($var)
done

shared_dir='/Users/steveho/hecateus/development/shared_data'

mkdir processed
mkdir interaction



###############################################################################
### Copy shared data and set names for nochange files
###############################################################################
for file in ${shared_dir}/*; do
cp $file processed/
done

for file in ${nochange}; do
    cp $file processed/
done

###rename chromhmm
cp $chromHMM processed/chromHMM_${tissue}.bed

###############################################################################
### CpG methylation 
### chr start end cpg_methyl
###############################################################################
### For bisulfite - liftover to hg38 if necessary
./liftOver CpgMethly_hippocampus_GSE46644.bed hg19ToHg38.over.chain.gz CpgMethly_hippocampus_GSE46644_lifted.bed CpgMethly_hippocampus_GSE46644_unlifted.bed

bedtools sort -i ${cpg} | bedtools merge -i - -d 200 | awk -v FS="\t" -v OFS="\t" '{print $1, $2, $3, "cpg_methyl"}' > processed/CpGmethyl_${tissue}_parsed.bed


###############################################################################
### Split chromatin loop file into two
###############################################################################
sort -k 1,1 -k2,2n ${chromatinloops} | awk -v FS="\t" -v OFS="\t"  '{print $1, $2, $3, "loop_"NR}' > ${chromatinloops}_1
sort -k 1,1 -k2,2n ${chromatinloops} | awk -v FS="\t" -v OFS="\t"  '{print $4, $5, $6, "loop_"NR}' > ${chromatinloops}_2
cat ${chromatinloops}_1 ${chromatinloops}_2 | sort -k 1,1 -k2,2n > processed/chromatinloops_${tissue}.txt


###############################################################################
### format enhancer atlas file
###############################################################################
sed -e 's/:/\t/g' -e 's/_EN/\tEN/g' -e 's/\$/\t/g' ${enhancers} | sort -k1,1 -k2,2n | awk -v FS="\t" -v OFS="\t" '{print $1, $2, "enhancer_ATLAS"}' | sed -e 's/-/\t/' > interaction/${enhancers}.tabbed


###############################################################################
### Add counts to TAD file
###############################################################################
awk -v FS="\t" -v OFS="\t" '{print $1, $2, $3, "tad_"NR}' ${tads} > processed/tads_${tissue}.txt



###############################################################################
### remove inactive entries from ensembl regulatory build
###############################################################################
gff2bed < ${regulatorybuild} | grep -v -e 'activity=INACTIVE\|open_chromatin_region\|CTCF_binding_site'  | sed -e 's/^/chr/g' -e 's/;/\t/g' -e 's/activity=//g' | awk -v FS="\t" -v OFS="\t" '{print $1, $2, $3, $8"_"$10}' > ${regulatorybuild}.bed

### cat the enhancers with regulatory build 
cat ${regulatorybuild}.bed ${enhancers}.tabbed.for_cat | sort -k1,1 -k2,2n > processed/regulatorybuild_plus_enhancers_${tissue}.bed









###############################################################################
### Remove header and get sizes for UCSC annotation files
###############################################################################
sed -e '1d' simple_repeats_hg38 > simple_repeats_hg38.txt
awk -v FS="\t" -v OFS="\t" '{print $2, $3, $4, $5_$1}' microsatellites_hg38.txt | sed -e '1d' > microsatellites_hg38_parsed.txt
awk -v FS="\t" -v OFS="\t" '{print $2, $3, $4, $5_$1}' cpgislands_hg38.txt | sed -e '1d' > cpgislands_hg38_parsed.txt

### change gnomad parsed file to bed
awk -v FS="\t" -v OFS="\t" '{print $1, $2-1, $2, $3}' gnomad_parsed_snps.txt > gnomad_snps.bed

### gnomad sv to bed file, hg38 version from ncbi 
awk -v FS="\t" -v OFS="\t" '{print $6, $9, $12, $3}' nstd166.GRCh38.variant_region.tsv > gnomadSV_hg38.bed

# awk '{print "chr"$0}' regulatoryensmbl_hg38.gff > regulatorybuild_ensembl_hg38_chr.gff
awk -v FS="\t" -v OFS="\t" '{print $6, $7, $8, $12, $13}' repeatmasker_hg38.txt | sed -e 1d > repeatmasker_hg38_parsed.txt

awk '$5 != "0.000000"' phastcons100way_hg38.bed > phastcons100way_hg38_0removed.bed
### for the gencode file, get type "genes" only

wget http://hgdownload.cse.ucsc.edu/gbdb/hg38/bbi/gc5BaseBw/gc5Base.bw 

# phastcons - converted to bedgraph, rounded $4, remove $4 == 0.0 
# round $4, merged if $4 is equal 
# awk -v FS="\t" -v OFS="\t" '{print $1, $2, $3, $4+0}' OFMT="%.1f"
# phastcons100way_0removed.bedgraph > phastcons100way_rounded.bedgraph
### 0.000 scores were first removed. Then the following
awk '$4 >= "0.90"' phastcons100way.bedgraph | bedtools merge -c 4 -o mean -i - > phastcons100way_parsed_.9.bedgraph
awk '$4 >= "0.95"' phastcons100way.bedgraph | bedtools merge -c 4 -o mean -i - > phastcons100way_parsed_.95.bedgraph


###############################################################################
### Get a lookup table for GTEx to convert between gene symbols and gencode ID
###############################################################################
sed 's/;/\t/g' gencode_v26_genes_only_with_GTEx_targets.bed | cut -f4,12 | sed -e 's/gene_name //g' -e 's/"//g' | sort -u > interaction/gencode_gene_symbol_lookup_table.txt



for file in chromatinloops.bed_dupes_removed tads.bed_dupes_removed; do
awk -v FS="\t" -v OFS="\t" '{print $0, "0"}' $file > $file.copy && mv $file.copy $file
done

cat *dupes_removed* | sort -k1,1 -k2,2n > all_concat_dupes_removed_sorted.bed


### convert from gtf to bed
awk '{ if ($0 ~ "transcript_id") print $0; else print $0" transcript_id \"\";"; }'  gencode.v26.annotation.gtf | gtf2bed - > gencode.v26.bed

### pybedtools script
import pybedtools

file = open("GTEX_geneswithTPM.txt", "r")
a = pybedtools.BedTool('gencode.v26.bed')
content = file.read()
content_list = content.split('\n')

subset = a.filter(lambda b: b[3] in content_list and b[7] == 'gene').saveas("gencode_v26_genes_only_with_GTEx_targets.bed")
subset.slop(g='hg38.chrom.sizes.txt', b=250000).saveas('gene_regions_250000.bed')






# for file in *;
# do
#     sed -i '1,2d' $file
#     count_1=$(head -n 1 $file | tr "\t" "\n"  | wc -l)
#     count=$(($count_1 - 2))
#     sed -i '1s/^/#1.2\n/' $file
#     sed -i "2s/^/56200\t$count\n/" $file
# done