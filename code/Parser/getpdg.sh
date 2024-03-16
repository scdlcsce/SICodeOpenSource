src_dir=$TODIR

total=$(find $src_dir -name "*.bin.zip" | wc -l)
echo $total
count=0

# generate cpg
for file in $(find $src_dir -name "*.bin.zip" -print)
do
	count=`expr $count + 1`
	echo "[$count/$total]: $file"
	echo ${file%".bin.zip"*}"_pdg.json"
	if [ -f  ${file%".bin.zip"*}"_pdg.json" ];then
        continue
    fi
	java -jar getpdg.jar $file
done
