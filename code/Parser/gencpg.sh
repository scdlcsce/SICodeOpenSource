src_dir=$TODIR

total=$(find $src_dir -name "*.c" | wc -l)
echo $total
count=0

for file in $(find $src_dir -name "*.c" -print)
do
	count=`expr $count + 1`
	echo "[$count/$total]: $file"
	java -jar gencpg.jar $file
done
