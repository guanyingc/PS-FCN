mkdir -p data/datasets
cd data/datasets

# Download real testing dataset
url="https://www.dropbox.com/s/hdnbh526tyvv68i/DiLiGenT.zip?dl=0"
name="DiLiGenT"

wget $url -O ${name}.zip
unzip ${name}.zip
rm ${name}.zip

cd ${name}/pmsData/
ls | sed '/objects.txt/d' > objects.txt
cp ballPNG/filenames.txt .

# Back to root directory
cd ../../../../

