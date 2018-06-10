
# extract
tar -xvf lookbook.tar
mkdir -p data
mv ./lookbook/data ./data/raw
rm -rf lookbook

mkdir -p data/prepro
python prepro.py --out_dir 'data/prepro' --data_root 'data/raw'


