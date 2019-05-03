pip3 install -r requirements.txt

mkdir -p input
mkdir -p features
mkdir -p models

wget -q -P models https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip
wget -q -P models https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip

cd models
unzip -o uncased_L-24_H-1024_A-16.zip
unzip -o cased_L-24_H-1024_A-16.zip

mkdir -p elmo
wget -q -P elmo https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json
wget -q -P elmo https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5
cd ..
