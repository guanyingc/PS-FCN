path="data/models/"
mkdir -p $path
cd $path

# Download pre-trained model
for model in "PS-FCN_B_S_32.pth.tar" "UPS-FCN_B_S_32.pth.tar"; do
    wget http://www.visionlab.cs.hku.hk/data/PS-FCN/models/${model}
done

# Back to root directory
cd ../
