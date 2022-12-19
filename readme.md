# Testing
Please place test images inside /split/test/[class] to get classification accuracy as required by the brief
Labels are taken from the directory to get the performance and are required

Unlabelled instances can also be predicted but are not implemented as it was not required.
img = open_image(loc)
print(learn.predict(img)[0])

# Structure

Train.ipynb builds the model
Test.py loads and runs the pickled model
split splits the images




# To install reqs

Developed for CUDA cores but will run on CPU.

python 3.7.6 must be used

pip install -r requirements.txt

pip install torch==1.5.1 torchvision==0.6.1 -f https://download.pytorch.org/whl/torch_stable.html
