## MNIST digit recognition in numpy.

To run the experiments execute './download_mnist.sh'.
The script will create the './data' folder 
download the MNIST data from http://yann.lecun.com
and extract the MNIST train and test samples.

To train and test the dense classifier run
'python train_dense_mnist.py'.
similarly, the convolutional neural net can be trained
and tested by running
'python train_cnn_mnist.py' .

Both scripts also do a test run, which will take a little while.
Running these I observed:
- A dense structure MNIST-test accuracy of 97.22% .
- A CNN structure MNIST-test accuracy of 95.21%.

I did not spent time tuning hyperparameters, but I think
we can safely conclude that both structures extract useful
patterns.
