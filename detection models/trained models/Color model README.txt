4/17/2021
This is a CNN model on Color that has an accuracy of ~98%

To load this model, myModel =  tf.keras.models.load_model([path to model])

Takes input shape (numSamples, 180, 135, 3) in float32 rgb values
Used data augmentation during training.

prediction key (labels are one-hot encoded):
	0,0,1	red
	0,1,0	purple
	1,0,0	green

Package version: 
tensorflow                2.1.0           gpu_py37h7db9008_0
tensorflow-base           2.1.0           gpu_py37h55f5790_0
tensorflow-estimator      2.1.0              pyhd54b08b_0
tensorflow-gpu            2.3.1                    pypi_0    pypi
tensorflow-gpu-estimator  2.3.0                    pypi_0    pypi
	
