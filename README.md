# model_training
training ECFPNUM-NN 100 epochs and the output properties are IE and EA.  
`python ECFPNUM -i input_data  -e 100 -p IE EA`

training ECFP_SYBYL 50 epochs and the output properties is IE.  
`python ECFPNUM -i input_data  -e 50 -p IE`

training SMILES-GRU 50 epochs and the output properties is IE.  
`python training_RNN.py -i input_data  -e 50 -p IE`
