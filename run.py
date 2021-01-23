# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from scripts.train import train_models
from tests.testnlp import test_models

if __name__ == '__main__':  


	#train_models()

	#model_name = 'my_model_nnlm-en-dim128-with-normalization.h5'
	#model_name  = 'my_model_nnlm-en-dim50.h5'
	model_name  = 'my_model_gnews-swivel-20dim.h5'

	test_models(model_name)







