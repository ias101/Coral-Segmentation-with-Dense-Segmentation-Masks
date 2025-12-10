# Intro
We fine-tune segformer from Nvidia models under marine dataset, 
in order to face ocean ecosystem challenages caused by humen activties.
# Set Up for Data Set And Environment
Due to geographic and ecosystem semilarity we decide to conduct our project on SEAFLOWER_BOLIVAR
and SEAFLOWER_COURTOWN.
To set up the dataset please move iamges from SEAFLOWER_BOLIVAR and SEAFLOWER_COURTOWN into iamge file.
Move masks_stitched from two dataset into mask file.
Then install dependences via requirements.txt
# Baseline Model
First run Ndataset.py to do the prerpossing and then run base_test.py.
you can get a file 'base_result' including prediction of baseline model
# Fine-tuned Model
Run re_dataset for preprocessing.To trian the model run re_trian.py and you will get pth file contains weights for each layer of the model.
After run re_test.py you can get 'improved_result' including all predicted masks.
# Metrics
Run the metric.py you can get two csv files that contains F1 score, Accuracy, and loss for
both model which are shown on poster
