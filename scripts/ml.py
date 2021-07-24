import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from  sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import logging
from git import Repo
import sys
import warnings

 

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)



import dvc.api


path='data/AdSmartABdata.csv'
repo='../abtest-ml'
version='v2'

data_url= dvc.api.get_url(
 path=path,
 repo=repo,
 rev=version 
 )

#print(data_url)
mlflow.set_experiment('ab-test-mlops')

def eval_metrics(actual,pred):
    rmse= np.sqrt(mean_squared_error(actual,pred))
    mae= mean_absolute_error(actual,pred)
    r2= r2_score(actual,pred)
    return rmse,mae,r2

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    np.random.seed(40)
    
    
data= pd.read_csv(data_url, sep=',')
    
mlflow.log_param('data_url',data_url)
mlflow.log_param('data_version',version)
mlflow.log_param('input_rows',data.shape[0])
mlflow.log_param('input_cols',data.shape[1])
    
train_size=0.7
   
train,test= train_test_split(data,train_size=0.7)
print(train.shape,test.shape)

test_size=0.3

test, valid = train_test_split(test,test_size=0.3 )

print(valid.shape)  
    
train_x=train.drop(['platform_os'],axis=1)
test_x=test.drop(['platform_os'], axis=1)
val_x=valid.drop(['platform_os'],axis=1)

train_y=train.drop(['browser'], axis=1)
test_y=test.drop(['browser'], axis=1)
val_y=valid.drop(['browser'],axis=1)


cols_x=pd.DataFrame(list(train_x.columns))
cols_x.to_csv('browser.csv', header=False, index=False)
mlflow.log_artifact('browser.csv')    

cols_y=pd.DataFrame(list(train_y.columns))
cols_y.to_csv('platform_os.csv', header=False, index=False)
mlflow.log_artifact('platform_os.csv')  
    
# alpha=float(sys.argv[1] )if len(sys.argv) > 1 else 0.5
# l1_ratio= float(sys.argv[2]) if len(sys.argv)> 2 else 0.5

# lr= ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
# lr.fit(train_x,train_y)

    
    