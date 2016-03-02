
# coding: utf-8

# In[8]:

import pandas as pd
import boto

a = pd.DataFrame([0,1,2])
print(a)
a.to_csv('res.csv')

### Connection to S3

s3 = boto.connect_s3(aws_access_key_id='', 
                     aws_secret_access_key='')

s3_bucket_p2 = s3.get_bucket('harvard_practicals2')
k = s3_bucket_p2.new_key('res.csv')
k.set_contents_from_filename('res.csv')

