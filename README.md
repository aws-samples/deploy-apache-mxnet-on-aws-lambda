# Deploying Apache MXNet on AWS Lambda
## Why Deploying Apache MXNet on AWS Lambda
When it comes to Amazon SageMaker model deployment, people usually deploy it on Amazon SageMaker endpoint, which can be more expensive in certain scenarios since the [cost is based on instance-hour](https://aws.amazon.com/sagemaker/pricing/). However, there is a gap for how to deploy MXNet Model on AWS Lambda, and this is a  greenfield for data scientists. Instead, we can directly deploy our output model to Lambda and benefit from serverless architectures.

This is a example of how to deploy your Apache MXNet model on Lambda, which you can benefit from Lambda pricing model. Note that there are some tradeoffs such as [cold start](https://aws.amazon.com/blogs/compute/new-for-aws-lambda-predictable-start-up-times-with-provisioned-concurrency/). Here we will use a pre-trained model in the Amazon SageMaker [End-to-End Multiclass Image Classification Example](https://github.com/aws/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/imageclassification_caltech/Image-classification-fulltraining.ipynb), which will export a model to S3 and we can deploy it on [SageMaker Endpoint](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html) or [AWS Lambda](https://aws.amazon.com/lambda). 

![圖片 1](https://user-images.githubusercontent.com/17841922/103724544-7f92f100-500f-11eb-9487-bcd34bf312c4.png)

If you want to deploy model to AWS Lambda and benefit from Serverless compute resource you need to  ***Dealing with Lambda's size limitations***

### Dealing with AWS Lambda's size limitations
Total size of this project is about ***300 MB***, Includes:
1. output image-classification model on Amazon S3 ***50 MB***
2. Apache MXNet 1.6.0 and dependency ***250 MB***
3. AWS Lambda Function to load Apache MXNet model less than ***1 MB***

AWS Lambda has some limitation of Deployment package (.zip file archive) size: 50 MB [Lambda Limitation](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html)
To deal with deployment package size. ***50 MB (zipped, for direct upload).*** We will...
1. Keep model on Amazon S3
2. Package Apache MXNet and dependency as AWS Lambda Layer

### Steps:
1. [Model Training as Shown in Amazon SageMaker Example](#train-model-with-amazon-sagemaker-example)
2. [Package Apache MXNet as AWS Lambda Layer](#package-apache-mxnet-as-aws-lambda-layer-on-your-local-machine)
3. [Use Apache MXNet for Inference](#use-apache-mxnet-for-inference-with-a-resnet-model)
4. [AWS Lambda Execution Result](#aws-lambda-execution-result)

## Model Training as Shown in Amazon SageMaker Example
[End-to-End Multiclass Image Classification Example](https://github.com/aws/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/imageclassification_caltech/Image-classification-fulltraining.ipynb) is an end-to-end example of distributed image classification algorithm. In this example, we will use the Amazon Sagemaker image classification algorithm to train on the [caltech-256 dataset](http://www.vision.caltech.edu/Image_Datasets/Caltech256/).

### Create model in your Amazon SageMaker Notework 
1. Login to your AWS Console -> Amazon SageMaker -> Notebook instances

![image](https://user-images.githubusercontent.com/38385178/103614634-038f9f00-4f64-11eb-9fc8-72542e69a78d.png)

2. **Create Notebook** instances with **ml.t2.medium**

![image](https://user-images.githubusercontent.com/38385178/103614774-59644700-4f64-11eb-8c78-5f0971431fa6.png)

3. Find your Instance in [Console](https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/notebook-instances)
 -> Open Jupyter -> Introduction to Amazon Algorithms -> Select **Image-classification-fulltraining.ipynb** and **Use** it

![image](https://user-images.githubusercontent.com/38385178/103614968-b19b4900-4f64-11eb-9ab8-7b40cce29909.png)

![image](https://user-images.githubusercontent.com/38385178/103615388-7b11fe00-4f65-11eb-8697-4ad6505b844b.png)

4. **Training parameters** with the parameters for the training job, and hyperparameters that are specific to the algorithm.
You can change those parameters according to your use case and scenario. Just **remember the number of training epochs**, which you need to provide when you deploy the model.

![image](https://user-images.githubusercontent.com/17841922/104262333-5c11ef80-54c2-11eb-9548-da370ed41460.png)

5. Follow instruction until **Create model** which will generate s3 path, and then copy it (it takes about 20 minutes to complete)

![image](https://user-images.githubusercontent.com/38385178/103617997-291fa700-4f6a-11eb-8b3e-1a2a4ef40ca2.png)


## Package Apache MXNet as AWS Lambda Layer on your local machine
1. Export your package dir, and clear file in target dir
```
export PKG_DIR="build/python/lib/python3.6/site-packages"
rm -rf ${PKG_DIR} && mkdir -p ${PKG_DIR}
```

2. Run commnad as following to pip install Apache MXNet and dependency, which target is your package.
```
docker run --rm -v "$PWD":/var/task "lambci/lambda:build-python3.6" /bin/sh -c "pip install -r requirements.txt -t ${PKG_DIR}; exit"
```

3. Replace `<<yourbucket>>` and `<<prefix-key>>` before compress the file in package folder and copy to s3.
```
zip -r python-mxnet.zip ${PKG_DIR}
aws s3 cp python-mxnet.zip s3://<<yourbucket>>/<<prefix-key>>/python-mxnet.zip
```

### AWS Lambda Layer settings
AWS Lambda -> Create Layers  

![image](https://user-images.githubusercontent.com/38385178/103614005-9f201000-4f62-11eb-98cb-7c8326fac2bc.png)


Select **python 3.6** (like above version) -> paste **s3://yourbucket/prefix-key/python-mxnet.zip** -> Create


![image](https://user-images.githubusercontent.com/38385178/103614354-5b79d600-4f63-11eb-87a1-c101d21ef033.png)


## Use Apache MXNet for Inference with a ResNet Model
Apache MXNet model load_checkpoint
1. Read-only file system except **/tmp**
2. Download Apache MXNet output model to **/tmp** and unzip it
3. Apache MXNet image provide method to read, resize and convert format

### The AWS Lambda code is as following:
```python
import json
import mxnet as mx
import numpy as np
import os
import boto3

## change your bucket and key here
bucket = '{{your-bucket-name}}'
key = '{{your-model-output-prefix}}/model.tar.gz'

boto3.resource('s3').Bucket(bucket).download_file(key, '/tmp/model.tar.gz')

os.system('tar -zxvf /tmp/model.tar.gz -C /tmp/')

def lambda_handler(event, context):
    ctx = mx.cpu()
    epoch = <the number of training epochs> # input your epoch when train model
    sym, args, aux = mx.model.load_checkpoint('/tmp/image-classification', epoch)
    
    fname = mx.test_utils.download(event['url'], dirname='/tmp/')
    img = mx.image.imread(fname)
    # convert into format (batch, RGB, width, height)
    img = mx.image.imresize(img, 224, 224) # resize
    img = img.transpose((2, 0, 1)) # Channel first
    img = img.expand_dims(axis=0) # batchify
    img = img.astype(dtype='float32')
    args['data'] = img
    
    softmax = mx.nd.random_normal(shape=(1,))
    args['softmax_label'] = softmax
    
    exe = sym.bind(ctx=ctx, args=args, aux_states=aux, grad_req='null')
    exe.forward()
    
    prob = exe.outputs[0].asnumpy() # 256 + 1 prob
    
    # remove single-dimensional entries from the shape of an array
    prob = np.squeeze(prob)
    
    # classfication labels
    labels = ['ak47', 'american-flag', 'backpack', 'baseball-bat', 'baseball-glove', 'basketball-hoop', 'bat', 'bathtub', 'bear', 'beer-mug', 'billiards', 'binoculars', 'birdbath', 'blimp', 'bonsai-101', 'boom-box', 'bowling-ball', 'bowling-pin', 'boxing-glove', 'brain-101', 'breadmaker', 'buddha-101', 'bulldozer', 'butterfly', 'cactus', 'cake', 'calculator', 'camel', 'cannon', 'canoe', 'car-tire', 'cartman', 'cd', 'centipede', 'cereal-box', 'chandelier-101', 'chess-board', 'chimp', 'chopsticks', 'cockroach', 'coffee-mug', 'coffin', 'coin', 'comet', 'computer-keyboard', 'computer-monitor', 'computer-mouse', 'conch', 'cormorant', 'covered-wagon', 'cowboy-hat', 'crab-101', 'desk-globe', 'diamond-ring', 'dice', 'dog', 'dolphin-101', 'doorknob', 'drinking-straw', 'duck', 'dumb-bell', 'eiffel-tower', 'electric-guitar-101', 'elephant-101', 'elk', 'ewer-101', 'eyeglasses', 'fern', 'fighter-jet', 'fire-extinguisher', 'fire-hydrant', 'fire-truck', 'fireworks', 'flashlight', 'floppy-disk', 'football-helmet', 'french-horn', 'fried-egg', 'frisbee', 'frog', 'frying-pan', 'galaxy', 'gas-pump', 'giraffe', 'goat', 'golden-gate-bridge', 'goldfish', 'golf-ball', 'goose', 'gorilla', 'grand-piano-101', 'grapes', 'grasshopper', 'guitar-pick', 'hamburger', 'hammock', 'harmonica', 'harp', 'harpsichord', 'hawksbill-101', 'head-phones', 'helicopter-101', 'hibiscus', 'homer-simpson', 'horse', 'horseshoe-crab', 'hot-air-balloon', 'hot-dog', 'hot-tub', 'hourglass', 'house-fly', 'human-skeleton', 'hummingbird', 'ibis-101', 'ice-cream-cone', 'iguana', 'ipod', 'iris', 'jesus-christ', 'joy-stick', 'kangaroo-101', 'kayak', 'ketch-101', 'killer-whale', 'knife', 'ladder', 'laptop-101', 'lathe', 'leopards-101', 'license-plate', 'lightbulb', 'light-house', 'lightning', 'llama-101', 'mailbox', 'mandolin', 'mars', 'mattress', 'megaphone', 'menorah-101', 'microscope', 'microwave', 'minaret', 'minotaur', 'motorbikes-101', 'mountain-bike', 'mushroom', 'mussels', 'necktie', 'octopus', 'ostrich', 'owl', 'palm-pilot', 'palm-tree', 'paperclip', 'paper-shredder', 'pci-card', 'penguin', 'people', 'pez-dispenser', 'photocopier', 'picnic-table', 'playing-card', 'porcupine', 'pram', 'praying-mantis', 'pyramid', 'raccoon', 'radio-telescope', 'rainbow', 'refrigerator', 'revolver-101', 'rifle', 'rotary-phone', 'roulette-wheel', 'saddle', 'saturn', 'school-bus', 'scorpion-101', 'screwdriver', 'segway', 'self-propelled-lawn-mower', 'sextant', 'sheet-music', 'skateboard', 'skunk', 'skyscraper', 'smokestack', 'snail', 'snake', 'sneaker', 'snowmobile', 'soccer-ball', 'socks', 'soda-can', 'spaghetti', 'speed-boat', 'spider', 'spoon', 'stained-glass', 'starfish-101', 'steering-wheel', 'stirrups', 'sunflower-101', 'superman', 'sushi', 'swan', 'swiss-army-knife', 'sword', 'syringe', 'tambourine', 'teapot', 'teddy-bear', 'teepee', 'telephone-box', 'tennis-ball', 'tennis-court', 'tennis-racket', 'theodolite', 'toaster', 'tomato', 'tombstone', 'top-hat', 'touring-bike', 'tower-pisa', 'traffic-light', 'treadmill', 'triceratops', 'tricycle', 'trilobite-101', 'tripod', 't-shirt', 'tuning-fork', 'tweezer', 'umbrella-101', 'unicorn', 'vcr', 'video-projector', 'washing-machine', 'watch-101', 'waterfall', 'watermelon', 'welding-mask', 'wheelbarrow', 'windmill', 'wine-bottle', 'xylophone', 'yarmulke', 'yo-yo', 'zebra', 'airplanes-101', 'car-side-101', 'faces-easy-101', 'greyhound', 'tennis-shoes', 'toad', 'clutter']
    
    a = np.argsort(prob)[::-1] # index number sorted by prob
    
    # print and append the top-5 to output
    output = []
    for i in a[0:5]:
        output.append({'label': labels[i], 'probability': str(prob[i])})
        print('probability=%f, class=%s' %(prob[i], labels[i]))
    
    return {'records': output}
```
### AWS Lambda function settings
1. AWS Lambda -> **Create Lambda** function and **Author from scratch**

![image](https://user-images.githubusercontent.com/38385178/103616796-0e4c3300-4f68-11eb-86a3-109b27b5b6ed.png)

2. runtime: python 3.6, expand Change default execution role 

![image](https://user-images.githubusercontent.com/38385178/103617088-8f0b2f00-4f68-11eb-9d05-fd49a4e1869e.png)

3. choose Amazon S3 object read-only permissions

![image](https://user-images.githubusercontent.com/38385178/103617279-f3c68980-4f68-11eb-9b96-23911d58dccd.png)

4. keep other setting default, and modify it after AWS Lambda function created 
5. Add **Lambda Layer**

<img width="960" alt="image" src="https://user-images.githubusercontent.com/38385178/103722492-0a252180-500b-11eb-9148-094695fb18e0.png">

6. **Custom Layer** and choice the layer your just create

<img width="832" alt="image" src="https://user-images.githubusercontent.com/38385178/103722581-448ebe80-500b-11eb-918f-78cdefb4cf06.png">

7. Basic settings -> Memory Size: **1024 MB**, Timeout: **10 seconds**

<img width="648" alt="image" src="https://user-images.githubusercontent.com/38385178/103722716-86b80000-500b-11eb-9b26-2e9a786a47bd.png">

<img width="668" alt="image" src="https://user-images.githubusercontent.com/38385178/103722883-e57d7980-500b-11eb-8b52-d6b7e0717e36.png">

8. create **Test** and **Configure test event**, paste json and **Create**

<img width="730" alt="image" src="https://user-images.githubusercontent.com/38385178/103723014-15c51800-500c-11eb-9b2c-35921118feae.png">

<img width="564" alt="image" src="https://user-images.githubusercontent.com/38385178/103723132-5c1a7700-500c-11eb-98f0-0be396f73001.png">

```
{
  "url": "https://img.etimg.com/thumb/msid-64170489,width-640,resizemode-4,imgsize-55512/the-wallace-sword.jpg"
}
```
8. **Deploy** AWS Lambda function and **Test** it!


<img width="960" alt="image" src="https://user-images.githubusercontent.com/38385178/103723358-d8ad5580-500c-11eb-9ca4-452a827c9c91.png">

<img width="962" alt="image" src="https://user-images.githubusercontent.com/38385178/103723438-05616d00-500d-11eb-9efb-7e149474d7ea.png">

## AWS Lambda Execution Result
Execution Result:
1. Duration: **637.31 ms**
2. Init Duration: **3885.41 ms**
3. Memory Size: **1024 MB**

<img width="1059" alt="image" src="https://user-images.githubusercontent.com/38385178/103723513-3b9eec80-500d-11eb-810d-f68ed740051e.png">

<img width="857" alt="image" src="https://user-images.githubusercontent.com/38385178/103725947-abfc3c80-5012-11eb-8af8-ccd07f06441f.png">


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
