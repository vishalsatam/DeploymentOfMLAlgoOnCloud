# Deploying Machine Learning Services on Cloud
### Assignment 3
### Team 1 - Manasi Dalvi & Vishal Satam

This project has been created to deploy the classification and prediction algorithms that we have developed for the Fredie Mac's dataset. More details available on the github URL :  https://github.com/vishalsatam/PredictiveModellingOnFreddieMacLoans  You will require login credentials from Freddie Mac's Single Family Loans dataset http://www.freddiemac.com/research/datasets/sf_loanlevel_dataset.html in order to execute the below docker image.

## Docker Image

The docker image has been created for preprocessing the data from Freddie Mac's website.

Pull the image
```
docker pull vishalsatam1988/assignment3
```
Run the summarization script
```
docker run -it vishalsatam1988/assignment3 sh /src/assignment3/downloadAndClean.sh "<username>" "<password>" <startyear> <endyear>
```
```
Eg : docker run -it vishalsatam1988/assignment3 sh /src/assignment3/downloadAndClean.sh "satam.v@husky.neu.edu" "Eq=yF?f3" 2005 2016
```

Commit the running container
```
docker commit <containerid> vishalsatam1988/assignment3
```

* You can use the origination file to upload directly to Microsoft Azure for building the Predictive Models. 
* For the performancesummary.csv file located at /src/assignment3/data/ you would have to do some pre-processing to perform Random Undersampling on the majority class before using the file in Microsoft Azure.
* For this process, additional memory is required and your docker machine might return a Segmentation Fault. So, you can run these pre-processing functions from the jupyter notebook.

For pre-processing the performancesummary.csv file, please open the Jupyter notebook to access the functions for pre-processing and use the function create_train_test_sample() to create the train and test files that can be uploaded to Microsoft Azure for building the Classification Models.

View results in Jupyter Notebook - Open /src/assignment3/RandomUnderSampling.ipynb
```
docker run -it -d -p 8888:8888 vishalsatam1988/assignment3 /bin/bash -c 'jupyter notebook --no-browser --allow-root --ip=* --NotebookApp.password="$PASSWD" "$@"'
```

### Instructions for building Web application have been given in the Flask Application folder