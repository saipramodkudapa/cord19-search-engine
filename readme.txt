# Exectute the below commands as part of the setup for executing the main program.
# Once this is done, execute armyants_project.py which is the main program which reads the data set
# and does Tf-IDF, PCA, Kmeans and LDA. This will generate the complete required information to act on
# the user search query and store them into 2 files (cluster_info.dms and keywords_cluster_info.dms)
# which can be found in the current directory.
# NOTE: these files are not directly readable as these are dumped using pickle package.

# After the above, execute input.py program which takes input(user query) from the command prompt
# and outputs list of Document IDs onto output shell.

## Packages to install
pip3 install kaggle
pip3 install -U spacy
pip3 install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz
pip3 install json


## Create a directory named "content" in the current directory (preferred root directory)
mkdir content/

## Create a directory named .kaggle in the current directory
mkdir .kaggle/

## Inside 'content' folder create another directory ".kaggle"
cd content
mkdir .kaggle/
cd ../

## Credentials for kaggle-cli api
token = {"username":"saipramodkudapa","key":"26273f7393c832d3b1283cdbec485327"}


## Writing the token to kaggle.json file
with open('/content/.kaggle/kaggle.json', 'w') as file:
    json.dump(token, file)

## Copying kaggle.json file
cp ./content/.kaggle/kaggle.json ~/.kaggle/kaggle.json

## set kaggle path
kaggle config set -n path -v{/content}

## Changing permission for kaggle.json
chmod 600 ./.kaggle/kaggle.json

##Downloading dataset into content directory
kaggle datasets download -d allen-institute-for-ai/CORD-19-research-challenge -p /content

cd content/
unzip \*.zip