# Topic Modeling using Contextual Cues
## Topic Modeling using Contextual Cues is described in the following paper (an update of the paper will be made as soon as it is published on IEEE):

Truică, Ciprian-Octavian and Apostol, Elena Simona and Leordeanu, Cătălin Adrian. **Topic modeling using contextual cues**. In *International Symposium on Symbolic and Numeric Algorithms for Scientific Computing (SYNASC2017)*. September 2017

### Folders
* src - contains the source code for the project
* dataset - contains the datasets in JSON format, to easily import into a MongoDB
* results - contains the test results

### Python packages
* numpy==1.13.0
* scipy==0.19.0
* scikit-learn==0.18.2
* nltk==3.2.4
* pymongo==3.4.0

### Database
The project uses a MongoDB (v3.4.4). In the database folder you will find a 2 scripts, one that imports all the data set into MongoDB (import.sh) and one that exports the data sets into JSON files (export.sh).

Modify accordigly to your system.

Please contact the authors to receive the full datasets.

### How to run
Run the bash script run.sh from the src folder after instaling the MongoDB and the Python packages and importing the dataset into MongoDB. Edit this script to point to the results folder (PATHR) and change the names of the databases (DB1, DB2,...), number of iterations (NUM_ITER) and number of tests (NUM_TESTS).
