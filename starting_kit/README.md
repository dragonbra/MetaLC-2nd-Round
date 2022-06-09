## How to use the Starting Kit?

### 1. Download the Starting Kit

### 2. Create a virtual environment:
```console
cd starting_kit/

pip install virtualenv

python -m virtualenv metaLC-challenge

source metaLC-challenge/bin/activate

python -m pip install -r requirements.txt
```
### 3. Implement your agent

Open the file: *sample_code_submission/agent.py*.

Implement your own agent by modifying all 4 functions in the file, following the code documentation.

We provided two sample agents for references:

- Random Search agent
- Average Rank agent

### 4. Test the implemented agent with the sample data provided
```console
cd starting_kit/

python ingestion_program/ingestion.py

python scoring_program/score.py
```

The results and visualizations will be written to these following files:

 *starting_kit/output/scores.txt*

 *starting_kit/output/scores.html*

NOTE: Performing well on the sample (synthetic) data does NOT guarantee your agent will perform well on the real data used for testing and ranking in our competition.

### 5. Submit the implemented agent to the competition on Codalab
Zip only two files: *agent.py* and *metadata*, WITHOUT the top directory.

Then, name the zipped file as you want and submit it to the DEVELOPMENT phase of the competition on Codalab.
