Get started by:
1. Set up a python venv
2. Run: pip install kagglehub
3. Run the Data_Download.py file to install the dataset
4. Store the path of where your dataset is stored in a .env file
5. Work on the project in the Jupyter Notebook

What we need to implement:
- Convert audio data to the coefficients and parameters used by the models
- Remove background noise
- Integrate background noise removal and the audio data conversion
- Create more training data (>100k rows; where 1 row = 1 second window)
- Retrain model based on training data
- Create demonstration system; make a fancy UI with lovable and allow users to record a short audio snippet, have the model give its prediction and confidence, and then generate a deepfake version of the short audio snippet and have the model predict that. 
    - Could be a two device system, to definitively prove we are not hardcoding the result.
    - Keep the demo under 2 minutes.