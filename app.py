from flask import Flask, render_template, request
from sklearn.externals import joblib
import os
import pandas as pd
import numpy as np

app = Flask(__name__, static_url_path='/static/')


@app.route('/')
def form():
    return render_template('index.html')


@app.route('/predict_revenue', methods=['POST', 'GET'])
def predicted_revenue():
    # get the parameters
    budget = float(request.form['budget'])
    Country = str(request.form['Country'])
    Language = str(request.form['Language'])
    Genre = str(request.form['Genre'])
    budget_log=np.log(budget)


    # load the X_columns file
    X_columns = joblib.load('model/X_columns.joblib')
    print(X_columns)

    # generate a dataframe with zeros
    df_prediction = pd.DataFrame(np.zeros((1, len(X_columns))), columns=X_columns)
    print(df_prediction)

    # change the dataframe according to the inputs
    df_prediction.at[0, 'budget_log'] = budget_log
    df_prediction.at[0, str("'"+ str(Country)+"'")] = 1.0
    df_prediction.at[0, str(Language)] = 1.0
    df_prediction.at[0, str("'"+str(Genre)+"'")] = 1.0
    print(df_prediction)

    # load the model and predict
    model = joblib.load('model/model.joblib')
    prediction = model.predict(df_prediction.head(1).values)
    predicted_revenue = prediction.round(1)[0]
    predicted_revenue=np.exp(predicted_revenue)
    predicted_revenue='${:,.0f}'.format(predicted_revenue)

    return render_template('results.html',
                           budget=int(budget),
                           Country=str("'"+ str(Country)+"'"),
                           Language=str(Language),
                           Genre=str("'"+str(Genre)+"'"),
                           predicted_revenue=str(predicted_revenue)
                           )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port,  debug=True)
