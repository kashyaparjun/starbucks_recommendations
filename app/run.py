import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Table
from sklearn.externals import joblib
from sqlalchemy import create_engine
from datetime import datetime


app = Flask(__name__)

# load data
# read in the json files
portfolio = pd.read_json('../data/portfolio.json', orient='records', lines=True)
profile = pd.read_json('../data/profile.json', orient='records', lines=True)
transcript = pd.read_json('../data/transcript.json', orient='records', lines=True)
filtered_data = joblib.load('../filtered_data.pkl')

# load model
channel_model = joblib.load("../channel_model.pkl")
difficulty_model = joblib.load("../difficulty_model.pkl")
duration_model = joblib.load("../duration_model.pkl")
offer_model = joblib.load("../offer_model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
        starting function
        INPUT:
            - 
        OUTPUT:
            - displays graphs and renders master.html
    """
    
    # Converting string date to datetime and adding a column containing only year of membership
    profile.became_member_on =  pd.to_datetime(profile.became_member_on, format='%Y%m%d')
    profile['became_member_on_year'] = profile.became_member_on.apply(lambda x: x.year)

    # Female membership data
    female_membership_data = profile[profile['gender'] == 'F'].groupby('became_member_on_year')['id'].count().to_frame()
    female_membership_data['year'] = female_membership_data.index

    # Male membership data
    male_membership_data = profile[profile['gender'] == 'M'].groupby('became_member_on_year')['id'].count().to_frame()
    male_membership_data['year'] = male_membership_data.index

    # Load sample data
    sample_data = filtered_data[['gender', 'age', 'income', 'became_member_on', 'difficulty', 'duration', 'offer_type']]

    sample_data = pd.concat([sample_data.head(), sample_data.tail()], axis=0)

    sample_data_values = sample_data.values

    filtered_data_sample_data = [[] for i in range(len(list(sample_data.columns)))]

    # Converting row wise data to column wise
    for i in range(len(sample_data_values)):
        for j in range(len(list(sample_data.columns))):
            filtered_data_sample_data[j].append(sample_data_values[i][j])


    
    graphs = [
        {
            'data': [
                Bar(
                    x=female_membership_data.year,
                    y=female_membership_data.id
                )
            ],

            'layout': {
                'title': 'Female membership start year',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Year"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=male_membership_data.year,
                    y=male_membership_data.id
                )
            ],

            'layout': {
                'title': 'Male membership start year',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Year"
                }
            }
        },
        {
            'data': [
                Table(
                    header=dict(values=sample_data.columns),
                    cells=dict(values=filtered_data_sample_data)
                )
            ],
            'layout': {
                'title': 'Sample input data'
            }
        }
    ]
    #graphs = []
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

def get_results(age, income, membership_days, gender_M, gender_F, gender_O):
    """
        Function to call all three models, and output the results
        Input:
            - age: Age of the customer
            - income: Income of the customer
            - membership_days: Number of days the customer is a member of Starbucks (today's date - membership date)
            - gender_M: 1 if customer is male
            - gender_F: 1 is customer is female
            - gender_O: 1 is customer is of other gender
        Ouput:
            - dictionary containing channel, difficulty and duration
    """
    inp = [[age, income, membership_days, gender_M, gender_F, gender_O]]
    channel = channel_model.predict(inp)[0]
    channels = []
    if channel[0] == 1:
        channels.append("email")
    if channel[1] == 1:
        channels.append("mobile")
    if channel[2] == 1:
        channels.append("social")
    if channel[3] == 1:
        channels.append("web")
    difficulty = difficulty_model.predict(inp)[0]
    duration = duration_model.predict(inp)[0]
    offers = offer_model.predict(inp)[0]
    offer = ""
    if offers[0] == 1:
        offer = "bogo"
    elif offers[1] == 1:
        offer = "discount"
    res = {
        "channels": channels,
        "difficulty": difficulty,
        "duration": duration,
        "offer": offer
    }
    return res

# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
        This function is called when the /go route is activated
        - Input:
            - request args
        - Output:
            - result from the ML models
    """
    # save user input in query
    age = request.args.get('age', '')
    gender = request.args.get('gender', '')
    income = request.args.get('income', '')
    membership_date = request.args.get('membership_date', '')

    # load the date into datetime to get date difference
    date = membership_date.split("-")
    join_date = datetime(int(date[0]), int(date[1]), int(date[2]))
    diff = (datetime.now() - join_date).days

    # encode gender
    gender_encoded = [0, 0, 0]
    if gender == 'M':
        gender_encoded[0] = 1
    elif gender == 'F':
        gender_encoded[1] = 1
    else:
        gender_encoded[2] = 1

    # use model to predict classification for query
    res = get_results(age, income, diff, gender_encoded[0], gender_encoded[1], gender_encoded[2])

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        age=age,
        gender=gender,
        income=income,
        membership_date=membership_date,
        result=res
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()