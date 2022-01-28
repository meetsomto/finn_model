# run 'pip install catboost'  in vscode terminal
from flask import Flask
from catboost import CatBoostClassifier
import pickle
from flask import request, jsonify

app = Flask(__name__)

# Mapping categorical features:

gender_of_respondent_Female_map = {'Male': 0, 'Female': 1}

location_type_map = {'Rural': 1, 'Urban': 0}

cellphone_access_map = {'Yes': 1, 'No': 0}

relationship_with_head_map = {'Spouse': 5, 'Head of Household': 1, 'Other relative': 3, 'Child': 0, 'Parent': 4,
                              'Other non-relatives': 2}

marital_status_map = {'Married/Living together': 2, 'Widowed': 4, 'Single/Never Married': 3, 'Divorced/Separated': 0,
                      'Dont know': 2}

country_map = {'Kenya': 'KE', 'Rwanda': 'RW', 'Tanzania': 'TA', 'Uganda': 'UG'}

country_keys = {'KE': {'Country_KE': 1, 'Country_RW': 0, 'Country_TA': 0, 'Country_UG': 0},
                'RW': {'Country_KE': 0, 'Country_RW': 1, 'Country_TA': 0, 'Country_UG': 0},
                'TA': {'Country_KE': 0, 'Country_RW': 0, 'Country_TA': 1, 'Country_UG': 0},
                'UG': {'Country_KE': 0, 'Country_RW': 0, 'Country_TA': 0, 'Country_UG': 1},
                }

education_level_map = {'Other/Dont know/RTA': 0, 'No formal education': 0, 'Primary education': 1,
                       'Vocational/Specialised training': 2, 'Secondary education': 3, 'Tertiary education': 4}

job_type_map = {'Self employed': 9, 'Government Dependent': 4, 'Formally employed Private': 3, 'Informally employed': 5,
                'Formally employed Government': 2,
                'Farming and Fishing': 1, 'Remittance Dependent': 8, 'Other Income': 7, 'Dont Know/Refuse to answer': 0,
                'No Income': 6}

bank_account_map = {0: "Yes", 1: "No"}  # map target variable


def predict_bank_account(Year: int, Location_type: str, Cellphone_access: str, Household_size: int, Age: int,
                         Relationship_with_head: str,
                         Marital_status: str, Education_level: str, Job_type: str, Gender_female: str, Country: str):
    # Load ML object and read model

    with open('model.pkl', 'rb') as pickle_file:
        model = pickle.load(pickle_file)

    # Transform argument passed into func as encoded variable using the dictionaries
    Gender_female = gender_of_respondent_Female_map[Gender_female]
    Location_type = location_type_map[Location_type]
    Cellphone_access = cellphone_access_map[Cellphone_access]
    Relationship_with_head = relationship_with_head_map[Relationship_with_head]
    Marital_status = marital_status_map[Marital_status]

    Education_level = education_level_map[Education_level]
    Job_type = job_type_map[Job_type]

    def country_transform(country_string):
        country_dict_key = country_keys[country_map[country_string]]

        countries_hotgroup = country_dict_key

        country_values = []
        [country_values.append(value) for key, value in countries_hotgroup.items()]

        # Country_Kenya = country_values[0]
        # Country_Rwanda= country_values[1]
        # Country_Tanzania  = country_values[2]
        # Country_Uganda = country_values[3]

        return country_values

    transform_country = country_transform(Country)
    Country_Kenya = transform_country[0]
    Country_Rwanda = transform_country[1]
    Country_Tanzania = transform_country[2]
    Country_Uganda = transform_country[3]

    # Prediction for an individual

    data = [Year, Location_type, Cellphone_access, Household_size, Age, Relationship_with_head, Marital_status,
            Education_level, Job_type, Gender_female, Country_Kenya, Country_Rwanda, Country_Tanzania, Country_Uganda]
    # print("Predictors", data)
    y_predict = model.predict([[Year, Location_type, Cellphone_access, Household_size, Age, Relationship_with_head,
                                Marital_status, Education_level, Job_type, Gender_female, Country_Kenya, Country_Rwanda,
                                Country_Tanzania, Country_Uganda]])

    y_predict = bank_account_map[y_predict[0]]

    return y_predict


@app.route("/")
def hello():
    return "A test web service for accessing a machine learning model for Binary Classification (individual likelihood of bank account ownership)"


@app.route('/bank', methods=['GET'])
def api_all():
    #    return jsonify(data_science_books)

    Year = int(request.args['Year'])
    Location_type = request.args['Location_type']
    Cellphone_access = request.args['Cellphone_access']
    Household_size = int(request.args['Household_size'])
    Age = int(request.args['Age'])
    Relationship_with_head = request.args['Relationship_with_head']
    Marital_status = request.args['Marital_status']
    Education_level = request.args['Education_level']
    Job_type = request.args['Job_type']
    Gender_female = request.args['Gender_female']
    Country = request.args['Country']

    bank = predict_bank_account(Year, Location_type, Cellphone_access, Household_size, Age, Relationship_with_head,
                                Marital_status, Education_level, Job_type, Gender_female, Country)

    # return(jsonify(bank))
    return (jsonify(account_exists=bank))
