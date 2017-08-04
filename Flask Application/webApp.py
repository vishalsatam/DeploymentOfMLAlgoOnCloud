from flask import Flask, render_template, request, jsonify
import atexit
import cf_deployment_tracker
import os
import json
import requests
 
# Emit Bluemix deployment event
cf_deployment_tracker.track()

app = Flask(__name__)

db_name = 'mydb'
client = None
db = None
'''
if 'VCAP_SERVICES' in os.environ:
    vcap = json.loads(os.getenv('VCAP_SERVICES'))
    print('Found VCAP_SERVICES')
    if 'cloudantNoSQLDB' in vcap:
        creds = vcap['cloudantNoSQLDB'][0]['credentials']
        user = creds['username']
        password = creds['password']
        url = 'https://' + creds['host']
        client = Cloudant(user, password, url=url, connect=True)
        db = client.create_database(db_name, throw_on_exists=False)
elif os.path.isfile('vcap-local.json'):
    with open('vcap-local.json') as f:
        vcap = json.load(f)
        print('Found local VCAP_SERVICES')
        creds = vcap['services']['cloudantNoSQLDB'][0]['credentials']
        user = creds['username']
        password = creds['password']
        url = 'https://' + creds['host']
        client = Cloudant(user, password, url=url, connect=True)
        db = client.create_database(db_name, throw_on_exists=False)
'''
# On Bluemix, get the port number from the environment variable PORT
# When running this app on the local machine, default the port to 8080
port = int(os.getenv('PORT', 8080))

def loadApiKeys(mltype):
    with open('apikeys.json') as data_file:    
            apikeys = json.load(data_file)
    if mltype=="classification":
        return apikeys['classification']
    elif mltype=="prediction":
        return apikeys['prediction']
    else:
        print("Algorithm doesn't exist")

@app.route('/')
def home():
    return render_template('prediction.html')

@app.route('/prediction')
def render_prediction():
    return render_template('prediction.html')

@app.route('/classification')
def render_classification():
    return render_template('classification.html')

# /* Endpoint to greet and add a new visitor to database.
# * Send a POST request to localhost:8080/api/visitors with body
# * {
# *     "name": "Bob"
# * }
# */
@app.route('/prediction/getPrediction', methods=['POST'])
def get_prediction():
    try:
        apikeys=loadApiKeys('prediction')
        if apikeys == None:
            print("Api Keys file has some issue")
            return_dict = {"predicted_interest_rate":"Some Error occured with api keys file"}
            return json.dumps(return_dict)
        else:
            credit_score=request.json['credit_score']
            og_first_time_home_buyer=request.json['og_first_time_home_buyer']
            og_upb=request.json['og_upb']
            og_loan_term=request.json['og_loan_term']
            og_quarter_year=request.json['og_quarter_year']
            og_seller_name=request.json['og_seller_name']
            og_servicer_name=request.json['og_servicer_name']
            algoType = request.json['algoType']
            #print(str(algoType)+"\t"+str(credit_score)+"\t"+str(og_first_time_home_buyer)+"\t"+str(og_upb)+"\t"+str(og_loan_term)+"\t"+str(og_quarter_year)+"\t"+str(og_seller_name)+"\t"+str(og_servicer_name))
            #make ai call
            if algoType=="pred_df":
                url=apikeys['boosteddecisiontree']['url']
                api_key=apikeys['boosteddecisiontree']['apikey']
            elif algoType=="pred_nn":
                url=apikeys['neuralnetwork']['url']
                api_key=apikeys['neuralnetwork']['apikey']
            elif algoType=="pred_lr":
                url=apikeys['linearregression']['url']
                api_key=apikeys['linearregression']['apikey']
            
            
            data =  {
            
                    "Inputs": {
            
                            "input1":
                            {
                                "ColumnNames": ["CREDIT_SCORE", "FIRST_HOME_BUYER_FLAG", "OG_UPB", "OG_LOANTERM", "SELLER_NAME", "SERVICE_NAME", "OG_QUARTERYEAR"],
                                "Values": [ [credit_score,og_first_time_home_buyer,og_upb,og_loan_term,og_seller_name,og_servicer_name,og_quarter_year]]
                            },        },
                        "GlobalParameters": {
            }
                }
            
            body = str.encode(json.dumps(data))
            
            #url = 'https://ussouthcentral.services.azureml.net/workspaces/5de0e8bd28f74cf9a40babb3f1799a53/services/300d6267d2f843c9a5975621ff077a09/execute?api-version=2.0&details=true'
            #api_key = 'wQWgTpa3GyVACzg7Q6jVDdwt5JEDnfdvqqG21PKDr+UHmZWRQJh1XfrtLVON846vEDEXoDgnruZ1s9zd4Drzyw==' # Replace this with the API key for the web service
            headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}
            
            response = requests.post(url, data=body,headers=headers)
            #print(response.content)
            
            response_json=json.loads(response.content)
            predicted_interest_rate=response_json['Results']['output1']['value']['Values'][0][7]
            
            if predicted_interest_rate == "":
                predicted_interest_rate = "Some error occured"
            
            return_dict = {"predicted_interest_rate":predicted_interest_rate}
            return json.dumps(return_dict)
    except:
        return_dict = {"predicted_interest_rate":"Some error occured"}
        return json.dumps(return_dict) 
    
@app.route('/classification/getClassification', methods=['POST'])
def get_classification():
    try:
        apikeys=loadApiKeys('classification')
        if apikeys == None:
            print("Api Keys file has some issue")
            classified_as="Some Error occured with api keys file"
            scored_probability = ""
            return_dict = {"classified_as":classified_as,"scored_probability":scored_probability}
            return json.dumps(return_dict)
        else:
            curr_act_upb=request.json['curr_act_upb']
            loan_age=request.json['loan_age']
            months_to_legal_maturity=request.json['months_to_legal_maturity']
            curr_interest_rate=request.json['crr_interest_rate']
            curr_deferred_upb=request.json['curr_deferred_upb']
            algoType = request.json['algoType']
            #print(curr_act_upb+"\t"+loan_age+"\t"+months_to_legal_maturity+"\t"+curr_interest_rate+"\t"+curr_deferred_upb)
            #make ai call
            if algoType=="pred_df":
                url=apikeys['decisionjungle']['url']
                api_key=apikeys['decisionjungle']['apikey']
            elif algoType=="pred_nn":
                url=apikeys['bayestwopoint']['url']
                api_key=apikeys['bayestwopoint']['apikey']
            elif algoType=="pred_lr":
                url=apikeys['logisticregression']['url']
                api_key = apikeys['logisticregression']['apikey']
                
            data =  {
            
                    "Inputs": {
            
                            "input1":
                            {
                                "ColumnNames": ["CUR_ACT_UPB", "LOAN_AGE", "MONTHS_LEGAL_MATURITY", "CURR_INTERESTRATE", "CURR_DEF_UPB"],
                                "Values": [[curr_act_upb, loan_age, months_to_legal_maturity, curr_interest_rate, curr_deferred_upb]]
                            },        },
                        "GlobalParameters": {
            }
                }
            
            body = str.encode(json.dumps(data))
            
            #url = 'https://ussouthcentral.services.azureml.net/workspaces/5de0e8bd28f74cf9a40babb3f1799a53/services/300d6267d2f843c9a5975621ff077a09/execute?api-version=2.0&details=true'
            #api_key = 'wQWgTpa3GyVACzg7Q6jVDdwt5JEDnfdvqqG21PKDr+UHmZWRQJh1XfrtLVON846vEDEXoDgnruZ1s9zd4Drzyw==' # Replace this with the API key for the web service
            headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}
            
            response = requests.post(url, data=body,headers=headers)
            #print(response.content)
            
            response_json=json.loads(response.content)
            if response_json['Results']['output1']['value']['Values'][0][5] == "0":
                scored_probability=response_json['Results']['output1']['value']['Values'][0][6]
                classified_as="Non-Delinquent"
            elif response_json['Results']['output1']['value']['Values'][0][5] == "1":
                scored_probability=response_json['Results']['output1']['value']['Values'][0][6]
                classified_as="Delinquent"
            else:
                classified_as="Some Error occured in Classification"
                scored_probability = ""
            return_dict = {"classified_as":classified_as,"scored_probability":scored_probability}
            return json.dumps(return_dict)
    except:
        return_dict = {"classified_as":"Some Error occured."}
        return json.dumps(return_dict)

# /**
#  * Endpoint to get a JSON array of all the visitors in the database
#  * REST API example:
#  * <code>
#  * GET http://localhost:8080/api/visitors
#  * </code>
#  *
#  * Response:
#  * [ "Bob", "Jane" ]
#  * @return An array of all the visitor names
#  */
@app.route('/api/visitors', methods=['POST'])
def put_visitor():
    user = request.json['name']
    if client:
        data = {'name':user}
        db.create_document(data)
        return 'Hello %s! I added you to the database.' % user
    else:
        print('No database')
        return 'Hello %s!' % user

@atexit.register
def shutdown():
    if client:
        client.disconnect()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
