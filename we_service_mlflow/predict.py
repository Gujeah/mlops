import pickle
from flask import Flask, request, jsonify

with open ('lin_reg.bin' , 'rb') as f_in:
    (dv, model)=pickle.load(f_in)


def prepare_features(ride):
    features={}
    features["PU_DO"]='%s_%s' % (ride["PULocationID"], ride["DOLocationID"])
    features["trip_distance"]=ride["trip_distance"]
    return features

def prediction(features):
    X=dv.transform(features)
    preds=model.predict(X)
    return preds[0]

app=Flask("predicting duration")
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride=request.get_json()
    features=prepare_features(ride)
    preds=prediction(features)
    result={ 
        'duration': preds
    }
    return jsonify(result)

if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)