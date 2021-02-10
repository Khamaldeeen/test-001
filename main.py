import uvicorn
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

# 2. Create app and model objects
app = FastAPI()
house_model = joblib.load("house_model.sav")


class Sales(BaseModel):
    type_ : str
    location : str 
    bed : int 
    bath : int 
    toilet : int 


# 3. Expose the prediction functionality, make a prediction from the passed

@app.get("/")
def index():
    return {"House": "Sales"}

@app.post("/predict/")
def PredictSales(data: Sales):
    data = data.dict()
    house = data["type_"]
    loc = data["location"]
    bed = data["bed"]
    bath = data["bath"]
    toilet = data["toilet"]
    
    pred = house_model.predict([house, loc, bed, bath, toilet])

    pred = int(pred)

 
    
    return {'Sales': pred}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    #allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
