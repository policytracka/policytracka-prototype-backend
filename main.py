from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import algorithm # all data processing and clustering algorithms will be in this file

app = FastAPI()
@app.get("/")
async def home():
    msg = {
        'message': 'This is home of policy tracka project prototype server',
        'api_endpoints': ['cluster_groups', 'cluster'], # please add more endpoints here as you add them
    }
    result = JSONResponse(content=msg)
    return result

@app.get("/api/cluster_groups")
async def cluster_groups():
    cluster = algorithm.get_cluster_groups()
    msg = {
        'message': 'cluster groups',
        'clusters': cluster,
    }
    result = JSONResponse(content=msg)
    return result

@app.get("/api/cluster")
async def cluster(policy: str):
    msg = {
        'message': 'cluster',
        'cluster': algorithm.get_cluster_of(policy), # temp
    }
    result = JSONResponse(content=msg)
    return result