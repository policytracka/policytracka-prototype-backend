from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import algorithm # all data processing and clustering algorithms will be in this file
import uvicorn

## allow CORS
origins = ['*']
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
async def home():
    msg = {
        'message': 'This is home of policy tracka project prototype server',
        'api_endpoints': ['cluster_groups', 'cluster', 'treemap', 'wordcloud', 'cluster_from_group'], # please add more endpoints here as you add them
    }
    result = JSONResponse(content=msg)
    return result

@app.get("/api/cluster_groups")
async def cluster_groups():
    cluster = algorithm.get_cluster_kmean()
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
        'cluster': algorithm.get_cluster_of(policy, return_id=False), # temp
    }
    result = JSONResponse(content=msg)
    return result

@app.get("/api/cluster_from_group")
async def cluster_from_group(group: int):
    id = group
    group = algorithm.get_treemap()[id]['name']
    data = algorithm.cluster_from_group(group)
    if len(data) == 0:
        msg = {
            'message': 'no_cluster_in_this_group',
            'group': {},
        }
        result = JSONResponse(content=msg)
        return result
    msg = {
        'message': 'cluster_in_this_group',
        'group': {
            'id': id,
            'name': group,
            'data': data,
        }
    }
    result = JSONResponse(content=msg)
    return result

@app.get("/api/treemap")
async def treemap():
    msg = {
        'message': 'treemap',
        'treemap': algorithm.get_treemap(), # temp
    }
    result = JSONResponse(content=msg)
    return result

@app.get("/api/wordcloud")
async def wordcloud():
    msg = {
        'message': 'wordcloud',
        'wordcloud': algorithm.get_wordcloud(), # temp
    }
    result = JSONResponse(content=msg)
    return result

@app.get("/api/political_party_icon")
async def political_party(id: int):
    msg = {
        'message': 'political_party',
        'img': algorithm.get_political_party_icon(id), # temp
    }
    result = JSONResponse(content=msg)
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)