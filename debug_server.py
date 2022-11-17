import uvicorn
from main import app

if __name__ == '__main__':
    uvicorn.run("debug_server:app", reload=True, host='localhost', port=5000)