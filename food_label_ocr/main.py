from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.routes import router  # 引入 API 路由

#--建立FastAPI應用
#--查看自動產生的API文件：http://127.0.0.1:8000/docs#/
app = FastAPI(
    title="食品標示 OCR 分析系統", 
    description="一個簡單的 OCR 服務，用於分析食品標示。"
)


#--掛載靜態文件目錄
app.mount("/static", StaticFiles(directory="static"), name="static")
#--啟用API路由
app.include_router(router)

@app.get("/")
async def root():
    return {"message": "歡迎使用食品標示 OCR 分析系統！"}

if __name__ == "__main__":
    #--使用uvicorn main:app前啟動即可，不必在腳本一開始就import uvicorn
    #--uvicorn屬於外部CLI工具，無須剛開始就硬綁server。
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)