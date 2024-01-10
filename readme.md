車牌OCR
===

### 第一步 安裝所需套件

假設你已安裝完python3，在專案目錄下用CMD開啟(絕對路徑不能有中文)

```cmd
建立虛擬環境病雞依賴安裝在橡木資料夾，避免汙染全域環境
python -m venv cuda

cuda\Scripts\activate

安裝所需套件
pip install -r PaddleOCR/requirements.txt

如果要使用gpu加速運行請額外安裝pytorch工具
這行指令由pyTroch官網獲得
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

如果要使用gpu加速運行請額外安裝paddleocr工具
pip install paddlepaddle-gpu==2.4.2.post117 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
```