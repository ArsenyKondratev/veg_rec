Импортируй пакеты в виртуальную среду
python -m venv venv
source venv/bin/activate  # для macOS/Linux
venv\Scripts\activate  # для Windows
pip install -r requirements.txt

Запусти сервер
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
