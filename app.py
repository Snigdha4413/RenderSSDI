from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import numpy as np
from scipy.stats import t, tstd
import os

app = Flask(__name__)

# Fix for Render's postgres:// vs postgresql:// requirement
uri = os.getenv("DATABASE_URL")
if uri and uri.startswith("postgres://"):
    uri = uri.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = uri
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

def twottest(a, b, alt, alpha=0.05):
    n1, n2 = len(a), len(b)
    x1, x2 = np.mean(a), np.mean(b)
    sd1, sd2 = tstd(a), tstd(b)
    se = np.sqrt((sd1**2 / n1) + (sd2**2 / n2))
    t_cal = (x1 - x2) / se
    df = n1 + n2 - 2
    if alt == 'two-sided':
        p = 2 * (1 - t.cdf(abs(t_cal), df))
        return {'t_cal': round(t_cal, 4), 'p_value': round(p, 4)}
    elif alt == 'greater':
        p = 1 - t.cdf(t_cal, df)
        return {'t_cal': round(t_cal, 4), 'p_value': round(p, 4)}
    else:
        p = t.cdf(t_cal, df)
        return {'t_cal': round(t_cal, 4), 'p_value': round(p, 4)}

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    columns = []
    try:
        # Dynamically fetch column names from your iris_data table
        raw_columns = db.session.execute('SELECT * FROM iris_data LIMIT 0').keys()
        columns = [col for col in raw_columns]
    except Exception as e:
        print(f"DB Error: {e}")

    if request.method == 'POST':
        col1 = request.form.get('column1')
        col2 = request.form.get('column2')
        alt = request.form.get('alt')
        data1 = db.session.execute(f'SELECT "{col1}" FROM iris_data').scalars().all()
        data2 = db.session.execute(f'SELECT "{col2}" FROM iris_data').scalars().all()
        if data1 and data2:
            result = twottest(data1, data2, alt)

    return render_template('index.html', result=result, columns=columns)
