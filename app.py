from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import numpy as np
from scipy.stats import t, tstd
import os

app = Flask(__name__)

# Database Configuration for Render
# Ensure you add DATABASE_URL to your Render Environment Variables
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
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
        t_pos = t.ppf(1 - alpha/2, df)
        t_neg = t.ppf(alpha/2, df)
        p = 2 * (1 - t.cdf(abs(t_cal), df))
        return {'t_cal': round(t_cal, 4), 't_pos': round(t_pos, 4), 't_neg': round(t_neg, 4), 'p_value': round(p, 4)}
    
    elif alt == 'greater':
        t_pos = t.ppf(1 - alpha, df)
        p = 1 - t.cdf(t_cal, df)
        return {'t_cal': round(t_cal, 4), 't_pos': round(t_pos, 4), 'p_value': round(p, 4)}
    
    else: # lesser
        t_neg = t.ppf(alpha, df)
        p = t.cdf(t_cal, df)
        return {'t_cal': round(t_cal, 4), 't_neg': round(t_neg, 4), 'p_value': round(p, 4)}

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    columns = []
    
    # Get column names from your iris_data table for the dropdowns
    try:
        raw_columns = db.session.execute('SELECT * FROM iris_data LIMIT 0').keys()
        columns = [col for col in raw_columns]
    except Exception as e:
        print(f"Database error: {e}")

    if request.method == 'POST':
        col1 = request.form.get('column1')
        col2 = request.form.get('column2')
        alt = request.form.get('alt')
        
        # Fetch data from SQL
        data1 = db.session.execute(f'SELECT "{col1}" FROM iris_data').scalars().all()
        data2 = db.session.execute(f'SELECT "{col2}" FROM iris_data').scalars().all()
        
        if data1 and data2:
            result = twottest(data1, data2, alt)

    return render_template('index.html', result=result, columns=columns)

if __name__ == '__main__':
    app.run(debug=True)
