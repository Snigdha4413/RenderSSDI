from flask import Flask, render_template, request
import numpy as np
from scipy.stats import t, tstd

app = Flask(__name__)

def twottest(a, b, alt, alpha=0.05):
    n1, n2 = len(a), len(b)
    x1, x2 = np.mean(a), np.mean(b)
    sd1, sd2 = tstd(a), tstd(b) # Using sample SD
    se = np.sqrt((sd1**2 / n1) + (sd2**2 / n2))
    t_cal = (x1 - x2) / se
    df = n1 + n2 - 2

    if alt == 'two-sided':
        t_pos = t.ppf(1 - alpha/2, df)
        t_neg = t.ppf(alpha/2, df)
        p = 2 * (1 - t.cdf(abs(t_cal), df)) # Standard two-sided p-value
        return {'t_cal': t_cal, 't_pos': t_pos, 't_neg': t_neg, 'p_value': p}
    
    elif alt == 'greater':
        t_pos = t.ppf(1 - alpha, df)
        p = 1 - t.cdf(t_cal, df)
        return {'t_cal': t_cal, 't_pos': t_pos, 'p_value': p}
    
    else: # lesser
        t_neg = t.ppf(alpha, df)
        p = t.cdf(t_cal, df)
        return {'t_cal': t_cal, 't_neg': t_neg, 'p_value': p}

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        # Simple comma-separated input parsing
        a = [float(x) for x in request.form['a'].split(',')]
        b = [float(x) for x in request.form['b'].split(',')]
        alt = request.form['alt']
        result = twottest(a, b, alt)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)